"""RFC 5322 / MIME bytes to ParsedEmail normalization.

    raw email bytes  ->  core.models.ParsedEmail

Pure and offline, per ARCHITECTURE.md section 3. This module never opens a
socket: no URL is fetched, no redirect is followed, no DNS or WHOIS lookup is
made, and no attachment is written to disk or executed. It only reads bytes
that were handed to it.

The parser is deliberately not coupled to Gmail. It takes raw RFC-822 bytes
from any source - a Gmail `format='raw'` fetch, a `.eml` file upload, a test
fixture - and knows nothing about where they came from.

Determinism
-----------
The same input bytes always produce an equivalent ParsedEmail. Nothing here
consults the clock, the network or a random source.

Message identifier
------------------
`ParsedEmail.message_id` is taken from the `Message-ID` header when one is
present. When the header is missing or empty, the identifier is derived
deterministically from the raw bytes as ``sha256:<hexdigest>``. This is a
content hash, never a random UUID, so re-parsing the same message yields the
same id and a cache lookup stays stable. The ``sha256:`` prefix makes a
synthesized id obvious at a glance.

Untrusted input
---------------
Every email is treated as hostile. Parsing uses the standard library's
tolerant facilities and recovers from malformed structure where it safely
can. HTML is scanned for URL-bearing attributes only - it is never rendered,
never executed, and no external resource it references is fetched.

What this module does not do: it extracts, it does not judge. No detection,
no scoring, no classification of content.
"""

from __future__ import annotations

import hashlib
import re
from email import message_from_bytes, policy
from email.message import Message
from email.parser import BytesParser
from email.utils import getaddresses, parseaddr

from bs4 import BeautifulSoup

from core.models import Attachment, ExtractedURL, ParsedEmail, URLSource

__all__ = ["EmailParseError", "parse_email"]


class EmailParseError(ValueError):
    """Raised when input cannot be represented as a ParsedEmail at all.

    Reserved for the case where a field the contract requires cannot be
    recovered - in practice a missing or unparseable sender address. The
    parser deliberately does not invent a placeholder sender, because the
    sender is security-relevant and a fabricated one would be indistinguishable
    downstream from a real one.
    """


# Absolute http(s) URLs only. Bare hostnames are not matched: without a scheme
# there is no reliable way to tell "example.com" from ordinary prose.
_TEXT_URL_RE = re.compile(r"https?://[^\s<>\"'`\[\]{}\\^|]+", re.IGNORECASE)

# Punctuation that commonly trails a URL in prose rather than belonging to it.
_TRAILING_PUNCTUATION = ".,;:!?\"'>*_~"

_URL_BEARING_ATTRS: tuple[tuple[str, str, URLSource], ...] = (
    ("a", "href", URLSource.ANCHOR_HREF),
    ("img", "src", URLSource.IMG_SRC),
    ("form", "action", URLSource.FORM_ACTION),
)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def parse_email(raw: bytes) -> ParsedEmail:
    """Normalize raw RFC-822 bytes into a ParsedEmail.

    Raises EmailParseError when no sender address can be recovered; see the
    class docstring for why that case is not papered over.
    """
    if not isinstance(raw, (bytes, bytearray)):
        raise TypeError(f"raw must be bytes, got {type(raw).__name__}")
    raw = bytes(raw)

    message = _parse_message(raw)

    headers = _collect_headers(message)
    from_display, from_addr = _parse_sender(message)
    if not from_addr:
        raise EmailParseError(
            "no sender address could be recovered from the From header; "
            "refusing to fabricate one"
        )

    body_text, body_html = _extract_bodies(message)

    return ParsedEmail(
        message_id=_message_id(message, raw),
        from_addr=from_addr,
        from_display=from_display,
        to_addrs=_address_list(message, "To"),
        subject=_header(message, "Subject"),
        body_text=body_text,
        body_html=body_html,
        reply_to=_reply_to(message),
        urls=_extract_urls(body_text, body_html),
        headers=headers,
        received_chain=list(headers.get("received", [])),
        attachments=_extract_attachments(message),
        raw=raw,
    )


# --------------------------------------------------------------------------
# Message construction
# --------------------------------------------------------------------------


def _parse_message(raw: bytes) -> Message:
    """Parse bytes into a message object, tolerating malformed input.

    Prefers the modern `policy.default` parser, which decodes RFC 2047 encoded
    headers for us. Falls back to the legacy compat32 parser if the modern one
    chokes on badly broken input, so a hostile message degrades rather than
    crashing the pipeline.
    """
    try:
        return BytesParser(policy=policy.default).parsebytes(raw)
    except Exception:  # noqa: BLE001 - untrusted input, any failure falls back
        return message_from_bytes(raw)


# --------------------------------------------------------------------------
# Headers
# --------------------------------------------------------------------------


def _header(message: Message, name: str) -> str:
    """One decoded header value, or "" when absent or undecodable."""
    try:
        value = message.get(name)
    except Exception:  # noqa: BLE001 - malformed header must not propagate
        return ""
    if value is None:
        return ""
    return _as_text(value)


def _as_text(value: object) -> str:
    """Render a header value as text without raising on malformed content."""
    try:
        text = str(value)
    except Exception:  # noqa: BLE001 - defective encoded word
        return ""
    # Header values may be folded across lines; normalize to a single line.
    return " ".join(text.split())


def _collect_headers(message: Message) -> dict[str, list[str]]:
    """All headers, lower-cased keys, values in the order they appeared.

    Multi-valued headers (Received, and any repeated header) keep every value,
    per the ARCHITECTURE.md section 2 contract.
    """
    headers: dict[str, list[str]] = {}
    try:
        items = message.items()
    except Exception:  # noqa: BLE001 - malformed header block
        return headers
    for name, value in items:
        try:
            key = str(name).lower()
        except Exception:  # noqa: BLE001
            continue
        headers.setdefault(key, []).append(_as_text(value))
    return headers


def _parse_sender(message: Message) -> tuple[str, str]:
    """(display name, address) from From. Either may be empty."""
    display, addr = parseaddr(_header(message, "From"))
    return display.strip(), addr.strip()


def _address_list(message: Message, name: str) -> list[str]:
    """Every address in a possibly repeated, possibly multi-value header.

    Order is preserved and duplicates are dropped, so the result is stable for
    identical input.
    """
    try:
        raw_values = message.get_all(name, [])
    except Exception:  # noqa: BLE001
        return []
    decoded = [_as_text(v) for v in raw_values]

    addresses: list[str] = []
    seen: set[str] = set()
    for _display, addr in getaddresses(decoded):
        addr = addr.strip()
        if addr and addr not in seen:
            seen.add(addr)
            addresses.append(addr)
    return addresses


def _reply_to(message: Message) -> str | None:
    """Reply-To address, or None when the header is absent or empty."""
    _display, addr = parseaddr(_header(message, "Reply-To"))
    return addr.strip() or None


def _message_id(message: Message, raw: bytes) -> str:
    """Message-ID header, or a deterministic content hash when absent."""
    message_id = _header(message, "Message-ID").strip()
    if message_id:
        return message_id
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


# --------------------------------------------------------------------------
# Bodies
# --------------------------------------------------------------------------


def _is_attachment_part(part: Message) -> bool:
    """True when a part should be treated as an attachment, not as a body."""
    try:
        disposition = (part.get_content_disposition() or "").lower()
    except Exception:  # noqa: BLE001
        disposition = ""
    if disposition == "attachment":
        return True
    if _filename(part):
        return True
    # A cid:-referenced part is an inline image, not the message body.
    return bool(_content_id(part))


def _part_text(part: Message) -> str:
    """Decoded text of one part, degrading rather than raising."""
    try:
        content = part.get_content()
        if isinstance(content, str):
            return content
    except Exception:  # noqa: BLE001 - unknown charset, broken encoding
        pass
    try:
        payload = part.get_payload(decode=True)
    except Exception:  # noqa: BLE001
        return ""
    if not isinstance(payload, (bytes, bytearray)):
        return ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


def _extract_bodies(message: Message) -> tuple[str, str | None]:
    """Return (body_text, body_html).

    Walks the whole tree, so multipart/alternative, multipart/mixed and nested
    structures are all handled: the first non-attachment text/plain and the
    first non-attachment text/html win.

    When a message carries HTML but no usable plaintext part, body_text is
    derived locally from that HTML.
    """
    text: str | None = None
    html: str | None = None

    try:
        parts = list(message.walk())
    except Exception:  # noqa: BLE001 - broken multipart structure
        parts = [message]

    for part in parts:
        try:
            if part.get_content_maintype() == "multipart":
                continue
            content_type = part.get_content_type()
        except Exception:  # noqa: BLE001
            continue
        if _is_attachment_part(part):
            continue
        if content_type == "text/plain" and text is None:
            text = _part_text(part)
        elif content_type == "text/html" and html is None:
            html = _part_text(part)

    if text is None and html is None:
        # A message can declare multipart and then carry no boundary at all
        # (StartBoundaryNotFoundDefect). The walk finds no usable subpart, but
        # the body is still sitting there in the payload - recover it rather
        # than reporting an empty body for a message that plainly has one.
        recovered = _recover_flat_body(message)
        if recovered:
            text = recovered

    if text is None and html is not None:
        text = _html_to_text(html)
    return (text or ""), html


def _recover_flat_body(message: Message) -> str:
    """Last-resort body recovery for structurally broken messages.

    Only applies when the message has no real subparts, so a well-formed
    multipart whose parts are all attachments still yields an empty body rather
    than a dump of MIME boundaries.
    """
    try:
        if message.is_multipart():
            return ""
    except Exception:  # noqa: BLE001
        return ""
    return _part_text(message)


def _html_to_text(html: str) -> str:
    """Local, offline HTML-to-text fallback.

    Parsed with the stdlib html.parser backend. Nothing is rendered, no script
    is run and no referenced resource is fetched.
    """
    soup = _soup(html)
    if soup is None:
        return ""
    for element in soup(["script", "style"]):
        element.decompose()
    return soup.get_text(separator="\n", strip=True)


def _soup(html: str) -> BeautifulSoup | None:
    """BeautifulSoup over untrusted HTML, or None if it cannot be parsed."""
    try:
        return BeautifulSoup(html, "html.parser")
    except Exception:  # noqa: BLE001 - hostile markup must not crash ingest
        return None


# --------------------------------------------------------------------------
# URLs
# --------------------------------------------------------------------------


def _extract_urls(body_text: str, body_html: str | None) -> list[ExtractedURL]:
    """Collect URLs from the HTML attributes and the plaintext body.

    HTML is scanned first so that when the same URL appears both as a link and
    as visible text, the richer anchor observation is the one kept.

    The plaintext body is always scanned, including when it was derived from
    HTML: that is the only way to catch a bare URL written as text and never
    wrapped in an anchor.

    Layer 2 fields are left untouched: redirect_chain stays empty, final_url
    stays None and redirect_depth is therefore 0. Nothing is fetched here.
    """
    found: list[ExtractedURL] = []
    if body_html:
        found.extend(_urls_from_html(body_html))
    if body_text:
        found.extend(_urls_from_text(body_text))
    return _dedupe(found)


def _urls_from_html(html: str) -> list[ExtractedURL]:
    """URLs from <a href>, <img src> and <form action>, in document order."""
    soup = _soup(html)
    if soup is None:
        return []

    found: list[ExtractedURL] = []
    for tag_name, attribute, source in _URL_BEARING_ATTRS:
        for element in soup.find_all(tag_name):
            value = element.get(attribute)
            if not isinstance(value, str):
                continue
            url = _normalize(value)
            if not url:
                continue
            anchor_text = None
            if source is URLSource.ANCHOR_HREF:
                anchor_text = element.get_text(strip=True) or None
            found.append(ExtractedURL(url=url, source=source, anchor_text=anchor_text))
    return found


def _urls_from_text(text: str) -> list[ExtractedURL]:
    """Absolute http(s) URLs written in plain text."""
    found: list[ExtractedURL] = []
    for match in _TEXT_URL_RE.finditer(text):
        url = _trim_trailing_punctuation(match.group(0))
        if url:
            found.append(ExtractedURL(url=url, source=URLSource.PLAIN_TEXT))
    return found


def _normalize(value: str) -> str:
    """Keep absolute http(s) URLs; drop everything else.

    Deliberately dropped, because they are not URLs Layer 2 can analyse:
    mailto:, tel:, cid: (inline image references, carried as attachment
    metadata instead), data:, javascript:, bare fragments, and scheme-relative
    or site-relative paths that cannot be resolved without a base URL - and
    resolving one would require knowing the sender's site, which we do not.
    """
    url = value.strip()
    if not url:
        return ""
    lowered = url.lower()
    if not (lowered.startswith("http://") or lowered.startswith("https://")):
        return ""
    return url


def _trim_trailing_punctuation(url: str) -> str:
    """Strip sentence punctuation that the regex swept up with the URL."""
    url = url.rstrip(_TRAILING_PUNCTUATION)
    # Only drop a closing paren when it has no opener inside the URL, so
    # https://en.wikipedia.org/wiki/Phishing_(email) survives intact.
    while url.endswith(")") and url.count(")") > url.count("("):
        url = url[:-1].rstrip(_TRAILING_PUNCTUATION)
    return url


def _dedupe(urls: list[ExtractedURL]) -> list[ExtractedURL]:
    """Deduplicate on (url, source), keeping first-seen order.

    Keying on the URL rather than on link text means two different URLs that
    share the same anchor text ("Click here") are both kept. Source is part of
    the key on purpose: the same URL reached as an anchor and as a form action
    is two distinct observations, and Layer 2 cares which is which.

    The first occurrence wins, so its anchor text is the one retained. Order is
    stable for identical input.
    """
    deduped: list[ExtractedURL] = []
    seen: set[tuple[str, URLSource]] = set()
    for url in urls:
        key = (url.url, url.source)
        if key not in seen:
            seen.add(key)
            deduped.append(url)
    return deduped


# --------------------------------------------------------------------------
# Attachments
# --------------------------------------------------------------------------


def _filename(part: Message) -> str:
    """Decoded attachment filename, or "" when the part is unnamed."""
    try:
        name = part.get_filename()
    except Exception:  # noqa: BLE001 - malformed parameter encoding
        return ""
    return _as_text(name) if name else ""


def _content_id(part: Message) -> str | None:
    """Content-ID with its angle brackets stripped, or None."""
    content_id = _header(part, "Content-ID").strip()
    if not content_id:
        return None
    return content_id.removeprefix("<").removesuffix(">") or None


def _payload_size(part: Message) -> int:
    """Size in bytes of the decoded payload, or 0 when it cannot be measured.

    The payload is decoded only far enough to measure it. Its contents are
    never inspected, stored, written to disk or executed.
    """
    try:
        payload = part.get_payload(decode=True)
    except Exception:  # noqa: BLE001
        return 0
    return len(payload) if isinstance(payload, (bytes, bytearray)) else 0


def _extract_attachments(message: Message) -> list[Attachment]:
    """Metadata for every attachment and cid:-referenced inline image.

    Metadata only - filename, content type, size and Content-ID. Nothing is
    saved, scanned or executed. Inline images are represented by carrying
    their Content-ID, which is what ParsedEmail.inline_images filters on.
    """
    attachments: list[Attachment] = []
    try:
        parts = list(message.walk())
    except Exception:  # noqa: BLE001
        return attachments

    for part in parts:
        try:
            if part.get_content_maintype() == "multipart":
                continue
        except Exception:  # noqa: BLE001
            continue
        if not _is_attachment_part(part):
            continue
        try:
            content_type = part.get_content_type()
        except Exception:  # noqa: BLE001
            content_type = "application/octet-stream"
        attachments.append(
            Attachment(
                filename=_filename(part),
                content_type=content_type or "application/octet-stream",
                size_bytes=_payload_size(part),
                content_id=_content_id(part),
            )
        )
    return attachments
