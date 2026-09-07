"""Layer 2 - URL and redirect chain analysis.

Signals specified in ARCHITECTURE.md section 4:
  l2.base64_email_param    base64-encoded address in query/fragment
  l2.redirect_depth        hop following, allow_redirects=False, cap 8
  l2.captcha_gate          Turnstile/hCaptcha/reCAPTCHA in rendered DOM
  l2.qr_url                QR decode of image and inline cid: attachments
  l2.domain_mismatch_brand anchor text brand vs. href target

Network layer: must run from sandboxed egress, never a home or campus IP.

**No signal is implemented yet.** This module currently carries only the
layer's foundation: the offline URL analysis seam that every one of the five
signals needs before it can do anything.

    ParsedEmail.urls ─┐
                      ├─> candidate_urls() ─> [URLCandidate] ─> a signal
    QR-decoded URLs ──┘        (offline)

What this foundation is for
---------------------------
Four of the five signals are questions about the *parts* of a URL, not about
the string: `base64_email_param` reads the query and fragment,
`domain_mismatch_brand` compares an anchor's brand claim against the href's
registrable domain, `redirect_depth` needs to know when two hops are the same
place, and `qr_url` per section 4 puts its decoded URLs back into "the URL
signal set" - which only exists if something defines what that set is. So the
foundation is one safe parse and one canonical form, computed once and shared,
rather than five signals each doing their own `urlsplit` and disagreeing at
the edges.

Extraction is **not** re-implemented here. `ingest/parser.py` already pulls
URLs from `<a href>`, `<img src>`, `<form action>` and the plaintext body, and
records the `URLSource` and anchor text for each; this module consumes
`ParsedEmail.urls` and adds the analysis view on top. Duplicating that would
give the pipeline two different ideas of what a message's URLs are.

Offline, and strictly so
------------------------
Nothing in this module opens a socket, resolves a name, follows a redirect or
renders anything. Parsing and canonicalizing a URL is pure string work, and it
has to stay that way: the hop-following and Playwright rendering that section 4
requires are the parts that must run from sandboxed egress, and keeping them
out of the foundation is what lets every signal's parsing be unit-tested
without a network at all.

Malformed input fails closed - `parse_url` returns None rather than raising or
guessing, and a URL that cannot be parsed is dropped from the candidate set
rather than passed on half-understood. A phishing URL is hostile input by
definition, and a parser that improvises on it produces confident nonsense.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence
from urllib.parse import parse_qsl, urlsplit

from core.models import ExtractedURL, ParsedEmail, URLSource
from layers.l1_headers import registrable_domain

__all__ = [
    "DEFAULT_PORTS",
    "URLCandidate",
    "candidate_urls",
    "canonical_form",
    "parse_url",
]

# Ports that carry no information because they are implied by the scheme.
# Stripped from the canonical form so `https://a.example` and
# `https://a.example:443` are recognized as one place.
DEFAULT_PORTS = {"http": 80, "https": 443}

# The schemes Layer 2 analyses. `ingest.parser` already drops everything else -
# mailto:, tel:, cid:, data:, javascript: - but a QR decode produces a raw
# string from an image and has had no such filtering, so the check lives here
# too rather than being assumed upstream.
_ANALYSABLE_SCHEMES = frozenset({"http", "https"})


@dataclass(frozen=True)
class URLCandidate:
    """One URL, parsed into the parts Layer 2's signals actually ask about.

    Frozen, like `DetectionSignal`: a candidate records how a URL was read and
    must not be edited afterwards.

    `raw` is the string exactly as it appeared, because that is what evidence
    must quote - a signal that reports a canonicalized URL is reporting
    something the message did not contain. `canonical` is for comparison only.

    `query_params` and `fragment_params` are parsed with `keep_blank_values`
    so `?redirect=` is visible as a present-but-empty parameter rather than
    vanishing; `l2.base64_email_param` reads both, since section 4 names the
    query *and* the fragment.

    `source` and `anchor_text` are carried through from `ingest.parser` -
    `l2.domain_mismatch_brand` needs the anchor text, and it is the parser, not
    this layer, that knows how to get it.
    """

    raw: str
    canonical: str
    scheme: str
    host: str
    port: int | None
    path: str
    query: str
    fragment: str
    registrable: str | None
    source: URLSource
    anchor_text: str | None = None
    query_params: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    fragment_params: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @property
    def is_ip_literal(self) -> bool:
        """True when the host is an address rather than a name.

        `registrable_domain` returns None for both an IP literal and a
        malformed host, and the two are different facts - a link to a bare IP
        is a Layer 2 observation, a broken host is not.
        """
        return self.registrable is None and _looks_like_address(self.host)


def _looks_like_address(host: str) -> bool:
    """Whether `host` is an IPv4 dotted quad or a bracketed IPv6 literal."""
    if host.startswith("[") and host.endswith("]"):
        return True
    octets = host.split(".")
    return len(octets) == 4 and all(
        octet.isdigit() and len(octet) <= 3 and int(octet) <= 255 for octet in octets
    )


def parse_url(url: str) -> tuple[str, str, int | None, str, str, str] | None:
    """Split a URL into (scheme, host, port, path, query, fragment), or None.

    Returns None - never raises, never guesses - for anything Layer 2 cannot
    analyse: an empty string, a scheme other than http(s), a missing host, or
    a value `urlsplit` rejects outright. Hostile input is the normal case here,
    so the failure mode has to be "this is not a URL I understand", not an
    exception the caller has to remember to catch or, worse, a partially
    parsed result that reads as though it were understood.

    The port is returned separately and only when explicitly present and valid;
    an out-of-range or non-numeric port makes the whole URL unparseable rather
    than being silently dropped, because `https://evil.example:99999/` is not a
    URL a browser will treat the way this parse would imply.
    """
    if not isinstance(url, str) or not url.strip():
        return None

    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return None

    scheme = parts.scheme.lower()
    if scheme not in _ANALYSABLE_SCHEMES:
        return None

    try:
        host = (parts.hostname or "").lower().rstrip(".")
    except ValueError:
        # urlsplit defers some host validation to attribute access.
        return None
    if not host:
        return None

    try:
        port = parts.port
    except ValueError:
        return None

    return scheme, host, port, parts.path, parts.query, parts.fragment


def canonical_form(url: str) -> str | None:
    """A comparison key for a URL, or None when it cannot be parsed.

    Deliberately conservative, and for the same reason `l4_intel.normalize_url`
    is: this exists so two spellings of one place are recognized as one place,
    not to canonicalize the web. It lowercases the scheme and host, drops a
    default port and a trailing dot on the host, and normalizes an empty path
    to "/". It does **not** touch case in the path or query, reorder or drop
    parameters, or unescape anything - all of which can change which resource
    a URL names, and any of which would mean checking a string the message
    never contained.

    The fragment is kept: `l2.base64_email_param` reads it, and two links that
    differ only after the `#` are two different candidates as far as this layer
    is concerned.
    """
    parsed = parse_url(url)
    if parsed is None:
        return None
    scheme, host, port, path, query, fragment = parsed

    authority = host
    if port is not None and DEFAULT_PORTS.get(scheme) != port:
        authority = f"{host}:{port}"

    canonical = f"{scheme}://{authority}{path or '/'}"
    if query:
        canonical = f"{canonical}?{query}"
    if fragment:
        canonical = f"{canonical}#{fragment}"
    return canonical


def _candidate(url: str, source: URLSource, anchor_text: str | None) -> URLCandidate | None:
    """Build one candidate, or None when the URL is not analysable."""
    parsed = parse_url(url)
    if parsed is None:
        return None
    scheme, host, port, path, query, fragment = parsed

    canonical = canonical_form(url)
    if canonical is None:  # pragma: no cover - parse_url already succeeded
        return None

    return URLCandidate(
        raw=url.strip(),
        canonical=canonical,
        scheme=scheme,
        host=host,
        port=port,
        path=path,
        query=query,
        fragment=fragment,
        registrable=registrable_domain(host),
        source=source,
        anchor_text=anchor_text,
        query_params=tuple(parse_qsl(query, keep_blank_values=True)),
        # A fragment carrying `a=b&c=d` is parsed the same way. One without an
        # `=` is an ordinary document anchor (`#section-2`), not a parameter
        # list, and parsing it would invent a parameter named after the anchor.
        # The raw `fragment` is kept either way, which is what a signal
        # scanning for an encoded address actually reads.
        fragment_params=(
            tuple(parse_qsl(fragment, keep_blank_values=True)) if "=" in fragment else ()
        ),
    )


def candidate_urls(
    email: ParsedEmail,
    *,
    extra: Sequence[ExtractedURL] = (),
) -> list[URLCandidate]:
    """The set of URLs Layer 2 analyses, parsed and deduplicated.

    Deduplicates on the canonical form, keeping the first observation, so a
    link repeated in the anchor, the plaintext part and an image src is one
    candidate carrying the anchor text that came with it. That differs from
    `ingest.parser`'s dedup, which keys on `(url, source)` because ingestion
    records *observations*; a signal asks about places, and the same place
    reached three ways is one question.

    `extra` is the seam ARCHITECTURE.md section 4 requires for `l2.qr_url`:
    "extracted URL re-enters the URL signal set". A QR decoder - which this
    module does not implement, and which is the only part of that signal that
    needs an image library - hands its findings in as `ExtractedURL`s with
    `URLSource.QR_CODE`, and they become candidates on exactly the same terms
    as the rest. Nothing here decodes an image, and the parameter is not a
    place for a caller to inject arbitrary analysis.

    Unparseable URLs are dropped rather than carried in a broken state. Never
    raises, and never touches the network.
    """
    candidates: list[URLCandidate] = []
    seen: set[str] = set()

    for extracted in (*email.urls, *extra):
        candidate = _candidate(extracted.url, extracted.source, extracted.anchor_text)
        if candidate is None or candidate.canonical in seen:
            continue
        seen.add(candidate.canonical)
        candidates.append(candidate)

    return candidates
