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
from enum import Enum
from typing import Protocol, Sequence, runtime_checkable
from urllib.parse import parse_qsl, urlsplit

from core.models import ExtractedURL, ParsedEmail, URLSource
from layers.l1_headers import registrable_domain

__all__ = [
    "DEFAULT_PORTS",
    "REDIRECT_FLAG_DEPTH",
    "REDIRECT_HOP_CAP",
    "RedirectFollower",
    "RedirectHop",
    "RedirectOutcome",
    "RedirectTrace",
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


# --------------------------------------------------------------------------
# The redirect result contract
# --------------------------------------------------------------------------
#
# ARCHITECTURE.md section 4 for `l2.redirect_depth`: "Follow hops,
# allow_redirects=False, cap 8. Flag > 2", with a per-hop timeout of 5s and a
# 15s total budget.
#
# **This is the record only. Nothing here follows a redirect.** Hop-following
# is network work that section 4 requires to run from sandboxed egress, and it
# is deliberately not in this module - see the `RedirectFollower` note below.
# What is here is the shape of the answer, so the follower, the signal and the
# tests can all be written against one contract.
#
# Why a separate record rather than filling in `ExtractedURL.redirect_chain`
# and `final_url`. Those fields exist and ingest leaves them empty, but writing
# to them would mean a mutable analysis result living inside the parsed
# message, shared by reference with every other layer, with no way to say "the
# chain is empty because nothing was followed" as distinct from "the chain is
# empty because the URL did not redirect". `URLCandidate` is frozen for the
# same reason. So a trace is its own immutable record, and the caller holds
# `ExtractedURL` and `RedirectTrace` side by side rather than one inside the
# other.

# Section 4: "cap 8" - a hard stop, not a budget to negotiate.
REDIRECT_HOP_CAP = 8

# Section 4: "Flag > 2". The threshold belongs to the signal, not to the
# follower; it lives here because the record is what the signal reads, and a
# constant named once cannot drift between the two.
REDIRECT_FLAG_DEPTH = 2


class RedirectOutcome(str, Enum):
    """How a redirect trace ended. The distinction scoring depends on.

    `SETTLED` and `CAPPED` are answers about the URL. `UNREACHABLE`,
    `TIMED_OUT` and `NOT_ATTEMPTED` are absences of information, and the signal
    that reads them must abstain rather than report a depth of 0 - the same
    rule Layers 1, 3 and 4 already follow. `is_answer` is the one predicate a
    caller needs, so no caller has to enumerate this set correctly.
    """

    # The chain ended at a non-redirect response. The depth is the real depth.
    SETTLED = "settled"

    # The hop cap was reached with the chain still redirecting. The depth is a
    # floor, not a measurement - `depth_is_exact` says so - but a chain that is
    # still redirecting after 8 hops is itself a finding, so this is an answer.
    CAPPED = "capped"

    # A hop could not be completed: refused, DNS failure, TLS failure, a
    # malformed Location, a response the follower could not read.
    UNREACHABLE = "unreachable"

    # The per-hop timeout or the total budget expired.
    TIMED_OUT = "timed_out"

    # No follower was configured, or egress was unavailable. Nothing was tried.
    NOT_ATTEMPTED = "not_attempted"

    @property
    def is_answer(self) -> bool:
        """True when the trace says something about the URL."""
        return self in (RedirectOutcome.SETTLED, RedirectOutcome.CAPPED)


@dataclass(frozen=True)
class RedirectHop:
    """One hop in a chain: a request that answered with a redirect.

    `url` is the URL that was requested, `location` the raw `Location` header
    it answered with, and `target` that header resolved against `url` - kept
    separately because a relative `Location` is normal and the raw header is
    what evidence should quote. `target` is None when the header was absent or
    could not be resolved, which is how a malformed redirect is recorded
    rather than guessed at.
    """

    url: str
    status_code: int
    location: str | None = None
    target: str | None = None
    elapsed_ms: int = 0


@dataclass(frozen=True)
class RedirectTrace:
    """The result of following one URL's redirects. Immutable.

    Neither `ExtractedURL` nor `URLCandidate` is modified to produce this: a
    trace is an observation *about* a URL and is held beside it.

    `url` is the URL the chain started from, exactly as the message contained
    it. `hops` is the redirect responses in order - so `depth` is `len(hops)`,
    and a URL that did not redirect has an empty tuple and a depth of 0, which
    is a genuine measurement rather than a missing one. `final_url` is where
    the chain came to rest, and is None whenever there is no such place -
    including every non-answer outcome, so a caller cannot read a final URL out
    of a trace that never reached one.

    `error` carries why an unsuccessful trace failed, in the same role it has
    on `DetectionSignal`: it is set when and only when the outcome is not an
    answer, so "this chain has depth 0" and "this chain was never followed"
    cannot be confused.
    """

    url: str
    outcome: RedirectOutcome
    hops: tuple[RedirectHop, ...] = ()
    final_url: str | None = None
    error: str | None = None
    elapsed_ms: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, RedirectOutcome):
            raise TypeError(
                f"outcome must be a RedirectOutcome, got {type(self.outcome).__name__}"
            )
        if not self.url or not self.url.strip():
            raise ValueError("url must not be empty")
        if len(self.hops) > REDIRECT_HOP_CAP:
            raise ValueError(
                f"a trace may not exceed the {REDIRECT_HOP_CAP}-hop cap, got {len(self.hops)}"
            )
        if self.outcome is RedirectOutcome.CAPPED and len(self.hops) != REDIRECT_HOP_CAP:
            raise ValueError(
                f"a capped trace has exactly {REDIRECT_HOP_CAP} hops, got {len(self.hops)}"
            )
        if self.outcome.is_answer:
            if self.error is not None:
                raise ValueError("an answered trace must not carry an error")
        else:
            # The invariant that keeps an outage from reading as a depth of 0.
            if not self.error:
                raise ValueError(f"a {self.outcome.value} trace must state why")
            if self.final_url is not None:
                raise ValueError(f"a {self.outcome.value} trace has no final URL")
        if self.outcome is RedirectOutcome.SETTLED and self.final_url is None:
            raise ValueError("a settled trace must record where it came to rest")
        if self.elapsed_ms < 0:
            raise ValueError(f"elapsed_ms must not be negative, got {self.elapsed_ms}")

    @property
    def depth(self) -> int:
        """Number of redirects followed. 0 for a URL that did not redirect."""
        return len(self.hops)

    @property
    def depth_is_exact(self) -> bool:
        """False when the cap stopped the walk, so `depth` is a lower bound."""
        return self.outcome is not RedirectOutcome.CAPPED

    @property
    def is_answer(self) -> bool:
        """True when this trace says something about the URL."""
        return self.outcome.is_answer

    @property
    def chain(self) -> tuple[str, ...]:
        """Every URL visited, start included, in order.

        The shape `ExtractedURL.redirect_chain` is written in, so a caller that
        wants to record the walk on the parsed message can, without this record
        having to reach into it.
        """
        visited = [self.url, *(hop.target for hop in self.hops if hop.target)]
        if self.final_url is not None and (not visited or visited[-1] != self.final_url):
            visited.append(self.final_url)
        return tuple(visited)

    @classmethod
    def not_attempted(cls, url: str, reason: str) -> RedirectTrace:
        """A trace for a URL nothing tried to follow.

        The state the pipeline is in today: there is no follower and no
        sandboxed egress, so this is what `l2.redirect_depth` will read and
        abstain on, rather than reporting every URL as direct.
        """
        return cls(url=url, outcome=RedirectOutcome.NOT_ATTEMPTED, error=reason)


# --------------------------------------------------------------------------
# The egress seam
# --------------------------------------------------------------------------
#
# `RedirectFollower` is the injection point ARCHITECTURE.md section 4's safety
# rules need, and it is the whole of what Layer 2 will know about the network.
# The same shape as `WhoisLookup`, `AITextDetector`, `ThreatIntelSource` and
# `ASNLookup`: a Protocol here, an implementation elsewhere, a fake in tests.
#
# **No implementation of this Protocol exists in this repository**, and adding
# one is not this change. Section 4 requires hop-following to run through
# restricted-network Docker with egress via a VPS or VPN, and states plainly
# that attacker infrastructure must never be fetched from a home or campus IP.
# Until that egress exists, `l2.redirect_depth` reads
# `RedirectTrace.not_attempted` and abstains, which is the honest state - and
# is why the contract above makes abstention impossible to confuse with a
# depth of 0.
#
# Note the contract is total: an implementation returns a `RedirectTrace` for
# every URL and does not raise, because a failed walk is a `RedirectTrace` with
# a non-answer outcome. That differs from the other seams, which signal failure
# by raising, and it is deliberate - a partial chain is itself evidence, and an
# exception would throw away the hops that were completed before the failure.


@runtime_checkable
class RedirectFollower(Protocol):
    """The one network-touching seam in Layer 2's redirect analysis.

    An implementation must honour the section 4 safety rules, none of which
    this module can enforce for it: `allow_redirects=False` so every hop is
    observed rather than collapsed by the client, a hard stop at
    `REDIRECT_HOP_CAP`, a 5s per-hop timeout inside a 15s total budget, no
    downloads executed, and egress through the sandboxed path.

    It returns a `RedirectTrace` in every case, including failure. It must not
    raise: a refused connection, a timeout or a malformed `Location` is a trace
    whose outcome is not an answer, carrying whatever hops were completed
    first.
    """

    def follow(self, url: str) -> RedirectTrace: ...
