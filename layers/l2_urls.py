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

import base64
import binascii
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, Sequence, runtime_checkable
from urllib.parse import parse_qsl, urlsplit

from core.models import (
    DetectionLayer,
    DetectionSignal,
    ExtractedURL,
    ParsedEmail,
    RiskLevel,
    URLSource,
)
from layers.l1_headers import BRAND_DOMAINS, registrable_domain

__all__ = [
    "BASE64_EMAIL_PARAM_SCORE",
    "BRAND_MISMATCH_SCORE",
    "DEFAULT_PORTS",
    "REDIRECT_CAPPED_SCORE",
    "REDIRECT_DEPTH_SCORE",
    "REDIRECT_FLAG_DEPTH",
    "REDIRECT_HOP_CAP",
    "RedirectFollower",
    "RedirectHop",
    "RedirectOutcome",
    "RedirectTrace",
    "URLCandidate",
    "analyze_base64_email_param",
    "analyze_domain_mismatch_brand",
    "brands_named_in",
    "analyze_redirect_depth",
    "decode_base64_email",
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


# --------------------------------------------------------------------------
# l2.redirect_depth
# --------------------------------------------------------------------------
#
# ARCHITECTURE.md section 4: "Follow hops, allow_redirects=False, cap 8.
# Flag > 2."
#
# This function follows nothing. It reads `RedirectTrace`s produced by an
# injected `RedirectFollower` and turns them into one signal - the same
# division Layer 1 uses between `analyze_domain_age` and `WhoisLookup`, and
# Layer 4 between `analyze_url_ioc` and `ThreatIntelSource`. No follower is
# constructed here, so the signal is fully exercisable with fakes and there is
# no path by which running the unit suite reaches the network.

# Hand-assigned for Phase A exactly as the Layer 1 and Layer 4 scores are, and
# refitted in Phase 5 against the labelled corpus.
#
# A chain of three or more hops is a moderate indicator: link-shortener stacks
# and marketing trackers legitimately produce them, so this sits with
# `l1.replyto_mismatch` (0.55) rather than with a confirmed IOC listing.
REDIRECT_DEPTH_SCORE = 0.55

# A chain *still redirecting* after eight hops is a stronger statement than a
# measured three. Nothing legitimate needs nine hops, and the reason the depth
# is inexact - the walk was stopped, not finished - is itself the finding.
REDIRECT_CAPPED_SCORE = 0.70

# Verdict bands from config/weights.yaml, applied to this signal's own score so
# its severity means what the composite's deliver/warn/block bands mean.
_WARN_THRESHOLD = 0.35
_BLOCK_THRESHOLD = 0.65


def _severity(score: float) -> RiskLevel:
    if score >= _BLOCK_THRESHOLD:
        return RiskLevel.HIGH
    if score >= _WARN_THRESHOLD:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _redirect_signal(
    score: float,
    severity: RiskLevel,
    evidence: str,
    *,
    metadata: dict[str, object],
    error: str | None = None,
) -> DetectionSignal:
    """Build the signal, recording `fired` as every other layer here does."""
    return DetectionSignal(
        layer=DetectionLayer.L2,
        name="redirect_depth",
        score=score,
        severity=severity,
        evidence=evidence,
        metadata={**metadata, "fired": score > 0.0 and error is None},
        error=error,
    )


def _describe(trace: RedirectTrace) -> dict[str, object]:
    """One trace as metadata. `exact` is what keeps a cap from reading as 8."""
    return {
        "url": trace.url,
        "outcome": trace.outcome.value,
        "depth": trace.depth,
        "exact": trace.depth_is_exact,
        "final_url": trace.final_url,
        "chain": list(trace.chain),
        "error": trace.error,
    }


def analyze_redirect_depth(
    email: ParsedEmail,
    *,
    follower: RedirectFollower | None = None,
    extra: Sequence[ExtractedURL] = (),
) -> DetectionSignal:
    """Flag messages whose links redirect more than twice before landing.

    Each candidate URL is walked independently, so one unreachable link does
    not stop the others being analysed and does not suppress a finding on a
    link that was walked successfully.

    **A capped chain is never reported as a measured depth of 8.** It is an
    answer - a chain still redirecting at the cap is a finding in its own right
    - but `depth` is a floor, and every place the number appears carries
    `exact: false` beside it in the metadata and "at least" in the evidence.

    Abstains, rather than reporting a clean result, when no URL could be walked
    - no follower configured, every walk unreachable or timed out - and also
    when nothing was found but some URL went unchecked, because "none of these
    links redirect" is a claim about links that were actually followed. A
    partial chain that failed mid-walk is an abstention carrying the hops it
    managed, never a depth-0 clean result.

    Evidence quotes `candidate.raw`, the URL as the message contained it, not
    the canonical form used for deduplication - reporting a rewritten URL would
    be reporting something the message did not say.

    Never raises. Never opens a socket, resolves a name or follows a redirect:
    all of that is the injected follower's job, behind the seam.
    """
    candidates = candidate_urls(email, extra=extra)

    metadata: dict[str, object] = {
        "urls_in_message": len(email.urls),
        "urls_checked": 0,
        "flag_depth": REDIRECT_FLAG_DEPTH,
        "hop_cap": REDIRECT_HOP_CAP,
        "traces": [],
        "findings": [],
        "abstentions": {},
        "deepest_exact": None,
    }

    if not candidates:
        # Genuine negative: the check ran over the zero URLs this message has.
        return _redirect_signal(
            0.0,
            RiskLevel.LOW,
            "The message contains no analysable URLs, so there was no redirect "
            "chain to follow.",
            metadata=metadata,
        )

    metadata["urls_checked"] = len(candidates)

    traces: list[RedirectTrace] = []
    for candidate in candidates:
        if follower is None:
            traces.append(
                RedirectTrace.not_attempted(candidate.raw, "no redirect follower configured")
            )
            continue
        try:
            trace = follower.follow(candidate.raw)
        except Exception as exc:  # noqa: BLE001 - the seam forbids this, so a
            # follower that raises is broken rather than authoritative.
            trace = RedirectTrace(
                url=candidate.raw,
                outcome=RedirectOutcome.UNREACHABLE,
                error=f"follower raised {type(exc).__name__}: {exc}",
            )
        if not isinstance(trace, RedirectTrace):
            trace = RedirectTrace(
                url=candidate.raw,
                outcome=RedirectOutcome.UNREACHABLE,
                error=f"follower returned {type(trace).__name__}, not a RedirectTrace",
            )
        traces.append(trace)

    answered = [trace for trace in traces if trace.is_answer]
    abstained = {trace.url: (trace.error or "not followed") for trace in traces if not trace.is_answer}
    findings = [trace for trace in answered if trace.depth > REDIRECT_FLAG_DEPTH]

    exact_depths = [trace.depth for trace in answered if trace.depth_is_exact]
    metadata.update(
        traces=[_describe(trace) for trace in traces],
        findings=[_describe(trace) for trace in findings],
        abstentions=abstained,
        deepest_exact=max(exact_depths) if exact_depths else None,
    )

    if findings:
        capped = [trace for trace in findings if not trace.depth_is_exact]
        score = REDIRECT_CAPPED_SCORE if capped else REDIRECT_DEPTH_SCORE
        described = "; ".join(
            f"{trace.url} redirects "
            f"{'at least ' if not trace.depth_is_exact else ''}{trace.depth} times"
            + ("" if trace.depth_is_exact else f" and was still redirecting at the "
               f"{REDIRECT_HOP_CAP}-hop cap")
            for trace in findings[:3]
        )
        more = f", and {len(findings) - 3} more" if len(findings) > 3 else ""
        caveat = (
            f" {len(abstained)} other URL(s) could not be followed."
            if abstained
            else ""
        )
        return _redirect_signal(
            score,
            _severity(score),
            f"{len(findings)} of the {len(candidates)} URL(s) in this message "
            f"redirect more than {REDIRECT_FLAG_DEPTH} times before landing: "
            f"{described}{more}.{caveat}",
            metadata=metadata,
        )

    if not answered:
        reasons = "; ".join(f"{url}: {reason}" for url, reason in abstained.items())
        return _redirect_signal(
            0.0,
            RiskLevel.LOW,
            f"None of the {len(candidates)} URL(s) in this message could be "
            f"followed ({reasons}), so their redirect chains were not measured. "
            f"This is an absence of information, not a clean result.",
            metadata=metadata,
            error=f"no URL could be followed ({reasons})",
        )

    if abstained:
        # Some links were walked and were shallow, but others were not walked
        # at all. "No link redirects more than twice" would be a claim about
        # URLs nobody followed.
        reasons = "; ".join(f"{url}: {reason}" for url, reason in abstained.items())
        return _redirect_signal(
            0.0,
            RiskLevel.LOW,
            f"{len(abstained)} of the {len(candidates)} URL(s) in this message "
            f"could not be followed ({reasons}), so the message was not fully "
            f"checked. This is an absence of information, not a clean result.",
            metadata=metadata,
            error=f"{len(abstained)} of {len(candidates)} URL(s) could not be followed",
        )

    deepest = max(trace.depth for trace in answered)
    return _redirect_signal(
        0.0,
        RiskLevel.LOW,
        f"All {len(candidates)} URL(s) in this message settle within "
        f"{REDIRECT_FLAG_DEPTH} redirect(s); the longest chain is {deepest}.",
        metadata=metadata,
    )


# --------------------------------------------------------------------------
# l2.base64_email_param
# --------------------------------------------------------------------------
#
# ARCHITECTURE.md section 4: "Regex for base64-encoded address in
# query/fragment; decode and confirm it parses as an email."
#
# Why this is worth a signal at all: a phishing link that already knows who it
# was sent to is a link built for one recipient. Tycoon-style kits carry the
# victim's address in the URL so the landing page can pre-fill the login form,
# which is both what makes the page convincing and what makes the link
# self-identifying. A legitimate marketing tracker encodes an opaque
# subscriber id, not a decodable mailbox.
#
# The spec's own false-positive control is the second half of the sentence:
# **decode and confirm it parses as an email**. Arbitrary base64 is ordinary in
# URLs - session tokens, encoded return paths, cache keys - so the finding is
# not "this looks encoded", it is "this decodes to an address". Nothing here
# fires on base64 alone.
#
# Entirely offline: string work over `URLCandidate` fields. No DNS, no request,
# and in particular no attempt to verify the decoded address exists.

# The base64 alphabet as it appears in a URL parameter. Standard alphabet only
# - `+/=` - because that is what section 4 says and nothing in this project
# requires the URL-safe `-_` variant. Padding arrives already percent-decoded
# by `parse_qsl`, so `%3D` is an `=` by the time it is seen here.
#
# The 12-character floor is not a heuristic about suspiciousness: it is the
# shortest input that can decode to anything email-shaped at all. The shortest
# plausible address is `a@b.co` at 6 bytes, which is 8 base64 characters, and
# 12 characters decode to 9 bytes - below that the regex cannot match a string
# that survives the email check anyway, and matching shorter would only mean
# decoding more noise to throw it away.
_BASE64_CANDIDATE_RE = re.compile(r"^[A-Za-z0-9+/]{12,}={0,2}$")

# The decoded value must parse as one address. Deliberately strict and
# deliberately not an RFC 5322 implementation: `email.utils.parseaddr` accepts
# a great deal that is not an address, so the shape is checked explicitly and
# the registrable domain must resolve against the vendored public suffix list,
# which is the same bar `l1.replyto_mismatch` applies. `x@localhost` and
# `a@b` therefore do not count, and neither does a sentence containing an `@`.
_DECODED_EMAIL_RE = re.compile(r"^[^\s@<>\"',;:\\]{1,64}@[A-Za-z0-9.-]{1,255}$")

# **Not specified by ARCHITECTURE.md.** Section 4 names the method for every
# signal but assigns a score to none of them, and config/weights.yaml carries
# layer weights and verdict bands only - no per-signal values. This number is
# therefore hand-assigned for Phase A on the same terms as every other score in
# this project (`_FAIL_SCORES` in Layer 1, `URL_IOC_SCORE` in Layer 4), and is
# refitted in Phase 5 against the labelled corpus.
#
# Placed at 0.70 - above `l2.redirect_depth`'s 0.55, below a confirmed
# threat-intel listing's 0.95. A link that carries the recipient's own address
# is close to conclusive evidence of targeting, but it is not proof of intent:
# some legitimate unsubscribe and preference-centre links do exactly this,
# which is precisely why it is not scored higher.
BASE64_EMAIL_PARAM_SCORE = 0.70


@dataclass(frozen=True)
class _EncodedAddress:
    """One decoded hit: where it was found, what it decoded to."""

    location: str  # "query" or "fragment"
    parameter: str | None  # None when the whole fragment carried the value
    encoded: str
    email: str


def decode_base64_email(value: str) -> str | None:
    """Decode `value` and return the email address it holds, or None.

    None - never an exception - for everything that is not the thing section 4
    describes: a value that is not base64-shaped, base64 that will not decode,
    bytes that are not UTF-8 text, and text that is not a single address on a
    domain with a public suffix. That last check is what stops arbitrary
    decodable base64 from becoming a finding.
    """
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not _BASE64_CANDIDATE_RE.match(candidate):
        return None

    try:
        raw = base64.b64decode(candidate, validate=True)
    except (binascii.Error, ValueError):
        return None

    try:
        decoded = raw.decode("utf-8").strip()
    except UnicodeDecodeError:
        return None

    if not _DECODED_EMAIL_RE.match(decoded):
        return None
    # An address whose domain has no public suffix is not one this project
    # treats as an address anywhere else - the same rule Layer 1 applies.
    if registrable_domain(decoded) is None:
        return None
    return decoded


def _encoded_addresses(candidate: URLCandidate) -> list[_EncodedAddress]:
    """Every decodable address carried by one URL's query and fragment.

    Both surfaces are inspected because section 4 names both. Values come from
    the already-parsed `query_params` / `fragment_params`, so a blank parameter
    stays visible as a present-but-empty value rather than vanishing - it
    simply cannot decode to anything.

    The bare fragment is inspected too, always: a kit that appends
    `#dGVzdEBleGFtcGxlLmNvbQ==` is putting the address in the fragment exactly
    as the spec describes. It is reported with `parameter: None` so evidence
    can say where it was. Note that base64 padding is an `=`, so such a
    fragment *also* parses as a one-entry parameter list with a blank value -
    which decodes to nothing, so the bare check is what actually finds it and
    no hit is reported twice.
    """
    found: list[_EncodedAddress] = []

    for location, params in (
        ("query", candidate.query_params),
        ("fragment", candidate.fragment_params),
    ):
        for name, value in params:
            email = decode_base64_email(value)
            if email is not None:
                found.append(
                    _EncodedAddress(
                        location=location, parameter=name, encoded=value, email=email
                    )
                )

    # Always, not only when the fragment held no parameters: base64 padding is
    # itself an `=`, so `#dGhpcmRAY29ycC5jb20=` parses as a parameter list with
    # one blank value *and* is a bare encoded address. Checking only one of the
    # two would miss whichever case the padding happened to produce.
    if candidate.fragment:
        email = decode_base64_email(candidate.fragment)
        if email is not None:
            found.append(
                _EncodedAddress(
                    location="fragment",
                    parameter=None,
                    encoded=candidate.fragment.strip(),
                    email=email,
                )
            )

    return found


def analyze_base64_email_param(
    email: ParsedEmail,
    *,
    extra: Sequence[ExtractedURL] = (),
) -> DetectionSignal:
    """Flag URLs carrying a base64-encoded recipient address.

    Each candidate URL is inspected independently over its query and fragment.
    A URL that cannot be parsed never becomes a candidate, so a malformed href
    costs that href and nothing else.

    Offline and total: `decode_base64_email` returns None rather than raising
    for every malformed input, so no parameter value can fail the analysis.

    Evidence quotes `candidate.raw`, the URL as the message contained it. The
    metadata carries the location, the parameter name and the decoded address -
    the finding is unreadable without them - and deliberately nothing else
    about the URL: the other parameters are not this signal's business and may
    hold session tokens.

    There is no abstention path. Every input this signal needs is already in
    the parsed message, so a URL either carries a decodable address or it does
    not, and "no encoded address" is a genuine negative rather than an absence
    of information. That is the whole difference between an offline signal and
    `l2.redirect_depth`.
    """
    candidates = candidate_urls(email, extra=extra)

    metadata: dict[str, object] = {
        "urls_in_message": len(email.urls),
        "urls_checked": len(candidates),
        "matches": [],
    }

    if not candidates:
        return _base64_signal(
            0.0,
            RiskLevel.LOW,
            "The message contains no analysable URLs, so none could carry an "
            "encoded address.",
            metadata=metadata,
        )

    matches: list[dict[str, object]] = []
    for candidate in candidates:
        for hit in _encoded_addresses(candidate):
            matches.append(
                {
                    "url": candidate.raw,
                    "location": hit.location,
                    "parameter": hit.parameter,
                    "encoded": hit.encoded,
                    "email": hit.email,
                }
            )

    metadata["matches"] = matches

    if not matches:
        return _base64_signal(
            0.0,
            RiskLevel.LOW,
            f"None of the {len(candidates)} URL(s) in this message carry a "
            f"base64-encoded email address in their query or fragment.",
            metadata=metadata,
        )

    described = "; ".join(
        f"{match['url']} carries {match['email']} base64-encoded in "
        + (
            f"the {match['location']} parameter {match['parameter']!r}"
            if match["parameter"] is not None
            else f"the {match['location']}"
        )
        for match in matches[:3]
    )
    more = f", and {len(matches) - 3} more" if len(matches) > 3 else ""
    return _base64_signal(
        BASE64_EMAIL_PARAM_SCORE,
        _severity(BASE64_EMAIL_PARAM_SCORE),
        f"{len(matches)} link parameter(s) in this message encode a recipient "
        f"address, so the link identifies who it was sent to: {described}{more}.",
        metadata=metadata,
    )


def _base64_signal(
    score: float,
    severity: RiskLevel,
    evidence: str,
    *,
    metadata: dict[str, object],
    error: str | None = None,
) -> DetectionSignal:
    return DetectionSignal(
        layer=DetectionLayer.L2,
        name="base64_email_param",
        score=score,
        severity=severity,
        evidence=evidence,
        metadata={**metadata, "fired": score > 0.0 and error is None},
        error=error,
    )


# --------------------------------------------------------------------------
# l2.domain_mismatch_brand
# --------------------------------------------------------------------------
#
# ARCHITECTURE.md section 4: "Anchor text names a brand, href points
# elsewhere."
#
# The brand corpus is `l1_headers.BRAND_DOMAINS`, imported rather than
# duplicated. It is already exactly the mapping this signal needs - brand token
# to the set of registrable domains that brand legitimately uses - and it is
# the only brand data in the repository. A second list would drift from the
# first, and the two signals would then disagree about who Microsoft is.
#
# Section 4 does not say which corpus Layer 2 should use; see the report. The
# reuse is a judgement, but it is the only one available that does not mean
# inventing a brand database.
#
# The rule is the same shape as `l1.display_name_impersonation`, deliberately:
# a fixed table, exact token matching after a fixed normalization, no edit
# distance and no similarity threshold. Fuzzy matching loose enough to catch
# real attacks is loose enough to call "Apple Valley Dental" an impersonation.
#
# **Why this does not fire on every anchor/domain mismatch.** A legitimate
# branded link routinely points at a third party - an ESP click-tracker, a
# CDN, a survey host - so "the domains differ" is not the finding. Two
# restraints, both taken from the Layer 1 signal's existing behaviour:
#
#   1. The anchor must *name* a brand as a token, not merely contain the
#      letters. "Click here to view your invoice" names nothing.
#   2. A destination whose own registrable domain carries the brand token does
#      not fire. `microsoft.com.evil.example` is caught by the domain check;
#      `microsoft-partner.co.uk` degrades to silence rather than to a false
#      accusation, which is how the Layer 1 signal handles a legitimate but
#      unlisted brand domain.
#
# That leaves a real residual false-positive surface - a genuine Microsoft mail
# whose link goes through an unlisted tracker still fires - which is why the
# score below is corroborating rather than conclusive.

# Same normalization as the Layer 1 signal's, applied to anchor text instead of
# to a display name. Not imported: the Layer 1 helpers are private to that
# module, and importing them would couple two signals through an unpublished
# interface. `BRAND_DOMAINS` is the shared thing, and it is public.
_ANCHOR_SPLIT_RE = re.compile(r"[^a-z0-9]+")

# **Not specified by ARCHITECTURE.md.** Section 4's Layer 2 table has no
# severity column and config/weights.yaml carries no per-signal scores, so this
# is hand-assigned for Phase A on the same terms as every other score in this
# project and refitted in Phase 5.
#
# 0.60, below the 0.65 block band and below `l2.base64_email_param`'s 0.70. A
# branded anchor pointing off-brand is a classic phishing shape, but the
# residual false positive above is real and common in legitimate bulk mail, so
# this corroborates rather than convicts. Note section 4 rates the analogous
# `l1.display_name_impersonation` "medium", which is the band this lands in.
BRAND_MISMATCH_SCORE = 0.60


def brands_named_in(anchor_text: str | None) -> list[str]:
    """The brands an anchor text names, in `BRAND_DOMAINS` order.

    Tokens are the folded words, adjacent pairs and the whole string joined -
    so "Micro Soft" and "Pay Pal" are caught, without matching two words that
    merely both appear somewhere in a long sentence. Deterministic, and empty
    for None, blank, or text naming nothing.
    """
    if not anchor_text or not anchor_text.strip():
        return []

    words = [w for w in _ANCHOR_SPLIT_RE.split(anchor_text.lower()) if w]
    if not words:
        return []

    tokens = set(words)
    tokens.update(a + b for a, b in zip(words, words[1:]))
    if len(words) > 1:
        tokens.add("".join(words))

    return [brand for brand in BRAND_DOMAINS if brand in tokens]


def _brand_verdict(candidate: URLCandidate) -> tuple[str, dict[str, object]] | None:
    """Classify one candidate, or None when its anchor names no brand.

    Returns `(verdict, detail)` where verdict is "mismatch", "legitimate" or
    "uncomparable" - the last when the anchor claims a brand but the href has
    no registrable domain to check it against, which is an absence of
    information rather than a clean link.
    """
    brands = brands_named_in(candidate.anchor_text)
    if not brands:
        return None

    detail: dict[str, object] = {
        "url": candidate.raw,
        "anchor_text": (candidate.anchor_text or "").strip(),
        "brands": brands,
        "destination": candidate.registrable,
    }

    if candidate.registrable is None:
        return "uncomparable", detail

    for brand in brands:
        if candidate.registrable in BRAND_DOMAINS[brand]:
            return "legitimate", {**detail, "matched_brand": brand}
        if brand in candidate.registrable:
            # The destination carries the brand token itself: an unlisted but
            # plausibly legitimate domain. Silence, not an accusation.
            return "legitimate", {**detail, "matched_brand": brand}

    return "mismatch", detail


def analyze_domain_mismatch_brand(
    email: ParsedEmail,
    *,
    extra: Sequence[ExtractedURL] = (),
) -> DetectionSignal:
    """Flag links whose anchor text names a brand the destination is not.

    "Sign in to Microsoft" pointing at `secure-login.tk` fires; the same anchor
    pointing at `microsoftonline.com` does not, and neither does "Click here"
    pointing anywhere. Each candidate is judged independently, so one link with
    no anchor text costs that link alone.

    Offline: a fixed table and exact token matching. No DNS, no request, no
    rendering. Never raises.

    Evidence names the brand the anchor claimed, the registrable domain the
    href actually points to, and why those disagree. It quotes `candidate.raw`
    and the anchor text and nothing else about the URL - query parameters are
    not this signal's business and may carry tokens.

    Abstains when a branded anchor's destination has no registrable domain to
    compare against - an IP-literal href under a brand anchor is not a clean
    link, it is an unanswerable one - and only when no other link produced a
    finding.
    """
    candidates = candidate_urls(email, extra=extra)

    metadata: dict[str, object] = {
        "urls_in_message": len(email.urls),
        "urls_checked": len(candidates),
        "branded_anchors": 0,
        "matches": [],
        "legitimate": [],
        "uncomparable": [],
    }

    if not candidates:
        return _brand_signal(
            0.0,
            RiskLevel.LOW,
            "The message contains no analysable URLs, so no anchor text could "
            "misrepresent a destination.",
            metadata=metadata,
        )

    matches: list[dict[str, object]] = []
    legitimate: list[dict[str, object]] = []
    uncomparable: list[dict[str, object]] = []

    for candidate in candidates:
        verdict = _brand_verdict(candidate)
        if verdict is None:
            continue
        outcome, detail = verdict
        if outcome == "mismatch":
            matches.append(detail)
        elif outcome == "legitimate":
            legitimate.append(detail)
        else:
            uncomparable.append(detail)

    metadata.update(
        branded_anchors=len(matches) + len(legitimate) + len(uncomparable),
        matches=matches,
        legitimate=legitimate,
        uncomparable=uncomparable,
    )

    if matches:
        described = "; ".join(
            f"{match['anchor_text']!r} names {match['brands'][0]} but "
            f"{match['url']} points to {match['destination']}"
            for match in matches[:3]
        )
        more = f", and {len(matches) - 3} more" if len(matches) > 3 else ""
        return _brand_signal(
            BRAND_MISMATCH_SCORE,
            _severity(BRAND_MISMATCH_SCORE),
            f"{len(matches)} link(s) in this message name a brand in their text "
            f"while pointing at a domain that brand does not use: {described}"
            f"{more}.",
            metadata=metadata,
        )

    if uncomparable:
        listed = "; ".join(
            f"{item['anchor_text']!r} names {item['brands'][0]} but {item['url']} "
            f"has no registrable domain"
            for item in uncomparable[:3]
        )
        return _brand_signal(
            0.0,
            RiskLevel.LOW,
            f"{len(uncomparable)} branded link(s) point at a destination with no "
            f"registrable domain, so the claim could not be checked ({listed}). "
            f"This is an absence of information, not a clean result.",
            metadata=metadata,
            error=f"{len(uncomparable)} branded link(s) had no comparable domain",
        )

    if legitimate:
        return _brand_signal(
            0.0,
            RiskLevel.LOW,
            f"{len(legitimate)} link(s) name a brand and point at a domain that "
            f"brand uses; none misrepresents its destination.",
            metadata=metadata,
        )

    return _brand_signal(
        0.0,
        RiskLevel.LOW,
        f"No link text in this message names a known brand, so none could "
        f"misrepresent its destination.",
        metadata=metadata,
    )


def _brand_signal(
    score: float,
    severity: RiskLevel,
    evidence: str,
    *,
    metadata: dict[str, object],
    error: str | None = None,
) -> DetectionSignal:
    return DetectionSignal(
        layer=DetectionLayer.L2,
        name="domain_mismatch_brand",
        score=score,
        severity=severity,
        evidence=evidence,
        metadata={**metadata, "fired": score > 0.0 and error is None},
        error=error,
    )
