"""Layer 1 - header and domain intelligence.

Signals specified in ARCHITECTURE.md section 4:
  l1.auth_fail                   read Gmail's Authentication-Results verdicts
  l1.replyto_mismatch            registrable domain of Reply-To != From
  l1.domain_age_lt_7d            WHOIS creation date (cached, 7-day TTL)
  l1.display_name_impersonation  brand token in display name, not in domain

All four are implemented. `analyze` runs the whole layer and is what the
orchestrator calls; the per-signal entry points stay public because each is
independently meaningful and independently testable.

Every signal accepts either a `ParsedEmail` or an `IngestedMessage`. The
latter is the ingestion boundary's own type and is preferred - it carries
`auth_headers()`, Gmail's already-extracted authentication headers - but the
orchestrator's `LayerCallable` contract is written in terms of `ParsedEmail`,
so both are read rather than forcing a change upstream.

Authentication-Results
----------------------
This module **reads verdicts, it does not compute them.** Per ARCHITECTURE.md
section 4, SPF and DKIM are deliberately not re-verified cryptographically:
the receiving MTA already did that work with the original connecting IP in
hand, and recorded the outcome in an `Authentication-Results` header
(RFC 8601). Re-deriving it here would cost more and produce a worse answer.
Consequently this module performs no DNS lookup, no signature check and no
network I/O of any kind - it is pure string parsing over headers the parser
already extracted.

One signal per method rather than a single aggregate `auth_fail`: SPF, DKIM
and DMARC fail independently and for different reasons, and a message can
easily pass one, fail another and have no verdict at all for the third. A
single signal cannot express "DKIM failed, DMARC absent" as simultaneously a
finding and an abstention, and the evidence string is what the UI shows.

Absence is not failure
----------------------
A method with no recorded verdict abstains: it emits a signal carrying
`error`, which `scoring/composite.py` excludes from the score rather than
counting as a 0.0 finding. A header that says `spf=pass` and a header that
says nothing about SPF at all are different facts, and collapsing the second
into "no threat found" is the failure mode ARCHITECTURE.md section 2 forbids.

`temperror` and `permerror` abstain for the same reason: the check could not
be completed, so there is no verdict to report either way.
"""

from __future__ import annotations

import asyncio
import os
import re
import threading
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

import tldextract

from core.models import DetectionLayer, DetectionSignal, ParsedEmail, RiskLevel

__all__ = [
    "AUTH_METHODS",
    "BRAND_DOMAINS",
    "DEFAULT_WHOIS_TIMEOUT",
    "WHOIS_UNAVAILABLE_TTL_SECONDS",
    "AuthVerdict",
    "DomainAge",
    "WhoisLookup",
    "WhoisTimeout",
    "WhoisUnavailable",
    "PythonWhoisLookup",
    "TimeLimitedWhoisLookup",
    "analyze",
    "analyze_async",
    "analyze_authentication_results",
    "analyze_display_name_impersonation",
    "analyze_domain_age",
    "analyze_reply_to_mismatch",
    "default_cache",
    "default_whois_lookup",
    "parse_authentication_results",
    "registrable_domain",
    "reset_default_cache",
]

# The methods reported on, in the order their signals are emitted.
AUTH_METHODS = ("spf", "dkim", "dmarc")

_HEADER_NAME = "authentication-results"

# One resinfo chunk: method[/version] = result. Trailing propspecs
# (header.i=, smtp.mailfrom=, ...) are matched by the caller's chunking and
# deliberately ignored - the verdict is the only thing being read.
_RESINFO_RE = re.compile(
    r"^\s*(?P<method>[A-Za-z][A-Za-z0-9-]*)\s*(?:/\s*\d+\s*)?=\s*(?P<result>[A-Za-z]+)"
)

# RFC 8601 result values, plus how each is treated. A result absent from this
# table is unrecognized and abstains rather than being guessed at.
_FAILING = {"fail", "softfail"}
_INCONCLUSIVE = {"temperror", "permerror"}
_BENIGN = {"pass", "none", "neutral", "policy"}

# Which verdict wins when several are recorded for one method - see
# `_worst`. Ordered so the most suspicious observation survives.
_SEVERITY_RANK = {
    "pass": 0,
    "none": 1,
    "neutral": 1,
    "policy": 1,
    "temperror": 2,
    "permerror": 2,
    "softfail": 3,
    "fail": 4,
}
_UNKNOWN_RANK = 2

# Per-method scoring for a hard failure. Hand-assigned for Phase A exactly as
# the composite weights are, and refitted in Phase 5 against the labelled
# corpus. DMARC is the strongest of the three because it is the one that
# expresses the domain owner's own published policy.
_FAIL_SCORES = {"spf": 0.60, "dkim": 0.60, "dmarc": 0.85}
_SOFTFAIL_SCORE = 0.30

_METHOD_LABELS = {"spf": "SPF", "dkim": "DKIM", "dmarc": "DMARC"}


@dataclass(frozen=True)
class AuthVerdict:
    """One method's recorded outcome, as read from the headers.

    `result` is lower-cased. `authserv_id` is the identity of the MTA that
    performed the check, which is what makes the evidence string traceable.
    `all_results` keeps every result seen for this method, so a conflict
    between two headers can be shown rather than silently resolved.
    """

    method: str
    result: str
    authserv_id: str | None = None
    all_results: tuple[str, ...] = ()

    @property
    def failed(self) -> bool:
        return self.result in _FAILING

    @property
    def inconclusive(self) -> bool:
        """True when the check could not be completed, or was not recognized."""
        return self.result in _INCONCLUSIVE or self.result not in _SEVERITY_RANK


# --------------------------------------------------------------------------
# Header parsing
# --------------------------------------------------------------------------


def _strip_comments(value: str) -> str:
    """Remove RFC 5322 parenthesized comments, honouring nesting and escapes.

    Comments routinely contain semicolons - Google writes
    `spf=pass (google.com: domain of x designates ...)` - so they have to go
    before the header can be split into resinfo chunks.
    """
    out: list[str] = []
    depth = 0
    i = 0
    while i < len(value):
        char = value[i]
        if char == "\\" and depth and i + 1 < len(value):
            i += 2  # escaped char inside a comment
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(char)
        i += 1
    return "".join(out)


def _parse_one_header(value: str) -> tuple[str | None, list[tuple[str, str]]]:
    """Parse one header into (authserv-id, [(method, result), ...]).

    Never raises. A chunk that does not look like a resinfo is skipped, so a
    malformed fragment costs that fragment and nothing else.
    """
    chunks = _strip_comments(value).split(";")
    if not chunks:
        return None, []

    # The authserv-id is the first chunk, optionally followed by a version.
    head = chunks[0].strip()
    authserv_id = head.split()[0] if head else None

    found: list[tuple[str, str]] = []
    for chunk in chunks[1:]:
        match = _RESINFO_RE.match(chunk)
        if match is None:
            continue  # propspec continuation, junk, or empty - not a verdict
        found.append((match.group("method").lower(), match.group("result").lower()))

    # A header may omit the authserv-id and lead with a resinfo. Recover that
    # case rather than silently discarding the first verdict.
    if head:
        leading = _RESINFO_RE.match(head)
        if leading is not None:
            authserv_id = None
            found.insert(
                0, (leading.group("method").lower(), leading.group("result").lower())
            )

    return authserv_id, found


def _worst(results: list[str]) -> str:
    """The most suspicious of several results recorded for one method.

    Taking the worst rather than the first is deliberate. Multiple DKIM
    signatures legitimately produce several verdicts, and an attacker can
    prepend a forged `Authentication-Results` header claiming a pass - the
    receiving MTA's own header is not necessarily the one read first here.
    Preferring the failure means a forged pass cannot mask a real failure.
    """
    return max(results, key=lambda r: _SEVERITY_RANK.get(r, _UNKNOWN_RANK))


def _parsed_email(source: object) -> ParsedEmail:
    """The `ParsedEmail` inside a message, whichever type was handed in.

    Accepting both keeps `IngestedMessage` the ingestion boundary's own type
    without rewriting the orchestrator's `ParsedEmail`-shaped contract.
    """
    if isinstance(source, ParsedEmail):
        return source
    email = getattr(source, "email", None)
    if isinstance(email, ParsedEmail):
        return email
    raise TypeError(
        "expected a ParsedEmail or an IngestedMessage, got "
        f"{type(source).__name__}"
    )


def _auth_result_headers(source: object) -> list[str]:
    """Every `Authentication-Results` header, from the best available source.

    An `IngestedMessage` is asked via `auth_headers()`, which is ingestion's
    own accessor for exactly these headers and already returns every
    occurrence. A bare `ParsedEmail` is read directly. Either way the result is
    the full list: a message legitimately carries several, and the disagreement
    between them is the thing worth seeing.
    """
    getter = getattr(source, "auth_headers", None)
    if callable(getter):
        try:
            headers = getter()
        except Exception:  # noqa: BLE001 - untrusted message, never fatal
            headers = None
        if isinstance(headers, dict):
            values = headers.get(_HEADER_NAME) or []
            return [v for v in values if isinstance(v, str)]

    values = _parsed_email(source).headers.get(_HEADER_NAME, [])
    return [v for v in values if isinstance(v, str)]


def parse_authentication_results(source: ParsedEmail | object) -> dict[str, AuthVerdict]:
    """Read every Authentication-Results header into per-method verdicts.

    Methods with no recorded verdict are simply absent from the mapping; this
    function does not invent a result for them. Method names and result values
    are lower-cased, so `SPF=Pass` and `spf=pass` are the same fact.

    Accepts a `ParsedEmail` or an `IngestedMessage`; the latter is read through
    `auth_headers()`.
    """
    headers = _auth_result_headers(source)

    collected: dict[str, list[str]] = {}
    authserv_for: dict[str, str] = {}
    for header in headers:
        try:
            authserv_id, results = _parse_one_header(header)
        except Exception:  # noqa: BLE001 - untrusted header, never fatal
            continue
        for method, result in results:
            collected.setdefault(method, []).append(result)
            if authserv_id and method not in authserv_for:
                authserv_for[method] = authserv_id

    return {
        method: AuthVerdict(
            method=method,
            result=_worst(results),
            authserv_id=authserv_for.get(method),
            all_results=tuple(results),
        )
        for method, results in collected.items()
    }


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------


def _reported_by(verdict: AuthVerdict) -> str:
    return f" as reported by {verdict.authserv_id}" if verdict.authserv_id else ""


def _conflict_note(verdict: AuthVerdict) -> str:
    """Names the other results when one method carries several."""
    if len(set(verdict.all_results)) <= 1:
        return ""
    seen = ", ".join(verdict.all_results)
    return f" This method reported several results ({seen}); the most severe was used."


def _signal(
    method: str,
    score: float,
    severity: RiskLevel,
    evidence: str,
    *,
    verdict: AuthVerdict | None = None,
    error: str | None = None,
) -> DetectionSignal:
    metadata: dict[str, object] = {"method": method}
    if verdict is not None:
        metadata.update(
            result=verdict.result,
            authserv_id=verdict.authserv_id,
            all_results=list(verdict.all_results),
        )
    # Same `fired` convention as the domain signals below, so every Layer 1
    # signal answers "did this fire" the same way.
    metadata["fired"] = score > 0.0 and error is None
    return DetectionSignal(
        layer=DetectionLayer.L1,
        name=f"{method}_fail",
        score=score,
        severity=severity,
        evidence=evidence,
        metadata=metadata,
        error=error,
    )


def _signal_for(method: str, verdict: AuthVerdict | None) -> DetectionSignal:
    label = _METHOD_LABELS[method]

    if verdict is None:
        # Absent is not failed. Abstain so scoring excludes it entirely.
        return _signal(
            method,
            0.0,
            RiskLevel.LOW,
            f"No {label} verdict was recorded in any Authentication-Results header.",
            error=f"no {label} result present",
        )

    detail = f"{method}={verdict.result}{_reported_by(verdict)}"

    if verdict.result in _INCONCLUSIVE:
        return _signal(
            method,
            0.0,
            RiskLevel.LOW,
            f"{label} could not be evaluated ({detail}).{_conflict_note(verdict)}",
            verdict=verdict,
            error=f"{label} returned {verdict.result}",
        )

    if verdict.inconclusive:  # unrecognized value - do not guess at it
        return _signal(
            method,
            0.0,
            RiskLevel.LOW,
            f"{label} reported an unrecognized result ({detail}).",
            verdict=verdict,
            error=f"unrecognized {label} result {verdict.result!r}",
        )

    if verdict.result == "fail":
        return _signal(
            method,
            _FAIL_SCORES[method],
            RiskLevel.HIGH if method == "dmarc" else RiskLevel.MEDIUM,
            f"{label} authentication failed ({detail}).{_conflict_note(verdict)}",
            verdict=verdict,
        )

    if verdict.result == "softfail":
        return _signal(
            method,
            _SOFTFAIL_SCORE,
            RiskLevel.LOW,
            f"{label} soft-failed ({detail}): the sending domain marks this "
            f"source as unauthorized but asks that it not be rejected."
            f"{_conflict_note(verdict)}",
            verdict=verdict,
        )

    if verdict.result == "none":
        return _signal(
            method,
            0.0,
            RiskLevel.LOW,
            f"{label} had no policy to evaluate ({detail}); the sending domain "
            f"publishes no {label} record. This is not an authentication failure.",
            verdict=verdict,
        )

    if verdict.result == "pass":
        return _signal(
            method,
            0.0,
            RiskLevel.LOW,
            f"{label} authentication passed ({detail}).{_conflict_note(verdict)}",
            verdict=verdict,
        )

    # neutral / policy - recorded, benign, and worth showing verbatim.
    return _signal(
        method,
        0.0,
        RiskLevel.LOW,
        f"{label} returned a non-committal result ({detail}), which is not a "
        f"failure.{_conflict_note(verdict)}",
        verdict=verdict,
    )


def analyze_authentication_results(
    source: ParsedEmail | object,
) -> list[DetectionSignal]:
    """Emit one signal per authentication method, in AUTH_METHODS order.

    Always returns three signals. A method that failed scores above zero; a
    method that passed scores 0.0 and is a genuine negative; a method with no
    usable verdict carries `error` and abstains from scoring altogether.

    Offline and total: no DNS, no network, no cryptography, and no exception
    escapes to the caller.
    """
    verdicts = parse_authentication_results(source)
    return [_signal_for(method, verdicts.get(method)) for method in AUTH_METHODS]


# --------------------------------------------------------------------------
# Registrable domains
# --------------------------------------------------------------------------
#
# `tldextract` is configured with `suffix_list_urls=()`, so it uses its bundled
# Public Suffix List snapshot and never touches the network. Layer 1 is an
# offline layer apart from WHOIS, and a detection layer that silently fetches a
# suffix list on first use is neither deterministic nor testable.
#
# The PSL's private section is included, so user-content hosts split properly:
# `attacker.github.io` and `victim.github.io` are different registrable domains
# rather than one shared `github.io`, which is what a hosted phishing page on a
# shared platform depends on being confused about.

_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=True)


def registrable_domain(value: str | None) -> str | None:
    """The registrable ("eTLD+1") domain of an address or hostname.

    `alice@mail.corp.co.uk` and `bob@corp.co.uk` both reduce to `corp.co.uk`,
    which is the comparison the architecture asks for: string equality would
    call those two different domains, and a naive last-two-labels rule would
    call `evil.co.uk` and `corp.co.uk` the same one.

    Returns None for anything without a public suffix - an empty value, a bare
    hostname like `localhost`, an IP literal, or junk. None means "no domain to
    compare", which callers must treat as unknown rather than as a mismatch.
    """
    if not isinstance(value, str):
        return None
    candidate = value.strip().strip("<>").strip()
    if not candidate:
        return None
    if "@" in candidate:
        candidate = candidate.rsplit("@", 1)[1].strip()
    if not candidate:
        return None
    try:
        extracted = _EXTRACT(candidate)
    except Exception:  # noqa: BLE001 - untrusted header value, never fatal
        return None
    # `registered_domain` is deprecated in tldextract 5.3+ in favour of
    # `top_domain_under_public_suffix`; read whichever this version exposes.
    domain = getattr(extracted, "top_domain_under_public_suffix", None)
    if domain is None:
        domain = getattr(extracted, "registered_domain", "")
    return domain.lower() or None


# --------------------------------------------------------------------------
# 2. Reply-To vs From
# --------------------------------------------------------------------------

_REPLYTO_SCORE = 0.55


def analyze_reply_to_mismatch(source: ParsedEmail | object) -> DetectionSignal:
    """Fire when Reply-To's registrable domain differs from From's.

    A reply address on a different registrable domain is the mechanism behind
    most business-email-compromise attempts: the message looks like it came
    from the finance director, and the reply goes to the attacker. Subdomains
    do **not** fire - `billing.corp.com` replying to `corp.com` is one
    organization talking to itself.

    Abstains, rather than reporting a clean result, when either domain cannot
    be determined. A malformed From header is a fact for another signal to
    judge; here it simply means the comparison could not be made, and saying
    "no mismatch" would be a claim this signal has no basis for.

    Offline: no DNS and no network. Never raises.
    """
    email = _parsed_email(source)

    from_domain = registrable_domain(email.from_addr)
    reply_to_raw = (email.reply_to or "").strip()

    if not reply_to_raw:
        # A message with no Reply-To simply replies to From. That is the
        # ordinary case and a genuine negative, not an abstention.
        return _domain_signal(
            "replyto_mismatch",
            0.0,
            RiskLevel.LOW,
            "No Reply-To header is present, so replies go to the From address.",
            metadata={"from_domain": from_domain, "reply_to_domain": None},
        )

    reply_domain = registrable_domain(reply_to_raw)

    if from_domain is None or reply_domain is None:
        missing = "From" if from_domain is None else "Reply-To"
        return _domain_signal(
            "replyto_mismatch",
            0.0,
            RiskLevel.LOW,
            f"The {missing} address has no registrable domain, so Reply-To and "
            f"From could not be compared.",
            metadata={"from_domain": from_domain, "reply_to_domain": reply_domain},
            error=f"no registrable domain for {missing}",
        )

    if from_domain == reply_domain:
        return _domain_signal(
            "replyto_mismatch",
            0.0,
            RiskLevel.LOW,
            f"Reply-To and From share the registrable domain {from_domain}.",
            metadata={"from_domain": from_domain, "reply_to_domain": reply_domain},
        )

    return _domain_signal(
        "replyto_mismatch",
        _REPLYTO_SCORE,
        RiskLevel.MEDIUM,
        f"Replies would go to {reply_domain}, not to the sending domain "
        f"{from_domain}: Reply-To <{reply_to_raw}> and From <{email.from_addr}> "
        f"are on different registrable domains.",
        metadata={"from_domain": from_domain, "reply_to_domain": reply_domain},
    )


# --------------------------------------------------------------------------
# 3. WHOIS domain age
# --------------------------------------------------------------------------

# Per ARCHITECTURE.md section 4: WHOIS is rate-limited and flaky, so a
# successful lookup is cached for 7 days, keyed on registrable domain.
WHOIS_TTL_SECONDS = 7 * 24 * 3600

# A *genuine negative* - the registry answered, and recorded no creation date
# for the domain - is cached for six hours, matching the intel-feed TTL in
# `storage.cache`. It is a real answer, so it is worth keeping; it is not worth
# keeping for a week, because a registry that starts publishing the date should
# be believed the same day.
WHOIS_NEGATIVE_TTL_SECONDS = 6 * 3600

# A lookup that could not be completed at all - a timeout, a refused socket, a
# quota rejection - is **not** an answer about the domain, and must never be
# stored as one. It gets a short cooldown instead: long enough that a broken
# registry is not re-dialled once per message, short enough that a domain is
# not blacked out over it. Real validation is what set this: a registry that
# answers correctly in ~10s exceeds the 5s budget, and under a six-hour
# negative entry that domain's age became permanently unknowable - the timeout
# recurring before the entry ever expired.
WHOIS_UNAVAILABLE_TTL_SECONDS = 5 * 60

# A domain registered within this window is the signal's subject. Named for the
# signal in the architecture table, `l1.domain_age_lt_7d`.
YOUNG_DOMAIN_DAYS = 7

_DOMAIN_AGE_SCORE = 0.65

_CACHE_KEY_PREFIX = "l1:whois:created"

# Marks a cache entry as a cooldown rather than an answer. A stored entry
# without it is a real WHOIS result: `found` true with a date, or false for a
# registry that recorded none.
_UNAVAILABLE = "unavailable"


class WhoisUnavailable(RuntimeError):
    """The lookup could not be completed, so nothing was learned.

    Raised for every condition that is *not* an answer about the domain: a
    timeout, a refused or reset socket, a quota rejection, an unparseable
    response. The distinction from a genuine negative is the one this class
    exists to keep: "the registry says there is no such domain" is a fact worth
    caching, while "the registry did not answer" is the absence of a fact and
    must never be stored as one.
    """


class WhoisTimeout(WhoisUnavailable, TimeoutError):
    """A WHOIS lookup exceeded its own budget.

    A slow registry is not a registry that said no. Real validation found one
    that answers correctly in ~10s against a 5s budget: treating that as a
    negative result cached the *absence* of an answer for six hours, and since
    the next attempt timed out too, the domain's age was never learned.

    So this is a `WhoisUnavailable`, and the caller abstains and schedules a
    retry rather than recording anything about the domain. `TimeoutError` stays
    in the bases so ordinary `except TimeoutError` handling still catches it.
    """


@dataclass(frozen=True)
class DomainAge:
    """A WHOIS creation date, or the absence of one.

    `created_at` is None when WHOIS answered but recorded no creation date -
    a real, cacheable answer, and distinct from a lookup that failed.
    """

    domain: str
    created_at: datetime | None


@runtime_checkable
class WhoisLookup(Protocol):
    """The one network-touching seam in Layer 1.

    A protocol so the signal can be exercised without WHOIS: the unit suite
    injects a fake and never opens a socket.

    The contract is two-way, and the distinction is the whole point:

    - **Returning** a `DomainAge` is an *answer about the domain*. A
      `created_at` of None is a genuine negative - the registry was reached and
      recorded no creation date - and is cached as such.
    - **Raising** means the lookup could not be completed, so nothing was
      learned about the domain. The caller abstains and does not store it as an
      answer. Implementations should raise `WhoisUnavailable` (or
      `WhoisTimeout`), but *any* exception is treated as non-authoritative: an
      unrecognized failure is precisely the case where claiming to know
      something would be wrong.
    """

    def creation_date(self, domain: str) -> DomainAge: ...


class PythonWhoisLookup:
    """`python-whois` behind the `WhoisLookup` protocol.

    Imported lazily, inside the call, so importing this module never pulls in
    the WHOIS client - the unit suite injects a fake and must not depend on it
    being installed or on any of its import-time behaviour.

    Its job beyond fetching is to sort the client's exceptions into the two
    halves of the protocol. `python-whois` distinguishes them itself:
    `WhoisDomainNotFoundError` means the registry replied "no match", which is
    an answer, while a quota rejection, a failed command or an unparseable
    response means the question never got answered. Only the first becomes a
    `DomainAge`; everything else is re-raised as `WhoisUnavailable` so no
    caller can mistake a transport failure for a fact about the domain.
    """

    def creation_date(self, domain: str) -> DomainAge:
        import whois  # noqa: PLC0415 - deliberately lazy, see docstring

        not_found = getattr(
            getattr(whois, "exceptions", None), "WhoisDomainNotFoundError", ()
        )
        try:
            record = whois.whois(domain)
        except not_found:
            # Authoritative: the registry was reached and has no such domain.
            # There is no creation date because there is nothing registered.
            return DomainAge(domain=domain, created_at=None)
        except Exception as exc:  # noqa: BLE001 - re-raised, never swallowed
            raise WhoisUnavailable(
                f"WHOIS lookup for {domain} could not be completed "
                f"({type(exc).__name__})"
            ) from exc

        created = getattr(record, "creation_date", None)
        if isinstance(created, list):
            # Several registrars report a list. The earliest is the domain's
            # actual birth; a later entry is a transfer or a re-registration.
            candidates = [c for c in created if isinstance(c, datetime)]
            created = min(candidates) if candidates else None
        if not isinstance(created, datetime):
            return DomainAge(domain=domain, created_at=None)
        return DomainAge(domain=domain, created_at=_as_utc(created))


def _as_utc(value: datetime) -> datetime:
    """WHOIS dates are conventionally UTC but often arrive naive."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def analyze_domain_age(
    source: ParsedEmail | object,
    *,
    lookup: WhoisLookup | None = None,
    cache: object | None = None,
    now: datetime | None = None,
) -> DetectionSignal:
    """Fire when the sending domain was registered less than 7 days ago.

    Phishing infrastructure is disposable, so a domain days old is worth
    knowing about. This is the only Layer 1 signal that touches the network,
    and it is the only one that can therefore be unavailable.

    **A failed lookup abstains.** It returns a signal carrying `error`, which
    scoring excludes, rather than a 0.0 that would read as "checked, and the
    domain is old" - the distinction ARCHITECTURE.md section 2 exists to
    protect. `lookup=None` abstains for the same reason: no lookup was
    configured, so nothing was checked.

    `cache` is a `storage.cache.Cache` (or anything with its `get`/`set`).
    Successes are cached for 7 days and failures for 6 hours, both keyed on the
    registrable domain, so a rate-limited registry is asked once rather than
    once per message. Never raises.
    """
    email = _parsed_email(source)
    domain = registrable_domain(email.from_addr)
    now = now or datetime.now(timezone.utc)

    if domain is None:
        return _domain_signal(
            "domain_age_lt_7d",
            0.0,
            RiskLevel.LOW,
            "The From address has no registrable domain, so its age could not "
            "be looked up.",
            metadata={"domain": None},
            error="no registrable domain in From",
        )

    if lookup is None:
        return _domain_signal(
            "domain_age_lt_7d",
            0.0,
            RiskLevel.LOW,
            f"No WHOIS lookup is configured, so the age of {domain} was not checked.",
            metadata={"domain": domain},
            error="no WHOIS lookup configured",
        )

    cached = _cache_get(cache, domain)
    if cached is not None:
        if cached.get("status") == _UNAVAILABLE:
            # A cooldown entry, not an answer. It suppresses re-dialling a
            # registry that just failed; it does not claim anything.
            return _whois_unavailable(
                domain, cached.get("reason") or "WHOIS lookup could not be completed",
                cached=True,
            )
        age = DomainAge(domain=domain, created_at=_from_iso(cached.get("created_at")))
        return _domain_age_signal(age, now, cached=True)

    try:
        age = lookup.creation_date(domain)
    except Exception as exc:  # noqa: BLE001 - any WHOIS client failure
        # Nothing was learned about the domain. Every exception lands here,
        # not only `WhoisUnavailable`: an unrecognized failure is exactly the
        # case where recording an answer would be wrong. The exception text can
        # carry the registry's raw response, so only its type is shown.
        reason = (
            f"WHOIS lookup timed out ({type(exc).__name__})"
            if isinstance(exc, TimeoutError)
            else f"WHOIS lookup could not be completed ({type(exc).__name__})"
        )
        _cache_set(cache, domain, {"status": _UNAVAILABLE, "reason": reason},
                   WHOIS_UNAVAILABLE_TTL_SECONDS)
        return _whois_unavailable(domain, reason, cached=False)

    if not isinstance(age, DomainAge):
        # A lookup that does not honour the protocol has told us nothing
        # either, so it is a cooldown rather than a negative result.
        reason = "WHOIS lookup returned an unusable result"
        _cache_set(cache, domain, {"status": _UNAVAILABLE, "reason": reason},
                   WHOIS_UNAVAILABLE_TTL_SECONDS)
        return _whois_unavailable(domain, reason, cached=False)

    if age.created_at is None:
        # A genuine negative: the registry was reached and recorded no creation
        # date. That is an answer, and it is the one the six-hour negative TTL
        # was written for.
        _cache_set(cache, domain, {"found": False, "created_at": None},
                   WHOIS_NEGATIVE_TTL_SECONDS)
        return _domain_age_signal(age, now, cached=False)

    _cache_set(
        cache,
        domain,
        {"found": True, "created_at": _as_utc(age.created_at).isoformat()},
        WHOIS_TTL_SECONDS,
    )
    return _domain_age_signal(age, now, cached=False)


def _domain_age_signal(age: DomainAge, now: datetime, *, cached: bool) -> DetectionSignal:
    source_note = " (from cache)" if cached else ""

    if age.created_at is None:
        # WHOIS answered, but recorded no creation date. That is an answer, and
        # it is still not a basis for calling the domain old.
        return _domain_signal(
            "domain_age_lt_7d",
            0.0,
            RiskLevel.LOW,
            f"WHOIS records no creation date for {age.domain}{source_note}, so "
            f"its age is unknown.",
            metadata={
                "domain": age.domain,
                "cached": cached,
                "created_at": None,
                # The registry was reached: this abstention is a fact about the
                # domain, unlike the one from a lookup that never completed.
                "authoritative": True,
            },
            error="WHOIS recorded no creation date",
        )

    # An injected lookup may hand back a naive datetime, as WHOIS records
    # conventionally do; normalize before any arithmetic.
    created_at = _as_utc(age.created_at)
    age_days = (now - created_at).total_seconds() / 86400
    created = created_at.date().isoformat()
    metadata = {
        "domain": age.domain,
        "cached": cached,
        "created_at": created_at.isoformat(),
        "age_days": round(age_days, 2),
    }

    if age_days < YOUNG_DOMAIN_DAYS:
        return _domain_signal(
            "domain_age_lt_7d",
            _DOMAIN_AGE_SCORE,
            RiskLevel.MEDIUM,
            f"The sending domain {age.domain} was registered on {created}, "
            f"{age_days:.1f} days ago{source_note} - under the "
            f"{YOUNG_DOMAIN_DAYS}-day threshold. Phishing infrastructure is "
            f"typically days old.",
            metadata=metadata,
        )

    return _domain_signal(
        "domain_age_lt_7d",
        0.0,
        RiskLevel.LOW,
        f"The sending domain {age.domain} was registered on {created}, "
        f"{age_days:.0f} days ago{source_note}.",
        metadata=metadata,
    )


def _whois_unavailable(domain: str, reason: str, *, cached: bool) -> DetectionSignal:
    """Abstain because the lookup did not complete - never a result.

    Distinct from the abstention for a registry that answered without a date:
    that one is a fact about the domain, this one is the absence of one, and
    only this one leaves the domain due for another attempt.
    """
    return _domain_signal(
        "domain_age_lt_7d",
        0.0,
        RiskLevel.LOW,
        f"The age of {domain} could not be determined: {reason}"
        f"{' (cached; the lookup will be retried once the cooldown expires)' if cached else ''}"
        f". This signal abstains rather than reporting the domain as established.",
        metadata={"domain": domain, "cached": cached, "authoritative": False},
        error=reason,
    )


def _cache_key(domain: str) -> str:
    return f"{_CACHE_KEY_PREFIX}:{domain}"


def _cache_get(cache: object | None, domain: str) -> dict | None:
    if cache is None:
        return None
    try:
        value = cache.get(_cache_key(domain))
    except Exception:  # noqa: BLE001 - a broken cache must not fail the layer
        return None
    return value if isinstance(value, dict) else None


def _cache_set(cache: object | None, domain: str, value: dict, ttl: float) -> None:
    if cache is None:
        return
    try:
        cache.set(_cache_key(domain), value, ttl)
    except Exception:  # noqa: BLE001 - a broken cache must not fail the layer
        return


def _from_iso(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return _as_utc(datetime.fromisoformat(value))
    except ValueError:
        return None


# --------------------------------------------------------------------------
# 4. Display-name brand impersonation
# --------------------------------------------------------------------------
#
# The architecture's rule, verbatim: "Display name contains brand token, From
# domain does not." Deterministic and explainable by construction - a fixed
# table, exact token matching, no scoring model and no edit-distance search.
# Fuzzy matching is deliberately avoided: "Michael Smith at Apple Valley Dental"
# is not an Apple impersonation, and a similarity threshold loose enough to
# catch real attacks is loose enough to say that it is.

# brand -> the registrable domains that brand legitimately sends from.
BRAND_DOMAINS: dict[str, frozenset[str]] = {
    "microsoft": frozenset({
        "microsoft.com", "microsoftonline.com", "office.com", "office365.com",
        "outlook.com", "live.com", "sharepointonline.com", "azure.com",
    }),
    "office365": frozenset({"microsoft.com", "microsoftonline.com", "office.com",
                            "office365.com"}),
    "onedrive": frozenset({"microsoft.com", "onedrive.com", "live.com"}),
    "sharepoint": frozenset({"microsoft.com", "sharepointonline.com"}),
    "google": frozenset({"google.com", "gmail.com", "googlemail.com", "youtube.com"}),
    "gmail": frozenset({"google.com", "gmail.com", "googlemail.com"}),
    "apple": frozenset({"apple.com", "icloud.com", "me.com"}),
    "icloud": frozenset({"apple.com", "icloud.com"}),
    "amazon": frozenset({"amazon.com", "amazon.co.uk", "amazonses.com", "aws.amazon.com"}),
    "paypal": frozenset({"paypal.com", "paypal.co.uk"}),
    "docusign": frozenset({"docusign.com", "docusign.net"}),
    "dropbox": frozenset({"dropbox.com", "dropboxmail.com"}),
    "adobe": frozenset({"adobe.com", "adobelogin.com"}),
    "linkedin": frozenset({"linkedin.com"}),
    "netflix": frozenset({"netflix.com"}),
    "facebook": frozenset({"facebook.com", "facebookmail.com", "fb.com"}),
    "instagram": frozenset({"instagram.com", "facebookmail.com"}),
    "whatsapp": frozenset({"whatsapp.com"}),
    "chase": frozenset({"chase.com"}),
    "wellsfargo": frozenset({"wellsfargo.com"}),
    "hsbc": frozenset({"hsbc.com", "hsbc.co.uk"}),
    "barclays": frozenset({"barclays.co.uk", "barclays.com"}),
    "santander": frozenset({"santander.co.uk", "santander.com"}),
    "coinbase": frozenset({"coinbase.com"}),
    "binance": frozenset({"binance.com"}),
    "fedex": frozenset({"fedex.com"}),
    "dhl": frozenset({"dhl.com", "dhl.de"}),
    "ups": frozenset({"ups.com"}),
}

# Digits and marks that stand in for letters in the display names attackers
# actually send: "Micros0ft", "PayPa1", "Amaz0n". A fixed substitution table,
# applied before exact matching - not a similarity search, so it adds no
# false-positive surface beyond these specific characters.
_CONFUSABLES = str.maketrans({
    "0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b",
    "$": "s", "@": "a", "!": "l", "|": "l",
})

_DISPLAY_NAME_SCORE = 0.60

# A brand token alone is not an accusation. "Apple Valley Dental" contains
# "apple"; so does a message from a genuine small business with the word in its
# name. What distinguishes an impersonation attempt is that the display name
# presents itself as the brand *itself* - either it is just the brand, or it
# pairs the brand with the service vocabulary a transactional notice uses.
# A fixed word list, so the rule stays explainable and testable; the cost is
# that "Microsoft Azure <x@attacker.tk>" stays silent, which is the direction
# to err in for a medium-precision signal.
_SERVICE_WORDS = frozenset({
    "account", "accounts", "admin", "alert", "alerts", "billing", "care",
    "customer", "helpdesk", "id", "info", "invoice", "mail", "mailer",
    "message", "no", "noreply", "notification", "notifications", "notice",
    "online", "pay", "payment", "payments", "reply", "secure", "security",
    "service", "services", "signin", "support", "team", "update", "updates",
    "verification", "verify",
})

# Non-alphanumerics are separators, so "Micro-soft" and "Micro soft" both
# reduce to the token "microsoft" alongside their split forms.
_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def _fold(value: str) -> str:
    """Normalize a display name for exact comparison.

    Case, accents, zero-width characters and the confusable digits above are
    all removed, because none of them change what a human reads. Nothing
    approximate happens here: two strings either fold to the same token or
    they do not.
    """
    decomposed = unicodedata.normalize("NFKD", value)
    stripped = "".join(
        ch for ch in decomposed
        if not unicodedata.combining(ch) and unicodedata.category(ch) != "Cf"
    )
    return stripped.casefold().translate(_CONFUSABLES)


def _display_name_words(display_name: str) -> list[str]:
    return [w for w in _TOKEN_SPLIT_RE.split(_fold(display_name)) if w]


def _display_name_tokens(words: list[str]) -> set[str]:
    """The tokens a brand could match: each word, adjacent pairs, and the whole.

    Adjacent pairs catch the split spellings attackers use - "Micro Soft Team"
    and "Pay Pal Service" - without matching words that are merely both present
    somewhere in a long name.
    """
    tokens = set(words)
    tokens.update(a + b for a, b in zip(words, words[1:]))
    if len(words) > 1:
        tokens.add("".join(words))
    return tokens


def _presents_as_brand(words: list[str], brand: str) -> bool:
    """True when the name claims to *be* the brand, not merely to contain it."""
    if "".join(words) == brand:
        return True  # the display name is exactly the brand
    return any(word in _SERVICE_WORDS for word in words)


def analyze_display_name_impersonation(
    source: ParsedEmail | object,
) -> DetectionSignal:
    """Fire when the display name names a brand the From domain does not belong to.

    "Microsoft Account Team <security@random-vps.tk>" fires; the same display
    name from `microsoftonline.com` does not, and neither does an ordinary
    human name. A brand token appearing in the sending domain itself also does
    not fire, so a legitimate-but-unlisted brand domain degrades to silence
    rather than to a false accusation.

    Offline: a fixed table, exact token matching after a fixed normalization.
    No model, no edit distance, no network. Never raises.
    """
    email = _parsed_email(source)
    display_name = (email.from_display or "").strip()
    from_domain = registrable_domain(email.from_addr)

    if not display_name:
        return _domain_signal(
            "display_name_impersonation",
            0.0,
            RiskLevel.LOW,
            "The From header carries no display name, so there is no name to "
            "compare against the sending domain.",
            metadata={"display_name": None, "from_domain": from_domain},
        )

    words = _display_name_words(display_name)
    tokens = _display_name_tokens(words)
    matched = [brand for brand in BRAND_DOMAINS if brand in tokens]

    if not matched:
        return _domain_signal(
            "display_name_impersonation",
            0.0,
            RiskLevel.LOW,
            f"The display name {display_name!r} names no known brand.",
            metadata={"display_name": display_name, "from_domain": from_domain},
        )

    if from_domain is None:
        return _domain_signal(
            "display_name_impersonation",
            0.0,
            RiskLevel.LOW,
            f"The display name {display_name!r} names {matched[0]}, but the From "
            f"address has no registrable domain to compare it against.",
            metadata={
                "display_name": display_name,
                "from_domain": None,
                "matched_brands": matched,
            },
            error="no registrable domain in From",
        )

    off_domain = [brand for brand in matched if from_domain not in BRAND_DOMAINS[brand]]

    if not off_domain:
        legitimate = matched[0]
        return _domain_signal(
            "display_name_impersonation",
            0.0,
            RiskLevel.LOW,
            f"The display name {display_name!r} names {legitimate}, and the "
            f"sending domain {from_domain} belongs to it.",
            metadata={
                "display_name": display_name,
                "from_domain": from_domain,
                "matched_brands": matched,
            },
        )

    impersonated = [b for b in off_domain if _presents_as_brand(words, b)]

    if not impersonated:
        return _domain_signal(
            "display_name_impersonation",
            0.0,
            RiskLevel.LOW,
            f"The display name {display_name!r} mentions {off_domain[0]} but does "
            f"not present itself as {off_domain[0]}, so it is not treated as an "
            f"impersonation attempt.",
            metadata={
                "display_name": display_name,
                "from_domain": from_domain,
                "matched_brands": matched,
            },
        )

    brand = impersonated[0]
    expected = ", ".join(sorted(BRAND_DOMAINS[brand]))
    return _domain_signal(
        "display_name_impersonation",
        _DISPLAY_NAME_SCORE,
        RiskLevel.MEDIUM,
        f"The display name {display_name!r} presents this message as {brand}, "
        f"but it was sent from {from_domain}, which is not one of {brand}'s "
        f"domains ({expected}).",
        metadata={
            "display_name": display_name,
            "from_domain": from_domain,
            "matched_brands": matched,
            "impersonated_brands": impersonated,
        },
    )


# --------------------------------------------------------------------------
# Signal construction shared by the three domain signals
# --------------------------------------------------------------------------


def _domain_signal(
    name: str,
    score: float,
    severity: RiskLevel,
    evidence: str,
    *,
    metadata: dict[str, object],
    error: str | None = None,
) -> DetectionSignal:
    """Build one signal, recording `fired` explicitly in the metadata.

    `DetectionSignal` has no `fired` field - a finding is a score above zero,
    and an abstention is an `error`. Stating it in the metadata anyway keeps
    "did this signal fire" answerable without re-deriving it from two other
    fields, which is what the UI and the tests both want.
    """
    return DetectionSignal(
        layer=DetectionLayer.L1,
        name=name,
        score=score,
        severity=severity,
        evidence=evidence,
        metadata={**metadata, "fired": score > 0.0 and error is None},
        error=error,
    )


# --------------------------------------------------------------------------
# The layer
# --------------------------------------------------------------------------


def analyze(
    source: ParsedEmail | object,
    *,
    whois_lookup: WhoisLookup | None = None,
    cache: object | None = None,
    now: datetime | None = None,
) -> list[DetectionSignal]:
    """Run every Layer 1 signal over one message.

    Six signals, always, in the order of the ARCHITECTURE.md section 4 table:
    three authentication verdicts, then Reply-To, domain age and display-name
    impersonation.

    Graceful degradation is per signal, not per layer. An unavailable WHOIS
    lookup abstains on its own signal alone - the layer still completes and the
    other five still report - which is what keeps a WHOIS outage from being
    indistinguishable from a clean message.
    """
    signals = list(analyze_authentication_results(source))
    signals.append(analyze_reply_to_mismatch(source))
    signals.append(
        analyze_domain_age(source, lookup=whois_lookup, cache=cache, now=now)
    )
    signals.append(analyze_display_name_impersonation(source))
    return signals


# --------------------------------------------------------------------------
# Orchestration seam
# --------------------------------------------------------------------------
#
# Everything above is synchronous and stays that way: the signals are string
# and date work, they are directly callable, and the unit suite exercises them
# without an event loop. The orchestrator, however, is async and expects a
# `LayerCallable` - a coroutine function taking the email and returning
# signals. `analyze_async` is that adapter, and it is deliberately thin.
#
# It exists for one reason beyond the await: `analyze` blocks on WHOIS, and
# blocking the event loop would stall the three layers running beside this one.
# Only the WHOIS signal is offloaded to a worker thread. The other five are
# pure in-process parsing and are cheaper to run inline than to hand across a
# thread boundary.


# Well under `core.orchestrator.DEFAULT_LAYER_TIMEOUTS[L1]`, which is 8s. The
# layer timeout is the orchestrator's backstop against a broken layer; it is
# not a WHOIS budget. If the registry is the slow one, this fires first and the
# other five signals still reach the caller.
DEFAULT_WHOIS_TIMEOUT = 5.0


class TimeLimitedWhoisLookup:
    """A `WhoisLookup` that gives up on a slow one.

    `python-whois` talks to a registry over a socket it does not let the caller
    bound, and a hung registry would otherwise hold the whole Layer 1 budget.
    The wrapped lookup runs on a daemon thread that is simply abandoned when it
    overruns - a blocking socket read cannot be cancelled from outside, and a
    non-daemon thread would keep the interpreter alive at exit.

    Raising is the point. `analyze_domain_age` turns any exception from
    `creation_date` into an abstention carrying `error`, and caches it
    negatively for six hours, so a timeout reaches the caller as "not checked"
    rather than as "checked, and the domain is established" - and one slow
    registry is asked once, not once per message.
    """

    def __init__(
        self,
        inner: WhoisLookup | None = None,
        timeout: float = DEFAULT_WHOIS_TIMEOUT,
    ) -> None:
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout!r}")
        self._inner = inner if inner is not None else PythonWhoisLookup()
        self._timeout = float(timeout)

    @property
    def timeout(self) -> float:
        return self._timeout

    def creation_date(self, domain: str) -> DomainAge:
        # A one-slot list rather than a Queue: the worker writes once and this
        # thread reads only after join(), so there is nothing to synchronize.
        outcome: list[tuple[bool, object]] = []

        def _run() -> None:
            try:
                outcome.append((True, self._inner.creation_date(domain)))
            except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
                outcome.append((False, exc))

        worker = threading.Thread(
            target=_run, name=f"whois-{domain}", daemon=True
        )
        worker.start()
        worker.join(self._timeout)

        if not outcome:
            raise WhoisTimeout(
                f"WHOIS lookup for {domain} exceeded {self._timeout:g}s"
            )

        succeeded, value = outcome[0]
        if succeeded:
            return value  # type: ignore[return-value]
        raise value  # type: ignore[misc]


# `analyze_async`'s defaults are resolved per call rather than at import, so
# this sentinel distinguishes "the caller said nothing" from an explicit
# `cache=None` / `whois_lookup=None`, both of which are meaningful.
_UNSET: object = object()

_DEFAULT_CACHE_PATH = "storage/cache.db"

_default_cache_lock = threading.Lock()
_default_cache: object | None = None
_default_cache_resolved = False


def default_cache() -> object | None:
    """The process-wide `storage.cache.Cache`, opened once and reused.

    This is the cache the architecture already specifies for WHOIS - not a
    second one. It is opened lazily, on first use, at `CACHE_DB_PATH` (the
    variable `.env.example` already defines) so that importing this module
    never creates a database, and so the unit suite - which passes its own
    cache or none - never touches the file.

    The TTLs stay where they were decided: `analyze_domain_age` writes
    successes with `WHOIS_TTL_SECONDS` and failures with
    `WHOIS_NEGATIVE_TTL_SECONDS`. This function only supplies the store.

    Returns None if the cache cannot be opened. An unwritable cache directory
    is a reason to look domains up every time, not a reason to fail the layer.
    """
    global _default_cache, _default_cache_resolved

    with _default_cache_lock:
        if not _default_cache_resolved:
            _default_cache_resolved = True
            try:
                from storage.cache import Cache  # noqa: PLC0415 - lazy, see docstring

                path = os.environ.get("CACHE_DB_PATH") or _DEFAULT_CACHE_PATH
                _default_cache = Cache(path)
            except Exception:  # noqa: BLE001 - a cache is an optimization, not a dependency
                _default_cache = None
        return _default_cache


def reset_default_cache() -> None:
    """Forget the memoized default cache. For tests that repoint CACHE_DB_PATH."""
    global _default_cache, _default_cache_resolved

    with _default_cache_lock:
        cache = _default_cache
        _default_cache = None
        _default_cache_resolved = False

    close = getattr(cache, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001 - closing a spent cache must not raise
            pass


def default_whois_lookup() -> WhoisLookup:
    """The real WHOIS client, bounded by `DEFAULT_WHOIS_TIMEOUT`."""
    return TimeLimitedWhoisLookup(PythonWhoisLookup(), DEFAULT_WHOIS_TIMEOUT)


async def analyze_async(
    source: ParsedEmail | object,
    *,
    whois_lookup: WhoisLookup | None | object = _UNSET,
    cache: object | None = _UNSET,
    now: datetime | None = None,
) -> list[DetectionSignal]:
    """`analyze` for an event loop. This is the orchestrator's `LayerCallable`.

    Same six signals, same order, same contents - the only difference is where
    the WHOIS call runs. The five offline signals are produced by `analyze`
    itself (with its lookup disabled, so it abstains on domain age in the usual
    way), and the domain-age signal is computed on a worker thread and
    substituted in. Nothing else crosses a thread boundary.

    Degradation stays per signal. A WHOIS timeout, a WHOIS failure, a broken
    cache or a missing client all abstain on `domain_age_lt_7d` alone; the
    other five are already computed by then and are returned regardless, so the
    orchestrator still sees a completed Layer 1.

    Omitting `whois_lookup` or `cache` uses the process defaults. Passing None
    explicitly is not the same thing and is honoured: `whois_lookup=None`
    abstains without looking anything up, and `cache=None` disables caching.
    """
    lookup = default_whois_lookup() if whois_lookup is _UNSET else whois_lookup

    # `whois_lookup=None` is `analyze`'s own abstention path; there is nothing
    # to offload, so do not pay for a thread to find that out.
    if lookup is None:
        return analyze(
            source, whois_lookup=None, cache=None if cache is _UNSET else cache, now=now
        )

    if cache is _UNSET:
        # Resolve the default cache only if there is something to cache.
        # `analyze_domain_age` abstains without a lookup when the From address
        # has no registrable domain, and opening a database to record that
        # would be a side effect with nothing behind it.
        store = default_cache() if registrable_domain(_parsed_email(source).from_addr) else None
    else:
        store = cache

    signals = analyze(source, whois_lookup=None, cache=None, now=now)
    age = await asyncio.to_thread(
        analyze_domain_age, source, lookup=lookup, cache=store, now=now
    )

    # Substitute by name rather than by index: the position of the domain-age
    # signal is `analyze`'s business, and this stays correct if it moves.
    replaced = False
    resolved: list[DetectionSignal] = []
    for signal in signals:
        if not replaced and signal.name == age.name:
            resolved.append(age)
            replaced = True
        else:
            resolved.append(signal)
    if not replaced:
        # `analyze` emitted no domain-age placeholder to swap. Append rather
        # than drop a signal that was genuinely computed.
        resolved.append(age)
    return resolved
