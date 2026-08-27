"""Layer 1 - header and domain intelligence.

Signals specified in ARCHITECTURE.md section 4:
  l1.auth_fail                   read Gmail's Authentication-Results verdicts
  l1.replyto_mismatch            registrable domain of Reply-To != From
  l1.domain_age_lt_7d            WHOIS creation date (cached, 7-day TTL)
  l1.display_name_impersonation  brand token in display name, not in domain

Only the first is implemented. The other three are Phase 2 tasks of their own
and nothing here is generalized in anticipation of them.

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

import re
from dataclasses import dataclass

from core.models import DetectionLayer, DetectionSignal, ParsedEmail, RiskLevel

__all__ = [
    "AUTH_METHODS",
    "AuthVerdict",
    "analyze_authentication_results",
    "parse_authentication_results",
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


def parse_authentication_results(email: ParsedEmail) -> dict[str, AuthVerdict]:
    """Read every Authentication-Results header into per-method verdicts.

    Methods with no recorded verdict are simply absent from the mapping; this
    function does not invent a result for them. Method names and result values
    are lower-cased, so `SPF=Pass` and `spf=pass` are the same fact.
    """
    headers = email.headers.get(_HEADER_NAME, [])

    collected: dict[str, list[str]] = {}
    authserv_for: dict[str, str] = {}
    for header in headers:
        if not isinstance(header, str):
            continue
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


def analyze_authentication_results(email: ParsedEmail) -> list[DetectionSignal]:
    """Emit one signal per authentication method, in AUTH_METHODS order.

    Always returns three signals. A method that failed scores above zero; a
    method that passed scores 0.0 and is a genuine negative; a method with no
    usable verdict carries `error` and abstains from scoring altogether.

    Offline and total: no DNS, no network, no cryptography, and no exception
    escapes to the caller.
    """
    verdicts = parse_authentication_results(email)
    return [_signal_for(method, verdicts.get(method)) for method in AUTH_METHODS]
