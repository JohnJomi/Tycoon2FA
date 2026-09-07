"""Provider-neutral threat-intelligence lookup: the seam Layer 4 depends on.

    Layer 4 -> ThreatIntelSource -> a provider -> some feed's own interface

This module is the whole of what Layer 4 is allowed to know about threat
intelligence. It names no feed, carries no endpoint, and imports no HTTP
client. Swapping or adding a source is a change in `layers/providers/`, not
here and not in `layers/l4_intel.py`.

The contract mirrors Layer 1's `WhoisLookup` and Layer 3's `AITextDetector`
deliberately, because it is the same shape of problem - one network-touching
seam, injected so the unit suite can exercise the layer without a socket, with
a two-way contract that keeps "the source answered" distinct from "the source
could not be reached":

- **Returning** an `IntelVerdict` is an answer about the indicator. `found`
  False is a genuine negative - the source was consulted and does not list it.
- **Raising** `IntelUnavailable` means nothing was learned, and the caller must
  not read the silence as "not listed". A feed that is down has no opinion.

The three free sources ARCHITECTURE.md section 4 names do not share an
interface, and this seam does not pretend otherwise
--------------------------------------------------------------------------
URLhaus answers per-indicator queries over HTTP. OpenPhish publishes a
downloadable snapshot and has no query endpoint on the free tier. PhishTank
distributes a bulk database dump. Forcing the latter two into a per-URL HTTP
call would mean inventing an endpoint that does not exist.

So the common capability - the only thing every source really offers - is
**"is this indicator present in this source"**, and that is what `check`
expresses. *How* a provider answers is its own business: URLhaus asks the
network per call, while a snapshot provider loads a feed once and answers from
memory. `refresh_interval_seconds` exists so a snapshot provider can state how
stale its answer may be; a query provider leaves it None.

`STIX/TAXII feeds would slot in at this interface, and so would a commercial
feed. Per ARCHITECTURE.md section 4, Proofpoint and Microsoft MSTIC are **not**
publicly accessible, so no provider here consumes them - the architecture
accommodates them, it does not claim to have them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

__all__ = [
    "ASNInfo",
    "ASNLookup",
    "ASNUnavailable",
    "INTEL_TTL_SECONDS",
    "IndicatorKind",
    "IntelTimeout",
    "IntelUnavailable",
    "IntelVerdict",
    "ThreatIntelSource",
]

# ARCHITECTURE.md section 4: "Cache all lookups in SQLite, 6-hour TTL." Stated
# here, next to the seam, so every caller and provider uses the same number.
INTEL_TTL_SECONDS = 6 * 3600


class IndicatorKind(str, Enum):
    """What an indicator is, so a source can decline what it does not cover.

    `DOMAIN` is declared now because `l4.domain_ioc` is the next signal and the
    seam must not need widening for it; nothing implements it yet.
    """

    URL = "url"
    DOMAIN = "domain"


class IntelUnavailable(RuntimeError):
    """The source could not be consulted, so nothing was learned.

    Raised for every condition that is *not* an answer about the indicator: a
    refused socket, a throttled request, a malformed response, a snapshot that
    could not be fetched or parsed, a missing API key. The caller must treat
    this as an absence of information - never as "not listed".
    """


class IntelTimeout(IntelUnavailable, TimeoutError):
    """The source did not answer inside its budget.

    A slow feed is not a feed that said "clean". Kept a subclass of
    `IntelUnavailable` so callers that only care about "no answer" need one
    except clause, with `TimeoutError` in the bases so ordinary timeout
    handling still catches it - the same arrangement as `WhoisTimeout` and
    `AITextTimeout`.
    """


@dataclass(frozen=True)
class IntelVerdict:
    """One source's answer about one indicator.

    `found` is the answer. `source` identifies who answered, so the evidence
    string can say which feed listed the URL rather than "threat intelligence"
    in the abstract. `reference` is a human-followable link to the entry where
    the source publishes one, and `detail` a short description of the threat -
    both optional, both for explainability, neither load-bearing.

    Frozen, like `DetectionSignal`: a verdict records what a source said and
    must not be edited afterwards.
    """

    found: bool
    source: str
    reference: str | None = None
    detail: str | None = None

    def as_cache_value(self) -> dict[str, object]:
        """The dict form stored in `storage.cache`.

        `{"found": false}` for a negative, which is the convention
        `storage/cache.py` documents: a negative lookup is cached by storing a
        value that *represents* absence, never by storing None.
        """
        value: dict[str, object] = {"found": self.found, "source": self.source}
        if self.reference is not None:
            value["reference"] = self.reference
        if self.detail is not None:
            value["detail"] = self.detail
        return value

    @classmethod
    def from_cache_value(cls, source: str, value: object) -> IntelVerdict | None:
        """Rebuild a verdict from a cached dict, or None if it is unreadable.

        A row this code cannot understand is treated as a miss rather than as
        an answer: a cache written by an older shape must not be able to
        fabricate a verdict.
        """
        if not isinstance(value, dict) or not isinstance(value.get("found"), bool):
            return None
        reference = value.get("reference")
        detail = value.get("detail")
        return cls(
            found=value["found"],
            source=str(value.get("source", source)),
            reference=reference if isinstance(reference, str) else None,
            detail=detail if isinstance(detail, str) else None,
        )


@runtime_checkable
class ThreatIntelSource(Protocol):
    """The whole of what Layer 4 requires of an intelligence source.

    `name` is stable and appears in signal metadata, so it is part of the
    contract rather than a label. `supports` lets a source decline an indicator
    kind it does not cover - a URL-only feed is not "unavailable" for a domain
    query, it simply has nothing to say and must not be counted as either a
    finding or an outage.

    `check` returns an `IntelVerdict` or raises `IntelUnavailable`. It must not
    fetch, resolve or otherwise touch the indicator's own infrastructure: this
    layer consults feeds about a URL, it never visits the URL.
    """

    name: str

    # None for a per-query source; a snapshot source states how often it
    # refreshes, so the caller can report how stale an answer may be.
    refresh_interval_seconds: float | None

    def supports(self, kind: IndicatorKind) -> bool: ...

    def check(self, indicator: str, kind: IndicatorKind) -> IntelVerdict: ...


# --------------------------------------------------------------------------
# ASN resolution
# --------------------------------------------------------------------------
#
# A second, smaller seam, for the one Layer 4 signal that asks a different
# question. `l4.hosting_flag` does not ask a feed "is this indicator listed";
# it asks "which network announces this host", and then matches that network
# against a static list. Two questions, two protocols - folding the second into
# `ThreatIntelSource` would mean a `check` whose return value means something
# different depending on who implemented it.


class ASNUnavailable(RuntimeError):
    """The host's network could not be determined, so nothing was learned.

    Raised for every condition that is not an answer: no resolver configured, a
    refused socket, a throttled service, a malformed response, a host the
    resolver cannot map. The caller abstains - a host whose ASN is unknown is
    not a host on a clean network.
    """


@dataclass(frozen=True)
class ASNInfo:
    """Which autonomous system announces a host.

    `asn` is the number alone (`13335`, never `"AS13335"`), because that is
    what a hosting list is keyed on. `name` and `prefix` are for explainability
    and may be absent; neither is load-bearing.
    """

    asn: int
    name: str | None = None
    prefix: str | None = None


@runtime_checkable
class ASNLookup(Protocol):
    """The whole of what Layer 4 requires of an ASN resolver.

    Returning an `ASNInfo` is an answer; raising `ASNUnavailable` means there
    is none. An implementation must not contact the host it is asked about -
    resolving a name or connecting to it would be reaching for attacker
    infrastructure, which ARCHITECTURE.md section 4 forbids. A resolver is
    expected to consult a routing dataset or a third-party service *about* the
    host instead.
    """

    def asn_for(self, host: str) -> ASNInfo: ...
