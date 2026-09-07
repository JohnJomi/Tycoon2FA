"""Layer 4 - threat intelligence correlation.

Signals specified in ARCHITECTURE.md section 4:
  l4.url_ioc        URLhaus + OpenPhish + PhishTank lookup on extracted URLs
                                                              [implemented]
  l4.domain_ioc     same, at registrable domain level         [implemented]
  l4.hosting_flag   ASN lookup against a static bulletproof list [implemented]

Free feeds only. Lookups cached in SQLite with a 6-hour TTL.

All three signals are implemented.

`hosting_flag` asks a different question from the other two - not "is this
indicator listed" but "which network announces this host, and is that network
a known bulletproof host" - so it uses the `ASNLookup` seam rather than
`ThreatIntelSource`, and matches the answer against a static list loaded from
disk. **Neither the resolver nor the list ships with this repository**, and
both must be supplied deliberately: an ASN dataset and a bulletproof-hosting
list are editorial judgements about real networks, and inventing either would
mean accusing real operators on the strength of a placeholder. Absent, the
signal abstains - see `analyze_hosting_flag`.

The two IOC signals are the same correlation over two different indicator
sets - one exact URL, one registrable domain - and share `_correlate`. They
are separate signals because they answer different questions and fail
independently: a feed that lists `http://evil.example/a` says nothing about
`http://evil.example/b`, while a domain listing covers both. A source that
covers one kind and not the other declines rather than failing.

This module names no feed. Every endpoint, request shape and response format
lives behind `layers.threat_intel.ThreatIntelSource` in `layers/providers/`,
and the three free sources do not share an interface - URLhaus answers
per-indicator queries, OpenPhish and PhishTank publish downloadable snapshots.
The seam is built around the one capability they do share, "is this indicator
present in this source", and the providers differ underneath it.

**This layer consults feeds about a URL. It never visits the URL.** Nothing
here resolves, fetches or otherwise touches attacker infrastructure; the only
hosts contacted are the feed publishers', and only by a provider.

Absence is not a negative
-------------------------
A source that could not be consulted has no opinion. If a feed is down, its
silence must never be read as "this URL is not listed" - so an unavailable
source is recorded as unavailable, its result is never cached, and a lookup
where *every* applicable source was unavailable abstains rather than reporting
a clean URL. A positive from any source that did answer is real evidence and
stands on its own, whatever the others managed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from core.models import DetectionLayer, DetectionSignal, ParsedEmail, RiskLevel
from layers.l1_headers import registrable_domain
from layers.threat_intel import (
    INTEL_TTL_SECONDS,
    ASNInfo,
    ASNLookup,
    ASNUnavailable,
    IndicatorKind,
    IntelUnavailable,
    IntelVerdict,
    ThreatIntelSource,
)

__all__ = [
    "BULLETPROOF_LIST_PATH",
    "DOMAIN_IOC_SCORE",
    "HOSTING_FLAG_SCORE",
    "HostingDataUnavailable",
    "HostingList",
    "HostingListEntry",
    "MAX_DOMAINS_PER_MESSAGE",
    "MAX_URLS_PER_MESSAGE",
    "URL_IOC_SCORE",
    "analyze_domain_ioc",
    "analyze_hosting_flag",
    "analyze_url_ioc",
    "cache_key",
    "default_asn_lookup",
    "default_sources",
    "load_hosting_list",
    "normalize_url",
    "registrable_domains",
]

# Hand-assigned for Phase A exactly as the Layer 1 weights are, and refitted in
# Phase 5 against the labelled corpus. Higher than any Layer 1 score because it
# is a different kind of claim: not "this looks wrong" but "a threat-intel feed
# has this exact URL on record as malicious". It is deliberately short of 1.0 -
# feeds carry false positives and stale entries, and a signal that cannot be
# wrong is a signal nothing can outweigh.
URL_IOC_SCORE = 0.95

# Slightly below `URL_IOC_SCORE`. A domain listing is a broader claim than a
# URL listing and correspondingly a little weaker as evidence about *this*
# message: the feed recorded that the domain hosted something malicious, not
# that the link in this mail is the malicious thing. Hand-assigned for Phase A
# like every other score here, and refitted in Phase 5.
DOMAIN_IOC_SCORE = 0.90

# Registrable domains collapse hard - a message with 40 links usually has two
# or three - so this cap is well clear of any realistic message and exists for
# the same reason the URL one does.
MAX_DOMAINS_PER_MESSAGE = 15

# A bounded amount of work per message. A mail with 400 URLs is not worth 400
# feed queries inside a 6s layer budget; the cap is reported in the metadata so
# a truncated check is visible rather than silent.
MAX_URLS_PER_MESSAGE = 25


def normalize_url(url: str) -> str:
    """Fold trivially-equivalent spellings together before lookup.

    Deliberately conservative: surrounding whitespace, and the case of the
    scheme and host only. Paths stay case-sensitive because they are, and
    nothing is unescaped, reordered or stripped - an aggressive normalizer
    would turn two different URLs into one and check the wrong string against
    the feeds. This exists to avoid paying twice for the same URL, not to
    canonicalize the web.
    """
    trimmed = url.strip()
    if "://" not in trimmed:
        return trimmed

    scheme, rest = trimmed.split("://", 1)
    host, separator, tail = rest.partition("/")
    return f"{scheme.lower()}://{host.lower()}{separator}{tail}"


def cache_key(source_name: str, indicator: str, kind: IndicatorKind) -> str:
    """A stable cache key that does not put a live phishing URL in the key.

    The indicator is hashed rather than embedded: cache keys end up in logs and
    in `sqlite3` dumps, and a raw clickable attacker URL in either is a hazard
    for whoever reads them. SHA-256 is fixed-width, so the key length is
    bounded regardless of how long the URL was.
    """
    digest = hashlib.sha256(indicator.encode("utf-8", "replace")).hexdigest()
    return f"intel:{kind.value}:{source_name}:{digest}"


def default_sources() -> list[ThreatIntelSource]:
    """The three free sources from ARCHITECTURE.md section 4.

    Constructed here and nowhere else, and imported lazily so that importing
    this layer does not drag in an HTTP client. Callers inject their own list;
    the unit suite injects fakes and never reaches this function.
    """
    from layers.providers.feed_snapshot import OpenPhishSource, PhishTankSource
    from layers.providers.urlhaus import URLhausSource

    return [URLhausSource(), OpenPhishSource(), PhishTankSource()]


def _lookup(
    source: ThreatIntelSource,
    indicator: str,
    kind: IndicatorKind,
    cache: object | None,
) -> IntelVerdict:
    """One source's verdict for one indicator, through the cache.

    Raises `IntelUnavailable` when the source could not answer. **An
    unavailable result is never written to the cache**: caching it would turn a
    six-hour feed outage into six hours of confident "not listed", which is the
    exact failure ARCHITECTURE.md section 2 forbids. Only real answers are
    stored, positive and negative alike, the negative as `{"found": false}` per
    the convention `storage/cache.py` documents.

    A broken cache never fails a lookup: a store that raises on read is a slow
    path, not a wrong answer, so the source is consulted instead.
    """
    key = cache_key(source.name, indicator, kind)

    if cache is not None:
        try:
            cached = cache.get(key)
        except Exception:
            cached = None
        if cached is not None:
            verdict = IntelVerdict.from_cache_value(source.name, cached)
            if verdict is not None:
                return verdict

    verdict = source.check(indicator, kind)  # IntelUnavailable propagates

    if cache is not None:
        try:
            cache.set(key, verdict.as_cache_value(), INTEL_TTL_SECONDS)
        except Exception:
            pass  # an unwritable cache costs a round trip, not a verdict
    return verdict


def _correlate(
    indicators: Sequence[str],
    kind: IndicatorKind,
    sources: Sequence[ThreatIntelSource],
    cache: object | None,
    label: str,
) -> tuple[list[str], dict[str, str], list[dict[str, object]]]:
    """Check every indicator against every source, and report what happened.

    Returns `(consulted, unavailable, matches)`: the sources that actually
    answered at least once, the ones that could not with the reason, and one
    entry per listed indicator under the key `label`. Stops at the first source
    that lists a given indicator - one listing is enough, and the rest would be
    paid for nothing.

    A source that raises anything other than `IntelUnavailable` is broken, not
    authoritative, and is recorded as unavailable exactly like a feed that is
    down. Shared by `l4.url_ioc` and `l4.domain_ioc`, which differ only in what
    they feed it.
    """
    consulted: list[str] = []
    unavailable: dict[str, str] = {}
    matches: list[dict[str, object]] = []

    for indicator in indicators:
        for source in sources:
            try:
                verdict = _lookup(source, indicator, kind, cache)
            except IntelUnavailable as exc:
                unavailable.setdefault(source.name, str(exc))
                continue
            except Exception as exc:  # noqa: BLE001 - a provider defect is an outage
                unavailable.setdefault(source.name, f"{type(exc).__name__}: {exc}")
                continue

            if source.name not in consulted:
                consulted.append(source.name)
            if verdict.found:
                matches.append(
                    {
                        label: indicator,
                        "source": verdict.source,
                        "reference": verdict.reference,
                        "detail": verdict.detail,
                    }
                )
                break

    return consulted, unavailable, matches


def analyze_url_ioc(
    email: ParsedEmail,
    *,
    sources: Sequence[ThreatIntelSource] | None = None,
    cache: object | None = None,
) -> DetectionSignal:
    """Check every URL in the message against the threat-intelligence sources.

    Deduplicates URLs first - a phishing mail repeats its one link in the
    anchor, the plain-text part and the image - and checks each distinct URL
    against each source that covers URLs, stopping at the first source that
    lists it. A message with no URLs is a **genuine negative**: there was
    nothing to find, the check ran, and it completed.

    Abstains only when there was something to check and nothing could check it:
    URLs present, and every applicable source unavailable for all of them.
    Partial availability is not abstention - a hit from a source that did
    answer is evidence regardless of what the others did, and a clean result
    records which sources were actually consulted so "not listed" can be read
    against the feeds that said it.

    Never raises. Never fetches the URLs themselves.
    """
    kind = IndicatorKind.URL
    active = [s for s in (default_sources() if sources is None else sources) if s.supports(kind)]

    seen: list[str] = []
    for extracted in email.urls:
        candidate = normalize_url(extracted.url)
        if candidate and candidate not in seen:
            seen.append(candidate)
    truncated = len(seen) > MAX_URLS_PER_MESSAGE
    checked_urls = seen[:MAX_URLS_PER_MESSAGE]

    metadata: dict[str, object] = {
        "urls_in_message": len(email.urls),
        "urls_checked": len(checked_urls),
        "urls_truncated": truncated,
        "sources": [s.name for s in active],
        "sources_consulted": [],
        "sources_unavailable": {},
        "matches": [],
    }

    if not checked_urls:
        # Genuine negative, not an abstention: the check completed over the
        # zero URLs this message contains.
        return _signal(
            0.0,
            RiskLevel.LOW,
            "The message contains no URLs, so there was nothing to check "
            "against the threat-intelligence feeds.",
            metadata=metadata,
        )

    if not active:
        return _signal(
            0.0,
            RiskLevel.LOW,
            f"No threat-intelligence source is configured for URLs, so the "
            f"{len(checked_urls)} URL(s) in this message were not checked. This "
            f"is an absence of information, not a clean result.",
            metadata=metadata,
            error="no URL intelligence source available",
        )

    consulted, unavailable, matches = _correlate(checked_urls, kind, active, cache, "url")

    metadata.update(
        sources_consulted=consulted,
        sources_unavailable=unavailable,
        matches=matches,
    )

    if matches:
        listed = ", ".join(
            f"{match['url']} ({match['source']})" for match in matches[:3]
        )
        more = f" and {len(matches) - 3} more" if len(matches) > 3 else ""
        return _signal(
            URL_IOC_SCORE,
            RiskLevel.HIGH,
            f"{len(matches)} of the {len(checked_urls)} URL(s) in this message "
            f"are listed as malicious by threat intelligence: {listed}{more}.",
            metadata=metadata,
        )

    if not consulted:
        reasons = "; ".join(f"{name}: {reason}" for name, reason in unavailable.items())
        return _signal(
            0.0,
            RiskLevel.LOW,
            f"None of the {len(active)} threat-intelligence source(s) could be "
            f"consulted, so the {len(checked_urls)} URL(s) in this message were "
            f"not checked ({reasons}). This is an absence of information, not a "
            f"clean result.",
            metadata=metadata,
            error=f"all URL intelligence sources unavailable ({reasons})",
        )

    consulted_names = ", ".join(consulted)
    caveat = (
        f" {len(unavailable)} other source(s) could not be consulted."
        if unavailable
        else ""
    )
    return _signal(
        0.0,
        RiskLevel.LOW,
        f"None of the {len(checked_urls)} URL(s) in this message are listed by "
        f"{consulted_names}.{caveat}",
        metadata=metadata,
    )


def _signal(
    score: float,
    severity: RiskLevel,
    evidence: str,
    *,
    metadata: dict[str, object],
    error: str | None = None,
) -> DetectionSignal:
    """Build the signal, recording `fired` as Layer 1 and Layer 3 both do."""
    return DetectionSignal(
        layer=DetectionLayer.L4,
        name="url_ioc",
        score=score,
        severity=severity,
        evidence=evidence,
        metadata={**metadata, "fired": score > 0.0 and error is None},
        error=error,
    )


def registrable_domains(email: ParsedEmail) -> list[str]:
    """The distinct registrable domains of the message's URLs, in first-seen order.

    Uses `l1_headers.registrable_domain`, which reads the vendored public
    suffix list and never fetches - there is one PSL implementation in this
    project and this is not a second one. Subdomains therefore collapse:
    `login.evil.example` and `mail.evil.example` are one indicator, which is
    the entire point of checking at this level.

    A URL whose host yields no registrable domain (an IP literal, a malformed
    href) is skipped rather than guessed at; `l4.url_ioc` still checks it as an
    exact URL.
    """
    domains: list[str] = []
    for extracted in email.urls:
        host = _host_of(extracted.url)
        domain = registrable_domain(host) if host else None
        if domain and domain not in domains:
            domains.append(domain)
    return domains


def _host_of(url: str) -> str | None:
    """The host portion of a URL, without resolving anything.

    Deliberately string work: `registrable_domain` expects a host or an
    address, and handing it a whole URL would have it parse a path as a label.
    No DNS, no connection - this layer never touches the domains it analyses.
    """
    trimmed = url.strip()
    if not trimmed:
        return None

    remainder = trimmed.split("://", 1)[1] if "://" in trimmed else trimmed
    authority = remainder.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    if "@" in authority:  # userinfo@host
        authority = authority.rsplit("@", 1)[1]
    if authority.startswith("["):  # IPv6 literal: no registrable domain
        return None
    host = authority.split(":", 1)[0].strip().rstrip(".")
    return host.lower() or None


def analyze_domain_ioc(
    email: ParsedEmail,
    *,
    sources: Sequence[ThreatIntelSource] | None = None,
    cache: object | None = None,
) -> DetectionSignal:
    """Check the registrable domain of every URL against the intel sources.

    The same correlation as `l4.url_ioc` over a coarser indicator, and it
    catches what the exact-URL check cannot: a feed listing one phishing URL on
    a domain tells you nothing about the *other* links on that domain, but the
    domain listing covers all of them.

    A message with no URLs, or none with a registrable domain, is a **genuine
    negative** - the check ran and there was nothing to find. Abstains only
    when there were domains to check and nothing could check them: a source
    that declines `DOMAIN` is not an outage, and if every source declines,
    nothing applicable was available and the signal says so rather than
    reporting a clean domain.

    Never raises. Never resolves or contacts the domains themselves.
    """
    kind = IndicatorKind.DOMAIN
    active = [s for s in (default_sources() if sources is None else sources) if s.supports(kind)]

    found_domains = registrable_domains(email)
    truncated = len(found_domains) > MAX_DOMAINS_PER_MESSAGE
    checked = found_domains[:MAX_DOMAINS_PER_MESSAGE]

    metadata: dict[str, object] = {
        "urls_in_message": len(email.urls),
        "domains_found": len(found_domains),
        "domains_checked": checked,
        "domains_truncated": truncated,
        "sources": [s.name for s in active],
        "sources_consulted": [],
        "sources_unavailable": {},
        "matches": [],
    }

    if not checked:
        return _domain_signal(
            0.0,
            RiskLevel.LOW,
            "The message contains no URLs with a registrable domain, so there "
            "was nothing to check against the threat-intelligence feeds.",
            metadata=metadata,
        )

    if not active:
        return _domain_signal(
            0.0,
            RiskLevel.LOW,
            f"No threat-intelligence source covers domain indicators, so the "
            f"{len(checked)} registrable domain(s) in this message were not "
            f"checked. This is an absence of information, not a clean result.",
            metadata=metadata,
            error="no domain intelligence source available",
        )

    consulted, unavailable, matches = _correlate(checked, kind, active, cache, "domain")
    metadata.update(
        sources_consulted=consulted,
        sources_unavailable=unavailable,
        matches=matches,
    )

    if matches:
        listed = ", ".join(f"{match['domain']} ({match['source']})" for match in matches[:3])
        more = f" and {len(matches) - 3} more" if len(matches) > 3 else ""
        return _domain_signal(
            DOMAIN_IOC_SCORE,
            RiskLevel.HIGH,
            f"{len(matches)} of the {len(checked)} registrable domain(s) in this "
            f"message are listed as malicious by threat intelligence: "
            f"{listed}{more}.",
            metadata=metadata,
        )

    if not consulted:
        reasons = "; ".join(f"{name}: {reason}" for name, reason in unavailable.items())
        return _domain_signal(
            0.0,
            RiskLevel.LOW,
            f"None of the {len(active)} applicable threat-intelligence source(s) "
            f"could be consulted, so the {len(checked)} registrable domain(s) in "
            f"this message were not checked ({reasons}). This is an absence of "
            f"information, not a clean result.",
            metadata=metadata,
            error=f"all domain intelligence sources unavailable ({reasons})",
        )

    caveat = (
        f" {len(unavailable)} other source(s) could not be consulted."
        if unavailable
        else ""
    )
    return _domain_signal(
        0.0,
        RiskLevel.LOW,
        f"None of the {len(checked)} registrable domain(s) in this message are "
        f"listed by {', '.join(consulted)}.{caveat}",
        metadata=metadata,
    )


def _domain_signal(
    score: float,
    severity: RiskLevel,
    evidence: str,
    *,
    metadata: dict[str, object],
    error: str | None = None,
) -> DetectionSignal:
    return DetectionSignal(
        layer=DetectionLayer.L4,
        name="domain_ioc",
        score=score,
        severity=severity,
        evidence=evidence,
        metadata={**metadata, "fired": score > 0.0 and error is None},
        error=error,
    )


# --------------------------------------------------------------------------
# 3. Bulletproof hosting
# --------------------------------------------------------------------------
#
# ARCHITECTURE.md section 4: "ASN lookup against a static bulletproof-hosting
# list." Two separable halves, kept separable: `ASNLookup` answers "which
# network announces this host", and `HostingList` answers "is this network on
# the list". Neither knows about the other, which is what lets the ASN source
# be swapped - a routing-table dump, a local dataset, a commercial service -
# without touching the matching logic, and lets the list be re-curated without
# touching the resolution.

# Hand-assigned for Phase A like every other score in this project, and
# refitted in Phase 5. Deliberately the weakest of the three Layer 4 signals,
# and deliberately below the 0.65 block threshold in `config/weights.yaml`:
# "this link is hosted on a network with a bad reputation" is a statement about
# the neighbourhood, not about the message. Legitimate sites do sit on bad
# networks. It is corroboration, not a verdict on its own.
HOSTING_FLAG_SCORE = 0.60

# Where the curated list is expected. **This file is not in the repository and
# is not created by this module.** It names real network operators, so it is
# an editorial artifact that has to be sourced and reviewed, not generated -
# and a placeholder full of invented ASNs would produce confident accusations
# against whoever happens to hold those numbers.
BULLETPROOF_LIST_PATH = Path(__file__).resolve().parent.parent / "data" / "bulletproof_asns.json"


class HostingDataUnavailable(IntelUnavailable):
    """The hosting list could not be used, so no host can be judged.

    Missing file, unreadable file, wrong shape, or no usable entries. Kept a
    subclass of `IntelUnavailable` so Layer 4's one "nothing was learned"
    except clause covers it.
    """


@dataclass(frozen=True)
class HostingListEntry:
    """One curated network on the bulletproof-hosting list."""

    asn: int
    name: str | None = None
    reason: str | None = None

    def describe(self) -> str:
        parts = [f"AS{self.asn}"]
        if self.name:
            parts.append(self.name)
        return " ".join(parts)


@dataclass(frozen=True)
class HostingList:
    """A loaded bulletproof-hosting list, keyed by ASN.

    `source` and `version` come from the file and exist so a finding can say
    *which* list flagged the network and how current it was - a reader cannot
    evaluate "flagged as bulletproof" without knowing who says so.
    """

    entries: dict[int, HostingListEntry]
    source: str | None = None
    version: str | None = None

    def match(self, asn: int) -> HostingListEntry | None:
        return self.entries.get(asn)


def load_hosting_list(path: str | Path | None = None) -> HostingList:
    """Load the static bulletproof-hosting list from disk. Never downloads.

    Expected shape::

        {"source": "...", "version": "2026-09-01",
         "entries": [{"asn": 64500, "name": "...", "reason": "..."}]}

    Raises `HostingDataUnavailable` for a missing, unreadable, malformed or
    empty file. An empty list is treated as unusable rather than as "no network
    is bulletproof": a file that parses to zero entries is far more likely to
    be a failed download than a curated statement that the problem does not
    exist, and trusting it would clear every host on the internet.
    """
    target = Path(path) if path is not None else BULLETPROOF_LIST_PATH

    if not target.is_file():
        raise HostingDataUnavailable(f"no bulletproof-hosting list at {target}")

    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HostingDataUnavailable(f"hosting list at {target} could not be read: {exc}") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("entries"), list):
        raise HostingDataUnavailable(f"hosting list at {target} has no entries array")

    entries: dict[int, HostingListEntry] = {}
    for raw in payload["entries"]:
        if not isinstance(raw, dict):
            continue
        asn = raw.get("asn")
        if isinstance(asn, bool) or not isinstance(asn, int):
            # A malformed row is skipped rather than guessed at; a list made
            # entirely of malformed rows falls through to the empty check.
            continue
        name = raw.get("name")
        reason = raw.get("reason")
        entries[asn] = HostingListEntry(
            asn=asn,
            name=name if isinstance(name, str) else None,
            reason=reason if isinstance(reason, str) else None,
        )

    if not entries:
        raise HostingDataUnavailable(f"hosting list at {target} contains no usable entries")

    source = payload.get("source")
    version = payload.get("version")
    return HostingList(
        entries=entries,
        source=source if isinstance(source, str) else None,
        version=version if isinstance(version, str) else None,
    )


def default_asn_lookup() -> ASNLookup | None:
    """The process-wide ASN resolver, or None when none is configured.

    **There is deliberately no default implementation.** Resolving an ASN needs
    either a routing dataset this repository does not carry or a third-party
    service this task does not add, and a stub that guessed would be worse than
    nothing. Returning None makes `l4.hosting_flag` abstain, which is the
    honest state until a resolver is wired in - Layer 1's `WhoisLookup` shows
    the shape the real one should take.
    """
    return None


def hosts_of(email: ParsedEmail) -> list[str]:
    """The distinct hosts of the message's URLs, in first-seen order.

    Hosts, not registrable domains: an ASN is announced for an address, so
    `cdn.evil-phish.com` and `evil-phish.com` can sit on different networks and
    are two separate questions. Reuses the same string-only parsing
    `l4.domain_ioc` uses - no DNS, no connection, nothing dereferenced.
    """
    hosts: list[str] = []
    for extracted in email.urls:
        host = _host_of(extracted.url)
        if host and host not in hosts:
            hosts.append(host)
    return hosts


def analyze_hosting_flag(
    email: ParsedEmail,
    *,
    asn_lookup: ASNLookup | None = None,
    hosting_list: HostingList | None = None,
    hosting_list_path: str | Path | None = None,
) -> DetectionSignal:
    """Flag URLs hosted on a network the static list marks as bulletproof.

    The dataset is checked **first, before the URLs are even counted**. A
    message with no URLs is a genuine negative only when there was a list to
    check against; without one, nothing was verified and the signal says so
    rather than reporting a clean message on the strength of an absent file.

    Every host must be resolved and checked for the result to be clean. A host
    whose ASN could not be determined is not a host on a good network, so a
    resolver failure abstains rather than quietly reducing the set of hosts
    that were actually judged. Findings survive that: a host already matched
    against the list is evidence regardless of what happened to the others.

    Never raises. Never resolves or contacts the URLs themselves - the resolver
    is asked *about* a host, and is contractually forbidden from visiting it.
    """
    metadata: dict[str, object] = {
        "hosts_in_message": 0,
        "hosts_checked": [],
        "hosts_unresolved": {},
        "matches": [],
        "list_source": None,
        "list_version": None,
        "list_entries": None,
    }

    try:
        listing = hosting_list if hosting_list is not None else load_hosting_list(hosting_list_path)
    except HostingDataUnavailable as exc:
        return _hosting_signal(
            0.0,
            RiskLevel.LOW,
            f"The bulletproof-hosting list is unavailable ({exc}), so no host in "
            f"this message was checked. This is an absence of information, not a "
            f"clean result.",
            metadata=metadata,
            error=str(exc),
        )

    metadata.update(
        list_source=listing.source,
        list_version=listing.version,
        list_entries=len(listing.entries),
    )

    hosts = hosts_of(email)
    metadata["hosts_in_message"] = len(hosts)

    if not hosts:
        return _hosting_signal(
            0.0,
            RiskLevel.LOW,
            "The message contains no URLs with a host, so there was no hosting "
            "network to check.",
            metadata=metadata,
        )

    resolver = asn_lookup if asn_lookup is not None else default_asn_lookup()
    if resolver is None:
        return _hosting_signal(
            0.0,
            RiskLevel.LOW,
            f"No ASN resolver is configured, so the network hosting the "
            f"{len(hosts)} host(s) in this message could not be determined. This "
            f"is an absence of information, not a clean result.",
            metadata=metadata,
            error="no ASN resolver configured",
        )

    checked: list[dict[str, object]] = []
    unresolved: dict[str, str] = {}
    matches: list[dict[str, object]] = []

    for host in hosts:
        try:
            info = resolver.asn_for(host)
        except ASNUnavailable as exc:
            unresolved[host] = str(exc)
            continue
        except Exception as exc:  # noqa: BLE001 - a broken resolver is not an answer
            unresolved[host] = f"{type(exc).__name__}: {exc}"
            continue

        if not isinstance(info, ASNInfo):
            unresolved[host] = f"resolver returned {type(info).__name__}, not an ASNInfo"
            continue

        checked.append({"host": host, "asn": info.asn, "network": info.name})
        entry = listing.match(info.asn)
        if entry is not None:
            matches.append(
                {
                    "host": host,
                    "asn": info.asn,
                    "network": info.name,
                    "entry": entry.describe(),
                    "reason": entry.reason,
                }
            )

    metadata.update(hosts_checked=checked, hosts_unresolved=unresolved, matches=matches)

    if matches:
        listed = ", ".join(
            f"{match['host']} on AS{match['asn']} ({match['entry']})" for match in matches[:3]
        )
        more = f" and {len(matches) - 3} more" if len(matches) > 3 else ""
        attribution = f" per the {listing.source} list" if listing.source else ""
        return _hosting_signal(
            HOSTING_FLAG_SCORE,
            RiskLevel.MEDIUM,
            f"{len(matches)} of the {len(hosts)} host(s) in this message are on "
            f"networks flagged as bulletproof hosting{attribution}: {listed}{more}.",
            metadata=metadata,
        )

    if unresolved:
        # No finding, and at least one host was never judged. Reporting "clean"
        # would be a claim about hosts nobody looked at.
        reasons = "; ".join(f"{host}: {reason}" for host, reason in unresolved.items())
        return _hosting_signal(
            0.0,
            RiskLevel.LOW,
            f"{len(unresolved)} of the {len(hosts)} host(s) in this message could "
            f"not be attributed to a network ({reasons}), so the message was not "
            f"fully checked. This is an absence of information, not a clean result.",
            metadata=metadata,
            error=f"ASN unavailable for {len(unresolved)} of {len(hosts)} host(s)",
        )

    return _hosting_signal(
        0.0,
        RiskLevel.LOW,
        f"All {len(hosts)} host(s) in this message resolve to networks that are "
        f"not on the bulletproof-hosting list.",
        metadata=metadata,
    )


def _hosting_signal(
    score: float,
    severity: RiskLevel,
    evidence: str,
    *,
    metadata: dict[str, object],
    error: str | None = None,
) -> DetectionSignal:
    return DetectionSignal(
        layer=DetectionLayer.L4,
        name="hosting_flag",
        score=score,
        severity=severity,
        evidence=evidence,
        metadata={**metadata, "fired": score > 0.0 and error is None},
        error=error,
    )
