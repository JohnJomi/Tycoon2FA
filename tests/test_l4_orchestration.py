"""Integration tests for Layer 4 running through the orchestrator.

Proves the wiring, not the signals - each signal's own behaviour is covered by
tests/test_l4_*.py. Every seam is injected: intelligence sources, cache, ASN
resolver and hosting list. No test contacts a feed, resolves a name, or opens a
socket, and no test writes a production dataset.
"""

from __future__ import annotations

import asyncio

import pytest

from core.models import (
    DetectionLayer,
    ExtractedURL,
    ParsedEmail,
    RiskLevel,
    URLSource,
)
from core.orchestrator import DEFAULT_LAYER_TIMEOUTS, DEFAULT_LAYERS, run_layers
from layers import l4_intel
from layers.l4_intel import (
    DOMAIN_IOC_SCORE,
    HOSTING_FLAG_SCORE,
    URL_IOC_SCORE,
    HostingList,
    HostingListEntry,
    Layer4Uninformative,
)
from layers.threat_intel import ASNInfo, ASNUnavailable, IndicatorKind, IntelUnavailable, IntelVerdict

L4_SIGNAL_NAMES = ["url_ioc", "domain_ioc", "hosting_flag"]

BAD_URL = "https://login.evil-phish.com/verify"
BAD_DOMAIN = "evil-phish.com"
BAD_HOST = "login.evil-phish.com"
FLAGGED_ASN = 64500  # RFC 5398 documentation range; a placeholder, not a real network


class FakeSource:
    """Answers about whatever it is told to list, for both indicator kinds."""

    refresh_interval_seconds = None

    def __init__(self, name: str, listed: set[str] | None = None) -> None:
        self.name = name
        self.listed = listed or set()
        self.calls: list[tuple[str, IndicatorKind]] = []

    def supports(self, kind: IndicatorKind) -> bool:
        return kind in (IndicatorKind.URL, IndicatorKind.DOMAIN)

    def check(self, indicator: str, kind: IndicatorKind) -> IntelVerdict:
        self.calls.append((indicator, kind))
        if indicator in self.listed:
            return IntelVerdict(found=True, source=self.name, detail="phishing")
        return IntelVerdict(found=False, source=self.name)


class DeadSource(FakeSource):
    def check(self, indicator: str, kind: IndicatorKind) -> IntelVerdict:
        self.calls.append((indicator, kind))
        raise IntelUnavailable(f"{self.name} is down")


class FakeResolver:
    def __init__(self, mapping: dict[str, int]) -> None:
        self.mapping = mapping
        self.calls: list[str] = []

    def asn_for(self, host: str) -> ASNInfo:
        self.calls.append(host)
        if host not in self.mapping:
            raise ASNUnavailable(f"no route data for {host}")
        return ASNInfo(asn=self.mapping[host], name="Placeholder BP")


def hosting_list() -> HostingList:
    return HostingList(
        entries={FLAGGED_ASN: HostingListEntry(FLAGGED_ASN, "Placeholder BP", "fixture")},
        source="test fixture",
        version="2026-09-01",
    )


@pytest.fixture
def email() -> ParsedEmail:
    return ParsedEmail(
        message_id="<m@example.com>",
        from_addr="sender@example.com",
        subject="Invoice",
        body_text="See the portal.",
        urls=[ExtractedURL(url=BAD_URL, source=URLSource.ANCHOR_HREF)],
    )


_MISSING = object()


def wired(*, sources=None, resolver=None, listing=_MISSING):
    """Layer 4 as a LayerCallable with every seam injected."""

    async def run(message: ParsedEmail):
        return await l4_intel.analyze_async(
            message,
            sources=sources if sources is not None else [FakeSource("urlhaus")],
            cache=None,
            asn_lookup=resolver,
            hosting_list=hosting_list() if listing is _MISSING else listing,
        )

    return run


def l4_of(results):
    return next(r for r in results if r.layer is DetectionLayer.L4)


# --- registration and normal execution ------------------------------------


def test_layer_four_is_registered_in_the_default_mapping() -> None:
    assert DEFAULT_LAYERS[DetectionLayer.L4] is l4_intel.analyze_async
    assert DEFAULT_LAYER_TIMEOUTS[DetectionLayer.L4] == 6.0


@pytest.mark.asyncio
async def test_all_three_signals_reach_the_result(email) -> None:
    results = await run_layers(
        email,
        layers={DetectionLayer.L4: wired(resolver=FakeResolver({BAD_HOST: 64501}))},
    )
    l4 = l4_of(results)

    assert [r.layer for r in results] == list(DetectionLayer)
    assert l4.completed is True
    assert l4.error is None
    assert [s.name for s in l4.signals] == L4_SIGNAL_NAMES
    assert all(s.layer is DetectionLayer.L4 for s in l4.signals)


def test_the_signals_run_in_the_architecture_order(email) -> None:
    """url_ioc, then domain_ioc, then hosting_flag - the section 4 table order."""
    source = FakeSource("urlhaus")
    resolver = FakeResolver({BAD_HOST: 64501})

    signals = l4_intel.analyze(
        email,
        sources=[source],
        cache=None,
        asn_lookup=resolver,
        hosting_list=hosting_list(),
    )

    assert [s.name for s in signals] == L4_SIGNAL_NAMES
    # The URL check happened before the domain check.
    assert [kind for _, kind in source.calls] == [IndicatorKind.URL, IndicatorKind.DOMAIN]
    assert resolver.calls == [BAD_HOST]


@pytest.mark.asyncio
async def test_each_seam_receives_the_indicator_it_is_responsible_for(email) -> None:
    source = FakeSource("urlhaus")
    resolver = FakeResolver({BAD_HOST: 64501})

    await run_layers(
        email, layers={DetectionLayer.L4: wired(sources=[source], resolver=resolver)}
    )

    assert (BAD_URL, IndicatorKind.URL) in source.calls
    assert (BAD_DOMAIN, IndicatorKind.DOMAIN) in source.calls
    assert resolver.calls == [BAD_HOST]  # the host, never the URL


# --- findings complete the layer ------------------------------------------


@pytest.mark.asyncio
async def test_a_url_ioc_finding_completes_the_layer(email) -> None:
    results = await run_layers(
        email,
        layers={DetectionLayer.L4: wired(sources=[FakeSource("urlhaus", {BAD_URL})])},
    )
    signals = {s.name: s for s in l4_of(results).signals}

    assert l4_of(results).completed is True
    assert signals["url_ioc"].score == URL_IOC_SCORE
    assert signals["url_ioc"].severity is RiskLevel.HIGH


@pytest.mark.asyncio
async def test_a_domain_ioc_finding_completes_the_layer(email) -> None:
    results = await run_layers(
        email,
        layers={DetectionLayer.L4: wired(sources=[FakeSource("urlhaus", {BAD_DOMAIN})])},
    )
    signals = {s.name: s for s in l4_of(results).signals}

    assert l4_of(results).completed is True
    assert signals["domain_ioc"].score == DOMAIN_IOC_SCORE


@pytest.mark.asyncio
async def test_a_hosting_only_finding_completes_the_layer(email) -> None:
    """Both feeds down, but the ASN check reached a conclusion on its own."""
    results = await run_layers(
        email,
        layers={
            DetectionLayer.L4: wired(
                sources=[DeadSource("urlhaus")],
                resolver=FakeResolver({BAD_HOST: FLAGGED_ASN}),
            )
        },
    )
    l4 = l4_of(results)
    signals = {s.name: s for s in l4.signals}

    assert l4.completed is True
    assert signals["hosting_flag"].score == HOSTING_FLAG_SCORE
    assert signals["url_ioc"].error is not None      # still an abstention
    assert signals["domain_ioc"].error is not None


@pytest.mark.asyncio
async def test_a_genuine_clean_check_completes_the_layer(email) -> None:
    results = await run_layers(
        email,
        layers={
            DetectionLayer.L4: wired(
                sources=[FakeSource("urlhaus")], resolver=FakeResolver({BAD_HOST: 64501})
            )
        },
    )
    l4 = l4_of(results)

    assert l4.completed is True
    assert all(s.error is None for s in l4.signals)
    assert all(s.score == 0.0 for s in l4.signals)


# --- fail-closed ----------------------------------------------------------


@pytest.mark.asyncio
async def test_all_signals_unavailable_marks_the_layer_incomplete(email) -> None:
    results = await run_layers(
        email,
        layers={DetectionLayer.L4: wired(sources=[DeadSource("urlhaus")], listing=None)},
    )
    l4 = l4_of(results)

    assert l4.completed is False
    assert l4.signals == []          # no fabricated negatives
    assert l4.error is not None      # and the reason is stated


@pytest.mark.asyncio
async def test_an_uninformative_layer_does_not_spend_its_weight(email) -> None:
    from scoring.composite import layer_contributions

    results = await run_layers(
        email,
        layers={DetectionLayer.L4: wired(sources=[DeadSource("urlhaus")], listing=None)},
    )

    assert DetectionLayer.L4 not in layer_contributions(results)


@pytest.mark.asyncio
async def test_one_signal_failing_never_becomes_a_clean_negative(email) -> None:
    """Feeds down, hosting clean: the layer completes, but the IOC signals abstain."""
    results = await run_layers(
        email,
        layers={
            DetectionLayer.L4: wired(
                sources=[DeadSource("urlhaus")], resolver=FakeResolver({BAD_HOST: 64501})
            )
        },
    )
    signals = {s.name: s for s in l4_of(results).signals}

    assert signals["url_ioc"].score == 0.0 and signals["url_ioc"].error is not None
    assert signals["domain_ioc"].error is not None
    assert signals["hosting_flag"].error is None  # this one really was checked
    assert "not a clean result" in signals["url_ioc"].evidence


@pytest.mark.asyncio
async def test_the_shipped_defaults_do_not_crash_or_report_clean() -> None:
    """No ASN resolver and no bulletproof dataset ship: hosting_flag abstains."""
    message = ParsedEmail(
        message_id="<m@example.com>", from_addr="sender@example.com", body_text="hi"
    )

    results = await run_layers(message)
    l4 = l4_of(results)

    # No URLs, so nothing was examined and the layer is not scored as clean.
    assert l4.completed is False
    assert l4.error is not None


def test_hosting_flag_abstains_without_its_dataset_rather_than_raising() -> None:
    message = ParsedEmail(
        message_id="<m@example.com>",
        from_addr="sender@example.com",
        urls=[ExtractedURL(url=BAD_URL, source=URLSource.ANCHOR_HREF)],
    )

    signals = l4_intel.analyze(message, sources=[FakeSource("urlhaus")], cache=None)
    hosting = next(s for s in signals if s.name == "hosting_flag")

    assert hosting.score == 0.0
    assert hosting.error is not None
    assert "not a clean result" in hosting.evidence


@pytest.mark.asyncio
async def test_a_crashing_layer_four_does_not_stop_the_analysis(email) -> None:
    async def exploding(_email):
        raise RuntimeError("feed client died")

    results = await run_layers(email, layers={DetectionLayer.L4: exploding})
    by_layer = {r.layer: r for r in results}

    assert by_layer[DetectionLayer.L4].completed is False
    assert "feed client died" in by_layer[DetectionLayer.L4].error
    assert by_layer[DetectionLayer.L1].completed is True
    assert by_layer[DetectionLayer.L1].signals


@pytest.mark.asyncio
async def test_a_slow_layer_four_is_bounded_by_its_own_timeout(email) -> None:
    async def slow(_email):
        await asyncio.sleep(1.0)
        return []

    results = await run_layers(
        email, layers={DetectionLayer.L4: slow}, timeouts={DetectionLayer.L4: 0.01}
    )
    by_layer = {r.layer: r for r in results}

    assert by_layer[DetectionLayer.L4].completed is False
    assert "timeout" in by_layer[DetectionLayer.L4].error
    assert by_layer[DetectionLayer.L1].completed is True


# --- the other layers are untouched ---------------------------------------


@pytest.mark.asyncio
async def test_the_other_layers_behave_as_before(email) -> None:
    results = await run_layers(
        email, layers={DetectionLayer.L4: wired(resolver=FakeResolver({BAD_HOST: 64501}))}
    )
    by_layer = {r.layer: r for r in results}

    assert by_layer[DetectionLayer.L1].completed is True
    assert [s.name for s in by_layer[DetectionLayer.L1].signals] == [
        "spf_fail",
        "dkim_fail",
        "dmarc_fail",
        "replyto_mismatch",
        "domain_age_lt_7d",
        "display_name_impersonation",
    ]
    assert by_layer[DetectionLayer.L2].completed is False
    assert "not implemented" in by_layer[DetectionLayer.L2].error
    # L3 is written but has no model artifacts here.
    assert by_layer[DetectionLayer.L3].completed is False


# --- no network -----------------------------------------------------------


@pytest.mark.asyncio
async def test_injected_seams_reach_no_network(email, monkeypatch) -> None:
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("layer 4 attempted network access")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket, "gethostbyname", forbidden)

    results = await run_layers(
        email,
        layers={
            DetectionLayer.L4: wired(
                sources=[FakeSource("urlhaus", {BAD_URL})],
                resolver=FakeResolver({BAD_HOST: FLAGGED_ASN}),
            )
        },
    )

    assert l4_of(results).completed is True


@pytest.mark.asyncio
async def test_analyze_always_returns_all_three_signals(email) -> None:
    """The sync entry point never raises; only the async adapter abstains."""
    signals = l4_intel.analyze(
        email, sources=[DeadSource("urlhaus")], cache=None, hosting_list=None
    )

    assert [s.name for s in signals] == L4_SIGNAL_NAMES
    with pytest.raises(Layer4Uninformative):
        await l4_intel.analyze_async(
            email, sources=[DeadSource("urlhaus")], cache=None, hosting_list=None
        )
