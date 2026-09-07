"""Unit tests for the l4.domain_ioc signal in layers/l4_intel.py.

Offline and deterministic. Sources are injected fakes except where a provider's
own parsing is under test, which drives `httpx.MockTransport`. No test opens a
socket, resolves a name, or contacts an analysed domain.
"""

from __future__ import annotations

import httpx
import pytest

from core.models import DetectionLayer, ExtractedURL, ParsedEmail, RiskLevel, URLSource
from layers.l4_intel import (
    DOMAIN_IOC_SCORE,
    MAX_DOMAINS_PER_MESSAGE,
    analyze_domain_ioc,
    cache_key,
    registrable_domains,
)
from layers.providers.feed_snapshot import OpenPhishSource
from layers.providers.urlhaus import URLhausSource
from layers.threat_intel import (
    INTEL_TTL_SECONDS,
    IndicatorKind,
    IntelUnavailable,
    IntelVerdict,
)
from storage.cache import Cache

BAD_DOMAIN = "evil-phish.com"
GOOD_DOMAIN = "example.com"


class FakeSource:
    """Covers DOMAIN by default; lists whatever it is told to."""

    refresh_interval_seconds = None

    def __init__(self, name: str, listed: set[str] | None = None, *, kinds=None) -> None:
        self.name = name
        self.listed = listed or set()
        self.calls: list[tuple[str, IndicatorKind]] = []
        self._kinds = kinds if kinds is not None else {IndicatorKind.DOMAIN}

    def supports(self, kind: IndicatorKind) -> bool:
        return kind in self._kinds

    def check(self, indicator: str, kind: IndicatorKind) -> IntelVerdict:
        self.calls.append((indicator, kind))
        if indicator in self.listed:
            return IntelVerdict(
                found=True,
                source=self.name,
                reference=f"https://{self.name}.example/entry",
                detail="phishing host",
            )
        return IntelVerdict(found=False, source=self.name)


class DeadSource(FakeSource):
    def check(self, indicator: str, kind: IndicatorKind) -> IntelVerdict:
        self.calls.append((indicator, kind))
        raise IntelUnavailable(f"{self.name} is down")


def email(*urls: str) -> ParsedEmail:
    return ParsedEmail(
        message_id="<m@example.com>",
        from_addr="sender@example.com",
        subject="Invoice",
        body_text="See attached.",
        urls=[ExtractedURL(url=u, source=URLSource.ANCHOR_HREF) for u in urls],
    )


def names(source: FakeSource) -> list[str]:
    return [indicator for indicator, _ in source.calls]


# --- extraction -----------------------------------------------------------


def test_no_urls_is_a_genuine_negative() -> None:
    source = FakeSource("urlhaus")
    signal = analyze_domain_ioc(email(), sources=[source])

    assert signal.layer is DetectionLayer.L4
    assert signal.name == "domain_ioc"
    assert signal.score == 0.0
    assert signal.severity is RiskLevel.LOW
    assert signal.error is None
    assert signal.metadata["fired"] is False
    assert signal.metadata["domains_checked"] == []
    assert source.calls == []


def test_a_url_without_a_registrable_domain_is_a_genuine_negative() -> None:
    """An IP literal has no registrable domain; url_ioc still checks it exactly."""
    signal = analyze_domain_ioc(email("http://192.0.2.10/login"), sources=[FakeSource("x")])

    assert signal.error is None
    assert signal.metadata["domains_checked"] == []


def test_one_registrable_domain_is_extracted() -> None:
    assert registrable_domains(email("https://www.evil-phish.com/login?a=1")) == [BAD_DOMAIN]


def test_multiple_domains_are_all_checked() -> None:
    source = FakeSource("urlhaus")
    signal = analyze_domain_ioc(
        email("https://a-corp.com/x", "https://b-corp.com/y", "https://c-corp.com/z"),
        sources=[source],
    )

    assert signal.metadata["domains_checked"] == ["a-corp.com", "b-corp.com", "c-corp.com"]
    assert names(source) == ["a-corp.com", "b-corp.com", "c-corp.com"]


def test_duplicate_domains_are_checked_once() -> None:
    source = FakeSource("urlhaus")
    analyze_domain_ioc(
        email("https://evil-phish.com/a", "https://evil-phish.com/b"), sources=[source]
    )

    assert names(source) == [BAD_DOMAIN]


def test_subdomains_collapse_to_one_registrable_domain() -> None:
    source = FakeSource("urlhaus")
    signal = analyze_domain_ioc(
        email(
            "https://login.evil-phish.com/a",
            "http://mail.evil-phish.com/b",
            "https://EVIL-PHISH.com/c",
        ),
        sources=[source],
    )

    assert names(source) == [BAD_DOMAIN]
    assert signal.metadata["urls_in_message"] == 3
    assert signal.metadata["domains_found"] == 1


def test_a_multi_part_public_suffix_is_honoured() -> None:
    """The vendored PSL is doing the work, not a naive last-two-labels split."""
    assert registrable_domains(email("https://shop.example.co.uk/x")) == ["example.co.uk"]


def test_userinfo_and_ports_do_not_confuse_extraction() -> None:
    assert registrable_domains(email("https://user@evil-phish.com:8443/x")) == [BAD_DOMAIN]


def test_the_domain_count_is_capped_and_reported() -> None:
    many = [f"https://d{i}-corp.com/x" for i in range(MAX_DOMAINS_PER_MESSAGE + 3)]
    source = FakeSource("urlhaus")

    signal = analyze_domain_ioc(email(*many), sources=[source])

    assert len(source.calls) == MAX_DOMAINS_PER_MESSAGE
    assert signal.metadata["domains_truncated"] is True


# --- findings -------------------------------------------------------------


@pytest.mark.parametrize("name", ["urlhaus", "otherfeed", "thirdfeed"])
def test_a_domain_listed_by_a_source_is_a_finding(name: str) -> None:
    sources = [FakeSource(n) for n in ("urlhaus", "otherfeed", "thirdfeed")]
    for source in sources:
        if source.name == name:
            source.listed = {BAD_DOMAIN}

    signal = analyze_domain_ioc(email("https://login.evil-phish.com/x"), sources=sources)

    assert signal.score == DOMAIN_IOC_SCORE
    assert signal.severity is RiskLevel.HIGH
    assert signal.error is None
    assert signal.metadata["fired"] is True
    assert signal.metadata["matches"] == [
        {
            "domain": BAD_DOMAIN,
            "source": name,
            "reference": f"https://{name}.example/entry",
            "detail": "phishing host",
        }
    ]
    assert BAD_DOMAIN in signal.evidence and name in signal.evidence


def test_the_domain_score_is_the_layer_four_convention_not_a_new_one() -> None:
    from layers.l4_intel import URL_IOC_SCORE

    assert 0.0 < DOMAIN_IOC_SCORE <= URL_IOC_SCORE


def test_absent_from_every_available_source_is_clean() -> None:
    signal = analyze_domain_ioc(
        email(f"https://{GOOD_DOMAIN}/x"), sources=[FakeSource("a"), FakeSource("b")]
    )

    assert signal.score == 0.0
    assert signal.error is None
    assert signal.metadata["matches"] == []
    assert signal.metadata["sources_consulted"] == ["a", "b"]
    assert signal.metadata["sources_unavailable"] == {}


def test_a_listing_stops_the_remaining_lookups() -> None:
    first = FakeSource("a", {BAD_DOMAIN})
    second = FakeSource("b", {BAD_DOMAIN})

    analyze_domain_ioc(email("https://evil-phish.com/x"), sources=[first, second])

    assert names(first) == [BAD_DOMAIN]
    assert second.calls == []


# --- availability ---------------------------------------------------------


def test_a_source_declining_domain_support_is_not_an_outage() -> None:
    url_only = FakeSource("openphish", kinds={IndicatorKind.URL})
    covering = FakeSource("urlhaus")

    signal = analyze_domain_ioc(email(f"https://{GOOD_DOMAIN}/x"), sources=[url_only, covering])

    assert signal.error is None
    assert signal.metadata["sources"] == ["urlhaus"]
    assert signal.metadata["sources_unavailable"] == {}
    assert url_only.calls == []


def test_every_source_declining_domain_support_abstains() -> None:
    signal = analyze_domain_ioc(
        email(f"https://{GOOD_DOMAIN}/x"),
        sources=[FakeSource("openphish", kinds={IndicatorKind.URL})],
    )

    assert signal.score == 0.0
    assert signal.error is not None
    assert "not a clean result" in signal.evidence


def test_partial_outage_does_not_hide_a_finding() -> None:
    signal = analyze_domain_ioc(
        email("https://evil-phish.com/x"),
        sources=[DeadSource("urlhaus"), FakeSource("otherfeed", {BAD_DOMAIN})],
    )

    assert signal.score == DOMAIN_IOC_SCORE
    assert signal.error is None
    assert signal.metadata["matches"][0]["source"] == "otherfeed"
    assert "urlhaus" in signal.metadata["sources_unavailable"]


def test_a_clean_result_records_consulted_and_unavailable_sources() -> None:
    signal = analyze_domain_ioc(
        email(f"https://{GOOD_DOMAIN}/x"),
        sources=[DeadSource("urlhaus"), FakeSource("otherfeed")],
    )

    assert signal.error is None
    assert signal.metadata["sources_consulted"] == ["otherfeed"]
    assert "urlhaus" in signal.metadata["sources_unavailable"]
    assert "could not be consulted" in signal.evidence


def test_all_applicable_sources_unavailable_abstains() -> None:
    signal = analyze_domain_ioc(
        email("https://evil-phish.com/x"), sources=[DeadSource("a"), DeadSource("b")]
    )

    assert signal.score == 0.0
    assert signal.metadata["fired"] is False
    assert signal.error is not None and "unavailable" in signal.error
    assert signal.metadata["sources_consulted"] == []
    assert "absence of information, not a clean result" in signal.evidence


def test_a_provider_raising_an_unexpected_error_is_an_outage() -> None:
    class Broken(FakeSource):
        def check(self, indicator, kind):
            raise ValueError("bad shape")

    signal = analyze_domain_ioc(email("https://evil-phish.com/x"), sources=[Broken("urlhaus")])

    assert signal.score == 0.0
    assert signal.error is not None
    assert "ValueError" in signal.metadata["sources_unavailable"]["urlhaus"]


# --- caching --------------------------------------------------------------


@pytest.fixture
def cache(tmp_path) -> Cache:
    with Cache(tmp_path / "cache.db") as store:
        yield store


def test_a_positive_verdict_is_cached_and_reused(cache) -> None:
    source = FakeSource("urlhaus", {BAD_DOMAIN})

    first = analyze_domain_ioc(email("https://evil-phish.com/a"), sources=[source], cache=cache)
    second = analyze_domain_ioc(email("https://evil-phish.com/b"), sources=[source], cache=cache)

    assert names(source) == [BAD_DOMAIN]  # the second was served from cache
    assert first.score == second.score == DOMAIN_IOC_SCORE


def test_a_negative_verdict_uses_the_found_false_convention(cache) -> None:
    source = FakeSource("urlhaus")

    analyze_domain_ioc(email(f"https://{GOOD_DOMAIN}/x"), sources=[source], cache=cache)
    stored = cache.get(cache_key("urlhaus", GOOD_DOMAIN, IndicatorKind.DOMAIN))

    assert stored == {"found": False, "source": "urlhaus"}
    analyze_domain_ioc(email(f"https://{GOOD_DOMAIN}/y"), sources=[source], cache=cache)
    assert names(source) == [GOOD_DOMAIN]


def test_an_unavailable_source_is_never_cached_as_a_negative(cache) -> None:
    dead = DeadSource("urlhaus")

    analyze_domain_ioc(email("https://evil-phish.com/x"), sources=[dead], cache=cache)

    assert cache.get(cache_key("urlhaus", BAD_DOMAIN, IndicatorKind.DOMAIN)) is None
    analyze_domain_ioc(email("https://evil-phish.com/x"), sources=[dead], cache=cache)
    assert names(dead) == [BAD_DOMAIN, BAD_DOMAIN]


def test_the_domain_and_url_caches_do_not_collide(cache) -> None:
    """Same string, different indicator kind, different key."""
    url_key = cache_key("urlhaus", BAD_DOMAIN, IndicatorKind.URL)
    domain_key = cache_key("urlhaus", BAD_DOMAIN, IndicatorKind.DOMAIN)

    assert url_key != domain_key
    assert domain_key.startswith("intel:domain:urlhaus:")


def test_the_cache_key_does_not_contain_the_domain() -> None:
    assert BAD_DOMAIN not in cache_key("urlhaus", BAD_DOMAIN, IndicatorKind.DOMAIN)


def test_an_unreadable_cache_row_is_treated_as_a_miss(cache) -> None:
    source = FakeSource("urlhaus", {BAD_DOMAIN})
    cache.set(
        cache_key("urlhaus", BAD_DOMAIN, IndicatorKind.DOMAIN), {"junk": 1}, INTEL_TTL_SECONDS
    )

    signal = analyze_domain_ioc(email("https://evil-phish.com/x"), sources=[source], cache=cache)

    assert signal.score == DOMAIN_IOC_SCORE


# --- provider parsing -----------------------------------------------------


def mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_urlhaus_queries_its_host_endpoint_for_a_domain() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = request.content.decode()
        return httpx.Response(
            200,
            json={"query_status": "ok", "urls": [{"url": "http://evil.example/a"}]},
        )

    source = URLhausSource(api_key="k", client=mock_client(handler))
    verdict = source.check(BAD_DOMAIN, IndicatorKind.DOMAIN)

    assert source.supports(IndicatorKind.DOMAIN) is True
    assert seen["url"].endswith("/v1/host/")
    assert f"host={BAD_DOMAIN}" in seen["body"].replace("%2E", ".")
    assert verdict.found is True
    assert "1 malicious URL" in verdict.detail


def test_urlhaus_reports_a_domain_miss_as_a_real_negative() -> None:
    source = URLhausSource(
        api_key="k",
        client=mock_client(lambda r: httpx.Response(200, json={"query_status": "no_results"})),
    )

    assert source.check(GOOD_DOMAIN, IndicatorKind.DOMAIN).found is False


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"query_status": "invalid_host"}),
        httpx.Response(200, json=["unexpected"]),
    ],
)
def test_urlhaus_malformed_domain_responses_fail_closed(response: httpx.Response) -> None:
    source = URLhausSource(api_key="k", client=mock_client(lambda r: response))

    with pytest.raises(IntelUnavailable):
        source.check(BAD_DOMAIN, IndicatorKind.DOMAIN)


def test_the_snapshot_feeds_decline_domain_indicators() -> None:
    """They publish URLs, not domains - declining is honest, not an outage."""
    source = OpenPhishSource()

    assert source.supports(IndicatorKind.DOMAIN) is False
    with pytest.raises(IntelUnavailable):
        source.check(BAD_DOMAIN, IndicatorKind.DOMAIN)


# --- no network -----------------------------------------------------------


def test_analysis_opens_no_socket_and_resolves_no_name(monkeypatch) -> None:
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("layer 4 attempted network access")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket, "gethostbyname", forbidden)

    signal = analyze_domain_ioc(
        email("https://login.evil-phish.com/x", f"https://{GOOD_DOMAIN}/y"),
        sources=[FakeSource("urlhaus", {BAD_DOMAIN})],
    )

    assert signal.score == DOMAIN_IOC_SCORE


def test_the_domain_is_handed_over_as_data_not_dereferenced() -> None:
    source = FakeSource("urlhaus")

    analyze_domain_ioc(email("https://evil-phish.com/x"), sources=[source])

    assert source.calls == [(BAD_DOMAIN, IndicatorKind.DOMAIN)]
