"""Unit tests for the l4.url_ioc signal in layers/l4_intel.py.

Offline and deterministic. Sources are injected fakes; the two provider tests
that exercise real parsing drive `httpx.MockTransport`, so no test in this file
opens a socket or contacts a feed.
"""

from __future__ import annotations

import json

import httpx
import pytest

from core.models import DetectionLayer, ExtractedURL, ParsedEmail, RiskLevel, URLSource
from layers.l4_intel import (
    MAX_URLS_PER_MESSAGE,
    URL_IOC_SCORE,
    analyze_url_ioc,
    cache_key,
    normalize_url,
)
from layers.providers.feed_snapshot import OpenPhishSource, PhishTankSource
from layers.providers.urlhaus import URLhausSource
from layers.threat_intel import (
    INTEL_TTL_SECONDS,
    IndicatorKind,
    IntelUnavailable,
    IntelVerdict,
)
from storage.cache import Cache

BAD = "http://evil.example/login"
GOOD = "https://example.com/hello"


class FakeSource:
    """Lists whatever it is told to, and records every indicator it was asked."""

    refresh_interval_seconds = None

    def __init__(self, name: str, listed: set[str] | None = None, *, kinds=None) -> None:
        self.name = name
        self.listed = listed or set()
        self.calls: list[str] = []
        self._kinds = kinds or {IndicatorKind.URL}

    def supports(self, kind: IndicatorKind) -> bool:
        return kind in self._kinds

    def check(self, indicator: str, kind: IndicatorKind) -> IntelVerdict:
        self.calls.append(indicator)
        if indicator in self.listed:
            return IntelVerdict(
                found=True,
                source=self.name,
                reference=f"https://{self.name}.example/entry",
                detail="phishing",
            )
        return IntelVerdict(found=False, source=self.name)


class DeadSource(FakeSource):
    def check(self, indicator: str, kind: IndicatorKind) -> IntelVerdict:
        self.calls.append(indicator)
        raise IntelUnavailable(f"{self.name} is down")


def email(*urls: str) -> ParsedEmail:
    return ParsedEmail(
        message_id="<m@example.com>",
        from_addr="sender@example.com",
        subject="Invoice",
        body_text="See attached.",
        urls=[ExtractedURL(url=u, source=URLSource.ANCHOR_HREF) for u in urls],
    )


# --- no URLs --------------------------------------------------------------


def test_a_message_with_no_urls_is_a_genuine_negative() -> None:
    source = FakeSource("urlhaus")
    signal = analyze_url_ioc(email(), sources=[source])

    assert signal.layer is DetectionLayer.L4
    assert signal.name == "url_ioc"
    assert signal.score == 0.0
    assert signal.severity is RiskLevel.LOW
    assert signal.error is None          # a negative, not an abstention
    assert signal.metadata["fired"] is False
    assert signal.metadata["urls_checked"] == 0
    assert source.calls == []            # nothing was looked up
    assert "no URLs" in signal.evidence


# --- findings, one per source ---------------------------------------------


@pytest.mark.parametrize("name", ["urlhaus", "openphish", "phishtank"])
def test_a_url_listed_by_a_source_is_a_finding(name: str) -> None:
    sources = [FakeSource(other) for other in ("urlhaus", "openphish", "phishtank")]
    for source in sources:
        if source.name == name:
            source.listed = {BAD}

    signal = analyze_url_ioc(email(BAD), sources=sources)

    assert signal.score == URL_IOC_SCORE
    assert signal.severity is RiskLevel.HIGH
    assert signal.error is None
    assert signal.metadata["fired"] is True
    assert signal.metadata["matches"] == [
        {
            "url": BAD,
            "source": name,
            "reference": f"https://{name}.example/entry",
            "detail": "phishing",
        }
    ]
    assert BAD in signal.evidence and name in signal.evidence


def test_a_url_absent_from_every_available_feed_is_clean() -> None:
    sources = [FakeSource("urlhaus"), FakeSource("openphish")]

    signal = analyze_url_ioc(email(GOOD), sources=sources)

    assert signal.score == 0.0
    assert signal.error is None
    assert signal.metadata["matches"] == []
    assert signal.metadata["sources_consulted"] == ["urlhaus", "openphish"]
    assert signal.metadata["sources_unavailable"] == {}


def test_a_listing_stops_the_remaining_lookups() -> None:
    first = FakeSource("urlhaus", {BAD})
    second = FakeSource("openphish", {BAD})

    analyze_url_ioc(email(BAD), sources=[first, second])

    assert first.calls == [BAD]
    assert second.calls == []


# --- partial availability -------------------------------------------------


def test_one_source_down_does_not_hide_another_source_finding() -> None:
    dead = DeadSource("urlhaus")
    alive = FakeSource("openphish", {BAD})

    signal = analyze_url_ioc(email(BAD), sources=[dead, alive])

    assert signal.score == URL_IOC_SCORE
    assert signal.error is None
    assert signal.metadata["matches"][0]["source"] == "openphish"
    assert "urlhaus" in signal.metadata["sources_unavailable"]


def test_a_clean_result_records_which_sources_could_not_be_consulted() -> None:
    signal = analyze_url_ioc(
        email(GOOD), sources=[DeadSource("urlhaus"), FakeSource("openphish")]
    )

    assert signal.score == 0.0
    assert signal.error is None  # one source did answer, so this is a real negative
    assert signal.metadata["sources_consulted"] == ["openphish"]
    assert "urlhaus" in signal.metadata["sources_unavailable"]
    assert "could not be consulted" in signal.evidence


def test_all_sources_unavailable_abstains() -> None:
    signal = analyze_url_ioc(
        email(BAD), sources=[DeadSource("urlhaus"), DeadSource("openphish")]
    )

    assert signal.score == 0.0
    assert signal.metadata["fired"] is False
    assert signal.error is not None and "unavailable" in signal.error
    assert signal.metadata["sources_consulted"] == []
    assert "absence of information, not a clean result" in signal.evidence


def test_no_configured_source_abstains() -> None:
    signal = analyze_url_ioc(email(BAD), sources=[])

    assert signal.error is not None
    assert "not a clean result" in signal.evidence


def test_a_source_that_does_not_cover_urls_is_skipped_not_counted() -> None:
    domain_only = FakeSource("domainfeed", kinds={IndicatorKind.DOMAIN})
    signal = analyze_url_ioc(email(GOOD), sources=[domain_only, FakeSource("urlhaus")])

    assert signal.error is None
    assert signal.metadata["sources"] == ["urlhaus"]
    assert domain_only.calls == []


def test_a_provider_raising_an_unexpected_error_is_an_outage_not_a_verdict() -> None:
    class Broken(FakeSource):
        def check(self, indicator, kind):
            raise ValueError("bad JSON shape")

    signal = analyze_url_ioc(email(BAD), sources=[Broken("urlhaus")])

    assert signal.score == 0.0
    assert signal.error is not None
    assert "ValueError" in signal.metadata["sources_unavailable"]["urlhaus"]


# --- deduplication --------------------------------------------------------


def test_duplicate_urls_are_checked_once() -> None:
    source = FakeSource("urlhaus")

    signal = analyze_url_ioc(email(BAD, BAD, " " + BAD + " "), sources=[source])

    assert source.calls == [BAD]
    assert signal.metadata["urls_in_message"] == 3
    assert signal.metadata["urls_checked"] == 1


def test_normalization_folds_only_scheme_and_host_case() -> None:
    assert normalize_url("HTTP://Evil.Example/Login") == "http://evil.example/Login"
    assert normalize_url("  https://a.example/x  ") == "https://a.example/x"
    # Paths differ, so these are two different URLs and both get checked.
    source = FakeSource("urlhaus")
    analyze_url_ioc(email("http://a.example/A", "http://a.example/b"), sources=[source])
    assert len(source.calls) == 2


def test_the_number_of_urls_checked_is_capped_and_reported() -> None:
    many = [f"http://a.example/{i}" for i in range(MAX_URLS_PER_MESSAGE + 5)]
    source = FakeSource("urlhaus")

    signal = analyze_url_ioc(email(*many), sources=[source])

    assert len(source.calls) == MAX_URLS_PER_MESSAGE
    assert signal.metadata["urls_truncated"] is True


# --- caching --------------------------------------------------------------


@pytest.fixture
def cache(tmp_path) -> Cache:
    with Cache(tmp_path / "cache.db") as store:
        yield store


def test_a_positive_verdict_is_cached_and_reused(cache) -> None:
    source = FakeSource("urlhaus", {BAD})

    first = analyze_url_ioc(email(BAD), sources=[source], cache=cache)
    second = analyze_url_ioc(email(BAD), sources=[source], cache=cache)

    assert source.calls == [BAD]  # the second lookup was served from cache
    assert first.score == second.score == URL_IOC_SCORE
    assert second.metadata["matches"][0]["source"] == "urlhaus"


def test_a_negative_verdict_is_cached_using_the_found_false_convention(cache) -> None:
    source = FakeSource("urlhaus")

    analyze_url_ioc(email(GOOD), sources=[source], cache=cache)
    stored = cache.get(cache_key("urlhaus", GOOD, IndicatorKind.URL))

    assert stored == {"found": False, "source": "urlhaus"}
    analyze_url_ioc(email(GOOD), sources=[source], cache=cache)
    assert source.calls == [GOOD]


def test_an_unavailable_source_is_never_cached_as_a_negative(cache) -> None:
    dead = DeadSource("urlhaus")

    analyze_url_ioc(email(BAD), sources=[dead], cache=cache)

    assert cache.get(cache_key("urlhaus", BAD, IndicatorKind.URL)) is None
    # And the next run asks again rather than inheriting a fabricated verdict.
    analyze_url_ioc(email(BAD), sources=[dead], cache=cache)
    assert dead.calls == [BAD, BAD]


def test_a_recovered_source_is_believed_after_an_outage(cache) -> None:
    dead = DeadSource("urlhaus")
    analyze_url_ioc(email(BAD), sources=[dead], cache=cache)

    recovered = FakeSource("urlhaus", {BAD})
    signal = analyze_url_ioc(email(BAD), sources=[recovered], cache=cache)

    assert signal.score == URL_IOC_SCORE


def test_the_cache_key_does_not_contain_the_url(cache) -> None:
    key = cache_key("urlhaus", BAD, IndicatorKind.URL)

    assert BAD not in key
    assert key.startswith("intel:url:urlhaus:")


def test_an_unreadable_cache_row_is_treated_as_a_miss(cache) -> None:
    source = FakeSource("urlhaus", {BAD})
    cache.set(cache_key("urlhaus", BAD, IndicatorKind.URL), {"junk": 1}, INTEL_TTL_SECONDS)

    signal = analyze_url_ioc(email(BAD), sources=[source], cache=cache)

    assert signal.score == URL_IOC_SCORE
    assert source.calls == [BAD]


def test_a_broken_cache_does_not_break_the_lookup() -> None:
    class BrokenCache:
        def get(self, key):
            raise RuntimeError("disk on fire")

        def set(self, key, value, ttl):
            raise RuntimeError("disk on fire")

    signal = analyze_url_ioc(
        email(BAD), sources=[FakeSource("urlhaus", {BAD})], cache=BrokenCache()
    )

    assert signal.score == URL_IOC_SCORE


# --- provider parsing (MockTransport, no sockets) -------------------------


def mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_urlhaus_parses_a_listing_and_a_miss() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Auth-Key"] == "k"
        body = request.content.decode()
        if "evil" in body:
            return httpx.Response(
                200,
                json={
                    "query_status": "ok",
                    "threat": "malware_download",
                    "urlhaus_reference": "https://urlhaus.abuse.ch/url/1/",
                },
            )
        return httpx.Response(200, json={"query_status": "no_results"})

    source = URLhausSource(api_key="k", client=mock_client(handler))

    hit = source.check(BAD, IndicatorKind.URL)
    assert hit.found is True and hit.detail == "malware_download"
    assert source.check(GOOD, IndicatorKind.URL).found is False


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"query_status": "http_post_expected"}),
        httpx.Response(200, json=["unexpected"]),
    ],
)
def test_urlhaus_parsing_failures_fail_closed(response: httpx.Response) -> None:
    source = URLhausSource(api_key="k", client=mock_client(lambda request: response))

    with pytest.raises(IntelUnavailable):
        source.check(BAD, IndicatorKind.URL)


def test_urlhaus_without_a_key_is_unavailable_not_negative() -> None:
    with pytest.raises(IntelUnavailable):
        URLhausSource(api_key="").check(BAD, IndicatorKind.URL)


def test_openphish_answers_from_one_downloaded_snapshot() -> None:
    downloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        downloads.append(str(request.url))
        return httpx.Response(200, text=f"# comment\n{BAD}\nhttp://other.example/x\n")

    source = OpenPhishSource(client=mock_client(handler))

    assert source.check(BAD, IndicatorKind.URL).found is True
    assert source.check(GOOD, IndicatorKind.URL).found is False
    assert len(downloads) == 1  # the snapshot is fetched once, not per URL


@pytest.mark.parametrize("response", [httpx.Response(500), httpx.Response(200, text="  \n")])
def test_openphish_feed_failures_fail_closed(response: httpx.Response) -> None:
    source = OpenPhishSource(client=mock_client(lambda request: response))

    with pytest.raises(IntelUnavailable):
        source.check(BAD, IndicatorKind.URL)


def test_phishtank_parses_its_bulk_dump_and_needs_a_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "appkey" in str(request.url)
        return httpx.Response(200, text=json.dumps([{"url": BAD, "verified": "yes"}]))

    source = PhishTankSource(api_key="appkey", client=mock_client(handler))
    assert source.check(BAD, IndicatorKind.URL).found is True

    with pytest.raises(IntelUnavailable):
        PhishTankSource(api_key="", client=mock_client(handler)).check(BAD, IndicatorKind.URL)


@pytest.mark.parametrize(
    "response",
    [httpx.Response(200, text="{"), httpx.Response(200, json={}), httpx.Response(200, json=[])],
)
def test_phishtank_parsing_failures_fail_closed(response: httpx.Response) -> None:
    source = PhishTankSource(api_key="k", client=mock_client(lambda request: response))

    with pytest.raises(IntelUnavailable):
        source.check(BAD, IndicatorKind.URL)


# --- no network -----------------------------------------------------------


def test_analysis_opens_no_socket(monkeypatch) -> None:
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("layer 4 attempted network access")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)

    signal = analyze_url_ioc(email(BAD, GOOD), sources=[FakeSource("urlhaus", {BAD})])

    assert signal.score == URL_IOC_SCORE


def test_the_layer_never_fetches_the_indicator_itself() -> None:
    """The URL is passed to the feed as data; nothing dereferences it."""
    source = FakeSource("urlhaus")

    analyze_url_ioc(email(BAD), sources=[source])

    assert source.calls == [BAD]  # handed over, not visited
