"""Integration tests for wiring Layer 1 into the async orchestrator.

Scope: the seam only. What each Layer 1 signal decides is settled in
`test_l1_headers.py` and `test_l1_domain_signals.py`; these tests cover the
adapter that puts those signals on an event loop - that the orchestrator calls
the real layer, that the blocking WHOIS call is offloaded rather than run on
the loop, that the existing cache is the one being used, and that a WHOIS
timeout abstains instead of quietly reporting a young domain as established.

Offline by construction: every WHOIS client here is a local double, and the
only cache is either an in-memory recorder or a real `storage.cache.Cache`
under tmp_path. No test opens a socket, and none reaches Gmail or OAuth.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from core.models import DetectionLayer, ParsedEmail
from core.orchestrator import DEFAULT_LAYERS, run_layers
from layers import l1_headers
from layers.l1_headers import (
    WHOIS_NEGATIVE_TTL_SECONDS,
    WHOIS_TTL_SECONDS,
    WHOIS_UNAVAILABLE_TTL_SECONDS,
    DomainAge,
    TimeLimitedWhoisLookup,
    WhoisTimeout,
    WhoisUnavailable,
    analyze,
    analyze_async,
)
from storage.cache import Cache

# Captured at import, before the autouse fixture below replaces it, so the one
# test that asserts about the real default cache can reach past the patch.
_REAL_DEFAULT_CACHE = l1_headers.default_cache

NOW = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)

# The six signals Layer 1 emits, in ARCHITECTURE.md section 4 order.
L1_SIGNAL_NAMES = [
    "spf_fail",
    "dkim_fail",
    "dmarc_fail",
    "replyto_mismatch",
    "domain_age_lt_7d",
    "display_name_impersonation",
]


@pytest.fixture(autouse=True)
def no_ambient_dependencies(monkeypatch):
    """No test in this module may fall through to a real registry or db file.

    Each test that wants a lookup or a cache passes its own; anything that does
    not is asserting about the default path and monkeypatches it explicitly.
    """
    monkeypatch.setattr(l1_headers, "default_whois_lookup", lambda: None)
    monkeypatch.setattr(l1_headers, "default_cache", lambda: None)


@pytest.fixture
def cache_file(tmp_path):
    """A real storage.cache.Cache, so TTL expiry is the real implementation."""
    with Cache(tmp_path / "l1-orchestration.sqlite") as c:
        yield c


@pytest.fixture
def email() -> ParsedEmail:
    return ParsedEmail(
        message_id="<wired@example.com>",
        from_addr="billing@corp-invoices.com",
        from_display="Microsoft Account Team",
        reply_to="attacker@elsewhere.net",
        # The parser lower-cases header names; `_HEADER_NAME` matches that.
        headers={"authentication-results": ["mx.google.com; spf=fail; dkim=pass; dmarc=fail"]},
    )


# --- WHOIS doubles ---------------------------------------------------------


class FakeWhois:
    """Answers immediately and records what it was asked."""

    def __init__(self, created_at: datetime | None = None) -> None:
        self.created_at = created_at
        self.calls: list[str] = []
        self.threads: list[str] = []

    def creation_date(self, domain: str) -> DomainAge:
        self.calls.append(domain)
        self.threads.append(threading.current_thread().name)
        return DomainAge(domain=domain, created_at=self.created_at)


class SlowWhois:
    """Blocks for `delay` seconds, the way a hung registry does."""

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.started = threading.Event()

    def creation_date(self, domain: str) -> DomainAge:
        self.started.set()
        time.sleep(self.delay)
        return DomainAge(domain=domain, created_at=NOW - timedelta(days=1))


class BrokenWhois:
    def creation_date(self, domain: str) -> DomainAge:
        raise ConnectionResetError("registry closed the connection")


class RecordingCache:
    """The `get`/`set` surface `analyze_domain_age` uses, with a log."""

    def __init__(self) -> None:
        self.store: dict[str, object] = {}
        self.gets: list[str] = []
        self.sets: list[tuple[str, object, float]] = []

    def get(self, key: str):
        self.gets.append(key)
        return self.store.get(key)

    def set(self, key: str, value, ttl_seconds: float) -> None:
        self.sets.append((key, value, ttl_seconds))
        self.store[key] = value


def _named(signals, name):
    return next(s for s in signals if s.name == name)


# --------------------------------------------------------------------------
# 1. The orchestrator calls the real Layer 1
# --------------------------------------------------------------------------


def test_the_default_l1_layer_is_the_real_implementation():
    assert DEFAULT_LAYERS[DetectionLayer.L1] is l1_headers.analyze_async


def test_no_orchestrator_stub_remains_for_layer_one():
    import core.orchestrator as orchestrator

    assert not hasattr(orchestrator, "stub_l1")
    assert not hasattr(orchestrator, "STUB_L1_SIGNAL_NAME")


@pytest.mark.asyncio
async def test_run_layers_returns_real_layer_one_signals(email, monkeypatch):
    monkeypatch.setattr(
        l1_headers, "default_whois_lookup", lambda: FakeWhois(NOW - timedelta(days=900))
    )

    results = await run_layers(email)
    l1 = next(r for r in results if r.layer is DetectionLayer.L1)

    assert l1.completed is True
    assert l1.error is None
    assert [s.name for s in l1.signals] == L1_SIGNAL_NAMES
    # A real detection, not a placeholder: this message really does forge a brand.
    assert _named(l1.signals, "display_name_impersonation").score > 0
    assert _named(l1.signals, "replyto_mismatch").score > 0
    assert _named(l1.signals, "spf_fail").score > 0


# --------------------------------------------------------------------------
# 2. The async contract, and the sync API it wraps
# --------------------------------------------------------------------------


def test_the_layer_callable_is_a_coroutine_function():
    assert inspect.iscoroutinefunction(l1_headers.analyze_async)


def test_the_layer_callable_needs_nothing_but_the_email():
    """The orchestrator passes one positional argument and no keywords."""
    signature = inspect.signature(l1_headers.analyze_async)
    required = [
        p
        for p in signature.parameters.values()
        if p.default is inspect.Parameter.empty and p.kind is not p.VAR_KEYWORD
    ]
    assert len(required) == 1


def test_the_synchronous_api_is_unchanged():
    """Direct callers, including the existing unit suite, still get sync functions."""
    for name in (
        "analyze",
        "analyze_authentication_results",
        "analyze_reply_to_mismatch",
        "analyze_domain_age",
        "analyze_display_name_impersonation",
    ):
        assert not inspect.iscoroutinefunction(getattr(l1_headers, name))


@pytest.mark.asyncio
async def test_async_and_sync_agree_signal_for_signal(email):
    lookup = FakeWhois(NOW - timedelta(days=3))

    expected = analyze(email, whois_lookup=FakeWhois(NOW - timedelta(days=3)), now=NOW)
    actual = await analyze_async(email, whois_lookup=lookup, cache=None, now=NOW)

    assert [s.name for s in actual] == [s.name for s in expected]
    assert [s.score for s in actual] == [s.score for s in expected]
    assert [s.evidence for s in actual] == [s.evidence for s in expected]
    assert [s.error for s in actual] == [s.error for s in expected]


@pytest.mark.asyncio
async def test_an_explicit_none_lookup_still_abstains(email):
    signals = await analyze_async(email, whois_lookup=None, cache=None, now=NOW)

    age = _named(signals, "domain_age_lt_7d")
    assert age.error == "no WHOIS lookup configured"
    assert age.score == 0.0
    assert [s.name for s in signals] == L1_SIGNAL_NAMES


# --------------------------------------------------------------------------
# 3. WHOIS does not run on the event loop
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_whois_call_runs_off_the_main_thread(email):
    lookup = FakeWhois(NOW - timedelta(days=2))

    await analyze_async(email, whois_lookup=lookup, cache=None, now=NOW)

    assert lookup.calls == ["corp-invoices.com"]
    assert lookup.threads[0] != threading.current_thread().name


@pytest.mark.asyncio
async def test_a_blocking_lookup_does_not_stall_the_event_loop(email):
    """The loop must keep scheduling while WHOIS blocks, or the sibling layers stall."""
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.005)

    beat = asyncio.create_task(ticker())
    await asyncio.sleep(0)  # let the ticker reach its first await
    try:
        await analyze_async(email, whois_lookup=SlowWhois(0.2), cache=None, now=NOW)
    finally:
        beat.cancel()

    # A blocked loop would have ticked once. ~40 are available in 0.2s.
    assert ticks > 5


@pytest.mark.asyncio
async def test_layer_one_still_completes_while_other_layers_run(email, monkeypatch):
    """The whole point of offloading: L2-L4 finish rather than waiting on WHOIS."""
    monkeypatch.setattr(l1_headers, "default_whois_lookup", lambda: SlowWhois(0.15))

    results = await run_layers(email)

    assert all(r.completed for r in results)
    assert [r.layer for r in results] == [
        DetectionLayer.L1,
        DetectionLayer.L2,
        DetectionLayer.L3,
        DetectionLayer.L4,
    ]
    # The three stub layers did not spend the WHOIS delay waiting their turn.
    for result in results[1:]:
        assert result.duration_ms < 100


# --------------------------------------------------------------------------
# 4. The existing cache is wired through
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_cache_is_consulted_and_written(email):
    cache = RecordingCache()

    await analyze_async(
        email, whois_lookup=FakeWhois(NOW - timedelta(days=2)), cache=cache, now=NOW
    )

    assert cache.gets == ["l1:whois:created:corp-invoices.com"]
    assert len(cache.sets) == 1
    key, value, ttl = cache.sets[0]
    assert key == "l1:whois:created:corp-invoices.com"
    assert value["found"] is True
    assert ttl == WHOIS_TTL_SECONDS == 7 * 24 * 3600


@pytest.mark.asyncio
async def test_a_genuine_negative_keeps_the_six_hour_negative_ttl(email):
    """A registry that answered without a date is a result, and is cached as one."""
    cache = RecordingCache()

    await analyze_async(email, whois_lookup=FakeWhois(None), cache=cache, now=NOW)

    _, value, ttl = cache.sets[0]
    assert value["found"] is False
    assert "status" not in value  # a real answer, not a cooldown marker
    assert ttl == WHOIS_NEGATIVE_TTL_SECONDS == 6 * 3600


@pytest.mark.asyncio
async def test_a_cached_domain_is_not_looked_up_twice(email):
    cache = RecordingCache()
    lookup = FakeWhois(NOW - timedelta(days=2))

    first = await analyze_async(email, whois_lookup=lookup, cache=cache, now=NOW)
    second = await analyze_async(email, whois_lookup=lookup, cache=cache, now=NOW)

    assert lookup.calls == ["corp-invoices.com"]
    assert _named(second, "domain_age_lt_7d").metadata["cached"] is True
    assert _named(first, "domain_age_lt_7d").score == _named(second, "domain_age_lt_7d").score


@pytest.mark.asyncio
async def test_the_default_cache_is_the_projects_sqlite_cache(email, tmp_path, monkeypatch):
    """Wired to `storage.cache.Cache` at CACHE_DB_PATH - not a second cache."""
    db = tmp_path / "cache.db"
    monkeypatch.setenv("CACHE_DB_PATH", str(db))
    monkeypatch.setattr(l1_headers, "default_cache", _REAL_DEFAULT_CACHE)
    l1_headers.reset_default_cache()
    try:
        cache = l1_headers.default_cache()
        assert isinstance(cache, Cache)

        await analyze_async(
            email, whois_lookup=FakeWhois(NOW - timedelta(days=2)), now=NOW
        )

        assert db.exists()
        assert cache.get("l1:whois:created:corp-invoices.com") == {
            "found": True,
            "created_at": (NOW - timedelta(days=2)).isoformat(),
        }
    finally:
        l1_headers.reset_default_cache()


@pytest.mark.asyncio
async def test_no_cache_is_opened_when_there_is_no_domain_to_look_up(
    tmp_path, monkeypatch
):
    """Opening a database to record "nothing to check" is a side effect for nothing."""
    db = tmp_path / "cache.db"
    monkeypatch.setenv("CACHE_DB_PATH", str(db))
    monkeypatch.setattr(l1_headers, "default_cache", _REAL_DEFAULT_CACHE)
    l1_headers.reset_default_cache()
    # `.invalid` has no public suffix, so there is no registrable domain.
    no_domain = ParsedEmail(
        message_id="<none@example.invalid>", from_addr="alerts@example.invalid"
    )
    try:
        signals = await analyze_async(
            no_domain, whois_lookup=FakeWhois(NOW), now=NOW
        )
    finally:
        l1_headers.reset_default_cache()

    assert _named(signals, "domain_age_lt_7d").error == "no registrable domain in From"
    assert not db.exists()


@pytest.mark.asyncio
async def test_a_broken_cache_does_not_sink_the_layer(email):
    class ExplodingCache:
        def get(self, key):
            raise RuntimeError("cache is unreadable")

        def set(self, key, value, ttl_seconds):
            raise RuntimeError("cache is unwritable")

    signals = await analyze_async(
        email, whois_lookup=FakeWhois(NOW - timedelta(days=2)), cache=ExplodingCache(), now=NOW
    )

    assert [s.name for s in signals] == L1_SIGNAL_NAMES
    assert _named(signals, "domain_age_lt_7d").score > 0


# --------------------------------------------------------------------------
# 5. The WHOIS timeout
# --------------------------------------------------------------------------


def test_a_slow_lookup_raises_rather_than_waiting(monkeypatch):
    bounded = TimeLimitedWhoisLookup(SlowWhois(5.0), timeout=0.05)

    started = time.perf_counter()
    with pytest.raises(WhoisTimeout):
        bounded.creation_date("slow-registry.com")
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0  # gave up on its own budget, not the caller's


def test_a_fast_lookup_is_returned_unchanged():
    inner = FakeWhois(NOW - timedelta(days=4))
    bounded = TimeLimitedWhoisLookup(inner, timeout=5.0)

    result = bounded.creation_date("corp.com")

    assert result == DomainAge(domain="corp.com", created_at=NOW - timedelta(days=4))


def test_an_inner_failure_is_propagated_not_masked_as_a_timeout():
    bounded = TimeLimitedWhoisLookup(BrokenWhois(), timeout=5.0)

    with pytest.raises(ConnectionResetError):
        bounded.creation_date("corp.com")


def test_a_non_positive_timeout_is_rejected():
    with pytest.raises(ValueError):
        TimeLimitedWhoisLookup(FakeWhois(), timeout=0)


def test_the_default_whois_timeout_is_under_the_layer_budget():
    from core.orchestrator import DEFAULT_LAYER_TIMEOUTS

    assert l1_headers.DEFAULT_WHOIS_TIMEOUT < DEFAULT_LAYER_TIMEOUTS[DetectionLayer.L1]


@pytest.mark.asyncio
async def test_a_whois_timeout_abstains_and_never_reports_the_domain_as_safe(email):
    bounded = TimeLimitedWhoisLookup(SlowWhois(5.0), timeout=0.05)

    signals = await analyze_async(email, whois_lookup=bounded, cache=None, now=NOW)
    age = _named(signals, "domain_age_lt_7d")

    assert age.score == 0.0
    assert age.error is not None
    assert "WhoisTimeout" in age.error
    # The distinction that matters: abstention, not an all-clear.
    assert "abstains" in age.evidence
    assert "registered" not in age.evidence


@pytest.mark.asyncio
async def test_a_timed_out_registry_is_NOT_negatively_cached(email):
    """The regression this fix exists for: a slow registry is not a negative result."""
    cache = RecordingCache()
    bounded = TimeLimitedWhoisLookup(SlowWhois(5.0), timeout=0.05)

    await analyze_async(email, whois_lookup=bounded, cache=cache, now=NOW)

    _, value, ttl = cache.sets[0]
    assert value["status"] == "unavailable"
    assert "found" not in value  # nothing was found, and nothing is claimed
    assert ttl == WHOIS_UNAVAILABLE_TTL_SECONDS
    assert ttl < WHOIS_NEGATIVE_TTL_SECONDS


# --------------------------------------------------------------------------
# 6. Graceful degradation stays per signal
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "lookup",
    [
        pytest.param(BrokenWhois(), id="failure"),
        pytest.param(TimeLimitedWhoisLookup(SlowWhois(5.0), timeout=0.05), id="timeout"),
    ],
)
async def test_the_other_five_signals_survive_a_whois_problem(email, lookup):
    signals = await analyze_async(email, whois_lookup=lookup, cache=None, now=NOW)

    assert [s.name for s in signals] == L1_SIGNAL_NAMES
    others = [s for s in signals if s.name != "domain_age_lt_7d"]
    assert all(s.error is None for s in others)
    assert _named(signals, "display_name_impersonation").score > 0
    assert _named(signals, "replyto_mismatch").score > 0


@pytest.mark.asyncio
async def test_the_orchestrator_still_sees_a_completed_layer_when_whois_times_out(
    email, monkeypatch
):
    monkeypatch.setattr(
        l1_headers,
        "default_whois_lookup",
        lambda: TimeLimitedWhoisLookup(SlowWhois(5.0), timeout=0.05),
    )

    results = await run_layers(email)
    l1 = next(r for r in results if r.layer is DetectionLayer.L1)

    # completed=True with one abstaining signal - not an incomplete layer.
    assert l1.completed is True
    assert l1.error is None
    assert len(l1.signals) == 6
    assert _named(l1.signals, "domain_age_lt_7d").error is not None


@pytest.mark.asyncio
async def test_a_whois_client_that_hangs_does_not_consume_the_layer_budget(
    email, monkeypatch
):
    monkeypatch.setattr(
        l1_headers,
        "default_whois_lookup",
        lambda: TimeLimitedWhoisLookup(SlowWhois(30.0), timeout=0.05),
    )

    started = time.perf_counter()
    results = await run_layers(email)
    elapsed = time.perf_counter() - started

    assert elapsed < 2.0  # nowhere near the 8s L1 timeout
    assert next(r for r in results if r.layer is DetectionLayer.L1).completed is True


# --------------------------------------------------------------------------
# 7. Authoritative results vs. lookups that never completed
#
# The regression these cover: a registry that answers correctly in ~10s against
# a 5s budget used to be recorded as a negative WHOIS result and cached for six
# hours. Since the retry timed out too, the domain's age became permanently
# unknowable. A timeout is now the absence of an answer, not an answer.
# --------------------------------------------------------------------------


class NoCreationDateWhois:
    """Reached the registry; it recorded no creation date. An authoritative no."""

    def __init__(self) -> None:
        self.calls = 0

    def creation_date(self, domain: str) -> DomainAge:
        self.calls += 1
        return DomainAge(domain=domain, created_at=None)


class QuotaWhois:
    """A registry that refused to answer. Not a fact about the domain."""

    def __init__(self) -> None:
        self.calls = 0

    def creation_date(self, domain: str) -> DomainAge:
        self.calls += 1
        raise WhoisUnavailable("quota exceeded")


def _entry(cache: RecordingCache) -> dict:
    return cache.sets[0][1]


def _ttl(cache: RecordingCache) -> float:
    return cache.sets[0][2]


@pytest.mark.asyncio
async def test_a_success_is_positively_cached_for_seven_days(email):
    cache = RecordingCache()

    signals = await analyze_async(
        email, whois_lookup=FakeWhois(NOW - timedelta(days=2)), cache=cache, now=NOW
    )

    assert _named(signals, "domain_age_lt_7d").score > 0
    assert _entry(cache) == {
        "found": True,
        "created_at": (NOW - timedelta(days=2)).isoformat(),
    }
    assert _ttl(cache) == WHOIS_TTL_SECONDS == 7 * 24 * 3600


@pytest.mark.asyncio
async def test_a_genuine_negative_abstains_but_is_marked_authoritative(email):
    cache = RecordingCache()

    signals = await analyze_async(
        email, whois_lookup=NoCreationDateWhois(), cache=cache, now=NOW
    )
    age = _named(signals, "domain_age_lt_7d")

    assert age.score == 0.0
    assert age.error == "WHOIS recorded no creation date"
    assert age.metadata["authoritative"] is True
    assert _entry(cache)["found"] is False
    assert _ttl(cache) == WHOIS_NEGATIVE_TTL_SECONDS


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "lookup, expected_error_fragment",
    [
        pytest.param(
            TimeLimitedWhoisLookup(SlowWhois(5.0), timeout=0.05),
            "timed out",
            id="timeout",
        ),
        pytest.param(QuotaWhois(), "could not be completed", id="quota"),
        pytest.param(BrokenWhois(), "could not be completed", id="network"),
    ],
)
async def test_an_incomplete_lookup_abstains_and_is_never_stored_as_a_result(
    email, lookup, expected_error_fragment
):
    cache = RecordingCache()

    signals = await analyze_async(email, whois_lookup=lookup, cache=cache, now=NOW)
    age = _named(signals, "domain_age_lt_7d")

    assert age.score == 0.0
    assert age.error is not None
    assert expected_error_fragment in age.error
    assert age.metadata["authoritative"] is False
    # The point of the fix: no negative result was recorded for this domain.
    assert _entry(cache)["status"] == "unavailable"
    assert "found" not in _entry(cache)
    assert _ttl(cache) == WHOIS_UNAVAILABLE_TTL_SECONDS
    assert _ttl(cache) < WHOIS_NEGATIVE_TTL_SECONDS


@pytest.mark.asyncio
async def test_the_timeout_error_names_the_timeout_specifically(email):
    bounded = TimeLimitedWhoisLookup(SlowWhois(5.0), timeout=0.05)

    signals = await analyze_async(email, whois_lookup=bounded, cache=None, now=NOW)
    age = _named(signals, "domain_age_lt_7d")

    assert "timed out" in age.error
    assert "WhoisTimeout" in age.error
    # Never phrased as an answer about the domain.
    assert "no creation date" not in age.error
    assert "registered" not in age.evidence


@pytest.mark.asyncio
async def test_a_domain_that_timed_out_is_retried_once_the_cooldown_expires(
    email, cache_file
):
    """A slow-but-healthy registry must not black the domain out for six hours."""
    import storage.cache as cache_module

    lookup = _SwitchableWhois(WhoisTimeout("too slow"))

    first = await analyze_async(email, whois_lookup=lookup, cache=cache_file, now=NOW)
    assert _named(first, "domain_age_lt_7d").error is not None
    assert lookup.calls == 1

    # Within the cooldown the registry is not re-dialled...
    await analyze_async(email, whois_lookup=lookup, cache=cache_file, now=NOW)
    assert lookup.calls == 1

    # ...but the cooldown is minutes, not the six hours a negative entry gets.
    real_time = cache_module.time.time
    lookup.answer = DomainAge(domain="corp-invoices.com", created_at=NOW - timedelta(days=2))
    try:
        cache_module.time.time = lambda: real_time() + WHOIS_UNAVAILABLE_TTL_SECONDS + 1
        recovered = await analyze_async(
            email, whois_lookup=lookup, cache=cache_file, now=NOW
        )
    finally:
        cache_module.time.time = real_time

    assert lookup.calls == 2  # retried
    age = _named(recovered, "domain_age_lt_7d")
    assert age.error is None
    assert age.score > 0  # and the domain's real age was finally learned


@pytest.mark.asyncio
async def test_the_cooldown_is_far_shorter_than_a_negative_result(email):
    assert WHOIS_UNAVAILABLE_TTL_SECONDS < WHOIS_NEGATIVE_TTL_SECONDS < WHOIS_TTL_SECONDS
    assert WHOIS_UNAVAILABLE_TTL_SECONDS <= 15 * 60


def test_the_layer_one_budget_is_unchanged():
    """The fix must not have bought correctness with a bigger budget."""
    from core.orchestrator import DEFAULT_LAYER_TIMEOUTS

    assert DEFAULT_LAYER_TIMEOUTS[DetectionLayer.L1] == 8.0
    assert l1_headers.DEFAULT_WHOIS_TIMEOUT == 5.0
    assert l1_headers.DEFAULT_WHOIS_TIMEOUT < DEFAULT_LAYER_TIMEOUTS[DetectionLayer.L1]


@pytest.mark.asyncio
async def test_l2_l4_still_complete_while_a_whois_lookup_times_out(email, monkeypatch):
    monkeypatch.setattr(
        l1_headers,
        "default_whois_lookup",
        lambda: TimeLimitedWhoisLookup(SlowWhois(30.0), timeout=0.05),
    )

    started = time.perf_counter()
    results = await run_layers(email)
    elapsed = time.perf_counter() - started

    assert elapsed < 2.0
    assert all(r.completed for r in results)
    for result in results[1:]:
        assert result.duration_ms < 100
    assert _named(results[0].signals, "domain_age_lt_7d").metadata["authoritative"] is False


class _SwitchableWhois:
    """Raises, then answers, so a retry after a cooldown can be observed."""

    def __init__(self, failure: BaseException) -> None:
        self.answer: DomainAge | None = None
        self.failure = failure
        self.calls = 0

    def creation_date(self, domain: str) -> DomainAge:
        self.calls += 1
        if self.answer is not None:
            return self.answer
        raise self.failure


# --------------------------------------------------------------------------
# 8. PythonWhoisLookup sorts the client's own exceptions into the two halves
#    of the protocol. Exercised against a stub module - no network, and no
#    dependency on python-whois being installed.
# --------------------------------------------------------------------------


def _stub_whois_module(monkeypatch, *, raises=None, record=None):
    import sys
    import types

    exceptions = types.ModuleType("whois.exceptions")

    class WhoisDomainNotFoundError(Exception):
        pass

    class WhoisQuotaExceededError(Exception):
        pass

    exceptions.WhoisDomainNotFoundError = WhoisDomainNotFoundError
    exceptions.WhoisQuotaExceededError = WhoisQuotaExceededError

    module = types.ModuleType("whois")
    module.exceptions = exceptions

    def _whois(domain):
        if raises is not None:
            raise raises(exceptions)
        return record

    module.whois = _whois
    monkeypatch.setitem(sys.modules, "whois", module)
    monkeypatch.setitem(sys.modules, "whois.exceptions", exceptions)
    return exceptions


def test_a_registry_saying_no_such_domain_is_an_authoritative_negative(monkeypatch):
    """"No match for domain" is an answer, and becomes DomainAge(created_at=None)."""
    _stub_whois_module(
        monkeypatch, raises=lambda ex: ex.WhoisDomainNotFoundError("No match")
    )

    age = l1_headers.PythonWhoisLookup().creation_date("nonexistent.com")

    assert age == DomainAge(domain="nonexistent.com", created_at=None)


def test_a_quota_rejection_is_not_an_answer_about_the_domain(monkeypatch):
    _stub_whois_module(
        monkeypatch, raises=lambda ex: ex.WhoisQuotaExceededError("slow down")
    )

    with pytest.raises(WhoisUnavailable) as caught:
        l1_headers.PythonWhoisLookup().creation_date("corp.com")

    # The registry's raw response is not carried into the message.
    assert "slow down" not in str(caught.value)
    assert "WhoisQuotaExceededError" in str(caught.value)


def test_a_transport_failure_is_not_an_answer_about_the_domain(monkeypatch):
    _stub_whois_module(monkeypatch, raises=lambda ex: ConnectionResetError("reset"))

    with pytest.raises(WhoisUnavailable):
        l1_headers.PythonWhoisLookup().creation_date("corp.com")


def test_a_record_with_a_creation_date_is_returned(monkeypatch):
    class Record:
        creation_date = datetime(2020, 1, 2, tzinfo=timezone.utc)

    _stub_whois_module(monkeypatch, record=Record())

    age = l1_headers.PythonWhoisLookup().creation_date("corp.com")

    assert age.created_at == datetime(2020, 1, 2, tzinfo=timezone.utc)


def test_a_client_failure_reaching_analyze_becomes_a_cooldown_not_a_negative(
    monkeypatch, email
):
    """End to end: client raises -> WhoisUnavailable -> cooldown, not a result."""
    _stub_whois_module(monkeypatch, raises=lambda ex: ex.WhoisQuotaExceededError("x"))
    cache = RecordingCache()

    signal = l1_headers.analyze_domain_age(
        email, lookup=l1_headers.PythonWhoisLookup(), cache=cache, now=NOW
    )

    assert signal.score == 0.0
    assert signal.metadata["authoritative"] is False
    assert cache.sets[0][1]["status"] == "unavailable"
    assert cache.sets[0][2] == WHOIS_UNAVAILABLE_TTL_SECONDS
