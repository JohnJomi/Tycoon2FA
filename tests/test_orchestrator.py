"""Unit tests for core/orchestrator.py.

Deterministic and offline: every layer used here is a small async double, and
no test touches the network, Gmail or OAuth. Timeouts are shrunk to
milliseconds so the suite stays fast.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from core.models import DetectionLayer, DetectionSignal, LayerResult, ParsedEmail, RiskLevel
from core.orchestrator import (
    DEFAULT_LAYER_TIMEOUTS,
    DEFAULT_TOTAL_TIMEOUT,
    STUB_L1_SIGNAL_NAME,
    run_layers,
)

ALL_LAYERS = [DetectionLayer.L1, DetectionLayer.L2, DetectionLayer.L3, DetectionLayer.L4]


@pytest.fixture
def email() -> ParsedEmail:
    return ParsedEmail(message_id="<orchestrated@example.com>", from_addr="sender@example.com")


def _signal(layer: DetectionLayer, name: str = "double") -> DetectionSignal:
    return DetectionSignal(
        layer=layer,
        name=name,
        score=0.4,
        severity=RiskLevel.MEDIUM,
        evidence="test double evidence",
    )


def _by_layer(results: list[LayerResult]) -> dict[DetectionLayer, LayerResult]:
    return {r.layer: r for r in results}


# --- async test doubles ----------------------------------------------------


def make_empty_layer():
    async def layer(_email):
        return []

    return layer


def make_signal_layer(layer_id: DetectionLayer, name: str = "double"):
    async def layer(_email):
        return [_signal(layer_id, name)]

    return layer


def make_slow_layer(delay: float):
    async def layer(_email):
        await asyncio.sleep(delay)
        return []

    return layer


def make_failing_layer(exc: Exception):
    async def layer(_email):
        raise exc

    return layer


def _layers(overrides=None):
    """All four layers as empty doubles, with the given overrides applied."""
    mapping = {layer: make_empty_layer() for layer in ALL_LAYERS}
    mapping.update(overrides or {})
    return mapping


# --------------------------------------------------------------------------
# 1, 10. All four layers execute and all four results come back
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_four_layers_execute(email):
    called: list[DetectionLayer] = []

    def recorder(layer_id):
        async def layer(_email):
            called.append(layer_id)
            return []

        return layer

    await run_layers(email, layers={layer: recorder(layer) for layer in ALL_LAYERS})

    assert sorted(called, key=int) == ALL_LAYERS


@pytest.mark.asyncio
async def test_result_contains_all_four_layers_in_order(email):
    results = await run_layers(email, layers=_layers())

    assert [r.layer for r in results] == ALL_LAYERS
    assert all(isinstance(r, LayerResult) for r in results)


@pytest.mark.asyncio
async def test_layers_receive_the_parsed_email(email):
    seen: list[ParsedEmail] = []

    async def layer(parsed):
        seen.append(parsed)
        return []

    await run_layers(email, layers={DetectionLayer.L1: layer})

    assert seen == [email]


# --------------------------------------------------------------------------
# 2. Concurrency, not serial execution
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_layers_run_concurrently_not_serially(email):
    delay = 0.10
    started = time.perf_counter()

    await run_layers(
        email,
        layers={layer: make_slow_layer(delay) for layer in ALL_LAYERS},
        timeouts={layer: 5.0 for layer in ALL_LAYERS},
    )

    elapsed = time.perf_counter() - started
    # Serial execution would take ~4x delay; concurrent takes ~1x.
    assert elapsed < delay * 2.5, f"layers appear to run serially ({elapsed:.3f}s)"


@pytest.mark.asyncio
async def test_layers_overlap_in_time(email):
    """Stronger than wall-clock: prove two layers are in flight at once."""
    concurrent = 0
    peak = 0

    async def layer(_email):
        nonlocal concurrent, peak
        concurrent += 1
        peak = max(peak, concurrent)
        await asyncio.sleep(0.05)
        concurrent -= 1
        return []

    await run_layers(email, layers={layer_id: layer for layer_id in ALL_LAYERS})

    assert peak == 4


# --------------------------------------------------------------------------
# 3-4, 11. Stub behaviour and the completed-empty vs incomplete distinction
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_l1_stub_returns_its_hardcoded_signal(email):
    results = _by_layer(await run_layers(email))
    l1 = results[DetectionLayer.L1]

    assert l1.completed is True
    assert len(l1.signals) == 1
    assert l1.signals[0].name == STUB_L1_SIGNAL_NAME
    assert l1.signals[0].layer is DetectionLayer.L1


@pytest.mark.asyncio
async def test_l1_stub_signal_is_marked_as_a_stub_not_a_real_detection(email):
    results = _by_layer(await run_layers(email))
    signal = results[DetectionLayer.L1].signals[0]

    assert "STUB" in signal.evidence.upper()
    assert signal.metadata.get("stub") is True


@pytest.mark.asyncio
async def test_default_l2_l3_l4_stubs_complete_with_no_signals(email):
    results = _by_layer(await run_layers(email))

    for layer in (DetectionLayer.L2, DetectionLayer.L3, DetectionLayer.L4):
        assert results[layer].completed is True
        assert results[layer].signals == []
        assert results[layer].error is None


@pytest.mark.asyncio
async def test_completed_empty_layer_is_distinguishable_from_an_incomplete_one(email):
    results = _by_layer(
        await run_layers(
            email,
            layers=_layers({DetectionLayer.L3: make_slow_layer(1.0)}),
            timeouts={DetectionLayer.L3: 0.01},
        )
    )

    found_nothing = results[DetectionLayer.L2]
    could_not_run = results[DetectionLayer.L3]

    # Both have zero signals, but they mean opposite things.
    assert found_nothing.signals == [] and could_not_run.signals == []
    assert found_nothing.completed is True
    assert found_nothing.error is None
    assert could_not_run.completed is False
    assert could_not_run.error is not None


# --------------------------------------------------------------------------
# 5-6. Timeout behaviour
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timed_out_layer_is_marked_incomplete_with_an_error(email):
    results = _by_layer(
        await run_layers(
            email,
            layers=_layers({DetectionLayer.L2: make_slow_layer(1.0)}),
            timeouts={DetectionLayer.L2: 0.01},
        )
    )
    l2 = results[DetectionLayer.L2]

    assert l2.completed is False
    assert l2.error is not None
    assert "timeout" in l2.error.lower()


@pytest.mark.asyncio
async def test_one_layer_timeout_does_not_prevent_the_others_completing(email):
    results = _by_layer(
        await run_layers(
            email,
            layers=_layers(
                {
                    DetectionLayer.L1: make_signal_layer(DetectionLayer.L1),
                    DetectionLayer.L2: make_slow_layer(1.0),
                }
            ),
            timeouts={DetectionLayer.L2: 0.01},
        )
    )

    assert results[DetectionLayer.L2].completed is False
    for layer in (DetectionLayer.L1, DetectionLayer.L3, DetectionLayer.L4):
        assert results[layer].completed is True
    assert len(results[DetectionLayer.L1].signals) == 1


@pytest.mark.asyncio
async def test_timeout_is_configurable_per_layer(email):
    results = _by_layer(
        await run_layers(
            email,
            layers={layer: make_slow_layer(0.05) for layer in ALL_LAYERS},
            timeouts={
                DetectionLayer.L1: 5.0,
                DetectionLayer.L2: 0.01,
                DetectionLayer.L3: 5.0,
                DetectionLayer.L4: 0.01,
            },
        )
    )

    assert results[DetectionLayer.L1].completed is True
    assert results[DetectionLayer.L2].completed is False
    assert results[DetectionLayer.L3].completed is True
    assert results[DetectionLayer.L4].completed is False


def test_default_timeouts_match_the_architecture_budget():
    assert DEFAULT_LAYER_TIMEOUTS[DetectionLayer.L1] == 8.0
    assert DEFAULT_LAYER_TIMEOUTS[DetectionLayer.L2] == 15.0
    assert DEFAULT_LAYER_TIMEOUTS[DetectionLayer.L3] == 10.0
    assert DEFAULT_LAYER_TIMEOUTS[DetectionLayer.L4] == 6.0
    assert DEFAULT_TOTAL_TIMEOUT == 20.0
    assert all(t < DEFAULT_TOTAL_TIMEOUT for t in DEFAULT_LAYER_TIMEOUTS.values())


@pytest.mark.asyncio
async def test_total_budget_backstop_reports_absence_not_a_clean_run(email):
    """If the whole gather overruns, every layer is unknown - never clean."""
    results = await run_layers(
        email,
        layers={layer: make_slow_layer(1.0) for layer in ALL_LAYERS},
        timeouts={layer: 5.0 for layer in ALL_LAYERS},
        total_timeout=0.02,
    )

    assert len(results) == 4
    assert all(r.completed is False for r in results)
    assert all(r.signals == [] for r in results)
    assert all("total budget" in r.error for r in results)
    # The elapsed time is recorded, not left at the default 0.
    assert all(r.duration_ms > 0 for r in results)


@pytest.mark.asyncio
async def test_total_timeout_can_be_disabled(email):
    results = await run_layers(
        email,
        layers=_layers({DetectionLayer.L2: make_slow_layer(0.05)}),
        timeouts={DetectionLayer.L2: 5.0},
        total_timeout=None,
    )

    assert all(r.completed for r in results)


# --------------------------------------------------------------------------
# 7-8. Failure isolation
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failed_layer_is_incomplete_and_records_the_error(email):
    results = _by_layer(
        await run_layers(
            email,
            layers=_layers(
                {DetectionLayer.L2: make_failing_layer(RuntimeError("upstream exploded"))}
            ),
        )
    )
    l2 = results[DetectionLayer.L2]

    assert l2.completed is False
    assert l2.error is not None
    assert "RuntimeError" in l2.error
    assert "upstream exploded" in l2.error


@pytest.mark.asyncio
async def test_one_layer_failure_does_not_prevent_the_others(email):
    results = _by_layer(
        await run_layers(
            email,
            layers=_layers(
                {
                    DetectionLayer.L1: make_signal_layer(DetectionLayer.L1),
                    DetectionLayer.L2: make_failing_layer(ValueError("bad state")),
                }
            ),
        )
    )

    assert results[DetectionLayer.L2].completed is False
    assert results[DetectionLayer.L1].completed is True
    assert len(results[DetectionLayer.L1].signals) == 1
    assert results[DetectionLayer.L3].completed is True
    assert results[DetectionLayer.L4].completed is True


@pytest.mark.asyncio
async def test_failed_and_timed_out_layers_produce_no_signals(email):
    results = _by_layer(
        await run_layers(
            email,
            layers=_layers(
                {
                    DetectionLayer.L2: make_failing_layer(RuntimeError("boom")),
                    DetectionLayer.L3: make_slow_layer(1.0),
                }
            ),
            timeouts={DetectionLayer.L3: 0.01},
        )
    )

    assert results[DetectionLayer.L2].signals == []
    assert results[DetectionLayer.L3].signals == []


@pytest.mark.asyncio
async def test_error_text_carries_no_stack_trace(email):
    results = _by_layer(
        await run_layers(
            email,
            layers=_layers({DetectionLayer.L2: make_failing_layer(RuntimeError("boom"))}),
        )
    )

    error = results[DetectionLayer.L2].error
    assert "Traceback" not in error
    assert "File \"" not in error
    assert error.count("\n") == 0


@pytest.mark.asyncio
async def test_a_failing_layer_is_never_reported_as_clean(email):
    """The core safety property: failure must not read as a genuine negative."""
    results = _by_layer(
        await run_layers(
            email,
            layers=_layers({DetectionLayer.L4: make_failing_layer(RuntimeError("intel feed 503"))}),
        )
    )
    l4 = results[DetectionLayer.L4]

    assert not (l4.completed is True and l4.signals == [])
    assert l4.completed is False


@pytest.mark.asyncio
async def test_layer_returning_none_is_incomplete_not_an_empty_success(email):
    """None is a broken layer, not a genuine negative."""

    async def returns_none(_email):
        return None

    results = _by_layer(await run_layers(email, layers={DetectionLayer.L2: returns_none}))
    l2 = results[DetectionLayer.L2]

    assert l2.completed is False
    assert l2.signals == []
    assert l2.error is not None
    assert "None" in l2.error


@pytest.mark.asyncio
async def test_a_none_returning_layer_does_not_stop_the_others(email):
    async def returns_none(_email):
        return None

    results = _by_layer(await run_layers(email, layers={DetectionLayer.L2: returns_none}))

    assert results[DetectionLayer.L2].completed is False
    for layer in (DetectionLayer.L1, DetectionLayer.L3, DetectionLayer.L4):
        assert results[layer].completed is True


@pytest.mark.asyncio
async def test_partial_custom_layer_mapping_still_runs_all_four_layers(email):
    """Supplying one layer overrides that default; it does not replace them all."""
    results = await run_layers(
        email, layers={DetectionLayer.L2: make_signal_layer(DetectionLayer.L2, "custom")}
    )

    assert [r.layer for r in results] == ALL_LAYERS
    assert results[1].signals[0].name == "custom"
    # The untouched defaults still ran.
    assert results[0].signals[0].name == STUB_L1_SIGNAL_NAME
    assert results[2].completed is True and results[2].signals == []
    assert results[3].completed is True and results[3].signals == []


# --------------------------------------------------------------------------
# 9. duration_ms
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_duration_ms_is_populated_for_every_layer(email):
    results = await run_layers(email)

    assert all(isinstance(r.duration_ms, int) and r.duration_ms >= 0 for r in results)


@pytest.mark.asyncio
async def test_duration_ms_reflects_a_slow_layer(email):
    results = _by_layer(
        await run_layers(
            email,
            layers=_layers({DetectionLayer.L3: make_slow_layer(0.05)}),
            timeouts={DetectionLayer.L3: 5.0},
        )
    )

    assert results[DetectionLayer.L3].duration_ms >= 40


@pytest.mark.asyncio
async def test_duration_ms_is_preserved_on_timeout(email):
    results = _by_layer(
        await run_layers(
            email,
            layers=_layers({DetectionLayer.L2: make_slow_layer(1.0)}),
            timeouts={DetectionLayer.L2: 0.05},
        )
    )

    assert results[DetectionLayer.L2].duration_ms >= 40


# --------------------------------------------------------------------------
# Scoring boundary
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_results_expose_what_scoring_needs_to_renormalize(email):
    results = await run_layers(
        email,
        layers=_layers(
            {
                DetectionLayer.L1: make_signal_layer(DetectionLayer.L1),
                DetectionLayer.L2: make_failing_layer(RuntimeError("down")),
            }
        ),
    )

    completed = [r.layer for r in results if r.completed]
    incomplete = [r.layer for r in results if not r.completed]

    assert completed == [DetectionLayer.L1, DetectionLayer.L3, DetectionLayer.L4]
    assert incomplete == [DetectionLayer.L2]


@pytest.mark.asyncio
async def test_orchestrator_does_not_score_or_decide_a_verdict(email):
    results = await run_layers(email)

    assert isinstance(results, list)
    assert all(isinstance(r, LayerResult) for r in results)
    assert not any(hasattr(r, "score") or hasattr(r, "verdict") for r in results)
