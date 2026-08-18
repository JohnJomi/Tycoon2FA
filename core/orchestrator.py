"""Async layer orchestration.

Per ARCHITECTURE.md section 5: run L1-L4 concurrently, bound each layer with
its own timeout, and degrade gracefully. A layer that times out or raises
comes back as `LayerResult(completed=False, signals=[])` so the later scoring
stage can redistribute its weight across the layers that did complete.

    ParsedEmail -> asyncio.gather(L1, L2, L3, L4) -> list[LayerResult]

The one rule that matters here: **an incomplete layer is never a clean
result.** A layer that ran and found nothing returns `completed=True` with an
empty signal list; a layer that timed out or crashed returns
`completed=False` with an error. Those two states stay distinguishable all the
way to scoring, because collapsing them turns an outage into an all-clear.

Scope: this module runs layers and reports what happened. It does not score,
does not decide a verdict, and does not know what any layer looks for.

TEMPORARY SCAFFOLD
------------------
The real detection layers do not exist yet. The stubs at the bottom of this
module stand in for them so the orchestration path can be exercised end to
end. They perform **no detection whatsoever** and are replaced by the real
`layers/` implementations in Phase 2.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence

from core.models import DetectionLayer, DetectionSignal, LayerResult, ParsedEmail, RiskLevel

__all__ = [
    "DEFAULT_LAYER_TIMEOUTS",
    "DEFAULT_TOTAL_TIMEOUT",
    "STUB_L1_SIGNAL_NAME",
    "LayerCallable",
    "run_layers",
]

# A layer is any coroutine taking the parsed email and returning its signals.
# The orchestrator - not the layer - decides completed/error/duration, so this
# conversion lives in exactly one place.
LayerCallable = Callable[[ParsedEmail], Awaitable[Sequence[DetectionSignal]]]

# Per-layer self-bounding, per ARCHITECTURE.md section 5. Overridable by the
# caller; nothing here hardcodes a timeout at a call site.
DEFAULT_LAYER_TIMEOUTS: Mapping[DetectionLayer, float] = {
    DetectionLayer.L1: 8.0,
    DetectionLayer.L2: 15.0,
    DetectionLayer.L3: 10.0,
    DetectionLayer.L4: 6.0,
}

# Wall-clock cap on the whole gather. A backstop only: every per-layer timeout
# is strictly below it, so it fires only if a layer refuses to be cancelled.
DEFAULT_TOTAL_TIMEOUT = 20.0

STUB_L1_SIGNAL_NAME = "orchestrator_stub"


def _elapsed_ms(started: float) -> int:
    """Milliseconds since a perf_counter start, never negative."""
    return max(0, round((time.perf_counter() - started) * 1000))


async def _run_layer(
    layer: DetectionLayer,
    run: LayerCallable,
    email: ParsedEmail,
    timeout: float,
) -> LayerResult:
    """Run one layer under its own timeout and convert the outcome.

    Never raises for a layer-level problem: a timeout or an exception becomes
    an incomplete LayerResult carrying the reason. No signals are invented for
    a layer that did not finish, and the elapsed time is recorded either way.
    """
    started = time.perf_counter()
    try:
        produced = await asyncio.wait_for(run(email), timeout=timeout)
        if produced is None:
            # A layer that returns nothing has not "found nothing" - it is
            # broken. Treating None as an empty success would report a
            # defective layer as a genuine negative.
            raise TypeError("layer returned None instead of a sequence of signals")
        signals = list(produced)
    except asyncio.TimeoutError:
        return LayerResult(
            layer=layer,
            completed=False,
            signals=[],
            error=f"timeout after {timeout:g}s",
            duration_ms=_elapsed_ms(started),
        )
    except Exception as exc:  # noqa: BLE001 - one layer must not sink the run
        # Type and message only. A stack trace is debugging output, not result
        # data, and this string is shown in the UI.
        return LayerResult(
            layer=layer,
            completed=False,
            signals=[],
            error=f"{type(exc).__name__}: {exc}",
            duration_ms=_elapsed_ms(started),
        )

    return LayerResult(
        layer=layer,
        completed=True,
        signals=signals,
        duration_ms=_elapsed_ms(started),
    )


async def run_layers(
    email: ParsedEmail,
    *,
    layers: Mapping[DetectionLayer, LayerCallable] | None = None,
    timeouts: Mapping[DetectionLayer, float] | None = None,
    total_timeout: float | None = DEFAULT_TOTAL_TIMEOUT,
) -> list[LayerResult]:
    """Run every layer concurrently and return one LayerResult per layer.

    Results come back in layer order (L1..L4) regardless of which finished
    first, so the output is deterministic. One layer timing out or raising
    does not stop the others: each is isolated inside its own wrapper before
    the gather ever sees it.

    `layers` and `timeouts` exist so callers and tests can substitute layer
    implementations and shrink the clock; both default to the real mapping.
    """
    # Start from the defaults so a partial mapping overrides only the layers
    # it names; L1-L4 always run.
    layer_map = dict(DEFAULT_LAYERS)
    if layers:
        layer_map.update(layers)
    timeout_map = dict(DEFAULT_LAYER_TIMEOUTS)
    if timeouts:
        timeout_map.update(timeouts)

    ordered = sorted(layer_map, key=int)
    pending = [
        _run_layer(layer, layer_map[layer], email, timeout_map.get(layer, DEFAULT_TOTAL_TIMEOUT))
        for layer in ordered
    ]

    started = time.perf_counter()
    gathered = asyncio.gather(*pending, return_exceptions=True)
    try:
        results = await (
            asyncio.wait_for(gathered, timeout=total_timeout)
            if total_timeout is not None
            else gathered
        )
    except asyncio.TimeoutError:
        # The backstop fired: a layer would not yield even to cancellation, so
        # nothing can be trusted to have finished. Report absence of
        # information for every layer rather than a fabricated all-clear.
        elapsed_ms = _elapsed_ms(started)
        return [
            LayerResult(
                layer=layer,
                completed=False,
                signals=[],
                error=f"pipeline exceeded its {total_timeout:g}s total budget",
                duration_ms=elapsed_ms,
            )
            for layer in ordered
        ]

    # _run_layer swallows layer-level failures, so an exception here would be a
    # defect in the wrapper itself. Report it as incomplete rather than raising
    # and losing the layers that did work.
    return [
        result
        if isinstance(result, LayerResult)
        else LayerResult(
            layer=layer,
            completed=False,
            signals=[],
            error=f"orchestrator error: {type(result).__name__}: {result}",
            duration_ms=_elapsed_ms(started),
        )
        for layer, result in zip(ordered, results)
    ]


# --------------------------------------------------------------------------
# TEMPORARY LAYER STUBS - scaffolding for Phase 1 only
#
# These are not detectors. They exist so the orchestration path can be tested
# before layers/ is implemented, and they are deleted once the real layers
# land in Phase 2. None of them looks at the email at all.
# --------------------------------------------------------------------------


async def stub_l1(email: ParsedEmail) -> list[DetectionSignal]:
    """Return one hardcoded placeholder signal. Performs no detection."""
    return [
        DetectionSignal(
            layer=DetectionLayer.L1,
            name=STUB_L1_SIGNAL_NAME,
            score=0.5,
            severity=RiskLevel.MEDIUM,
            evidence=(
                "TEMPORARY STUB: hardcoded placeholder emitted by "
                "core/orchestrator.py. No header, domain or authentication "
                "check has been performed."
            ),
            metadata={"stub": True},
        )
    ]


async def stub_empty(email: ParsedEmail) -> list[DetectionSignal]:
    """Complete with no signals. A genuine negative, not an abstention."""
    return []


DEFAULT_LAYERS: Mapping[DetectionLayer, LayerCallable] = {
    DetectionLayer.L1: stub_l1,
    DetectionLayer.L2: stub_empty,
    DetectionLayer.L3: stub_empty,
    DetectionLayer.L4: stub_empty,
}
