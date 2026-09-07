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

Layers 1, 3 and 4 are real: `DEFAULT_LAYERS` points at each layer's own async
adapter. Layer 2 is not written yet, and says so: they report `completed=False`, which is the same state a timeout
produces and means "no information", not "nothing found". Scoring then
redistributes their weight onto the layers that did run, so an unwritten layer
cannot dilute a real finding.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence

from core.models import DetectionLayer, DetectionSignal, LayerResult, ParsedEmail
from layers import l1_headers, l3_nlp, l4_intel

__all__ = [
    "DEFAULT_LAYERS",
    "unimplemented_layer",
    "DEFAULT_LAYER_TIMEOUTS",
    "DEFAULT_TOTAL_TIMEOUT",
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
# The layer mapping
#
# L1, L3 and L4 are the real implementations. L2 is not written yet.
#
# An unwritten layer must not return `[]`. A layer that completes with no
# signals is making a claim - "I ran, and I found nothing" - and scoring counts
# that claim as a genuine 0.0 at the layer's full configured weight. With three
# unwritten layers carrying 0.70 of the weight between them, a Layer 1 DMARC
# failure scoring 0.85 came out as a 0.255 composite: LOW, for a message whose
# authentication genuinely failed.
#
# `completed=False` is the state ARCHITECTURE.md section 2 already provides for
# exactly this - it is what a timeout produces, and it means "no information".
# `scoring.composite` then redistributes the weight across the layers that did
# run, so the same message scores 0.85 and reads HIGH. Raising is how a
# LayerCallable reports that it could not produce signals, so this needs no new
# vocabulary in either module.
# --------------------------------------------------------------------------


def unimplemented_layer(layer: DetectionLayer, description: str) -> LayerCallable:
    """A layer that has not been written, and reports itself as such.

    Not a stub that returns nothing: that would be indistinguishable from a
    layer that ran and found the message clean.
    """

    async def run(email: ParsedEmail) -> list[DetectionSignal]:
        raise NotImplementedError(
            f"{layer.name} ({description}) is not implemented yet"
        )

    run.__name__ = f"unimplemented_{layer.name.lower()}"
    run.__qualname__ = run.__name__
    return run


# `analyze_async` is Layer 1's own adapter onto LayerCallable, and every
# argument beyond the email defaults: the WHOIS client, its timeout and the
# shared cache are the layer's decisions, not the orchestrator's. This module
# still knows nothing about what any layer looks for.
DEFAULT_LAYERS: Mapping[DetectionLayer, LayerCallable] = {
    DetectionLayer.L1: l1_headers.analyze_async,
    DetectionLayer.L2: unimplemented_layer(DetectionLayer.L2, "URL & redirect chain"),
    DetectionLayer.L3: l3_nlp.analyze_async,
    DetectionLayer.L4: l4_intel.analyze_async,
}
