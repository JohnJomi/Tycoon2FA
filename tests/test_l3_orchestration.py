"""Integration tests for Layer 3 running through the orchestrator.

Proves the wiring, not the signals - each signal's own behaviour is covered by
tests/test_l3_*.py. Every model seam is injected, so no test here loads a
checkpoint, fits anything, or opens a socket.
"""

from __future__ import annotations

import asyncio

import pytest

from core.models import DetectionLayer, ParsedEmail, RiskLevel
from core.orchestrator import DEFAULT_LAYER_TIMEOUTS, DEFAULT_LAYERS, run_layers
from layers import l3_nlp
from layers.l3_nlp import FUSION_FEATURE_ORDER, Layer3Uninformative

L3_SIGNAL_NAMES = ["zero_width", "urgency", "perplexity", "burstiness", "fusion"]

BODY = (
    "Your account will be suspended within 24 hours. Verify your password "
    "immediately or you will lose access today. Please confirm your billing "
    "details now to avoid the suspension of your mailbox. Act quickly. "
    "Failure to respond within the stated period will result in the permanent "
    "closure of the account and the loss of all stored messages. Confirm your "
    "identity using the secure link that was provided to you earlier today."
)


class FakeUrgency:
    classes_ = [0, 1]

    def predict_proba(self, texts):
        return [[0.2, 0.8] for _ in texts]


class FakeLM:
    """One token per word, fixed per-token log-likelihood."""

    def encode(self, text: str):
        return [1] * len(text.split())

    def token_log_likelihoods(self, token_ids):
        return [-2.0] * (len(token_ids) - 1)


class FakeFusion:
    classes_ = [0, 1]

    def __init__(self) -> None:
        self.rows = []

    def predict_proba(self, rows):
        self.rows.extend(rows)
        return [[0.1, 0.9] for _ in rows]


@pytest.fixture
def email() -> ParsedEmail:
    return ParsedEmail(
        message_id="<m@example.com>",
        from_addr="sender@example.com",
        subject="Immediate action required",
        body_text=BODY,
    )


def wired(fusion: FakeFusion | None = None):
    """Layer 3 as a LayerCallable with every model seam injected."""

    async def run(email: ParsedEmail):
        return await l3_nlp.analyze_async(
            email,
            urgency_model=FakeUrgency(),
            language_model=FakeLM(),
            fusion_model=fusion or FakeFusion(),
        )

    return run


# --- the layer runs through the normal pipeline ---------------------------


def test_layer_three_is_registered_in_the_default_mapping() -> None:
    assert DEFAULT_LAYERS[DetectionLayer.L3] is l3_nlp.analyze_async
    assert DEFAULT_LAYER_TIMEOUTS[DetectionLayer.L3] == 10.0


@pytest.mark.asyncio
async def test_l3_signals_reach_the_result_through_the_pipeline(email) -> None:
    results = await run_layers(email, layers={DetectionLayer.L3: wired()})
    l3 = next(r for r in results if r.layer is DetectionLayer.L3)

    assert [r.layer for r in results] == list(DetectionLayer)
    assert l3.completed is True
    assert l3.error is None
    assert [s.name for s in l3.signals] == L3_SIGNAL_NAMES
    assert all(s.layer is DetectionLayer.L3 for s in l3.signals)
    assert l3.duration_ms >= 0


@pytest.mark.asyncio
async def test_the_fused_verdict_is_the_signal_that_carries_a_score(email) -> None:
    results = await run_layers(email, layers={DetectionLayer.L3: wired()})
    signals = {s.name: s for s in next(r for r in results if r.layer is DetectionLayer.L3).signals}

    assert signals["fusion"].score == pytest.approx(0.9)
    assert signals["fusion"].severity is RiskLevel.HIGH
    assert signals["perplexity"].score == 0.0  # a measurement, not a verdict
    assert signals["burstiness"].score == 0.0


# --- fusion receives the component measurements ---------------------------


@pytest.mark.asyncio
async def test_fusion_receives_the_four_component_measurements(email) -> None:
    fusion = FakeFusion()

    results = await run_layers(email, layers={DetectionLayer.L3: wired(fusion)})
    signals = {s.name: s for s in next(r for r in results if r.layer is DetectionLayer.L3).signals}

    assert len(fusion.rows) == 1
    row = fusion.rows[0]
    assert len(row) == len(FUSION_FEATURE_ORDER)
    # Each column is the measurement the matching signal published upstream.
    assert row[0] == signals["zero_width"].metadata["total"]
    assert row[1] == signals["urgency"].metadata["probability"]
    assert row[2] == signals["perplexity"].metadata["mean_nll"]
    assert row[3] == signals["burstiness"].metadata["perplexity_stddev"]
    assert signals["fusion"].metadata["features"] == dict(zip(FUSION_FEATURE_ORDER, row))


@pytest.mark.asyncio
async def test_components_are_computed_before_fusion(email) -> None:
    """Fusion cannot have run first: it was handed values only they produce."""
    fusion = FakeFusion()

    await run_layers(email, layers={DetectionLayer.L3: wired(fusion)})

    assert fusion.rows and all(value is not None for value in fusion.rows[0])


# --- fail-closed ----------------------------------------------------------


@pytest.mark.asyncio
async def test_a_missing_model_abstains_rather_than_reporting_a_clean_layer(email) -> None:
    """No artifacts on this machine: every model-backed signal abstains."""
    results = await run_layers(email)
    l3 = next(r for r in results if r.layer is DetectionLayer.L3)

    assert l3.completed is False
    assert l3.signals == []          # no fabricated negatives
    assert l3.error is not None      # and the reason is stated


@pytest.mark.asyncio
async def test_an_uninformative_layer_does_not_dilute_a_layer_one_finding(email) -> None:
    """The reason `Layer3Uninformative` exists: weight is redistributed, not spent."""
    from scoring.composite import layer_contributions

    results = await run_layers(email)
    contributions = layer_contributions(results)

    assert DetectionLayer.L3 not in contributions


@pytest.mark.asyncio
async def test_a_real_finding_still_completes_without_any_model(email) -> None:
    """zero-width evidence is real evidence, even with every model missing."""
    obfuscated = ParsedEmail(
        message_id="<m@example.com>",
        from_addr="sender@example.com",
        subject="Pay​ment",
        body_text=BODY,
    )

    results = await run_layers(obfuscated)
    l3 = next(r for r in results if r.layer is DetectionLayer.L3)

    assert l3.completed is True
    assert [s.name for s in l3.signals] == L3_SIGNAL_NAMES
    assert {s.name for s in l3.signals if s.error} == {"urgency", "perplexity", "burstiness", "fusion"}


@pytest.mark.asyncio
async def test_a_crashing_layer_three_does_not_stop_the_analysis(email) -> None:
    async def exploding(_email):
        raise RuntimeError("model server died")

    results = await run_layers(email, layers={DetectionLayer.L3: exploding})
    by_layer = {r.layer: r for r in results}

    assert by_layer[DetectionLayer.L3].completed is False
    assert "model server died" in by_layer[DetectionLayer.L3].error
    assert by_layer[DetectionLayer.L1].completed is True
    assert by_layer[DetectionLayer.L1].signals


@pytest.mark.asyncio
async def test_a_slow_layer_three_is_bounded_by_its_own_timeout(email) -> None:
    async def slow(_email):
        await asyncio.sleep(1.0)
        return []

    results = await run_layers(
        email,
        layers={DetectionLayer.L3: slow},
        timeouts={DetectionLayer.L3: 0.01},
    )
    by_layer = {r.layer: r for r in results}

    assert by_layer[DetectionLayer.L3].completed is False
    assert "timeout" in by_layer[DetectionLayer.L3].error
    assert by_layer[DetectionLayer.L1].completed is True


# --- existing behaviour is untouched --------------------------------------


@pytest.mark.asyncio
async def test_layer_one_and_the_other_layers_are_unchanged(email) -> None:
    results = await run_layers(email, layers={DetectionLayer.L3: wired()})
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
    # L4 is written now; this message's URLs are not checkable without feeds.
    assert by_layer[DetectionLayer.L4].completed is False
    assert by_layer[DetectionLayer.L4].error is not None


@pytest.mark.asyncio
async def test_the_layer_does_not_construct_its_own_models(email, monkeypatch) -> None:
    """Injected seams are used as given; no loader is reached."""
    for loader in ("default_urgency_model", "default_perplexity_model", "default_fusion_model"):
        monkeypatch.setattr(
            f"layers.l3_nlp.{loader}",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError(f"{loader} was called")),
        )

    results = await run_layers(email, layers={DetectionLayer.L3: wired()})

    assert next(r for r in results if r.layer is DetectionLayer.L3).completed is True


@pytest.mark.asyncio
async def test_analyze_always_returns_all_five_signals(email) -> None:
    """The sync entry point never raises; only the async adapter abstains."""
    signals = l3_nlp.analyze(email)

    assert [s.name for s in signals] == L3_SIGNAL_NAMES
    with pytest.raises(Layer3Uninformative):
        await l3_nlp.analyze_async(email)
