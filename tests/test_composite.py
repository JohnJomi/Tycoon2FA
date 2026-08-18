"""Unit tests for scoring/composite.py. Deterministic and offline."""

from __future__ import annotations

import pytest

from core.models import DetectionLayer, DetectionSignal, LayerResult, RiskLevel
from scoring.composite import (
    ScoringWeights,
    UnscoreableError,
    layer_contributions,
    layer_score,
    load_weights,
    risk_level,
    score,
)

WEIGHTS = ScoringWeights(
    layer_weights={
        DetectionLayer.L1: 0.30,
        DetectionLayer.L2: 0.30,
        DetectionLayer.L3: 0.20,
        DetectionLayer.L4: 0.20,
    },
    deliver_threshold=0.35,
    block_threshold=0.65,
)


def sig(layer: DetectionLayer, value: float, *, error: str | None = None) -> DetectionSignal:
    return DetectionSignal(
        layer=layer,
        name="test_signal",
        score=value,
        severity=RiskLevel.MEDIUM,
        evidence="test evidence",
        error=error,
    )


def done(layer: DetectionLayer, *signals: DetectionSignal) -> LayerResult:
    return LayerResult(layer=layer, completed=True, signals=list(signals), duration_ms=1)


def failed(layer: DetectionLayer, reason: str = "timeout after 15s") -> LayerResult:
    return LayerResult(layer=layer, completed=False, signals=[], error=reason, duration_ms=1)


def all_layers(**scores: float) -> list[LayerResult]:
    """Completed results for L1-L4, e.g. all_layers(l1=0.8) -> others score 0."""
    out = []
    for layer in DetectionLayer:
        value = scores.get(layer.name.lower())
        out.append(done(layer, sig(layer, value)) if value is not None else done(layer))
    return out


# --- configuration ---------------------------------------------------------


def test_weights_come_from_the_project_config_file():
    loaded = load_weights()

    assert loaded.layer_weights == WEIGHTS.layer_weights
    assert (loaded.deliver_threshold, loaded.block_threshold) == (0.35, 0.65)


def test_weights_path_is_overridable(tmp_path):
    config = tmp_path / "w.yaml"
    config.write_text(
        "layers: {l1: 1.0, l2: 0.0, l3: 0.0, l4: 0.0}\nthresholds: {deliver: 0.1, block: 0.2}\n"
    )

    loaded = load_weights(config)

    assert loaded.layer_weights[DetectionLayer.L1] == 1.0
    assert loaded.block_threshold == 0.2


def test_malformed_config_is_rejected(tmp_path):
    config = tmp_path / "w.yaml"
    config.write_text("layers: {l1: 0.3}\nthresholds: {deliver: 0.35, block: 0.65}\n")

    with pytest.raises(UnscoreableError, match="L2"):
        load_weights(config)


# --- normal weighted scoring ----------------------------------------------


def test_weighted_sum_across_completed_layers():
    results = all_layers(l1=1.0, l2=0.5, l3=0.0, l4=0.0)

    # 0.30*1.0 + 0.30*0.5 + 0.20*0 + 0.20*0
    assert score("<m@e>", results, weights=WEIGHTS).score == pytest.approx(0.45)


def test_all_layers_clean_scores_zero_and_bands_low():
    assessment = score("<m@e>", all_layers(), weights=WEIGHTS)

    assert assessment.score == 0.0
    assert assessment.level is RiskLevel.LOW
    assert assessment.layers_completed == list(DetectionLayer)


def test_layer_score_takes_the_strongest_signal():
    result = done(DetectionLayer.L1, sig(DetectionLayer.L1, 0.2), sig(DetectionLayer.L1, 0.9))

    assert layer_score(result) == 0.9


def test_abstaining_signals_do_not_contribute():
    """A signal with an error could not be computed; it is not a 0.0 finding."""
    result = done(DetectionLayer.L1, sig(DetectionLayer.L1, 0.9, error="body too short"))

    assert layer_score(result) == 0.0


def test_completed_layer_with_no_signals_scores_zero():
    assert layer_score(done(DetectionLayer.L4)) == 0.0


# --- renormalization -------------------------------------------------------


def test_incomplete_layer_weight_is_redistributed():
    results = [
        done(DetectionLayer.L1, sig(DetectionLayer.L1, 1.0)),
        failed(DetectionLayer.L2),
        done(DetectionLayer.L3),
        done(DetectionLayer.L4),
    ]

    # L2's 0.30 is redistributed; L1 now carries 0.30/0.70 of the total.
    assert score("<m@e>", results, weights=WEIGHTS).score == pytest.approx(0.30 / 0.70)


def test_l2_timeout_does_not_pull_the_score_toward_clean():
    """The ROADMAP acceptance criterion for this module."""
    with_l2_clean = [
        done(DetectionLayer.L1, sig(DetectionLayer.L1, 1.0)),
        done(DetectionLayer.L2),
        done(DetectionLayer.L3),
        done(DetectionLayer.L4),
    ]
    with_l2_timeout = [
        done(DetectionLayer.L1, sig(DetectionLayer.L1, 1.0)),
        failed(DetectionLayer.L2),
        done(DetectionLayer.L3),
        done(DetectionLayer.L4),
    ]

    clean = score("<m@e>", with_l2_clean, weights=WEIGHTS).score
    timed_out = score("<m@e>", with_l2_timeout, weights=WEIGHTS).score

    assert timed_out > clean
    assert clean == pytest.approx(0.30)
    assert timed_out == pytest.approx(0.30 / 0.70)


def test_incomplete_layer_is_not_scored_as_zero():
    """If L2 were scored 0 rather than renormalized, the score would drop."""
    results = [
        done(DetectionLayer.L1, sig(DetectionLayer.L1, 1.0)),
        failed(DetectionLayer.L2),
        done(DetectionLayer.L3),
        done(DetectionLayer.L4),
    ]
    naive_if_treated_as_clean = 0.30 * 1.0

    assert score("<m@e>", results, weights=WEIGHTS).score > naive_if_treated_as_clean


def test_renormalized_weights_still_sum_to_one():
    results = [
        done(DetectionLayer.L1, sig(DetectionLayer.L1, 1.0)),
        failed(DetectionLayer.L2),
        done(DetectionLayer.L3, sig(DetectionLayer.L3, 1.0)),
        done(DetectionLayer.L4, sig(DetectionLayer.L4, 1.0)),
    ]

    assert score("<m@e>", results, weights=WEIGHTS).score == pytest.approx(1.0)


def test_only_completed_layers_are_recorded():
    results = [
        done(DetectionLayer.L1),
        failed(DetectionLayer.L2),
        done(DetectionLayer.L3),
        done(DetectionLayer.L4),
    ]

    assessment = score("<m@e>", results, weights=WEIGHTS)

    assert assessment.layers_completed == [
        DetectionLayer.L1,
        DetectionLayer.L3,
        DetectionLayer.L4,
    ]


def test_contributions_sum_to_the_composite_score():
    results = [
        done(DetectionLayer.L1, sig(DetectionLayer.L1, 0.8)),
        failed(DetectionLayer.L2),
        done(DetectionLayer.L3, sig(DetectionLayer.L3, 0.4)),
        done(DetectionLayer.L4),
    ]

    contributions = layer_contributions(results, WEIGHTS)
    assessment = score("<m@e>", results, weights=WEIGHTS)

    assert DetectionLayer.L2 not in contributions
    assert sum(contributions.values()) == pytest.approx(assessment.score)


# --- all layers incomplete -------------------------------------------------


def test_all_layers_incomplete_raises_rather_than_scoring_clean():
    results = [failed(layer) for layer in DetectionLayer]

    with pytest.raises(UnscoreableError, match="no layer completed"):
        score("<m@e>", results, weights=WEIGHTS)


def test_empty_result_list_raises():
    with pytest.raises(UnscoreableError):
        score("<m@e>", [], weights=WEIGHTS)


def test_completed_layers_carrying_no_weight_raise():
    zero = ScoringWeights(
        layer_weights={layer: 0.0 for layer in DetectionLayer},
        deliver_threshold=0.35,
        block_threshold=0.65,
    )

    with pytest.raises(UnscoreableError, match="no configured weight"):
        score("<m@e>", all_layers(), weights=zero)


# --- risk bands ------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, RiskLevel.LOW),
        (0.349, RiskLevel.LOW),
        (0.35, RiskLevel.MEDIUM),  # deliver threshold is inclusive of warn
        (0.5, RiskLevel.MEDIUM),
        (0.649, RiskLevel.MEDIUM),
        (0.65, RiskLevel.HIGH),  # block threshold is inclusive of block
        (1.0, RiskLevel.HIGH),
    ],
)
def test_risk_band_boundaries(value, expected):
    assert risk_level(value, WEIGHTS) is expected


def test_band_is_applied_to_the_assessment():
    high = score("<m@e>", all_layers(l1=1.0, l2=1.0, l3=1.0, l4=1.0), weights=WEIGHTS)
    low = score("<m@e>", all_layers(), weights=WEIGHTS)

    assert high.level is RiskLevel.HIGH and high.score == pytest.approx(1.0)
    assert low.level is RiskLevel.LOW


def test_scoring_produces_no_operational_action():
    """Risk level only; deliver/warn/block belongs to the later policy stage."""
    assessment = score("<m@e>", all_layers(l1=1.0), weights=WEIGHTS)

    assert assessment.level in set(RiskLevel)
    assert not hasattr(assessment, "verdict")
    assert not hasattr(assessment, "action")


# --- explainability --------------------------------------------------------


def test_assessment_retains_every_signal_including_from_failed_runs():
    l1_signal = sig(DetectionLayer.L1, 0.8)
    l3_signal = sig(DetectionLayer.L3, 0.4, error="body too short")
    results = [
        done(DetectionLayer.L1, l1_signal),
        failed(DetectionLayer.L2),
        done(DetectionLayer.L3, l3_signal),
        done(DetectionLayer.L4),
    ]

    assessment = score("<m@e>", results, weights=WEIGHTS)

    assert assessment.signals == [l1_signal, l3_signal]
    assert assessment.message_id == "<m@e>"


def test_summary_is_passed_through():
    assessment = score("<m@e>", all_layers(), weights=WEIGHTS, summary="nothing found")

    assert assessment.summary == "nothing found"
