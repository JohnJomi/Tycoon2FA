"""Unit tests for the l3.fusion signal in layers/l3_nlp.py.

Offline and deterministic. Fusion models are either fitted in-test from
`training.train_fusion.build_model` over a fixed table, or hand-written fakes;
no test loads a production artifact, downloads anything, or opens a socket.
"""

from __future__ import annotations

import pytest

from core.models import DetectionLayer, DetectionSignal, ParsedEmail, RiskLevel
from layers.l3_nlp import (
    FUSION_FEATURE_ORDER,
    FUSION_MODEL_PATH,
    URGENCY_MODEL_PATH,
    FusionUnavailable,
    analyze_fusion,
    default_fusion_model,
    extract_fusion_features,
    load_fusion_model,
    reset_default_fusion_model,
    verdict_thresholds,
)
from training.train_fusion import PHISH_LABEL, build_model, save_model, train


def signal(name: str, metadata: dict, *, error: str | None = None) -> DetectionSignal:
    return DetectionSignal(
        layer=DetectionLayer.L3,
        name=name,
        score=0.0,
        severity=RiskLevel.LOW,
        evidence=f"{name} evidence",
        metadata={**metadata, "fired": False},
        error=error,
    )


def features(
    *,
    zero_width: int | None = 0,
    urgency: float | None = 0.1,
    perplexity: float | None = 4.0,
    burstiness: float | None = 2.0,
    abstain: str | None = None,
) -> list[DetectionSignal]:
    """The four upstream signals, shaped as the analyzers actually emit them."""
    return [
        signal(
            "zero_width",
            {"total": zero_width},
            error=("abstained" if abstain == "zero_width" else None),
        ),
        signal(
            "urgency",
            {"probability": urgency},
            error=("classifier unavailable" if abstain == "urgency" else None),
        ),
        signal(
            "perplexity",
            {"mean_nll": perplexity},
            error=("body too short" if abstain == "perplexity" else None),
        ),
        signal(
            "burstiness",
            {"perplexity_stddev": burstiness, "length_stddev": 5.0},
            error=("too few sentences" if abstain == "burstiness" else None),
        ),
    ]


class FakeFusion:
    """Probability rises with the urgency column. `classes_` order is a parameter."""

    def __init__(self, classes=(0, PHISH_LABEL)) -> None:
        self.classes_ = list(classes)
        self.calls = 0

    def predict_proba(self, rows):
        self.calls += 1
        out = []
        for row in rows:
            phish = min(1.0, max(0.0, float(row[1])))
            pair = {PHISH_LABEL: phish, 0: 1.0 - phish}
            out.append([pair[label] for label in self.classes_])
        return out


@pytest.fixture(scope="module")
def fitted():
    rows = [
        [0, 0.05, 3.0, 1.0],
        [0, 0.10, 3.5, 1.5],
        [0, 0.08, 3.2, 1.2],
        [3, 0.90, 8.0, 6.0],
        [5, 0.95, 8.5, 6.5],
        [4, 0.88, 7.8, 6.2],
    ]
    return train(rows, [0, 0, 0, PHISH_LABEL, PHISH_LABEL, PHISH_LABEL])


# --- feature extraction ---------------------------------------------------


def test_features_are_read_in_the_declared_order() -> None:
    row = extract_fusion_features(
        features(zero_width=2, urgency=0.7, perplexity=5.5, burstiness=3.25)
    )

    assert FUSION_FEATURE_ORDER == ("zero_width", "urgency", "perplexity", "burstiness")
    assert row == [2.0, 0.7, 5.5, 3.25]


def test_features_come_from_the_upstream_signals_not_recomputation() -> None:
    """Fusion never touches a language model: arbitrary metadata flows straight through."""
    row = extract_fusion_features(features(perplexity=99.0))

    assert row[2] == 99.0


@pytest.mark.parametrize("which", FUSION_FEATURE_ORDER)
def test_an_abstaining_upstream_signal_is_not_treated_as_zero(which: str) -> None:
    with pytest.raises(FusionUnavailable, match=which):
        extract_fusion_features(features(abstain=which))


def test_a_missing_upstream_signal_abstains() -> None:
    partial = [s for s in features() if s.name != "burstiness"]

    with pytest.raises(FusionUnavailable, match="burstiness"):
        extract_fusion_features(partial)


def test_a_none_measurement_abstains() -> None:
    with pytest.raises(FusionUnavailable, match="perplexity"):
        extract_fusion_features(features(perplexity=None))


def test_a_non_finite_measurement_abstains() -> None:
    with pytest.raises(FusionUnavailable, match="non-finite"):
        extract_fusion_features(features(burstiness=float("nan")))


# --- scoring --------------------------------------------------------------


def test_valid_fusion_over_all_four_features(fitted) -> None:
    result = analyze_fusion(
        features(zero_width=4, urgency=0.92, perplexity=8.2, burstiness=6.3), model=fitted
    )

    assert result.layer is DetectionLayer.L3
    assert result.name == "fusion"
    assert result.error is None
    assert 0.0 <= result.score <= 1.0
    assert result.score > 0.5
    assert result.metadata["fired"] is True
    assert result.metadata["probability"] == result.score
    assert result.metadata["features"] == {
        "zero_width": 4.0,
        "urgency": 0.92,
        "perplexity": 8.2,
        "burstiness": 6.3,
    }
    assert result.evidence


def test_clean_features_score_below_phishing_features(fitted) -> None:
    calm = analyze_fusion(features(zero_width=0, urgency=0.05, perplexity=3.1, burstiness=1.1), model=fitted)
    nasty = analyze_fusion(features(zero_width=5, urgency=0.95, perplexity=8.4, burstiness=6.4), model=fitted)

    assert calm.score < nasty.score
    assert calm.error is None and nasty.error is None


def test_positive_class_is_located_by_label_not_column() -> None:
    normal = analyze_fusion(features(urgency=0.9), model=FakeFusion((0, PHISH_LABEL)))
    reversed_ = analyze_fusion(features(urgency=0.9), model=FakeFusion((PHISH_LABEL, 0)))

    assert normal.score == pytest.approx(0.9)
    assert reversed_.score == pytest.approx(0.9)


def test_a_model_without_the_positive_class_abstains() -> None:
    result = analyze_fusion(features(), model=FakeFusion((0, 2)))

    assert result.score == 0.0
    assert result.error is not None and "class" in result.error


# --- severity -------------------------------------------------------------


def test_severity_uses_the_configured_thresholds() -> None:
    warn, block = verdict_thresholds()
    assert (warn, block) == (0.35, 0.65)

    below = analyze_fusion(features(urgency=warn - 0.01), model=FakeFusion())
    at_warn = analyze_fusion(features(urgency=warn), model=FakeFusion())
    at_block = analyze_fusion(features(urgency=block), model=FakeFusion())

    assert below.severity is RiskLevel.LOW
    assert at_warn.severity is RiskLevel.MEDIUM
    assert at_block.severity is RiskLevel.HIGH


# --- abstention -----------------------------------------------------------


@pytest.mark.parametrize("which", FUSION_FEATURE_ORDER)
def test_upstream_abstention_abstains_the_fusion(which: str, fitted) -> None:
    result = analyze_fusion(features(abstain=which), model=fitted)

    assert result.score == 0.0
    assert result.metadata["fired"] is False
    assert result.error is not None and which in result.error
    assert result.metadata["probability"] is None
    assert "absence of information, not a clean result" in result.evidence


def test_missing_artifact_abstains(tmp_path) -> None:
    reset_default_fusion_model()
    try:
        result = analyze_fusion(features(), model_path=tmp_path / "absent.joblib")
    finally:
        reset_default_fusion_model()

    assert result.score == 0.0
    assert result.error is not None and "absent.joblib" in result.error
    assert "not a clean result" in result.evidence


def test_corrupt_artifact_is_rejected(tmp_path) -> None:
    corrupt = tmp_path / "fusion_clf.joblib"
    corrupt.write_bytes(b"not a joblib file")

    with pytest.raises(FusionUnavailable):
        load_fusion_model(corrupt)


def test_an_artifact_that_is_not_a_classifier_is_rejected(tmp_path) -> None:
    import joblib

    path = tmp_path / "wrong.joblib"
    joblib.dump([1, 2, 3], path)

    with pytest.raises(FusionUnavailable):
        load_fusion_model(path)


def test_inference_failure_abstains_rather_than_propagating() -> None:
    class Broken:
        classes_ = [0, PHISH_LABEL]

        def predict_proba(self, rows):
            raise RuntimeError("solver exploded")

    result = analyze_fusion(features(), model=Broken())

    assert result.score == 0.0
    assert result.error is not None and "solver exploded" in result.error
    assert result.metadata["probability"] is None


# --- artifacts and determinism -------------------------------------------


def test_the_fusion_artifact_is_not_the_urgency_artifact() -> None:
    assert FUSION_MODEL_PATH != URGENCY_MODEL_PATH
    assert FUSION_MODEL_PATH.name == "fusion_clf.joblib"


def test_round_trips_through_a_persisted_artifact(tmp_path, fitted) -> None:
    path = save_model(fitted, tmp_path / "models" / "fusion_clf.joblib")
    loaded = load_fusion_model(path)
    inputs = features(zero_width=4, urgency=0.9, perplexity=8.0, burstiness=6.0)

    assert analyze_fusion(inputs, model=loaded).score == pytest.approx(
        analyze_fusion(inputs, model=fitted).score
    )
    assert (tmp_path / "models" / "urgency_clf.joblib").exists() is False


def test_repeated_inference_is_deterministic(fitted) -> None:
    inputs = features(zero_width=2, urgency=0.6, perplexity=6.0, burstiness=4.0)
    scores = {analyze_fusion(inputs, model=fitted).score for _ in range(5)}

    assert len(scores) == 1


def test_the_singleton_is_loaded_once_and_a_failure_is_not_cached(monkeypatch) -> None:
    attempts = []

    def flaky(path=None):
        attempts.append(path)
        if len(attempts) == 1:
            raise FusionUnavailable("artifact missing")
        return FakeFusion()

    monkeypatch.setattr("layers.l3_nlp.load_fusion_model", flaky)
    reset_default_fusion_model()
    try:
        with pytest.raises(FusionUnavailable):
            default_fusion_model()
        first = default_fusion_model()
        assert default_fusion_model() is first
        assert len(attempts) == 2
    finally:
        reset_default_fusion_model()


def test_analysis_opens_no_socket_and_loads_no_model(monkeypatch, fitted) -> None:
    """Scoring with an injected model must not reach the network or the disk."""
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("fusion attempted network access")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(
        "layers.l3_nlp.load_fusion_model",
        lambda path=None: (_ for _ in ()).throw(AssertionError("fusion loaded a model")),
    )

    result = analyze_fusion(features(urgency=0.8), model=fitted)

    assert result.error is None


def test_build_model_is_a_logistic_regression() -> None:
    from sklearn.linear_model import LogisticRegression

    assert isinstance(build_model(), LogisticRegression)


def test_a_fitted_model_exposes_coefficients_per_feature(fitted) -> None:
    """The coef_ values are a deliverable, one per named feature."""
    assert fitted.coef_.shape == (1, len(FUSION_FEATURE_ORDER))
