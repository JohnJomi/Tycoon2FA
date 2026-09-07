"""Unit tests for the l3.perplexity signal in layers/l3_nlp.py.

No test downloads a checkpoint, imports torch or opens a socket. Each injects
a deterministic arithmetic-only `CausalLanguageModel`, so the mean per-token
NLL is a number the test computes by hand and asserts exactly.
"""

from __future__ import annotations

import math

import pytest

from core.models import DetectionLayer, ParsedEmail, RiskLevel
from layers.l3_nlp import (
    PERPLEXITY_MAX_TOKENS,
    PERPLEXITY_MIN_TOKENS,
    PerplexityUnavailable,
    analyze_perplexity,
    compute_perplexity,
    default_perplexity_model,
    load_perplexity_model,
    reset_default_perplexity_model,
)


class FakeLM:
    """One token per whitespace-separated word, one fixed log-likelihood each.

    Deterministic by construction and independent of any vocabulary, so the
    expected mean NLL is just `-log_likelihood`.
    """

    def __init__(self, log_likelihood: float = -2.0) -> None:
        self.log_likelihood = log_likelihood
        self.calls = 0

    def encode(self, text: str):
        return [1] * len(text.split())

    def token_log_likelihoods(self, token_ids):
        self.calls += 1
        return [self.log_likelihood] * (len(token_ids) - 1)


def body(tokens: int, word: str = "word") -> str:
    return " ".join([word] * tokens)


def email(text: str, *, subject: str = "Quarterly update") -> ParsedEmail:
    return ParsedEmail(
        message_id="<m@example.com>",
        from_addr="sender@example.com",
        subject=subject,
        body_text=text,
    )


def test_mean_nll_is_the_negated_mean_log_likelihood() -> None:
    result = compute_perplexity(FakeLM(-2.0), body(100))

    assert result.mean_nll == pytest.approx(2.0)
    assert result.perplexity == pytest.approx(math.exp(2.0))
    assert result.token_count == 100
    assert result.truncated is False


def test_the_measurement_uses_n_minus_one_predictions() -> None:
    """A model whose likelihoods do not match the shift is rejected, not averaged."""

    class Unshifted(FakeLM):
        def token_log_likelihoods(self, token_ids):
            return [self.log_likelihood] * len(token_ids)

    with pytest.raises(PerplexityUnavailable, match="expected"):
        compute_perplexity(Unshifted(), body(100))


def test_text_is_truncated_to_the_architecture_window() -> None:
    model = FakeLM(-1.5)
    result = compute_perplexity(model, body(PERPLEXITY_MAX_TOKENS + 250))

    assert result.token_count == PERPLEXITY_MAX_TOKENS
    assert result.truncated is True
    assert result.mean_nll == pytest.approx(1.5)


def test_ordinary_text_produces_a_measurement_signal() -> None:
    signal = analyze_perplexity(email(body(120)), model=FakeLM(-3.25))

    assert signal.layer is DetectionLayer.L3
    assert signal.name == "perplexity"
    assert signal.error is None
    assert signal.severity is RiskLevel.LOW
    assert signal.metadata["mean_nll"] == pytest.approx(3.25)
    assert signal.metadata["perplexity"] == pytest.approx(math.exp(3.25))
    assert signal.metadata["token_count"] == 120
    assert signal.metadata["truncated"] is False
    assert "3.250" in signal.evidence


def test_the_signal_is_a_measurement_not_a_risk_score() -> None:
    """ARCHITECTURE.md defines no nats-to-0-1 mapping, so none is invented."""
    low = analyze_perplexity(email(body(120)), model=FakeLM(-0.5))
    high = analyze_perplexity(email(body(120)), model=FakeLM(-9.0))

    assert low.score == 0.0 and high.score == 0.0
    assert low.metadata["fired"] is False and high.metadata["fired"] is False
    assert high.metadata["mean_nll"] > low.metadata["mean_nll"]


@pytest.mark.parametrize("tokens", [0, 1, PERPLEXITY_MIN_TOKENS - 1])
def test_short_and_empty_bodies_abstain(tokens: int) -> None:
    signal = analyze_perplexity(email(body(tokens)), model=FakeLM())

    assert signal.score == 0.0
    assert signal.error == "body too short"
    assert signal.metadata["mean_nll"] is None
    assert signal.metadata["perplexity"] is None
    assert "abstention, not a clean result" in signal.evidence


def test_a_body_at_the_floor_is_measured() -> None:
    signal = analyze_perplexity(email(body(PERPLEXITY_MIN_TOKENS)), model=FakeLM(-2.0))

    assert signal.error is None
    assert signal.metadata["token_count"] == PERPLEXITY_MIN_TOKENS


def test_unavailable_model_abstains_without_fabricating_a_measurement() -> None:
    reset_default_perplexity_model()
    try:
        signal = analyze_perplexity(email(body(120)), model_name="no-such-checkpoint")
    finally:
        reset_default_perplexity_model()

    assert signal.score == 0.0
    assert signal.error
    assert signal.metadata["mean_nll"] is None
    assert signal.metadata["perplexity"] is None
    assert "not a clean result" in signal.evidence


def test_load_raises_rather_than_downloading_a_missing_checkpoint() -> None:
    with pytest.raises(PerplexityUnavailable):
        load_perplexity_model("definitely-not-a-real-checkpoint-name")


def test_inference_failure_abstains_rather_than_propagating() -> None:
    class Broken(FakeLM):
        def token_log_likelihoods(self, token_ids):
            raise RuntimeError("cuda is on fire")

    signal = analyze_perplexity(email(body(120)), model=Broken())

    assert signal.score == 0.0
    assert signal.error is not None and "cuda is on fire" in signal.error
    assert signal.metadata["mean_nll"] is None


def test_non_finite_output_abstains() -> None:
    class Infinite(FakeLM):
        def token_log_likelihoods(self, token_ids):
            return [float("-inf")] * (len(token_ids) - 1)

    signal = analyze_perplexity(email(body(120)), model=Infinite())

    assert signal.error is not None
    assert signal.metadata["perplexity"] is None


def test_repeated_inference_is_deterministic() -> None:
    model = FakeLM(-2.75)
    message = email(body(120))
    results = {analyze_perplexity(message, model=model).metadata["mean_nll"] for _ in range(5)}

    assert len(results) == 1
    assert model.calls == 5


def test_the_model_singleton_is_loaded_once_and_resettable(monkeypatch) -> None:
    loads = []

    def fake_load(name=None):
        loads.append(name)
        return FakeLM(-1.0)

    monkeypatch.setattr("layers.l3_nlp.load_perplexity_model", fake_load)
    reset_default_perplexity_model()
    try:
        first = default_perplexity_model()
        second = default_perplexity_model()
        assert first is second
        assert len(loads) == 1

        reset_default_perplexity_model()
        assert default_perplexity_model() is not first
        assert len(loads) == 2
    finally:
        reset_default_perplexity_model()


def test_a_failed_load_is_not_cached(monkeypatch) -> None:
    """An artifact that appears after a deploy is picked up without a restart."""
    attempts = []

    def flaky(name=None):
        attempts.append(name)
        if len(attempts) == 1:
            raise PerplexityUnavailable("checkpoint missing")
        return FakeLM(-1.0)

    monkeypatch.setattr("layers.l3_nlp.load_perplexity_model", flaky)
    reset_default_perplexity_model()
    try:
        with pytest.raises(PerplexityUnavailable):
            default_perplexity_model()
        assert default_perplexity_model() is not None
        assert len(attempts) == 2
    finally:
        reset_default_perplexity_model()
