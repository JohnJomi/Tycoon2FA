"""Unit tests for the l3.burstiness signal in layers/l3_nlp.py.

No test downloads a checkpoint, imports torch or opens a socket: each injects
a deterministic arithmetic-only `CausalLanguageModel` through the same seam
`l3.perplexity` uses, so both dispersions are numbers the test computes by hand.
"""

from __future__ import annotations

import math
import statistics

import pytest

from core.models import DetectionLayer, ParsedEmail, RiskLevel
from layers.l3_nlp import (
    BURSTINESS_MIN_SENTENCES,
    BURSTINESS_MIN_SENTENCE_TOKENS,
    PerplexityUnavailable,
    analyze_burstiness,
    compute_burstiness,
    reset_default_perplexity_model,
    split_sentences,
)


class FakeLM:
    """One token per word; per-token log-likelihood chosen by sentence length.

    `by_length` lets a test make one sentence harder than another without
    touching anything else, so a length-varying corpus and a
    perplexity-varying corpus can be built independently.
    """

    def __init__(self, log_likelihood: float = -2.0, by_length: dict[int, float] | None = None):
        self.log_likelihood = log_likelihood
        self.by_length = by_length or {}
        self.calls = 0

    def encode(self, text: str):
        return [1] * len(text.split())

    def token_log_likelihoods(self, token_ids):
        self.calls += 1
        value = self.by_length.get(len(token_ids), self.log_likelihood)
        return [value] * (len(token_ids) - 1)


def sentences(*lengths: int) -> str:
    """A body of sentences with the given word counts."""
    return " ".join(" ".join(["word"] * n) + "." for n in lengths)


def email(text: str) -> ParsedEmail:
    return ParsedEmail(
        message_id="<m@example.com>",
        from_addr="sender@example.com",
        subject="Quarterly update",
        body_text=text,
    )


def test_sentence_splitting_is_conservative_and_deterministic() -> None:
    text = 'One sentence here. A second one! And a third? "Quoted." Done.'
    first = split_sentences(text)

    assert first == [
        "One sentence here.",
        "A second one!",
        "And a third?",
        '"Quoted."',
        "Done.",
    ]
    assert split_sentences(text) == first


def test_uniform_sentences_have_zero_dispersion() -> None:
    result = compute_burstiness(FakeLM(-2.0), sentences(10, 10, 10, 10))

    assert result.sentence_count == 4
    assert result.length_stddev == pytest.approx(0.0)
    assert result.perplexity_stddev == pytest.approx(0.0)
    assert result.mean_perplexity == pytest.approx(math.exp(2.0))
    assert result.mean_length == pytest.approx(10.0)


def test_varying_sentence_length_moves_only_the_length_dispersion() -> None:
    """Same difficulty per token, different lengths: lengths vary, perplexity does not."""
    lengths = (6, 12, 24, 30)
    result = compute_burstiness(FakeLM(-2.0), sentences(*lengths))

    assert result.sentence_lengths == lengths
    assert result.length_stddev == pytest.approx(statistics.stdev(lengths))
    assert result.length_stddev > 0
    assert result.perplexity_stddev == pytest.approx(0.0)


def test_varying_sentence_perplexity_is_reported_independently() -> None:
    """Difficulty varies far more than length, and each is measured on its own."""
    result = compute_burstiness(
        FakeLM(by_length={10: -1.0, 11: -5.0, 12: -3.0}), sentences(10, 11, 12)
    )

    expected = [math.exp(1.0), math.exp(5.0), math.exp(3.0)]
    assert result.sentence_perplexities == pytest.approx(tuple(expected))
    assert result.perplexity_stddev == pytest.approx(statistics.stdev(expected))
    assert result.length_stddev == pytest.approx(statistics.stdev([10, 11, 12]))
    assert result.perplexity_stddev > result.length_stddev


def test_equal_difficulty_gives_zero_perplexity_dispersion() -> None:
    result = compute_burstiness(FakeLM(-1.0), sentences(10, 10, 10))

    assert result.perplexity_stddev == pytest.approx(0.0)


def test_signal_reports_both_measurements_separately() -> None:
    signal = analyze_burstiness(email(sentences(6, 12, 24, 30)), model=FakeLM(-2.0))

    assert signal.layer is DetectionLayer.L3
    assert signal.name == "burstiness"
    assert signal.error is None
    assert signal.severity is RiskLevel.LOW
    assert signal.metadata["length_stddev"] == pytest.approx(statistics.stdev([6, 12, 24, 30]))
    assert signal.metadata["perplexity_stddev"] == pytest.approx(0.0)
    assert signal.metadata["sentence_count"] == 4
    assert signal.metadata["sentence_lengths"] == [6, 12, 24, 30]
    assert signal.evidence


def test_the_signal_is_a_measurement_not_a_risk_score() -> None:
    even = analyze_burstiness(email(sentences(10, 10, 10, 10)), model=FakeLM(-2.0))
    uneven = analyze_burstiness(email(sentences(4, 20, 45, 8)), model=FakeLM(-2.0))

    assert even.score == 0.0 and uneven.score == 0.0
    assert even.metadata["fired"] is False and uneven.metadata["fired"] is False
    assert uneven.metadata["length_stddev"] > even.metadata["length_stddev"]


def test_too_few_sentences_abstains() -> None:
    signal = analyze_burstiness(email(sentences(20, 20)), model=FakeLM(-2.0))

    assert signal.score == 0.0
    assert signal.error is not None and "sentence" in signal.error
    assert signal.metadata["perplexity_stddev"] is None
    assert signal.metadata["length_stddev"] is None
    assert "abstention, not a clean result" in signal.evidence


def test_fragments_below_the_token_floor_do_not_count_as_sentences() -> None:
    short = BURSTINESS_MIN_SENTENCE_TOKENS - 1
    signal = analyze_burstiness(email(sentences(short, short, short, short)), model=FakeLM(-2.0))

    assert signal.error is not None
    assert signal.metadata["sentence_count"] is None


@pytest.mark.parametrize("text", ["", "   \n  ", "Hello."])
def test_empty_or_short_body_abstains(text: str) -> None:
    signal = analyze_burstiness(email(text), model=FakeLM(-2.0))

    assert signal.score == 0.0
    assert signal.error is not None
    assert signal.metadata["length_stddev"] is None


def test_a_body_at_the_sentence_floor_is_measured() -> None:
    signal = analyze_burstiness(email(sentences(8, 12, 16)), model=FakeLM(-2.0))

    assert signal.error is None
    assert signal.metadata["sentence_count"] == BURSTINESS_MIN_SENTENCES


def test_model_unavailable_abstains_without_fabricating_a_dispersion() -> None:
    reset_default_perplexity_model()
    try:
        signal = analyze_burstiness(
            email(sentences(10, 20, 30)), model_name="no-such-checkpoint"
        )
    finally:
        reset_default_perplexity_model()

    assert signal.score == 0.0
    assert signal.error
    assert signal.metadata["perplexity_stddev"] is None
    assert signal.metadata["length_stddev"] is None
    assert "not a clean result" in signal.evidence


def test_inference_failure_abstains_rather_than_propagating() -> None:
    class Broken(FakeLM):
        def token_log_likelihoods(self, token_ids):
            raise RuntimeError("cuda is on fire")

    signal = analyze_burstiness(email(sentences(10, 20, 30)), model=Broken())

    assert signal.score == 0.0
    assert signal.error is not None and "cuda is on fire" in signal.error
    assert signal.metadata["perplexity_stddev"] is None


def test_inference_failure_does_not_look_like_too_few_sentences() -> None:
    class Broken(FakeLM):
        def token_log_likelihoods(self, token_ids):
            raise RuntimeError("boom")

    signal = analyze_burstiness(email(sentences(10, 20, 30)), model=Broken())

    assert "absence of information" in signal.evidence


def test_repeated_analysis_is_deterministic() -> None:
    model = FakeLM(by_length={10: -1.0, 20: -4.0, 30: -2.5})
    message = email(sentences(10, 20, 30))
    seen = {
        (
            analyze_burstiness(message, model=model).metadata["perplexity_stddev"],
            analyze_burstiness(message, model=model).metadata["length_stddev"],
        )
        for _ in range(5)
    }

    assert len(seen) == 1


def test_compute_raises_rather_than_returning_a_partial_result() -> None:
    with pytest.raises(PerplexityUnavailable):
        compute_burstiness(FakeLM(-2.0), sentences(30, 30))
