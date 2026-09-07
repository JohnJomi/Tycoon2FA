"""Unit tests for the provider-neutral seam in layers/ai_text.py.

Offline and total: this module defines a protocol, a frozen result and two
exceptions, and imports no HTTP client. Nothing here can open a socket.
"""

from __future__ import annotations

import math
import pathlib

import pytest

from layers.ai_text import (
    AITextDetector,
    AITextTimeout,
    AITextUnavailable,
    AITextVerdict,
    NullAITextDetector,
)


# --------------------------------------------------------------------------
# AITextVerdict
# --------------------------------------------------------------------------


def test_a_verdict_carries_the_normalized_probability_and_its_provenance():
    verdict = AITextVerdict(0.2534, "walter_writes", model="main_detector - 1", word_count=52)

    assert verdict.ai_generated_probability == pytest.approx(0.2534)
    assert verdict.provider == "walter_writes"
    assert verdict.model == "main_detector - 1"
    assert verdict.word_count == 52


def test_a_verdict_is_frozen():
    """A verdict records what a provider said; it is not edited afterwards."""
    verdict = AITextVerdict(0.5, "p")

    with pytest.raises(Exception):
        verdict.ai_generated_probability = 0.9  # type: ignore[misc]


@pytest.mark.parametrize("value", [-0.01, 1.01, 2.0, math.nan, math.inf])
def test_a_probability_outside_the_unit_interval_is_rejected(value):
    """An un-normalized score is a provider bug, not a low score."""
    with pytest.raises(ValueError):
        AITextVerdict(value, "p")


@pytest.mark.parametrize("value", ["0.5", None, True])
def test_a_non_numeric_probability_is_rejected(value):
    with pytest.raises(TypeError):
        AITextVerdict(value, "p")


@pytest.mark.parametrize("provider", ["", "   "])
def test_a_verdict_must_name_its_provider(provider):
    with pytest.raises(ValueError):
        AITextVerdict(0.5, provider)


def test_a_negative_word_count_is_rejected():
    with pytest.raises(ValueError):
        AITextVerdict(0.5, "p", word_count=-1)


def test_the_seam_module_names_no_vendor():
    """The whole point of the abstraction: swapping providers touches only
    `layers/providers/`, so no vendor may appear in the contract itself."""
    source = (
        pathlib.Path(__file__).resolve().parent.parent / "layers" / "ai_text.py"
    ).read_text().lower()

    for vendor in ("walter", "sapling", "openai", "gptzero"):
        assert vendor not in source, f"{vendor} leaked into the provider-neutral seam"


def test_the_verdict_is_provider_neutral():
    """No vendor name may appear in the contract Layer 3 depends on."""
    fields = set(AITextVerdict.__dataclass_fields__)

    assert fields == {"ai_generated_probability", "provider", "model", "word_count"}
    for vendor in ("walter", "sapling", "gpt2"):
        assert not any(vendor in name for name in fields)


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


def test_a_timeout_is_a_kind_of_unavailable():
    """One except clause catches "no answer"; TimeoutError still catches this."""
    assert issubclass(AITextTimeout, AITextUnavailable)
    assert issubclass(AITextTimeout, TimeoutError)
    assert issubclass(AITextUnavailable, RuntimeError)


# --------------------------------------------------------------------------
# NullAITextDetector
# --------------------------------------------------------------------------


def test_the_null_detector_satisfies_the_protocol():
    assert isinstance(NullAITextDetector(), AITextDetector)


def test_the_null_detector_abstains_rather_than_reporting_human():
    """Returning 0.0 would claim the text was checked and found human-written."""
    with pytest.raises(AITextUnavailable):
        NullAITextDetector().detect("any text at all")


def test_the_null_detector_makes_no_network_request(monkeypatch):
    import httpx

    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("the null detector must not open a connection")

    monkeypatch.setattr(httpx.Client, "send", explode)
    monkeypatch.setattr(httpx.Client, "request", explode)

    with pytest.raises(AITextUnavailable):
        NullAITextDetector().detect("word " * 100)
