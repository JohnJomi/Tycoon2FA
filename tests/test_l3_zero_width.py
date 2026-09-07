"""Unit tests for the l3.zero_width signal in layers/l3_nlp.py.

Deterministic and offline: the signal is a compiled character class over the
subject and text body, so nothing here loads a model or opens a socket.
"""

from __future__ import annotations

import pytest

from core.models import DetectionLayer, ParsedEmail, RiskLevel
from layers.l3_nlp import ZERO_WIDTH_CHARS, ZERO_WIDTH_SCORE, analyze_zero_width


def email(body: str = "", *, subject: str = "Invoice") -> ParsedEmail:
    return ParsedEmail(
        message_id="<m@example.com>",
        from_addr="sender@example.com",
        subject=subject,
        body_text=body,
    )


def test_clean_text_is_a_genuine_negative() -> None:
    signal = analyze_zero_width(email("Please review the attached invoice."))

    assert signal.layer is DetectionLayer.L3
    assert signal.name == "zero_width"
    assert signal.score == 0.0
    assert signal.severity is RiskLevel.LOW
    assert signal.error is None
    assert signal.metadata["fired"] is False
    assert signal.metadata["total"] == 0
    assert signal.evidence


@pytest.mark.parametrize("char", ZERO_WIDTH_CHARS, ids=lambda c: f"U+{ord(c):04X}")
def test_each_supported_codepoint_fires(char: str) -> None:
    signal = analyze_zero_width(email(f"Verify your acc{char}ount now."))

    assert signal.score == ZERO_WIDTH_SCORE
    assert signal.severity is RiskLevel.MEDIUM
    assert signal.error is None
    assert signal.metadata["fired"] is True
    assert signal.metadata["total"] == 1
    assert signal.metadata["counts"] == {f"U+{ord(char):04X}": 1}
    assert f"U+{ord(char):04X}" in signal.evidence


def test_subject_is_scanned_as_well_as_body() -> None:
    signal = analyze_zero_width(email("clean body", subject="Pay​ment due"))

    assert signal.metadata["fired"] is True
    assert signal.metadata["fields"] == {"subject": ["U+200B"]}


def test_multiple_occurrences_are_counted_per_codepoint() -> None:
    body = "a​b​c⁠d"
    signal = analyze_zero_width(email(body, subject="e﻿ f"))

    assert signal.metadata["total"] == 4
    assert signal.metadata["counts"] == {"U+200B": 2, "U+2060": 1, "U+FEFF": 1}
    assert signal.metadata["fields"] == {
        "subject": ["U+FEFF"],
        "body_text": ["U+200B", "U+2060"],
    }
    # One finding, not one per character: the score is a verdict about the
    # message, not a tally.
    assert signal.score == ZERO_WIDTH_SCORE


def test_empty_text_is_a_negative_not_an_exception() -> None:
    signal = analyze_zero_width(email("", subject=""))

    assert signal.score == 0.0
    assert signal.error is None
    assert signal.metadata["total"] == 0


def test_ordinary_non_ascii_text_does_not_fire() -> None:
    body = (
        "Grüße aus München — the naïve café charged €5.\n"
        "Ελληνικά, Русский, 日本語, العربية.\n"
        "Non-breaking space and an em—dash."
    )
    signal = analyze_zero_width(email(body, subject="Rückfrage – Café"))

    assert signal.score == 0.0
    assert signal.metadata["fired"] is False
