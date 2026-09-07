"""Unit tests for the l3.urgency signal in layers/l3_nlp.py.

Offline and deterministic. No test loads the production artifact: each fits a
small pipeline from `training.train_urgency.build_pipeline` over a fixed
in-file corpus and injects it, so the suite exercises the real TF-IDF ->
LogisticRegression shape without depending on a trained model existing.
"""

from __future__ import annotations

import pytest

from core.models import DetectionLayer, ParsedEmail, RiskLevel
from layers import l3_nlp
from layers.l3_nlp import (
    URGENCY_MIN_CHARS,
    URGENCY_WARN_THRESHOLD,
    UrgencyUnavailable,
    analyze_urgency,
    load_urgency_model,
    reset_default_urgency_model,
)
from training.train_urgency import PHISH_LABEL, build_pipeline, save_model

HAM = [
    "Hi Sam, attached are the notes from Tuesday's planning meeting.",
    "The quarterly report is ready whenever you have a moment to review it.",
    "Thanks for lunch yesterday - let me know if Thursday works for a follow up.",
    "Please find the updated project timeline attached for your reference.",
    "Reminder that the office will be closed on Monday for the public holiday.",
    "Could you send over the vendor contact details when you get a chance?",
]

PHISH = [
    "URGENT: your account will be suspended within 24 hours, verify immediately.",
    "Immediate action required - confirm your password now or lose access today.",
    "Final warning: your mailbox will be deactivated unless you verify right now.",
    "Act now! Your account has been locked, click here immediately to restore it.",
    "Security alert: verify your credentials immediately or your account is closed.",
    "Your payment failed - update your billing details urgently to avoid suspension.",
]


@pytest.fixture(scope="module")
def model():
    pipeline = build_pipeline()
    pipeline.fit(HAM + PHISH, [0] * len(HAM) + [PHISH_LABEL] * len(PHISH))
    return pipeline


def email(body: str, *, subject: str = "") -> ParsedEmail:
    return ParsedEmail(
        message_id="<m@example.com>",
        from_addr="sender@example.com",
        subject=subject,
        body_text=body,
    )


def test_ordinary_business_text_scores_low(model) -> None:
    signal = analyze_urgency(
        email(
            "Hi Alex, here are the minutes from the planning meeting on Tuesday. "
            "No rush - have a look whenever you get a chance this week.",
            subject="Meeting notes",
        ),
        model=model,
    )

    assert signal.layer is DetectionLayer.L3
    assert signal.name == "urgency"
    assert signal.error is None
    assert signal.score < URGENCY_WARN_THRESHOLD
    assert signal.severity is RiskLevel.LOW
    assert signal.metadata["probability"] == signal.score
    assert signal.evidence


def test_phishing_style_urgency_scores_high(model) -> None:
    signal = analyze_urgency(
        email(
            "URGENT: your account will be suspended within 24 hours. "
            "Verify your password immediately or you will lose access today.",
            subject="Immediate action required",
        ),
        model=model,
    )

    assert signal.error is None
    assert signal.score > URGENCY_WARN_THRESHOLD
    assert signal.severity in (RiskLevel.MEDIUM, RiskLevel.HIGH)
    assert signal.metadata["fired"] is True


def test_urgent_text_outscores_calm_text(model) -> None:
    """The ordering is the claim; the absolute numbers move when refit."""
    calm = analyze_urgency(email(HAM[0] + " " + HAM[1]), model=model)
    urgent = analyze_urgency(email(PHISH[0] + " " + PHISH[1]), model=model)

    assert urgent.score > calm.score


def test_mixed_text_lands_between_the_two(model) -> None:
    mixed = analyze_urgency(
        email(
            "Hi Sam, attached are the notes from Tuesday's planning meeting. "
            "Separately, please verify your account immediately to avoid suspension.",
            subject="Meeting notes and account",
        ),
        model=model,
    )
    calm = analyze_urgency(email(HAM[0] + " " + HAM[1]), model=model)
    urgent = analyze_urgency(email(PHISH[0] + " " + PHISH[1]), model=model)

    assert calm.score < mixed.score < urgent.score
    assert mixed.error is None


@pytest.mark.parametrize("body, subject", [("", ""), ("  \n ", ""), ("Thanks!", "Hi")])
def test_short_text_abstains_rather_than_scoring(model, body: str, subject: str) -> None:
    signal = analyze_urgency(email(body, subject=subject), model=model)

    assert signal.score == 0.0
    assert signal.error == "text too short to classify"
    assert signal.metadata["fired"] is False
    assert signal.metadata["probability"] is None
    assert signal.metadata["chars"] < URGENCY_MIN_CHARS


def test_missing_artifact_abstains_and_does_not_fabricate_a_score(tmp_path) -> None:
    reset_default_urgency_model()
    try:
        signal = analyze_urgency(
            email("Please verify your account immediately or it will be suspended."),
            model_path=tmp_path / "absent.joblib",
        )
    finally:
        reset_default_urgency_model()

    assert signal.score == 0.0
    assert signal.error is not None
    assert "absent.joblib" in signal.error
    assert signal.metadata["fired"] is False
    assert signal.metadata["probability"] is None
    # An outage must not read as an all-clear.
    assert "not a clean result" in signal.evidence


def test_unreadable_artifact_raises_unavailable(tmp_path) -> None:
    corrupt = tmp_path / "urgency_clf.joblib"
    corrupt.write_bytes(b"not a joblib file")

    with pytest.raises(UrgencyUnavailable):
        load_urgency_model(corrupt)


def test_artifact_that_is_not_a_classifier_is_rejected(tmp_path) -> None:
    import joblib

    path = tmp_path / "wrong.joblib"
    joblib.dump({"not": "a model"}, path)

    with pytest.raises(UrgencyUnavailable):
        load_urgency_model(path)


def test_a_failing_model_abstains_rather_than_propagating(model) -> None:
    class Broken:
        classes_ = [0, PHISH_LABEL]

        def predict_proba(self, texts):
            raise RuntimeError("boom")

    signal = analyze_urgency(email(PHISH[0] + " " + PHISH[1]), model=Broken())

    assert signal.score == 0.0
    assert signal.error is not None
    assert "boom" in signal.error


def test_round_trips_through_a_persisted_artifact(tmp_path, model) -> None:
    path = save_model(model, tmp_path / "models" / "urgency_clf.joblib")
    loaded = load_urgency_model(path)

    body = PHISH[0] + " " + PHISH[1]
    assert analyze_urgency(email(body), model=loaded).score == pytest.approx(
        analyze_urgency(email(body), model=model).score
    )


def test_inference_is_deterministic(model) -> None:
    message = email(
        "Immediate action required: confirm your password now or lose access.",
        subject="Security alert",
    )
    scores = {analyze_urgency(message, model=model).score for _ in range(5)}

    assert len(scores) == 1


def test_module_does_not_import_sklearn() -> None:
    """The layer talks to a protocol; the model's shape is a training concern."""
    source = (l3_nlp.__file__ or "")
    assert source
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    assert "sklearn" not in text
