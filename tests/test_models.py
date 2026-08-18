"""Unit tests for the core data contracts in core/models.py.

Scope is deliberately narrow: construction, defaults and validation of the
dataclasses. There is no detection logic to test yet.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from core.models import (
    Attachment,
    LayerResult,
    DetectionLayer,
    DetectionSignal,
    ExtractedURL,
    ParsedEmail,
    RiskAssessment,
    RiskLevel,
    URLSource,
)


# --------------------------------------------------------------------------
# 1. Risk level enum
# --------------------------------------------------------------------------


def test_risk_level_has_exactly_three_members():
    assert [level.name for level in RiskLevel] == ["LOW", "MEDIUM", "HIGH"]


def test_risk_level_values_are_lowercase_strings():
    assert RiskLevel.LOW.value == "low"
    assert RiskLevel.MEDIUM.value == "medium"
    assert RiskLevel.HIGH.value == "high"


def test_risk_level_is_constructible_from_its_value():
    assert RiskLevel("high") is RiskLevel.HIGH


def test_unknown_risk_level_value_is_rejected():
    with pytest.raises(ValueError):
        RiskLevel("critical")


# --------------------------------------------------------------------------
# 2. Detection layer enum
# --------------------------------------------------------------------------


def test_detection_layer_has_exactly_four_members():
    assert [layer.name for layer in DetectionLayer] == ["L1", "L2", "L3", "L4"]


def test_detection_layer_values_are_one_through_four():
    assert [int(layer) for layer in DetectionLayer] == [1, 2, 3, 4]


def test_detection_layer_is_constructible_from_its_number():
    assert DetectionLayer(3) is DetectionLayer.L3


def test_unknown_detection_layer_value_is_rejected():
    with pytest.raises(ValueError):
        DetectionLayer(5)


# --------------------------------------------------------------------------
# 3. ParsedEmail
# --------------------------------------------------------------------------


def test_parsed_email_minimal_construction_applies_defaults():
    email = ParsedEmail(message_id="<abc@example.com>", from_addr="sender@example.com")

    assert email.message_id == "<abc@example.com>"
    assert email.from_addr == "sender@example.com"
    assert email.from_display == ""
    assert email.subject == ""
    assert email.body_text == ""
    assert email.body_html is None
    assert email.reply_to is None
    assert email.raw is None
    assert email.to_addrs == []
    assert email.urls == []
    assert email.headers == {}
    assert email.received_chain == []
    assert email.attachments == []


def test_parsed_email_full_construction_round_trips_fields():
    url = ExtractedURL(url="https://example.com/login", source=URLSource.ANCHOR_HREF)
    attachment = Attachment(filename="invoice.pdf", content_type="application/pdf", size_bytes=1024)

    email = ParsedEmail(
        message_id="<full@example.com>",
        from_addr="billing@example.com",
        from_display="Billing Team",
        to_addrs=["a@example.org", "b@example.org"],
        subject="Invoice attached",
        body_text="See attached.",
        body_html="<p>See attached.</p>",
        reply_to="not-billing@elsewhere.test",
        urls=[url],
        headers={"authentication-results": ["spf=pass"]},
        received_chain=["mx1.example.com", "mx2.example.com"],
        attachments=[attachment],
        raw=b"From: billing@example.com\r\n",
    )

    assert email.from_display == "Billing Team"
    assert email.to_addrs == ["a@example.org", "b@example.org"]
    assert email.reply_to == "not-billing@elsewhere.test"
    assert email.urls == [url]
    assert email.headers["authentication-results"] == ["spf=pass"]
    assert email.received_chain[0] == "mx1.example.com"
    assert email.attachments == [attachment]
    assert email.raw == b"From: billing@example.com\r\n"


def test_parsed_email_mutable_defaults_are_not_shared_between_instances():
    first = ParsedEmail(message_id="<1@example.com>", from_addr="a@example.com")
    second = ParsedEmail(message_id="<2@example.com>", from_addr="b@example.com")

    first.to_addrs.append("victim@example.org")

    assert second.to_addrs == []


def test_parsed_email_inline_images_selects_only_cid_referenced_parts():
    inline = Attachment(filename="logo.png", content_type="image/png", content_id="logo123")
    regular = Attachment(filename="invoice.pdf", content_type="application/pdf")

    email = ParsedEmail(
        message_id="<mixed@example.com>",
        from_addr="a@example.com",
        attachments=[inline, regular],
    )

    assert email.inline_images == [inline]


@pytest.mark.parametrize("bad_value", ["", "   "])
def test_parsed_email_rejects_blank_message_id(bad_value):
    with pytest.raises(ValueError, match="message_id"):
        ParsedEmail(message_id=bad_value, from_addr="a@example.com")


def test_parsed_email_allows_an_empty_sender():
    """A message with no recoverable From is still representable.

    The absent sender is evidence for Layer 1, so ingest must be able to hand
    the message on rather than rejecting it.
    """
    email = ParsedEmail(message_id="<x@example.com>", from_addr="")

    assert email.from_addr == ""
    assert email.from_display == ""


def test_parsed_email_allows_an_empty_display_name_with_a_real_address():
    email = ParsedEmail(message_id="<x@example.com>", from_addr="a@example.com", from_display="")

    assert email.from_addr == "a@example.com"
    assert email.from_display == ""


def test_parsed_email_with_empty_sender_keeps_its_other_fields():
    email = ParsedEmail(
        message_id="<x@example.com>",
        from_addr="",
        subject="Still parsed",
        to_addrs=["victim@example.org"],
    )

    assert email.subject == "Still parsed"
    assert email.to_addrs == ["victim@example.org"]


def test_parsed_email_still_requires_a_message_id():
    with pytest.raises(ValueError, match="message_id"):
        ParsedEmail(message_id="", from_addr="a@example.com")


# --------------------------------------------------------------------------
# 4. ExtractedURL
# --------------------------------------------------------------------------


def test_extracted_url_minimal_construction_leaves_layer2_fields_unresolved():
    url = ExtractedURL(url="https://example.com/a", source=URLSource.PLAIN_TEXT)

    assert url.anchor_text is None
    assert url.redirect_chain == []
    assert url.final_url is None
    assert url.redirect_depth == 0


def test_extracted_url_records_a_resolved_redirect_chain():
    url = ExtractedURL(
        url="https://short.test/abc",
        source=URLSource.ANCHOR_HREF,
        anchor_text="View invoice",
        redirect_chain=["https://hop1.test", "https://hop2.test"],
        final_url="https://final.test/login",
    )

    assert url.anchor_text == "View invoice"
    assert url.redirect_depth == 2
    assert url.final_url == "https://final.test/login"


def test_extracted_url_accepts_every_defined_source():
    for source in URLSource:
        assert ExtractedURL(url="https://example.com", source=source).source is source


def test_extracted_url_rejects_blank_url():
    with pytest.raises(ValueError, match="url"):
        ExtractedURL(url="", source=URLSource.PLAIN_TEXT)


def test_extracted_url_rejects_non_enum_source():
    with pytest.raises(TypeError, match="source"):
        ExtractedURL(url="https://example.com", source="anchor_href")


# --------------------------------------------------------------------------
# 5. DetectionSignal
# --------------------------------------------------------------------------


def test_detection_signal_minimal_construction_applies_defaults():
    signal = DetectionSignal(
        layer=DetectionLayer.L1,
        name="domain_age",
        score=0.8,
        severity=RiskLevel.HIGH,
        evidence="Sending domain registered 2 days ago.",
    )

    assert signal.metadata == {}
    assert signal.error is None
    assert signal.qualified_name == "L1/domain_age"


@pytest.mark.parametrize(
    ("layer", "name"),
    [
        (DetectionLayer.L1, "domain_age"),
        (DetectionLayer.L2, "redirect_depth"),
        (DetectionLayer.L2, "base64_email_parameter"),
        (DetectionLayer.L3, "zero_width_character"),
        (DetectionLayer.L3, "urgency_score"),
        (DetectionLayer.L4, "threat_intelligence_match"),
    ],
)
def test_one_generic_signal_model_represents_any_layer_and_name(layer, name):
    signal = DetectionSignal(
        layer=layer,
        name=name,
        score=0.5,
        severity=RiskLevel.MEDIUM,
        evidence="example evidence",
    )

    assert signal.layer is layer
    assert signal.name == name
    assert signal.qualified_name == f"{layer.name}/{name}"


def test_detection_signal_carries_structured_metadata():
    signal = DetectionSignal(
        layer=DetectionLayer.L2,
        name="redirect_depth",
        score=0.6,
        severity=RiskLevel.MEDIUM,
        evidence="Followed 4 hops before reaching the landing page.",
        metadata={"depth": 4, "final_url": "https://final.test"},
    )

    assert signal.metadata["depth"] == 4


def test_detection_signal_can_abstain_by_recording_an_error():
    signal = DetectionSignal(
        layer=DetectionLayer.L3,
        name="perplexity",
        score=0.0,
        severity=RiskLevel.LOW,
        evidence="Perplexity not computed.",
        error="body too short",
    )

    assert signal.error == "body too short"


def test_detection_signal_is_frozen():
    signal = DetectionSignal(
        layer=DetectionLayer.L1,
        name="domain_age",
        score=0.2,
        severity=RiskLevel.LOW,
        evidence="Sending domain registered 6 years ago.",
    )

    with pytest.raises(FrozenInstanceError):
        signal.score = 0.9


@pytest.mark.parametrize("score", [0.0, 0.5, 1.0])
def test_detection_signal_accepts_scores_across_the_unit_interval(score):
    signal = DetectionSignal(
        layer=DetectionLayer.L4,
        name="threat_intelligence_match",
        score=score,
        severity=RiskLevel.LOW,
        evidence="example evidence",
    )

    assert signal.score == score


@pytest.mark.parametrize("bad_score", [-0.1, 1.1, float("nan"), float("inf")])
def test_detection_signal_rejects_scores_outside_the_unit_interval(bad_score):
    with pytest.raises(ValueError, match="score"):
        DetectionSignal(
            layer=DetectionLayer.L1,
            name="domain_age",
            score=bad_score,
            severity=RiskLevel.LOW,
            evidence="example evidence",
        )


def test_detection_signal_rejects_non_numeric_score():
    with pytest.raises(TypeError, match="score"):
        DetectionSignal(
            layer=DetectionLayer.L1,
            name="domain_age",
            score="high",
            severity=RiskLevel.LOW,
            evidence="example evidence",
        )


def test_detection_signal_requires_evidence():
    with pytest.raises(ValueError, match="evidence"):
        DetectionSignal(
            layer=DetectionLayer.L1,
            name="domain_age",
            score=0.5,
            severity=RiskLevel.MEDIUM,
            evidence="",
        )


def test_detection_signal_requires_a_name():
    with pytest.raises(ValueError, match="name"):
        DetectionSignal(
            layer=DetectionLayer.L1,
            name="",
            score=0.5,
            severity=RiskLevel.MEDIUM,
            evidence="example evidence",
        )


def test_detection_signal_rejects_non_enum_layer():
    with pytest.raises(TypeError, match="layer"):
        DetectionSignal(
            layer=1,
            name="domain_age",
            score=0.5,
            severity=RiskLevel.MEDIUM,
            evidence="example evidence",
        )


def test_detection_signal_rejects_non_enum_severity():
    with pytest.raises(TypeError, match="severity"):
        DetectionSignal(
            layer=DetectionLayer.L1,
            name="domain_age",
            score=0.5,
            severity="high",
            evidence="example evidence",
        )


# --------------------------------------------------------------------------
# LayerResult invariants
# --------------------------------------------------------------------------


def _layer_signal():
    return DetectionSignal(
        layer=DetectionLayer.L1,
        name="example",
        score=0.5,
        severity=RiskLevel.MEDIUM,
        evidence="example evidence",
    )


def test_layer_result_minimal_construction():
    result = LayerResult(layer=DetectionLayer.L2, completed=True)

    assert result.signals == []
    assert result.error is None
    assert result.duration_ms == 0


def test_completed_layer_may_carry_signals():
    result = LayerResult(layer=DetectionLayer.L1, completed=True, signals=[_layer_signal()])

    assert len(result.signals) == 1


def test_completed_layer_must_not_carry_an_error():
    with pytest.raises(ValueError, match="completed layer must not carry an error"):
        LayerResult(layer=DetectionLayer.L1, completed=True, error="timeout")


def test_incomplete_layer_must_state_a_reason():
    with pytest.raises(ValueError, match="did not complete"):
        LayerResult(layer=DetectionLayer.L1, completed=False)


@pytest.mark.parametrize("blank", ["", "   "])
def test_incomplete_layer_rejects_a_blank_reason(blank):
    with pytest.raises(ValueError, match="did not complete"):
        LayerResult(layer=DetectionLayer.L1, completed=False, error=blank)


def test_incomplete_layer_must_not_carry_signals():
    """An abstention that also reports findings would be a contradiction."""
    with pytest.raises(ValueError, match="must not carry signals"):
        LayerResult(
            layer=DetectionLayer.L1,
            completed=False,
            signals=[_layer_signal()],
            error="timeout after 8s",
        )


def test_layer_result_rejects_a_non_enum_layer():
    with pytest.raises(TypeError, match="layer"):
        LayerResult(layer=1, completed=True)


def test_layer_result_rejects_a_non_bool_completed():
    with pytest.raises(TypeError, match="completed"):
        LayerResult(layer=DetectionLayer.L1, completed="yes")


def test_layer_result_rejects_a_non_int_duration():
    with pytest.raises(TypeError, match="duration_ms"):
        LayerResult(layer=DetectionLayer.L1, completed=True, duration_ms=12.5)


def test_layer_result_rejects_a_negative_duration():
    with pytest.raises(ValueError, match="duration_ms"):
        LayerResult(layer=DetectionLayer.L1, completed=True, duration_ms=-1)


# --------------------------------------------------------------------------
# 6. RiskAssessment
# --------------------------------------------------------------------------


def _signal(layer=DetectionLayer.L1, name="domain_age", score=0.5):
    return DetectionSignal(
        layer=layer,
        name=name,
        score=score,
        severity=RiskLevel.MEDIUM,
        evidence="example evidence",
    )


def test_risk_assessment_minimal_construction_applies_defaults():
    assessment = RiskAssessment(
        message_id="<abc@example.com>",
        score=0.2,
        level=RiskLevel.LOW,
    )

    assert assessment.signals == []
    assert assessment.summary is None
    assert assessment.layers_completed == []
    assert isinstance(assessment.scored_at, datetime)
    assert assessment.scored_at.tzinfo is timezone.utc


def test_risk_assessment_holds_signals_and_summary():
    signals = [_signal(), _signal(layer=DetectionLayer.L2, name="redirect_depth")]

    assessment = RiskAssessment(
        message_id="<abc@example.com>",
        score=0.71,
        level=RiskLevel.HIGH,
        signals=signals,
        summary="Newly registered domain and a 4-hop redirect chain.",
        layers_completed=[DetectionLayer.L1, DetectionLayer.L2],
    )

    assert assessment.signals == signals
    assert assessment.summary.startswith("Newly registered")
    assert assessment.layers_completed == [DetectionLayer.L1, DetectionLayer.L2]


def test_risk_assessment_groups_signals_by_layer():
    l1 = _signal(layer=DetectionLayer.L1, name="domain_age")
    l2 = _signal(layer=DetectionLayer.L2, name="redirect_depth")

    assessment = RiskAssessment(
        message_id="<abc@example.com>",
        score=0.5,
        level=RiskLevel.MEDIUM,
        signals=[l1, l2],
    )

    assert assessment.signals_for(DetectionLayer.L1) == [l1]
    assert assessment.signals_for(DetectionLayer.L2) == [l2]
    assert assessment.signals_for(DetectionLayer.L3) == []


def test_risk_assessment_mutable_defaults_are_not_shared_between_instances():
    first = RiskAssessment(message_id="<1@example.com>", score=0.1, level=RiskLevel.LOW)
    second = RiskAssessment(message_id="<2@example.com>", score=0.1, level=RiskLevel.LOW)

    first.signals.append(_signal())

    assert second.signals == []


@pytest.mark.parametrize("bad_score", [-0.5, 1.5, float("nan")])
def test_risk_assessment_rejects_scores_outside_the_unit_interval(bad_score):
    with pytest.raises(ValueError, match="score"):
        RiskAssessment(message_id="<abc@example.com>", score=bad_score, level=RiskLevel.LOW)


def test_risk_assessment_rejects_blank_message_id():
    with pytest.raises(ValueError, match="message_id"):
        RiskAssessment(message_id="", score=0.1, level=RiskLevel.LOW)


def test_risk_assessment_rejects_non_enum_level():
    with pytest.raises(TypeError, match="level"):
        RiskAssessment(message_id="<abc@example.com>", score=0.1, level="low")


# --------------------------------------------------------------------------
# 7. Attachment
# --------------------------------------------------------------------------


def test_attachment_defaults_to_a_non_inline_part():
    attachment = Attachment(filename="invoice.pdf", content_type="application/pdf")

    assert attachment.size_bytes == 0
    assert attachment.content_id is None
    assert attachment.is_inline is False


def test_attachment_with_content_id_is_inline():
    attachment = Attachment(filename="logo.png", content_type="image/png", content_id="logo123")

    assert attachment.is_inline is True


def test_attachment_rejects_negative_size():
    with pytest.raises(ValueError, match="size_bytes"):
        Attachment(filename="x.pdf", content_type="application/pdf", size_bytes=-1)


def test_attachment_rejects_blank_content_type():
    with pytest.raises(ValueError, match="content_type"):
        Attachment(filename="x.pdf", content_type="")


# --------------------------------------------------------------------------
# 8. Readable representations
# --------------------------------------------------------------------------


def test_dataclasses_have_readable_reprs():
    signal = _signal()
    assert "DetectionSignal" in repr(signal)
    assert "domain_age" in repr(signal)

    email = ParsedEmail(message_id="<abc@example.com>", from_addr="a@example.com")
    assert "ParsedEmail" in repr(email)
