"""Unit tests for the Layer 2 QR decode contract in layers/l2_urls.py.

The record and the seam only - no image is decoded, `pyzbar` is not installed,
and no concrete decoder exists. These tests pin the invariants that keep a
corrupt image from being read as an image with no QR code in it, and prove the
decoded payloads re-enter the URL candidate set through the existing seam.

An autouse guard forbids every socket and DNS entry point for the whole file.
"""

from __future__ import annotations

import sys

import pytest

from core.models import Attachment, ExtractedURL, ParsedEmail, URLSource
from layers.l2_urls import (
    QR_IMAGE_CONTENT_TYPES,
    QRDecodeResult,
    QRDecoder,
    QROutcome,
    QRPayload,
    candidate_urls,
)

QR_URL = "https://evil-phish.com/verify?u=1"
FAILING = [
    QROutcome.UNSUPPORTED,
    QROutcome.UNREADABLE,
    QROutcome.TIMED_OUT,
    QROutcome.NOT_ATTEMPTED,
]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("the QR contract suite touched the network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket, "gethostbyname", forbidden)


def attachment(
    filename: str = "invoice.png",
    content_type: str = "image/png",
    content_id: str | None = "img001",
) -> Attachment:
    return Attachment(
        filename=filename, content_type=content_type, size_bytes=2048, content_id=content_id
    )


def decoded(*payloads: QRPayload, part: Attachment | None = None) -> QRDecodeResult:
    return QRDecodeResult(
        attachment=part or attachment(), outcome=QROutcome.DECODED, payloads=payloads
    )


# --- 1, 2, 3, 4. successful decodes ---------------------------------------


def test_a_decode_finding_one_url_records_it_verbatim() -> None:
    result = decoded(QRPayload(QR_URL, "QRCODE"))

    assert result.is_answer is True
    assert result.found_any is True
    assert result.error is None
    assert result.payloads[0].data == QR_URL
    assert result.payloads[0].symbol == "QRCODE"


def test_a_decode_can_report_several_symbols() -> None:
    result = decoded(
        QRPayload(QR_URL, "QRCODE"),
        QRPayload("https://second-phish.com/x", "QRCODE"),
        QRPayload("WIFI:S=guest;T=WPA;", "QRCODE"),
    )

    assert len(result.payloads) == 3
    assert [p.data for p in result.payloads] == [
        QR_URL,
        "https://second-phish.com/x",
        "WIFI:S=guest;T=WPA;",
    ]


def test_an_image_with_no_symbol_is_a_genuine_negative() -> None:
    """Read successfully, carried nothing - not the same as unreadable."""
    result = decoded()

    assert result.is_answer is True
    assert result.found_any is False
    assert result.payloads == ()
    assert result.error is None


def test_a_symbol_that_is_not_a_url_is_still_recorded() -> None:
    """The contract does not validate URLs; candidate_urls decides that later."""
    result = decoded(QRPayload("BEGIN:VCARD\nFN:Nobody\nEND:VCARD"))

    assert result.found_any is True
    assert result.as_extracted_urls()[0].url.startswith("BEGIN:VCARD")


# --- 5, 6, 7. failures and invalid combinations ---------------------------


@pytest.mark.parametrize(
    "outcome, reason",
    [
        (QROutcome.UNSUPPORTED, "application/pdf is not an image"),
        (QROutcome.UNREADABLE, "truncated PNG: cannot open image"),
        (QROutcome.TIMED_OUT, "decoder gave up after its budget"),
        (QROutcome.NOT_ATTEMPTED, "no decoder configured"),
    ],
)
def test_a_failed_decode_is_not_an_answer(outcome: QROutcome, reason: str) -> None:
    result = QRDecodeResult(attachment=attachment(), outcome=outcome, error=reason)

    assert result.is_answer is False
    assert outcome.is_answer is False
    assert result.found_any is False
    assert result.error == reason
    assert result.payloads == ()


@pytest.mark.parametrize("outcome", FAILING)
def test_a_failed_decode_must_state_why(outcome: QROutcome) -> None:
    with pytest.raises(ValueError, match="state why"):
        QRDecodeResult(attachment=attachment(), outcome=outcome)


@pytest.mark.parametrize("outcome", FAILING)
def test_a_failed_decode_cannot_report_payloads(outcome: QROutcome) -> None:
    """It read no image, so it cannot have found anything in one."""
    with pytest.raises(ValueError, match="cannot"):
        QRDecodeResult(
            attachment=attachment(),
            outcome=outcome,
            error="boom",
            payloads=(QRPayload(QR_URL),),
        )


@pytest.mark.parametrize("outcome", FAILING)
def test_a_failed_decode_never_reads_as_a_clean_image(outcome: QROutcome) -> None:
    """The invariant this contract exists for."""
    result = QRDecodeResult(attachment=attachment(), outcome=outcome, error="corrupt")

    assert result.found_any is False   # but ...
    assert result.is_answer is False   # ... this is what a caller must check
    assert result.as_extracted_urls() == []


def test_a_decoded_result_may_not_carry_an_error() -> None:
    with pytest.raises(ValueError, match="must not carry an error"):
        QRDecodeResult(
            attachment=attachment(), outcome=QROutcome.DECODED, error="boom"
        )


def test_the_outcome_and_attachment_types_are_enforced() -> None:
    with pytest.raises(TypeError, match="QROutcome"):
        QRDecodeResult(attachment=attachment(), outcome="decoded")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Attachment"):
        QRDecodeResult(attachment="invoice.png", outcome=QROutcome.DECODED)  # type: ignore[arg-type]


def test_elapsed_time_may_not_be_negative() -> None:
    with pytest.raises(ValueError, match="elapsed_ms"):
        QRDecodeResult(
            attachment=attachment(), outcome=QROutcome.DECODED, elapsed_ms=-1
        )


# --- 8. empty payloads ----------------------------------------------------


def test_an_empty_decoded_payload_is_rejected() -> None:
    """A decoder reporting one has malfunctioned; it must not reach the candidates."""
    with pytest.raises(ValueError, match="must not be empty"):
        QRPayload("")


def test_a_non_string_payload_is_rejected() -> None:
    with pytest.raises(TypeError, match="str"):
        QRPayload(b"https://evil-phish.com/x")  # type: ignore[arg-type]


def test_whitespace_is_preserved_not_stripped() -> None:
    """Verbatim means verbatim; trimming is a decision for whoever consumes it."""
    assert QRPayload("  https://evil-phish.com/x  ").data == "  https://evil-phish.com/x  "


# --- 9, 10. verbatim payloads and provenance ------------------------------


def test_the_decoded_string_is_preserved_exactly() -> None:
    messy = "HTTPS://Evil-Phish.com:443/Verify?u=dGVzdA%3D%3D#frag"
    result = decoded(QRPayload(messy))

    assert result.payloads[0].data == messy
    assert result.as_extracted_urls()[0].url == messy


def test_attachment_provenance_is_preserved() -> None:
    part = attachment(filename="scan.jpg", content_type="image/jpeg", content_id="cid42")
    result = decoded(QRPayload(QR_URL), part=part)

    assert result.attachment is part
    assert result.provenance == {
        "filename": "scan.jpg",
        "content_type": "image/jpeg",
        "content_id": "cid42",
        "inline": True,
    }


def test_an_ordinary_attachment_is_distinguishable_from_an_inline_image() -> None:
    """Section 2: Layer 2 needs both, and inline is derived from content_id."""
    plain = decoded(QRPayload(QR_URL), part=attachment(content_id=None))

    assert plain.provenance["inline"] is False
    assert plain.provenance["content_id"] is None


def test_the_image_content_types_cover_the_formats_worth_decoding() -> None:
    assert "image/png" in QR_IMAGE_CONTENT_TYPES
    assert "image/jpeg" in QR_IMAGE_CONTENT_TYPES
    assert "application/pdf" not in QR_IMAGE_CONTENT_TYPES


# --- 11. immutability and non-mutation ------------------------------------


def test_the_records_are_immutable() -> None:
    result = decoded(QRPayload(QR_URL))

    with pytest.raises(Exception):
        result.outcome = QROutcome.UNREADABLE  # type: ignore[misc]
    with pytest.raises(Exception):
        result.payloads = ()  # type: ignore[misc]
    with pytest.raises(Exception):
        result.payloads[0].data = "changed"  # type: ignore[misc]


def test_nothing_upstream_is_mutated() -> None:
    part = attachment()
    message = ParsedEmail(
        message_id="<m@example.com>",
        from_addr="s@example.com",
        urls=[ExtractedURL(url="https://good.example/x", source=URLSource.ANCHOR_HREF)],
        attachments=[part],
    )

    result = decoded(QRPayload(QR_URL), part=part)
    result.as_extracted_urls()

    assert message.attachments[0].filename == "invoice.png"
    assert message.attachments[0].content_id == "img001"
    assert not hasattr(message.attachments[0], "payload")
    assert len(message.urls) == 1
    assert message.urls[0].url == "https://good.example/x"


def test_the_attachment_model_still_carries_no_bytes() -> None:
    """Section 2: 'The payload is deliberately not carried on the contract.'"""
    fields = set(Attachment.__dataclass_fields__)

    assert fields == {"filename", "content_type", "size_bytes", "content_id"}


# --- 12, 13. the seam and re-entry ----------------------------------------


def test_a_decoder_is_anything_with_the_decode_method() -> None:
    class FakeDecoder:
        def __init__(self) -> None:
            self.calls: list[tuple[bytes, str]] = []

        def decode(self, payload: bytes, part: Attachment) -> QRDecodeResult:
            self.calls.append((payload, part.filename))
            return decoded(QRPayload(QR_URL, "QRCODE"), part=part)

    decoder = FakeDecoder()
    part = attachment()

    assert isinstance(decoder, QRDecoder)
    result = decoder.decode(b"\x89PNG fake bytes", part)
    assert decoder.calls == [(b"\x89PNG fake bytes", "invoice.png")]
    assert result.found_any is True


def test_a_fake_decoder_can_express_every_outcome() -> None:
    class ScriptedDecoder:
        def decode(self, payload: bytes, part: Attachment) -> QRDecodeResult:
            if part.content_type not in QR_IMAGE_CONTENT_TYPES:
                return QRDecodeResult(
                    attachment=part,
                    outcome=QROutcome.UNSUPPORTED,
                    error=f"{part.content_type} is not an image",
                )
            if not payload:
                return QRDecodeResult(
                    attachment=part, outcome=QROutcome.UNREADABLE, error="empty image"
                )
            return decoded(QRPayload(QR_URL), part=part)

    decoder = ScriptedDecoder()

    assert isinstance(decoder, QRDecoder)
    assert decoder.decode(b"x", attachment()).outcome is QROutcome.DECODED
    assert decoder.decode(b"", attachment()).outcome is QROutcome.UNREADABLE
    assert (
        decoder.decode(b"x", attachment(content_type="application/pdf")).outcome
        is QROutcome.UNSUPPORTED
    )


def test_decoded_urls_re_enter_the_candidate_set() -> None:
    """Section 4: 'extracted URL re-enters the URL signal set'."""
    message = ParsedEmail(
        message_id="<m@example.com>",
        from_addr="s@example.com",
        urls=[ExtractedURL(url="https://good.example/a", source=URLSource.ANCHOR_HREF)],
    )
    result = decoded(QRPayload(QR_URL), QRPayload("https://second-phish.com/x"))

    candidates = candidate_urls(message, extra=result.as_extracted_urls())

    assert [c.host for c in candidates] == [
        "good.example",
        "evil-phish.com",
        "second-phish.com",
    ]
    assert candidates[1].source is URLSource.QR_CODE
    assert candidates[1].raw == QR_URL


def test_non_url_payloads_are_dropped_by_the_existing_candidate_filter() -> None:
    """The contract validates nothing; candidate_urls is where that already lives."""
    message = ParsedEmail(message_id="<m@example.com>", from_addr="s@example.com")
    result = decoded(
        QRPayload("WIFI:S=guest;T=WPA;"),
        QRPayload("mailto:victim@example.com"),
        QRPayload(QR_URL),
    )

    candidates = candidate_urls(message, extra=result.as_extracted_urls())

    assert [c.raw for c in candidates] == [QR_URL]


def test_a_failed_decode_injects_nothing_into_the_candidate_set() -> None:
    message = ParsedEmail(message_id="<m@example.com>", from_addr="s@example.com")
    result = QRDecodeResult(
        attachment=attachment(), outcome=QROutcome.UNREADABLE, error="corrupt"
    )

    assert candidate_urls(message, extra=result.as_extracted_urls()) == []


def test_not_attempted_is_the_state_this_repository_ships_in() -> None:
    result = QRDecodeResult.not_attempted(attachment(), "pyzbar is not installed")

    assert result.outcome is QROutcome.NOT_ATTEMPTED
    assert result.is_answer is False
    assert result.as_extracted_urls() == []


# --- 15, 16. nothing decoded, nothing installed ---------------------------


def test_pyzbar_is_not_installed_and_is_never_imported() -> None:
    assert "pyzbar" not in sys.modules

    with pytest.raises(ModuleNotFoundError):
        __import__("pyzbar")

    assert "pyzbar" not in sys.modules


def test_no_decoder_implementation_ships_in_this_repository() -> None:
    """There is no decoder, and no bytes to give one - see the module note."""
    import layers.l2_urls as module

    concrete = [
        name
        for name in dir(module)
        if isinstance(getattr(module, name), type)
        and name != "QRDecoder"
        and hasattr(getattr(module, name), "decode")
    ]
    assert concrete == []
