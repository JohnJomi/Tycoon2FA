"""Unit tests for the l2.qr_url signal in layers/l2_urls.py.

The decoder is always an injected fake: `pyzbar` is not installed and no
concrete decoder exists. Attachment bytes come from real parsed messages
through `ingest.parser.attachment_payload`. An autouse guard forbids every
socket and DNS entry point for the whole file.
"""

from __future__ import annotations

import sys
from email.message import EmailMessage

import pytest

from core.models import Attachment, DetectionLayer, ParsedEmail, RiskLevel, URLSource
from ingest.parser import parse_email
from layers.l2_urls import (
    QR_IMAGE_CONTENT_TYPES,
    QR_URL_SCORE,
    QRDecodeResult,
    QROutcome,
    QRPayload,
    analyze_qr_url,
    eligible_qr_attachments,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"fake image bytes" * 4
JPEG = b"\xff\xd8\xff\xe0" + b"second image" * 4
PDF = b"%PDF-1.4 not an image"

QR_URL = "https://evil-phish.com/verify?u=1"
OTHER_URL = "https://second-phish.com/x"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("the qr_url suite touched the network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket, "gethostbyname", forbidden)


def build(*parts: tuple[bytes, str, str, str | None], body_url: str | None = None) -> bytes:
    message = EmailMessage()
    message["From"] = "Sender <sender@example.com>"
    message["To"] = "victim@corp-invoices.com"
    message["Subject"] = "Invoice"
    message["Message-ID"] = "<m@example.com>"
    message.set_content(f"Scan the code. {body_url or ''}")

    for payload, content_type, name, cid in parts:
        maintype, _, subtype = content_type.partition("/")
        kwargs: dict[str, object] = {
            "maintype": maintype,
            "subtype": subtype,
            "filename": name,
        }
        if cid is not None:
            kwargs["cid"] = f"<{cid}>"
            kwargs["disposition"] = "inline"
        message.add_attachment(payload, **kwargs)  # type: ignore[arg-type]
    return message.as_bytes()


def email(*parts, body_url: str | None = None) -> ParsedEmail:
    return parse_email(build(*parts, body_url=body_url))


class FakeDecoder:
    """Returns scripted payloads keyed by filename. Records what it was given."""

    def __init__(self, scripted: dict[str, list[str]] | None = None, **outcomes) -> None:
        self.scripted = scripted or {}
        self.outcomes: dict[str, QROutcome] = outcomes.get("outcomes", {})
        self.calls: list[tuple[str, int]] = []

    def decode(self, payload: bytes, attachment: Attachment) -> QRDecodeResult:
        self.calls.append((attachment.filename, len(payload)))
        outcome = self.outcomes.get(attachment.filename)
        if outcome is not None:
            return QRDecodeResult(
                attachment=attachment, outcome=outcome, error=f"{outcome.value} image"
            )
        data = self.scripted.get(attachment.filename, [])
        return QRDecodeResult(
            attachment=attachment,
            outcome=QROutcome.DECODED,
            payloads=tuple(QRPayload(d, "QRCODE") for d in data),
        )


# --- 1, 2, 3, 4. findings -------------------------------------------------


def test_one_image_carrying_a_url_is_a_finding() -> None:
    message = email((PNG, "image/png", "invoice.png", None))
    decoder = FakeDecoder({"invoice.png": [QR_URL]})

    signal = analyze_qr_url(message, decoder=decoder)

    assert signal.layer is DetectionLayer.L2
    assert signal.name == "qr_url"
    assert signal.score == QR_URL_SCORE
    assert signal.severity is RiskLevel.HIGH
    assert signal.error is None
    assert signal.metadata["fired"] is True
    assert signal.metadata["images_inspected"] == 1
    assert signal.metadata["matches"] == [
        {
            "filename": "invoice.png",
            "content_type": "image/png",
            "content_id": None,
            "inline": False,
            "url": QR_URL,
            "host": "evil-phish.com",
        }
    ]
    assert QR_URL in signal.evidence
    assert decoder.calls == [("invoice.png", len(PNG))]


def test_several_urls_from_one_image_are_all_reported() -> None:
    message = email((PNG, "image/png", "a.png", None))
    signal = analyze_qr_url(message, decoder=FakeDecoder({"a.png": [QR_URL, OTHER_URL]}))

    assert [m["url"] for m in signal.metadata["matches"]] == [QR_URL, OTHER_URL]


def test_several_images_are_decoded_independently() -> None:
    message = email(
        (PNG, "image/png", "one.png", None), (JPEG, "image/jpeg", "two.jpg", None)
    )
    decoder = FakeDecoder({"one.png": [QR_URL], "two.jpg": [OTHER_URL]})

    signal = analyze_qr_url(message, decoder=decoder)

    assert decoder.calls == [("one.png", len(PNG)), ("two.jpg", len(JPEG))]
    assert [m["filename"] for m in signal.metadata["matches"]] == ["one.png", "two.jpg"]


def test_an_inline_image_is_decoded_and_its_provenance_recorded() -> None:
    message = email((PNG, "image/png", "logo.png", "cid42"))
    signal = analyze_qr_url(message, decoder=FakeDecoder({"logo.png": [QR_URL]}))

    match = signal.metadata["matches"][0]
    assert match["inline"] is True
    assert match["content_id"] == "cid42"
    assert "inline image" in signal.evidence


# --- 5, 10, 11, 12. payload handling --------------------------------------


def test_a_non_url_payload_is_recorded_but_is_not_a_finding() -> None:
    message = email((PNG, "image/png", "wifi.png", None))
    signal = analyze_qr_url(
        message, decoder=FakeDecoder({"wifi.png": ["WIFI:S=guest;T=WPA;"]})
    )

    assert signal.score == 0.0
    assert signal.error is None            # inspected successfully: a clean result
    assert signal.metadata["matches"] == []
    assert signal.metadata["non_url_payloads"][0]["payload"] == "WIFI:S=guest;T=WPA;"
    assert "encode no web address" in signal.evidence


@pytest.mark.parametrize(
    "payload", ["mailto:victim@example.com", "tel:+441234567890", "just some text"]
)
def test_other_non_http_payloads_do_not_become_findings(payload: str) -> None:
    message = email((PNG, "image/png", "a.png", None))
    signal = analyze_qr_url(message, decoder=FakeDecoder({"a.png": [payload]}))

    assert signal.score == 0.0
    assert signal.metadata["matches"] == []


def test_the_same_url_in_two_images_becomes_one_candidate() -> None:
    message = email(
        (PNG, "image/png", "one.png", None), (JPEG, "image/jpeg", "two.jpg", None)
    )
    signal = analyze_qr_url(
        message, decoder=FakeDecoder({"one.png": [QR_URL], "two.jpg": [QR_URL]})
    )

    assert len(signal.metadata["matches"]) == 1
    assert signal.metadata["matches"][0]["filename"] == "one.png"  # first seen wins


def test_a_qr_url_already_in_the_body_is_deduplicated() -> None:
    message = email((PNG, "image/png", "a.png", None), body_url=QR_URL)
    signal = analyze_qr_url(message, decoder=FakeDecoder({"a.png": [QR_URL]}))

    # The body observation came first, so the QR one is not a separate candidate.
    assert signal.metadata["matches"] == []
    assert signal.score == 0.0


def test_the_qr_source_is_what_marks_a_candidate_as_concealed() -> None:
    from layers.l2_urls import candidate_urls

    message = email((PNG, "image/png", "a.png", None))
    result = QRDecodeResult(
        attachment=message.attachments[0],
        outcome=QROutcome.DECODED,
        payloads=(QRPayload(QR_URL),),
    )

    candidates = candidate_urls(message, extra=result.as_extracted_urls())

    assert [c.source for c in candidates] == [URLSource.QR_CODE]


# --- 6, 7, 8, 19. abstention ----------------------------------------------


def test_an_unreadable_image_abstains() -> None:
    message = email((PNG, "image/png", "corrupt.png", None))
    decoder = FakeDecoder(outcomes={"corrupt.png": QROutcome.UNREADABLE})

    signal = analyze_qr_url(message, decoder=decoder)

    assert signal.score == 0.0
    assert signal.metadata["fired"] is False
    assert signal.error is not None
    assert "absence of information, not a clean result" in signal.evidence
    assert signal.metadata["failures"][0]["filename"] == "corrupt.png"


def test_no_decoder_injected_abstains() -> None:
    """The state this repository ships in: pyzbar is not installed."""
    message = email((PNG, "image/png", "a.png", None))

    signal = analyze_qr_url(message)

    assert signal.score == 0.0
    assert signal.error == "no QR decoder configured"
    assert signal.metadata["images_inspected"] == 0
    assert len(signal.metadata["failures"]) == 1
    assert "not a clean result" in signal.evidence


def test_a_decoder_that_raises_is_an_abstention_not_a_verdict() -> None:
    class Broken:
        def decode(self, payload: bytes, attachment: Attachment) -> QRDecodeResult:
            raise RuntimeError("zbar segfaulted")

    signal = analyze_qr_url(email((PNG, "image/png", "a.png", None)), decoder=Broken())

    assert signal.score == 0.0
    assert signal.error is not None
    assert "zbar segfaulted" in signal.metadata["failures"][0]["error"]


def test_a_decoder_returning_the_wrong_type_is_an_abstention() -> None:
    class Wrong:
        def decode(self, payload: bytes, attachment: Attachment):
            return ["https://evil-phish.com/x"]

    signal = analyze_qr_url(email((PNG, "image/png", "a.png", None)), decoder=Wrong())

    assert signal.error is not None
    assert "not a QRDecodeResult" in signal.metadata["failures"][0]["error"]


def test_an_oversized_payload_fails_safely() -> None:
    message = email((PNG, "image/png", "big.png", None))
    decoder = FakeDecoder({"big.png": [QR_URL]})

    signal = analyze_qr_url(message, decoder=decoder, max_payload_bytes=4)

    assert signal.score == 0.0
    assert signal.error is not None
    assert decoder.calls == []  # never handed bytes it could not bound
    assert "above the" in signal.metadata["failures"][0]["error"]


def test_a_message_without_raw_bytes_abstains_rather_than_reporting_clean() -> None:
    part = Attachment(filename="a.png", content_type="image/png", size_bytes=5)
    message = ParsedEmail(
        message_id="<m@example.com>", from_addr="s@example.com", attachments=[part]
    )

    signal = analyze_qr_url(message, decoder=FakeDecoder({"a.png": [QR_URL]}))

    assert signal.score == 0.0
    assert signal.error is not None
    assert "raw bytes" in signal.metadata["failures"][0]["error"]


def test_a_finding_survives_a_sibling_failure() -> None:
    message = email(
        (PNG, "image/png", "good.png", None), (JPEG, "image/jpeg", "bad.jpg", None)
    )
    decoder = FakeDecoder(
        {"good.png": [QR_URL]}, outcomes={"bad.jpg": QROutcome.UNREADABLE}
    )

    signal = analyze_qr_url(message, decoder=decoder)

    assert signal.score == QR_URL_SCORE
    assert signal.error is None                     # the finding stands
    assert signal.metadata["matches"][0]["url"] == QR_URL
    assert len(signal.metadata["failures"]) == 1    # but the gap is reported
    assert "could not be inspected" in signal.evidence


def test_a_clean_verdict_is_not_issued_while_an_image_is_uninspected() -> None:
    message = email(
        (PNG, "image/png", "clean.png", None), (JPEG, "image/jpeg", "bad.jpg", None)
    )
    decoder = FakeDecoder({"clean.png": []}, outcomes={"bad.jpg": QROutcome.UNREADABLE})

    signal = analyze_qr_url(message, decoder=decoder)

    assert signal.score == 0.0
    assert signal.error is not None
    assert signal.metadata["images_inspected"] == 1


# --- 9, 13. eligibility and clean results ---------------------------------


def test_a_message_with_no_images_is_a_genuine_negative() -> None:
    decoder = FakeDecoder()
    signal = analyze_qr_url(email((PDF, "application/pdf", "doc.pdf", None)), decoder=decoder)

    assert signal.score == 0.0
    assert signal.error is None
    assert signal.metadata["images_eligible"] == 0
    assert decoder.calls == []
    assert "no image attachments" in signal.evidence


def test_a_message_with_no_attachments_at_all_is_a_genuine_negative() -> None:
    signal = analyze_qr_url(email(), decoder=FakeDecoder())

    assert signal.score == 0.0
    assert signal.error is None


def test_an_image_with_no_code_is_a_genuine_negative() -> None:
    message = email((PNG, "image/png", "photo.png", None))
    signal = analyze_qr_url(message, decoder=FakeDecoder({"photo.png": []}))

    assert signal.score == 0.0
    assert signal.error is None
    assert signal.metadata["images_inspected"] == 1
    assert "carry no encoded URL" in signal.evidence


def test_only_image_parts_are_offered_to_the_decoder() -> None:
    message = email(
        (PDF, "application/pdf", "doc.pdf", None),
        (PNG, "image/png", "a.png", None),
        (JPEG, "image/jpeg", "b.jpg", "cid1"),
    )

    eligible = eligible_qr_attachments(message)

    assert [a.filename for a in eligible] == ["a.png", "b.jpg"]
    assert all(a.content_type in QR_IMAGE_CONTENT_TYPES for a in eligible)


# --- 14, 15, 16, 17, 18. evidence, purity, determinism --------------------


def test_the_decoded_url_is_quoted_verbatim() -> None:
    messy = "HTTPS://Evil-Phish.com:443/Verify?u=1&session=SECRET"
    message = email((PNG, "image/png", "a.png", None))

    signal = analyze_qr_url(message, decoder=FakeDecoder({"a.png": [messy]}))

    assert signal.metadata["matches"][0]["url"] == messy
    assert messy in signal.evidence


def test_evidence_and_metadata_expose_no_image_contents() -> None:
    message = email((PNG, "image/png", "a.png", "cid7"))
    signal = analyze_qr_url(message, decoder=FakeDecoder({"a.png": [QR_URL]}))

    match = signal.metadata["matches"][0]
    assert set(match) == {"filename", "content_type", "content_id", "inline", "url", "host"}
    assert set(signal.metadata) == {
        "attachments_in_message",
        "images_eligible",
        "images_inspected",
        "matches",
        "non_url_payloads",
        "failures",
        "fired",
    }
    assert "PNG" not in str(signal.metadata)


def test_the_message_and_attachments_are_not_mutated() -> None:
    message = email((PNG, "image/png", "a.png", "cid1"))
    before = (
        message.attachments[0].filename,
        message.attachments[0].content_type,
        message.attachments[0].size_bytes,
        message.attachments[0].content_id,
    )
    urls_before = list(message.urls)

    analyze_qr_url(message, decoder=FakeDecoder({"a.png": [QR_URL]}))

    assert (
        message.attachments[0].filename,
        message.attachments[0].content_type,
        message.attachments[0].size_bytes,
        message.attachments[0].content_id,
    ) == before
    assert message.urls == urls_before
    assert not hasattr(message.attachments[0], "payload")


def test_repeated_analysis_is_deterministic() -> None:
    message = email(
        (PNG, "image/png", "one.png", None), (JPEG, "image/jpeg", "two.jpg", None)
    )

    results = {
        (
            analyze_qr_url(
                message, decoder=FakeDecoder({"one.png": [QR_URL], "two.jpg": [OTHER_URL]})
            ).score,
            analyze_qr_url(
                message, decoder=FakeDecoder({"one.png": [QR_URL], "two.jpg": [OTHER_URL]})
            ).evidence,
        )
        for _ in range(5)
    }

    assert len(results) == 1


def test_the_signal_never_dereferences_a_payload() -> None:
    """The socket guard proves it; this pins the intent."""
    message = email((PNG, "image/png", "a.png", None))
    signal = analyze_qr_url(message, decoder=FakeDecoder({"a.png": [QR_URL]}))

    assert signal.score == QR_URL_SCORE  # reached without any network call


def test_pyzbar_is_never_imported() -> None:
    assert "pyzbar" not in sys.modules
    analyze_qr_url(email((PNG, "image/png", "a.png", None)), decoder=FakeDecoder())
    assert "pyzbar" not in sys.modules
