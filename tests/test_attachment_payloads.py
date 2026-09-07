"""Unit tests for attachment payload retrieval in ingest/parser.py.

`Attachment` stays metadata only, per ARCHITECTURE.md section 2; the bytes are
recovered on demand from `ParsedEmail.raw`, which the parser already keeps.
Nothing here decodes a QR code or imports pyzbar, and an autouse guard forbids
every socket and DNS entry point.
"""

from __future__ import annotations

import sys
from email.message import EmailMessage

import pytest

from core.models import Attachment, ParsedEmail
from ingest.parser import (
    MAX_ATTACHMENT_PAYLOAD_BYTES,
    AttachmentPayloadUnavailable,
    attachment_payload,
    parse_email,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"fake image data" * 4
JPEG = b"\xff\xd8\xff\xe0" + b"other image bytes" * 3
PDF = b"%PDF-1.4 not an image"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("the attachment payload suite touched the network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket, "gethostbyname", forbidden)


def build(*parts: tuple[bytes, str, str, str | None]) -> bytes:
    """An RFC-822 message carrying the given (payload, maintype/subtype, name, cid)."""
    message = EmailMessage()
    message["From"] = "Sender <sender@example.com>"
    message["To"] = "victim@corp-invoices.com"
    message["Subject"] = "Invoice"
    message["Message-ID"] = "<m@example.com>"
    message.set_content("Please scan the code.")

    for payload, content_type, name, cid in parts:
        maintype, _, subtype = content_type.partition("/")
        kwargs: dict[str, object] = {"maintype": maintype, "subtype": subtype}
        kwargs["filename"] = name
        if cid is not None:
            kwargs["cid"] = f"<{cid}>"
            kwargs["disposition"] = "inline"
        message.add_attachment(payload, **kwargs)  # type: ignore[arg-type]
    return message.as_bytes()


# --- 1 & 2. ordinary and inline images ------------------------------------


def test_an_ordinary_image_attachment_payload_is_retrievable() -> None:
    email = parse_email(build((PNG, "image/png", "invoice.png", None)))
    (part,) = email.attachments

    assert attachment_payload(email, part) == PNG
    assert part.content_type == "image/png"
    assert part.is_inline is False


def test_an_inline_image_payload_is_retrievable() -> None:
    email = parse_email(build((PNG, "image/png", "logo.png", "img001")))
    (part,) = email.attachments

    assert part.is_inline is True
    assert part.content_id is not None
    assert attachment_payload(email, part) == PNG


def test_inline_images_are_reachable_through_the_derived_filter() -> None:
    """Section 2: `inline_images` is derived; there is one attachment list."""
    email = parse_email(
        build(
            (PNG, "image/png", "logo.png", "cid-logo"),
            (JPEG, "image/jpeg", "scan.jpg", None),
        )
    )

    (inline,) = email.inline_images

    assert attachment_payload(email, inline) == PNG
    assert len(email.attachments) == 2


# --- 3, 7, 8. mapping attachments to payloads -----------------------------


def test_several_attachments_map_to_their_own_payloads() -> None:
    email = parse_email(
        build(
            (PNG, "image/png", "one.png", None),
            (JPEG, "image/jpeg", "two.jpg", None),
            (PDF, "application/pdf", "three.pdf", None),
        )
    )

    payloads = [attachment_payload(email, a) for a in email.attachments]

    assert payloads == [PNG, JPEG, PDF]


def test_the_content_id_identifies_the_right_inline_part() -> None:
    email = parse_email(
        build(
            (PNG, "image/png", "a.png", "first"),
            (JPEG, "image/jpeg", "b.jpg", "second"),
        )
    )

    by_cid = {a.content_id: a for a in email.attachments}

    assert attachment_payload(email, by_cid["first"]) == PNG
    assert attachment_payload(email, by_cid["second"]) == JPEG


def test_duplicate_filenames_still_map_to_distinct_payloads() -> None:
    """Ordinal position is the mapping; metadata alone would be ambiguous."""
    first = b"\x89PNG first payload"
    second = b"\x89PNG second payload"
    email = parse_email(
        build(
            (first, "image/png", "invoice.png", None),
            (second, "image/png", "invoice.png", None),
        )
    )

    a, b = email.attachments

    assert a.filename == b.filename == "invoice.png"
    assert attachment_payload(email, a) == first
    assert attachment_payload(email, b) == second


def test_an_attachment_from_another_message_is_refused() -> None:
    email = parse_email(build((PNG, "image/png", "a.png", None)))
    stranger = Attachment(filename="elsewhere.png", content_type="image/png", size_bytes=10)

    with pytest.raises(AttachmentPayloadUnavailable, match="not an attachment"):
        attachment_payload(email, stranger)


# --- 4. non-image parts ---------------------------------------------------


def test_a_non_image_attachment_is_retrievable_but_not_a_qr_candidate() -> None:
    """Filtering by content type belongs to Layer 2, which owns the format list."""
    from layers.l2_urls import QR_IMAGE_CONTENT_TYPES

    email = parse_email(
        build((PDF, "application/pdf", "statement.pdf", None), (PNG, "image/png", "q.png", None))
    )

    images = [a for a in email.attachments if a.content_type in QR_IMAGE_CONTENT_TYPES]

    assert [a.filename for a in images] == ["q.png"]
    assert attachment_payload(email, images[0]) == PNG


# --- 5 & 6. failure and emptiness -----------------------------------------


def test_a_message_without_raw_bytes_fails_safely() -> None:
    """A ParsedEmail built directly carries no raw, so nothing can be recovered."""
    part = Attachment(filename="a.png", content_type="image/png", size_bytes=3)
    email = ParsedEmail(
        message_id="<m@example.com>", from_addr="s@example.com", attachments=[part]
    )

    with pytest.raises(AttachmentPayloadUnavailable, match="no raw bytes"):
        attachment_payload(email, part)


def test_a_truncated_message_does_not_raise_something_unexpected() -> None:
    email = parse_email(build((PNG, "image/png", "a.png", None)))
    truncated = ParsedEmail(
        message_id=email.message_id,
        from_addr=email.from_addr,
        attachments=list(email.attachments),
        raw=email.raw[: len(email.raw) // 3],
    )

    try:
        payload = attachment_payload(truncated, truncated.attachments[0])
    except AttachmentPayloadUnavailable:
        pass  # the honest outcome
    else:
        assert isinstance(payload, bytes)  # a tolerant parser recovered something


def test_an_empty_attachment_returns_empty_bytes_rather_than_failing() -> None:
    """An empty part is a real thing a message can contain, not a failure."""
    email = parse_email(build((b"", "image/png", "empty.png", None)))
    (part,) = email.attachments

    assert part.size_bytes == 0
    assert attachment_payload(email, part) == b""


def test_a_wrong_type_is_rejected_loudly() -> None:
    email = parse_email(build((PNG, "image/png", "a.png", None)))

    with pytest.raises(TypeError, match="Attachment"):
        attachment_payload(email, "a.png")  # type: ignore[arg-type]


# --- 11. bounds -----------------------------------------------------------


def test_a_payload_above_the_limit_is_refused_rather_than_loaded() -> None:
    email = parse_email(build((PNG, "image/png", "big.png", None)))
    (part,) = email.attachments

    with pytest.raises(AttachmentPayloadUnavailable, match="above the"):
        attachment_payload(email, part, max_bytes=4)


def test_a_payload_exactly_at_the_limit_is_allowed() -> None:
    email = parse_email(build((PNG, "image/png", "a.png", None)))
    (part,) = email.attachments

    assert attachment_payload(email, part, max_bytes=len(PNG)) == PNG


def test_the_default_limit_is_bounded_and_sane() -> None:
    assert MAX_ATTACHMENT_PAYLOAD_BYTES == 16 * 1024 * 1024


def test_a_non_positive_limit_is_rejected() -> None:
    email = parse_email(build((PNG, "image/png", "a.png", None)))

    with pytest.raises(ValueError, match="max_bytes"):
        attachment_payload(email, email.attachments[0], max_bytes=0)


# --- 9 & 10. nothing existing changed -------------------------------------


def test_attachment_metadata_is_unchanged_by_retrieval() -> None:
    email = parse_email(build((PNG, "image/png", "invoice.png", "cid9")))
    (part,) = email.attachments
    before = (part.filename, part.content_type, part.size_bytes, part.content_id)

    attachment_payload(email, part)

    assert (part.filename, part.content_type, part.size_bytes, part.content_id) == before
    assert part.size_bytes == len(PNG)


def test_the_attachment_model_still_carries_no_payload_field() -> None:
    """Section 2: 'The payload is deliberately not carried on the contract.'"""
    assert set(Attachment.__dataclass_fields__) == {
        "filename",
        "content_type",
        "size_bytes",
        "content_id",
    }


def test_the_parsed_message_is_not_mutated_by_retrieval() -> None:
    email = parse_email(build((PNG, "image/png", "a.png", None)))
    before_raw = email.raw
    before_urls = list(email.urls)

    attachment_payload(email, email.attachments[0])

    assert email.raw is before_raw
    assert email.urls == before_urls
    assert len(email.attachments) == 1


def test_repeated_retrieval_is_deterministic_and_holds_no_extra_copy() -> None:
    email = parse_email(build((PNG, "image/png", "a.png", None)))
    part = email.attachments[0]

    first = attachment_payload(email, part)
    second = attachment_payload(email, part)

    assert first == second == PNG
    # The bytes live in `raw`; nothing is cached onto the model.
    assert not hasattr(part, "payload")
    assert not hasattr(email, "_payloads")


# --- 12 & 13. isolation and the QR seam -----------------------------------


def test_pyzbar_is_not_imported() -> None:
    assert "pyzbar" not in sys.modules


def test_the_bytes_flow_into_the_existing_qr_decoder_seam() -> None:
    """The contract's `decode(payload, attachment)` is fed straight from here."""
    from layers.l2_urls import (
        QR_IMAGE_CONTENT_TYPES,
        QRDecodeResult,
        QRDecoder,
        QROutcome,
        QRPayload,
        candidate_urls,
    )

    class FakeDecoder:
        def __init__(self) -> None:
            self.seen: list[tuple[str, int]] = []

        def decode(self, payload: bytes, part: Attachment) -> QRDecodeResult:
            self.seen.append((part.filename, len(payload)))
            return QRDecodeResult(
                attachment=part,
                outcome=QROutcome.DECODED,
                payloads=(QRPayload("https://evil-phish.com/qr", "QRCODE"),),
            )

    email = parse_email(
        build(
            (PDF, "application/pdf", "statement.pdf", None),
            (PNG, "image/png", "invoice.png", "cid1"),
        )
    )
    decoder = FakeDecoder()
    assert isinstance(decoder, QRDecoder)

    results = [
        decoder.decode(attachment_payload(email, a), a)
        for a in email.attachments
        if a.content_type in QR_IMAGE_CONTENT_TYPES
    ]

    assert decoder.seen == [("invoice.png", len(PNG))]
    extra = [url for result in results for url in result.as_extracted_urls()]
    candidates = candidate_urls(email, extra=extra)

    assert [c.raw for c in candidates] == ["https://evil-phish.com/qr"]
