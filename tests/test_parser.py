"""Unit tests for ingest/parser.py.

All fixtures are small literal byte strings defined in this file. Nothing is
downloaded, and no test opens a socket - the parser is offline by design and
these tests exercise it as such.
"""

from __future__ import annotations

import pytest

from core.models import ParsedEmail, RiskAssessment, RiskLevel, URLSource
from ingest.parser import parse_email

# --------------------------------------------------------------------------
# Fixtures - raw RFC-822 bytes
# --------------------------------------------------------------------------

SIMPLE_TEXT = b"""\
Message-ID: <simple@example.com>
From: Alice Sender <alice@example.com>
To: bob@example.org
Subject: Quarterly report
Date: Tue, 18 Aug 2026 09:00:00 +0000
Received: from mx1.example.com (mx1.example.com [192.0.2.1]) by mx2.example.org
Received: from mail.example.com (mail.example.com [192.0.2.9]) by mx1.example.com
Content-Type: text/plain; charset="utf-8"

Hello Bob,

The quarterly report is ready at https://reports.example.com/q3 today.

Alice
"""

HTML_ONLY = b"""\
Message-ID: <htmlonly@example.com>
From: "Billing" <billing@example.com>
To: victim@example.org
Subject: Invoice
Content-Type: text/html; charset="utf-8"

<html><body>
  <p>Your invoice is ready.</p>
  <a href="https://payments.example.com/invoice">View invoice</a>
  <script>var x = 1;</script>
</body></html>
"""

MULTIPART_ALTERNATIVE = b"""\
Message-ID: <alt@example.com>
From: News <news@example.com>
To: reader@example.org
Subject: Weekly digest
Content-Type: multipart/alternative; boundary="BOUND1"

--BOUND1
Content-Type: text/plain; charset="utf-8"

Plain version: https://news.example.com/story
--BOUND1
Content-Type: text/html; charset="utf-8"

<html><body><a href="https://news.example.com/story">Read the story</a></body></html>
--BOUND1--
"""

# Subject and display name are RFC 2047 encoded (base64 and quoted-printable).
ENCODED_HEADERS = b"""\
Message-ID: <encoded@example.com>
From: =?utf-8?q?Sicherheits=2DTeam?= <security@example.com>
To: =?utf-8?B?VGVzdCBVc2Vy?= <user@example.org>
Reply-To: attacker@elsewhere.test
Subject: =?utf-8?B?RHJpbmdlbmQ6IEtvbnRvIGdlc3BlcnJ0?=
Content-Type: text/plain; charset="utf-8"

Body.
"""

MULTIPLE_RECIPIENTS = b"""\
Message-ID: <many@example.com>
From: sender@example.com
To: first@example.org, "Second Person" <second@example.org>
To: third@example.org
Cc: carbon@example.org
Subject: Team update
Content-Type: text/plain; charset="utf-8"

Body.
"""

HTML_URL_SOURCES = b"""\
Message-ID: <urls@example.com>
From: sender@example.com
To: victim@example.org
Subject: Mixed URL sources
Content-Type: text/html; charset="utf-8"

<html><body>
  <a href="https://anchor.example.com/a">Click here</a>
  <a href="https://anchor.example.com/b">Click here</a>
  <img src="https://tracker.example.com/pixel.gif" alt="">
  <form action="https://harvest.example.com/post"><input name="pw"></form>
  <a href="mailto:someone@example.com">Mail us</a>
  <a href="/relative/path">Relative</a>
  <a href="#section">Fragment</a>
  <img src="cid:logo123">
</body></html>
"""

DUPLICATE_URLS = b"""\
Message-ID: <dupes@example.com>
From: sender@example.com
To: victim@example.org
Subject: Duplicates
Content-Type: text/html; charset="utf-8"

<html><body>
  <a href="https://example.com/same">First link</a>
  <a href="https://example.com/same">Second link</a>
  <form action="https://example.com/same"><input name="x"></form>
</body></html>
"""

PLAINTEXT_URLS = b"""\
Message-ID: <texturls@example.com>
From: sender@example.com
To: victim@example.org
Subject: Text URLs
Content-Type: text/plain; charset="utf-8"

Visit https://one.example.com/path now.
Also http://two.example.com/page, and see
https://en.wikipedia.org/wiki/Phishing_(email) for detail.
Not a url: example.com/nope
"""

WITH_ATTACHMENT = b"""\
Message-ID: <attach@example.com>
From: sender@example.com
To: victim@example.org
Subject: Invoice attached
Content-Type: multipart/mixed; boundary="BOUND2"

--BOUND2
Content-Type: text/plain; charset="utf-8"

See attached.
--BOUND2
Content-Type: application/pdf; name="invoice.pdf"
Content-Disposition: attachment; filename="invoice.pdf"
Content-Transfer-Encoding: base64

SGVsbG8gUERGIGJ5dGVz
--BOUND2--
"""

WITH_INLINE_IMAGE = b"""\
Message-ID: <inline@example.com>
From: sender@example.com
To: victim@example.org
Subject: Branded message
Content-Type: multipart/related; boundary="BOUND3"

--BOUND3
Content-Type: text/html; charset="utf-8"

<html><body><img src="cid:logo123"><p>Hello</p></body></html>
--BOUND3
Content-Type: image/png
Content-ID: <logo123>
Content-Disposition: inline; filename="logo.png"
Content-Transfer-Encoding: base64

iVBORw0KGgo=
--BOUND3--
"""

# Nested: mixed containing an alternative, plus an attachment.
NESTED_MIME = b"""\
Message-ID: <nested@example.com>
From: sender@example.com
To: victim@example.org
Subject: Nested structure
Content-Type: multipart/mixed; boundary="OUTER"

--OUTER
Content-Type: multipart/alternative; boundary="INNER"

--INNER
Content-Type: text/plain; charset="utf-8"

Plain inside nested structure.
--INNER
Content-Type: text/html; charset="utf-8"

<html><body><a href="https://nested.example.com/link">Nested link</a></body></html>
--INNER--
--OUTER
Content-Type: application/zip; name="archive.zip"
Content-Disposition: attachment; filename="archive.zip"

not really a zip
--OUTER--
"""

# Boundary declared in the header never appears in the body.
MALFORMED_BOUNDARY = b"""\
Message-ID: <malformed@example.com>
From: broken@example.com
To: victim@example.org
Subject: Broken multipart
Content-Type: multipart/mixed; boundary="NEVER-APPEARS"

This body claims to be multipart but has no boundary markers at all.
Visit https://broken.example.com/path
"""

NO_MESSAGE_ID = b"""\
From: sender@example.com
To: victim@example.org
Subject: No message id header
Content-Type: text/plain; charset="utf-8"

Body without a Message-ID header.
"""

NO_SENDER = b"""\
Message-ID: <nosender@example.com>
To: victim@example.org
Subject: No From header
Content-Type: text/html; charset="utf-8"

<html><body><a href="https://nosender.example.com/login">Sign in</a></body></html>
"""

EMPTY_SENDER = b"""\
Message-ID: <emptysender@example.com>
From:
To: victim@example.org
Subject: Empty From header
Content-Type: text/plain; charset="utf-8"

Body with an empty From header.
"""

UNPARSEABLE_SENDER = b"""\
Message-ID: <badsender@example.com>
From: <<<not an address at all>>>
To: victim@example.org
Subject: Unparseable From header
Content-Type: text/plain; charset="utf-8"

Body with a From header that is not an address.
"""


# --------------------------------------------------------------------------
# 1. Simple plain-text email
# --------------------------------------------------------------------------


def test_simple_plaintext_email_populates_core_fields():
    email = parse_email(SIMPLE_TEXT)

    assert isinstance(email, ParsedEmail)
    assert email.message_id == "<simple@example.com>"
    assert email.from_addr == "alice@example.com"
    assert email.from_display == "Alice Sender"
    assert email.to_addrs == ["bob@example.org"]
    assert email.subject == "Quarterly report"
    assert "The quarterly report is ready" in email.body_text
    assert email.body_html is None
    assert email.reply_to is None
    assert email.attachments == []
    assert email.raw == SIMPLE_TEXT


def test_headers_are_lowercased_and_multi_valued():
    email = parse_email(SIMPLE_TEXT)

    assert email.headers["subject"] == ["Quarterly report"]
    assert email.headers["from"] == ["Alice Sender <alice@example.com>"]
    assert len(email.headers["received"]) == 2


def test_received_chain_preserves_header_order():
    email = parse_email(SIMPLE_TEXT)

    assert len(email.received_chain) == 2
    assert email.received_chain[0].startswith("from mx1.example.com")
    assert email.received_chain[1].startswith("from mail.example.com")
    assert email.received_chain == email.headers["received"]


# --------------------------------------------------------------------------
# 2. HTML email
# --------------------------------------------------------------------------


def test_html_only_email_keeps_html_and_derives_text_fallback():
    email = parse_email(HTML_ONLY)

    assert email.body_html is not None
    assert "<a href=" in email.body_html
    # Fallback text is derived locally from the HTML.
    assert "Your invoice is ready." in email.body_text
    assert "<p>" not in email.body_text


def test_html_to_text_fallback_drops_script_contents():
    email = parse_email(HTML_ONLY)

    assert "var x = 1" not in email.body_text


# --------------------------------------------------------------------------
# 3. multipart/alternative
# --------------------------------------------------------------------------


def test_multipart_alternative_keeps_both_bodies():
    email = parse_email(MULTIPART_ALTERNATIVE)

    assert "Plain version:" in email.body_text
    assert email.body_html is not None
    assert "Read the story" in email.body_html


def test_nested_multipart_is_walked():
    email = parse_email(NESTED_MIME)

    assert "Plain inside nested structure." in email.body_text
    assert email.body_html is not None
    assert "Nested link" in email.body_html
    assert [a.filename for a in email.attachments] == ["archive.zip"]


# --------------------------------------------------------------------------
# 4 & 5. Encoded headers
# --------------------------------------------------------------------------


def test_encoded_subject_is_decoded():
    email = parse_email(ENCODED_HEADERS)

    assert email.subject == "Dringend: Konto gesperrt"


def test_encoded_display_name_is_decoded():
    email = parse_email(ENCODED_HEADERS)

    assert email.from_display == "Sicherheits-Team"
    assert email.from_addr == "security@example.com"


def test_encoded_recipient_display_name_still_yields_address():
    email = parse_email(ENCODED_HEADERS)

    assert email.to_addrs == ["user@example.org"]


# --------------------------------------------------------------------------
# 6. Reply-To
# --------------------------------------------------------------------------


def test_reply_to_is_extracted_when_present():
    email = parse_email(ENCODED_HEADERS)

    assert email.reply_to == "attacker@elsewhere.test"


def test_reply_to_is_none_when_absent():
    assert parse_email(SIMPLE_TEXT).reply_to is None


# --------------------------------------------------------------------------
# 7. Multiple recipients
# --------------------------------------------------------------------------


def test_multiple_recipients_across_repeated_headers():
    email = parse_email(MULTIPLE_RECIPIENTS)

    assert email.to_addrs == [
        "first@example.org",
        "second@example.org",
        "third@example.org",
    ]


def test_cc_is_not_folded_into_to_addrs_but_is_kept_in_headers():
    email = parse_email(MULTIPLE_RECIPIENTS)

    assert "carbon@example.org" not in email.to_addrs
    assert email.headers["cc"] == ["carbon@example.org"]


# --------------------------------------------------------------------------
# 8-11. URL extraction by source
# --------------------------------------------------------------------------


def _urls_by_source(email: ParsedEmail, source: URLSource) -> list[str]:
    return [u.url for u in email.urls if u.source is source]


def test_anchor_href_urls_are_extracted_with_anchor_text():
    email = parse_email(HTML_URL_SOURCES)

    anchors = [u for u in email.urls if u.source is URLSource.ANCHOR_HREF]
    assert [u.url for u in anchors] == [
        "https://anchor.example.com/a",
        "https://anchor.example.com/b",
    ]
    assert all(u.anchor_text == "Click here" for u in anchors)


def test_image_src_urls_are_extracted():
    email = parse_email(HTML_URL_SOURCES)

    assert _urls_by_source(email, URLSource.IMG_SRC) == [
        "https://tracker.example.com/pixel.gif"
    ]


def test_form_action_urls_are_extracted():
    email = parse_email(HTML_URL_SOURCES)

    assert _urls_by_source(email, URLSource.FORM_ACTION) == [
        "https://harvest.example.com/post"
    ]
    
def test_html_url_observations_preserve_document_order():
    raw = b"""\
Message-ID: <order@example.com>
From: sender@example.com
To: victim@example.org
Subject: Interleaved URL order
Content-Type: text/html; charset="utf-8"

<html><body>
  <img src="https://example.com/image1.png">
  <a href="https://example.com/anchor1">First</a>
  <form action="https://example.com/form1"></form>
  <a href="https://example.com/anchor2">Second</a>
  <img src="https://example.com/image2.png">
</body></html>
"""

    email = parse_email(raw)

    assert [(url.source, url.url) for url in email.urls] == [
        (URLSource.IMG_SRC, "https://example.com/image1.png"),
        (URLSource.ANCHOR_HREF, "https://example.com/anchor1"),
        (URLSource.FORM_ACTION, "https://example.com/form1"),
        (URLSource.ANCHOR_HREF, "https://example.com/anchor2"),
        (URLSource.IMG_SRC, "https://example.com/image2.png"),
    ]

def test_non_http_and_unresolvable_hrefs_are_not_extracted():
    email = parse_email(HTML_URL_SOURCES)
    urls = [u.url for u in email.urls]

    assert not any(u.startswith("mailto:") for u in urls)
    assert not any(u.startswith("cid:") for u in urls)
    assert "/relative/path" not in urls
    assert "#section" not in urls


def test_plaintext_urls_are_extracted():
    email = parse_email(PLAINTEXT_URLS)

    urls = _urls_by_source(email, URLSource.PLAIN_TEXT)
    assert "https://one.example.com/path" in urls
    assert "http://two.example.com/page" in urls


def test_plaintext_url_extraction_trims_sentence_punctuation():
    email = parse_email(PLAINTEXT_URLS)
    urls = _urls_by_source(email, URLSource.PLAIN_TEXT)

    assert "https://one.example.com/path" in urls  # trailing "." removed
    assert "http://two.example.com/page" in urls  # trailing "," removed


def test_plaintext_url_extraction_keeps_balanced_parentheses():
    email = parse_email(PLAINTEXT_URLS)

    assert "https://en.wikipedia.org/wiki/Phishing_(email)" in _urls_by_source(
        email, URLSource.PLAIN_TEXT
    )


def test_bare_hostname_without_scheme_is_not_treated_as_a_url():
    email = parse_email(PLAINTEXT_URLS)

    assert not any("nope" in u.url for u in email.urls)


def test_layer2_fields_are_left_unresolved_by_the_parser():
    for raw in (HTML_URL_SOURCES, PLAINTEXT_URLS, MULTIPART_ALTERNATIVE):
        for url in parse_email(raw).urls:
            assert url.redirect_chain == []
            assert url.final_url is None
            assert url.redirect_depth == 0


# --------------------------------------------------------------------------
# 12. Duplicate URL handling
# --------------------------------------------------------------------------


def test_identical_url_from_the_same_source_is_deduplicated():
    email = parse_email(DUPLICATE_URLS)

    anchors = _urls_by_source(email, URLSource.ANCHOR_HREF)
    assert anchors == ["https://example.com/same"]


def test_first_occurrence_anchor_text_is_the_one_kept():
    email = parse_email(DUPLICATE_URLS)

    anchor = next(u for u in email.urls if u.source is URLSource.ANCHOR_HREF)
    assert anchor.anchor_text == "First link"


def test_same_url_from_a_different_source_is_kept_as_a_separate_observation():
    email = parse_email(DUPLICATE_URLS)

    assert _urls_by_source(email, URLSource.FORM_ACTION) == ["https://example.com/same"]


def test_distinct_urls_sharing_anchor_text_are_both_kept():
    email = parse_email(HTML_URL_SOURCES)

    anchors = [u for u in email.urls if u.source is URLSource.ANCHOR_HREF]
    assert len({u.url for u in anchors}) == 2
    assert len({u.anchor_text for u in anchors}) == 1


# --------------------------------------------------------------------------
# 13. Attachment metadata
# --------------------------------------------------------------------------


def test_attachment_metadata_is_extracted():
    email = parse_email(WITH_ATTACHMENT)

    assert len(email.attachments) == 1
    attachment = email.attachments[0]
    assert attachment.filename == "invoice.pdf"
    assert attachment.content_type == "application/pdf"
    assert attachment.size_bytes == len(b"Hello PDF bytes")
    assert attachment.content_id is None
    assert attachment.is_inline is False


def test_attachment_part_is_not_mistaken_for_the_body():
    email = parse_email(WITH_ATTACHMENT)

    assert email.body_text.strip() == "See attached."


# --------------------------------------------------------------------------
# 14. CID inline image metadata
# --------------------------------------------------------------------------


def test_inline_image_metadata_is_extracted():
    email = parse_email(WITH_INLINE_IMAGE)

    assert len(email.attachments) == 1
    image = email.attachments[0]
    assert image.content_type == "image/png"
    assert image.filename == "logo.png"
    assert image.content_id == "logo123"
    assert image.is_inline is True


def test_inline_images_are_reachable_through_the_parsed_email_contract():
    email = parse_email(WITH_INLINE_IMAGE)

    assert [i.content_id for i in email.inline_images] == ["logo123"]


def test_inline_image_part_does_not_become_the_html_body():
    email = parse_email(WITH_INLINE_IMAGE)

    assert email.body_html is not None
    assert "Hello" in email.body_html


# --------------------------------------------------------------------------
# 15. Malformed input
# --------------------------------------------------------------------------


def test_malformed_multipart_boundary_does_not_raise():
    email = parse_email(MALFORMED_BOUNDARY)

    assert email.from_addr == "broken@example.com"
    assert email.subject == "Broken multipart"


def test_malformed_multipart_body_is_recovered_not_silently_dropped():
    email = parse_email(MALFORMED_BOUNDARY)

    assert "This body claims to be multipart" in email.body_text
    assert [u.url for u in email.urls] == ["https://broken.example.com/path"]


def test_wellformed_multipart_with_only_attachments_has_no_body_dump():
    raw = b"""\
Message-ID: <onlyattach@example.com>
From: sender@example.com
Subject: Attachment only
Content-Type: multipart/mixed; boundary="B"

--B
Content-Type: application/pdf; name="a.pdf"
Content-Disposition: attachment; filename="a.pdf"

payload
--B--
"""
    email = parse_email(raw)

    assert email.body_text == ""
    assert "--B" not in email.body_text
    assert [a.filename for a in email.attachments] == ["a.pdf"]


def test_input_that_is_not_an_email_at_all_does_not_raise():
    email = parse_email(b"this is not an email, just some bytes\x00\xff")

    assert email.from_addr == ""
    assert email.message_id.startswith("sha256:")


def test_empty_input_does_not_raise():
    email = parse_email(b"")

    assert email.from_addr == ""
    assert email.subject == ""
    assert email.message_id.startswith("sha256:")


def test_headers_only_message_yields_empty_body():
    raw = b"Message-ID: <hdrs@example.com>\nFrom: a@example.com\nSubject: Only headers\n\n"
    email = parse_email(raw)

    assert email.body_text == ""
    assert email.body_html is None
    assert email.urls == []


def test_non_bytes_input_is_rejected():
    with pytest.raises(TypeError):
        parse_email("already a string")


# --------------------------------------------------------------------------
# Missing / empty / unparseable sender
#
# Principle: the parser extracts what exists; detection decides whether what
# exists is suspicious. An absent sender is evidence, not a reason to reject.
# --------------------------------------------------------------------------


def test_email_without_a_from_header_parses_successfully():
    email = parse_email(NO_SENDER)

    assert isinstance(email, ParsedEmail)
    assert email.from_addr == ""
    assert email.from_display == ""


def test_email_with_an_empty_from_header_parses_successfully():
    email = parse_email(EMPTY_SENDER)

    assert email.from_addr == ""
    assert email.from_display == ""


def test_email_with_an_unparseable_from_header_parses_successfully():
    email = parse_email(UNPARSEABLE_SENDER)

    assert email.from_addr == ""
    assert email.subject == "Unparseable From header"


def test_parser_does_not_invent_a_sender():
    for raw in (NO_SENDER, EMPTY_SENDER, UNPARSEABLE_SENDER):
        email = parse_email(raw)

        assert email.from_addr == ""
        assert email.from_display == ""
        assert "@" not in email.from_addr
        assert "unknown" not in email.from_addr.lower()
        assert "invalid" not in email.from_addr.lower()


def test_other_fields_survive_a_missing_sender():
    email = parse_email(NO_SENDER)

    assert email.message_id == "<nosender@example.com>"
    assert email.to_addrs == ["victim@example.org"]
    assert email.subject == "No From header"
    assert [u.url for u in email.urls] == ["https://nosender.example.com/login"]
    assert email.urls[0].source is URLSource.ANCHOR_HREF


def test_a_present_but_unparseable_from_header_is_still_recorded():
    """The From header is recorded even when no address can be pulled from it.

    Note the stdlib address-header parser normalizes badly broken input when
    rendering it, so `headers["from"]` holds its normalized form rather than
    the original bytes. The untouched original remains on `ParsedEmail.raw`,
    which is where a detector should look if it needs byte fidelity.
    """
    email = parse_email(UNPARSEABLE_SENDER)

    assert "from" in email.headers
    assert email.from_addr == ""
    assert b"not an address at all" in email.raw


def test_a_message_with_no_from_header_records_no_from_key():
    assert "from" not in parse_email(NO_SENDER).headers


def test_email_without_a_sender_can_reach_a_later_stage():
    """A senderless ParsedEmail is a valid input to downstream consumers.

    This asserts only that the object flows on intact - it is not a detector
    and encodes no detection behaviour.
    """
    email = parse_email(NO_SENDER)

    def downstream_consumer(parsed: ParsedEmail) -> RiskAssessment:
        return RiskAssessment(
            message_id=parsed.message_id,
            score=0.0,
            level=RiskLevel.LOW,
            layers_completed=[],
        )

    assessment = downstream_consumer(email)

    assert assessment.message_id == email.message_id

# --------------------------------------------------------------------------
# 16. Deterministic Message-ID fallback
# --------------------------------------------------------------------------


def test_message_id_header_is_used_when_present():
    assert parse_email(SIMPLE_TEXT).message_id == "<simple@example.com>"


def test_missing_message_id_falls_back_to_a_content_hash():
    import hashlib

    email = parse_email(NO_MESSAGE_ID)

    assert email.message_id == f"sha256:{hashlib.sha256(NO_MESSAGE_ID).hexdigest()}"


def test_fallback_message_id_is_stable_across_repeated_parses():
    first = parse_email(NO_MESSAGE_ID).message_id
    second = parse_email(NO_MESSAGE_ID).message_id

    assert first == second
    assert first.startswith("sha256:")


def test_fallback_message_id_differs_for_different_content():
    other = NO_MESSAGE_ID.replace(b"No message id header", b"Different subject line")

    assert parse_email(NO_MESSAGE_ID).message_id != parse_email(other).message_id


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [SIMPLE_TEXT, HTML_ONLY, MULTIPART_ALTERNATIVE, NESTED_MIME, WITH_INLINE_IMAGE],
)
def test_parsing_the_same_bytes_twice_is_equivalent(raw):
    first = parse_email(raw)
    second = parse_email(raw)

    assert first == second
