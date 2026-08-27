"""Unit tests for ingest/pipeline.py.

Gmail is faked throughout. No test opens a socket, runs an OAuth flow or
touches real credentials: a fake client stands in for `GmailClient`, and the
fixtures are shaped like real `users.messages.list` / `users.messages.get`
responses.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone

import pytest

from core.models import ParsedEmail
from ingest.gmail_client import GmailClientError
from ingest.parser import EmailParseError
from ingest.pipeline import (
    Checkpoint,
    GmailIngestor,
    InMemoryCheckpoint,
    InMemorySeenMessageStore,
    IngestedMessage,
    SeenMessageStore,
)

PLAIN_EMAIL = b"""\
Message-ID: <plain@example.com>
From: Alice Sender <alice@example.com>
To: bob@example.org, carol@example.org
Subject: Quarterly report
Date: Mon, 3 Mar 2025 09:00:00 +0000
Authentication-Results: mx.google.com; spf=pass smtp.mailfrom=example.com; dkim=pass
Received-SPF: pass (google.com: domain of alice@example.com designates 1.2.3.4)
Content-Type: text/plain; charset="utf-8"

The plain text body.
"""

HTML_EMAIL = b"""\
Message-ID: <html@example.com>
From: pay@bank.example
To: victim@example.org
Subject: Verify your account
Content-Type: text/html; charset="utf-8"

<html><body><p>Please <a href="https://phish.example/login">sign in</a>.</p></body></html>
"""

MULTIPART_EMAIL = b"""\
Message-ID: <multi@example.com>
From: sender@example.com
To: recipient@example.org
Subject: Invoice attached
MIME-Version: 1.0
Content-Type: multipart/mixed; boundary="OUTER"

--OUTER
Content-Type: multipart/alternative; boundary="INNER"

--INNER
Content-Type: text/plain; charset="utf-8"

Plain alternative.
--INNER
Content-Type: text/html; charset="utf-8"

<html><body>HTML alternative with <a href="https://example.com/pay">a link</a>.</body></html>
--INNER--
--OUTER
Content-Type: application/pdf; name="invoice.pdf"
Content-Disposition: attachment; filename="invoice.pdf"
Content-Transfer-Encoding: base64

JVBERi0xLjQK
--OUTER--
"""


def _b64url(raw: bytes) -> str:
    """Encode as Gmail does: URL-safe base64 with the padding stripped."""
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _resource(
    gmail_id: str,
    raw: bytes,
    *,
    thread_id: str | None = "t-1",
    labels: list[str] | None = None,
    internal_date: str | None = "1740992400000",  # 2025-03-03T09:00:00Z
    size: int = 1024,
) -> dict:
    """A Gmail `Message` resource in `format='raw'` shape."""
    resource: dict = {"id": gmail_id, "raw": _b64url(raw), "sizeEstimate": size}
    if thread_id is not None:
        resource["threadId"] = thread_id
    if labels is not None:
        resource["labelIds"] = labels
    if internal_date is not None:
        resource["internalDate"] = internal_date
    return resource


class FakeClient:
    """Stands in for GmailClient, recording what the ingestor asked for."""

    def __init__(self, *, pages=None, resources=None, list_error=None, get_errors=None):
        self._pages = list(pages or [])
        self._resources = dict(resources or {})
        self._list_error = list_error
        self._get_errors = dict(get_errors or {})
        self.queries: list[str] = []
        self.max_results: list[int | None] = []
        self.fetched: list[str] = []

    def iter_message_refs(self, query, *, max_results=None):
        self.queries.append(query)
        self.max_results.append(max_results)
        if self._list_error is not None:
            raise self._list_error

        yielded = 0
        for page in self._pages:
            for ref in page:
                yield ref
                yielded += 1
                if max_results is not None and yielded >= max_results:
                    return

    def fetch_message_resource(self, gmail_id):
        self.fetched.append(gmail_id)
        if gmail_id in self._get_errors:
            raise self._get_errors[gmail_id]
        return self._resources[gmail_id]


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------


def test_normalizes_gmail_metadata_and_content():
    client = FakeClient(
        pages=[[{"id": "m1", "threadId": "t-9"}]],
        resources={
            "m1": _resource("m1", PLAIN_EMAIL, thread_id="t-9", labels=["INBOX", "UNREAD"])
        },
    )

    (message,) = list(GmailIngestor(client).ingest("in:inbox"))

    assert message.gmail_id == "m1"
    assert message.thread_id == "t-9"
    assert message.label_ids == ("INBOX", "UNREAD")
    assert message.size_estimate == 1024
    assert message.internal_date == datetime(2025, 3, 3, 9, 0, tzinfo=timezone.utc)

    assert message.sender == "alice@example.com"
    assert message.sender_display == "Alice Sender"
    assert message.recipients == ["bob@example.org", "carol@example.org"]
    assert message.subject == "Quarterly report"
    assert "The plain text body." in message.body_text
    assert isinstance(message.email, ParsedEmail)


def test_gmail_id_and_rfc822_message_id_are_distinct():
    """Gmail's id is not the Message-ID header; both must survive."""
    client = FakeClient(
        pages=[[{"id": "gmail-abc", "threadId": "t-1"}]],
        resources={"gmail-abc": _resource("gmail-abc", PLAIN_EMAIL)},
    )

    (message,) = list(GmailIngestor(client).ingest())

    assert message.gmail_id == "gmail-abc"
    assert message.rfc822_message_id == "<plain@example.com>"


def test_auth_headers_are_surfaced_for_layer_one():
    client = FakeClient(
        pages=[[{"id": "m1", "threadId": "t-1"}]],
        resources={"m1": _resource("m1", PLAIN_EMAIL)},
    )

    (message,) = list(GmailIngestor(client).ingest())
    auth = message.auth_headers()

    assert "spf=pass" in auth["authentication-results"][0]
    assert auth["received-spf"][0].startswith("pass")
    # Absent headers are absent keys, not empty strings.
    assert "dkim-signature" not in auth


def test_label_helpers_reflect_gmail_folders():
    client = FakeClient(
        pages=[[{"id": "s1", "threadId": "t-1"}]],
        resources={"s1": _resource("s1", HTML_EMAIL, labels=["SPAM"])},
    )

    (message,) = list(GmailIngestor(client).ingest("in:spam"))

    assert message.is_spam is True
    assert message.is_inbox is False
    assert message.is_trash is False


def test_thread_id_falls_back_to_the_listing_ref():
    client = FakeClient(
        pages=[[{"id": "m1", "threadId": "t-from-list"}]],
        resources={"m1": _resource("m1", PLAIN_EMAIL, thread_id=None)},
    )

    (message,) = list(GmailIngestor(client).ingest())

    assert message.thread_id == "t-from-list"


def test_missing_or_odd_metadata_does_not_sink_the_message():
    resource = {"id": "m1", "raw": _b64url(PLAIN_EMAIL), "internalDate": "not-a-number"}
    client = FakeClient(pages=[[{"id": "m1"}]], resources={"m1": resource})

    (message,) = list(GmailIngestor(client).ingest())

    assert message.internal_date is None
    assert message.label_ids == ()
    assert message.size_estimate == 0
    assert message.subject == "Quarterly report"


def test_ingested_message_rejects_a_non_parsed_email():
    with pytest.raises(TypeError):
        IngestedMessage(gmail_id="m1", thread_id="t", email="not an email")


def test_ingested_message_requires_a_gmail_id():
    parsed = ParsedEmail(message_id="<x@example.com>", from_addr="a@example.com")
    with pytest.raises(ValueError):
        IngestedMessage(gmail_id="  ", thread_id="t", email=parsed)


# --------------------------------------------------------------------------
# MIME / body extraction
# --------------------------------------------------------------------------


def test_plain_text_message_has_no_html_body():
    client = FakeClient(
        pages=[[{"id": "m1"}]], resources={"m1": _resource("m1", PLAIN_EMAIL)}
    )

    (message,) = list(GmailIngestor(client).ingest())

    assert message.body_text.strip() == "The plain text body."
    assert message.body_html is None
    assert message.attachments == []


def test_html_only_message_yields_html_and_a_text_rendering():
    client = FakeClient(
        pages=[[{"id": "m1"}]], resources={"m1": _resource("m1", HTML_EMAIL)}
    )

    (message,) = list(GmailIngestor(client).ingest())

    assert message.body_html is not None
    assert "phish.example" in message.body_html
    assert "sign in" in message.body_text
    assert any(u.url == "https://phish.example/login" for u in message.email.urls)


def test_nested_multipart_yields_both_bodies_and_attachment_metadata():
    client = FakeClient(
        pages=[[{"id": "m1"}]], resources={"m1": _resource("m1", MULTIPART_EMAIL)}
    )

    (message,) = list(GmailIngestor(client).ingest())

    assert "Plain alternative." in message.body_text
    assert message.body_html is not None and "HTML alternative" in message.body_html

    (attachment,) = message.attachments
    assert attachment.filename == "invoice.pdf"
    assert attachment.content_type == "application/pdf"
    assert attachment.is_inline is False


# --------------------------------------------------------------------------
# Pagination
# --------------------------------------------------------------------------


def test_messages_from_every_page_are_ingested_in_order():
    pages = [
        [{"id": "m1"}, {"id": "m2"}],
        [{"id": "m3"}],
    ]
    resources = {i: _resource(i, PLAIN_EMAIL) for i in ("m1", "m2", "m3")}
    client = FakeClient(pages=pages, resources=resources)

    ingested = list(GmailIngestor(client).ingest("in:anywhere"))

    assert [m.gmail_id for m in ingested] == ["m1", "m2", "m3"]


def test_max_results_is_passed_through_to_the_listing():
    client = FakeClient(
        pages=[[{"id": "m1"}, {"id": "m2"}]],
        resources={i: _resource(i, PLAIN_EMAIL) for i in ("m1", "m2")},
    )

    ingested = list(GmailIngestor(client).ingest("in:inbox", max_results=1))

    assert [m.gmail_id for m in ingested] == ["m1"]
    assert client.max_results == [1]


def test_ingest_is_lazy_and_stops_fetching_when_the_caller_stops():
    """Taking one message must not fetch the rest of the mailbox."""
    pages = [[{"id": "m1"}, {"id": "m2"}, {"id": "m3"}]]
    resources = {i: _resource(i, PLAIN_EMAIL) for i in ("m1", "m2", "m3")}
    client = FakeClient(pages=pages, resources=resources)

    stream = GmailIngestor(client).ingest()
    next(stream)

    assert client.fetched == ["m1"]


def test_a_listing_failure_propagates_rather_than_looking_like_an_empty_mailbox():
    client = FakeClient(pages=[], list_error=GmailClientError("could not list"))

    with pytest.raises(GmailClientError):
        list(GmailIngestor(client).ingest())


# --------------------------------------------------------------------------
# Deduplication
# --------------------------------------------------------------------------


def test_an_id_repeated_across_pages_is_ingested_once():
    pages = [[{"id": "m1"}, {"id": "m2"}], [{"id": "m1"}]]
    resources = {i: _resource(i, PLAIN_EMAIL) for i in ("m1", "m2")}
    client = FakeClient(pages=pages, resources=resources)

    result = GmailIngestor(client).ingest_all()

    assert [m.gmail_id for m in result.messages] == ["m1", "m2"]
    assert result.skipped_duplicates == 1
    assert client.fetched == ["m1", "m2"]


def test_a_previously_seen_message_is_never_fetched_again():
    client = FakeClient(
        pages=[[{"id": "m1"}, {"id": "m2"}]],
        resources={i: _resource(i, PLAIN_EMAIL) for i in ("m1", "m2")},
    )
    store = InMemorySeenMessageStore({"m1"})

    result = GmailIngestor(client, seen_store=store).ingest_all()

    assert [m.gmail_id for m in result.messages] == ["m2"]
    assert client.fetched == ["m2"]  # no round trip spent on the duplicate
    assert result.skipped_duplicates == 1


def test_the_seen_store_persists_across_runs_of_the_same_ingestor():
    client = FakeClient(
        pages=[[{"id": "m1"}]], resources={"m1": _resource("m1", PLAIN_EMAIL)}
    )
    ingestor = GmailIngestor(client)

    assert len(list(ingestor.ingest())) == 1
    assert list(ingestor.ingest()) == []
    assert client.fetched == ["m1"]


def test_in_memory_seen_store_round_trips():
    store = InMemorySeenMessageStore()
    assert store.has_seen("m1") is False
    store.mark_seen("m1")
    assert store.has_seen("m1") is True
    assert len(store) == 1
    assert isinstance(store, SeenMessageStore)


# --------------------------------------------------------------------------
# Incremental ingestion
# --------------------------------------------------------------------------


def test_the_first_run_does_not_narrow_the_query():
    client = FakeClient(
        pages=[[{"id": "m1"}]], resources={"m1": _resource("m1", PLAIN_EMAIL)}
    )

    list(GmailIngestor(client, checkpoint=InMemoryCheckpoint()).ingest("in:inbox"))

    assert client.queries == ["in:inbox"]


def test_the_checkpoint_advances_to_the_newest_message_seen():
    resources = {
        "m1": _resource("m1", PLAIN_EMAIL, internal_date="1000000"),
        "m2": _resource("m2", PLAIN_EMAIL, internal_date="3000000"),
        "m3": _resource("m3", PLAIN_EMAIL, internal_date="2000000"),
    }
    client = FakeClient(pages=[[{"id": i} for i in resources]], resources=resources)
    checkpoint = InMemoryCheckpoint()

    list(GmailIngestor(client, checkpoint=checkpoint).ingest())

    assert checkpoint.load() == 3000  # epoch milliseconds -> seconds


def test_a_later_run_narrows_the_query_with_the_checkpoint():
    client = FakeClient(pages=[[]], resources={})
    checkpoint = InMemoryCheckpoint(1740992400)

    list(GmailIngestor(client, checkpoint=checkpoint).ingest("in:inbox"))

    assert client.queries == ["in:inbox after:1740992400"]


def test_the_checkpoint_bound_is_the_whole_query_when_none_was_given():
    client = FakeClient(pages=[[]], resources={})

    list(GmailIngestor(client, checkpoint=InMemoryCheckpoint(500)).ingest(""))

    assert client.queries == ["after:500"]


def test_the_checkpoint_never_moves_backwards():
    checkpoint = InMemoryCheckpoint(5000)
    checkpoint.save(100)
    assert checkpoint.load() == 5000
    checkpoint.save(9000)
    assert checkpoint.load() == 9000
    assert isinstance(checkpoint, Checkpoint)


def test_a_run_with_no_datable_messages_leaves_the_checkpoint_alone():
    resource = {"id": "m1", "raw": _b64url(PLAIN_EMAIL)}  # no internalDate
    client = FakeClient(pages=[[{"id": "m1"}]], resources={"m1": resource})
    checkpoint = InMemoryCheckpoint(4242)

    list(GmailIngestor(client, checkpoint=checkpoint).ingest())

    assert checkpoint.load() == 4242


# --------------------------------------------------------------------------
# Error handling
# --------------------------------------------------------------------------


def test_a_fetch_failure_is_recorded_and_the_run_continues():
    client = FakeClient(
        pages=[[{"id": "bad"}, {"id": "good"}]],
        resources={"good": _resource("good", PLAIN_EMAIL)},
        get_errors={"bad": GmailClientError("could not fetch Gmail message bad")},
    )

    result = GmailIngestor(client).ingest_all()

    assert [m.gmail_id for m in result.messages] == ["good"]
    (failure,) = result.failures
    assert failure.gmail_id == "bad"
    assert failure.reason == "the message could not be fetched from Gmail"
    assert isinstance(failure.error, GmailClientError)


def test_an_unparseable_message_is_recorded_and_the_run_continues():
    client = FakeClient(
        pages=[[{"id": "bad"}, {"id": "good"}]],
        resources={"good": _resource("good", PLAIN_EMAIL)},
        get_errors={"bad": EmailParseError("not a message")},
    )

    result = GmailIngestor(client).ingest_all()

    assert [m.gmail_id for m in result.messages] == ["good"]
    assert result.failures[0].reason == "the message could not be parsed"


def test_a_resource_without_a_raw_payload_is_a_failure_not_a_crash():
    client = FakeClient(
        pages=[[{"id": "m1"}]], resources={"m1": {"id": "m1", "threadId": "t-1"}}
    )

    result = GmailIngestor(client).ingest_all()

    assert result.messages == []
    assert result.failures[0].gmail_id == "m1"


def test_a_corrupt_base64_payload_is_a_failure_not_a_crash():
    client = FakeClient(
        pages=[[{"id": "m1"}]],
        resources={"m1": {"id": "m1", "raw": "!!! not base64 !!!"}},
    )

    result = GmailIngestor(client).ingest_all()

    assert result.messages == []
    assert result.failures[0].reason == "the message could not be fetched from Gmail"


def test_a_failed_message_is_not_retried_within_the_same_ingestor():
    client = FakeClient(
        pages=[[{"id": "bad"}]],
        resources={},
        get_errors={"bad": EmailParseError("not a message")},
    )
    ingestor = GmailIngestor(client)

    ingestor.ingest_all()
    ingestor.ingest_all()

    assert client.fetched == ["bad"]


def test_failures_are_dropped_silently_by_the_streaming_form():
    client = FakeClient(
        pages=[[{"id": "bad"}, {"id": "good"}]],
        resources={"good": _resource("good", PLAIN_EMAIL)},
        get_errors={"bad": GmailClientError("nope")},
    )

    ingested = list(GmailIngestor(client).ingest())

    assert [m.gmail_id for m in ingested] == ["good"]


def test_failure_records_never_carry_message_content():
    client = FakeClient(
        pages=[[{"id": "m1"}]],
        resources={},
        get_errors={"m1": GmailClientError("could not fetch Gmail message m1")},
    )

    (failure,) = GmailIngestor(client).ingest_all().failures

    assert "Quarterly report" not in failure.reason
    assert "alice@example.com" not in failure.reason


def test_result_counts_summarize_the_run():
    client = FakeClient(
        pages=[[{"id": "m1"}, {"id": "m1"}, {"id": "bad"}]],
        resources={"m1": _resource("m1", PLAIN_EMAIL)},
        get_errors={"bad": EmailParseError("nope")},
    )

    result = GmailIngestor(client).ingest_all()

    assert result.ingested_count == 1
    assert result.skipped_duplicates == 1
    assert len(result.failures) == 1
