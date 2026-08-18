"""Unit tests for ingest/gmail_client.py.

Every Gmail interaction is faked. No test opens a socket, runs a real OAuth
flow, launches a browser, or reads real credentials: the Gmail service is
injected, and the auth helpers are monkeypatched at the module boundary.
"""

from __future__ import annotations

import base64
import json

import pytest

from core.models import ParsedEmail
from ingest import gmail_client as gmail_module
from ingest.gmail_client import GmailClient, GmailClientError
from ingest.parser import EmailParseError

SAMPLE_EMAIL = b"""\
Message-ID: <gmail-sample@example.com>
From: Alice Sender <alice@example.com>
To: bob@example.org
Subject: Fetched through Gmail
Content-Type: text/plain; charset="utf-8"

Body retrieved via the Gmail API.
"""


def _b64url(raw: bytes) -> str:
    """Encode as Gmail does: URL-safe base64 with the padding stripped."""
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


class FakeExecutable:
    """Stands in for a Google API request object."""

    def __init__(self, result=None, error: Exception | None = None):
        self._result = result
        self._error = error

    def execute(self):
        if self._error is not None:
            raise self._error
        return self._result


class FakeMessages:
    """Records the arguments Gmail was called with, and replays canned results."""

    def __init__(self, *, get_result=None, get_error=None, list_pages=None, list_error=None):
        self._get_result = get_result
        self._get_error = get_error
        self._list_pages = list(list_pages or [])
        self._list_error = list_error
        self.get_calls: list[dict] = []
        self.list_calls: list[dict] = []

    def get(self, **kwargs):
        self.get_calls.append(kwargs)
        return FakeExecutable(result=self._get_result, error=self._get_error)

    def list(self, **kwargs):
        self.list_calls.append(kwargs)
        if self._list_error is not None:
            return FakeExecutable(error=self._list_error)
        page = self._list_pages.pop(0) if self._list_pages else {}
        return FakeExecutable(result=page)


class FakeService:
    """Minimal stand-in for the Gmail discovery service."""

    def __init__(self, messages: FakeMessages):
        self._messages = messages

    def users(self):
        return self

    def messages(self):
        return self._messages


def _client(messages: FakeMessages, **kwargs) -> GmailClient:
    return GmailClient(service=FakeService(messages), **kwargs)


# --------------------------------------------------------------------------
# 1-3. Raw decoding, handoff to the parser, and the returned ParsedEmail
# --------------------------------------------------------------------------


def test_gmail_raw_payload_is_decoded_to_the_original_bytes():
    messages = FakeMessages(get_result={"raw": _b64url(SAMPLE_EMAIL)})

    assert _client(messages).fetch_raw("m1") == SAMPLE_EMAIL


def test_unpadded_base64url_payload_is_decoded_correctly():
    """Gmail strips base64 padding; the client must restore it."""
    for body in (b"a", b"ab", b"abc", b"abcd"):
        raw = b"From: a@b.c\r\n\r\n" + body
        encoded = _b64url(raw)
        assert "=" not in encoded
        messages = FakeMessages(get_result={"raw": encoded})
        assert _client(messages).fetch_raw("m1") == raw


def test_base64url_specific_alphabet_is_handled():
    """`-` and `_` must decode, i.e. standard base64 alone would be wrong."""
    raw = bytes([0xFB, 0xFF, 0xBF]) + b" From: a@b.c"
    encoded = _b64url(raw)
    assert "-" in encoded or "_" in encoded
    messages = FakeMessages(get_result={"raw": encoded})

    assert _client(messages).fetch_raw("m1") == raw


def test_decoded_bytes_are_passed_unchanged_to_the_parser(monkeypatch):
    seen: dict[str, object] = {}

    def fake_parse_email(raw):
        seen["raw"] = raw
        return ParsedEmail(message_id="<stub@example.com>", from_addr="stub@example.com")

    monkeypatch.setattr(gmail_module, "parse_email", fake_parse_email)
    messages = FakeMessages(get_result={"raw": _b64url(SAMPLE_EMAIL)})

    _client(messages).fetch_message("m1")

    assert seen["raw"] == SAMPLE_EMAIL
    assert isinstance(seen["raw"], bytes), "parser must receive bytes, never str"


def test_parsed_email_from_the_parser_is_returned_to_the_caller():
    messages = FakeMessages(get_result={"raw": _b64url(SAMPLE_EMAIL)})

    email = _client(messages).fetch_message("m1")

    assert isinstance(email, ParsedEmail)
    assert email.message_id == "<gmail-sample@example.com>"
    assert email.from_addr == "alice@example.com"
    assert email.subject == "Fetched through Gmail"


def test_client_does_not_reimplement_parsing():
    """The client's only job is transport; parsed detail comes from the parser."""
    messages = FakeMessages(get_result={"raw": _b64url(SAMPLE_EMAIL)})

    email = _client(messages).fetch_message("m1")

    assert email.raw == SAMPLE_EMAIL
    assert email.to_addrs == ["bob@example.org"]


# --------------------------------------------------------------------------
# 4-5. Request arguments
# --------------------------------------------------------------------------


def test_message_id_is_passed_through_to_messages_get():
    messages = FakeMessages(get_result={"raw": _b64url(SAMPLE_EMAIL)})

    _client(messages).fetch_raw("abc123")

    assert messages.get_calls[0]["id"] == "abc123"
    assert messages.get_calls[0]["userId"] == "me"


def test_format_raw_is_used_explicitly():
    messages = FakeMessages(get_result={"raw": _b64url(SAMPLE_EMAIL)})

    _client(messages).fetch_raw("m1")

    assert messages.get_calls[0]["format"] == "raw"


def test_user_id_is_configurable():
    messages = FakeMessages(get_result={"raw": _b64url(SAMPLE_EMAIL)})

    _client(messages, user_id="someone@example.com").fetch_raw("m1")

    assert messages.get_calls[0]["userId"] == "someone@example.com"


# --------------------------------------------------------------------------
# 6-7. Listing and pagination
# --------------------------------------------------------------------------


def test_list_query_is_passed_through_unchanged():
    messages = FakeMessages(list_pages=[{"messages": [{"id": "a"}]}])

    _client(messages).list_message_ids("in:inbox newer_than:7d")

    assert messages.list_calls[0]["q"] == "in:inbox newer_than:7d"
    assert messages.list_calls[0]["userId"] == "me"


def test_listing_follows_pagination_across_pages():
    messages = FakeMessages(
        list_pages=[
            {"messages": [{"id": "a"}, {"id": "b"}], "nextPageToken": "T1"},
            {"messages": [{"id": "c"}], "nextPageToken": "T2"},
            {"messages": [{"id": "d"}]},
        ]
    )

    assert _client(messages).list_message_ids("in:inbox") == ["a", "b", "c", "d"]
    assert len(messages.list_calls) == 3


def test_page_token_is_sent_on_subsequent_requests_only():
    messages = FakeMessages(
        list_pages=[
            {"messages": [{"id": "a"}], "nextPageToken": "T1"},
            {"messages": [{"id": "b"}]},
        ]
    )

    _client(messages).list_message_ids("in:inbox")

    assert "pageToken" not in messages.list_calls[0]
    assert messages.list_calls[1]["pageToken"] == "T1"


def test_listing_stops_at_max_results_without_fetching_more_pages():
    messages = FakeMessages(
        list_pages=[
            {"messages": [{"id": "a"}, {"id": "b"}], "nextPageToken": "T1"},
            {"messages": [{"id": "c"}]},
        ]
    )

    assert _client(messages).list_message_ids("in:inbox", max_results=2) == ["a", "b"]
    assert len(messages.list_calls) == 1


def test_listing_handles_an_empty_result_set():
    messages = FakeMessages(list_pages=[{}])

    assert _client(messages).list_message_ids("in:inbox label:nothing") == []


def test_listing_does_not_request_message_bodies():
    """Listing must stay cheap: ids only, no format/payload requests."""
    messages = FakeMessages(list_pages=[{"messages": [{"id": "a"}]}])

    _client(messages).list_message_ids("in:inbox")

    assert "format" not in messages.list_calls[0]
    assert messages.get_calls == []


def test_non_positive_max_results_returns_nothing_without_calling_gmail():
    messages = FakeMessages(list_pages=[{"messages": [{"id": "a"}]}])

    assert _client(messages).list_message_ids("in:inbox", max_results=0) == []
    assert messages.list_calls == []


# --------------------------------------------------------------------------
# 8-9. Error boundaries
# --------------------------------------------------------------------------


def test_gmail_get_failure_becomes_a_client_error():
    messages = FakeMessages(get_error=RuntimeError("HTTP 503 backend error"))

    with pytest.raises(GmailClientError) as excinfo:
        _client(messages).fetch_raw("m1")

    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_gmail_list_failure_becomes_a_client_error():
    messages = FakeMessages(list_error=RuntimeError("HTTP 429 rate limited"))

    with pytest.raises(GmailClientError):
        _client(messages).list_message_ids("in:inbox")


def test_missing_raw_field_becomes_a_client_error():
    messages = FakeMessages(get_result={"id": "m1"})

    with pytest.raises(GmailClientError, match="raw payload"):
        _client(messages).fetch_raw("m1")


def test_undecodable_payload_becomes_a_client_error_not_a_parser_error():
    messages = FakeMessages(get_result={"raw": "!!!not base64!!!"})

    with pytest.raises(GmailClientError, match="base64url"):
        _client(messages).fetch_raw("m1")


def test_parser_failure_is_not_swallowed_or_converted(monkeypatch):
    """A malformed message must not surface as a transport error, nor succeed."""

    def exploding_parser(raw):
        raise EmailParseError("input could not be parsed as an email")

    monkeypatch.setattr(gmail_module, "parse_email", exploding_parser)
    messages = FakeMessages(get_result={"raw": _b64url(SAMPLE_EMAIL)})

    with pytest.raises(EmailParseError):
        _client(messages).fetch_message("m1")


def test_transport_failure_and_parser_failure_are_distinguishable():
    transport = FakeMessages(get_error=RuntimeError("network down"))
    with pytest.raises(GmailClientError):
        _client(transport).fetch_raw("m1")

    assert not issubclass(EmailParseError, GmailClientError)
    assert not issubclass(GmailClientError, EmailParseError)


# --------------------------------------------------------------------------
# 10. Errors must not leak credential material
# --------------------------------------------------------------------------


SECRET = "ya29.SUPER-SECRET-REFRESH-TOKEN-VALUE"


def test_unreadable_token_file_error_reports_the_path_not_the_contents(tmp_path):
    token_file = tmp_path / "token.json"
    token_file.write_text(json.dumps({"refresh_token": SECRET, "malformed": True}))

    client = GmailClient(token_file=token_file, client_secrets_file=tmp_path / "cs.json")

    with pytest.raises(GmailClientError) as excinfo:
        client._load_cached_credentials()

    message = str(excinfo.value)
    assert SECRET not in message
    assert str(token_file) in message


def test_missing_client_secrets_error_names_the_path_only(tmp_path):
    client = GmailClient(
        token_file=tmp_path / "token.json",
        client_secrets_file=tmp_path / "absent-client-secret.json",
    )

    with pytest.raises(GmailClientError) as excinfo:
        client._run_installed_app_flow()

    assert "absent-client-secret.json" in str(excinfo.value)
    assert SECRET not in str(excinfo.value)


# --------------------------------------------------------------------------
# 11-13. Credential lifecycle
# --------------------------------------------------------------------------


class FakeCredentials:
    def __init__(self, *, valid=True, expired=False, refresh_token=None):
        self.valid = valid
        self.expired = expired
        self.refresh_token = refresh_token
        self.refreshed = False

    def refresh(self, _request):
        self.refreshed = True
        self.valid = True
        self.expired = False

    def to_json(self):
        return json.dumps({"token": "fake"})


def _stub_credentials_loader(monkeypatch, credentials):
    """Make Credentials.from_authorized_user_file return a canned object."""
    monkeypatch.setattr(
        gmail_module.Credentials,
        "from_authorized_user_file",
        staticmethod(lambda *a, **k: credentials),
    )


def test_valid_cached_token_is_reused_without_refresh_or_oauth(monkeypatch, tmp_path):
    token_file = tmp_path / "token.json"
    token_file.write_text("{}")
    cached = FakeCredentials(valid=True)
    _stub_credentials_loader(monkeypatch, cached)

    def fail_flow(self):
        raise AssertionError("OAuth flow must not run when a valid token exists")

    monkeypatch.setattr(GmailClient, "_run_installed_app_flow", fail_flow)
    client = GmailClient(token_file=token_file)

    assert client._credentials() is cached
    assert cached.refreshed is False
    # A reused token is not rewritten.
    assert token_file.read_text() == "{}"


def test_expired_refreshable_token_is_refreshed_and_saved(monkeypatch, tmp_path):
    token_file = tmp_path / "token.json"
    token_file.write_text("{}")
    cached = FakeCredentials(valid=False, expired=True, refresh_token="refresh-me")
    _stub_credentials_loader(monkeypatch, cached)
    monkeypatch.setattr(gmail_module, "Request", lambda *a, **k: object())

    def fail_flow(self):
        raise AssertionError("OAuth flow must not run when the token can be refreshed")

    monkeypatch.setattr(GmailClient, "_run_installed_app_flow", fail_flow)

    credentials = GmailClient(token_file=token_file)._credentials()

    assert credentials.refreshed is True
    assert json.loads(token_file.read_text()) == {"token": "fake"}


def test_missing_credentials_trigger_the_oauth_flow(monkeypatch, tmp_path):
    token_file = tmp_path / "absent-token.json"
    fresh = FakeCredentials(valid=True)
    calls: list[str] = []

    def fake_flow(self):
        calls.append("flow")
        return fresh

    monkeypatch.setattr(GmailClient, "_run_installed_app_flow", fake_flow)

    credentials = GmailClient(token_file=token_file)._credentials()

    assert calls == ["flow"]
    assert credentials is fresh
    assert token_file.exists(), "the new token should be cached"


def test_expired_token_without_a_refresh_token_falls_back_to_the_oauth_flow(
    monkeypatch, tmp_path
):
    token_file = tmp_path / "token.json"
    token_file.write_text("{}")
    _stub_credentials_loader(
        monkeypatch, FakeCredentials(valid=False, expired=True, refresh_token=None)
    )
    fresh = FakeCredentials(valid=True)
    monkeypatch.setattr(GmailClient, "_run_installed_app_flow", lambda self: fresh)

    assert GmailClient(token_file=token_file)._credentials() is fresh


def test_refresh_failure_becomes_a_client_error(monkeypatch, tmp_path):
    token_file = tmp_path / "token.json"
    token_file.write_text("{}")

    class RevokedCredentials(FakeCredentials):
        def refresh(self, _request):
            raise RuntimeError("token has been revoked")

    _stub_credentials_loader(
        monkeypatch, RevokedCredentials(valid=False, expired=True, refresh_token="r")
    )
    monkeypatch.setattr(gmail_module, "Request", lambda *a, **k: object())

    with pytest.raises(GmailClientError, match="refreshed"):
        GmailClient(token_file=token_file)._credentials()


def test_saved_token_file_is_not_world_readable(monkeypatch, tmp_path):
    token_file = tmp_path / "nested" / "token.json"
    monkeypatch.setattr(
        GmailClient, "_run_installed_app_flow", lambda self: FakeCredentials(valid=True)
    )

    GmailClient(token_file=token_file)._credentials()

    assert token_file.stat().st_mode & 0o077 == 0


# --------------------------------------------------------------------------
# Configuration and construction
# --------------------------------------------------------------------------


def test_paths_default_to_the_gitignored_credentials_directory(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLIENT_SECRETS_FILE", raising=False)
    monkeypatch.delenv("GMAIL_TOKEN_FILE", raising=False)

    client = GmailClient()

    assert str(client.token_file) == ".credentials/token.json"
    assert str(client.client_secrets_file) == ".credentials/client_secret.json"


def test_paths_are_configurable_through_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("GOOGLE_CLIENT_SECRETS_FILE", str(tmp_path / "cs.json"))
    monkeypatch.setenv("GMAIL_TOKEN_FILE", str(tmp_path / "tok.json"))

    client = GmailClient()

    assert client.token_file == tmp_path / "tok.json"
    assert client.client_secrets_file == tmp_path / "cs.json"


def test_constructing_a_client_performs_no_authentication(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("constructing a GmailClient must not authenticate")

    monkeypatch.setattr(GmailClient, "_credentials", fail)
    monkeypatch.setattr(gmail_module, "build", fail)

    GmailClient()  # must not raise


def test_only_the_readonly_scope_is_requested():
    assert gmail_module.SCOPES == ["https://www.googleapis.com/auth/gmail.readonly"]
