"""Gmail transport: raw RFC-822 bytes out of Gmail and into the parser.

    Gmail API -> base64url raw -> bytes -> parse_email() -> ParsedEmail

This module owns transport and authentication only. It does not parse mail:
`ingest.parser` owns MIME, headers, bodies, URLs and attachments, and the sole
handoff between them is a `bytes` object. Nothing here inspects message
content beyond decoding the payload Gmail returned.

Per ARCHITECTURE.md section 3:

- OAuth 2.0 installed-app flow, scope `gmail.readonly` **only**. The scope is
  a Google *restricted* scope; this project stays in Testing mode permanently
  and is never submitted for verification.
- Messages are always fetched with ``format='raw'``. ``format='full'`` loses
  fidelity and complicates MIME walking, so it is not used.
- The cached token lives under `.credentials/`, which is gitignored.

Security
--------
Gmail content is untrusted. This module never logs or prints tokens, refresh
tokens, client secrets, message bodies or attachment content, and it never
writes a fetched message to disk. Error messages carry file *paths* and Gmail
message ids, never file contents or credential material.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from core.models import ParsedEmail
from ingest.parser import parse_email

__all__ = ["GmailClient", "GmailClientError", "GMAIL_READONLY_SCOPE"]

# Read-only, and deliberately the only scope this project ever requests.
GMAIL_READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
SCOPES = [GMAIL_READONLY_SCOPE]

DEFAULT_CLIENT_SECRETS_FILE = ".credentials/client_secret.json"
DEFAULT_TOKEN_FILE = ".credentials/token.json"


class GmailClientError(RuntimeError):
    """A Gmail transport or authentication failure.

    Raised at the client boundary so callers can tell "Gmail did not give us
    the message" apart from "the message was malformed". Parser failures are
    deliberately *not* wrapped in this: `ingest.parser.EmailParseError`
    propagates untouched.
    """


def _decode_raw(raw: str | bytes) -> bytes:
    """Decode Gmail's base64url `raw` field into RFC-822 bytes.

    Gmail returns URL-safe base64 and omits padding, so padding is restored
    before decoding. Decoding is strict: `validate=True` rejects any character
    outside the base64url alphabet instead of silently discarding it, so a
    corrupted payload fails loudly rather than yielding truncated mail.

    The result stays as bytes: it is handed to the parser without ever
    becoming a str, so no charset guess is made here.
    """
    try:
        # Encoding lives inside the boundary: a non-ASCII payload is a
        # malformed payload, not an unhandled UnicodeEncodeError.
        if isinstance(raw, str):
            raw = raw.encode("ascii", errors="strict")
        padding = b"=" * (-len(raw) % 4)
        return base64.b64decode(raw + padding, altchars=b"-_", validate=True)
    except Exception as exc:  # noqa: BLE001 - malformed payload from Gmail
        raise GmailClientError("Gmail returned a raw payload that is not valid base64url") from exc


class GmailClient:
    """A small read-only Gmail client.

    Pass `service` to inject an already-built Gmail API service; that is the
    only seam the tests need, and it keeps real credentials out of them.
    Otherwise the service is built lazily on first use, so constructing a
    client never triggers an OAuth flow.
    """

    def __init__(
        self,
        *,
        client_secrets_file: str | os.PathLike[str] | None = None,
        token_file: str | os.PathLike[str] | None = None,
        service: object | None = None,
        user_id: str = "me",
    ) -> None:
        self.client_secrets_file = Path(
            client_secrets_file
            or os.environ.get("GOOGLE_CLIENT_SECRETS_FILE", DEFAULT_CLIENT_SECRETS_FILE)
        )
        self.token_file = Path(
            token_file or os.environ.get("GMAIL_TOKEN_FILE", DEFAULT_TOKEN_FILE)
        )
        self.user_id = user_id
        self._service = service

    # ---------------------------------------------------------------- auth

    def _load_cached_credentials(self) -> Credentials | None:
        """Read the cached token, or None when there is not a usable one."""
        if not self.token_file.exists():
            return None
        try:
            return Credentials.from_authorized_user_file(str(self.token_file), SCOPES)
        except Exception as exc:  # noqa: BLE001 - corrupt or foreign token file
            # Path only. The file's contents are credential material.
            raise GmailClientError(
                f"cached credentials at {self.token_file} could not be read"
            ) from exc

    def _run_installed_app_flow(self) -> Credentials:
        """Run the standard installed-app OAuth flow on a loopback port.

        This is google-auth-oauthlib's own local redirect listener, not a web
        server this project implements or exposes.
        """
        if not self.client_secrets_file.exists():
            raise GmailClientError(
                f"OAuth client secrets file not found at {self.client_secrets_file}; "
                "download it from the Google Cloud console and place it there"
            )
        try:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(self.client_secrets_file), SCOPES
            )
            return flow.run_local_server(port=0)
        except Exception as exc:  # noqa: BLE001 - user cancelled, bad config
            raise GmailClientError("the Gmail OAuth flow did not complete") from exc

    def _save_credentials(self, credentials: Credentials) -> None:
        """Cache the token in the gitignored credentials directory.

        The file is created 0600 by `os.open`, never written world-readable
        and tightened afterwards - there is no window in which the token sits
        on disk with default permissions. An already-existing file is
        tightened *before* it is rewritten, closing the same window on the
        overwrite path. The containing directory is 0700.
        """
        try:
            parent = self.token_file.parent
            parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            parent.chmod(0o700)  # a pre-existing directory may be looser

            if self.token_file.exists():
                self.token_file.chmod(0o600)

            descriptor = os.open(
                self.token_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
            )
            with os.fdopen(descriptor, "w") as handle:
                handle.write(credentials.to_json())

            self.token_file.chmod(0o600)
        except Exception as exc:  # noqa: BLE001
            raise GmailClientError(
                f"could not write the Gmail token to {self.token_file}"
            ) from exc

    def _credentials(self) -> Credentials:
        """Reuse, refresh, or obtain credentials, in that order of preference."""
        credentials = self._load_cached_credentials()

        if credentials is not None and credentials.valid:
            return credentials

        if (
            credentials is not None
            and credentials.expired
            and credentials.refresh_token
        ):
            try:
                credentials.refresh(Request())
            except Exception as exc:  # noqa: BLE001 - revoked or offline
                raise GmailClientError(
                    "the cached Gmail token could not be refreshed"
                ) from exc
        else:
            credentials = self._run_installed_app_flow()

        self._save_credentials(credentials)
        return credentials

    def service(self) -> object:
        """The Gmail API service, built on first use and reused thereafter."""
        if self._service is None:
            try:
                self._service = build(
                    "gmail", "v1", credentials=self._credentials(), cache_discovery=False
                )
            except GmailClientError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise GmailClientError("could not build the Gmail API service") from exc
        return self._service

    # ------------------------------------------------------------- fetching

    def _messages(self):  # noqa: ANN202 - Google's resource objects are untyped
        return self.service().users().messages()

    def fetch_raw(self, message_id: str) -> bytes:
        """Raw RFC-822 bytes for one message, via ``format='raw'``."""
        try:
            message = (
                self._messages()
                .get(userId=self.user_id, id=message_id, format="raw")
                .execute()
            )
        except GmailClientError:
            raise
        except Exception as exc:  # noqa: BLE001 - HttpError and transport faults
            raise GmailClientError(f"could not fetch Gmail message {message_id}") from exc

        raw = message.get("raw") if isinstance(message, dict) else None
        if not raw:
            raise GmailClientError(
                f"Gmail message {message_id} came back without a raw payload"
            )
        return _decode_raw(raw)

    def fetch_message(self, message_id: str) -> ParsedEmail:
        """Fetch one message and hand its bytes to the parser.

        Parsing happens outside the transport error boundary on purpose: a
        malformed message raises the parser's own error, so a bad email is
        never mistaken for a Gmail outage, and a Gmail outage is never
        mistaken for a bad email.
        """
        raw = self.fetch_raw(message_id)
        return parse_email(raw)

    # ------------------------------------------------------------- listing

    def list_message_ids(self, query: str, *, max_results: int | None = None) -> list[str]:
        """Message ids matching a caller-supplied Gmail search query.

        The query is always the caller's - nothing is hardcoded here. Only ids
        are requested: `users.messages.list` returns ids and thread ids, never
        bodies, so listing an inbox does not download any mail. Pagination is
        followed until Gmail stops returning a page token or `max_results` is
        reached.
        """
        if max_results is not None and max_results <= 0:
            return []

        message_ids: list[str] = []
        page_token: str | None = None
        seen_page_tokens: set[str] = set()

        while True:
            request_args: dict[str, object] = {"userId": self.user_id, "q": query}
            if page_token:
                request_args["pageToken"] = page_token
            if max_results is not None:
                request_args["maxResults"] = max_results - len(message_ids)

            try:
                response = self._messages().list(**request_args).execute()
            except GmailClientError:
                raise
            except Exception as exc:  # noqa: BLE001 - HttpError and transport faults
                raise GmailClientError("could not list Gmail messages") from exc

            if not isinstance(response, dict):
                # Neither silently truncate the listing nor echo the payload
                # back to the caller.
                raise GmailClientError(
                    "Gmail returned an unexpected response while listing messages"
                )

            for message in response.get("messages") or []:
                message_id = message.get("id") if isinstance(message, dict) else None
                if message_id:
                    message_ids.append(message_id)
                    if max_results is not None and len(message_ids) >= max_results:
                        return message_ids

            page_token = response.get("nextPageToken")
            if not page_token:
                break
            if page_token in seen_page_tokens:
                # Gmail handed back a token we have already followed. Without
                # a max_results ceiling that would loop forever, so stop with
                # what we have rather than spinning.
                break
            seen_page_tokens.add(page_token)

        return message_ids
