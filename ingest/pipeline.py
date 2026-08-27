"""Gmail ingestion: Gmail's mailbox view, normalized for the detection layers.

    GmailClient -> Gmail Message resource -> parse_email() -> IngestedMessage

`ingest.gmail_client` owns transport and auth; `ingest.parser` owns MIME. This
module owns neither. It is the seam between them and the rest of the system:
it walks a Gmail query with pagination, drops messages it has already handled,
merges the Gmail-side metadata that lives outside the RFC-822 payload with the
`ParsedEmail` the parser produced, and yields one `IngestedMessage` per
message.

Everything downstream consumes `IngestedMessage`. Nothing downstream imports
`googleapiclient`, knows what a page token is, or knows that `labelIds` is
spelled in camelCase - that is the entire point of this layer.

Per ARCHITECTURE.md section 3 messages are fetched `format='raw'`, and the
scope stays `gmail.readonly`. This module reads; it never modifies, labels,
sends or deletes.

Security
--------
Gmail content is untrusted. Nothing here logs or prints subjects, bodies,
addresses or attachment content, and nothing is written to disk. Failure
records carry a Gmail message id and a reason, never message content.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from core.models import Attachment, ParsedEmail
from ingest.gmail_client import GmailClient, GmailClientError, decode_raw_payload
from ingest.parser import EmailParseError, parse_email

__all__ = [
    "IngestedMessage",
    "IngestionFailure",
    "IngestionResult",
    "GmailIngestor",
    "SeenMessageStore",
    "InMemorySeenMessageStore",
    "Checkpoint",
    "InMemoryCheckpoint",
]

# Gmail's own label ids for the folders Layer 1 cares about telling apart.
SPAM_LABEL = "SPAM"
INBOX_LABEL = "INBOX"
TRASH_LABEL = "TRASH"

# Email authentication headers Layer 1 reads. Surfaced by name so a consumer
# does not have to know the RFC spelling or the case Gmail happened to use.
AUTH_HEADERS = (
    "authentication-results",
    "received-spf",
    "dkim-signature",
    "arc-authentication-results",
)


# --------------------------------------------------------------------------
# The normalized message
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class IngestedMessage:
    """One Gmail message, normalized into Tycoon2FA's own representation.

    Two halves, deliberately kept distinct rather than flattened:

    - `email` is the parsed RFC-822 content (`ParsedEmail`, ARCHITECTURE.md
      section 2), which is what the detection layers already consume.
    - the remaining fields are Gmail-side metadata that exists *only* in the
      API resource: thread membership, labels, and Gmail's own receipt time.
      None of it is recoverable from the message bytes.

    Frozen: an ingested message is a record of what the mailbox held at fetch
    time. Layers annotate their own `DetectionSignal`s; they do not edit this.

    `internal_date` is Gmail's receipt timestamp, always timezone-aware UTC. It
    is preferred over the `Date:` header for ordering because `Date:` is
    attacker-controlled - a phishing message can claim any send time it likes,
    while `internalDate` is stamped by Gmail on arrival.
    """

    gmail_id: str
    thread_id: str | None
    email: ParsedEmail
    label_ids: tuple[str, ...] = ()
    internal_date: datetime | None = None
    size_estimate: int = 0
    history_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.gmail_id, str) or not self.gmail_id.strip():
            raise ValueError("gmail_id must be a non-empty string")
        if not isinstance(self.email, ParsedEmail):
            raise TypeError(
                f"email must be a ParsedEmail, got {type(self.email).__name__}"
            )

    # -- content, forwarded from the parsed email ---------------------------
    #
    # Thin accessors so a consumer can treat an IngestedMessage as "the
    # message" without reaching through `.email` for the common fields, while
    # ParsedEmail stays the single place the content actually lives.

    @property
    def sender(self) -> str:
        """The From address. May be empty - see `ParsedEmail`."""
        return self.email.from_addr

    @property
    def sender_display(self) -> str:
        return self.email.from_display

    @property
    def recipients(self) -> list[str]:
        return list(self.email.to_addrs)

    @property
    def subject(self) -> str:
        return self.email.subject

    @property
    def body_text(self) -> str:
        return self.email.body_text

    @property
    def body_html(self) -> str | None:
        return self.email.body_html

    @property
    def attachments(self) -> list[Attachment]:
        """Attachment *metadata*; payloads are never carried, per section 2."""
        return list(self.email.attachments)

    @property
    def rfc822_message_id(self) -> str:
        """The `Message-ID:` header - not Gmail's id. The two differ."""
        return self.email.message_id

    # -- Gmail-side metadata -----------------------------------------------

    @property
    def is_spam(self) -> bool:
        return SPAM_LABEL in self.label_ids

    @property
    def is_inbox(self) -> bool:
        return INBOX_LABEL in self.label_ids

    @property
    def is_trash(self) -> bool:
        return TRASH_LABEL in self.label_ids

    def auth_headers(self) -> dict[str, list[str]]:
        """The email-authentication headers Layer 1 consumes, lowercase-keyed.

        Only headers actually present are returned, and every one is returned
        in full: a message can legitimately carry several `Received-SPF` or
        `DKIM-Signature` headers, and dropping the duplicates would hide
        exactly the disagreement Layer 1 is looking for. An absent header is an
        absent key rather than an empty string, so "Gmail asserted nothing"
        stays distinguishable from "Gmail asserted a blank result".
        """
        lowered = {name.lower(): values for name, values in self.email.headers.items()}
        return {
            name: list(lowered[name]) for name in AUTH_HEADERS if lowered.get(name)
        }


@dataclass(frozen=True)
class IngestionFailure:
    """One message that could not be ingested, and why.

    A failure is data, not an exception, because one unparseable message in a
    mailbox of 75 must not end the run. `reason` is a short human-readable
    string and `error` the original exception; neither ever carries message
    content.
    """

    gmail_id: str
    reason: str
    error: Exception | None = None


@dataclass
class IngestionResult:
    """The outcome of one `ingest_all` run.

    Carries what succeeded, what failed and what was skipped as already-seen,
    so a caller can tell "nothing new" apart from "everything broke" - the
    same completed/incomplete distinction `LayerResult` draws in section 2.
    """

    messages: list[IngestedMessage] = field(default_factory=list)
    failures: list[IngestionFailure] = field(default_factory=list)
    skipped_duplicates: int = 0

    @property
    def ingested_count(self) -> int:
        return len(self.messages)


# --------------------------------------------------------------------------
# Pluggable state
# --------------------------------------------------------------------------


@runtime_checkable
class SeenMessageStore(Protocol):
    """Records which Gmail ids have already been ingested.

    A protocol, not a table. ARCHITECTURE.md does not yet specify persistence
    for ingestion state, so this module declines to invent a schema: the
    in-memory implementation below is the default, and a SQLite- or
    `storage.cache`-backed one can be dropped in later without touching the
    ingestor.
    """

    def has_seen(self, gmail_id: str) -> bool: ...

    def mark_seen(self, gmail_id: str) -> None: ...


class InMemorySeenMessageStore:
    """Per-process dedupe. Forgets everything when the process exits.

    Correct for a single run - the same message is never fetched twice, even
    if Gmail returns it on two pages - and honest about being nothing more.
    """

    def __init__(self, seen: set[str] | None = None) -> None:
        self._seen: set[str] = set(seen or ())

    def has_seen(self, gmail_id: str) -> bool:
        return gmail_id in self._seen

    def mark_seen(self, gmail_id: str) -> None:
        self._seen.add(gmail_id)

    def __len__(self) -> int:
        return len(self._seen)


@runtime_checkable
class Checkpoint(Protocol):
    """A high-water mark for incremental ingestion.

    `load` returns the epoch-seconds timestamp of the newest message already
    ingested, or None on a first run. `save` advances it.
    """

    def load(self) -> int | None: ...

    def save(self, epoch_seconds: int) -> None: ...


class InMemoryCheckpoint:
    """The default checkpoint: real within a process, gone after it."""

    def __init__(self, epoch_seconds: int | None = None) -> None:
        self._value = epoch_seconds

    def load(self) -> int | None:
        return self._value

    def save(self, epoch_seconds: int) -> None:
        if self._value is None or epoch_seconds > self._value:
            self._value = epoch_seconds


# --------------------------------------------------------------------------
# The ingestor
# --------------------------------------------------------------------------


def _internal_date(resource: dict) -> datetime | None:
    """Gmail's `internalDate` (epoch milliseconds, as a string) as UTC.

    Returns None rather than raising when the field is missing or unparseable:
    an odd timestamp is not a reason to discard an otherwise good message, and
    a None here simply means this message cannot advance the checkpoint.
    """
    value = resource.get("internalDate")
    if value is None:
        return None
    try:
        millis = int(value)
    except (TypeError, ValueError):
        return None
    try:
        return datetime.fromtimestamp(millis / 1000, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _label_ids(resource: dict) -> tuple[str, ...]:
    """`labelIds` as a tuple of strings, tolerating a missing or odd field."""
    labels = resource.get("labelIds")
    if not isinstance(labels, list):
        return ()
    return tuple(label for label in labels if isinstance(label, str) and label)


def _size_estimate(resource: dict) -> int:
    try:
        size = int(resource.get("sizeEstimate", 0))
    except (TypeError, ValueError):
        return 0
    return max(size, 0)


class GmailIngestor:
    """Walks a Gmail query and yields normalized messages.

    Composed of, not derived from, `GmailClient`: the client stays a pure
    transport, and this object stays testable with a fake in its place.

    Dedupe is at the ingestion boundary, before the fetch, so a message the
    store has already seen costs no Gmail round trip at all - which matters
    both for quota and because `users.messages.list` legitimately repeats ids
    across pages when the mailbox changes mid-pagination.
    """

    def __init__(
        self,
        client: GmailClient,
        *,
        seen_store: SeenMessageStore | None = None,
        checkpoint: Checkpoint | None = None,
    ) -> None:
        self.client = client
        self.seen_store: SeenMessageStore = seen_store or InMemorySeenMessageStore()
        self.checkpoint = checkpoint

    # ------------------------------------------------------------ querying

    def _effective_query(self, query: str) -> str:
        """The caller's query, narrowed by the checkpoint when there is one.

        Incremental ingestion is expressed in Gmail's own query language
        (`after:<epoch seconds>`) rather than in a local database, because
        ARCHITECTURE.md has not yet specified ingestion persistence. That keeps
        the narrowing server-side - Gmail does not return the old pages at all,
        so an incremental run is cheap rather than merely quiet - and leaves
        the `Checkpoint` protocol as the single seam to swap when persistence
        is designed.

        The bound is inclusive of its own second, so the newest already-seen
        message can come back once more; the seen-store drops it. Overlapping
        by a second is the safe direction: the alternative risks skipping a
        message that arrived in the same second as the checkpoint.

        The caller's query is parenthesized before the bound is appended.
        Gmail's `OR` binds looser than the implicit `AND` between terms, so a
        bare `in:spam OR in:inbox after:N` means "spam, or inbox-since-N" - the
        bound silently fails to apply to the first branch and the run re-ingests
        the whole of it. `(in:spam OR in:inbox) after:N` is what was meant.
        """
        if self.checkpoint is None:
            return query
        since = self.checkpoint.load()
        if since is None:
            return query
        bound = f"after:{max(int(since), 0)}"
        return f"({query.strip()}) {bound}" if query.strip() else bound

    # ----------------------------------------------------------- ingestion

    def ingest(
        self, query: str = "", *, max_results: int | None = None
    ) -> Iterator[IngestedMessage]:
        """Yield normalized messages for a Gmail query, newest pages first.

        A generator: pagination, fetching and parsing are interleaved, so a
        caller can process the first message before the last page is listed,
        and can stop early without listing the rest.

        `max_results` bounds how many *refs are listed*, not how many messages
        are yielded, since duplicates and failures are filtered afterwards.
        That keeps the Gmail-side cost predictable, which is the quantity worth
        bounding.

        Messages that fail to fetch or parse are skipped. Use `ingest_all` when
        the failures themselves matter; a `GmailClientError` that indicates the
        *listing* failed still propagates, because a broken listing means the
        run saw an unknown fraction of the mailbox and must not look like a
        clean pass.
        """
        for message in self._ingest(query, max_results=max_results, result=None):
            yield message

    def ingest_all(
        self, query: str = "", *, max_results: int | None = None
    ) -> IngestionResult:
        """Ingest eagerly, keeping the failures and the duplicate count.

        The reporting form of `ingest`. Prefer it wherever "how much of the
        mailbox did we actually see" is a question worth answering.
        """
        result = IngestionResult()
        result.messages.extend(self._ingest(query, max_results=max_results, result=result))
        return result

    def _ingest(
        self,
        query: str,
        *,
        max_results: int | None,
        result: IngestionResult | None,
    ) -> Iterator[IngestedMessage]:
        newest_seen: int | None = None

        refs = self.client.iter_message_refs(
            self._effective_query(query), max_results=max_results
        )

        for ref in refs:
            gmail_id = ref.get("id")
            if not gmail_id:
                continue

            if self.seen_store.has_seen(gmail_id):
                if result is not None:
                    result.skipped_duplicates += 1
                continue

            try:
                message = self._fetch_one(gmail_id, ref.get("threadId"))
            except (GmailClientError, EmailParseError, ValueError) as exc:
                # One bad message is not a bad run. Record it and move on -
                # a malformed phishing message is exactly the kind of thing
                # that fails to parse, and losing the whole mailbox over it
                # would be the wrong trade.
                if result is not None:
                    result.failures.append(
                        IngestionFailure(
                            gmail_id=gmail_id, reason=_failure_reason(exc), error=exc
                        )
                    )
                # Whether the message is marked seen turns on whether the
                # failure can ever resolve itself. A parse failure is a
                # property of the message: it will fail identically next run,
                # so marking it seen stops a permanent retry loop. A
                # GmailClientError is a property of the *call* - a timeout, a
                # 429, a 5xx - and marking it seen would suppress a message
                # the mailbox still holds, permanently, on the strength of one
                # bad minute. It is left unseen so a later run retries it.
                if not isinstance(exc, GmailClientError):
                    self.seen_store.mark_seen(gmail_id)
                continue

            self.seen_store.mark_seen(gmail_id)

            if message.internal_date is not None:
                stamp = int(message.internal_date.timestamp())
                if newest_seen is None or stamp > newest_seen:
                    newest_seen = stamp

            yield message

        # Advanced only after the listing has been walked to the end. Advancing
        # per message would let an exception mid-run leave the checkpoint ahead
        # of the messages actually handed downstream, permanently skipping the
        # remainder.
        if self.checkpoint is not None and newest_seen is not None:
            self.checkpoint.save(newest_seen)

    def _fetch_one(self, gmail_id: str, thread_id: str | None) -> IngestedMessage:
        """Fetch and normalize one message.

        MIME structure - plain text, HTML, multipart, nested multipart - is
        `ingest.parser`'s job and is not re-implemented here; this method only
        joins the parser's output to Gmail's metadata.
        """
        resource = self.client.fetch_message_resource(gmail_id)

        raw = resource.get("raw")
        if not raw:
            raise GmailClientError(
                f"Gmail message {gmail_id} came back without a raw payload"
            )

        parsed = parse_email(decode_raw_payload(raw))

        return IngestedMessage(
            gmail_id=gmail_id,
            # Prefer the resource's own threadId; the listing ref is only a
            # fallback for a Gmail response that omitted it.
            thread_id=resource.get("threadId") or thread_id,
            email=parsed,
            label_ids=_label_ids(resource),
            internal_date=_internal_date(resource),
            size_estimate=_size_estimate(resource),
            history_id=resource.get("historyId"),
        )


def _failure_reason(exc: Exception) -> str:
    """A short reason string that never carries message content."""
    if isinstance(exc, EmailParseError):
        return "the message could not be parsed"
    if isinstance(exc, GmailClientError):
        return "the message could not be fetched from Gmail"
    return "the message could not be normalized"
