"""Snapshot-backed `ThreatIntelSource`s: OpenPhish and PhishTank.

    ThreatIntelSource -> SnapshotFeedSource -> one downloaded feed -> a set

Neither of these two free sources answers per-URL queries. OpenPhish's free
tier publishes a plain-text feed of currently-active phishing URLs; PhishTank
distributes a bulk database dump. ARCHITECTURE.md section 4 names both, so the
seam accommodates how they actually work rather than inventing a query
endpoint: the provider downloads the feed once, holds it in memory, and answers
membership locally.

That is a real difference in semantics and is reported as one -
`refresh_interval_seconds` states how stale an answer may be, so a caller can
say "absent from a feed refreshed within the last hour" rather than implying a
live check.

The feed is fetched from the *feed publisher*. No indicator's own host is ever
contacted.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterable

import httpx

from layers.threat_intel import IndicatorKind, IntelTimeout, IntelUnavailable, IntelVerdict

__all__ = ["OpenPhishSource", "PhishTankSource", "SnapshotFeedSource"]

# One hour: well inside the 6-hour lookup TTL in ARCHITECTURE.md section 4, so
# a cached verdict never outlives the snapshot that produced it by much, and
# far longer than a message-processing burst - the feed is fetched once per
# process per hour, not once per email.
DEFAULT_REFRESH_SECONDS = 3600.0

# Generous next to `DEFAULT_LAYER_TIMEOUTS[L4]` of 6s because this is a whole
# feed, not one query - but still under it, so a slow publisher makes this
# source unavailable rather than the layer time out.
DEFAULT_TIMEOUT_SECONDS = 5.0


class SnapshotFeedSource:
    """Shared machinery for a source that answers from a downloaded feed.

    Subclasses supply the URL and the parser. This class owns fetching,
    refresh timing, thread-safety and the failure contract - a feed that cannot
    be fetched or parsed makes the source *unavailable*, never negative, so an
    outage can never be read as "this URL is not listed".

    A stale snapshot is deliberately kept and used when a refresh fails: an
    hour-old answer from a feed that lists 30-day-old phishing URLs is worth
    far more than no answer, and the staleness is reported in the verdict's
    metadata rather than hidden. Only a source that has *never* loaded a
    snapshot is unavailable.
    """

    name = "snapshot"
    feed_url = ""

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        refresh_interval_seconds: float = DEFAULT_REFRESH_SECONDS,
        now: object = None,
    ) -> None:
        self._client = client
        self._timeout = float(timeout)
        self.refresh_interval_seconds = float(refresh_interval_seconds)
        self._now = now if callable(now) else time.monotonic
        self._lock = threading.Lock()
        self._entries: frozenset[str] | None = None
        self._loaded_at: float | None = None

    # -- subclass hooks ----------------------------------------------------

    def parse(self, body: str) -> Iterable[str]:  # pragma: no cover - abstract
        raise NotImplementedError

    def request_headers(self) -> dict[str, str]:
        return {}

    def resolve_feed_url(self) -> str:
        """The URL to download. Overridden where a key is part of the path."""
        return self.feed_url

    # -- the seam ----------------------------------------------------------

    def supports(self, kind: IndicatorKind) -> bool:
        return kind is IndicatorKind.URL

    def check(self, indicator: str, kind: IndicatorKind) -> IntelVerdict:
        if not self.supports(kind):
            raise IntelUnavailable(f"{self.name} does not cover {kind.value} indicators")

        entries, age = self._snapshot()
        found = indicator in entries
        detail = f"listed in the {self.name} feed" if found else None
        return IntelVerdict(
            found=found,
            source=self.name,
            detail=(f"{detail}, snapshot {age:.0f}s old" if detail else None),
        )

    # -- snapshot management ----------------------------------------------

    def _snapshot(self) -> tuple[frozenset[str], float]:
        """The current entry set and its age, refreshing it if it is stale."""
        with self._lock:
            now = self._now()
            fresh = (
                self._entries is not None
                and self._loaded_at is not None
                and (now - self._loaded_at) < self.refresh_interval_seconds
            )
            if fresh:
                return self._entries, now - self._loaded_at

            try:
                entries = frozenset(self.parse(self._fetch()))
            except IntelUnavailable:
                if self._entries is not None:
                    # Keep serving the stale snapshot; see the class docstring.
                    return self._entries, now - (self._loaded_at or now)
                raise

            self._entries = entries
            self._loaded_at = now
            return entries, 0.0

    def _fetch(self) -> str:
        headers = self.request_headers()
        url = self.resolve_feed_url()
        try:
            if self._client is not None:
                response = self._client.get(url, headers=headers, timeout=self._timeout)
            else:
                with httpx.Client(timeout=self._timeout, follow_redirects=True) as client:
                    response = client.get(url, headers=headers)
        except httpx.TimeoutException as exc:
            raise IntelTimeout(f"{self.name} feed timed out after {self._timeout:g}s") from exc
        except httpx.HTTPError as exc:
            raise IntelUnavailable(
                f"{self.name} feed request failed: {type(exc).__name__}"
            ) from exc

        if response.status_code != 200:
            raise IntelUnavailable(f"{self.name} feed returned HTTP {response.status_code}")
        return response.text


class OpenPhishSource(SnapshotFeedSource):
    """OpenPhish's free community feed: one URL per line, plain text."""

    name = "openphish"
    feed_url = "https://openphish.com/feed.txt"

    def parse(self, body: str) -> Iterable[str]:
        lines = [line.strip() for line in body.splitlines()]
        entries = [line for line in lines if line and not line.startswith("#")]
        if not entries:
            # An empty feed is far more likely to be an error page or a
            # truncated download than a world with no phishing in it, and
            # trusting it would answer "not listed" for every URL.
            raise IntelUnavailable(f"{self.name} feed contained no entries")
        return entries


class PhishTankSource(SnapshotFeedSource):
    """PhishTank's bulk database dump: a JSON array of verified entries.

    The download is keyed per application. Without a key PhishTank serves the
    anonymous URL heavily rate-limited, so the key is read from the
    environment; its absence makes the source unavailable rather than negative.
    """

    name = "phishtank"
    feed_url = "https://data.phishtank.com/data/{key}/online-valid.json"
    API_KEY_ENV = "PHISHTANK_APP_KEY"
    USER_AGENT_ENV = "PHISHTANK_USER_AGENT"

    def __init__(self, *, api_key: str | None = None, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._api_key = api_key if api_key is not None else os.environ.get(self.API_KEY_ENV)

    def request_headers(self) -> dict[str, str]:
        # PhishTank asks every downloader to identify itself and rejects the
        # default client UA.
        return {"User-Agent": os.environ.get(self.USER_AGENT_ENV, "phishtank/tycoon2fa")}

    def resolve_feed_url(self) -> str:
        if not self._api_key:
            raise IntelUnavailable(f"{self.API_KEY_ENV} is not set")
        return self.feed_url.format(key=self._api_key)

    def parse(self, body: str) -> Iterable[str]:
        try:
            payload = json.loads(body)
        except ValueError as exc:
            raise IntelUnavailable(f"{self.name} feed was not valid JSON") from exc
        if not isinstance(payload, list):
            raise IntelUnavailable(
                f"{self.name} feed was {type(payload).__name__}, not a list"
            )
        entries = [
            entry["url"]
            for entry in payload
            if isinstance(entry, dict) and isinstance(entry.get("url"), str)
        ]
        if not entries:
            raise IntelUnavailable(f"{self.name} feed contained no entries")
        return entries
