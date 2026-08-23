"""TTL-aware SQLite cache for WHOIS, threat-intel and sandbox results.

Negative lookups are cached too, so a rate-limited WHOIS does not turn into a
retry storm. TTLs per ARCHITECTURE.md: WHOIS 7 days, intel feeds 6 hours -
callers pass their own ttl_seconds per call; this module has no opinion on
what a given key's TTL should be.

Synchronous and file-backed. Nothing upstream awaits this cache, so it stays
a plain sqlite3 wrapper rather than adopting async for its own sake.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

__all__ = ["Cache"]


class Cache:
    """A persistent key/value store with per-entry TTL expiration.

    Values are JSON-serialized, so anything json.dumps can handle (str, int,
    float, bool, None, list, dict) may be cached. Expiration is checked at
    read time against the stored expiry timestamp; expired rows are treated
    as a miss and lazily deleted rather than swept on a timer.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_entries (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                expires_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    def set(self, key: str, value: Any, ttl_seconds: float) -> None:
        """Store `value` under `key`, expiring `ttl_seconds` from now."""
        expires_at = time.time() + ttl_seconds
        self._conn.execute(
            "INSERT INTO cache_entries (key, value, expires_at) VALUES (?, ?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value,"
            " expires_at = excluded.expires_at",
            (key, json.dumps(value), expires_at),
        )
        self._conn.commit()

    def get(self, key: str) -> Any | None:
        """Return the cached value for `key`, or None on miss or expiry."""
        row = self._conn.execute(
            "SELECT value, expires_at FROM cache_entries WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None

        value, expires_at = row
        if expires_at <= time.time():
            self._conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
            self._conn.commit()
            return None

        return json.loads(value)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Cache:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
