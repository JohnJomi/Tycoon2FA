"""TTL-aware SQLite cache for WHOIS, threat-intel and sandbox results.

Negative lookups are cached too, so a rate-limited WHOIS does not turn into a
retry storm. TTLs per ARCHITECTURE.md: WHOIS 7 days, intel feeds 6 hours -
callers pass their own ttl_seconds per call; this module has no opinion on
what a given key's TTL should be.

Synchronous and file-backed. Nothing upstream awaits this cache, so it stays
a plain sqlite3 wrapper rather than adopting async for its own sake.

`get` returns None for a miss, so None is not a storable value - see `set`.
A "negative lookup" is cached by storing a value that *represents* absence
(``{"found": false}``, ``[]``), not by storing None, which would be
indistinguishable from never having been cached at all.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

__all__ = ["Cache"]


class Cache:
    """A persistent key/value store with per-entry TTL expiration.

    Values are JSON-serialized, so anything json.dumps can handle (str, int,
    float, bool, list, dict) may be cached - except None, which the API
    reserves for "miss". Expiration is checked at read time against the stored
    expiry timestamp; expired rows are treated as a miss and lazily deleted
    rather than swept on a timer.

    One connection guarded by one lock. The connection is shared across
    threads (`check_same_thread=False`), so every statement runs under
    `self._lock` - sqlite3 connection objects are not safe to use concurrently
    even when SQLite itself is compiled thread-safe.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        with self._lock:
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
        """Store `value` under `key`, expiring `ttl_seconds` from now.

        Overwriting an existing key replaces its value *and* its expiry, so a
        refresh extends the entry rather than inheriting the old deadline.

        Raises ValueError for a None value: `get` returns None to mean "miss",
        so a stored None could never be read back as a hit.
        """
        if value is None:
            raise ValueError(
                "None cannot be cached: get() returns None to signal a miss. "
                "Represent a negative lookup with a value such as "
                '{"found": false}.'
            )

        expires_at = time.time() + ttl_seconds
        encoded = json.dumps(value)
        with self._lock:
            self._conn.execute(
                "INSERT INTO cache_entries (key, value, expires_at) VALUES (?, ?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value,"
                " expires_at = excluded.expires_at",
                (key, encoded, expires_at),
            )
            self._conn.commit()

    def get(self, key: str) -> Any | None:
        """Return the cached value for `key`, or None on miss or expiry."""
        with self._lock:
            row = self._conn.execute(
                "SELECT value, expires_at FROM cache_entries WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None

            value, expires_at = row
            if expires_at <= time.time():
                # Delete the exact row this read observed. Matching on
                # expires_at as well as key means a concurrent set() that
                # refreshed the entry between the SELECT and the DELETE is left
                # alone - otherwise this reader would evict a fresh value.
                self._conn.execute(
                    "DELETE FROM cache_entries WHERE key = ? AND expires_at = ?",
                    (key, expires_at),
                )
                self._conn.commit()
                return None

        return json.loads(value)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Cache:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
