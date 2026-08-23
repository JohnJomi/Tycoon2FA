"""Unit tests for the SQLite TTL cache in storage/cache.py.

Scope is deliberately narrow: set/get, TTL expiration, cache-miss behaviour
and persistence across reopening the same database file.
"""

from __future__ import annotations

import sqlite3
import threading
import time

import pytest

from storage.cache import Cache


# --------------------------------------------------------------------------
# 1. Basic set/get
# --------------------------------------------------------------------------


def test_set_then_get_returns_stored_value(tmp_path):
    cache = Cache(tmp_path / "cache.sqlite3")
    cache.set("example.com", {"age_days": 3}, ttl_seconds=3600)

    assert cache.get("example.com") == {"age_days": 3}


def test_get_missing_key_is_a_cache_miss(tmp_path):
    cache = Cache(tmp_path / "cache.sqlite3")

    assert cache.get("never-set") is None


# --------------------------------------------------------------------------
# 2. TTL expiration
# --------------------------------------------------------------------------


def test_entry_before_ttl_expiry_is_a_hit(tmp_path, monkeypatch):
    cache = Cache(tmp_path / "cache.sqlite3")
    now = 1_000_000.0
    monkeypatch.setattr("storage.cache.time.time", lambda: now)

    cache.set("example.com", "clean", ttl_seconds=60)
    monkeypatch.setattr("storage.cache.time.time", lambda: now + 59)

    assert cache.get("example.com") == "clean"


def test_entry_after_ttl_expiry_is_a_miss(tmp_path, monkeypatch):
    cache = Cache(tmp_path / "cache.sqlite3")
    now = 1_000_000.0
    monkeypatch.setattr("storage.cache.time.time", lambda: now)

    cache.set("example.com", "clean", ttl_seconds=60)
    monkeypatch.setattr("storage.cache.time.time", lambda: now + 61)

    assert cache.get("example.com") is None


def test_set_overwrites_existing_key_and_ttl(tmp_path, monkeypatch):
    cache = Cache(tmp_path / "cache.sqlite3")
    now = 1_000_000.0
    monkeypatch.setattr("storage.cache.time.time", lambda: now)

    cache.set("example.com", "first", ttl_seconds=60)
    cache.set("example.com", "second", ttl_seconds=60)

    assert cache.get("example.com") == "second"


# --------------------------------------------------------------------------
# 3. Persistence across reopening the database
# --------------------------------------------------------------------------


def test_values_survive_reopening_the_cache(tmp_path):
    db_path = tmp_path / "cache.sqlite3"
    Cache(db_path).set("example.com", "clean", ttl_seconds=3600)

    reopened = Cache(db_path)
    assert reopened.get("example.com") == "clean"


def test_expired_entries_are_not_returned_after_reopening(tmp_path, monkeypatch):
    db_path = tmp_path / "cache.sqlite3"
    now = 1_000_000.0
    monkeypatch.setattr("storage.cache.time.time", lambda: now)
    Cache(db_path).set("example.com", "clean", ttl_seconds=60)

    monkeypatch.setattr("storage.cache.time.time", lambda: now + 61)
    reopened = Cache(db_path)

    assert reopened.get("example.com") is None


def test_creates_db_file_and_parent_directories_when_missing(tmp_path):
    db_path = tmp_path / "nested" / "dir" / "cache.sqlite3"

    Cache(db_path).set("k", "v", ttl_seconds=60)

    assert db_path.exists()


# --------------------------------------------------------------------------
# 4. None is not a storable value
# --------------------------------------------------------------------------


def test_setting_none_is_rejected(tmp_path):
    """get() returns None for a miss, so a stored None could never read back."""
    cache = Cache(tmp_path / "cache.sqlite3")

    with pytest.raises(ValueError, match="None cannot be cached"):
        cache.set("example.com", None, ttl_seconds=60)


def test_rejecting_none_does_not_disturb_an_existing_entry(tmp_path):
    cache = Cache(tmp_path / "cache.sqlite3")
    cache.set("example.com", "clean", ttl_seconds=60)

    with pytest.raises(ValueError):
        cache.set("example.com", None, ttl_seconds=60)

    assert cache.get("example.com") == "clean"


def test_a_negative_lookup_is_cached_as_a_value_not_as_none(tmp_path):
    """The documented way to cache "WHOIS found nothing"."""
    cache = Cache(tmp_path / "cache.sqlite3")
    cache.set("nx.example", {"found": False}, ttl_seconds=60)

    assert cache.get("nx.example") == {"found": False}


# --------------------------------------------------------------------------
# 5. Lazy expiry must not evict a concurrently refreshed entry
# --------------------------------------------------------------------------


def test_expiry_delete_does_not_remove_a_refreshed_entry(tmp_path, monkeypatch):
    """Reproduces the read-expired / refresh / delete interleaving exactly.

    A second Cache over the same file refreshes the key in the window between
    this get()'s SELECT and its DELETE. The delete is scoped to the expiry the
    SELECT observed, so the fresh row must survive.
    """
    db_path = tmp_path / "cache.sqlite3"
    now = 1_000_000.0
    real_time = time.time

    reader = Cache(db_path)
    monkeypatch.setattr("storage.cache.time.time", lambda: now)
    reader.set("example.com", "stale", ttl_seconds=60)

    writer = Cache(db_path)
    refreshed = False

    def clock_that_refreshes_mid_get() -> float:
        # get() calls time.time() once, after its SELECT and before its
        # DELETE - exactly the window the race occupies.
        nonlocal refreshed
        if not refreshed:
            refreshed = True
            monkeypatch.setattr("storage.cache.time.time", lambda: now + 61)
            writer.set("example.com", "fresh", ttl_seconds=60)
            monkeypatch.setattr("storage.cache.time.time", clock_that_refreshes_mid_get)
        return now + 61

    monkeypatch.setattr("storage.cache.time.time", clock_that_refreshes_mid_get)

    # The reader saw a stale row, so it correctly reports a miss...
    assert reader.get("example.com") is None
    assert refreshed

    # ...but it must not have deleted the value the writer just stored.
    monkeypatch.setattr("storage.cache.time.time", lambda: now + 70)
    assert reader.get("example.com") == "fresh"

    monkeypatch.setattr("storage.cache.time.time", real_time)


def test_expiry_delete_still_removes_an_unrefreshed_entry(tmp_path, monkeypatch):
    """The conditional delete must not break ordinary lazy expiration."""
    db_path = tmp_path / "cache.sqlite3"
    cache = Cache(db_path)
    now = 1_000_000.0
    monkeypatch.setattr("storage.cache.time.time", lambda: now)
    cache.set("example.com", "stale", ttl_seconds=60)

    monkeypatch.setattr("storage.cache.time.time", lambda: now + 61)
    assert cache.get("example.com") is None

    remaining = sqlite3.connect(db_path).execute(
        "SELECT COUNT(*) FROM cache_entries WHERE key = ?", ("example.com",)
    ).fetchone()[0]
    assert remaining == 0


# --------------------------------------------------------------------------
# 6. Overwriting refreshes the TTL, not just the value
# --------------------------------------------------------------------------


def test_overwrite_refreshes_the_ttl(tmp_path, monkeypatch):
    """The second set's TTL governs; the first deadline must not be inherited."""
    cache = Cache(tmp_path / "cache.sqlite3")
    now = 1_000_000.0
    monkeypatch.setattr("storage.cache.time.time", lambda: now)

    cache.set("example.com", "first", ttl_seconds=10)
    cache.set("example.com", "second", ttl_seconds=600)

    # Past the first TTL, well inside the second.
    monkeypatch.setattr("storage.cache.time.time", lambda: now + 11)
    assert cache.get("example.com") == "second"

    # And the second TTL is genuinely enforced, not merely long.
    monkeypatch.setattr("storage.cache.time.time", lambda: now + 601)
    assert cache.get("example.com") is None


def test_overwrite_can_shorten_the_ttl(tmp_path, monkeypatch):
    cache = Cache(tmp_path / "cache.sqlite3")
    now = 1_000_000.0
    monkeypatch.setattr("storage.cache.time.time", lambda: now)

    cache.set("example.com", "first", ttl_seconds=600)
    cache.set("example.com", "second", ttl_seconds=10)

    monkeypatch.setattr("storage.cache.time.time", lambda: now + 11)
    assert cache.get("example.com") is None


# --------------------------------------------------------------------------
# 7. Concurrent access over the shared connection
# --------------------------------------------------------------------------


def test_concurrent_get_and_set_are_safe(tmp_path):
    """The lock must keep the shared sqlite3 connection usable under threads."""
    cache = Cache(tmp_path / "cache.sqlite3")
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def hammer(worker: int) -> None:
        try:
            barrier.wait()
            for i in range(50):
                cache.set(f"key-{worker}", {"worker": worker, "i": i}, ttl_seconds=60)
                cache.get(f"key-{(worker + 1) % 8}")
        except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
            errors.append(exc)

    threads = [threading.Thread(target=hammer, args=(w,)) for w in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    for worker in range(8):
        assert cache.get(f"key-{worker}") == {"worker": worker, "i": 49}
