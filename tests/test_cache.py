"""Unit tests for the SQLite TTL cache in storage/cache.py.

Scope is deliberately narrow: set/get, TTL expiration, cache-miss behaviour
and persistence across reopening the same database file.
"""

from __future__ import annotations

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
