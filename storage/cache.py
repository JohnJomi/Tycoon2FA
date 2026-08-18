"""TTL-aware SQLite cache for WHOIS, threat-intel and sandbox results.

Negative lookups are cached too, so a rate-limited WHOIS does not turn into a
retry storm. TTLs per ARCHITECTURE.md: WHOIS 7 days, intel feeds 6 hours.

Not implemented - Phase 0 scaffold.
"""
