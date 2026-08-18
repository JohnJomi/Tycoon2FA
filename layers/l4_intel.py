"""Layer 4 - threat intelligence correlation.

Signals specified in ARCHITECTURE.md section 4:
  l4.url_ioc        URLhaus + OpenPhish + PhishTank lookup on extracted URLs
  l4.domain_ioc     same, at registrable domain level
  l4.hosting_flag   ASN lookup against a static bulletproof-hosting list

Free feeds only. Lookups cached in SQLite with a 6-hour TTL.

Not implemented - Phase 0 scaffold.
"""
