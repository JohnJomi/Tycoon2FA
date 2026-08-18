"""Layer 1 - header and domain intelligence.

Signals specified in ARCHITECTURE.md section 4:
  l1.auth_fail                   read Gmail's Authentication-Results verdicts
  l1.replyto_mismatch            registrable domain of Reply-To != From
  l1.domain_age_lt_7d            WHOIS creation date (cached, 7-day TTL)
  l1.display_name_impersonation  brand token in display name, not in domain

Not implemented - Phase 0 scaffold.
"""
