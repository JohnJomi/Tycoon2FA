"""Layer 2 - URL and redirect chain analysis.

Signals specified in ARCHITECTURE.md section 4:
  l2.base64_email_param    base64-encoded address in query/fragment
  l2.redirect_depth        hop following, allow_redirects=False, cap 8
  l2.captcha_gate          Turnstile/hCaptcha/reCAPTCHA in rendered DOM
  l2.qr_url                QR decode of image and inline cid: attachments
  l2.domain_mismatch_brand anchor text brand vs. href target

Network layer: must run from sandboxed egress, never a home or campus IP.

Not implemented - Phase 0 scaffold.
"""
