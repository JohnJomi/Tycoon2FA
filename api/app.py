"""FastAPI application.

Endpoint surface specified in ARCHITECTURE.md section 8:
  POST /auth/login              OAuth redirect
  GET  /messages?limit=25       inbox list with cached verdicts
  POST /analyze/{message_id}    run pipeline, return RiskAssessment
  GET  /analyze/{message_id}    cached assessment
  POST /analyze/upload          .eml upload (demo path, no Gmail needed)
  GET  /health                  per-layer readiness incl. GPT-2 load state

Not implemented - Phase 0 scaffold. No app object is defined yet.
"""
