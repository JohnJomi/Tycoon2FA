"""Core data contracts shared by every other module.

Per ARCHITECTURE.md section 2, this module defines:
  - Signal          (name, layer, fired, value, weight, evidence, error)
  - LayerResult     (layer, signals, completed, duration_ms)
  - ParsedEmail     (normalized RFC-822 representation)
  - RiskAssessment  (score, verdict, signals, layers_completed, scored_at)

This module depends on nothing; everything else depends on it.

Not implemented - Phase 0 scaffold.
"""
