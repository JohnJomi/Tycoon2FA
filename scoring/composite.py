"""Composite risk scoring.

Per ARCHITECTURE.md section 6: weighted sum over layer signals, weight
renormalization when a layer did not complete, and verdict bands read from
config/weights.yaml - never hardcoded. Phase A weights are hand-assigned;
Phase B refits them and re-derives thresholds from the ROC curve.

Not implemented - Phase 0 scaffold. Nothing reads config/weights.yaml yet.
"""
