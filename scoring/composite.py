"""Composite risk scoring.

Per ARCHITECTURE.md section 6: a weighted sum over the layers, with the
weights and thresholds read from `config/weights.yaml` - never hardcoded here.
Phase B refits both; nothing in this module needs to change when it does.

    list[LayerResult] -> RiskAssessment(score, level, signals)

Renormalization is the point of this module. An incomplete layer contributes
no score, and its weight is redistributed proportionally across the layers
that did complete. A layer that timed out therefore does not drag the score
toward clean - the message is judged on the evidence that was actually
gathered.

This module produces a *risk level* (LOW / MEDIUM / HIGH), not an operational
action. Mapping a level onto deliver / warn / block is the later policy stage
(ARCHITECTURE.md section 6); this prototype takes no action.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from core.models import (
    DetectionLayer,
    DetectionSignal,
    LayerResult,
    RiskAssessment,
    RiskLevel,
)

__all__ = [
    "DEFAULT_WEIGHTS_PATH",
    "ScoringWeights",
    "UnscoreableError",
    "layer_contributions",
    "layer_score",
    "load_weights",
    "score",
]

DEFAULT_WEIGHTS_PATH = "config/weights.yaml"


class UnscoreableError(RuntimeError):
    """Raised when no layer completed, so there is nothing to score.

    Deliberately an error rather than a score: with every layer missing there
    is no evidence at all, and emitting 0.0/LOW would turn a total outage into
    an all-clear - the exact failure ARCHITECTURE.md section 2 forbids.
    """


@dataclass(frozen=True)
class ScoringWeights:
    """Layer weights and risk-band thresholds, as loaded from configuration."""

    layer_weights: dict[DetectionLayer, float]
    deliver_threshold: float
    block_threshold: float

    def weight_for(self, layer: DetectionLayer) -> float:
        return self.layer_weights.get(layer, 0.0)


def load_weights(path: str | os.PathLike[str] | None = None) -> ScoringWeights:
    """Read weights and thresholds from YAML.

    The path comes from the argument, else the WEIGHTS_CONFIG environment
    variable, else the project default. No production weight is written in
    Python.
    """
    config_path = Path(path or os.environ.get("WEIGHTS_CONFIG", DEFAULT_WEIGHTS_PATH))
    try:
        raw = yaml.safe_load(config_path.read_text()) or {}
    except Exception as exc:  # noqa: BLE001 - missing or malformed config
        raise UnscoreableError(f"could not read scoring weights from {config_path}") from exc

    layers_section = raw.get("layers") or {}
    thresholds = raw.get("thresholds") or {}

    layer_weights: dict[DetectionLayer, float] = {}
    for layer in DetectionLayer:
        value = layers_section.get(layer.name.lower())
        if value is None:
            raise UnscoreableError(f"{config_path} is missing a weight for {layer.name}")
        weight = float(value)
        if weight < 0:
            raise UnscoreableError(f"{layer.name} weight must not be negative, got {weight}")
        layer_weights[layer] = weight

    try:
        deliver = float(thresholds["deliver"])
        block = float(thresholds["block"])
    except (KeyError, TypeError, ValueError) as exc:
        raise UnscoreableError(f"{config_path} is missing deliver/block thresholds") from exc

    if not 0.0 <= deliver <= block <= 1.0:
        raise UnscoreableError(
            f"thresholds must satisfy 0 <= deliver <= block <= 1, got {deliver} and {block}"
        )

    return ScoringWeights(
        layer_weights=layer_weights, deliver_threshold=deliver, block_threshold=block
    )


def _scoring_signals(result: LayerResult) -> list[DetectionSignal]:
    """Signals eligible to contribute: those that were actually computed.

    A signal carrying an `error` abstained - it could not be computed - so it
    is excluded rather than counted as a 0.0 finding.
    """
    return [s for s in result.signals if s.error is None]


def layer_score(result: LayerResult) -> float:
    """One layer's 0.0-1.0 score: the strength of its strongest signal.

    A layer that completed and found nothing scores 0.0, which is a genuine
    negative. Taking the maximum keeps the result bounded and stops a pile of
    weak signals from diluting one strong one. ARCHITECTURE.md does not
    specify the within-layer rule for Phase A; Phase B replaces this with the
    fitted model.
    """
    scores = [s.score for s in _scoring_signals(result)]
    return max(scores) if scores else 0.0


def layer_contributions(
    results: list[LayerResult], weights: ScoringWeights | None = None
) -> dict[DetectionLayer, float]:
    """Each completed layer's contribution to the composite, for explainability.

    Values sum to the composite score. Incomplete layers are absent: they
    contributed nothing and their weight went elsewhere.
    """
    weights = weights or load_weights()
    completed = [r for r in results if r.completed]
    total = sum(weights.weight_for(r.layer) for r in completed)
    if total <= 0:
        return {}
    return {
        r.layer: (weights.weight_for(r.layer) / total) * layer_score(r) for r in completed
    }


def risk_level(score_value: float, weights: ScoringWeights) -> RiskLevel:
    """Band a composite score, using the thresholds from configuration."""
    if score_value >= weights.block_threshold:
        return RiskLevel.HIGH
    if score_value >= weights.deliver_threshold:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def score(
    message_id: str,
    results: list[LayerResult],
    *,
    weights: ScoringWeights | None = None,
    summary: str | None = None,
) -> RiskAssessment:
    """Combine layer results into a RiskAssessment.

    Only completed layers contribute. Their weights are renormalized to sum to
    1 so a missing layer redistributes its share rather than scoring as 0 -
    the timeout does not pull the verdict toward clean.

    Raises UnscoreableError when no layer completed.
    """
    weights = weights or load_weights()

    completed = [r for r in results if r.completed]
    if not completed:
        raise UnscoreableError(
            "no layer completed, so there is no evidence to score; "
            "refusing to emit a clean result"
        )

    total_weight = sum(weights.weight_for(r.layer) for r in completed)
    if total_weight <= 0:
        raise UnscoreableError(
            "the layers that completed carry no configured weight, so the "
            "message cannot be scored"
        )

    composite = sum(
        (weights.weight_for(r.layer) / total_weight) * layer_score(r) for r in completed
    )
    composite = min(1.0, max(0.0, composite))  # guard float drift only

    signals: list[DetectionSignal] = []
    for result in results:
        signals.extend(result.signals)

    return RiskAssessment(
        message_id=message_id,
        score=composite,
        level=risk_level(composite, weights),
        signals=signals,
        summary=summary,
        layers_completed=[r.layer for r in completed],
    )
