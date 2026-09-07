"""Train the Layer 3 fusion model.

Per ARCHITECTURE.md section 4: a LogisticRegression over the four Layer 3
features - zero-width, urgency, perplexity, burstiness - replacing the
hand-summing of those features. Its `coef_` values are a deliverable: they are
the concrete statement of what Layer 3 actually learned each feature is worth.

Same separation as `train_urgency.py`. This module owns the model's shape and
the feature order; `layers/l3_nlp.py` owns the signal and knows only that
something with `predict_proba` and `classes_` was loaded off disk. It never
imports scikit-learn and never fits anything.

The artifact is `models/fusion_clf.joblib` - deliberately a different file from
`models/urgency_clf.joblib`. They are different models over different feature
spaces, and one must never be loaded in place of the other.

Corpus assembly and the CLI are still unimplemented: fusion trains on the
Layer 3 features extracted from the labelled corpus, which Phase 5 builds.
`build_model` and `save_model` are complete, because they are what the artifact
contract is made of and what the unit suite fits its fixtures with.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from sklearn.linear_model import LogisticRegression

__all__ = ["FUSION_FEATURE_ORDER", "PHISH_LABEL", "build_model", "save_model", "train"]

# The feature vector's column order, fixed here and mirrored by
# `layers.l3_nlp.FUSION_FEATURE_ORDER`. A persisted model is meaningless
# without it: the columns are four unrelated quantities on four different
# scales, so scoring them in a different order produces a confident wrong
# answer rather than an error.
FUSION_FEATURE_ORDER = (
    "zero_width",
    "urgency",
    "perplexity",
    "burstiness",
)

# The positive class. `layers.l3_nlp` locates this label in `classes_` rather
# than assuming a column index, so a model fitted with its classes in the other
# order still scores correctly.
PHISH_LABEL = 1


def build_model(*, random_state: int = 0) -> LogisticRegression:
    """The ARCHITECTURE.md section 4 fusion model.

    No vectorizer and no pipeline: the input is already four numbers. The
    features are on very different scales (a count, a probability, nats per
    token, a standard deviation), so a scaler would normally belong here - it
    is deliberately omitted for now because `coef_` is a deliverable and
    coefficients on the raw features are the ones that can be read aloud. If
    the Phase 5 refit shows the solver struggling, that is the first thing to
    revisit, and it is a change to this file alone.
    """
    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state,
    )


def train(
    features: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    random_state: int = 0,
) -> LogisticRegression:
    """Fit the fusion model. Rows follow `FUSION_FEATURE_ORDER`."""
    model = build_model(random_state=random_state)
    model.fit([list(row) for row in features], list(labels))
    return model


def save_model(model: LogisticRegression, path: str | Path) -> Path:
    """Persist a fitted model to `path`, creating the directory if needed."""
    import joblib

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)
    return destination
