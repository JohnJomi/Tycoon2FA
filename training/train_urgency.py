"""Train the Layer 3 urgency classifier.

Per ARCHITECTURE.md section 4 and ROADMAP.md Phase 2: TF-IDF (word 1-2gram +
char 3-5gram) -> LogisticRegression over the Nazario phishing corpus vs. Enron
ham, persisted as `models/urgency_clf.joblib` with held-out accuracy recorded
in `eval/results/`.

This module owns the *shape* of the model. `layers/l3_nlp.py` owns the signal
and knows only that something with a `predict_proba` was loaded off disk - it
never imports scikit-learn, never fits anything, and never learns what the
feature space looks like. That separation is what lets the classifier be
refitted, or replaced outright, without touching the detection layer.

The char n-grams are not decoration: per ARCHITECTURE.md they survive the
zero-width obfuscation that `l3.zero_width` detects, so a body whose keywords
have been broken up with U+200B still scores through the character view.

Corpus loading and the CLI are deliberately still unimplemented - no corpus is
downloaded here. `build_pipeline` and `save_model` are complete, because they
are what the artifact contract is made of and what the unit suite fits its
fixtures with.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

__all__ = ["PHISH_LABEL", "build_pipeline", "save_model", "train"]

# The positive class. Persisted models must use this label for "urgent /
# phishing-style", because `layers.l3_nlp` reads the probability of exactly
# this class out of `classes_` rather than assuming column order.
PHISH_LABEL = 1


def build_pipeline(*, random_state: int = 0) -> Pipeline:
    """The ARCHITECTURE.md section 4 model: TF-IDF union -> LogisticRegression.

    Word 1-2 grams carry the phrasing ("act now", "your account will be");
    char_wb 3-5 grams carry the sub-word shape and are what survives
    obfuscation and misspelling. `char_wb` rather than `char` so n-grams do not
    straddle word boundaries and re-learn the word view.

    `random_state` is set on the solver so a refit over the same corpus
    produces the same coefficients - the `coef_` values are a deliverable.
    """
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=1,
                    lowercase=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    sublinear_tf=True,
                    min_df=1,
                    lowercase=True,
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def train(texts: Sequence[str], labels: Sequence[int], *, random_state: int = 0) -> Pipeline:
    """Fit the pipeline. `labels` uses `PHISH_LABEL` for the positive class."""
    pipeline = build_pipeline(random_state=random_state)
    pipeline.fit(list(texts), list(labels))
    return pipeline


def save_model(pipeline: Pipeline, path: str | Path) -> Path:
    """Persist a fitted pipeline to `path`, creating the directory if needed."""
    import joblib

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, destination)
    return destination
