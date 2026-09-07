"""Layer 3 - NLP / ML body analysis (project centerpiece).

Signals specified in ARCHITECTURE.md section 4:
  l3.zero_width    regex over zero-width and BOM codepoints   [implemented]
  l3.urgency       TF-IDF (word 1-2gram + char 3-5gram) -> LR   [implemented]
  l3.perplexity    GPT-2 (124M) mean per-token NLL              [implemented]
  l3.burstiness    std-dev of per-sentence PPL and of sentence length
  l3.fusion        LogisticRegression over the four features above

`zero_width`, `urgency` and `perplexity` are implemented. Burstiness and
fusion stay unimplemented here rather than being stubbed, so that a message
analysed today carries two honest absences instead of two fabricated 0.0s.

`zero_width` is the deterministic one: a compiled character class over the
subject and text body, no model, no corpus, no dependency on anything loaded
at startup.

`urgency` scores a persisted classifier that `training/train_urgency.py` fits
and writes to `models/urgency_clf.joblib`. **This module never fits, never
downloads and never reaches the network**; it loads one local artifact, once,
and calls `predict_proba` on it. It does not import scikit-learn at all - the
artifact is loaded through `joblib` inside the loader, and everything above
that is expressed against the `UrgencyModel` protocol, so the model's shape
stays a training concern.

`perplexity` measures GPT-2's mean per-token negative log-likelihood over the
body. It is a **measurement, not a verdict**: ARCHITECTURE.md defines no
mapping from nats-per-token to a 0-1 risk score, and inventing one here would
be a number nothing calibrated. The figure is carried in the metadata for
`l3.fusion` to consume, and the signal's own `score` stays 0.0 - see
`analyze_perplexity`.

A missing, unreadable or unusable artifact **fails closed**: the signal
abstains with an `error`, exactly as Layer 1 abstains on an unreachable WHOIS
server. It never returns 0.0, because "no classifier was available" and "this
message is calm" are different facts and collapsing them would report an
outage as an all-clear.

AI-generated-text detection is **not** part of this module. It lives behind
`layers/ai_text.py`, the provider-neutral seam; nothing here names a provider
or reaches for one.
"""

from __future__ import annotations

import math
import re
import threading
import unicodedata
from pathlib import Path
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from core.models import DetectionLayer, DetectionSignal, ParsedEmail, RiskLevel

__all__ = [
    "PERPLEXITY_MAX_TOKENS",
    "PERPLEXITY_MIN_TOKENS",
    "PERPLEXITY_MODEL_NAME",
    "URGENCY_BLOCK_THRESHOLD",
    "URGENCY_MIN_CHARS",
    "URGENCY_MODEL_PATH",
    "URGENCY_WARN_THRESHOLD",
    "ZERO_WIDTH_CHARS",
    "ZERO_WIDTH_RE",
    "ZERO_WIDTH_SCORE",
    "CausalLanguageModel",
    "PerplexityResult",
    "PerplexityUnavailable",
    "UrgencyModel",
    "UrgencyUnavailable",
    "analyze_perplexity",
    "compute_perplexity",
    "analyze_urgency",
    "analyze_zero_width",
    "default_perplexity_model",
    "default_urgency_model",
    "load_perplexity_model",
    "load_urgency_model",
    "reset_default_perplexity_model",
    "reset_default_urgency_model",
]

# The codepoints from the ARCHITECTURE.md section 4 table, in order.
#
# All five are invisible when rendered, which is the whole point: inserted
# between the letters of a brand name or a keyword they defeat exact-match
# string rules while leaving the text looking untouched to the reader. None of
# them has a legitimate use in the plain-text body of a business email. U+200C
# and U+200D do have legitimate uses in Arabic, Indic and emoji-sequence text,
# so this signal is scored as evidence rather than as a verdict - see the
# score note below.
ZERO_WIDTH_CHARS: tuple[str, ...] = (
    "\u200b",  # ZERO WIDTH SPACE
    "\u200c",  # ZERO WIDTH NON-JOINER
    "\u200d",  # ZERO WIDTH JOINER
    "\u2060",  # WORD JOINER
    "\ufeff",  # ZERO WIDTH NO-BREAK SPACE / BOM
)

ZERO_WIDTH_RE = re.compile("[" + "".join(ZERO_WIDTH_CHARS) + "]")

# Hand-assigned for Phase A exactly as the Layer 1 weights are, and refitted in
# Phase 5 against the labelled corpus. Deliberately below the Layer 1 hard-fail
# scores: zero-width characters are a strong indicator of deliberate filter
# evasion, but they also arrive by accident from copy-paste out of a rendered
# web page, and U+200C/U+200D are load-bearing in several scripts.
ZERO_WIDTH_SCORE = 0.55


def _codepoint_label(char: str) -> str:
    """`U+200B ZERO WIDTH SPACE` - the evidence string is what the UI shows."""
    name = unicodedata.name(char, "UNNAMED")
    return f"U+{ord(char):04X} {name}"


def analyze_zero_width(source: ParsedEmail) -> DetectionSignal:
    """Fire when zero-width or BOM codepoints appear in the subject or body.

    Scans `subject` and `body_text` only. The HTML body is deliberately not
    scanned here: its zero-width characters are frequently the mail client's
    own layout artefacts rather than the sender's, and judging them needs the
    rendered text, which Layer 2 produces.

    A message with neither field populated is a genuine negative, not an
    abstention: there was text to look at (however little), the check ran to
    completion, and it found nothing. Deterministic, offline, never raises.
    """
    fields = (("subject", source.subject or ""), ("body_text", source.body_text or ""))

    counts: dict[str, int] = {}
    locations: dict[str, list[str]] = {}
    for field_name, text in fields:
        for match in ZERO_WIDTH_RE.finditer(text):
            char = match.group()
            counts[char] = counts.get(char, 0) + 1
            locations.setdefault(field_name, [])
            if char not in locations[field_name]:
                locations[field_name].append(char)

    total = sum(counts.values())
    # Ordered by ZERO_WIDTH_CHARS so the evidence string is stable across runs
    # regardless of the order the characters happened to appear in.
    found = [c for c in ZERO_WIDTH_CHARS if c in counts]
    metadata = {
        "total": total,
        "counts": {f"U+{ord(c):04X}": counts[c] for c in found},
        "fields": {name: [f"U+{ord(c):04X}" for c in chars] for name, chars in locations.items()},
        "fired": total > 0,
    }

    if total == 0:
        return DetectionSignal(
            layer=DetectionLayer.L3,
            name="zero_width",
            score=0.0,
            severity=RiskLevel.LOW,
            evidence="No zero-width or byte-order-mark codepoints in the subject or text body.",
            metadata=metadata,
        )

    where = " and ".join(sorted(locations))
    listed = ", ".join(f"{_codepoint_label(c)} x{counts[c]}" for c in found)
    return DetectionSignal(
        layer=DetectionLayer.L3,
        name="zero_width",
        score=ZERO_WIDTH_SCORE,
        severity=RiskLevel.MEDIUM,
        evidence=(
            f"{total} invisible codepoint{'s' if total != 1 else ''} in the {where}: "
            f"{listed}. Zero-width characters are not visible to the reader and "
            f"are used to break up keywords that exact-match filters look for."
        ),
        metadata=metadata,
    )


# --------------------------------------------------------------------------
# 2. Urgency classifier
# --------------------------------------------------------------------------
#
# The model seam, kept the same shape as Layer 1's `WhoisLookup` and the
# `AITextDetector` in `layers/ai_text.py`: a protocol, an injectable default,
# and one exception class meaning "nothing was learned". The unit suite fits a
# small pipeline of its own and passes it in, so no test depends on the
# production artifact existing.


class UrgencyUnavailable(RuntimeError):
    """The urgency classifier could not be used, so nothing was learned.

    Raised for every condition that is *not* an answer about the text: no
    artifact on disk, an artifact that will not deserialize, an object that is
    not a classifier, a classifier that raises. One class rather than a
    hierarchy, matching `WhoisUnavailable` - the caller's decision is the same
    in every case, and the specific reason belongs in the message, which is
    what the abstaining signal's `error` carries.
    """


@runtime_checkable
class UrgencyModel(Protocol):
    """The whole of what this layer requires of the classifier.

    `classes_` is read rather than assumed: the positive column's index is
    looked up by label, so a model persisted with its classes in the other
    order scores correctly instead of scoring inverted and silently.
    """

    classes_: object

    def predict_proba(self, texts: list[str]) -> object: ...


# Where `training/train_urgency.py` writes the fitted pipeline. Relative to the
# repository root, resolved from this file so the working directory does not
# matter.
URGENCY_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "urgency_clf.joblib"

# The label `training.train_urgency.PHISH_LABEL` fits the positive class under.
# Duplicated as a literal rather than imported: importing the training module
# would drag scikit-learn into every analysis process to read one integer.
_PHISH_LABEL = 1

# Verdict bands from config/weights.yaml, applied to this signal's own score so
# its severity means the same thing the composite's bands do.
URGENCY_WARN_THRESHOLD = 0.35
URGENCY_BLOCK_THRESHOLD = 0.65

# Below this, the classifier is scoring noise. A one-word body has no phrasing
# for the word n-grams to read, and abstaining is the honest answer - the same
# reasoning ARCHITECTURE.md applies to short-body perplexity.
URGENCY_MIN_CHARS = 24

_model_lock = threading.Lock()
_model_cache: UrgencyModel | None = None


def load_urgency_model(path: str | Path | None = None) -> UrgencyModel:
    """Load the persisted classifier from disk. Never fits, never downloads.

    Raises `UrgencyUnavailable` if the artifact is absent, unreadable, or not
    something that can score text. The caller abstains rather than guessing.
    """
    target = Path(path) if path is not None else URGENCY_MODEL_PATH

    if not target.is_file():
        raise UrgencyUnavailable(f"no urgency model artifact at {target}")

    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - joblib ships with the model deps
        raise UrgencyUnavailable(f"joblib is not installed: {exc}") from exc

    try:
        model = joblib.load(target)
    except Exception as exc:
        raise UrgencyUnavailable(f"urgency model at {target} could not be loaded: {exc}") from exc

    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        raise UrgencyUnavailable(
            f"artifact at {target} is a {type(model).__name__}, not a probability classifier"
        )
    return model


def default_urgency_model(path: str | Path | None = None) -> UrgencyModel:
    """The process-wide singleton, loaded once.

    ARCHITECTURE.md is explicit that a per-email load destroys the latency
    budget. A failed load is *not* cached: an artifact that appears after a
    deploy should be picked up without a restart.
    """
    global _model_cache
    with _model_lock:
        if _model_cache is None:
            _model_cache = load_urgency_model(path)
        return _model_cache


def reset_default_urgency_model() -> None:
    """Drop the cached singleton. For tests and for a post-refit reload."""
    global _model_cache
    with _model_lock:
        _model_cache = None


def _urgency_signal(
    score: float,
    severity: RiskLevel,
    evidence: str,
    *,
    metadata: dict[str, object],
    error: str | None = None,
) -> DetectionSignal:
    return DetectionSignal(
        layer=DetectionLayer.L3,
        name="urgency",
        score=score,
        severity=severity,
        evidence=evidence,
        metadata={**metadata, "fired": score > 0.0 and error is None},
        error=error,
    )


def _positive_probability(model: UrgencyModel, text: str) -> float:
    """P(urgent) for one text, reading the positive column by label."""
    classes = list(getattr(model, "classes_", []))
    try:
        column = classes.index(_PHISH_LABEL)
    except ValueError as exc:
        raise UrgencyUnavailable(
            f"urgency model has no {_PHISH_LABEL!r} class; classes are {classes!r}"
        ) from exc

    try:
        row = model.predict_proba([text])[0]
    except Exception as exc:
        raise UrgencyUnavailable(f"urgency model failed to score the text: {exc}") from exc

    try:
        return float(row[column])
    except (IndexError, TypeError, ValueError) as exc:
        raise UrgencyUnavailable(f"urgency model returned no usable probability: {exc}") from exc


def _urgency_severity(score: float) -> RiskLevel:
    if score >= URGENCY_BLOCK_THRESHOLD:
        return RiskLevel.HIGH
    if score >= URGENCY_WARN_THRESHOLD:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def analyze_urgency(
    source: ParsedEmail,
    *,
    model: UrgencyModel | None = None,
    model_path: str | Path | None = None,
) -> DetectionSignal:
    """Score the subject and text body for phishing-style urgency.

    The score is the classifier's probability of the urgent class, reported as
    it is rather than thresholded into a 0/1 - a 0.9 and a 0.4 are different
    findings and the composite is entitled to see which one it got.

    `model` is injected by the unit suite; in production the module-level
    singleton is used. Deterministic: the same text scores the same number
    every time, because inference over a fitted LogisticRegression has no
    sampling in it. Offline - nothing here opens a socket.

    Abstains, with an `error` and a 0.0 that scoring excludes rather than
    counts, when the classifier is unavailable or the text is too short to
    judge. Never raises.
    """
    text = "\n".join(part for part in ((source.subject or ""), (source.body_text or "")) if part)
    stripped = text.strip()
    metadata: dict[str, object] = {"chars": len(stripped), "probability": None}

    if len(stripped) < URGENCY_MIN_CHARS:
        return _urgency_signal(
            0.0,
            RiskLevel.LOW,
            f"Subject and body together are {len(stripped)} characters, below the "
            f"{URGENCY_MIN_CHARS}-character floor, so the urgency classifier was not run.",
            metadata=metadata,
            error="text too short to classify",
        )

    try:
        classifier = model if model is not None else default_urgency_model(model_path)
        probability = _positive_probability(classifier, text)
    except UrgencyUnavailable as exc:
        return _urgency_signal(
            0.0,
            RiskLevel.LOW,
            "The urgency classifier was unavailable, so this message's language "
            "was not assessed. This is an absence of information, not a clean result.",
            metadata=metadata,
            error=str(exc),
        )

    score = min(1.0, max(0.0, probability))
    metadata["probability"] = score
    severity = _urgency_severity(score)
    verdict = "reads as urgent" if score >= URGENCY_WARN_THRESHOLD else "reads as ordinary"
    return _urgency_signal(
        score,
        severity,
        f"The urgency classifier scored this message {score:.2f} on the "
        f"phishing-urgency axis over {len(stripped)} characters of subject and "
        f"body: the language {verdict}.",
        metadata=metadata,
    )


# --------------------------------------------------------------------------
# 3. GPT-2 perplexity
# --------------------------------------------------------------------------
#
# ARCHITECTURE.md section 4: "GPT-2 (124M) mean per-token negative
# log-likelihood", truncated to 512 tokens, skipped under ~40 tokens.
#
# The seam is the same shape as the urgency one, for the same reason: this
# module must not learn what a Hugging Face model looks like. Everything above
# `load_perplexity_model` is expressed against `CausalLanguageModel`, which is
# two methods wide and says nothing about tensors, devices or checkpoints.
# `_TransformersCausalLM` is the only place `torch` and `transformers` are
# named, it is constructed only by the loader, and the unit suite never reaches
# it - the tests inject their own arithmetic-only model, so no test downloads a
# checkpoint or imports torch.


class PerplexityUnavailable(RuntimeError):
    """The language model could not be used, so nothing was learned.

    Same contract as `UrgencyUnavailable`: no checkpoint on disk or in the
    cache, a checkpoint that will not load, a model that raises mid-inference.
    The caller abstains; it never converts the failure into a number.
    """


@runtime_checkable
class CausalLanguageModel(Protocol):
    """The whole of what this layer requires of a language model.

    `encode` turns text into token ids. `token_log_likelihoods` returns
    log p(token_i | token_<i) in nats, one entry per *predicted* token - so for
    n input ids it returns n-1 values, because the first token is conditioned
    on nothing and has no likelihood under a causal model. That shift is the
    implementation's job, and `_TransformersCausalLM` documents how it does it.
    """

    def encode(self, text: str) -> Sequence[int]: ...

    def token_log_likelihoods(self, token_ids: Sequence[int]) -> Sequence[float]: ...


# ARCHITECTURE.md section 4 names the checkpoint and both bounds.
PERPLEXITY_MODEL_NAME = "gpt2"
PERPLEXITY_MAX_TOKENS = 512
PERPLEXITY_MIN_TOKENS = 40


@dataclass(frozen=True)
class PerplexityResult:
    """One measurement over one body.

    `mean_nll` is the metric ARCHITECTURE.md specifies, in nats per token.
    `perplexity` is `exp(mean_nll)`, carried alongside because it is the
    figure the UI shows and the one a reader recognizes - it is a
    presentation of the same number, not a second measurement.
    """

    mean_nll: float
    perplexity: float
    token_count: int
    truncated: bool


_perplexity_lock = threading.Lock()
_perplexity_cache: CausalLanguageModel | None = None


class _TransformersCausalLM:
    """GPT-2 behind `CausalLanguageModel`. The only torch-aware code here.

    Constructed by `load_perplexity_model` and nowhere else. Kept in this file
    rather than in `training/` because nothing about it is trained: it is an
    adapter over a pretrained checkpoint, and it has no fitting step to own.
    """

    def __init__(self, tokenizer: object, model: object, torch_module: object) -> None:
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch_module

    def encode(self, text: str) -> Sequence[int]:
        return list(self._tokenizer.encode(text))

    def token_log_likelihoods(self, token_ids: Sequence[int]) -> Sequence[float]:
        """Per-token log-likelihood from shifted logits and labels.

        The shift is the whole correctness question. Position i's logits
        predict token i+1, so the logits are dropped at the last position and
        the labels at the first; averaging without that shift scores every
        token against the distribution for the *previous* one and produces a
        confidently wrong number. `log_softmax` is taken over the vocabulary
        and the label's own entry is gathered out, which is the definition of
        log p(token_i | token_<i) rather than any library's own "perplexity".
        """
        torch = self._torch
        ids = torch.tensor([list(token_ids)], dtype=torch.long)
        with torch.no_grad():
            logits = self._model(ids).logits

        shifted_logits = logits[:, :-1, :]
        shifted_labels = ids[:, 1:]
        log_probs = torch.log_softmax(shifted_logits, dim=-1)
        gathered = log_probs.gather(-1, shifted_labels.unsqueeze(-1)).squeeze(-1)
        return [float(value) for value in gathered[0]]


def load_perplexity_model(model_name: str | None = None) -> CausalLanguageModel:
    """Load GPT-2 from the local Hugging Face cache. Never downloads.

    `local_files_only=True` on both loads: ARCHITECTURE.md forbids network
    access during analysis, and a checkpoint that has to be fetched at scoring
    time is exactly the cold start the risk table warns about. An absent
    checkpoint raises `PerplexityUnavailable` and the signal abstains, which is
    the honest outcome - fetching it here would trade a stated absence for an
    unbounded stall.
    """
    name = model_name or PERPLEXITY_MODEL_NAME

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise PerplexityUnavailable(f"transformers/torch are not installed: {exc}") from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(name, local_files_only=True)
        model.eval()
    except Exception as exc:
        raise PerplexityUnavailable(f"GPT-2 checkpoint {name!r} could not be loaded: {exc}") from exc

    return _TransformersCausalLM(tokenizer, model, torch)


def default_perplexity_model(model_name: str | None = None) -> CausalLanguageModel:
    """The process-wide singleton, loaded once.

    ARCHITECTURE.md: "GPT-2 loaded once as a module-level singleton. Loading
    per email is the single easiest way to destroy the latency budget." As
    with the urgency model, a *failed* load is not cached.
    """
    global _perplexity_cache
    with _perplexity_lock:
        if _perplexity_cache is None:
            _perplexity_cache = load_perplexity_model(model_name)
        return _perplexity_cache


def reset_default_perplexity_model() -> None:
    """Drop the cached singleton. For tests and for a deliberate reload."""
    global _perplexity_cache
    with _perplexity_lock:
        _perplexity_cache = None


def compute_perplexity(model: CausalLanguageModel, text: str) -> PerplexityResult:
    """Mean per-token negative log-likelihood over `text`, and its exponential.

    Truncates to `PERPLEXITY_MAX_TOKENS` before scoring and raises
    `PerplexityUnavailable` below `PERPLEXITY_MIN_TOKENS`, per
    ARCHITECTURE.md section 4. Raises rather than returning a sentinel so the
    caller has one thing to catch and cannot accidentally average a placeholder
    into the fusion features later.
    """
    try:
        token_ids = list(model.encode(text))
    except Exception as exc:
        raise PerplexityUnavailable(f"tokenization failed: {exc}") from exc

    total = len(token_ids)
    if total < PERPLEXITY_MIN_TOKENS:
        raise PerplexityUnavailable("body too short")

    truncated = total > PERPLEXITY_MAX_TOKENS
    window = token_ids[:PERPLEXITY_MAX_TOKENS]

    try:
        log_likelihoods = list(model.token_log_likelihoods(window))
    except Exception as exc:
        raise PerplexityUnavailable(f"language model failed to score the text: {exc}") from exc

    # n input tokens yield n-1 predictions; the first token is conditioned on
    # nothing. A model returning anything else is not implementing the shift.
    if len(log_likelihoods) != len(window) - 1:
        raise PerplexityUnavailable(
            f"language model returned {len(log_likelihoods)} log-likelihoods "
            f"for {len(window)} tokens; expected {len(window) - 1}"
        )

    try:
        values = [float(value) for value in log_likelihoods]
    except (TypeError, ValueError) as exc:
        raise PerplexityUnavailable(f"language model returned non-numeric output: {exc}") from exc

    if not all(math.isfinite(value) for value in values):
        raise PerplexityUnavailable("language model returned a non-finite log-likelihood")

    mean_nll = -sum(values) / len(values)
    try:
        perplexity = math.exp(mean_nll)
    except OverflowError as exc:
        raise PerplexityUnavailable(f"perplexity overflowed at {mean_nll} nats/token") from exc

    return PerplexityResult(
        mean_nll=mean_nll,
        perplexity=perplexity,
        token_count=len(window),
        truncated=truncated,
    )


def _perplexity_signal(
    evidence: str,
    *,
    metadata: dict[str, object],
    error: str | None = None,
) -> DetectionSignal:
    """Always score 0.0. See `analyze_perplexity` for why that is not a verdict."""
    return DetectionSignal(
        layer=DetectionLayer.L3,
        name="perplexity",
        score=0.0,
        severity=RiskLevel.LOW,
        evidence=evidence,
        metadata={**metadata, "fired": False},
        error=error,
    )


def analyze_perplexity(
    source: ParsedEmail,
    *,
    model: CausalLanguageModel | None = None,
    model_name: str | None = None,
) -> DetectionSignal:
    """Measure GPT-2 mean per-token NLL over the message body.

    **This signal reports a measurement, not a risk.** ARCHITECTURE.md defines
    `l3.perplexity` as a number in nats per token and specifies no mapping from
    it to the 0-1 risk scale; `l3.fusion` is the component that learns what a
    given perplexity is worth. So `score` is 0.0 in every branch and the
    measurement lives in `metadata["mean_nll"]` / `metadata["perplexity"]`,
    where fusion will read it. Squeezing nats into 0-1 with a hand-picked curve
    would be a number nothing calibrated, presented as though something had.

    The body text alone is scored, not the subject: a subject line is a handful
    of tokens of headline-register English, and folding it in shifts the mean
    without saying anything about how the message was written.

    Abstains - `error` set, `mean_nll` and `perplexity` `None` - when the model
    is unavailable, when it fails mid-inference, or when the body is under
    `PERPLEXITY_MIN_TOKENS`. Never fabricates a measurement. Never raises, and
    never opens a socket: `load_perplexity_model` is `local_files_only`.
    """
    text = (source.body_text or "").strip()
    metadata: dict[str, object] = {
        "mean_nll": None,
        "perplexity": None,
        "token_count": None,
        "truncated": False,
        "model": model_name or PERPLEXITY_MODEL_NAME,
    }

    try:
        language_model = model if model is not None else default_perplexity_model(model_name)
        result = compute_perplexity(language_model, text)
    except PerplexityUnavailable as exc:
        reason = str(exc)
        if reason == "body too short":
            evidence = (
                f"The body is under {PERPLEXITY_MIN_TOKENS} tokens, which is too "
                f"short for perplexity to mean anything, so it was not measured. "
                f"This is an abstention, not a clean result."
            )
        else:
            evidence = (
                "The GPT-2 perplexity model was unavailable, so this message's "
                "text was not measured. This is an absence of information, not a "
                "clean result."
            )
        return _perplexity_signal(evidence, metadata=metadata, error=reason)

    metadata.update(
        mean_nll=result.mean_nll,
        perplexity=result.perplexity,
        token_count=result.token_count,
        truncated=result.truncated,
    )
    window = (
        f" over the first {result.token_count} of more than {PERPLEXITY_MAX_TOKENS} tokens"
        if result.truncated
        else f" over {result.token_count} tokens"
    )
    return _perplexity_signal(
        f"GPT-2 mean per-token negative log-likelihood is {result.mean_nll:.3f} "
        f"nats{window}, a perplexity of {result.perplexity:.1f}. This is a "
        f"measurement of the text, not a risk score on its own.",
        metadata=metadata,
    )
