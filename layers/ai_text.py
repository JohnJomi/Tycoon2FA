"""Provider-neutral AI-generated-text detection: the seam Layer 3 depends on.

    Layer 3 -> AITextDetector -> a provider -> some vendor's API

This module is the whole of what Layer 3 is allowed to know. It names no
vendor, carries no endpoint, and imports no HTTP client. Swapping the provider
- to a local model, or to a different service - is a change in
`layers/providers/`, not here and not in Layer 3.

The contract mirrors Layer 1's `WhoisLookup` deliberately, because it is the
same shape of problem: one network-touching seam, injected so the unit suite
can exercise the layer without a socket, with a two-way contract that keeps
"the detector answered" distinct from "the detector could not be reached".

**Returning** an `AITextVerdict` is an answer about the text. **Raising** means
nothing was learned, and the caller must abstain rather than record a 0.0 -
which would read as "checked, and this text is human-written". That is the
distinction ARCHITECTURE.md section 2 exists to protect, and it is exactly the
one a detection layer cannot afford to collapse.

Scope note: this module defines the seam only. Nothing here scores a message,
and no signal is emitted from it. Per the provider calibration recorded under
`reports/`, no provider's probability is fit to drive the composite risk score
yet.

This file names no vendor, by design and by test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = [
    "AITextDetector",
    "AITextTimeout",
    "AITextUnavailable",
    "AITextVerdict",
    "NullAITextDetector",
]


class AITextUnavailable(RuntimeError):
    """The detection could not be completed, so nothing was learned.

    Raised for every condition that is *not* an answer about the text: no API
    key configured, a rejected key, a throttled request, a refused socket, a
    malformed response, text the provider will not accept. The caller abstains.

    One class rather than a hierarchy per failure mode, matching
    `WhoisUnavailable`: the caller's decision is the same in every case - do
    not claim to know - and the specific reason belongs in the message, which
    is what the abstaining signal's `error` will carry.
    """


class AITextTimeout(AITextUnavailable, TimeoutError):
    """The provider did not answer inside its budget.

    A slow provider is not a provider that said "human". Kept a subclass of
    `AITextUnavailable` so callers that only care about "no answer" need one
    except clause, with `TimeoutError` in the bases so ordinary timeout
    handling still catches it - the same arrangement as `WhoisTimeout`.
    """


@dataclass(frozen=True)
class AITextVerdict:
    """One provider's answer about one piece of text.

    Provider-neutral on purpose. `ai_generated_probability` is the normalized
    0.0-1.0 quantity every provider is reduced to, so Layer 3 never learns what
    a given vendor happens to call its score or how that score is shaped.
    `provider` and `model` are metadata *about* the answer rather than part of
    it - enough to attribute a result and to tell two providers' numbers apart
    in a stored verdict, without letting either leak into the layer's logic.

    Frozen, like every other result type in this project: a verdict is a record
    of what a provider said and is not edited afterwards.

    A verdict is not a risk score. It says how machine-written the text looks,
    which is a different question from whether the message is phishing, and the
    calibration is explicit that the two must not be conflated yet.
    """

    ai_generated_probability: float
    provider: str
    model: str | None = None
    word_count: int = 0

    def __post_init__(self) -> None:
        value = self.ai_generated_probability
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                "ai_generated_probability must be a number, got "
                f"{type(value).__name__}"
            )
        if math.isnan(value) or math.isinf(value):
            raise ValueError(
                f"ai_generated_probability must be finite, got {value!r}"
            )
        if not 0.0 <= value <= 1.0:
            # A provider that returns something outside the unit interval has
            # not been normalized, and guessing at its scale would be worse
            # than refusing it.
            raise ValueError(
                f"ai_generated_probability must be between 0.0 and 1.0, got {value!r}"
            )
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("provider must be a non-empty string")
        if isinstance(self.word_count, bool) or not isinstance(self.word_count, int):
            raise TypeError(
                f"word_count must be an int, got {type(self.word_count).__name__}"
            )
        if self.word_count < 0:
            raise ValueError(f"word_count must not be negative, got {self.word_count!r}")


@runtime_checkable
class AITextDetector(Protocol):
    """The one network-touching seam Layer 3 will depend on.

    Implementations return an `AITextVerdict` or raise `AITextUnavailable`
    (or `AITextTimeout`). Any other exception is still treated as
    non-authoritative by the caller: an unrecognized failure is precisely the
    case where claiming to know something would be wrong.
    """

    def detect(self, text: str) -> AITextVerdict: ...


class NullAITextDetector:
    """A detector that always abstains, and never opens a socket.

    The honest default for "no provider is configured". It is not a detector
    that finds nothing - that would be a claim - so it raises rather than
    returning a 0.0 verdict, and the caller's abstention path handles it the
    same way it handles an outage.

    Useful in three places: as the default when no key is present, in tests
    that must not reach the network, and as the switch that turns AI-text
    detection off without removing the code that calls it.
    """

    provider = "null"

    def detect(self, text: str) -> AITextVerdict:
        raise AITextUnavailable("no AI-text detector is configured")
