"""Core data contracts shared by every other module.

Per ARCHITECTURE.md section 2 this module is the base of the dependency graph:
everything else imports it and it imports nothing beyond the standard library.

It is a data-contract module only. It performs no detection, no scoring, no
parsing, no I/O and no network access. Layers populate these structures; the
scoring component consumes them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Any

__all__ = [
    "RiskLevel",
    "DetectionLayer",
    "URLSource",
    "Attachment",
    "ExtractedURL",
    "ParsedEmail",
    "DetectionSignal",
    "LayerResult",
    "RiskAssessment",
]


# --------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------


class RiskLevel(str, Enum):
    """Final risk banding for an assessment, and severity for a single signal.

    Maps onto the deliver / warn / block verdict bands in ARCHITECTURE.md
    section 6: LOW = deliver, MEDIUM = warn, HIGH = block. String values so the
    enum serializes cleanly to JSON and YAML later.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DetectionLayer(IntEnum):
    """The four independent detection layers.

    An IntEnum because ARCHITECTURE.md numbers the layers 1..4 and orders them;
    this keeps the numeric contract while giving callers named members.
    """

    L1 = 1  # Header & domain intelligence
    L2 = 2  # URL & redirect chain analysis
    L3 = 3  # NLP & body content analysis
    L4 = 4  # Live threat intelligence correlation


class URLSource(str, Enum):
    """Where in the email a URL was found.

    Mirrors the extraction sites listed in ARCHITECTURE.md section 3, plus
    QR_CODE for URLs decoded from images by Layer 2.
    """

    ANCHOR_HREF = "anchor_href"
    IMG_SRC = "img_src"
    FORM_ACTION = "form_action"
    PLAIN_TEXT = "plain_text"
    HEADER = "header"
    QR_CODE = "qr_code"


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------


def _require_non_empty(value: str, field_name: str) -> None:
    """Reject a missing or whitespace-only required string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_unit_interval(value: float, field_name: str) -> None:
    """Reject a score outside the normalized 0.0-1.0 range."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number, got {type(value).__name__}")
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"{field_name} must be a finite number, got {value!r}")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0, got {value!r}")


# --------------------------------------------------------------------------
# Email structures
# --------------------------------------------------------------------------


@dataclass
class Attachment:
    """Metadata for one attachment or inline image.

    Metadata only - the payload is deliberately not carried here. An inline
    (cid:-referenced) image is one whose content_id is set, which is what
    Layer 2 needs when hunting for QR codes.
    """

    filename: str
    content_type: str
    size_bytes: int = 0
    content_id: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.content_type, "content_type")
        if self.size_bytes < 0:
            raise ValueError(f"size_bytes must not be negative, got {self.size_bytes!r}")

    @property
    def is_inline(self) -> bool:
        """True when the part is referenced from the HTML body by cid:."""
        return self.content_id is not None


@dataclass
class ExtractedURL:
    """A URL found in an email, with room for what Layer 2 learns later.

    Construction is offline and purely descriptive. redirect_chain and
    final_url stay empty/None until Layer 2 resolves them; this module never
    follows a redirect.
    """

    url: str
    source: URLSource
    anchor_text: str | None = None
    redirect_chain: list[str] = field(default_factory=list)
    final_url: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.url, "url")
        if not isinstance(self.source, URLSource):
            raise TypeError(f"source must be a URLSource, got {type(self.source).__name__}")

    @property
    def redirect_depth(self) -> int:
        """Number of hops recorded so far. 0 when unresolved or direct."""
        return len(self.redirect_chain)


@dataclass
class ParsedEmail:
    """A normalized email handed to the detection pipeline.

    Deliberately not coupled to Gmail or to any MIME parser: ingest/parser.py
    populates it from an RFC-822 message, but nothing here knows how. `raw` is
    optional so an email can be constructed in tests without carrying bytes.

    `from_addr` and `from_display` may be empty. A message with a missing,
    empty or unparseable From header is still a message worth analysing - the
    absent sender is itself evidence for Layer 1 - so ingest represents what
    was actually there rather than rejecting the message or inventing a
    placeholder. The parser extracts what exists; detection decides whether
    what exists is suspicious.
    """

    message_id: str
    from_addr: str
    from_display: str = ""
    to_addrs: list[str] = field(default_factory=list)
    subject: str = ""
    body_text: str = ""
    body_html: str | None = None
    reply_to: str | None = None
    urls: list[ExtractedURL] = field(default_factory=list)
    headers: dict[str, list[str]] = field(default_factory=dict)
    received_chain: list[str] = field(default_factory=list)
    attachments: list[Attachment] = field(default_factory=list)
    raw: bytes | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.message_id, "message_id")

    @property
    def inline_images(self) -> list[Attachment]:
        """Attachments referenced inline by cid:, per ARCHITECTURE.md section 2."""
        return [a for a in self.attachments if a.is_inline]


# --------------------------------------------------------------------------
# Detection output
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectionSignal:
    """One piece of evidence produced by any detection layer.

    Deliberately generic: there is no per-layer subclass, and nothing here
    assumes a given signal name belongs to a given layer. A layer identifies
    itself via `layer` and names its own signal, e.g. DetectionLayer.L2 /
    "redirect_depth".

    Frozen, per ARCHITECTURE.md section 2 - a signal is a record of what was
    observed and must not be edited after the fact.

    `evidence` is required and must be non-empty: per ARCHITECTURE.md a bare
    score demonstrates nothing, and the evidence strings are what the UI shows.
    `error` carries the reason a signal could not be computed, so a layer can
    abstain honestly instead of emitting a fabricated 0.0.
    """

    layer: DetectionLayer
    name: str
    score: float
    severity: RiskLevel
    evidence: str
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.layer, DetectionLayer):
            raise TypeError(f"layer must be a DetectionLayer, got {type(self.layer).__name__}")
        if not isinstance(self.severity, RiskLevel):
            raise TypeError(f"severity must be a RiskLevel, got {type(self.severity).__name__}")
        _require_non_empty(self.name, "name")
        _require_non_empty(self.evidence, "evidence")
        _require_unit_interval(self.score, "score")

    @property
    def qualified_name(self) -> str:
        """Stable identifier of the form "L2/redirect_depth"."""
        return f"{self.layer.name}/{self.name}"


@dataclass
class LayerResult:
    """What one detection layer returned, per ARCHITECTURE.md section 2.

    **`completed=False` means the layer failed or abstained and must not be
    interpreted as "no threat found."** A completed layer with zero signals is
    a genuine negative: it ran, it looked, it found nothing. An incomplete
    layer is an absence of information - it timed out, crashed, or could not
    reach a dependency. Collapsing the two into "clean" silently turns an
    outage into an all-clear, which is why scoring renormalizes over completed
    layers rather than scoring a missing layer as 0.

    `error` carries the reason a layer could not complete. `duration_ms` feeds
    the latency budget in section 5.
    """

    layer: DetectionLayer
    completed: bool
    signals: list[DetectionSignal] = field(default_factory=list)
    error: str | None = None
    duration_ms: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.layer, DetectionLayer):
            raise TypeError(f"layer must be a DetectionLayer, got {type(self.layer).__name__}")
        if not isinstance(self.completed, bool):
            raise TypeError(f"completed must be a bool, got {type(self.completed).__name__}")
        if self.duration_ms < 0:
            raise ValueError(f"duration_ms must not be negative, got {self.duration_ms!r}")


@dataclass
class RiskAssessment:
    """The output of the future scoring pipeline for one message.

    This module stores the result; it does not compute it. Combining signals
    into `score` and banding it into `level` is scoring/composite.py's job.
    """

    message_id: str
    score: float
    level: RiskLevel
    signals: list[DetectionSignal] = field(default_factory=list)
    summary: str | None = None
    layers_completed: list[DetectionLayer] = field(default_factory=list)
    scored_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        _require_non_empty(self.message_id, "message_id")
        _require_unit_interval(self.score, "score")
        if not isinstance(self.level, RiskLevel):
            raise TypeError(f"level must be a RiskLevel, got {type(self.level).__name__}")

    def signals_for(self, layer: DetectionLayer) -> list[DetectionSignal]:
        """Signals contributed by one layer, for grouped display in the UI."""
        return [s for s in self.signals if s.layer is layer]
