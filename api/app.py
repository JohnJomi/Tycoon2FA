"""FastAPI application.

Endpoint surface specified in ARCHITECTURE.md section 8:
  POST /auth/login              OAuth redirect
  GET  /messages?limit=25       inbox list with cached verdicts
  POST /analyze/{message_id}    run pipeline, return RiskAssessment
  GET  /analyze/{message_id}    cached assessment
  POST /analyze/upload          .eml upload (demo path, no Gmail needed)
  GET  /health                  per-layer readiness incl. GPT-2 load state

**Only `POST /analyze/upload` is implemented.** Section 8 calls it the path
that "makes the whole system demonstrable without OAuth", and it is the one
endpoint the frontend needs to exist at all. The other five are left unwritten
rather than stubbed: a route returning a placeholder is indistinguishable to a
caller from one that works, which is the same failure the layer contract
spends so much effort avoiding.

    uploaded bytes -> parse_email -> run_layers -> score -> JSON

This module owns transport and serialization and nothing else. It does not
score, does not decide a verdict, and does not know what any layer looks for -
it calls the same three functions the CLI calls, in the same order, and
converts what comes back.

Why the response carries `layers` as well as `assessment`
---------------------------------------------------------
`RiskAssessment` records which layers completed, but not *why* the others did
not: `layers_completed` is a list of the ones that ran, and a layer's `error`
and `duration_ms` live on `LayerResult`, which the assessment does not carry.
Section 9 requires the detail view to mark degraded layers "not completed,"
never blank, and it cannot do that from the assessment alone. So both are
serialized, exactly as `cli/__main__.py` prints both.

Enum encoding
-------------
`RiskLevel` is serialized as its value ("low"/"medium"/"high"). `DetectionLayer`
is serialized as its **name** ("L1".."L4") rather than its int value, because
that is the spelling the rest of the payload already uses:
`DetectionSignal.qualified_name` is "L2/redirect_depth", and the CLI report
prints `layer.name`. Emitting `1` beside `"L2/redirect_depth"` in the same
document would make the client parse two spellings of one identity.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from core.models import (
    DetectionLayer,
    DetectionSignal,
    LayerResult,
    ParsedEmail,
    RiskAssessment,
)
from core.orchestrator import run_layers
from ingest.parser import EmailParseError, parse_email
from scoring.composite import UnscoreableError, layer_contributions, load_weights, score

__all__ = ["ALLOWED_ORIGINS", "app"]

# The Vite dev server. Deliberately not a wildcard: this endpoint accepts a
# file and returns an analysis of it, and there is no reason for an arbitrary
# origin to be able to post to it from a browser. Deployment adds its own
# origin here rather than this list growing speculatively.
ALLOWED_ORIGINS = ["http://localhost:5173"]

app = FastAPI(
    title="Tycoon2FA",
    description="Phishing detection pipeline. Upload an .eml, get a scored assessment.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Serialization
#
# Explicit, field by field. The models are plain dataclasses holding enums and
# a datetime, none of which FastAPI serializes on its own, and a response
# schema inferred from them would drift silently the next time a field moves.
# --------------------------------------------------------------------------


def _signal_json(signal: DetectionSignal) -> dict[str, Any]:
    """One DetectionSignal.

    `error` is preserved rather than folded into the score: a signal carrying
    an error abstained, and the UI has to show that as "unavailable" instead of
    as a 0.0 finding. `metadata` is passed through as the layer built it.
    """
    return {
        "layer": signal.layer.name,
        "name": signal.name,
        "qualified_name": signal.qualified_name,
        "score": signal.score,
        "severity": signal.severity.value,
        "evidence": signal.evidence,
        "metadata": dict(signal.metadata),
        "error": signal.error,
    }


def _layer_json(result: LayerResult) -> dict[str, Any]:
    """One LayerResult, including the reason an incomplete layer did not finish."""
    return {
        "layer": result.layer.name,
        "completed": result.completed,
        "signals": [_signal_json(s) for s in result.signals],
        "error": result.error,
        "duration_ms": result.duration_ms,
    }


def _assessment_json(assessment: RiskAssessment) -> dict[str, Any]:
    return {
        "message_id": assessment.message_id,
        "score": assessment.score,
        "level": assessment.level.value,
        "signals": [_signal_json(s) for s in assessment.signals],
        "layers_completed": [layer.name for layer in assessment.layers_completed],
        "scored_at": assessment.scored_at.isoformat(),
        "summary": assessment.summary,
    }


def _contributions_json(
    contributions: dict[DetectionLayer, float],
) -> dict[str, float]:
    """Layer -> contribution, keyed by layer name.

    Values sum to the composite score. Layers that did not complete are absent,
    which is `layer_contributions`' own convention and not a loss here.
    """
    return {layer.name: value for layer, value in contributions.items()}


def _message_json(email: ParsedEmail) -> dict[str, Any]:
    """The message metadata `cli/__main__.py` already reports.

    The same six facts its "Message" block prints, and no more: this is a
    detection result, not a mail reader, and the body is not echoed back.
    """
    return {
        "message_id": email.message_id,
        "from_addr": email.from_addr,
        "from_display": email.from_display,
        "subject": email.subject,
        "url_count": len(email.urls),
        "attachment_count": len(email.attachments),
    }


# --------------------------------------------------------------------------
# POST /analyze/upload
# --------------------------------------------------------------------------


@app.post("/analyze/upload")
async def analyze_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    """Analyze one uploaded .eml and return its assessment.

    The pipeline is called directly - `parse_email`, `run_layers`, `score` -
    rather than through `cli.__main__._analyze`, which wraps the orchestrator
    in `asyncio.run` and cannot be called from inside a running event loop.

    400 when the upload is empty or cannot be parsed as a message. 503 when
    nothing could be scored: `UnscoreableError` means no layer completed, and
    section 2 forbids reporting that as a clean result, so it is reported as
    the outage it is rather than as a verdict.
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(
            status_code=400,
            detail="the uploaded file is empty, so there is no message to analyze",
        )

    try:
        email = parse_email(raw)
    except EmailParseError as exc:
        raise HTTPException(
            status_code=400, detail=f"the upload could not be parsed as an email: {exc}"
        ) from exc

    results = await run_layers(email)

    try:
        weights = load_weights()
        assessment = score(email.message_id, results, weights=weights)
        contributions = layer_contributions(results, weights)
    except UnscoreableError as exc:
        # Not a 500: the request was fine and the pipeline ran. No layer
        # reached a conclusion, so there is no evidence to score, and emitting
        # a clean 0.0 would turn an outage into an all-clear.
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "assessment": _assessment_json(assessment),
        "layers": [_layer_json(r) for r in results],
        "contributions": _contributions_json(contributions),
        "message": _message_json(email),
    }
