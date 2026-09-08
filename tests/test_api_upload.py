"""Unit tests for POST /analyze/upload in api/app.py.

Transport and serialization only. The pipeline itself is tested by the layer
suites; what matters here is that what the layers produced survives the trip
into JSON without losing the distinctions the layer contract is built on:

  - an abstaining signal (`error` set) must not arrive looking like a 0.0
    finding, and
  - an incomplete layer must not arrive looking like a layer that ran and
    found nothing.

`run_layers` is replaced with a fake in every test. It is the seam the
endpoint imports, so patching it keeps these tests off the network and away
from the GPT-2 checkpoint, and it lets one test hand the endpoint a pipeline
where nothing completed - the 503 case, which is otherwise reachable only by
breaking the machine. The `LayerResult` and `DetectionSignal` objects the fake
returns are real ones, so the serializers are exercised against the actual
models rather than against stand-ins.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from api.app import ALLOWED_ORIGINS, app
from core.models import DetectionLayer, DetectionSignal, LayerResult, RiskLevel
from ingest.parser import EmailParseError

EML = b"""\
From: Microsoft Account Team <security@account-verify-ms.example>
To: victim@corp.example
Subject: Action required
Message-ID: <api-test-0001@account-verify-ms.example>
Content-Type: text/html

<html><body><a href="https://evil.example/login">Microsoft</a></body></html>
"""


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def upload(client: TestClient, raw: bytes = EML):
    return client.post(
        "/analyze/upload", files={"file": ("message.eml", raw, "message/rfc822")}
    )


# --------------------------------------------------------------------------
# Fake pipelines
# --------------------------------------------------------------------------


def _signal(layer, name, sc, *, error=None) -> DetectionSignal:
    return DetectionSignal(
        layer=layer,
        name=name,
        score=sc,
        severity=RiskLevel.HIGH if sc >= 0.65 else RiskLevel.LOW,
        evidence=f"evidence for {name}",
        metadata={"fired": sc > 0.0, "checked": ["a", "b"]},
        error=error,
    )


def _mixed_results() -> list[LayerResult]:
    """One of each state the UI has to be able to tell apart."""
    return [
        LayerResult(
            layer=DetectionLayer.L1,
            completed=True,
            signals=[
                _signal(DetectionLayer.L1, "dmarc_fail", 0.85),
                # Abstained: computed nothing, and must not read as a finding.
                _signal(DetectionLayer.L1, "domain_age_lt_7d", 0.0, error="no WHOIS client"),
            ],
            duration_ms=30,
        ),
        # Completed with no signals at all: a genuine negative.
        LayerResult(layer=DetectionLayer.L2, completed=True, signals=[], duration_ms=1),
        # Did not complete: an absence of information, never an all-clear.
        LayerResult(
            layer=DetectionLayer.L3,
            completed=False,
            signals=[],
            error="Layer3Uninformative: no Layer 3 signal reached a conclusion",
            duration_ms=5494,
        ),
        LayerResult(
            layer=DetectionLayer.L4,
            completed=True,
            signals=[_signal(DetectionLayer.L4, "url_ioc", 0.0)],
            duration_ms=9,
        ),
    ]


def _nothing_completed() -> list[LayerResult]:
    return [
        LayerResult(layer=layer, completed=False, signals=[], error="dependency missing")
        for layer in DetectionLayer
    ]


@pytest.fixture
def wired(monkeypatch):
    """Patch the orchestrator seam with a fake returning the given results."""

    def install(results):
        async def fake_run_layers(email, **kwargs):
            return results

        monkeypatch.setattr("api.app.run_layers", fake_run_layers)

    return install


# --------------------------------------------------------------------------
# 1. A valid upload is analyzed
# --------------------------------------------------------------------------


def test_a_valid_eml_upload_returns_200(client, wired):
    wired(_mixed_results())

    assert upload(client).status_code == 200


def test_the_response_carries_all_four_top_level_sections(client, wired):
    wired(_mixed_results())

    body = upload(client).json()

    assert sorted(body) == ["assessment", "contributions", "layers", "message"]


def test_the_whole_body_is_json_serializable(client, wired):
    """No dataclass, enum or datetime escapes into the payload."""
    wired(_mixed_results())

    json.dumps(upload(client).json())  # raises if anything did


# --------------------------------------------------------------------------
# 2. The assessment
# --------------------------------------------------------------------------


def test_the_assessment_preserves_every_field(client, wired):
    wired(_mixed_results())

    assessment = upload(client).json()["assessment"]

    assert sorted(assessment) == [
        "layers_completed",
        "level",
        "message_id",
        "score",
        "scored_at",
        "signals",
        "summary",
    ]
    assert assessment["message_id"] == "<api-test-0001@account-verify-ms.example>"
    # L1 0.85 and L2/L4 clean, renormalized over the three that completed.
    assert assessment["score"] == pytest.approx(0.85 * (0.30 / 0.80))
    assert assessment["level"] in {"low", "medium", "high"}
    assert assessment["layers_completed"] == ["L1", "L2", "L4"]


def test_scored_at_is_an_iso_8601_string(client, wired):
    from datetime import datetime

    wired(_mixed_results())

    scored_at = upload(client).json()["assessment"]["scored_at"]

    assert isinstance(scored_at, str)
    assert datetime.fromisoformat(scored_at).tzinfo is not None


def test_summary_is_carried_through_as_null(client, wired):
    """`score()` never generates one, so the field is present and null."""
    wired(_mixed_results())

    assert upload(client).json()["assessment"]["summary"] is None


# --------------------------------------------------------------------------
# 3. Signals
# --------------------------------------------------------------------------


def test_a_signal_preserves_every_field(client, wired):
    wired(_mixed_results())

    signals = upload(client).json()["assessment"]["signals"]
    dmarc = next(s for s in signals if s["name"] == "dmarc_fail")

    assert sorted(dmarc) == [
        "error",
        "evidence",
        "layer",
        "metadata",
        "name",
        "qualified_name",
        "score",
        "severity",
    ]
    assert dmarc["layer"] == "L1"
    assert dmarc["qualified_name"] == "L1/dmarc_fail"
    assert dmarc["score"] == 0.85
    assert dmarc["severity"] == "high"
    assert dmarc["evidence"] == "evidence for dmarc_fail"
    assert dmarc["metadata"] == {"fired": True, "checked": ["a", "b"]}
    assert dmarc["error"] is None


def test_an_abstaining_signal_keeps_its_error_and_does_not_look_like_a_finding(
    client, wired
):
    """The distinction the whole layer contract rests on, at the JSON boundary."""
    wired(_mixed_results())

    signals = upload(client).json()["assessment"]["signals"]
    abstained = next(s for s in signals if s["name"] == "domain_age_lt_7d")

    assert abstained["error"] == "no WHOIS client"
    assert abstained["score"] == 0.0
    # A client can tell this apart from a real 0.0 by the error alone.
    fired_clean = next(s for s in signals if s["name"] == "url_ioc")
    assert fired_clean["error"] is None and fired_clean["score"] == 0.0


# --------------------------------------------------------------------------
# 4. Layers - the degraded states section 9 requires the UI to show
# --------------------------------------------------------------------------


def test_a_layer_preserves_every_field(client, wired):
    wired(_mixed_results())

    layers = upload(client).json()["layers"]

    assert [l["layer"] for l in layers] == ["L1", "L2", "L3", "L4"]
    assert sorted(layers[0]) == [
        "completed",
        "duration_ms",
        "error",
        "layer",
        "signals",
    ]


def test_an_incomplete_layer_survives_serialization_with_its_reason(client, wired):
    wired(_mixed_results())

    l3 = next(l for l in upload(client).json()["layers"] if l["layer"] == "L3")

    assert l3["completed"] is False
    assert "Layer3Uninformative" in l3["error"]
    assert l3["signals"] == []
    assert l3["duration_ms"] == 5494


def test_a_completed_empty_layer_is_distinguishable_from_an_incomplete_one(
    client, wired
):
    """Both have no signals. Only one of them is a clean result."""
    wired(_mixed_results())

    layers = {l["layer"]: l for l in upload(client).json()["layers"]}

    assert layers["L2"]["completed"] is True
    assert layers["L2"]["error"] is None
    assert layers["L3"]["completed"] is False
    assert layers["L3"]["error"] is not None
    assert layers["L2"]["signals"] == layers["L3"]["signals"] == []


# --------------------------------------------------------------------------
# 5. Contributions and message metadata
# --------------------------------------------------------------------------


def test_contributions_are_keyed_by_layer_name_and_omit_incomplete_layers(
    client, wired
):
    wired(_mixed_results())

    body = upload(client).json()
    contributions = body["contributions"]

    assert sorted(contributions) == ["L1", "L2", "L4"]
    assert "L3" not in contributions
    assert sum(contributions.values()) == pytest.approx(body["assessment"]["score"])


def test_message_metadata_comes_from_the_parsed_email(client, wired):
    wired(_mixed_results())

    message = upload(client).json()["message"]

    assert message == {
        "message_id": "<api-test-0001@account-verify-ms.example>",
        "from_addr": "security@account-verify-ms.example",
        "from_display": "Microsoft Account Team",
        "subject": "Action required",
        "url_count": 1,
        "attachment_count": 0,
    }


# --------------------------------------------------------------------------
# 6. Errors
# --------------------------------------------------------------------------


def test_an_unparseable_upload_is_a_400(client, wired, monkeypatch):
    """Both stdlib parsers accept nearly any bytes, so the seam is patched.

    What is under test is the mapping, not the parser: `EmailParseError` must
    reach the client as a bad request rather than as a generic 500.
    """
    wired(_mixed_results())

    def boom(raw):
        raise EmailParseError("input could not be parsed as an email")

    monkeypatch.setattr("api.app.parse_email", boom)

    response = upload(client)

    assert response.status_code == 400
    assert "could not be parsed" in response.json()["detail"]


def test_an_empty_upload_is_a_400(client, wired):
    wired(_mixed_results())

    response = upload(client, raw=b"")

    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_a_pipeline_where_nothing_completed_is_a_503(client, wired):
    """`UnscoreableError` is an outage, not a verdict, and never a clean 0.0."""
    wired(_nothing_completed())

    response = upload(client)

    assert response.status_code == 503
    assert "no layer completed" in response.json()["detail"]


def test_a_missing_file_field_is_a_422(client, wired):
    """FastAPI's own validation. Recorded so a change to it is visible."""
    wired(_mixed_results())

    assert client.post("/analyze/upload").status_code == 422


# --------------------------------------------------------------------------
# 7. CORS
# --------------------------------------------------------------------------


def test_the_vite_dev_origin_is_allowed(client):
    response = client.options(
        "/analyze/upload",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_the_allowed_origin_is_on_the_actual_response_too(client, wired):
    wired(_mixed_results())

    response = client.post(
        "/analyze/upload",
        files={"file": ("message.eml", EML, "message/rfc822")},
        headers={"Origin": "http://localhost:5173"},
    )

    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_an_unlisted_origin_is_not_granted_access(client):
    """Not a wildcard: an arbitrary site must not be able to post here."""
    response = client.options(
        "/analyze/upload",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.headers.get("access-control-allow-origin") != "https://evil.example"


def test_the_allowlist_holds_only_the_vite_dev_origin(client):
    assert ALLOWED_ORIGINS == ["http://localhost:5173"]
