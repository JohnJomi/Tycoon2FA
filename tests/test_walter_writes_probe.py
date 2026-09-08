"""Tests for scripts/probe_walter_writes.py.

The probe is a throwaway connectivity check, but its response handling is
worth pinning: it is the only place so far that has met this API, and what it
learned about the contract is the input to the Layer 3 design.

Every test serves the response from an `httpx.MockTransport`. **No test makes
a real API call** - the repository has no integration-test convention, and a
suite that spends metered credits on every run would be a bad one to inherit.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import httpx
import pytest

_PROBE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "probe_walter_writes.py"
_spec = importlib.util.spec_from_file_location("probe_walter_writes", _PROBE_PATH)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


def client_returning(response: httpx.Response) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(lambda request: response))


# The shape the live API actually returned, reduced to the fields that matter.
SUCCESS_BODY = {
    "ai_score": 0.99,
    "result": "ai",
    "word_count": 75,
    "credits_remaining": 1766,
    "items": [{"ai_score": 0.99, "prediction": "ai-generated", "text": "..."}],
    "status": "success",
}


def test_a_success_response_is_parsed_into_status_and_body():
    with client_returning(httpx.Response(200, json=SUCCESS_BODY)) as client:
        status, body = probe.detect(client, "text")

    assert status == 200
    assert body["result"] == "ai"
    assert body["ai_score"] == pytest.approx(0.99)


def test_an_error_response_is_returned_rather_than_raised():
    """A 400 is information about the contract, not a crash."""
    error = {"error": "Invalid payload", "code": "invalid_payload"}
    with client_returning(httpx.Response(400, json=error)) as client:
        status, body = probe.detect(client, "text")

    assert status == 400
    assert body["code"] == "invalid_payload"


def test_a_non_json_body_is_reported_without_dumping_the_page():
    """A WAF challenge page is identifiable but must not flood the output."""
    with client_returning(httpx.Response(200, text="<html>" + "x" * 5000)) as client:
        status, body = probe.detect(client, "text")

    assert status == 200
    assert isinstance(body, str) and body.startswith("<non-JSON body:")
    assert len(body) < 300


def test_the_request_carries_the_content_field_the_api_requires():
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read().decode()
        return httpx.Response(200, json=SUCCESS_BODY)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        probe.detect(client, "the sample text")

    assert '"content"' in str(seen["body"])
    assert "the sample text" in str(seen["body"])


def test_a_timeout_is_reported_and_does_not_propagate(capsys):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        ok = probe.report("sample", "word " * probe.MIN_WORDS, client)

    assert ok is False
    assert "FAILED" in capsys.readouterr().out


def test_a_sample_under_the_word_minimum_is_skipped_without_a_call(capsys):
    """The API rejects under 50 words; the probe must not spend a credit to learn that."""
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no request should be made for a too-short sample")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        ok = probe.report("short", "too short", client)

    assert ok is False
    assert "SKIPPED" in capsys.readouterr().out


def test_a_missing_api_key_stops_before_any_request(monkeypatch, capsys):
    monkeypatch.setattr(probe, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv(probe.API_KEY_ENV, raising=False)

    assert probe.main() == 1
    assert probe.API_KEY_ENV in capsys.readouterr().err


def test_the_bundled_samples_meet_the_documented_word_minimum():
    for label, text in probe.SAMPLES.items():
        assert len(text.split()) >= probe.MIN_WORDS, label
