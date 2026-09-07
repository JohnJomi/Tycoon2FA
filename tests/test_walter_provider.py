"""Unit tests for the Walter Writes provider.

Every response is served from an `httpx.MockTransport`. **No test reaches the
real API** - it is metered per word and throttled at roughly five requests a
minute, so a suite that called it would be both expensive and flaky.

The fixtures mirror the response shapes recorded in
`reports/walter_writes_calibration.md`, since the vendor's documentation was
unreachable and those observations are the only contract there is.
"""

from __future__ import annotations

import httpx
import pytest

from layers.ai_text import AITextDetector, AITextTimeout, AITextUnavailable, AITextVerdict
from layers.providers.walter import API_KEY_ENV, ENDPOINT, MIN_WORDS, WalterWritesDetector

LONG_TEXT = "word " * 60  # comfortably over the 50-word minimum

# A real 200 body, reduced to the fields the provider reads.
SUCCESS_BODY = {
    "ai_score": 0.2534,
    "result": "human",
    "word_count": 52,
    "service_name": "main_detector - 1",
    "items": [{"ai_score": 0.0534, "prediction": "original", "text": "..."}],
    "credits_remaining": 437,
}


def detector_for(handler, **kwargs) -> WalterWritesDetector:
    """A detector whose transport is a mock, never a socket."""
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return WalterWritesDetector(kwargs.pop("api_key", "test-key"), client=client, **kwargs)


def responding(response: httpx.Response):
    return lambda request: response


def never_called(request: httpx.Request) -> httpx.Response:  # pragma: no cover
    raise AssertionError("no HTTP request should have been made")


# --------------------------------------------------------------------------
# Success
# --------------------------------------------------------------------------


def test_the_provider_satisfies_the_detector_protocol():
    assert isinstance(WalterWritesDetector("k"), AITextDetector)


def test_a_successful_response_becomes_a_normalized_verdict():
    detector = detector_for(responding(httpx.Response(200, json=SUCCESS_BODY)))

    verdict = detector.detect(LONG_TEXT)

    assert isinstance(verdict, AITextVerdict)
    assert verdict.ai_generated_probability == pytest.approx(0.2534)
    assert verdict.provider == "walter_writes"
    assert verdict.model == "main_detector - 1"
    assert verdict.word_count == 52  # the provider's own count, which it bills on


def test_the_score_is_taken_whole_and_not_recomputed_from_sentences():
    """Calibration: the whole-text score is not the mean of `items`."""
    body = dict(SUCCESS_BODY, ai_score=0.1866,
                items=[{"ai_score": 0.9960, "prediction": "ai-generated", "text": "..."}])
    detector = detector_for(responding(httpx.Response(200, json=body)))

    verdict = detector.detect(LONG_TEXT)

    assert verdict.ai_generated_probability == pytest.approx(0.1866)


def test_the_request_uses_the_documented_endpoint_method_auth_and_payload():
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("X-API-Key")
        seen["content_type"] = request.headers.get("Content-Type")
        seen["json"] = request.read().decode()
        return httpx.Response(200, json=SUCCESS_BODY)

    detector_for(handler, api_key="secret-key").detect(LONG_TEXT)

    assert seen["method"] == "POST"
    assert seen["url"] == ENDPOINT
    assert seen["key"] == "secret-key"
    assert "application/json" in str(seen["content_type"])
    assert '"content"' in str(seen["json"])


# --------------------------------------------------------------------------
# Abstentions that cost no request
# --------------------------------------------------------------------------


def test_text_under_the_word_minimum_is_refused_without_an_http_request():
    detector = detector_for(never_called)

    with pytest.raises(AITextUnavailable) as raised:
        detector.detect("only a handful of words here")

    assert str(MIN_WORDS) in str(raised.value)


@pytest.mark.parametrize("text", ["", "   ", None])
def test_empty_text_is_refused_without_an_http_request(text):
    with pytest.raises(AITextUnavailable):
        detector_for(never_called).detect(text)


@pytest.mark.parametrize("key", [None, "", "   "])
def test_a_missing_api_key_abstains_without_an_http_request(key):
    with pytest.raises(AITextUnavailable) as raised:
        detector_for(never_called, api_key=key).detect(LONG_TEXT)

    assert API_KEY_ENV in str(raised.value)


def test_the_api_key_is_read_from_the_environment_when_not_passed(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "from-environment")

    assert WalterWritesDetector()._api_key == "from-environment"


def test_a_short_body_is_refused_before_the_key_is_even_checked():
    """A message too short to score must never spend a request or a credit."""
    with pytest.raises(AITextUnavailable):
        detector_for(never_called, api_key=None).detect("three words only")


# --------------------------------------------------------------------------
# HTTP failures
# --------------------------------------------------------------------------


def test_a_rejected_key_is_reported_as_a_configuration_fault():
    detector = detector_for(
        responding(httpx.Response(401, json={"error": "Invalid API key"})),
    )

    with pytest.raises(AITextUnavailable) as raised:
        detector.detect(LONG_TEXT)

    assert "401" in str(raised.value) and API_KEY_ENV in str(raised.value)


def test_throttling_is_surfaced_with_the_wait_and_is_not_retried():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(
            429, json={"error": "Request was throttled. Expected available in 57 seconds."}
        )

    with pytest.raises(AITextUnavailable) as raised:
        detector_for(handler).detect(LONG_TEXT)

    assert "429" in str(raised.value)
    assert "57" in str(raised.value)  # sent only in the body, not a header
    assert len(calls) == 1, "the provider must not retry a throttled request"


@pytest.mark.parametrize("status", [400, 403, 404, 500, 503])
def test_any_other_non_2xx_status_abstains(status):
    detector = detector_for(
        responding(httpx.Response(status, json={"error": "no", "code": "service_error"}))
    )

    with pytest.raises(AITextUnavailable) as raised:
        detector.detect(LONG_TEXT)

    assert str(status) in str(raised.value)


def test_a_timeout_raises_the_timeout_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(AITextTimeout):
        detector_for(handler).detect(LONG_TEXT)


def test_a_network_failure_is_unavailable_not_a_verdict():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(AITextUnavailable) as raised:
        detector_for(handler).detect(LONG_TEXT)

    assert "ConnectError" in str(raised.value)


# --------------------------------------------------------------------------
# Malformed successes - a 200 is not automatically an answer
# --------------------------------------------------------------------------


def test_a_non_json_success_body_abstains():
    detector = detector_for(responding(httpx.Response(200, text="<html>blocked</html>")))

    with pytest.raises(AITextUnavailable):
        detector.detect(LONG_TEXT)


@pytest.mark.parametrize(
    "body",
    [
        {},                                  # no ai_score at all
        {"ai_score": None},                  # present but null
        {"ai_score": "high"},                # non-numeric
        {"ai_score": 1.5},                   # outside the unit interval
        {"ai_score": -0.2},
        [1, 2, 3],                           # not an object
    ],
)
def test_a_malformed_success_body_abstains_rather_than_inventing_a_score(body):
    detector = detector_for(responding(httpx.Response(200, json=body)))

    with pytest.raises(AITextUnavailable):
        detector.detect(LONG_TEXT)


def test_a_missing_word_count_falls_back_to_the_local_count():
    body = {"ai_score": 0.5}
    detector = detector_for(responding(httpx.Response(200, json=body)))

    verdict = detector.detect(LONG_TEXT)

    assert verdict.word_count == len(LONG_TEXT.split())
    assert verdict.model is None


# --------------------------------------------------------------------------
# The key must not escape
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "handler",
    [
        responding(httpx.Response(401, json={"error": "bad key"})),
        responding(httpx.Response(429, json={"error": "throttled"})),
        responding(httpx.Response(500, json={"error": "boom", "code": "x"})),
        responding(httpx.Response(200, text="not json")),
    ],
)
def test_no_error_message_contains_the_api_key(handler):
    detector = detector_for(handler, api_key="wltr-super-secret-value")

    with pytest.raises(AITextUnavailable) as raised:
        detector.detect(LONG_TEXT)

    assert "wltr-super-secret-value" not in str(raised.value)
