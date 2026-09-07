"""Walter Writes behind the `AITextDetector` seam.

    AITextDetector -> WalterWritesDetector -> POST /api/detector/

The only module in the project that knows this vendor exists. The endpoint,
the `X-API-Key` header, the `content` request field and the `ai_score`
response field are all confined here; `layers.ai_text` and Layer 3 see an
`AITextVerdict` or an `AITextUnavailable` and nothing else.

Everything below was established empirically in
`reports/walter_writes_calibration.md`, because the vendor's published
documentation was unreachable (Cloudflare 403) throughout testing. That is
worth stating plainly: this is an observed contract, not a documented one, and
it can change without warning. The identity checks and the strict response
parsing are what keep an unannounced change loud rather than silent.

Deliberately absent, and not oversights:

- **No retries.** The API throttles at roughly five requests per minute and
  states the wait only in the 429 body. A retry policy needs to be designed
  against that budget, not improvised per call.
- **No caching.** Walter bills per word, so caching matters - and for that
  reason it deserves its own design pass against `storage.cache` rather than
  an ad-hoc dict here.
- **No scoring.** Per the calibration, this provider's probability is not fit
  to influence the composite risk score, and nothing here emits a signal.

Security: the API key is read from the environment, is never logged, and never
appears in an exception message. Response bodies are not echoed either - they
contain the submitted text, and an error string that quotes them would put
email content into logs.
"""

from __future__ import annotations

import os
import re

import httpx

from layers.ai_text import AITextTimeout, AITextUnavailable, AITextVerdict

__all__ = ["WalterWritesDetector"]

# Confirmed against the live service; see the calibration report. The probe at
# `scripts/probe_walter_writes.py` targets the same URL - that script is a
# throwaway and this constant, not it, is what production reads.
ENDPOINT = "https://developer-portal.walterwrites.ai/api/detector/"

API_KEY_ENV = "WALTER_WRITES_API_KEY"

PROVIDER_NAME = "walter_writes"

# The API rejects shorter text with a 400. Enforced here so a message that
# cannot be scored costs neither a round trip nor a credit.
MIN_WORDS = 50

# Comfortably inside `core.orchestrator.DEFAULT_LAYER_TIMEOUTS[L3]`, which is
# 10s, so this fires first and the layer's own budget stays a backstop rather
# than the operative limit - the arrangement Layer 1 uses for WHOIS.
DEFAULT_TIMEOUT_SECONDS = 8.0

_THROTTLE_WAIT = re.compile(r"available in (\d+) seconds")


class WalterWritesDetector:
    """`AITextDetector` backed by the Walter Writes detection API.

    `client` exists so the unit suite can inject an `httpx.MockTransport`; when
    it is not supplied a client is created per call and closed again, which is
    the right default for a detector used once per message.

    Passing no `api_key` reads it from the environment at construction. A
    missing key is not raised here: the object stays constructible so callers
    can build one unconditionally, and the failure surfaces as an ordinary
    abstention on the call that actually needed it.
    """

    provider = PROVIDER_NAME

    def __init__(
        self,
        api_key: str | None = None,
        *,
        endpoint: str = ENDPOINT,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else os.environ.get(API_KEY_ENV)
        self.endpoint = endpoint
        self.timeout = timeout
        self._client = client

    # ------------------------------------------------------------------ api

    def detect(self, text: str) -> AITextVerdict:
        """Score one piece of text, or raise `AITextUnavailable`.

        Never returns a fabricated probability. Every path that does not end in
        a parsed `ai_score` raises, so the caller abstains instead of recording
        a number the provider did not give.
        """
        words = len((text or "").split())
        if words < MIN_WORDS:
            # A property of the input, not a failure of the provider - but the
            # outcome is the same: there is no verdict, so there is nothing to
            # report. Checked before the key so a short body never spends a
            # request.
            raise AITextUnavailable(
                f"text is {words} words, under the provider's {MIN_WORDS}-word minimum"
            )

        if not (self._api_key or "").strip():
            raise AITextUnavailable(
                f"no API key configured; set {API_KEY_ENV}"
            )

        response = self._post(text)

        if response.status_code == 401:
            # A configuration fault, not an outage. Named as such so it is
            # actionable, without echoing the key or the response.
            raise AITextUnavailable(
                f"provider rejected the API key (HTTP 401); check {API_KEY_ENV}"
            )
        if response.status_code == 429:
            # Throttled. Deliberately not retried here - see the module
            # docstring. The stated wait is surfaced because the provider sends
            # it only in the body, with no Retry-After header.
            raise AITextUnavailable(
                f"provider throttled the request (HTTP 429){self._throttle_note(response)}"
            )
        if not 200 <= response.status_code < 300:
            raise AITextUnavailable(
                f"provider returned HTTP {response.status_code}"
                f"{self._error_code_note(response)}"
            )

        return self._verdict(response, words)

    # -------------------------------------------------------------- helpers

    def _post(self, text: str) -> httpx.Response:
        """One request, with transport failures translated at the boundary."""
        headers = {"X-API-Key": self._api_key, "Content-Type": "application/json"}
        payload = {"content": text}
        try:
            if self._client is not None:
                return self._client.post(self.endpoint, json=payload, headers=headers)
            with httpx.Client(timeout=self.timeout) as client:
                return client.post(self.endpoint, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise AITextTimeout(
                f"provider did not respond within {self.timeout:g}s"
            ) from exc
        except httpx.HTTPError as exc:
            # Type only. An httpx error's message carries the request URL, and
            # some carry more; none of it belongs in an error string that will
            # be shown as a signal's `error`.
            raise AITextUnavailable(
                f"could not reach the provider ({type(exc).__name__})"
            ) from exc

    def _verdict(self, response: httpx.Response, words: int) -> AITextVerdict:
        """Normalize a 200 into an `AITextVerdict`.

        `ai_score` is taken as the probability directly - it is already a
        0.0-1.0 quantity. It is deliberately **not** recomputed from the
        sentence-level `items`: the calibration found the whole-text score is
        not their mean (0.1866 overall against a 0.9960 sentence maximum in one
        case), so the aggregation rule is unknown and inventing one would put a
        number in the record that the provider never asserted.
        """
        try:
            body = response.json()
        except ValueError as exc:
            raise AITextUnavailable("provider returned a non-JSON response") from exc

        if not isinstance(body, dict):
            raise AITextUnavailable("provider returned an unexpected response shape")

        if "ai_score" not in body:
            raise AITextUnavailable("provider response carried no ai_score")

        try:
            probability = float(body["ai_score"])
        except (TypeError, ValueError) as exc:
            raise AITextUnavailable("provider returned a non-numeric ai_score") from exc

        model = body.get("service_name")
        try:
            return AITextVerdict(
                ai_generated_probability=probability,
                provider=PROVIDER_NAME,
                model=model if isinstance(model, str) else None,
                word_count=self._word_count(body, words),
            )
        except (TypeError, ValueError) as exc:
            # An out-of-range or non-finite score reaches here. It is a
            # malformed answer, not a low one.
            raise AITextUnavailable(f"provider returned an unusable ai_score ({exc})") from exc

    @staticmethod
    def _word_count(body: dict, fallback: int) -> int:
        """The provider's own word count, which is what it bills on."""
        value = body.get("word_count")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return fallback
        return value

    @staticmethod
    def _body_field(response: httpx.Response, field: str) -> str:
        """One named field from an error body, or "" - never the whole body."""
        try:
            body = response.json()
        except ValueError:
            return ""
        value = body.get(field) if isinstance(body, dict) else None
        return value if isinstance(value, str) else ""

    @classmethod
    def _throttle_note(cls, response: httpx.Response) -> str:
        match = _THROTTLE_WAIT.search(cls._body_field(response, "error"))
        return f"; provider suggests retrying in {match.group(1)}s" if match else ""

    @classmethod
    def _error_code_note(cls, response: httpx.Response) -> str:
        code = cls._body_field(response, "code")
        return f" ({code})" if code else ""
