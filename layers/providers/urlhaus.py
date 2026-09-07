"""URLhaus behind the `ThreatIntelSource` seam - a per-indicator query source.

    ThreatIntelSource -> URLhausSource -> POST /v1/url/  or  /v1/host/

abuse.ch's URLhaus is the one of the three free sources in ARCHITECTURE.md
section 4 that genuinely answers per-indicator queries, so this provider asks
the network once per indicator. It has two such endpoints - one for an exact
URL, one for a host - which is why it is the source that covers both
`IndicatorKind.URL` and `IndicatorKind.DOMAIN`. The endpoints, the form fields,
and the `query_status` response vocabulary are confined to this module.

The URL under investigation is sent to abuse.ch as a query parameter; it is
never fetched. Nothing here connects to the indicator's own host.
"""

from __future__ import annotations

import os

import httpx

from layers.threat_intel import IndicatorKind, IntelTimeout, IntelUnavailable, IntelVerdict

__all__ = ["URLhausSource"]

URL_ENDPOINT = "https://urlhaus-api.abuse.ch/v1/url/"
HOST_ENDPOINT = "https://urlhaus-api.abuse.ch/v1/host/"

# Which endpoint and form field each indicator kind uses. The host endpoint
# takes a hostname or a registrable domain in `host`; the URL endpoint takes an
# exact URL in `url`. Same response vocabulary either way.
_ENDPOINTS = {
    IndicatorKind.URL: (URL_ENDPOINT, "url"),
    IndicatorKind.DOMAIN: (HOST_ENDPOINT, "host"),
}

PROVIDER_NAME = "urlhaus"

# abuse.ch began requiring an Auth-Key for the API in 2024. Absent, the source
# is unavailable rather than negative: an unauthenticated 401 says nothing
# about the URL.
API_KEY_ENV = "URLHAUS_AUTH_KEY"

# Comfortably inside `core.orchestrator.DEFAULT_LAYER_TIMEOUTS[L4]`, which is
# 6s, so this fires first and the layer budget stays a backstop - the same
# arrangement Layer 1 uses for WHOIS and Layer 3 for Walter.
DEFAULT_TIMEOUT_SECONDS = 4.0

# `query_status` values that are answers. Anything else - including
# `http_post_expected`, `invalid_url` and any value not listed - is a refusal
# to answer, not a clean verdict.
_LISTED = "ok"
_NOT_LISTED = "no_results"


class URLhausSource:
    """`ThreatIntelSource` backed by the URLhaus lookup API.

    `client` exists so the unit suite can inject an `httpx.MockTransport`; when
    it is not supplied a client is created per call and closed again, matching
    `WalterWritesDetector`.
    """

    name = PROVIDER_NAME
    refresh_interval_seconds = None  # queried live, never snapshotted

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._api_key = api_key if api_key is not None else os.environ.get(API_KEY_ENV)
        self._client = client
        self._timeout = float(timeout)

    def supports(self, kind: IndicatorKind) -> bool:
        return kind in _ENDPOINTS

    def check(self, indicator: str, kind: IndicatorKind) -> IntelVerdict:
        if not self.supports(kind):
            raise IntelUnavailable(f"{self.name} does not cover {kind.value} indicators")
        if not self._api_key:
            raise IntelUnavailable(f"{API_KEY_ENV} is not set")

        endpoint, field = _ENDPOINTS[kind]
        headers = {"Auth-Key": self._api_key}
        payload_data = {field: indicator}
        try:
            if self._client is not None:
                response = self._client.post(
                    endpoint, data=payload_data, headers=headers, timeout=self._timeout
                )
            else:
                with httpx.Client(timeout=self._timeout) as client:
                    response = client.post(endpoint, data=payload_data, headers=headers)
        except httpx.TimeoutException as exc:
            raise IntelTimeout(f"{self.name} timed out after {self._timeout:g}s") from exc
        except httpx.HTTPError as exc:
            raise IntelUnavailable(f"{self.name} request failed: {type(exc).__name__}") from exc

        if response.status_code != 200:
            # Including 4xx: a rejected query is a query that was not answered.
            raise IntelUnavailable(f"{self.name} returned HTTP {response.status_code}")

        try:
            payload = response.json()
        except ValueError as exc:
            raise IntelUnavailable(f"{self.name} returned a non-JSON body") from exc

        if not isinstance(payload, dict):
            raise IntelUnavailable(f"{self.name} returned {type(payload).__name__}, not an object")

        status = payload.get("query_status")
        if status == _NOT_LISTED:
            return IntelVerdict(found=False, source=self.name)
        if status != _LISTED:
            # An unrecognized status is exactly the case where claiming to know
            # something would be wrong.
            raise IntelUnavailable(f"{self.name} returned query_status={status!r}")

        # The URL endpoint reports the threat inline; the host endpoint returns
        # the URLs seen on that host instead, so the count is what is worth
        # saying about a domain.
        threat = payload.get("threat")
        if threat is None and kind is IndicatorKind.DOMAIN:
            urls = payload.get("urls")
            if isinstance(urls, list) and urls:
                threat = f"{len(urls)} malicious URL(s) recorded on this host"
        reference = payload.get("urlhaus_reference")
        return IntelVerdict(
            found=True,
            source=self.name,
            reference=reference if isinstance(reference, str) else None,
            detail=str(threat) if isinstance(threat, str) else None,
        )
