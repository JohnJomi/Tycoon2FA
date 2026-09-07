"""`RedirectFollower` over HTTP: the concrete hop-walker for l2.redirect_depth.

    RedirectFollower -> HttpRedirectFollower -> one GET per hop, no auto-follow

The only module in the project that performs Layer 2's network work. The client,
the per-hop and total budgets, and the egress configuration are confined here;
`layers.l2_urls` sees a `RedirectTrace` and nothing else.

What this does, per ARCHITECTURE.md section 4
---------------------------------------------
"Follow hops, `allow_redirects=False`, cap 8", per-hop timeout 5s inside a
total budget. Every hop is requested individually with redirects disabled, so
each `Location` is observed rather than collapsed by the client - the whole
point of the signal is the *shape* of the chain, and a client that followed
redirects for us would return the destination and throw the evidence away.

What this deliberately does not do
----------------------------------
- **No rendering and no JavaScript.** This is an HTTP client, not a browser. A
  meta-refresh or a `location.href` hop is invisible here, and that is a stated
  limitation of the signal rather than something to fix with Playwright, which
  section 4 assigns to `l2.captcha_gate` under its own sandbox rules.
- **No body is downloaded.** Each hop is streamed and closed as soon as the
  status line and headers are in hand, so a redirect chain that ends at a
  10MB payload costs nothing to observe. Section 4: never execute downloads.
- **No DNS or socket calls of its own.** Name resolution and connection are
  `httpx`'s, through whatever transport it was given; nothing here touches
  `socket` directly, which is what lets the tests forbid it outright.
- **No caching.** Redirect chains rotate fast and section 4 sets no TTL for
  them, so caching needs its own decision rather than an inherited one.

Egress
------
Section 4 is emphatic: attacker infrastructure must never be fetched from a
home or campus IP, and hop-following belongs behind restricted-network Docker
with egress via a VPS or VPN. **This module hard-codes no proxy and carries no
credentials.** It takes an injected `client`, or a `proxy` argument, or reads
one from `REDIRECT_PROXY_ENV` - and if none is configured it still works
directly, which is correct for a developer running against a lab URL and wrong
for production. Deployment owns that choice; the code only makes it expressible.
"""

from __future__ import annotations

import os
import time
from urllib.parse import urljoin

import httpx

from layers.l2_urls import (
    REDIRECT_HOP_CAP,
    RedirectHop,
    RedirectOutcome,
    RedirectTrace,
    canonical_form,
    parse_url,
)

__all__ = [
    "DEFAULT_PER_HOP_TIMEOUT",
    "DEFAULT_TOTAL_BUDGET",
    "REDIRECT_PROXY_ENV",
    "REDIRECT_STATUS_CODES",
    "HttpRedirectFollower",
]

# Section 4: "Per-hop timeout 5s".
DEFAULT_PER_HOP_TIMEOUT = 5.0

# Section 4 says "total budget 15s" - but `DEFAULT_LAYER_TIMEOUTS[L2]` is also
# 15s, and Layer 2 has five signals to run inside it. A follower allowed the
# layer's entire budget would leave the other four nothing, and the layer
# timeout would start firing as the operative limit instead of as the
# orchestrator's backstop against a broken layer.
#
# 12s keeps the spec's intent - eight 5s hops must not run to 40s - while
# staying strictly inside the layer, which is the arrangement every other
# network seam here already uses: WHOIS 5s under L1's 8s, URLhaus 4s under
# L4's 6s. Both budgets are constructor arguments, so a deployment that wants
# the literal 15s can ask for it.
DEFAULT_TOTAL_BUDGET = 12.0

# Read only when neither `client` nor `proxy` is supplied. Named, not
# defaulted: there is no sensible default proxy and inventing one would be a
# deployment decision made in the wrong place.
REDIRECT_PROXY_ENV = "REDIRECT_EGRESS_PROXY"

# The redirect statuses that carry a Location. 300 (Multiple Choices) is
# excluded: it is not a redirect a client follows without a choice being made.
REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})

# httpx builds the redirect request eagerly - `_send_handling_redirects` calls
# `_build_redirect_request` whenever a response has a redirect location, even
# with `follow_redirects=False`, purely to populate `response.next_request`. A
# `Location: javascript:...` or `Location: ::junk::` therefore raises
# `httpx.InvalidURL` out of the send, destroying exactly the evidence this
# signal exists to record.
#
# The response event hook below runs *before* that build (see the ordering in
# `httpx._client`), and defuses it: the raw header is copied to a private
# header this module reads, and the real `Location` is removed when it is not
# something we would follow anyway. httpx then sees a non-redirect response and
# returns it intact, hop and all.
_RAW_LOCATION_HEADER = "x-tycoon-raw-location"
_HOOK_INSTALLED = "_tycoon_redirect_hook"


def _defuse_unfollowable_location(response: httpx.Response) -> None:
    """Preserve a Location httpx would refuse to parse. Never raises."""
    try:
        if response.status_code not in REDIRECT_STATUS_CODES:
            return
        location = response.headers.get("location")
        if location is None:
            return
        response.headers[_RAW_LOCATION_HEADER] = location
        if _resolve(str(response.request.url), location) is None:
            # Not a hop this module would follow, so httpx has no reason to
            # parse it - and every reason not to.
            del response.headers["location"]
    except Exception:  # noqa: BLE001 - a hook must never break the request
        return


def _install_hook(client: httpx.Client) -> httpx.Client:
    """Add the hook once, idempotently, to whichever client is in use."""
    if not getattr(client, _HOOK_INSTALLED, False):
        client.event_hooks["response"] = [
            *client.event_hooks.get("response", []),
            _defuse_unfollowable_location,
        ]
        setattr(client, _HOOK_INSTALLED, True)
    return client


# A browser-shaped identity. Some phishing infrastructure serves a different
# chain - or none - to an obvious tool, and the point is to observe the chain a
# victim would get. Not configurable per call; this is not a fingerprinting
# surface worth widening.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


class HttpRedirectFollower:
    """`RedirectFollower` backed by an httpx client with auto-redirects off.

    `client` exists so the unit suite can inject an `httpx.MockTransport`; when
    it is not supplied a client is created per call and closed again, matching
    `WalterWritesDetector` and `URLhausSource`.

    Total by contract: `follow` returns a `RedirectTrace` for every input and
    raises nothing. A refused connection, a timeout, a loop or a malformed
    `Location` is a trace whose outcome is not an answer, carrying whatever
    hops completed before the failure.
    """

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        proxy: str | None = None,
        per_hop_timeout: float = DEFAULT_PER_HOP_TIMEOUT,
        total_budget: float = DEFAULT_TOTAL_BUDGET,
        hop_cap: int = REDIRECT_HOP_CAP,
    ) -> None:
        if hop_cap < 1 or hop_cap > REDIRECT_HOP_CAP:
            raise ValueError(
                f"hop_cap must be between 1 and the {REDIRECT_HOP_CAP}-hop "
                f"architecture cap, got {hop_cap!r}"
            )
        self._client = client
        self._proxy = proxy if proxy is not None else os.environ.get(REDIRECT_PROXY_ENV)
        self._per_hop_timeout = float(per_hop_timeout)
        self._total_budget = float(total_budget)
        self._hop_cap = int(hop_cap)

    # -- the seam ----------------------------------------------------------

    def follow(self, url: str) -> RedirectTrace:
        """Walk one URL's redirect chain. Never raises."""
        started = time.monotonic()

        if parse_url(url) is None:
            return RedirectTrace(
                url=url if url and url.strip() else "about:invalid",
                outcome=RedirectOutcome.UNREACHABLE,
                error="not an analysable http(s) URL",
            )

        try:
            if self._client is not None:
                return self._walk(url, _install_hook(self._client), started)
            with self._new_client() as client:
                return self._walk(url, client, started)
        except Exception as exc:  # noqa: BLE001 - the contract forbids leaking
            # Reaching here means a defect in the walk itself rather than a
            # network failure, which `_walk` already converts. Still a trace:
            # the caller must never have to catch anything from this seam.
            return RedirectTrace(
                url=url,
                outcome=RedirectOutcome.UNREACHABLE,
                error=f"redirect follower failed: {type(exc).__name__}: {exc}",
                elapsed_ms=_ms_since(started),
            )

    # -- internals ---------------------------------------------------------

    def _new_client(self) -> httpx.Client:
        kwargs: dict[str, object] = {
            "timeout": self._per_hop_timeout,
            "follow_redirects": False,  # section 4: allow_redirects=False
            "headers": {"User-Agent": _USER_AGENT},
        }
        if self._proxy:
            kwargs["proxy"] = self._proxy
        return _install_hook(httpx.Client(**kwargs))  # type: ignore[arg-type]

    def _walk(self, url: str, client: httpx.Client, started: float) -> RedirectTrace:
        hops: list[RedirectHop] = []
        current = url
        # Canonical forms, so `https://a.example/x` and `https://A.example:443/x`
        # are recognized as the same place rather than looping forever.
        visited = {canonical_form(url) or url}

        while True:
            remaining = self._total_budget - (time.monotonic() - started)
            if remaining <= 0:
                return self._failed(
                    url, hops, RedirectOutcome.TIMED_OUT,
                    f"total budget of {self._total_budget:g}s exhausted after "
                    f"{len(hops)} hop(s)",
                    started,
                )

            hop_started = time.monotonic()
            try:
                status, location = self._request(client, current, min(remaining, self._per_hop_timeout))
            except httpx.TimeoutException:
                return self._failed(
                    url, hops, RedirectOutcome.TIMED_OUT,
                    f"hop {len(hops) + 1} timed out after "
                    f"{self._per_hop_timeout:g}s at {current}",
                    started,
                )
            except httpx.HTTPError as exc:
                return self._failed(
                    url, hops, RedirectOutcome.UNREACHABLE,
                    f"hop {len(hops) + 1} failed at {current}: {type(exc).__name__}",
                    started,
                )

            if status not in REDIRECT_STATUS_CODES:
                # The chain came to rest. This is the answer.
                return RedirectTrace(
                    url=url,
                    outcome=RedirectOutcome.SETTLED,
                    hops=tuple(hops),
                    final_url=current,
                    elapsed_ms=_ms_since(started),
                )

            target = _resolve(current, location)
            hops.append(
                RedirectHop(
                    url=current,
                    status_code=status,
                    location=location,
                    target=target,
                    elapsed_ms=_ms_since(hop_started),
                )
            )

            if location is None or not location.strip():
                return self._failed(
                    url, hops, RedirectOutcome.UNREACHABLE,
                    f"hop {len(hops)} returned {status} with no Location header",
                    started,
                )
            if target is None:
                return self._failed(
                    url, hops, RedirectOutcome.UNREACHABLE,
                    f"hop {len(hops)} returned {status} to an unusable Location "
                    f"{location!r}",
                    started,
                )

            key = canonical_form(target) or target
            if key in visited:
                return self._failed(
                    url, hops, RedirectOutcome.UNREACHABLE,
                    f"redirect loop: hop {len(hops)} returned to {target}",
                    started,
                )
            visited.add(key)

            if len(hops) >= self._hop_cap:
                # Still redirecting at the cap. A real answer, and a finding in
                # its own right - `depth_is_exact` records that it is a floor.
                return RedirectTrace(
                    url=url,
                    outcome=RedirectOutcome.CAPPED,
                    hops=tuple(hops),
                    elapsed_ms=_ms_since(started),
                )

            current = target

    def _request(
        self, client: httpx.Client, url: str, timeout: float
    ) -> tuple[int, str | None]:
        """One hop. Streams so no body is downloaded - section 4's rule.

        Returns the status and the raw `Location` header. The response is
        closed by the context manager before the body is touched.
        """
        with client.stream(
            "GET", url, follow_redirects=False, timeout=timeout
        ) as response:
            # The hook copies the header here before httpx can reject it; the
            # real `location` is still authoritative when it survived.
            raw = response.headers.get(_RAW_LOCATION_HEADER)
            location = response.headers.get("location", raw)
            return response.status_code, location

    def _failed(
        self,
        url: str,
        hops: list[RedirectHop],
        outcome: RedirectOutcome,
        reason: str,
        started: float,
    ) -> RedirectTrace:
        """A non-answer trace that keeps the hops already completed."""
        return RedirectTrace(
            url=url,
            outcome=outcome,
            hops=tuple(hops),
            error=reason,
            elapsed_ms=_ms_since(started),
        )


def _resolve(base: str, location: str | None) -> str | None:
    """Resolve a `Location` against the URL that returned it.

    Standard RFC 3986 joining via `urljoin`, so a relative `/next`, a
    scheme-relative `//host/x` and an absolute URL all work. Returns None -
    never a guess - when the header is absent, unjoinable, or resolves to
    something outside http(s): a `Location: javascript:...` or a `data:` URL is
    not a hop this layer follows, and treating it as one would mean handing an
    unanalysable string to the next request.
    """
    if location is None or not location.strip():
        return None
    try:
        joined = urljoin(base, location.strip())
    except ValueError:
        return None

    if parse_url(joined) is None:
        return None

    # `parse_url` reads structure; it does not judge whether a host is legal.
    # `https://exa mple.com/x` splits cleanly and is still not a URL anything
    # can request, so the client's own parser gets the final say - and gets it
    # here, where the answer is "do not follow this", rather than inside a send
    # where it would raise.
    try:
        httpx.URL(joined)
    except (httpx.InvalidURL, ValueError, UnicodeError):
        return None
    return joined


def _ms_since(started: float) -> int:
    return max(0, round((time.monotonic() - started) * 1000))
