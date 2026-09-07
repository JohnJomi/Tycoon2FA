"""Unit tests for HttpRedirectFollower in layers/providers/redirect_follower.py.

Every hop is served by an `httpx.MockTransport`. A module-scoped autouse guard
forbids `socket.connect`, `socket.connect_ex`, `getaddrinfo` and
`gethostbyname` for the whole file, so a test that accidentally reached the
real network would fail rather than silently succeed.
"""

from __future__ import annotations

import httpx
import pytest

from layers.l2_urls import REDIRECT_HOP_CAP, RedirectFollower, RedirectOutcome
from layers.providers.redirect_follower import (
    DEFAULT_PER_HOP_TIMEOUT,
    DEFAULT_TOTAL_BUDGET,
    REDIRECT_PROXY_ENV,
    REDIRECT_STATUS_CODES,
    HttpRedirectFollower,
)

START = "https://links.evil-phish.com/a"
FINAL = "https://evil-phish.com/login"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Nothing in this file may open a socket or resolve a name."""
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("the redirect follower test suite touched the network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket, "gethostbyname", forbidden)


def follower(handler, **kwargs) -> HttpRedirectFollower:
    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    return HttpRedirectFollower(client=client, **kwargs)


def chain_handler(mapping: dict[str, tuple[int, str | None]], terminal: int = 200):
    """Serve a redirect map: url -> (status, Location). Anything else settles."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url in mapping:
            status, location = mapping[url]
            headers = {} if location is None else {"location": location}
            return httpx.Response(status, headers=headers)
        return httpx.Response(terminal, text="ok")

    return handler


# --- configuration --------------------------------------------------------


def test_it_satisfies_the_seam() -> None:
    assert isinstance(HttpRedirectFollower(), RedirectFollower)


def test_the_budgets_sit_inside_the_layer_timeout() -> None:
    """Section 4's per-hop 5s, with a total strictly under L2's own 15s."""
    from core.orchestrator import DEFAULT_LAYER_TIMEOUTS
    from core.models import DetectionLayer

    assert DEFAULT_PER_HOP_TIMEOUT == 5.0
    assert DEFAULT_TOTAL_BUDGET < DEFAULT_LAYER_TIMEOUTS[DetectionLayer.L2]


def test_no_proxy_is_hard_coded(monkeypatch) -> None:
    monkeypatch.delenv(REDIRECT_PROXY_ENV, raising=False)

    assert HttpRedirectFollower()._proxy is None
    assert HttpRedirectFollower(proxy="http://egress.internal:3128")._proxy == (
        "http://egress.internal:3128"
    )

    monkeypatch.setenv(REDIRECT_PROXY_ENV, "http://from-env:3128")
    assert HttpRedirectFollower()._proxy == "http://from-env:3128"


def test_the_hop_cap_cannot_be_raised_above_the_architecture_cap() -> None:
    with pytest.raises(ValueError, match="cap"):
        HttpRedirectFollower(hop_cap=REDIRECT_HOP_CAP + 1)


# --- 1. no redirect -------------------------------------------------------


def test_a_url_that_does_not_redirect_settles_at_depth_zero() -> None:
    trace = follower(chain_handler({})).follow(START)

    assert trace.outcome is RedirectOutcome.SETTLED
    assert trace.depth == 0
    assert trace.hops == ()
    assert trace.final_url == START
    assert trace.url == START
    assert trace.error is None
    assert trace.is_answer is True


def test_the_original_url_is_preserved_exactly() -> None:
    messy = "HTTPS://Links.Evil-Phish.com:443/a?x=1"
    trace = follower(chain_handler({})).follow(messy)

    assert trace.url == messy  # not canonicalized


# --- 2. relative redirect -------------------------------------------------


def test_a_relative_location_is_resolved_against_the_hop_that_sent_it() -> None:
    trace = follower(
        chain_handler({START: (302, "/next/page")})
    ).follow(START)

    assert trace.outcome is RedirectOutcome.SETTLED
    assert trace.depth == 1
    assert trace.hops[0].target == "https://links.evil-phish.com/next/page"
    assert trace.final_url == "https://links.evil-phish.com/next/page"


def test_a_scheme_relative_location_inherits_the_scheme() -> None:
    trace = follower(
        chain_handler({START: (302, "//evil-phish.com/login")})
    ).follow(START)

    assert trace.hops[0].target == "https://evil-phish.com/login"


# --- 3. multiple redirects ------------------------------------------------


def test_a_multi_hop_chain_records_every_hop_in_order() -> None:
    trace = follower(
        chain_handler(
            {
                START: (302, "https://mid1.example/b"),
                "https://mid1.example/b": (301, "https://mid2.example/c"),
                "https://mid2.example/c": (307, FINAL),
            }
        )
    ).follow(START)

    assert trace.outcome is RedirectOutcome.SETTLED
    assert trace.depth == 3
    assert [h.status_code for h in trace.hops] == [302, 301, 307]
    assert [h.url for h in trace.hops] == [START, "https://mid1.example/b", "https://mid2.example/c"]
    assert trace.final_url == FINAL
    assert trace.chain == (START, "https://mid1.example/b", "https://mid2.example/c", FINAL)


@pytest.mark.parametrize("status", sorted(REDIRECT_STATUS_CODES))
def test_every_redirect_status_is_followed(status: int) -> None:
    trace = follower(chain_handler({START: (status, FINAL)})).follow(START)

    assert trace.depth == 1
    assert trace.final_url == FINAL


def test_a_300_multiple_choices_is_a_terminus_not_a_hop() -> None:
    trace = follower(chain_handler({START: (300, FINAL)})).follow(START)

    assert trace.outcome is RedirectOutcome.SETTLED
    assert trace.depth == 0


# --- 4. the cap -----------------------------------------------------------


def test_a_chain_still_redirecting_at_the_cap_is_capped_not_settled() -> None:
    endless = {f"https://h{i}.example/x": (302, f"https://h{i + 1}.example/x") for i in range(50)}
    trace = follower(chain_handler(endless)).follow("https://h0.example/x")

    assert trace.outcome is RedirectOutcome.CAPPED
    assert trace.depth == REDIRECT_HOP_CAP
    assert trace.depth_is_exact is False   # a floor, not a measurement
    assert trace.is_answer is True
    assert trace.final_url is None
    assert trace.error is None


def test_the_cap_is_never_exceeded() -> None:
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        n = len(requested)
        return httpx.Response(302, headers={"location": f"https://h{n}.example/x"})

    trace = follower(handler).follow("https://h0.example/x")

    assert len(requested) == REDIRECT_HOP_CAP
    assert trace.depth == REDIRECT_HOP_CAP


def test_a_chain_that_settles_exactly_at_the_cap_is_settled() -> None:
    mapping = {
        f"https://h{i}.example/x": (302, f"https://h{i + 1}.example/x")
        for i in range(REDIRECT_HOP_CAP - 1)
    }
    trace = follower(chain_handler(mapping)).follow("https://h0.example/x")

    assert trace.outcome is RedirectOutcome.SETTLED
    assert trace.depth == REDIRECT_HOP_CAP - 1


# --- 5. loops -------------------------------------------------------------


def test_a_two_url_loop_is_caught_rather_than_walked_to_the_cap() -> None:
    trace = follower(
        chain_handler({START: (302, FINAL), FINAL: (302, START)})
    ).follow(START)

    assert trace.outcome is RedirectOutcome.UNREACHABLE
    assert trace.is_answer is False
    assert "loop" in trace.error
    assert trace.depth == 2          # the hops that were completed are kept
    assert trace.final_url is None


def test_a_self_loop_is_caught_immediately() -> None:
    trace = follower(chain_handler({START: (302, START)})).follow(START)

    assert trace.outcome is RedirectOutcome.UNREACHABLE
    assert "loop" in trace.error
    assert trace.depth == 1


def test_a_loop_is_detected_across_equivalent_spellings() -> None:
    """`https://a.example/x` and `https://A.example:443/x` are one place."""
    trace = follower(
        chain_handler(
            {
                "https://a.example/x": (302, "https://b.example/y"),
                "https://b.example/y": (302, "https://A.example:443/x"),
            }
        )
    ).follow("https://a.example/x")

    assert trace.outcome is RedirectOutcome.UNREACHABLE
    assert "loop" in trace.error


# --- 6 & 7. bad Location headers ------------------------------------------


def test_a_redirect_with_no_location_header_is_unreachable() -> None:
    trace = follower(chain_handler({START: (302, None)})).follow(START)

    assert trace.outcome is RedirectOutcome.UNREACHABLE
    assert "no Location" in trace.error
    assert trace.depth == 1
    assert trace.hops[0].location is None
    assert trace.hops[0].target is None
    assert trace.final_url is None


def test_a_location_the_client_itself_rejects_is_unreachable() -> None:
    """Whatever httpx refuses to request is not a hop this module walks."""
    trace = follower(chain_handler({START: (302, "::junk::")})).follow(START)

    assert trace.outcome is RedirectOutcome.UNREACHABLE
    assert trace.is_answer is False
    assert trace.final_url is None


def test_an_empty_location_header_is_unreachable() -> None:
    trace = follower(chain_handler({START: (302, "   ")})).follow(START)

    assert trace.outcome is RedirectOutcome.UNREACHABLE
    assert trace.depth == 1


@pytest.mark.parametrize(
    "location",
    ["http://", "https://[unterminated/x", "http:///nohost"],
)
def test_a_malformed_location_is_unreachable_not_a_guess(location: str) -> None:
    trace = follower(chain_handler({START: (302, location)})).follow(START)

    assert trace.outcome is RedirectOutcome.UNREACHABLE
    assert trace.depth == 1
    assert trace.hops[0].location == location
    assert trace.hops[0].target is None


# --- 8. non-HTTP(S) targets -----------------------------------------------


@pytest.mark.parametrize(
    "location",
    [
        "javascript:alert(1)",
        "data:text/html;base64,PHNjcmlwdD4=",
        "mailto:victim@example.com",
        "ftp://files.example.com/x",
    ],
)
def test_a_non_http_target_is_not_followed(location: str) -> None:
    trace = follower(chain_handler({START: (302, location)})).follow(START)

    assert trace.outcome is RedirectOutcome.UNREACHABLE
    assert trace.depth == 1
    assert trace.hops[0].location == location   # recorded as evidence
    assert trace.hops[0].target is None         # but never walked
    assert "unusable Location" in trace.error


def test_a_url_that_is_not_analysable_is_rejected_without_a_request() -> None:
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(200)

    trace = follower(handler).follow("javascript:alert(1)")

    assert trace.outcome is RedirectOutcome.UNREACHABLE
    assert requested == []


# --- 9 & 10. transport failures -------------------------------------------


def test_a_connection_failure_is_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    trace = follower(handler).follow(START)

    assert trace.outcome is RedirectOutcome.UNREACHABLE
    assert trace.is_answer is False
    assert "ConnectError" in trace.error
    assert trace.final_url is None


def test_a_timeout_is_reported_as_a_timeout_not_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("too slow")

    trace = follower(handler).follow(START)

    assert trace.outcome is RedirectOutcome.TIMED_OUT
    assert trace.is_answer is False
    assert "timed out" in trace.error


def test_the_total_budget_stops_a_long_chain() -> None:
    """A chain that is slow rather than deep still terminates."""
    endless = {f"https://h{i}.example/x": (302, f"https://h{i + 1}.example/x") for i in range(50)}
    follow = follower(chain_handler(endless), total_budget=0.0)

    trace = follow.follow("https://h0.example/x")

    assert trace.outcome is RedirectOutcome.TIMED_OUT
    assert "budget" in trace.error


def test_an_unexpected_error_never_escapes_the_seam() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("something nobody predicted")

    trace = follower(handler).follow(START)

    assert trace.outcome is RedirectOutcome.UNREACHABLE
    assert "something nobody predicted" in trace.error


# --- 11. partial chains ---------------------------------------------------


def test_hops_completed_before_a_failure_are_preserved() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        calls.append(url)
        if len(calls) <= 2:
            return httpx.Response(302, headers={"location": f"https://mid{len(calls)}.example/x"})
        raise httpx.ConnectError("refused on the third hop")

    trace = follower(handler).follow(START)

    assert trace.outcome is RedirectOutcome.UNREACHABLE
    assert trace.depth == 2
    assert [h.url for h in trace.hops] == [START, "https://mid1.example/x"]
    assert trace.final_url is None


def test_hops_are_preserved_when_a_later_hop_times_out() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if len(calls) == 1:
            return httpx.Response(302, headers={"location": "https://mid.example/x"})
        raise httpx.ReadTimeout("stalled")

    trace = follower(handler).follow(START)

    assert trace.outcome is RedirectOutcome.TIMED_OUT
    assert trace.depth == 1
    assert trace.hops[0].url == START


# --- 12. raw vs resolved --------------------------------------------------


def test_the_raw_location_is_kept_separately_from_the_resolved_target() -> None:
    trace = follower(chain_handler({START: (302, "/next?x=1")})).follow(START)

    assert trace.hops[0].location == "/next?x=1"
    assert trace.hops[0].target == "https://links.evil-phish.com/next?x=1"


def test_an_absolute_location_is_kept_verbatim_as_well() -> None:
    trace = follower(chain_handler({START: (302, FINAL)})).follow(START)

    assert trace.hops[0].location == FINAL
    assert trace.hops[0].target == FINAL


# --- section 4 mechanics --------------------------------------------------


def test_the_client_never_follows_redirects_itself() -> None:
    """Section 4: allow_redirects=False. Every hop must be observed."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if str(request.url) == START:
            return httpx.Response(302, headers={"location": FINAL})
        return httpx.Response(200)

    trace = follower(handler).follow(START)

    assert [str(r.url) for r in seen] == [START, FINAL]
    assert trace.depth == 1  # the client did not collapse the hop


def test_no_response_body_is_downloaded() -> None:
    """Section 4: never execute downloads. Each hop is streamed and closed."""
    read = []

    class WatchfulStream(httpx.SyncByteStream):
        def __iter__(self):
            read.append(True)
            yield b"x" * 1024

        def close(self) -> None:
            pass

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=WatchfulStream())

    trace = follower(handler).follow(START)

    assert trace.outcome is RedirectOutcome.SETTLED
    assert read == []  # the body was never iterated


def test_every_execution_returns_a_valid_trace() -> None:
    """The contract is total: a trace for every input, an exception for none."""
    cases = [
        chain_handler({}),
        chain_handler({START: (302, None)}),
        chain_handler({START: (302, "javascript:alert(1)")}),
        chain_handler({START: (302, START)}),
    ]

    for handler in cases:
        trace = follower(handler).follow(START)
        assert trace.url == START
        assert isinstance(trace.outcome, RedirectOutcome)
        assert (trace.error is None) is trace.is_answer
