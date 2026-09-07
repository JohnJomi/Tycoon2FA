"""Unit tests for the Layer 2 redirect result contract in layers/l2_urls.py.

The record only - no follower exists, nothing is fetched, and no signal reads
these yet. These tests pin the invariants that keep an unfollowed or failed
chain from being read as a chain of depth 0.
"""

from __future__ import annotations

import pytest

from layers.l2_urls import (
    REDIRECT_FLAG_DEPTH,
    REDIRECT_HOP_CAP,
    RedirectFollower,
    RedirectHop,
    RedirectOutcome,
    RedirectTrace,
)

START = "https://links.evil-phish.com/a"
FINAL = "https://evil-phish.com/login"


def hop(url: str, status: int = 302, target: str | None = None) -> RedirectHop:
    return RedirectHop(url=url, status_code=status, location=target, target=target)


def hops(count: int) -> tuple[RedirectHop, ...]:
    return tuple(hop(f"https://h{i}.example/x", target=f"https://h{i + 1}.example/x")
                 for i in range(count))


# --- the architecture's numbers -------------------------------------------


def test_the_caps_match_the_architecture_specification() -> None:
    """Section 4: 'cap 8. Flag > 2'."""
    assert REDIRECT_HOP_CAP == 8
    assert REDIRECT_FLAG_DEPTH == 2


# --- a settled chain ------------------------------------------------------


def test_a_direct_url_has_a_real_depth_of_zero() -> None:
    trace = RedirectTrace(url=START, outcome=RedirectOutcome.SETTLED, final_url=START)

    assert trace.depth == 0
    assert trace.depth_is_exact is True
    assert trace.is_answer is True
    assert trace.error is None
    assert trace.chain == (START,)


def test_a_redirecting_chain_records_every_hop_in_order() -> None:
    chain = (
        hop(START, 302, "https://mid.example/b"),
        hop("https://mid.example/b", 301, FINAL),
    )
    trace = RedirectTrace(
        url=START, outcome=RedirectOutcome.SETTLED, hops=chain, final_url=FINAL
    )

    assert trace.depth == 2
    assert trace.chain == (START, "https://mid.example/b", FINAL)
    assert [h.status_code for h in trace.hops] == [302, 301]


def test_the_raw_location_header_is_kept_alongside_the_resolved_target() -> None:
    """A relative Location is normal; evidence should quote what was sent."""
    one = RedirectHop(url=START, status_code=302, location="/next", target="https://links.evil-phish.com/next")

    assert one.location == "/next"
    assert one.target == "https://links.evil-phish.com/next"


def test_a_hop_with_an_unresolvable_location_records_no_target() -> None:
    one = RedirectHop(url=START, status_code=302, location="::junk::", target=None)

    assert one.target is None


# --- the cap --------------------------------------------------------------


def test_a_capped_chain_is_an_answer_but_not_an_exact_depth() -> None:
    trace = RedirectTrace(
        url=START, outcome=RedirectOutcome.CAPPED, hops=hops(REDIRECT_HOP_CAP)
    )

    assert trace.depth == REDIRECT_HOP_CAP
    assert trace.depth_is_exact is False   # a floor, not a measurement
    assert trace.is_answer is True
    assert trace.final_url is None         # it never came to rest
    assert trace.error is None


def test_a_trace_may_not_exceed_the_hop_cap() -> None:
    with pytest.raises(ValueError, match="cap"):
        RedirectTrace(
            url=START, outcome=RedirectOutcome.SETTLED, hops=hops(REDIRECT_HOP_CAP + 1)
        )


def test_a_capped_trace_must_actually_have_reached_the_cap() -> None:
    with pytest.raises(ValueError, match="capped"):
        RedirectTrace(url=START, outcome=RedirectOutcome.CAPPED, hops=hops(3))


# --- abstention is not depth zero -----------------------------------------


@pytest.mark.parametrize(
    "outcome",
    [RedirectOutcome.UNREACHABLE, RedirectOutcome.TIMED_OUT, RedirectOutcome.NOT_ATTEMPTED],
)
def test_a_non_answer_outcome_is_not_an_answer(outcome: RedirectOutcome) -> None:
    trace = RedirectTrace(url=START, outcome=outcome, error="something went wrong")

    assert trace.is_answer is False
    assert outcome.is_answer is False
    assert trace.final_url is None
    assert trace.error


@pytest.mark.parametrize(
    "outcome",
    [RedirectOutcome.UNREACHABLE, RedirectOutcome.TIMED_OUT, RedirectOutcome.NOT_ATTEMPTED],
)
def test_a_non_answer_must_state_why(outcome: RedirectOutcome) -> None:
    """The invariant that stops an outage from reading as a clean direct link."""
    with pytest.raises(ValueError, match="state why"):
        RedirectTrace(url=START, outcome=outcome)


@pytest.mark.parametrize(
    "outcome",
    [RedirectOutcome.UNREACHABLE, RedirectOutcome.TIMED_OUT, RedirectOutcome.NOT_ATTEMPTED],
)
def test_a_non_answer_may_not_claim_a_final_url(outcome: RedirectOutcome) -> None:
    with pytest.raises(ValueError, match="no final URL"):
        RedirectTrace(url=START, outcome=outcome, error="boom", final_url=FINAL)


def test_an_answered_trace_may_not_carry_an_error() -> None:
    with pytest.raises(ValueError, match="must not carry an error"):
        RedirectTrace(
            url=START, outcome=RedirectOutcome.SETTLED, final_url=FINAL, error="boom"
        )


def test_a_settled_trace_must_record_where_it_came_to_rest() -> None:
    with pytest.raises(ValueError, match="came to rest"):
        RedirectTrace(url=START, outcome=RedirectOutcome.SETTLED)


def test_a_failed_walk_keeps_the_hops_it_completed() -> None:
    """A partial chain is evidence; the contract does not discard it."""
    trace = RedirectTrace(
        url=START,
        outcome=RedirectOutcome.TIMED_OUT,
        hops=hops(3),
        error="per-hop timeout after 5s",
    )

    assert trace.depth == 3
    assert trace.is_answer is False


def test_not_attempted_is_the_state_the_pipeline_ships_in() -> None:
    trace = RedirectTrace.not_attempted(START, "no sandboxed egress configured")

    assert trace.outcome is RedirectOutcome.NOT_ATTEMPTED
    assert trace.is_answer is False
    assert trace.depth == 0            # but is_answer is False, so it is not a measurement
    assert "egress" in trace.error


# --- basic validation -----------------------------------------------------


def test_the_outcome_must_be_a_redirect_outcome() -> None:
    with pytest.raises(TypeError, match="RedirectOutcome"):
        RedirectTrace(url=START, outcome="settled", final_url=FINAL)  # type: ignore[arg-type]


@pytest.mark.parametrize("url", ["", "   "])
def test_a_trace_must_name_the_url_it_started_from(url: str) -> None:
    with pytest.raises(ValueError, match="url"):
        RedirectTrace(url=url, outcome=RedirectOutcome.SETTLED, final_url=FINAL)


def test_elapsed_time_may_not_be_negative() -> None:
    with pytest.raises(ValueError, match="elapsed_ms"):
        RedirectTrace(
            url=START, outcome=RedirectOutcome.SETTLED, final_url=FINAL, elapsed_ms=-1
        )


def test_the_records_are_immutable() -> None:
    trace = RedirectTrace(url=START, outcome=RedirectOutcome.SETTLED, final_url=FINAL)

    with pytest.raises(Exception):
        trace.final_url = "https://elsewhere.example/"  # type: ignore[misc]
    with pytest.raises(Exception):
        trace.hops[0:0]  # tuples cannot be appended to
        trace.hops = ()  # type: ignore[misc]


def test_the_parsed_message_is_not_mutated_to_carry_a_trace() -> None:
    """The reason this record exists: ExtractedURL and URLCandidate stay clean."""
    from core.models import ExtractedURL, ParsedEmail, URLSource
    from layers.l2_urls import candidate_urls

    message = ParsedEmail(
        message_id="<m@example.com>",
        from_addr="s@example.com",
        urls=[ExtractedURL(url=START, source=URLSource.ANCHOR_HREF)],
    )
    (candidate,) = candidate_urls(message)

    RedirectTrace(
        url=candidate.raw,
        outcome=RedirectOutcome.SETTLED,
        hops=(hop(START, 302, FINAL),),
        final_url=FINAL,
    )

    assert message.urls[0].redirect_chain == []
    assert message.urls[0].final_url is None
    assert candidate.canonical == "https://links.evil-phish.com/a"


# --- the egress seam ------------------------------------------------------


def test_a_follower_is_anything_with_the_follow_method() -> None:
    class FakeFollower:
        def follow(self, url: str) -> RedirectTrace:
            return RedirectTrace(url=url, outcome=RedirectOutcome.SETTLED, final_url=url)

    follower = FakeFollower()

    assert isinstance(follower, RedirectFollower)
    assert follower.follow(START).is_answer is True


def test_no_follower_implementation_ships_in_this_repository() -> None:
    """Hop-following needs sandboxed egress, which does not exist yet."""
    import layers.l2_urls as module

    concrete = [
        name
        for name in dir(module)
        if isinstance(getattr(module, name), type)
        and name != "RedirectFollower"
        and hasattr(getattr(module, name), "follow")
    ]
    assert concrete == []


def test_the_contract_requires_no_network(monkeypatch) -> None:
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("the redirect contract attempted network access")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)

    trace = RedirectTrace(
        url=START, outcome=RedirectOutcome.SETTLED, hops=(hop(START, 302, FINAL),), final_url=FINAL
    )

    assert trace.chain == (START, FINAL)
