"""Unit tests for the l2.redirect_depth signal in layers/l2_urls.py.

The follower is always an injected fake, so nothing here walks a chain, opens a
socket or resolves a name - an autouse guard forbids all four socket entry
points for the whole file. The production egress infrastructure does not exist
yet, and this suite must stay green without it.
"""

from __future__ import annotations

import pytest

from core.models import (
    DetectionLayer,
    ExtractedURL,
    ParsedEmail,
    RiskLevel,
    URLSource,
)
from layers.l2_urls import (
    REDIRECT_CAPPED_SCORE,
    REDIRECT_DEPTH_SCORE,
    REDIRECT_FLAG_DEPTH,
    REDIRECT_HOP_CAP,
    RedirectHop,
    RedirectOutcome,
    RedirectTrace,
    analyze_redirect_depth,
)

A = "https://links.evil-phish.com/a"
B = "https://tracker.example.net/b"
FINAL = "https://evil-phish.com/login"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("the redirect_depth suite touched the network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket, "gethostbyname", forbidden)


def email(*urls: str) -> ParsedEmail:
    return ParsedEmail(
        message_id="<m@example.com>",
        from_addr="sender@example.com",
        subject="Invoice",
        body_text="See the portal.",
        urls=[ExtractedURL(url=u, source=URLSource.ANCHOR_HREF) for u in urls],
    )


def hops(count: int, start: str = A) -> tuple[RedirectHop, ...]:
    urls = [start] + [f"https://h{i}.example/x" for i in range(1, count)]
    return tuple(
        RedirectHop(url=urls[i], status_code=302, location=f"https://h{i + 1}.example/x",
                    target=f"https://h{i + 1}.example/x")
        for i in range(count)
    )


def settled(url: str, depth: int) -> RedirectTrace:
    return RedirectTrace(
        url=url,
        outcome=RedirectOutcome.SETTLED,
        hops=hops(depth, url),
        final_url=FINAL if depth else url,
    )


def capped(url: str) -> RedirectTrace:
    return RedirectTrace(url=url, outcome=RedirectOutcome.CAPPED, hops=hops(REDIRECT_HOP_CAP, url))


def failed(url: str, outcome: RedirectOutcome, reason: str, depth: int = 0) -> RedirectTrace:
    return RedirectTrace(url=url, outcome=outcome, hops=hops(depth, url), error=reason)


class FakeFollower:
    """Returns a scripted trace per URL. Records what it was asked to walk."""

    def __init__(self, traces: dict[str, RedirectTrace]) -> None:
        self.traces = traces
        self.calls: list[str] = []

    def follow(self, url: str) -> RedirectTrace:
        self.calls.append(url)
        return self.traces.get(url, settled(url, 0))


def by_url(metadata, url: str) -> dict:
    return next(t for t in metadata["traces"] if t["url"] == url)


# --- clean results --------------------------------------------------------


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_a_chain_at_or_under_the_threshold_is_clean(depth: int) -> None:
    signal = analyze_redirect_depth(
        email(A), follower=FakeFollower({A: settled(A, depth)})
    )

    assert signal.layer is DetectionLayer.L2
    assert signal.name == "redirect_depth"
    assert signal.score == 0.0
    assert signal.severity is RiskLevel.LOW
    assert signal.error is None            # a negative, not an abstention
    assert signal.metadata["fired"] is False
    assert signal.metadata["deepest_exact"] == depth
    assert by_url(signal.metadata, A)["exact"] is True


def test_the_threshold_matches_the_architecture() -> None:
    """Section 4: 'Flag > 2'."""
    assert REDIRECT_FLAG_DEPTH == 2
    clean = analyze_redirect_depth(email(A), follower=FakeFollower({A: settled(A, 2)}))
    flagged = analyze_redirect_depth(email(A), follower=FakeFollower({A: settled(A, 3)}))

    assert clean.score == 0.0
    assert flagged.score > 0.0


def test_a_message_with_no_urls_is_a_genuine_negative() -> None:
    follower = FakeFollower({})
    signal = analyze_redirect_depth(email(), follower=follower)

    assert signal.score == 0.0
    assert signal.error is None
    assert follower.calls == []
    assert "no analysable URLs" in signal.evidence


# --- findings -------------------------------------------------------------


def test_an_exact_depth_of_three_is_a_finding() -> None:
    signal = analyze_redirect_depth(email(A), follower=FakeFollower({A: settled(A, 3)}))

    assert signal.score == REDIRECT_DEPTH_SCORE
    assert signal.severity is RiskLevel.MEDIUM
    assert signal.error is None
    assert signal.metadata["fired"] is True
    assert len(signal.metadata["findings"]) == 1
    assert signal.metadata["findings"][0]["exact"] is True
    assert signal.metadata["deepest_exact"] == 3
    assert "redirects 3 times" in signal.evidence
    assert "at least" not in signal.evidence


def test_a_deeper_exact_chain_is_also_a_finding() -> None:
    signal = analyze_redirect_depth(email(A), follower=FakeFollower({A: settled(A, 6)}))

    assert signal.score == REDIRECT_DEPTH_SCORE
    assert signal.metadata["findings"][0]["depth"] == 6
    assert signal.metadata["findings"][0]["exact"] is True


def test_a_capped_chain_is_a_finding_that_never_claims_an_exact_depth() -> None:
    signal = analyze_redirect_depth(email(A), follower=FakeFollower({A: capped(A)}))

    assert signal.score == REDIRECT_CAPPED_SCORE
    assert signal.severity is RiskLevel.HIGH
    assert signal.error is None
    finding = signal.metadata["findings"][0]
    assert finding["depth"] == REDIRECT_HOP_CAP
    assert finding["exact"] is False          # a floor, not a measurement
    assert finding["final_url"] is None
    assert "at least 8" in signal.evidence
    assert "still redirecting" in signal.evidence


def test_a_capped_chain_is_not_counted_as_a_measured_depth() -> None:
    """`deepest_exact` reports only depths that were actually measured."""
    signal = analyze_redirect_depth(email(A), follower=FakeFollower({A: capped(A)}))

    assert signal.metadata["deepest_exact"] is None


def test_a_capped_chain_outscores_an_exact_deep_chain() -> None:
    deep = analyze_redirect_depth(email(A), follower=FakeFollower({A: settled(A, 5)}))
    at_cap = analyze_redirect_depth(email(A), follower=FakeFollower({A: capped(A)}))

    assert at_cap.score > deep.score


# --- abstentions ----------------------------------------------------------


@pytest.mark.parametrize(
    "outcome, reason",
    [
        (RedirectOutcome.TIMED_OUT, "hop 1 timed out after 5s"),
        (RedirectOutcome.UNREACHABLE, "hop 1 failed: ConnectError"),
        (RedirectOutcome.NOT_ATTEMPTED, "no sandboxed egress configured"),
    ],
)
def test_a_non_answer_is_an_abstention_not_a_clean_result(
    outcome: RedirectOutcome, reason: str
) -> None:
    signal = analyze_redirect_depth(
        email(A), follower=FakeFollower({A: failed(A, outcome, reason)})
    )

    assert signal.score == 0.0
    assert signal.metadata["fired"] is False
    assert signal.error is not None
    assert reason in signal.error
    assert "absence of information, not a clean result" in signal.evidence
    assert by_url(signal.metadata, A)["outcome"] == outcome.value


def test_no_follower_configured_abstains() -> None:
    """The state the pipeline ships in: there is no sandboxed egress yet."""
    signal = analyze_redirect_depth(email(A))

    assert signal.score == 0.0
    assert signal.error is not None
    assert "no redirect follower configured" in signal.error
    assert by_url(signal.metadata, A)["outcome"] == RedirectOutcome.NOT_ATTEMPTED.value


def test_a_partial_failed_chain_keeps_its_hops_and_does_not_become_clean() -> None:
    signal = analyze_redirect_depth(
        email(A),
        follower=FakeFollower({A: failed(A, RedirectOutcome.TIMED_OUT, "stalled on hop 3", 2)}),
    )

    assert signal.score == 0.0
    assert signal.error is not None
    trace = by_url(signal.metadata, A)
    assert trace["depth"] == 2               # the hops it managed are preserved
    assert trace["final_url"] is None
    assert trace["error"] == "stalled on hop 3"
    assert signal.metadata["deepest_exact"] is None
    assert signal.metadata["findings"] == []


def test_a_follower_that_raises_is_treated_as_unreachable() -> None:
    class Broken:
        def follow(self, url: str) -> RedirectTrace:
            raise RuntimeError("egress proxy died")

    signal = analyze_redirect_depth(email(A), follower=Broken())

    assert signal.score == 0.0
    assert signal.error is not None
    assert "egress proxy died" in by_url(signal.metadata, A)["error"]


def test_a_follower_returning_the_wrong_type_is_treated_as_unreachable() -> None:
    class Wrong:
        def follow(self, url: str):
            return 7

    signal = analyze_redirect_depth(email(A), follower=Wrong())

    assert signal.error is not None
    assert "not a RedirectTrace" in by_url(signal.metadata, A)["error"]


# --- multiple URLs --------------------------------------------------------


def test_one_unfollowable_url_does_not_suppress_a_finding_on_another() -> None:
    signal = analyze_redirect_depth(
        email(A, B),
        follower=FakeFollower(
            {
                A: failed(A, RedirectOutcome.UNREACHABLE, "refused"),
                B: settled(B, 4),
            }
        ),
    )

    assert signal.score == REDIRECT_DEPTH_SCORE
    assert signal.error is None                       # the finding stands
    assert [f["url"] for f in signal.metadata["findings"]] == [B]
    assert "refused" in signal.metadata["abstentions"][A]
    assert "could not be followed" in signal.evidence  # the gap is still reported


def test_a_clean_verdict_is_not_issued_over_urls_nobody_followed() -> None:
    signal = analyze_redirect_depth(
        email(A, B),
        follower=FakeFollower(
            {A: failed(A, RedirectOutcome.TIMED_OUT, "stalled"), B: settled(B, 1)}
        ),
    )

    assert signal.score == 0.0
    assert signal.error is not None
    assert "not fully checked" in signal.evidence
    assert signal.metadata["deepest_exact"] == 1      # what was measured is kept


def test_all_urls_clean_is_a_clean_result() -> None:
    signal = analyze_redirect_depth(
        email(A, B), follower=FakeFollower({A: settled(A, 0), B: settled(B, 2)})
    )

    assert signal.score == 0.0
    assert signal.error is None
    assert signal.metadata["abstentions"] == {}
    assert "longest chain is 2" in signal.evidence


def test_every_url_is_followed_independently() -> None:
    follower = FakeFollower({A: settled(A, 1), B: settled(B, 1)})

    analyze_redirect_depth(email(A, B), follower=follower)

    assert follower.calls == [A, B]


def test_mixed_findings_and_caps_report_the_stronger_score() -> None:
    signal = analyze_redirect_depth(
        email(A, B), follower=FakeFollower({A: settled(A, 3), B: capped(B)})
    )

    assert signal.score == REDIRECT_CAPPED_SCORE
    assert len(signal.metadata["findings"]) == 2


# --- deduplication and raw evidence ---------------------------------------


def test_duplicate_canonical_urls_are_followed_once() -> None:
    follower = FakeFollower({A: settled(A, 3)})
    message = ParsedEmail(
        message_id="<m@example.com>",
        from_addr="sender@example.com",
        urls=[
            ExtractedURL(url=A, source=URLSource.ANCHOR_HREF),
            ExtractedURL(url="HTTPS://Links.Evil-Phish.com:443/a", source=URLSource.PLAIN_TEXT),
            ExtractedURL(url=A, source=URLSource.IMG_SRC),
        ],
    )

    signal = analyze_redirect_depth(message, follower=follower)

    assert follower.calls == [A]
    assert signal.metadata["urls_in_message"] == 3
    assert signal.metadata["urls_checked"] == 1


def test_the_raw_url_is_followed_and_quoted_not_the_canonical_form() -> None:
    messy = "HTTPS://Links.Evil-Phish.com/A?x=1"
    follower = FakeFollower({messy: settled(messy, 3)})
    message = ParsedEmail(
        message_id="<m@example.com>",
        from_addr="sender@example.com",
        urls=[ExtractedURL(url=messy, source=URLSource.ANCHOR_HREF)],
    )

    signal = analyze_redirect_depth(message, follower=follower)

    assert follower.calls == [messy]
    assert messy in signal.evidence
    assert "https://links.evil-phish.com/A?x=1" not in signal.evidence
    assert signal.metadata["findings"][0]["url"] == messy


def test_a_qr_decoded_url_re_enters_the_analysis() -> None:
    decoded = [ExtractedURL(url=B, source=URLSource.QR_CODE)]
    follower = FakeFollower({B: settled(B, 4)})

    signal = analyze_redirect_depth(email(), follower=follower, extra=decoded)

    assert follower.calls == [B]
    assert signal.score == REDIRECT_DEPTH_SCORE


# --- determinism and isolation --------------------------------------------


def test_repeated_analysis_is_deterministic() -> None:
    traces = {A: settled(A, 3), B: capped(B)}
    message = email(A, B)

    results = {
        (
            analyze_redirect_depth(message, follower=FakeFollower(traces)).score,
            analyze_redirect_depth(message, follower=FakeFollower(traces)).evidence,
        )
        for _ in range(5)
    }

    assert len(results) == 1


def test_the_signal_never_follows_a_redirect_itself() -> None:
    """Every walk goes through the seam; the function has no client of its own."""
    follower = FakeFollower({A: settled(A, 2)})

    analyze_redirect_depth(email(A), follower=follower)

    assert follower.calls == [A]


def test_the_parsed_message_is_not_mutated() -> None:
    message = email(A)

    analyze_redirect_depth(message, follower=FakeFollower({A: settled(A, 3)}))

    assert message.urls[0].redirect_chain == []
    assert message.urls[0].final_url is None
