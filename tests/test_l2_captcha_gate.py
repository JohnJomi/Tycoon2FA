"""Unit tests for the l2.captcha_gate signal in layers/l2_urls.py.

The renderer is always an injected fake: Playwright is not installed and no
concrete renderer exists. Redirect traces are built directly from the existing
contract - nothing here follows a redirect. An autouse guard forbids every
socket and DNS entry point for the whole file.
"""

from __future__ import annotations

import sys

import pytest

from core.models import (
    DetectionLayer,
    ExtractedURL,
    ParsedEmail,
    RiskLevel,
    URLSource,
)
from layers.l2_urls import (
    CAPTCHA_GATE_SCORE,
    ChallengeEvidence,
    ChallengeKind,
    RedirectHop,
    RedirectOutcome,
    RedirectTrace,
    RenderOutcome,
    RenderResult,
    analyze_captcha_gate,
    candidate_urls,
    terminal_urls,
)

LINK = "https://links.evil-phish.com/a"
TERMINAL = "https://evil-phish.com/login"
SECOND = "https://tracker.example.net/b"
SECOND_TERMINAL = "https://second-phish.com/signin"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("the captcha_gate suite touched the network")

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


def settled(url: str, final: str) -> RedirectTrace:
    hops = (
        ()
        if url == final
        else (RedirectHop(url=url, status_code=302, location=final, target=final),)
    )
    return RedirectTrace(
        url=url, outcome=RedirectOutcome.SETTLED, hops=hops, final_url=final
    )


def unfollowed(url: str, outcome: RedirectOutcome = RedirectOutcome.TIMED_OUT) -> RedirectTrace:
    return RedirectTrace(url=url, outcome=outcome, error="chain not followed")


def challenge(kind: ChallengeKind, marker: str) -> ChallengeEvidence:
    return ChallengeEvidence(kind=kind, marker=marker, detail="script[src]")


def rendered(url: str, *challenges: ChallengeEvidence) -> RenderResult:
    return RenderResult(
        url=url,
        outcome=RenderOutcome.RENDERED,
        challenges=challenges,
        final_url=url,
        status_code=200,
        title="Sign in",
    )


def failed(url: str, outcome: RenderOutcome, reason: str) -> RenderResult:
    return RenderResult(url=url, outcome=outcome, error=reason)


class FakeRenderer:
    """Returns scripted results per URL, and records what it was asked to render."""

    def __init__(self, scripted: dict[str, RenderResult] | None = None) -> None:
        self.scripted = scripted or {}
        self.calls: list[str] = []

    def render(self, url: str) -> RenderResult:
        self.calls.append(url)
        return self.scripted.get(url) or rendered(url)


# --- 1, 2, 3, 4. findings -------------------------------------------------


@pytest.mark.parametrize(
    "kind, marker",
    [
        (ChallengeKind.TURNSTILE, "challenges.cloudflare.com/turnstile/v0/api.js"),
        (ChallengeKind.HCAPTCHA, "hcaptcha.com/1/api.js"),
        (ChallengeKind.RECAPTCHA, "google.com/recaptcha/api.js"),
    ],
)
def test_each_supported_challenge_kind_is_a_finding(
    kind: ChallengeKind, marker: str
) -> None:
    renderer = FakeRenderer({TERMINAL: rendered(TERMINAL, challenge(kind, marker))})

    signal = analyze_captcha_gate(
        email(LINK), renderer=renderer, traces=[settled(LINK, TERMINAL)]
    )

    assert signal.layer is DetectionLayer.L2
    assert signal.name == "captcha_gate"
    assert signal.score == CAPTCHA_GATE_SCORE
    assert signal.severity is RiskLevel.HIGH
    assert signal.error is None
    assert signal.metadata["fired"] is True
    match = signal.metadata["matches"][0]
    assert match["kinds"] == [kind.value]
    assert match["markers"] == [marker]
    assert renderer.calls == [TERMINAL]


def test_several_challenge_kinds_on_one_page_are_all_reported() -> None:
    page = rendered(
        TERMINAL,
        challenge(ChallengeKind.TURNSTILE, "cf-turnstile"),
        challenge(ChallengeKind.RECAPTCHA, "g-recaptcha"),
    )
    signal = analyze_captcha_gate(
        email(LINK),
        renderer=FakeRenderer({TERMINAL: page}),
        traces=[settled(LINK, TERMINAL)],
    )

    match = signal.metadata["matches"][0]
    assert match["kinds"] == ["turnstile", "recaptcha"]
    assert match["markers"] == ["cf-turnstile", "g-recaptcha"]
    assert "turnstile, recaptcha" in signal.evidence


# --- 5. clean -------------------------------------------------------------


def test_a_rendered_page_with_no_challenge_is_a_genuine_clean_result() -> None:
    signal = analyze_captcha_gate(
        email(LINK), renderer=FakeRenderer(), traces=[settled(LINK, TERMINAL)]
    )

    assert signal.score == 0.0
    assert signal.error is None
    assert signal.metadata["fired"] is False
    assert signal.metadata["pages_rendered"] == 1
    assert signal.metadata["matches"] == []
    assert signal.metadata["clean_pages"][0]["terminal_url"] == TERMINAL
    assert "without a Turnstile, hCaptcha or reCAPTCHA" in signal.evidence


def test_a_direct_link_is_its_own_terminal_page() -> None:
    renderer = FakeRenderer()
    signal = analyze_captcha_gate(
        email(LINK), renderer=renderer, traces=[settled(LINK, LINK)]
    )

    assert renderer.calls == [LINK]
    assert signal.error is None


# --- 6, 7, 8, 9. abstention ----------------------------------------------


@pytest.mark.parametrize(
    "outcome, reason",
    [
        (RenderOutcome.UNREACHABLE, "net::ERR_CONNECTION_REFUSED"),
        (RenderOutcome.TIMED_OUT, "context killed after the hard timeout"),
        (RenderOutcome.NOT_ATTEMPTED, "sandbox unavailable"),
    ],
)
def test_a_failed_render_is_an_abstention(outcome: RenderOutcome, reason: str) -> None:
    renderer = FakeRenderer({TERMINAL: failed(TERMINAL, outcome, reason)})

    signal = analyze_captcha_gate(
        email(LINK), renderer=renderer, traces=[settled(LINK, TERMINAL)]
    )

    assert signal.score == 0.0
    assert signal.metadata["fired"] is False
    assert signal.error is not None
    assert reason in signal.metadata["failures"][0]["error"]
    assert "absence of information, not a clean result" in signal.evidence


def test_no_renderer_configured_abstains() -> None:
    """The state this repository ships in: there is no Playwright sandbox."""
    signal = analyze_captcha_gate(email(LINK), traces=[settled(LINK, TERMINAL)])

    assert signal.score == 0.0
    assert signal.error == "no page renderer configured"
    assert signal.metadata["pages_rendered"] == 0
    assert signal.metadata["failures"][0]["terminal_url"] == TERMINAL
    assert "not a clean result" in signal.evidence


@pytest.mark.parametrize(
    "outcome",
    [RedirectOutcome.TIMED_OUT, RedirectOutcome.UNREACHABLE, RedirectOutcome.NOT_ATTEMPTED],
)
def test_a_url_whose_chain_was_not_followed_has_no_terminal_page(
    outcome: RedirectOutcome,
) -> None:
    renderer = FakeRenderer()

    signal = analyze_captcha_gate(
        email(LINK), renderer=renderer, traces=[unfollowed(LINK, outcome)]
    )

    assert signal.score == 0.0
    assert signal.error == "no terminal URL was available to render"
    assert renderer.calls == []          # nothing was rendered on a guess
    assert "not a clean result" in signal.evidence


def test_no_traces_at_all_abstains_without_rendering() -> None:
    """The pipeline state today: no follower, so no chain is ever resolved."""
    renderer = FakeRenderer()

    signal = analyze_captcha_gate(email(LINK), renderer=renderer)

    assert signal.score == 0.0
    assert signal.error is not None
    assert renderer.calls == []


def test_a_capped_chain_offers_no_terminal_page() -> None:
    """A chain still redirecting at the cap never came to rest by contract."""
    capped = RedirectTrace(
        url=LINK,
        outcome=RedirectOutcome.CAPPED,
        hops=tuple(
            RedirectHop(
                url=f"https://h{i}.example/x",
                status_code=302,
                location=f"https://h{i + 1}.example/x",
                target=f"https://h{i + 1}.example/x",
            )
            for i in range(8)
        ),
    )
    renderer = FakeRenderer()

    signal = analyze_captcha_gate(email(LINK), renderer=renderer, traces=[capped])

    assert renderer.calls == []
    assert signal.error is not None


# --- 10, 11, 12. several URLs --------------------------------------------


def test_a_finding_survives_a_sibling_render_failure() -> None:
    renderer = FakeRenderer(
        {
            TERMINAL: rendered(TERMINAL, challenge(ChallengeKind.TURNSTILE, "cf-turnstile")),
            SECOND_TERMINAL: failed(SECOND_TERMINAL, RenderOutcome.TIMED_OUT, "hung"),
        }
    )

    signal = analyze_captcha_gate(
        email(LINK, SECOND),
        renderer=renderer,
        traces=[settled(LINK, TERMINAL), settled(SECOND, SECOND_TERMINAL)],
    )

    assert signal.score == CAPTCHA_GATE_SCORE
    assert signal.error is None                       # the finding stands
    assert len(signal.metadata["matches"]) == 1
    assert len(signal.metadata["failures"]) == 1
    assert "could not be rendered" in signal.evidence  # the gap is still reported


def test_several_clean_pages_are_a_clean_result() -> None:
    signal = analyze_captcha_gate(
        email(LINK, SECOND),
        renderer=FakeRenderer(),
        traces=[settled(LINK, TERMINAL), settled(SECOND, SECOND_TERMINAL)],
    )

    assert signal.score == 0.0
    assert signal.error is None
    assert signal.metadata["pages_rendered"] == 2
    assert signal.metadata["failures"] == []


def test_a_clean_verdict_is_not_issued_while_a_page_is_unrendered() -> None:
    renderer = FakeRenderer(
        {SECOND_TERMINAL: failed(SECOND_TERMINAL, RenderOutcome.UNREACHABLE, "refused")}
    )

    signal = analyze_captcha_gate(
        email(LINK, SECOND),
        renderer=renderer,
        traces=[settled(LINK, TERMINAL), settled(SECOND, SECOND_TERMINAL)],
    )

    assert signal.score == 0.0
    assert signal.error is not None
    assert signal.metadata["pages_rendered"] == 1
    assert "not fully checked" in signal.evidence


def test_each_url_is_rendered_independently() -> None:
    renderer = FakeRenderer()

    analyze_captcha_gate(
        email(LINK, SECOND),
        renderer=renderer,
        traces=[settled(LINK, TERMINAL), settled(SECOND, SECOND_TERMINAL)],
    )

    assert renderer.calls == [TERMINAL, SECOND_TERMINAL]


def test_only_urls_with_a_terminal_page_are_rendered() -> None:
    renderer = FakeRenderer(
        {TERMINAL: rendered(TERMINAL, challenge(ChallengeKind.HCAPTCHA, "h-captcha"))}
    )

    signal = analyze_captcha_gate(
        email(LINK, SECOND),
        renderer=renderer,
        traces=[settled(LINK, TERMINAL), unfollowed(SECOND)],
    )

    assert renderer.calls == [TERMINAL]
    assert signal.score == CAPTCHA_GATE_SCORE
    assert signal.metadata["urls_checked"] == 2
    assert signal.metadata["urls_with_terminal"] == 1


# --- 13, 14, 15. evidence and provenance ---------------------------------


def test_the_terminal_url_is_preserved_in_metadata() -> None:
    signal = analyze_captcha_gate(
        email(LINK),
        renderer=FakeRenderer(
            {TERMINAL: rendered(TERMINAL, challenge(ChallengeKind.TURNSTILE, "cf-turnstile"))}
        ),
        traces=[settled(LINK, TERMINAL)],
    )

    match = signal.metadata["matches"][0]
    assert match["terminal_url"] == TERMINAL
    assert match["rendered_url"] == TERMINAL


def test_the_original_url_is_what_evidence_quotes() -> None:
    messy = "HTTPS://Links.Evil-Phish.com:443/A?x=1"
    signal = analyze_captcha_gate(
        email(messy),
        renderer=FakeRenderer(
            {TERMINAL: rendered(TERMINAL, challenge(ChallengeKind.TURNSTILE, "cf-turnstile"))}
        ),
        traces=[settled(messy, TERMINAL)],
    )

    assert messy in signal.evidence
    assert signal.metadata["matches"][0]["url"] == messy


def test_the_challenge_marker_is_preserved() -> None:
    marker = "challenges.cloudflare.com/turnstile/v0/api.js?render=explicit"
    signal = analyze_captcha_gate(
        email(LINK),
        renderer=FakeRenderer(
            {TERMINAL: rendered(TERMINAL, challenge(ChallengeKind.TURNSTILE, marker))}
        ),
        traces=[settled(LINK, TERMINAL)],
    )

    assert signal.metadata["matches"][0]["markers"] == [marker]


def test_terminal_urls_maps_only_answered_settled_chains() -> None:
    message = email(LINK, SECOND)
    candidates = candidate_urls(message)
    mapping = terminal_urls(
        candidates, [settled(LINK, TERMINAL), unfollowed(SECOND)]
    )

    assert mapping == {LINK: TERMINAL}


# --- 16, 17, 18, 19, 20. safety ------------------------------------------


def test_a_renderer_that_raises_is_an_abstention_not_a_verdict() -> None:
    class Broken:
        def render(self, url: str) -> RenderResult:
            raise RuntimeError("the browser context crashed")

    signal = analyze_captcha_gate(
        email(LINK), renderer=Broken(), traces=[settled(LINK, TERMINAL)]
    )

    assert signal.score == 0.0
    assert signal.error is not None
    assert "the browser context crashed" in signal.metadata["failures"][0]["error"]


def test_a_renderer_returning_the_wrong_type_is_an_abstention() -> None:
    class Wrong:
        def render(self, url: str):
            return {"captcha": True}

    signal = analyze_captcha_gate(
        email(LINK), renderer=Wrong(), traces=[settled(LINK, TERMINAL)]
    )

    assert signal.error is not None
    assert "not a RenderResult" in signal.metadata["failures"][0]["error"]


def test_repeated_analysis_is_deterministic() -> None:
    traces = [settled(LINK, TERMINAL), settled(SECOND, SECOND_TERMINAL)]
    page = rendered(TERMINAL, challenge(ChallengeKind.TURNSTILE, "cf-turnstile"))
    message = email(LINK, SECOND)

    results = {
        (
            analyze_captcha_gate(
                message, renderer=FakeRenderer({TERMINAL: page}), traces=traces
            ).score,
            analyze_captcha_gate(
                message, renderer=FakeRenderer({TERMINAL: page}), traces=traces
            ).evidence,
        )
        for _ in range(5)
    }

    assert len(results) == 1


def test_nothing_upstream_is_mutated() -> None:
    message = email(LINK)
    trace = settled(LINK, TERMINAL)
    candidates_before = candidate_urls(message)

    analyze_captcha_gate(
        message,
        renderer=FakeRenderer(
            {TERMINAL: rendered(TERMINAL, challenge(ChallengeKind.TURNSTILE, "cf-turnstile"))}
        ),
        traces=[trace],
    )

    assert message.urls[0].url == LINK
    assert message.urls[0].redirect_chain == []
    assert message.urls[0].final_url is None
    assert trace.final_url == TERMINAL
    assert trace.hops[0].target == TERMINAL
    assert [c.raw for c in candidate_urls(message)] == [c.raw for c in candidates_before]


def test_the_signal_never_follows_a_redirect_itself() -> None:
    """There is one hop-following mechanism in this layer, and it is not here."""
    renderer = FakeRenderer()

    analyze_captcha_gate(
        email(LINK), renderer=renderer, traces=[settled(LINK, TERMINAL)]
    )

    assert renderer.calls == [TERMINAL]  # the terminal page only, never a hop


def test_playwright_is_never_imported() -> None:
    assert "playwright" not in sys.modules

    analyze_captcha_gate(
        email(LINK), renderer=FakeRenderer(), traces=[settled(LINK, TERMINAL)]
    )

    assert "playwright" not in sys.modules


def test_no_renderer_implementation_ships_in_this_repository() -> None:
    """Nothing in this layer can load a page: no browser, no egress, no fetch.

    `_BoundedRenderer` satisfies the protocol structurally but renders nothing
    itself - it is the per-message render cap, and it either delegates to the
    injected renderer or returns NOT_ATTEMPTED. It is excluded by name rather
    than by shape so a real renderer appearing here would still fail this.
    """
    import layers.l2_urls as module

    concrete = [
        name
        for name in dir(module)
        if isinstance(getattr(module, name), type)
        and name not in ("PageRenderer", "_BoundedRenderer")
        and hasattr(getattr(module, name), "render")
    ]
    assert concrete == []


def test_a_message_with_no_urls_is_a_genuine_negative() -> None:
    signal = analyze_captcha_gate(email(), renderer=FakeRenderer())

    assert signal.score == 0.0
    assert signal.error is None
    assert "no analysable URLs" in signal.evidence
