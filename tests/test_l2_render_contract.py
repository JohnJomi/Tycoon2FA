"""Unit tests for the Layer 2 render contract in layers/l2_urls.py.

The record and the seam only - no renderer exists, no browser is launched, and
no detection is implemented. These tests pin the invariants that keep a failed
render from being read as a page with no captcha on it.

An autouse guard forbids every socket and DNS entry point, and a separate test
asserts that importing or exercising this contract never imports Playwright.
"""

from __future__ import annotations

import sys

import pytest

from layers.l2_urls import (
    CHALLENGE_MARKERS,
    ChallengeEvidence,
    ChallengeKind,
    PageRenderer,
    RenderOutcome,
    RenderResult,
)

LANDING = "https://evil-phish.com/login"

FAILING = [RenderOutcome.UNREACHABLE, RenderOutcome.TIMED_OUT, RenderOutcome.NOT_ATTEMPTED]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("the render contract suite touched the network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket, "gethostbyname", forbidden)


def turnstile() -> ChallengeEvidence:
    return ChallengeEvidence(
        kind=ChallengeKind.TURNSTILE,
        marker="challenges.cloudflare.com/turnstile/v0/api.js",
        detail="script[src]",
    )


# --- 1. an ordinary rendered page -----------------------------------------


def test_a_rendered_page_with_no_challenge_is_a_genuine_clean_result() -> None:
    result = RenderResult(
        url=LANDING,
        outcome=RenderOutcome.RENDERED,
        final_url=LANDING,
        status_code=200,
        title="Sign in",
    )

    assert result.is_answer is True
    assert result.has_challenge is False
    assert result.challenges == ()
    assert result.kinds == ()
    assert result.error is None


# --- 2. a challenged page -------------------------------------------------


def test_a_rendered_page_with_a_challenge_records_its_evidence() -> None:
    result = RenderResult(
        url=LANDING,
        outcome=RenderOutcome.RENDERED,
        challenges=(turnstile(),),
        final_url=LANDING,
        status_code=200,
    )

    assert result.is_answer is True
    assert result.has_challenge is True
    assert result.kinds == (ChallengeKind.TURNSTILE,)
    assert result.challenges[0].marker.startswith("challenges.cloudflare.com")
    assert result.challenges[0].detail == "script[src]"


def test_the_three_challenge_kinds_are_the_three_the_architecture_names() -> None:
    """Section 4: 'detect Turnstile/hCaptcha/reCAPTCHA in DOM'."""
    assert [k.value for k in ChallengeKind] == ["turnstile", "hcaptcha", "recaptcha"]
    assert set(CHALLENGE_MARKERS) == set(ChallengeKind)
    assert all(markers for markers in CHALLENGE_MARKERS.values())


def test_several_challenges_deduplicate_by_kind_in_first_seen_order() -> None:
    result = RenderResult(
        url=LANDING,
        outcome=RenderOutcome.RENDERED,
        challenges=(
            ChallengeEvidence(ChallengeKind.HCAPTCHA, "h-captcha"),
            turnstile(),
            ChallengeEvidence(ChallengeKind.HCAPTCHA, "hcaptcha.com/1/api.js"),
        ),
        final_url=LANDING,
        status_code=200,
    )

    assert result.kinds == (ChallengeKind.HCAPTCHA, ChallengeKind.TURNSTILE)


def test_a_challenge_must_carry_the_marker_that_matched() -> None:
    """'A captcha was detected' is not evidence; section 2 requires the why."""
    with pytest.raises(ValueError, match="marker"):
        ChallengeEvidence(kind=ChallengeKind.TURNSTILE, marker="   ")


def test_a_challenge_kind_must_be_a_challenge_kind() -> None:
    with pytest.raises(TypeError, match="ChallengeKind"):
        ChallengeEvidence(kind="turnstile", marker="cf-turnstile")  # type: ignore[arg-type]


# --- 3, 4, 5. the failing outcomes ----------------------------------------


@pytest.mark.parametrize(
    "outcome, reason",
    [
        (RenderOutcome.UNREACHABLE, "net::ERR_CONNECTION_REFUSED"),
        (RenderOutcome.TIMED_OUT, "context killed after the hard timeout"),
        (RenderOutcome.NOT_ATTEMPTED, "no sandboxed egress configured"),
    ],
)
def test_a_failed_render_is_not_an_answer(outcome: RenderOutcome, reason: str) -> None:
    result = RenderResult(url=LANDING, outcome=outcome, error=reason)

    assert result.is_answer is False
    assert outcome.is_answer is False
    assert result.error == reason
    assert result.final_url is None
    assert result.status_code is None


def test_not_attempted_is_the_state_this_repository_ships_in() -> None:
    result = RenderResult.not_attempted(LANDING, "no page renderer configured")

    assert result.outcome is RenderOutcome.NOT_ATTEMPTED
    assert result.is_answer is False
    assert result.has_challenge is False
    assert "renderer" in result.error


# --- 7 & 8. invalid combinations ------------------------------------------


@pytest.mark.parametrize("outcome", FAILING)
def test_a_failed_render_must_state_why(outcome: RenderOutcome) -> None:
    with pytest.raises(ValueError, match="state why"):
        RenderResult(url=LANDING, outcome=outcome)


@pytest.mark.parametrize("outcome", FAILING)
def test_a_failed_render_cannot_report_challenges(outcome: RenderOutcome) -> None:
    """There was no DOM to inspect, so it cannot have found anything in one."""
    with pytest.raises(ValueError, match="cannot"):
        RenderResult(
            url=LANDING, outcome=outcome, error="boom", challenges=(turnstile(),)
        )


@pytest.mark.parametrize("outcome", FAILING)
def test_a_failed_render_cannot_claim_a_page(outcome: RenderOutcome) -> None:
    with pytest.raises(ValueError, match="no page"):
        RenderResult(url=LANDING, outcome=outcome, error="boom", final_url=LANDING)
    with pytest.raises(ValueError, match="no page"):
        RenderResult(url=LANDING, outcome=outcome, error="boom", status_code=200)


@pytest.mark.parametrize("outcome", FAILING)
def test_a_failed_render_never_reads_as_a_clean_page(outcome: RenderOutcome) -> None:
    """The invariant this whole contract exists for."""
    result = RenderResult(url=LANDING, outcome=outcome, error="browser died")

    assert result.has_challenge is False   # but ...
    assert result.is_answer is False       # ... this is what a caller must check
    assert result.challenges == ()


def test_a_rendered_page_may_not_carry_an_error() -> None:
    with pytest.raises(ValueError, match="must not carry an error"):
        RenderResult(url=LANDING, outcome=RenderOutcome.RENDERED, error="boom")


def test_the_outcome_must_be_a_render_outcome() -> None:
    with pytest.raises(TypeError, match="RenderOutcome"):
        RenderResult(url=LANDING, outcome="rendered")  # type: ignore[arg-type]


@pytest.mark.parametrize("url", ["", "   "])
def test_a_result_must_name_the_url_it_rendered(url: str) -> None:
    with pytest.raises(ValueError, match="url"):
        RenderResult(url=url, outcome=RenderOutcome.RENDERED)


def test_elapsed_time_may_not_be_negative() -> None:
    with pytest.raises(ValueError, match="elapsed_ms"):
        RenderResult(url=LANDING, outcome=RenderOutcome.RENDERED, elapsed_ms=-1)


def test_the_url_rendered_is_preserved_exactly() -> None:
    messy = "HTTPS://Evil-Phish.com:443/Login?x=1"
    result = RenderResult(url=messy, outcome=RenderOutcome.RENDERED)

    assert result.url == messy


# --- 6. immutability ------------------------------------------------------


def test_the_records_are_immutable() -> None:
    result = RenderResult(url=LANDING, outcome=RenderOutcome.RENDERED)
    challenge = turnstile()

    with pytest.raises(Exception):
        result.outcome = RenderOutcome.UNREACHABLE  # type: ignore[misc]
    with pytest.raises(Exception):
        challenge.marker = "changed"  # type: ignore[misc]
    with pytest.raises(Exception):
        result.challenges = (challenge,)  # type: ignore[misc]


def test_nothing_upstream_is_mutated_by_building_a_result() -> None:
    from core.models import ExtractedURL, ParsedEmail, URLSource
    from layers.l2_urls import candidate_urls

    message = ParsedEmail(
        message_id="<m@example.com>",
        from_addr="s@example.com",
        urls=[ExtractedURL(url=LANDING, source=URLSource.ANCHOR_HREF)],
    )
    (candidate,) = candidate_urls(message)

    RenderResult(
        url=candidate.raw,
        outcome=RenderOutcome.RENDERED,
        challenges=(turnstile(),),
        final_url=LANDING,
        status_code=200,
    )

    assert message.urls[0].final_url is None
    assert message.urls[0].redirect_chain == []
    assert candidate.raw == LANDING


# --- 9, 10, 11. the seam, and what must not happen ------------------------


def test_a_renderer_is_anything_with_the_render_method() -> None:
    class FakeRenderer:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def render(self, url: str) -> RenderResult:
            self.calls.append(url)
            return RenderResult(
                url=url,
                outcome=RenderOutcome.RENDERED,
                challenges=(turnstile(),),
                final_url=url,
                status_code=200,
            )

    renderer = FakeRenderer()

    assert isinstance(renderer, PageRenderer)
    result = renderer.render(LANDING)
    assert renderer.calls == [LANDING]
    assert result.has_challenge is True


def test_a_fake_renderer_can_express_every_outcome() -> None:
    """The seam is total: a fake covers success and every failure without raising."""
    scripted = {
        LANDING: RenderResult(url=LANDING, outcome=RenderOutcome.RENDERED),
        "https://slow.example/": RenderResult(
            url="https://slow.example/", outcome=RenderOutcome.TIMED_OUT, error="hung"
        ),
        "https://dead.example/": RenderResult(
            url="https://dead.example/", outcome=RenderOutcome.UNREACHABLE, error="refused"
        ),
    }

    class ScriptedRenderer:
        def render(self, url: str) -> RenderResult:
            return scripted.get(url) or RenderResult.not_attempted(url, "not scripted")

    renderer = ScriptedRenderer()

    assert isinstance(renderer, PageRenderer)
    assert [renderer.render(u).outcome for u in scripted] == [
        RenderOutcome.RENDERED,
        RenderOutcome.TIMED_OUT,
        RenderOutcome.UNREACHABLE,
    ]
    assert renderer.render("https://unknown.example/").is_answer is False


def test_no_renderer_implementation_ships_in_this_repository() -> None:
    """Rendering needs the Docker sandbox and egress that do not exist yet."""
    import layers.l2_urls as module

    concrete = [
        name
        for name in dir(module)
        if isinstance(getattr(module, name), type)
        # `_BoundedRenderer` is the per-message render cap: it renders nothing
        # itself, and either delegates or returns NOT_ATTEMPTED. Excluded by
        # name, so a real renderer appearing here would still fail this.
        and name not in ("PageRenderer", "_BoundedRenderer")
        and hasattr(getattr(module, name), "render")
    ]
    assert concrete == []


def test_playwright_is_never_imported() -> None:
    """No browser is launched, and none can be: the dependency is not installed."""
    assert "playwright" not in sys.modules

    RenderResult(
        url=LANDING,
        outcome=RenderOutcome.RENDERED,
        challenges=(turnstile(),),
        final_url=LANDING,
        status_code=200,
    )

    assert "playwright" not in sys.modules


def test_no_detection_is_implemented_yet() -> None:
    """CHALLENGE_MARKERS records the requirement; nothing reads it."""
    import inspect

    import layers.l2_urls as module

    source = inspect.getsource(module)
    # The table is defined once and never consulted - detection is a later change.
    assert source.count("CHALLENGE_MARKERS") == 2  # __all__ entry and the definition
