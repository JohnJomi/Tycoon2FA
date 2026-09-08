"""Integration tests for the Layer 2 entry point and its wiring into the
orchestrator.

Scope is the *integration*, not the five signals - each of those has its own
focused suite and none of them is retested here. What is tested is the work the
layer entry point does on their behalf: computing the candidate set once,
walking each redirect chain once, handing the same trace objects to both
signals that read them, bounding how many pages a message can make the browser
render, and turning "the layer learned nothing" into the orchestrator's
incomplete state rather than a clean 0.0 at Layer 2's 0.30 weight.

Every seam is an injected fake. An autouse guard forbids every socket and DNS
entry point for the whole file: no network, no browser, no QR decoder, no DNS.
"""

from __future__ import annotations

from email.message import EmailMessage

import pytest

from core.models import (
    DetectionLayer,
    ExtractedURL,
    LayerResult,
    ParsedEmail,
    RiskLevel,
    URLSource,
)
from core.orchestrator import DEFAULT_LAYERS, DEFAULT_LAYER_TIMEOUTS, run_layers
from ingest.parser import parse_email
from layers import l2_urls
from layers.l2_urls import (
    CAPTCHA_GATE_SCORE,
    MAX_RENDERED_PAGES_PER_EMAIL,
    ChallengeEvidence,
    ChallengeKind,
    Layer2Uninformative,
    QRDecodeResult,
    QROutcome,
    QRPayload,
    RedirectHop,
    RedirectOutcome,
    RedirectTrace,
    RenderOutcome,
    RenderResult,
    analyze,
    analyze_async,
    candidate_urls,
)
from scoring.composite import ScoringWeights, layer_contributions, score

DIRECT = "https://direct.evil-phish.com/login"
LINK = "https://links.evil-phish.com/a"
TERMINAL = "https://evil-phish.com/signin"
CAPPED_LINK = "https://loop.evil-phish.com/z"
LONE = "https://lonely.example.net/x"

PNG = b"\x89PNG\r\n\x1a\n" + b"fake image bytes" * 4
QR_URL = "https://qr.evil-phish.com/verify"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("the L2 orchestration suite touched the network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket, "gethostbyname", forbidden)


# --------------------------------------------------------------------------
# Message builders and fakes
# --------------------------------------------------------------------------


def email(*urls: str, anchor: str | None = None) -> ParsedEmail:
    return ParsedEmail(
        message_id="<m@example.com>",
        from_addr="sender@example.com",
        subject="Invoice",
        body_text="See the portal.",
        urls=[
            ExtractedURL(url=u, source=URLSource.ANCHOR_HREF, anchor_text=anchor)
            for u in urls
        ],
    )


def email_with_image(*, body_url: str | None = None) -> ParsedEmail:
    """A real parsed message carrying a PNG, so payload bytes are available."""
    message = EmailMessage()
    message["From"] = "Sender <sender@example.com>"
    message["To"] = "victim@corp-invoices.com"
    message["Subject"] = "Invoice"
    message["Message-ID"] = "<m@example.com>"
    message.set_content(f"Scan the code. {body_url or ''}")
    message.add_attachment(PNG, maintype="image", subtype="png", filename="qr.png")
    return parse_email(message.as_bytes())


def settled(url: str, final: str) -> RedirectTrace:
    hops = (
        ()
        if url == final
        else (RedirectHop(url=url, status_code=302, location=final, target=final),)
    )
    return RedirectTrace(url=url, outcome=RedirectOutcome.SETTLED, hops=hops, final_url=final)


def capped(url: str) -> RedirectTrace:
    hops = tuple(
        RedirectHop(
            url=f"{url}/{i}", status_code=302, location=f"{url}/{i + 1}", target=f"{url}/{i + 1}"
        )
        for i in range(l2_urls.REDIRECT_HOP_CAP)
    )
    return RedirectTrace(url=url, outcome=RedirectOutcome.CAPPED, hops=hops)


class FakeFollower:
    """Answers from a table, and records every URL it was asked about."""

    def __init__(self, table: dict[str, RedirectTrace] | None = None) -> None:
        self.table = table or {}
        self.calls: list[str] = []

    def follow(self, url: str) -> RedirectTrace:
        self.calls.append(url)
        return self.table.get(url) or settled(url, url)


class FakeRenderer:
    """Renders from a table, and records every URL it was asked to render."""

    def __init__(self, gated: dict[str, ChallengeKind] | None = None) -> None:
        self.gated = gated or {}
        self.calls: list[str] = []

    def render(self, url: str) -> RenderResult:
        self.calls.append(url)
        kind = self.gated.get(url)
        challenges = (
            (ChallengeEvidence(kind=kind, marker=f"{kind.value}-widget"),) if kind else ()
        )
        return RenderResult(
            url=url,
            outcome=RenderOutcome.RENDERED,
            challenges=challenges,
            final_url=url,
            status_code=200,
        )


class FakeDecoder:
    def __init__(self, payloads: tuple[str, ...] = ()) -> None:
        self.payloads = payloads
        self.calls = 0

    def decode(self, payload: bytes, attachment) -> QRDecodeResult:
        self.calls += 1
        return QRDecodeResult(
            attachment=attachment,
            outcome=QROutcome.DECODED,
            payloads=tuple(QRPayload(data=d, symbol="QRCODE") for d in self.payloads),
        )


def by_name(signals) -> dict[str, object]:
    return {s.name: s for s in signals}


# --------------------------------------------------------------------------
# A. The candidate set is computed once for the whole layer
# --------------------------------------------------------------------------


def test_candidate_urls_is_computed_once_for_the_whole_layer(monkeypatch):
    """Five signals, one candidate set.

    Left to themselves each signal rebuilds the set from the message; the entry
    point exists so the parse happens once and every signal is answering about
    the same URLs.
    """
    calls: list[int] = []
    real = l2_urls.candidate_urls

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(l2_urls, "candidate_urls", counting)

    signals = analyze(email(DIRECT, LINK), follower=FakeFollower(), renderer=FakeRenderer())

    assert len(calls) == 1
    assert len(signals) == 5


def test_every_signal_sees_the_same_candidate_objects(monkeypatch):
    """Not merely the same count - the identical objects."""
    seen: list[list[object]] = []
    real = l2_urls.candidate_urls

    for name in (
        "analyze_base64_email_param",
        "analyze_redirect_depth",
        "analyze_captcha_gate",
        "analyze_qr_url",
        "analyze_domain_mismatch_brand",
    ):
        original = getattr(l2_urls, name)

        def spy(email, *, _original=original, **kwargs):
            seen.append(kwargs.get("candidates"))
            return _original(email, **kwargs)

        monkeypatch.setattr(l2_urls, name, spy)

    analyze(email(DIRECT, LINK), follower=FakeFollower())

    assert len(seen) == 5
    first = seen[0]
    assert first is not None
    for other in seen[1:]:
        assert other is first


# --------------------------------------------------------------------------
# B, C. One walk per candidate, and the same traces reach both readers
# --------------------------------------------------------------------------


def test_the_follower_runs_exactly_once_per_candidate():
    """Two signals read redirect traces; the chains are walked once, not twice.

    These are requests to infrastructure the attacker controls. Walking them
    per signal would double the load and let the second walk disagree with the
    first.
    """
    follower = FakeFollower({LINK: settled(LINK, TERMINAL)})

    analyze(email(DIRECT, LINK), follower=follower, renderer=FakeRenderer())

    assert follower.calls == [DIRECT, LINK]


def test_a_url_repeated_in_the_message_is_walked_once():
    """First-observation canonical dedup survives the entry point."""
    follower = FakeFollower()
    message = ParsedEmail(
        message_id="<m@example.com>",
        from_addr="sender@example.com",
        urls=[
            ExtractedURL(url=DIRECT, source=URLSource.ANCHOR_HREF, anchor_text="Sign in"),
            ExtractedURL(url=DIRECT, source=URLSource.PLAIN_TEXT),
            ExtractedURL(url=f"{DIRECT}", source=URLSource.IMG_SRC),
        ],
    )

    analyze(message, follower=follower)

    assert follower.calls == [DIRECT]


def test_the_same_trace_objects_reach_redirect_depth_and_captcha_gate(monkeypatch):
    """Object identity, so no second walk can have produced them."""
    captured: dict[str, object] = {}
    for name in ("analyze_redirect_depth", "analyze_captcha_gate"):
        original = getattr(l2_urls, name)

        def spy(email, *, _name=name, _original=original, **kwargs):
            captured[_name] = kwargs.get("traces")
            return _original(email, **kwargs)

        monkeypatch.setattr(l2_urls, name, spy)

    analyze(email(DIRECT, LINK), follower=FakeFollower(), renderer=FakeRenderer())

    depth_traces = captured["analyze_redirect_depth"]
    gate_traces = captured["analyze_captcha_gate"]
    assert depth_traces is gate_traces
    assert len(depth_traces) == 2


def test_captcha_gate_never_follows_a_redirect_of_its_own():
    """There is one hop-following mechanism in this layer, and it is the seam.

    The follower's call log is the whole proof: two candidates, two calls, and
    the renderer still received the terminal URLs.
    """
    follower = FakeFollower({LINK: settled(LINK, TERMINAL)})
    renderer = FakeRenderer()

    analyze(email(DIRECT, LINK), follower=follower, renderer=renderer)

    assert len(follower.calls) == 2
    assert set(renderer.calls) == {DIRECT, TERMINAL}


# --------------------------------------------------------------------------
# D-G. Which pages are rendered, and which are never rendered
# --------------------------------------------------------------------------


def test_a_settled_direct_url_is_rendered_as_its_own_terminal_page():
    renderer = FakeRenderer()

    analyze(email(DIRECT), follower=FakeFollower(), renderer=renderer)

    assert renderer.calls == [DIRECT]


def test_a_redirected_chain_renders_the_final_url_not_the_link():
    renderer = FakeRenderer()

    analyze(
        email(LINK), follower=FakeFollower({LINK: settled(LINK, TERMINAL)}), renderer=renderer
    )

    assert renderer.calls == [TERMINAL]


def test_a_capped_chain_is_never_rendered():
    """A capped chain has no final URL, so there is no terminal page to look at.

    Rendering an intermediate hop would be looking at the wrong page and
    reporting the answer as if it were about the destination.
    """
    renderer = FakeRenderer()

    signals = by_name(
        analyze(
            email(CAPPED_LINK),
            follower=FakeFollower({CAPPED_LINK: capped(CAPPED_LINK)}),
            renderer=renderer,
        )
    )

    assert renderer.calls == []
    assert signals["captcha_gate"].error is not None


@pytest.mark.parametrize(
    "outcome",
    [RedirectOutcome.UNREACHABLE, RedirectOutcome.TIMED_OUT, RedirectOutcome.NOT_ATTEMPTED],
)
def test_an_unfollowed_chain_is_never_rendered(outcome):
    renderer = FakeRenderer()
    trace = RedirectTrace(url=LINK, outcome=outcome, error="chain not followed")

    analyze(email(LINK), follower=FakeFollower({LINK: trace}), renderer=renderer)

    assert renderer.calls == []


def test_a_url_with_no_trace_is_never_assumed_to_be_its_own_terminal_page():
    """Without a followed chain the layer does not know a URL is terminal.

    With no follower there is no trace that answers, so nothing is rendered -
    the alternative, rendering the link itself, is a claim the layer has no
    basis for.
    """
    renderer = FakeRenderer()

    signals = by_name(analyze(email(DIRECT), follower=None, renderer=renderer))

    assert renderer.calls == []
    assert signals["captcha_gate"].error is not None


# --------------------------------------------------------------------------
# H. The render cap
# --------------------------------------------------------------------------


def test_the_render_cap_is_a_small_explicit_constant():
    assert isinstance(MAX_RENDERED_PAGES_PER_EMAIL, int)
    assert 1 <= MAX_RENDERED_PAGES_PER_EMAIL <= 5


def test_render_count_is_bounded_however_many_links_a_message_carries():
    """The number of browser renders is ours to choose, not the attacker's."""
    urls = [f"https://evil-phish.com/link-{i}" for i in range(40)]
    renderer = FakeRenderer()

    analyze(email(*urls), follower=FakeFollower(), renderer=renderer)

    assert len(renderer.calls) == MAX_RENDERED_PAGES_PER_EMAIL


def test_the_cap_takes_the_first_eligible_terminals_in_candidate_order():
    """Deterministic: the same message always renders the same pages."""
    urls = [f"https://evil-phish.com/link-{i}" for i in range(6)]
    renderer = FakeRenderer()

    analyze(email(*urls), follower=FakeFollower(), renderer=renderer)

    assert renderer.calls == urls[:MAX_RENDERED_PAGES_PER_EMAIL]


def test_pages_past_the_cap_are_uncheckable_not_clean():
    """A page nobody looked at must never be reported as a page with no captcha."""
    urls = [f"https://evil-phish.com/link-{i}" for i in range(6)]

    signals = by_name(analyze(email(*urls), follower=FakeFollower(), renderer=FakeRenderer()))
    gate = signals["captcha_gate"]

    assert gate.score == 0.0
    assert gate.error is not None
    assert gate.metadata["pages_rendered"] == MAX_RENDERED_PAGES_PER_EMAIL
    assert len(gate.metadata["failures"]) == len(urls) - MAX_RENDERED_PAGES_PER_EMAIL
    assert "render cap" in gate.metadata["failures"][0]["error"]


def test_a_finding_inside_the_cap_still_fires():
    """The cap bounds the work; it does not suppress what the work found."""
    urls = [f"https://evil-phish.com/link-{i}" for i in range(6)]

    signals = by_name(
        analyze(
            email(*urls),
            follower=FakeFollower(),
            renderer=FakeRenderer({urls[0]: ChallengeKind.TURNSTILE}),
        )
    )

    assert signals["captcha_gate"].score == CAPTCHA_GATE_SCORE
    assert signals["captcha_gate"].error is None


# --------------------------------------------------------------------------
# The QR seam: decoded once, re-entering the shared candidate set
# --------------------------------------------------------------------------


def test_a_decoded_qr_url_joins_the_shared_candidate_set_and_is_walked():
    """Section 4's "extracted URL re-enters the URL signal set", once.

    The proof that the re-entry happened before the shared work: the follower
    was asked about the QR URL, which appears nowhere in the message text.
    """
    follower = FakeFollower()
    decoder = FakeDecoder((QR_URL,))

    signals = by_name(
        analyze(email_with_image(), follower=follower, decoder=decoder, renderer=FakeRenderer())
    )

    assert QR_URL in follower.calls
    assert signals["qr_url"].score == l2_urls.QR_URL_SCORE


def test_each_image_is_decoded_exactly_once():
    """The entry point decodes for the candidate set and for the signal at once."""
    decoder = FakeDecoder((QR_URL,))

    analyze(email_with_image(), follower=FakeFollower(), decoder=decoder)

    assert decoder.calls == 1


# --------------------------------------------------------------------------
# I-L. Layer2Uninformative
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_offline_signals_clean_while_the_network_ones_abstained_is_uninformative():
    """The case this class exists for.

    base64 clean + brand clean, with the redirect walk, the render and the
    decode all unavailable, is not a clean Layer 2 - it is an unexamined one.
    Scoring it 0.0 at 0.30 would let an outage argue a message is safe.
    """
    with pytest.raises(Layer2Uninformative):
        await analyze_async(email(DIRECT))


@pytest.mark.asyncio
async def test_no_renderer_is_uninformative_when_no_other_signal_found_anything():
    """The browser is the missing piece; the rest of the layer found nothing."""
    with pytest.raises(Layer2Uninformative) as excinfo:
        await analyze_async(email(DIRECT), follower=FakeFollower(), renderer=None)

    assert "captcha_gate" in str(excinfo.value)


@pytest.mark.asyncio
async def test_no_decoder_is_uninformative_when_the_message_carries_an_image():
    """An undecoded image is not an image with nothing in it.

    Every other signal here answers cleanly; the layer is still uninformative,
    because the one surface this message actually leads with went uninspected.
    """
    message = email_with_image(body_url=DIRECT)

    with pytest.raises(Layer2Uninformative) as excinfo:
        await analyze_async(
            message, follower=FakeFollower(), renderer=FakeRenderer(), decoder=None
        )

    assert "qr_url" in str(excinfo.value)


@pytest.mark.asyncio
async def test_a_finding_survives_another_signal_abstaining():
    """A URL that hides a recipient address hides one whether or not the
    browser was available."""
    encoded = "dmljdGltQGNvcnAtaW52b2ljZXMuY29t"  # victim@corp-invoices.com
    message = email(f"https://evil-phish.com/go?e={encoded}")

    signals = await analyze_async(message, follower=FakeFollower(), renderer=None)

    names = by_name(signals)
    assert names["base64_email_param"].score > 0.0
    assert names["captcha_gate"].error is not None


@pytest.mark.asyncio
async def test_a_fully_analysed_clean_message_is_an_informative_clean_result():
    """Every applicable surface was actually looked at and nothing was found.

    This one *should* consume Layer 2's weight: it is a real negative, not an
    absence of information.
    """
    signals = await analyze_async(
        email(DIRECT), follower=FakeFollower(), renderer=FakeRenderer(), decoder=FakeDecoder()
    )

    assert len(signals) == 5
    assert all(s.error is None for s in signals)
    assert all(s.score == 0.0 for s in signals)


def blank_email() -> ParsedEmail:
    """A message with no URLs and no attachments - no Layer 2 surface at all."""
    return ParsedEmail(
        message_id="<m@example.com>",
        from_addr="sender@example.com",
        subject="Lunch",
        body_text="See you at one.",
    )


@pytest.mark.asyncio
async def test_a_message_with_no_urls_is_an_informative_clean_result():
    """Layer 2 examined the URL surface and found nothing on it to inspect.

    That is a conclusion, not an absence of information: no signal abstained,
    nothing went unchecked, and there is no missing dependency hiding behind
    the answer. So it consumes Layer 2's 0.30 like any other genuine negative.
    """
    signals = await analyze_async(
        blank_email(), follower=FakeFollower(), renderer=FakeRenderer(), decoder=FakeDecoder()
    )

    assert len(signals) == 5
    assert all(s.error is None for s in signals)
    assert all(s.score == 0.0 for s in signals)


@pytest.mark.asyncio
async def test_a_message_with_no_urls_is_not_uninformative_even_with_no_seams():
    """The missing browser and decoder are irrelevant when there is nothing to
    give them: no signal abstains, so nothing went uninspected."""
    signals = await analyze_async(blank_email())

    assert all(s.error is None for s in signals)


@pytest.mark.asyncio
async def test_a_message_with_no_urls_completes_at_the_orchestrator():
    async def l2(email):
        return await analyze_async(email)

    results = {
        r.layer: r for r in await run_layers(blank_email(), layers={DetectionLayer.L2: l2})
    }
    l2_result = results[DetectionLayer.L2]

    assert l2_result.completed is True
    assert l2_result.error is None
    assert [s.name for s in l2_result.signals] == [
        "base64_email_param",
        "redirect_depth",
        "captcha_gate",
        "qr_url",
        "domain_mismatch_brand",
    ]


def test_a_message_with_no_urls_consumes_layer_twos_weight():
    """The counterpart of `test_an_uninformative_l2_does_not_consume_its_030_weight`.

    L1 at 0.85 beside a genuinely clean L2: the 0.30 is spent, and the
    composite is pulled down to 0.425. That is the correct outcome here - a
    real negative is allowed to argue, where an outage is not.
    """
    assessment = score(
        "<m@example.com>",
        [
            completed(DetectionLayer.L1, 0.85),
            completed(DetectionLayer.L2, 0.0),
            incomplete(DetectionLayer.L3),
            incomplete(DetectionLayer.L4),
        ],
        weights=WEIGHTS,
    )

    assert assessment.score == pytest.approx(0.425)


@pytest.mark.asyncio
async def test_a_message_with_urls_but_no_seams_is_still_uninformative():
    """The no-URL carve-out must not leak into the case it sits next to.

    One URL and nothing configured to look at it: three signals abstain and the
    layer is uninformative, exactly as before this distinction was drawn.
    """
    with pytest.raises(Layer2Uninformative):
        await analyze_async(email(DIRECT))


@pytest.mark.asyncio
async def test_a_message_with_an_image_but_no_decoder_is_still_uninformative():
    """An undecoded image is a surface that went uninspected, URLs or not."""
    with pytest.raises(Layer2Uninformative):
        await analyze_async(
            email_with_image(), follower=FakeFollower(), renderer=FakeRenderer(), decoder=None
        )


def test_analyze_itself_never_raises_uninformative():
    """The distinction belongs to the adapter that speaks to the orchestrator.

    `analyze` always returns all five signals, in section 4's table order.
    """
    signals = analyze(email(DIRECT))

    assert [s.name for s in signals] == [
        "base64_email_param",
        "redirect_depth",
        "captcha_gate",
        "qr_url",
        "domain_mismatch_brand",
    ]
    assert all(s.layer is DetectionLayer.L2 for s in signals)


def test_provider_metadata_stays_metadata():
    """Nothing about the follower, renderer or decoder leaks into a name."""
    signals = analyze(
        email(LINK),
        follower=FakeFollower({LINK: settled(LINK, TERMINAL)}),
        renderer=FakeRenderer({TERMINAL: ChallengeKind.TURNSTILE}),
    )

    for signal in signals:
        assert "Fake" not in signal.name
        assert "Fake" not in signal.evidence
    gate = by_name(signals)["captcha_gate"]
    assert gate.metadata["matches"][0]["terminal_url"] == TERMINAL


# --------------------------------------------------------------------------
# M-O. The orchestrator and the composite
# --------------------------------------------------------------------------


WEIGHTS = ScoringWeights(
    layer_weights={
        DetectionLayer.L1: 0.30,
        DetectionLayer.L2: 0.30,
        DetectionLayer.L3: 0.20,
        DetectionLayer.L4: 0.20,
    },
    deliver_threshold=0.35,
    block_threshold=0.65,
)


def completed(layer: DetectionLayer, value: float) -> LayerResult:
    from core.models import DetectionSignal

    return LayerResult(
        layer=layer,
        completed=True,
        signals=[
            DetectionSignal(
                layer=layer,
                name="probe",
                score=value,
                severity=RiskLevel.HIGH if value >= 0.65 else RiskLevel.LOW,
                evidence="probe",
            )
        ],
    )


def incomplete(layer: DetectionLayer) -> LayerResult:
    return LayerResult(layer=layer, completed=False, signals=[], error="no information")


def test_l2_is_wired_into_the_default_layer_mapping():
    assert DEFAULT_LAYERS[DetectionLayer.L2] is l2_urls.analyze_async
    assert DEFAULT_LAYER_TIMEOUTS[DetectionLayer.L2] == 15.0


@pytest.mark.asyncio
async def test_an_uninformative_l2_reports_incomplete_to_the_orchestrator():
    async def l2(email):
        return await analyze_async(email)

    results = {r.layer: r for r in await run_layers(email(DIRECT), layers={DetectionLayer.L2: l2})}

    assert results[DetectionLayer.L2].completed is False
    assert results[DetectionLayer.L2].signals == []
    assert "Layer2Uninformative" in results[DetectionLayer.L2].error


def test_an_uninformative_l2_does_not_consume_its_030_weight():
    """The whole point: the 0.30 is redistributed, not spent on a false clean.

    L1 alone at 0.85. Scored against a completed-but-clean L2 the composite is
    0.425 - MEDIUM. With L2 incomplete its weight goes to the layers that ran,
    and the DMARC failure reads 0.85: HIGH.
    """
    with_l2_clean = score(
        "<m@example.com>",
        [
            completed(DetectionLayer.L1, 0.85),
            completed(DetectionLayer.L2, 0.0),
            incomplete(DetectionLayer.L3),
            incomplete(DetectionLayer.L4),
        ],
        weights=WEIGHTS,
    )
    with_l2_absent = score(
        "<m@example.com>",
        [
            completed(DetectionLayer.L1, 0.85),
            incomplete(DetectionLayer.L2),
            incomplete(DetectionLayer.L3),
            incomplete(DetectionLayer.L4),
        ],
        weights=WEIGHTS,
    )

    assert with_l2_clean.score == pytest.approx(0.425)
    assert with_l2_clean.level is RiskLevel.MEDIUM
    assert with_l2_absent.score == pytest.approx(0.85)
    assert with_l2_absent.level is RiskLevel.HIGH

    contributions = layer_contributions(
        [completed(DetectionLayer.L1, 0.85), incomplete(DetectionLayer.L2)], WEIGHTS
    )
    assert DetectionLayer.L2 not in contributions


@pytest.mark.asyncio
async def test_an_l2_finding_changes_the_final_orchestration_result():
    """A captcha gate on the terminal page moves the composite, at 0.30."""
    message = email(LINK)

    async def l2(_email):
        return await analyze_async(
            _email,
            follower=FakeFollower({LINK: settled(LINK, TERMINAL)}),
            renderer=FakeRenderer({TERMINAL: ChallengeKind.TURNSTILE}),
            decoder=FakeDecoder(),
        )

    results = await run_layers(message, layers={DetectionLayer.L2: l2})
    by_layer = {r.layer: r for r in results}

    assert by_layer[DetectionLayer.L2].completed is True
    gate = by_name(by_layer[DetectionLayer.L2].signals)["captcha_gate"]
    assert gate.score == CAPTCHA_GATE_SCORE

    scored_with = score(
        "<m@example.com>",
        [incomplete(DetectionLayer.L1), by_layer[DetectionLayer.L2]],
        weights=WEIGHTS,
    )
    scored_without = score(
        "<m@example.com>",
        [incomplete(DetectionLayer.L1), incomplete(DetectionLayer.L2), completed(DetectionLayer.L3, 0.0)],
        weights=WEIGHTS,
    )

    assert scored_with.score == pytest.approx(CAPTCHA_GATE_SCORE)
    assert scored_with.level is RiskLevel.HIGH
    assert scored_without.score == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_layers_one_three_and_four_are_unchanged_by_the_l2_wiring():
    """L2 was added beside them; nothing about them moved."""
    message = email(DIRECT)
    results = {r.layer: r for r in await run_layers(message)}

    assert results[DetectionLayer.L1].completed is True
    assert [s.name for s in results[DetectionLayer.L1].signals] == [
        "spf_fail",
        "dkim_fail",
        "dmarc_fail",
        "replyto_mismatch",
        "domain_age_lt_7d",
        "display_name_impersonation",
    ]

    # No models, no feeds and no ASN resolver on this machine: L3 and L4 report
    # incomplete for their own reasons, exactly as they did before.
    for layer in (DetectionLayer.L3, DetectionLayer.L4):
        assert results[layer].completed is False
        assert results[layer].signals == []
        assert "Layer2Uninformative" not in (results[layer].error or "")
