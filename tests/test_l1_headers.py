"""Unit tests for the Authentication-Results analyzer in layers/l1_headers.py.

Deterministic and offline. Fixtures are synthetic RFC-822 bytes run through
the real `ingest.parser`, so the tests exercise the same header extraction the
pipeline uses rather than a hand-built ParsedEmail that could drift from it.

No test performs DNS, network or cryptographic work, because the module under
test does none: it reads verdicts the receiving MTA already recorded.
"""

from __future__ import annotations

import pytest

from core.models import DetectionLayer, ParsedEmail, RiskLevel
from ingest.parser import parse_email
from layers.l1_headers import (
    AUTH_METHODS,
    analyze_authentication_results,
    parse_authentication_results,
)


def email_with(*auth_results: str, extra_headers: str = "") -> ParsedEmail:
    """A parsed message carrying the given Authentication-Results header(s)."""
    headers = "".join(f"Authentication-Results: {value}\n" for value in auth_results)
    raw = (
        b"From: Sender <sender@example.invalid>\n"
        b"To: john@example.com\n"
        b"Subject: Test message\n"
        b"Message-ID: <l1-test@example.invalid>\n"
        + headers.encode()
        + extra_headers.encode()
        + b"Content-Type: text/plain; charset=\"utf-8\"\n"
        b"\n"
        b"Body text.\n"
    )
    return parse_email(raw)


def signals_by_method(email: ParsedEmail) -> dict[str, object]:
    return {
        signal.metadata["method"]: signal
        for signal in analyze_authentication_results(email)
    }


GOOGLE_HEADER = (
    "mx.google.com; "
    "dkim=pass header.i=@example.invalid header.s=sel header.b=AbCd; "
    "spf=pass (google.com: domain of sender@example.invalid designates "
    "209.85.220.41 as permitted sender) smtp.mailfrom=sender@example.invalid; "
    "dmarc=pass (p=REJECT sp=REJECT dis=NONE) header.from=example.invalid"
)


# --------------------------------------------------------------------------
# 0. ARC-Authentication-Results is deliberately not read
# --------------------------------------------------------------------------
#
# `ingest.pipeline.AUTH_HEADERS` surfaces `ARC-Authentication-Results`, but
# ARCHITECTURE.md section 4 specifies the three verdicts as read from
# `Authentication-Results`, and this layer honours that. ARC records what some
# *earlier* hop asserted, not what the receiving MTA concluded, and folding it
# into `_worst()` would let an upstream forwarder's stale or forged verdict
# decide a message the receiver itself authenticated. These tests pin the
# existing behaviour so the omission stays a decision rather than drift.


def test_arc_authentication_results_alone_does_not_produce_verdicts():
    """An ARC-only message abstains; it does not inherit the upstream claim."""
    email = email_with(
        extra_headers=(
            "ARC-Authentication-Results: i=1; mx.google.com; "
            "spf=pass; dkim=pass; dmarc=pass\n"
        )
    )

    assert parse_authentication_results(email) == {}
    for signal in analyze_authentication_results(email):
        assert signal.score == 0.0
        assert signal.error is not None


def test_an_arc_header_does_not_override_the_receivers_own_verdict():
    """A failing ARC assertion cannot contaminate a receiver-recorded pass."""
    email = email_with(
        GOOGLE_HEADER,
        extra_headers=(
            "ARC-Authentication-Results: i=1; upstream.example; "
            "spf=fail; dkim=fail; dmarc=fail\n"
        ),
    )

    signals = signals_by_method(email)

    for method in AUTH_METHODS:
        assert signals[method].metadata["result"] == "pass"
        assert signals[method].score == 0.0
        assert signals[method].metadata["fired"] is False


# --------------------------------------------------------------------------
# 1. Contract shape
# --------------------------------------------------------------------------


def test_emits_one_signal_per_method_in_order():
    signals = analyze_authentication_results(email_with(GOOGLE_HEADER))

    assert [s.metadata["method"] for s in signals] == list(AUTH_METHODS)


def test_signals_belong_to_layer_one_and_are_named_per_method():
    signals = analyze_authentication_results(email_with(GOOGLE_HEADER))

    assert all(s.layer is DetectionLayer.L1 for s in signals)
    assert [s.qualified_name for s in signals] == [
        "L1/spf_fail",
        "L1/dkim_fail",
        "L1/dmarc_fail",
    ]


def test_every_signal_carries_non_empty_evidence():
    signals = analyze_authentication_results(email_with(GOOGLE_HEADER))

    assert all(s.evidence.strip() for s in signals)


# --------------------------------------------------------------------------
# 2. Pass and fail, per method
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", AUTH_METHODS)
def test_pass_does_not_fire_and_is_a_genuine_negative(method):
    """A pass scores 0.0 with no error: it ran, it looked, it found nothing."""
    signal = signals_by_method(email_with(f"mx.google.com; {method}=pass"))[method]

    assert signal.score == 0.0
    assert signal.error is None
    assert signal.severity is RiskLevel.LOW


@pytest.mark.parametrize("method", AUTH_METHODS)
def test_fail_fires_with_a_positive_score(method):
    signal = signals_by_method(email_with(f"mx.google.com; {method}=fail"))[method]

    assert signal.score > 0.0
    assert signal.error is None


def test_spf_fail_evidence_names_the_method_and_result():
    signal = signals_by_method(email_with("mx.google.com; spf=fail"))["spf"]

    assert "SPF" in signal.evidence
    assert "spf=fail" in signal.evidence
    assert "mx.google.com" in signal.evidence


def test_dkim_fail_evidence_names_the_method_and_result():
    signal = signals_by_method(email_with("mx.google.com; dkim=fail"))["dkim"]

    assert "DKIM" in signal.evidence
    assert "dkim=fail" in signal.evidence


def test_dmarc_fail_evidence_names_the_method_and_result():
    signal = signals_by_method(email_with("mx.google.com; dmarc=fail"))["dmarc"]

    assert "DMARC" in signal.evidence
    assert "dmarc=fail" in signal.evidence


def test_dmarc_failure_is_the_most_severe_of_the_three():
    """DMARC expresses the domain owner's own published policy."""
    failing = email_with("mx.google.com; spf=fail; dkim=fail; dmarc=fail")
    signals = signals_by_method(failing)

    assert signals["dmarc"].severity is RiskLevel.HIGH
    assert signals["spf"].severity is RiskLevel.MEDIUM
    assert signals["dkim"].severity is RiskLevel.MEDIUM
    assert signals["dmarc"].score > signals["spf"].score


def test_softfail_fires_more_weakly_than_a_hard_fail():
    soft = signals_by_method(email_with("mx.google.com; spf=softfail"))["spf"]
    hard = signals_by_method(email_with("mx.google.com; spf=fail"))["spf"]

    assert 0.0 < soft.score < hard.score
    assert soft.error is None


def test_all_three_pass_produces_no_fired_signal():
    signals = analyze_authentication_results(email_with(GOOGLE_HEADER))

    assert all(s.score == 0.0 and s.error is None for s in signals)


# --------------------------------------------------------------------------
# 3. Absence is not failure
# --------------------------------------------------------------------------


def test_missing_header_entirely_abstains_for_every_method():
    signals = analyze_authentication_results(email_with())

    assert all(s.error is not None for s in signals)
    assert all(s.score == 0.0 for s in signals)


def test_missing_individual_method_abstains_only_for_that_method():
    signals = signals_by_method(email_with("mx.google.com; spf=pass; dkim=pass"))

    assert signals["spf"].error is None
    assert signals["dkim"].error is None
    assert signals["dmarc"].error is not None
    assert "DMARC" in signals["dmarc"].evidence


def test_absent_method_is_not_reported_as_a_failure():
    """The whole point: no verdict must never become a fabricated finding."""
    signal = signals_by_method(email_with("mx.google.com; spf=pass"))["dmarc"]

    assert signal.score == 0.0
    assert "failed" not in signal.evidence.lower()


@pytest.mark.parametrize("result", ["temperror", "permerror"])
def test_transient_and_permanent_errors_abstain(result):
    signal = signals_by_method(email_with(f"mx.google.com; spf={result}"))["spf"]

    assert signal.error is not None
    assert signal.score == 0.0


def test_none_result_is_not_a_failure():
    """spf=none means the domain publishes no record, not that it failed."""
    signal = signals_by_method(email_with("mx.google.com; spf=none"))["spf"]

    assert signal.score == 0.0
    assert signal.error is None
    assert "not an authentication failure" in signal.evidence


def test_neutral_result_is_not_a_failure():
    signal = signals_by_method(email_with("mx.google.com; spf=neutral"))["spf"]

    assert signal.score == 0.0
    assert signal.error is None


def test_unrecognized_result_abstains_rather_than_guessing():
    signal = signals_by_method(email_with("mx.google.com; spf=wibble"))["spf"]

    assert signal.error is not None
    assert signal.score == 0.0


# --------------------------------------------------------------------------
# 4. Header syntax variations
# --------------------------------------------------------------------------


def test_multiple_authentication_results_headers_are_all_read():
    email = email_with(
        "mx.google.com; spf=pass",
        "mx.google.com; dkim=fail",
        "mx.google.com; dmarc=pass",
    )
    signals = signals_by_method(email)

    assert signals["spf"].score == 0.0
    assert signals["dkim"].score > 0.0
    assert signals["dmarc"].score == 0.0


def test_multiple_results_within_one_header_are_all_read():
    signals = signals_by_method(
        email_with("mx.google.com; spf=pass; dkim=fail; dmarc=fail")
    )

    assert signals["spf"].score == 0.0
    assert signals["dkim"].score > 0.0
    assert signals["dmarc"].score > 0.0


def test_mixed_case_method_names_are_recognized():
    signals = signals_by_method(email_with("mx.google.com; SPF=fail; DKIM=fail"))

    assert signals["spf"].score > 0.0
    assert signals["dkim"].score > 0.0


def test_mixed_case_result_values_are_recognized():
    signals = signals_by_method(email_with("mx.google.com; spf=FAIL; dkim=Pass"))

    assert signals["spf"].score > 0.0
    assert signals["dkim"].score == 0.0
    assert signals["dkim"].error is None


def test_extra_propspec_parameters_are_ignored():
    signals = signals_by_method(
        email_with(
            "mx.google.com; dkim=fail header.i=@example.invalid header.s=sel "
            "header.b=AbCdEf; spf=pass smtp.mailfrom=sender@example.invalid"
        )
    )

    assert signals["dkim"].score > 0.0
    assert signals["spf"].score == 0.0


def test_comments_containing_semicolons_do_not_break_chunking():
    """Google writes `spf=pass (google.com: domain of ... )` - with colons and
    parens around text that would otherwise split the header."""
    signals = signals_by_method(email_with(GOOGLE_HEADER))

    assert all(s.error is None for s in signals.values())
    assert signals["dmarc"].metadata["result"] == "pass"


def test_method_version_tokens_are_accepted():
    signal = signals_by_method(email_with("mx.google.com; spf/1=fail"))["spf"]

    assert signal.score > 0.0


def test_authserv_id_is_recorded_in_metadata():
    signal = signals_by_method(email_with("mx.google.com; spf=fail"))["spf"]

    assert signal.metadata["authserv_id"] == "mx.google.com"


def test_authserv_id_with_a_version_token_is_handled():
    signal = signals_by_method(email_with("mx.google.com 1; spf=fail"))["spf"]

    assert signal.metadata["authserv_id"] == "mx.google.com"
    assert signal.score > 0.0


def test_header_without_an_authserv_id_still_yields_the_verdict():
    signal = signals_by_method(email_with("spf=fail"))["spf"]

    assert signal.score > 0.0


# --------------------------------------------------------------------------
# 5. Conflicting verdicts
# --------------------------------------------------------------------------


def test_the_most_severe_verdict_wins_when_results_conflict():
    """A forged header claiming a pass must not mask a real failure."""
    email = email_with("attacker.invalid; dkim=pass", "mx.google.com; dkim=fail")
    signal = signals_by_method(email)["dkim"]

    assert signal.score > 0.0
    assert signal.metadata["result"] == "fail"


def test_conflicting_verdicts_are_disclosed_in_the_evidence():
    email = email_with("attacker.invalid; dkim=pass", "mx.google.com; dkim=fail")
    signal = signals_by_method(email)["dkim"]

    assert "several results" in signal.evidence
    assert sorted(signal.metadata["all_results"]) == ["fail", "pass"]


def test_multiple_dkim_signatures_in_one_header_keep_every_result():
    signal = signals_by_method(
        email_with("mx.google.com; dkim=pass header.i=@a; dkim=fail header.i=@b")
    )["dkim"]

    assert signal.metadata["all_results"] == ["pass", "fail"]
    assert signal.score > 0.0


# --------------------------------------------------------------------------
# 6. Malformed input must not crash the analysis
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "header",
    [
        "",
        ";;;;",
        "mx.google.com;",
        "mx.google.com; = ; spf",
        "mx.google.com; spf=",
        "(unterminated comment",
        "mx.google.com; spf=pass (unterminated",
        "!!! not a header at all !!!",
        "mx.google.com; 123=456",
    ],
)
def test_malformed_headers_do_not_raise(header):
    signals = analyze_authentication_results(email_with(header))

    assert len(signals) == len(AUTH_METHODS)


def test_a_malformed_fragment_does_not_discard_a_valid_one():
    """One bad chunk costs that chunk, not the whole header."""
    signals = signals_by_method(email_with("mx.google.com; @@@garbage@@@; spf=fail"))

    assert signals["spf"].score > 0.0


def test_irrelevant_methods_are_ignored():
    """iprev, auth and others are valid RFC 8601 methods we do not report on."""
    signals = signals_by_method(
        email_with("mx.google.com; iprev=pass; auth=pass; spf=fail")
    )

    assert signals["spf"].score > 0.0
    assert set(signals) == set(AUTH_METHODS)


def test_parse_returns_only_methods_actually_present():
    verdicts = parse_authentication_results(email_with("mx.google.com; spf=pass"))

    assert set(verdicts) == {"spf"}
    assert verdicts["spf"].result == "pass"


# --------------------------------------------------------------------------
# 7. Real-message shapes
# --------------------------------------------------------------------------


def test_folded_header_is_read_correctly():
    """Real Authentication-Results headers are folded across several lines."""
    raw = (
        b"From: Sender <sender@example.invalid>\n"
        b"Message-ID: <folded@example.invalid>\n"
        b"Authentication-Results: mx.google.com;\n"
        b"       dkim=fail header.i=@example.invalid;\n"
        b"       spf=softfail smtp.mailfrom=sender@example.invalid;\n"
        b"       dmarc=fail header.from=example.invalid\n"
        b"Content-Type: text/plain\n"
        b"\n"
        b"Body.\n"
    )
    signals = signals_by_method(parse_email(raw))

    assert signals["dkim"].score > 0.0
    assert signals["spf"].metadata["result"] == "softfail"
    assert signals["dmarc"].severity is RiskLevel.HIGH


def test_the_repository_sample_message_is_analyzed_without_error():
    """sample.eml carries no Authentication-Results, so all three abstain."""
    from pathlib import Path

    raw = (Path(__file__).resolve().parent.parent / "sample.eml").read_bytes()
    signals = analyze_authentication_results(parse_email(raw))

    assert len(signals) == 3
    assert all(s.error is not None for s in signals)


# --------------------------------------------------------------------------
# Real-message Authentication-Results fixtures
#
# The five REAL_* headers below were captured verbatim from real messages in
# the mailbox the ingestion layer was validated against (only the local-part
# of one bounce address is shortened). They are the roadmap's "correctly
# reports pass/fail on 5 real messages" acceptance check, and they exist
# because hand-written fixtures drift towards the shapes the parser already
# handles - these carry real propspec ordering, real base64 with `/` and `+`
# in it, multi-signature DKIM, and comments full of semicolons.
#
# Gmail rejects most hard authentication failures outright, so the failing
# fixtures that follow are real-world *shaped* - taken from the header formats
# Outlook/Proofpoint/Zoho emit - rather than captured from this mailbox.
# --------------------------------------------------------------------------

REAL_GITHUB = (
    "mx.google.com; dkim=pass header.i=@github.com header.s=pf2023 "
    'header.b="B4/Nx918"; spf=pass (google.com: domain of '
    "notifications@github.com designates 192.30.252.143 as permitted sender) "
    "smtp.mailfrom=notifications@github.com; dmarc=pass (p=QUARANTINE "
    "sp=REJECT dis=NONE) header.from=github.com"
)

REAL_AMAZON_SES = (
    "mx.google.com; dkim=pass header.i=@no-reply.hack2skill.com "
    'header.s=qkdrj4pyxom2jekriifuhnv6ej45rhms header.b="CjN/tcuK"; '
    "dkim=pass header.i=@amazonses.com header.s=33l4t57s6hxgsng3hsnbfahbdkoubkgb "
    "header.b=oNLIz2He; spf=pass (google.com: domain of "
    "010e01a04152a146-000000@ap-southeast-1.amazonses.com designates "
    "23.251.232.54 as permitted sender) "
    "smtp.mailfrom=010e01a04152a146-000000@ap-southeast-1.amazonses.com"
)

REAL_NOBROKER = (
    "mx.google.com; dkim=pass header.i=@homeservices.nobroker.in header.s=kmnb1 "
    "header.b=PLTjhYD3; spf=pass (google.com: domain of "
    "info@homeservices.nobroker.in designates 103.162.246.221 as permitted "
    "sender) smtp.mailfrom=info@homeservices.nobroker.in; dmarc=pass "
    "(p=REJECT sp=REJECT dis=NONE) header.from=homeservices.nobroker.in"
)

REAL_EMARSYS = (
    "mx.google.com; dkim=pass header.i=@hello.bitdefender.com header.s=key2 "
    "header.b=Ch67h+5+; dkim=pass header.i=@emarsys.net "
    "header.s=emarsys-2048b header.b=FDQIehUC; spf=pass (google.com: domain "
    "of suite25@xpressus.emsmtp.us designates 83.68.134.99 as permitted "
    "sender) smtp.mailfrom=suite25@xpressus.emsmtp.us; dmarc=pass "
    "(p=REJECT sp=REJECT dis=NONE) header.from=hello.bitdefender.com"
)

REAL_SES_MANOCHA = (
    "mx.google.com; dkim=pass header.i=@manochaacademy.com "
    "header.s=grgopexakp5gxfpj5pymd5ju4k7akvjq header.b=NqHoeyLp; dkim=pass "
    "header.i=@amazonses.com header.s=rlntogby6xsxlfnvyxwnvvhttakdsqto "
    "header.b=lYvx8iow; spf=pass (google.com: domain of "
    "0109019fbbdc2d96-000000@ses.manochaacademy.com designates 76.223.180.123 "
    "as permitted sender) "
    "smtp.mailfrom=0109019fbbdc2d96-000000@ses.manochaacademy.com"
)

# A genuinely captured hard failure: three DKIM signatures, the third of which
# failed verification. Exactly the case the "most severe wins" rule exists for.
REAL_AWS_EDUCATE = (
    "mx.google.com; dkim=pass header.i=@awseducate.com "
    "header.s=xelyu5nablrrqj5scckqloieecubrbgu header.b=uFVnJDDH; dkim=pass "
    "header.i=@amazonses.com header.s=hsbnp7p3ensaochzwyq5wwmceodymuwv "
    "header.b=S83lLXjM; dkim=fail header.i=@awseducate.com "
    "header.s=alteducatedkimkey header.b=WrqNguPP; spf=pass (google.com: "
    "domain of 010101a012f4697c-000000@us-west-2.amazonses.com designates "
    "54.240.27.199 as permitted sender) "
    "smtp.mailfrom=010101a012f4697c-000000@us-west-2.amazonses.com"
)

REAL_MESSAGES = [
    ("github", REAL_GITHUB, {"spf": "pass", "dkim": "pass", "dmarc": "pass"}),
    ("amazon-ses", REAL_AMAZON_SES, {"spf": "pass", "dkim": "pass"}),
    ("nobroker", REAL_NOBROKER, {"spf": "pass", "dkim": "pass", "dmarc": "pass"}),
    ("emarsys", REAL_EMARSYS, {"spf": "pass", "dkim": "pass", "dmarc": "pass"}),
    ("ses-manocha", REAL_SES_MANOCHA, {"spf": "pass", "dkim": "pass"}),
    ("aws-educate", REAL_AWS_EDUCATE, {"spf": "pass", "dkim": "fail"}),
]


@pytest.mark.parametrize(
    "name,header,expected", REAL_MESSAGES, ids=[m[0] for m in REAL_MESSAGES]
)
def test_verdicts_on_five_real_messages(name, header, expected):
    """The roadmap's acceptance check: pass/fail read correctly on 5 real messages."""
    verdicts = parse_authentication_results(email_with(header))

    for method, result in expected.items():
        assert verdicts[method].result == result, f"{name}: {method}"

    for signal in analyze_authentication_results(email_with(header)):
        method = signal.metadata["method"]
        if method in expected:
            assert signal.error is None, f"{name}: {method} should not abstain"
            fired = expected[method] == "fail"
            assert signal.metadata["fired"] is fired, f"{name}: {method}"
            assert (signal.score > 0.0) is fired, f"{name}: {method}"
        else:
            # A method the MTA recorded nothing for abstains rather than
            # reporting a clean result.
            assert signal.error is not None, f"{name}: {method} must abstain"


def test_a_real_failed_dkim_signature_is_not_masked_by_two_passes():
    """The real awseducate.com message signs three times; one signature fails.

    Reading the first verdict would report a clean pass. The most severe wins,
    so the failure survives - and the evidence discloses the disagreement
    rather than resolving it silently.
    """
    verdicts = parse_authentication_results(email_with(REAL_AWS_EDUCATE))

    assert verdicts["dkim"].all_results == ("pass", "pass", "fail")
    assert verdicts["dkim"].result == "fail"
    assert verdicts["dkim"].failed is True

    signal = signals_by_method(email_with(REAL_AWS_EDUCATE))["dkim"]
    assert signal.score > 0.0
    assert signal.metadata["fired"] is True
    assert "pass, pass, fail" in signal.evidence


def test_a_real_multi_signature_message_keeps_every_dkim_verdict():
    verdicts = parse_authentication_results(email_with(REAL_AMAZON_SES))

    assert verdicts["dkim"].all_results == ("pass", "pass")
    assert verdicts["dkim"].authserv_id == "mx.google.com"


def test_real_headers_are_traceable_to_the_reporting_mta():
    for _name, header, _expected in REAL_MESSAGES:
        verdicts = parse_authentication_results(email_with(header))
        assert all(v.authserv_id == "mx.google.com" for v in verdicts.values())


# Real-world *shaped* failures. Gmail rarely delivers a hard fail, so these
# reproduce the header formats other MTAs emit rather than being captured.
REAL_SHAPED_FAILURES = [
    (
        "outlook-spf-fail",
        "spf=fail (sender IP is 45.155.205.13) "
        "smtp.mailfrom=paypa1-secure.top; dkim=none (message not signed) "
        "header.d=none; dmarc=fail action=oreject header.from=paypa1-secure.top;"
        "compauth=fail reason=000",
        {"spf": "fail", "dkim": "none", "dmarc": "fail"},
    ),
    (
        "proofpoint-dkim-fail",
        "mx1.example.net; dkim=fail reason=\"signature verification failed\" "
        "header.d=chase.com header.b=Qk3Lm9; spf=softfail "
        "smtp.mailfrom=bounce@mailer.example.ru; dmarc=fail (p=reject) "
        "header.from=chase.com",
        {"spf": "softfail", "dkim": "fail", "dmarc": "fail"},
    ),
    (
        "zoho-temperror",
        "mx.zohomail.com; spf=temperror (DNS timeout) "
        "smtp.mailfrom=alerts@corp.example; dkim=permerror (bad key record) "
        "header.i=@corp.example",
        {"spf": "temperror", "dkim": "permerror"},
    ),
]


@pytest.mark.parametrize(
    "name,header,expected", REAL_SHAPED_FAILURES, ids=[m[0] for m in REAL_SHAPED_FAILURES]
)
def test_failing_real_world_shaped_headers_are_read_correctly(name, header, expected):
    verdicts = parse_authentication_results(email_with(header))

    for method, result in expected.items():
        assert verdicts[method].result == result, f"{name}: {method}"


def test_a_hard_failure_scores_and_an_inconclusive_one_abstains():
    signals = signals_by_method(email_with(REAL_SHAPED_FAILURES[1][1]))

    assert signals["dmarc"].score > 0.0
    assert signals["dmarc"].error is None
    assert signals["dmarc"].metadata["fired"] is True

    inconclusive = signals_by_method(email_with(REAL_SHAPED_FAILURES[2][1]))
    assert inconclusive["spf"].score == 0.0
    assert inconclusive["spf"].error is not None
    assert inconclusive["spf"].metadata["fired"] is False


def test_a_message_with_no_authentication_results_abstains_on_every_method():
    signals = signals_by_method(email_with())

    assert set(signals) == set(AUTH_METHODS)
    assert all(s.error is not None and s.score == 0.0 for s in signals.values())
