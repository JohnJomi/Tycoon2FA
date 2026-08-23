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
