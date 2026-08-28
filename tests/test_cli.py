"""Unit tests for the `python -m cli analyze` entry point.

Offline and deterministic. Layer 1 is now real, but the sample message sends
from `.invalid` - which has no registrable domain - so its WHOIS signal
abstains before any lookup is attempted and nothing here touches the network,
Gmail or OAuth. Scope is the CLI's own job - input validation, pipeline
invocation, output and exit status - not the pipeline's behaviour, which its
own tests already cover.
"""

from __future__ import annotations

import pytest

from cli.__main__ import main

VALID_EML = b"""From: "Account Team" <alerts@example.invalid>
To: john@example.com
Subject: Unusual sign-in activity
Message-ID: <cli-test-0001@example.invalid>
Content-Type: text/plain; charset="utf-8"

Verify your account: https://example.invalid/verify
"""


@pytest.fixture
def eml(tmp_path):
    path = tmp_path / "message.eml"
    path.write_bytes(VALID_EML)
    return path


# --------------------------------------------------------------------------
# 1. The happy path
# --------------------------------------------------------------------------


def test_analyze_valid_eml_exits_zero(eml):
    assert main(["analyze", str(eml)]) == 0


def test_analyze_prints_the_risk_verdict(eml, capsys):
    main(["analyze", str(eml)])
    out = capsys.readouterr().out

    assert "Verdict" in out
    assert "score" in out
    assert any(level in out for level in ("LOW", "MEDIUM", "HIGH"))


def test_analyze_prints_the_message_and_its_signals(eml, capsys):
    main(["analyze", str(eml)])
    out = capsys.readouterr().out

    assert "<cli-test-0001@example.invalid>" in out
    assert "Unusual sign-in activity" in out
    # Layer 1's signals are reported under their qualified names.
    assert "L1/replyto_mismatch" in out
    assert "L1/display_name_impersonation" in out


def test_analyze_reports_every_layer_status(eml, capsys):
    main(["analyze", str(eml)])
    out = capsys.readouterr().out

    for layer in ("L1", "L2", "L3", "L4"):
        assert layer in out


# --------------------------------------------------------------------------
# 2. Invalid input
# --------------------------------------------------------------------------


def test_missing_file_exits_non_zero(tmp_path, capsys):
    exit_code = main(["analyze", str(tmp_path / "absent.eml")])

    assert exit_code != 0
    assert "no such file" in capsys.readouterr().err


def test_directory_argument_exits_non_zero(tmp_path, capsys):
    exit_code = main(["analyze", str(tmp_path)])

    assert exit_code != 0
    assert "not a file" in capsys.readouterr().err


def test_missing_subcommand_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main([])

    assert exit_info.value.code != 0


# --------------------------------------------------------------------------
# 3. Malformed input
# --------------------------------------------------------------------------


def test_malformed_email_is_analyzed_not_crashed(tmp_path, capsys):
    """The parser is deliberately tolerant; garbage in still yields a verdict."""
    path = tmp_path / "garbage.eml"
    path.write_bytes(b"\xff\xfe not remotely an email \x00\x01")

    exit_code = main(["analyze", str(path)])

    assert exit_code == 0
    assert "Verdict" in capsys.readouterr().out


def test_empty_file_is_handled_cleanly(tmp_path, capsys):
    path = tmp_path / "empty.eml"
    path.write_bytes(b"")

    exit_code = main(["analyze", str(path)])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "no sender" in out


# --------------------------------------------------------------------------
# 4. Pipeline failure
# --------------------------------------------------------------------------


def test_unscoreable_pipeline_exits_non_zero(eml, monkeypatch, capsys):
    """A scoring failure is reported, not swallowed into a clean verdict."""
    from scoring.composite import UnscoreableError

    def explode(*_args, **_kwargs):
        raise UnscoreableError("no layer completed")

    monkeypatch.setattr("cli.__main__.score", explode)

    assert main(["analyze", str(eml)]) != 0
    assert "no layer completed" in capsys.readouterr().err


def test_weights_are_resolved_independently_of_the_working_directory(eml, tmp_path, monkeypatch):
    """The documented command must work from outside the project root."""
    monkeypatch.chdir(tmp_path)

    assert main(["analyze", str(eml)]) == 0
