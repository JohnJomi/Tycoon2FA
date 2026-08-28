"""`python -m cli` - the Phase 1 exit criterion.

    python -m cli analyze sample.eml

Reads an RFC-822 file, runs it through the existing pipeline and prints the
signal table plus the composite verdict:

    parse_email -> run_layers -> score -> printed report

Layer 1 behind `run_layers` is real; Layers 2-4 are still stubs, so the
composite score is partial and the report says so. Replacing the remaining
stubs requires no change here.

Exit status: 0 on a completed analysis, 1 on unreadable input or a pipeline
failure, 2 on a usage error (argparse's own convention).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Sequence

from core.models import DetectionLayer, LayerResult, ParsedEmail, RiskAssessment
from core.orchestrator import run_layers
from ingest.parser import EmailParseError, parse_email
from scoring.composite import UnscoreableError, load_weights, score

__all__ = ["main"]

EXIT_OK = 0
EXIT_FAILURE = 1

# scoring/composite.py resolves its default weights path against the current
# working directory. The CLI can be invoked from anywhere, so it resolves the
# project default itself; WEIGHTS_CONFIG still wins when it is set.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_WEIGHTS = _PROJECT_ROOT / "config" / "weights.yaml"


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


def _read(path: Path) -> bytes:
    """Read the .eml file, or raise ValueError with a usable message."""
    if not path.exists():
        raise ValueError(f"no such file: {path}")
    if not path.is_file():
        raise ValueError(f"not a file: {path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValueError(f"could not read {path}: {exc}") from exc


def _analyze(raw: bytes) -> tuple[ParsedEmail, list[LayerResult], RiskAssessment]:
    """Run the existing pipeline over raw message bytes."""
    email = parse_email(raw)
    results = asyncio.run(run_layers(email))
    weights = load_weights(os.environ.get("WEIGHTS_CONFIG") or _DEFAULT_WEIGHTS)
    return email, results, score(email.message_id, results, weights=weights)


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def _rule(title: str) -> str:
    return f"\n{title}\n{'-' * len(title)}"


def _format_report(
    email: ParsedEmail, results: list[LayerResult], assessment: RiskAssessment
) -> str:
    lines: list[str] = []

    lines.append(_rule("Message"))
    lines.append(f"  message-id : {email.message_id}")
    lines.append(f"  from       : {email.from_display or '(no display name)'} "
                 f"<{email.from_addr or 'no sender'}>")
    lines.append(f"  subject    : {email.subject or '(no subject)'}")
    lines.append(f"  urls       : {len(email.urls)}")
    lines.append(f"  attachments: {len(email.attachments)}")

    lines.append(_rule("Layers"))
    for result in results:
        if result.completed:
            status = f"completed, {len(result.signals)} signal(s)"
        else:
            # An incomplete layer is an absence of evidence, never an
            # all-clear - the report has to keep those distinguishable.
            status = f"NOT COMPLETED - {result.error}"
        lines.append(f"  {result.layer.name}  {status}  [{result.duration_ms} ms]")

    lines.append(_rule("Signals"))
    if assessment.signals:
        for layer in DetectionLayer:
            for signal in assessment.signals_for(layer):
                marker = "abstained" if signal.error else f"{signal.score:.2f}"
                lines.append(
                    f"  {signal.qualified_name:<28} {marker:>9}  "
                    f"{signal.severity.value}"
                )
                lines.append(f"      {signal.error or signal.evidence}")
    else:
        lines.append("  (no signals fired)")

    lines.append(_rule("Verdict"))
    lines.append(f"  score : {assessment.score:.3f}")
    lines.append(f"  level : {assessment.level.value.upper()}")
    completed = ", ".join(layer.name for layer in assessment.layers_completed) or "none"
    lines.append(f"  scored over: {completed}")

    lines.append(
        "\nNOTE: Layer 1 is implemented; Layers 2-4 are still stubs, so this "
        "score reflects header and domain signals only."
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cli", description="Tycoon2FA phishing detection pipeline."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    analyze = subcommands.add_parser("analyze", help="analyze one .eml file")
    analyze.add_argument("path", type=Path, help="path to an RFC-822 .eml file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        raw = _read(args.path)
        email, results, assessment = _analyze(raw)
    except (ValueError, EmailParseError, UnscoreableError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    except Exception as exc:  # noqa: BLE001 - the CLI reports, it does not trace
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_FAILURE

    print(_format_report(email, results, assessment))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
