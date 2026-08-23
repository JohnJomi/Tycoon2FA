"""Command-line entry point for the detection pipeline.

Thin by design: argument parsing, input validation, one pipeline call and
formatted output. All detection, orchestration and scoring live in
`ingest/`, `core/` and `scoring/`; nothing is reimplemented here.
"""
