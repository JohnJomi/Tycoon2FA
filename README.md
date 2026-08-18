# Tycoon2FA

A research prototype for detecting Tycoon 2FA / AiTM (adversary-in-the-middle)
phishing email. It reads raw RFC-822 messages from a Gmail mailbox in read-only
mode and runs them through four independent detection layers, each of which
emits signals with human-readable evidence rather than a bare number. The
layers are combined into a single composite risk score with a deliver / warn /
block verdict. The design goal is defensibility: every claim in this README is
meant to be backed by a measurement produced by `eval/`.

**This is a research prototype.** It is not production software, it is not
publicly deployed, it is read-only, and it takes no remediation action.

## Detection layers

| Layer | Focus |
|---|---|
| L1 | Header and domain intelligence - authentication results, Reply-To mismatch, domain age, display-name impersonation |
| L2 | URL and redirect chain - redirect depth, base64 email parameters, CAPTCHA gating, QR-embedded URLs |
| L3 | NLP / ML body analysis - Unicode obfuscation, urgency classification, perplexity and burstiness |
| L4 | Threat intelligence correlation - URL and domain IOC lookups, hosting reputation |

None of these layers is implemented yet.

## Status

**Phase 0 - repository bootstrap.** This repository currently contains the
directory and module scaffold only. Every Python module is a documented
placeholder; no ingest, detection, scoring, training, API or frontend
functionality exists. There are no results to report and no detection
capability to claim.

## Documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md) - data contracts, layer specifications,
  orchestration, scoring and API surface. Source of truth.
- [`ROADMAP.md`](ROADMAP.md) - phased build plan with acceptance criteria per
  task.

## Requirements

Python 3.11+. Baseline dependencies are in `requirements.txt`; layer-specific
and ML dependencies are added in the phase that introduces them.
