# ARCHITECTURE.md

Multi-layer detection pipeline for Tycoon 2FA / AiTM phishing email.

**Status:** research prototype. Not production, not public. Designed to be
defensible under technical questioning — every claim in the README must be
backed by a number produced by `eval/`.

---

## 1. System overview

```
Gmail API (readonly)
        │  users.messages.get(format='raw')  → RFC-822 bytes
        ▼
   ingest/parser.py  ──────────────► ParsedEmail (normalized)
        │
        ▼
   asyncio.gather(L1, L2, L3, L4)   ── 20s wall-clock cap
        │       │    │   │   │
        │       │    │   │   └── L4  threat intel   (network)
        │       │    │   └────── L3  NLP / ML       (CPU, model-backed)
        │       │    └────────── L2  URL + redirect (network + browser)
        │       └─────────────── L1  headers/domain (network, cached)
        ▼
   scoring/composite.py  → RiskAssessment(score, level, signals[])
        │
        ▼
   FastAPI  →  React dashboard (inbox list + signal evidence table)
```

Layers are independent. None imports another. All communicate only through
the `Signal` contract defined below. This is the single most important
constraint in the codebase — it is what makes the ablation study possible.

---

## 2. Core data contracts

Defined in `core/models.py`. Everything else depends on this module; it
depends on nothing. It holds data contracts only: no detection, no scoring,
no parsing, no I/O. Standard library only — `dataclasses`, `enum`, `datetime`.

`core/models.py` is the implementation source of truth. This section describes
what is implemented there.

### Enumerations

```python
class RiskLevel(str, Enum):     # LOW / MEDIUM / HIGH
class DetectionLayer(IntEnum):  # L1=1, L2=2, L3=3, L4=4
class URLSource(str, Enum):     # where in the email a URL was found
```

`RiskLevel` is the **risk classification produced by the future composite
scoring stage** (`scoring/composite.py`). It is also reused as the per-signal
`severity`. It is a statement about *risk*, not about what to do next — see
section 6 for the separate operational action.

`DetectionLayer` is an `IntEnum` so the layers keep their documented 1..4
numbering while callers use named members.

`URLSource` covers the extraction sites in section 3 — `ANCHOR_HREF`,
`IMG_SRC`, `FORM_ACTION`, `PLAIN_TEXT`, `HEADER` — plus `QR_CODE` for URLs
Layer 2 decodes out of images.

### DetectionSignal

```python
@dataclass(frozen=True)
class DetectionSignal:
    layer: DetectionLayer        # which layer produced this
    name: str                    # layer-local id, e.g. "redirect_depth"
    score: float                 # 0.0–1.0 normalized strength
    severity: RiskLevel          # LOW / MEDIUM / HIGH
    evidence: str                # human-readable. REQUIRED, never empty.
    metadata: dict[str, Any]     # optional structured detail
    error: str | None = None     # set when the signal could not be computed

    @property
    def qualified_name(self) -> str:   # "L2/redirect_depth"
```

**One generic model, layer-independent.** There are no `L1Signal` /
`L2Signal` subclasses and no per-signal-type class hierarchy. A layer
identifies itself through `layer` and names its own signal, so
`L1/domain_age`, `L2/base64_email_parameter` and `L4/threat_intelligence_match`
are all the same type. This is what makes the ablation study in section 5
possible: signals can be grouped, filtered and dropped by layer generically.

Frozen: a signal records what was observed and is not edited afterwards.

**`evidence` is required and validated non-empty.** The evidence strings are
the demo — a bare score demonstrates nothing, and the UI detail view in
section 9 renders them directly. Explainability is a hard requirement of the
contract, not a nicety.

**`error` is abstention, and abstention is not a clean result.** A signal with
`error` set could not be computed — WHOIS was rate-limited, the body was too
short for perplexity. That is categorically different from a signal that ran
and found nothing suspicious. A layer must never convert a failure into a
confident `score=0.0`; it sets `error`, states why, and the UI shows the
abstention rather than a false all-clear.

`score` is validated to a finite 0.0–1.0. NaN and infinity are rejected.

Note that `DetectionSignal` carries **no weight field**. Weights are scoring
configuration, not properties of an observation — see section 6.

### ParsedEmail

```python
@dataclass
class ParsedEmail:
    message_id: str
    from_addr: str
    from_display: str = ""
    to_addrs: list[str]                    # recipients
    subject: str = ""
    body_text: str = ""                    # plaintext part, or html→text fallback
    body_html: str | None = None
    reply_to: str | None = None
    urls: list[ExtractedURL]               # deduped; NOT list[str] — see below
    headers: dict[str, list[str]]          # multi-valued, case-normalized keys
    received_chain: list[str]
    attachments: list[Attachment]          # metadata only, no payload
    raw: bytes | None = None               # original RFC-822 bytes where available

    @property
    def inline_images(self) -> list[Attachment]:   # derived: cid:-referenced parts
```

The normalized email handed to the pipeline. **Not coupled to Gmail and not
coupled to any particular MIME parser** — `ingest/parser.py` populates it from
an RFC-822 message, but nothing in the contract knows how.

`urls` holds `ExtractedURL` objects, **not plain strings** — Layer 2 needs to
know where each URL came from and to write resolved redirect data back onto it.
The list is deduped, and is populated from HTML hrefs, `<img src>`,
`<form action>` and plaintext extraction (section 3).

`raw` is optional so an email can be constructed without carrying bytes (tests,
and any future non-Gmail ingest path); it holds the original RFC-822 bytes
where a source provides them.

`inline_images` is **derived, not stored**: it filters `attachments` for parts
with a `content_id`, i.e. the `cid:`-referenced images. There is one attachment
list, not two. Layer 2 needs both ordinary image attachments and inline images
when hunting for QR codes.

`Attachment` is metadata only — `filename`, `content_type`, `size_bytes`,
optional `content_id`, and a derived `is_inline`. The payload is deliberately
not carried on the contract.

### ExtractedURL

```python
@dataclass
class ExtractedURL:
    url: str                        # the URL as found in the email
    source: URLSource               # where it was found
    anchor_text: str | None = None  # visible text, where there was any
    redirect_chain: list[str]       # hops, once Layer 2 has resolved them
    final_url: str | None = None    # terminal URL, once resolved

    @property
    def redirect_depth(self) -> int:   # derived: len(redirect_chain)
```

**Observations only.** This model fetches nothing, follows nothing and
analyses nothing — construction is pure and offline. `redirect_chain` and
`final_url` stay empty/`None` until `layers/l2_urls.py` resolves them and
writes them back; `redirect_depth` is derived from the chain, not measured
here. `anchor_text` is retained because `l2.domain_mismatch_brand` needs to
compare visible text against the href target.

### RiskAssessment

```python
@dataclass
class RiskAssessment:
    message_id: str
    score: float                          # 0.0–1.0 composite
    level: RiskLevel                      # LOW / MEDIUM / HIGH
    signals: list[DetectionSignal]
    summary: str | None = None            # optional human-readable rationale
    layers_completed: list[DetectionLayer]  # for renormalization audit
    scored_at: datetime                   # UTC, defaults to now

    def signals_for(self, layer: DetectionLayer) -> list[DetectionSignal]:
```

The output of the scoring pipeline for one message. **This model stores the
result; it does not compute it.** Combining signals into `score` and banding
that into `level` is `scoring/composite.py`'s job — the contract has no
scoring logic in it.

`layers_completed` records which layers actually ran, so a degraded assessment
is auditable after the fact (section 5). `signals_for()` supports the
layer-grouped evidence table in the UI (section 9).

### LayerResult — specified, not yet implemented

Each detection layer returns a result conceptually containing:

| Field | Meaning |
|---|---|
| `layer` | which layer produced this result |
| `completed` | whether the layer actually ran to completion |
| `signals` | the signals it produced (possibly none) |
| `error` | optional reason the layer could not complete |
| `duration_ms` | wall-clock time the layer took, for the latency budget in section 5 |

**The critical semantic: `completed=False` means the layer failed or abstained
and must not be interpreted as "no threat found."**

A completed layer that produced zero suspicious signals is a genuine negative:
it ran, it looked, it found nothing. A layer with `completed=False` is an
absence of information — it timed out, crashed, or could not reach a
dependency. Collapsing those two states into "clean" is the single most
dangerous error this pipeline could make, because it silently converts an
outage into an all-clear on live mail. Scoring must therefore renormalize over
completed layers (section 6) rather than scoring a missing layer as 0, and the
UI must render an incomplete layer as "not completed" rather than blank
(section 9).

`LayerResult` is **not implemented in `core/models.py` yet.** It lands with
`core/orchestrator.py`, which is the first component that needs it.

---

## 3. Ingest

`ingest/gmail_client.py`

- OAuth 2.0 installed-app flow, scope `gmail.readonly` **only**.
- `gmail.readonly` is a Google *restricted* scope. Public verification takes
  weeks. **Stay in Testing mode** — up to 100 test users, no review needed.
  This project never leaves testing mode.
- Token cached to `.credentials/token.json`, gitignored.
- Always `format='raw'`. Returns base64url RFC-822 → decode → feed to
  `email.message_from_bytes`. Do not use `format='full'`; it loses fidelity
  and complicates MIME walking.

`ingest/parser.py`

- `email` stdlib for structure, `BeautifulSoup` for HTML body.
- URL extraction from: `<a href>`, `<img src>`, plaintext regex, and
  `<form action>`.
- Never fetch anything during parse. Parsing is pure and offline.

---

## 4. Layer specifications

### Layer 1 — Header & domain intelligence
`layers/l1_headers.py`

| Signal | Method | Expected precision |
|---|---|---|
| `l1.auth_fail` | Read Gmail's `Authentication-Results` header for spf/dkim/dmarc verdicts | high |
| `l1.replyto_mismatch` | Registrable domain of Reply-To ≠ From | high |
| `l1.domain_age_lt_7d` | WHOIS creation date | medium |
| `l1.display_name_impersonation` | Display name contains brand token, From domain does not | medium |

Do **not** re-verify SPF/DKIM cryptographically. Gmail already did it and
recorded the verdict in `Authentication-Results`. Re-verification costs hours
of work and produces a worse answer (you no longer have the original
connecting IP).

WHOIS is rate-limited and flaky. Mandatory SQLite cache with 7-day TTL,
keyed on registrable domain. Cache negative lookups too.

### Layer 2 — URL & redirect chain
`layers/l2_urls.py`

| Signal | Method |
|---|---|
| `l2.base64_email_param` | Regex for base64-encoded address in query/fragment; decode and confirm it parses as an email |
| `l2.redirect_depth` | Follow hops, `allow_redirects=False`, cap 8. Flag > 2 |
| `l2.captcha_gate` | Playwright render of terminal URL; detect Turnstile/hCaptcha/reCAPTCHA in DOM |
| `l2.qr_url` | `pyzbar` decode of image attachments + inline images; extracted URL re-enters the URL signal set |
| `l2.domain_mismatch_brand` | Anchor text names a brand, href points elsewhere |

**QR decoding is the differentiator.** Most academic phishing detectors don't
do it, and per the research doc it is Tycoon's primary SEG bypass. Give it
prominence.

**Safety, non-negotiable:**
- Playwright runs in Docker, `--network` restricted, no host mount.
- Egress via VPS or VPN. **Never fetch attacker infrastructure from a home
  or campus IP.**
- Per-hop timeout 5s, total budget 15s, hard hop cap 8.
- Never execute downloads; `accept_downloads=False`.

### Layer 3 — NLP / ML body analysis  ★ centerpiece
`layers/l3_nlp.py`

| Signal | Method |
|---|---|
| `l3.zero_width` | Regex `[\u200b\u200c\u200d\u2060\ufeff]` in body/subject |
| `l3.urgency` | TF-IDF (word 1–2gram + char 3–5gram) → LogisticRegression |
| `l3.perplexity` | GPT-2 (124M) mean per-token negative log-likelihood |
| `l3.burstiness` | Std-dev of per-sentence PPL, and of sentence length |
| `l3.fusion` | LogisticRegression over the four features above |

Implementation notes:

- GPT-2 loaded **once** as a module-level singleton at app startup. Loading
  per email is the single easiest way to destroy the latency budget.
- Truncate to 512 tokens. Compute per-sentence PPL first, then aggregate —
  burstiness needs the per-sentence vector anyway, so mean PPL is free.
- Skip perplexity entirely on bodies under ~40 tokens and emit
  `error="body too short"`. Short-text perplexity is noise; better to abstain
  than to emit a confident wrong number.
- Char n-grams in the urgency vectorizer are deliberate: they survive the
  zero-width obfuscation that L3 also detects.
- Urgency classifier trains on Nazario phishing corpus vs. Enron ham. Both
  public. Persist as `models/urgency_clf.joblib`, version the training script.
- Fusion LR replaces hand-summing L3 features. Its `coef_` values are a
  deliverable — they are the concrete detail that makes this section
  defensible in conversation.

**Known limitation, to be stated in the README, not hidden:** perplexity-based
LLM-text detection is weak on short text and has documented false-positive
bias against non-native English writers. Report the AUC you actually measure,
including if it's poor. An honest weak number is worth more than a
strong unverifiable claim.

### Layer 4 — Threat intelligence correlation
`layers/l4_intel.py`

| Signal | Method |
|---|---|
| `l4.url_ioc` | URLhaus + OpenPhish + PhishTank lookup on all extracted URLs |
| `l4.domain_ioc` | Same, registrable domain level |
| `l4.hosting_flag` | ASN lookup against a static bulletproof-hosting list |

Free feeds only. **Proofpoint and Microsoft MSTIC feeds are not publicly
accessible** — the research doc's claim otherwise must be corrected. Reframe
as: "the architecture consumes STIX/TAXII feeds; commercial feeds would slot
in at this interface."

Cache all lookups in SQLite, 6-hour TTL.

---

## 5. Orchestration

`core/orchestrator.py`

```python
results = await asyncio.gather(
    *(run_layer(l, email) for l in LAYERS),
    return_exceptions=True,
)
```

- Wall-clock cap 20s on the gather.
- Each layer additionally self-bounds: L1 8s, L2 15s, L3 10s, L4 6s.
- **Graceful degradation:** a layer that times out or raises returns
  `LayerResult(completed=False, signals=[])` (contract in section 2). Its
  configured weight is redistributed proportionally across completed layers.
  A timeout must never silently become "clean" — `completed=False` is an
  absence of information, not a negative finding.
- `layers_completed` is recorded on every `RiskAssessment` so degraded results
  are auditable after the fact.

L1/L3 are CPU-bound; wrap in `run_in_executor` so they don't block the loop.

---

## 6. Scoring

`scoring/composite.py`

**Phase A — pre-evaluation (hand-assigned):**

```
L1 0.30 | L2 0.30 | L3 0.20 | L4 0.20
thresholds: deliver < 0.35 ≤ warn < 0.65 ≤ block
```

These are the values currently in `config/weights.yaml`.

**Phase B — post-evaluation (fitted):** LogisticRegression over the full
signal vector on the labelled corpus. Thresholds re-derived from the ROC
curve, not guessed.

Choose the operating point with **false positives weighted more heavily than
false negatives**. A blocked legitimate invoice is a business incident; a
missed phish is one more item for the next control to catch. State the chosen
operating point and the reasoning in the README.

### Weights belong to scoring, not to signals

Weights live in `config/weights.yaml`, never hardcoded — and never carried on
individual `DetectionSignal` objects. A signal is an observation: this is what
was seen, this is how strongly, this is the evidence. How much that observation
counts toward the composite is a *policy* decision that changes when the model
is refit in Phase B, and it is owned by `scoring/composite.py` reading the
configuration. Keeping weights off the signal means refitting changes one YAML
file rather than every layer, and the same recorded signals can be rescored
under different weightings — which is exactly what the ablation study needs.

### Risk level and operational action are two stages

The document uses two vocabularies. They are **not** the same concept and must
not be conflated:

| Stage | Values | Produced by | Meaning |
|---|---|---|---|
| **Risk level** | `LOW` / `MEDIUM` / `HIGH` | composite scoring | *How risky is this message?* |
| **Operational action** | `deliver` / `warn` / `block` | policy stage | *What should be done about it?* |

`RiskLevel` (section 2) is the classification the scoring stage produces from
the composite score. The operational action is what a deployment decides to do
with that classification, using the thresholds in `config/weights.yaml`.

The separation matters because the mapping is a deployment policy, not a
property of the detection. The same `HIGH` classification may warrant `block`
in one deployment and `warn` in another; this prototype is read-only and takes
no action at all (section 11), so in practice the action is advisory.

**The score/level → action mapping is not implemented.** It belongs to the
later scoring/policy stage, and the thresholds it will consume are already
present in `config/weights.yaml`.

---

## 7. Repo layout

```
tycoon-detect/
├── core/
│   ├── models.py           # DetectionSignal, ParsedEmail, ExtractedURL, RiskAssessment
│   └── orchestrator.py
├── ingest/
│   ├── gmail_client.py
│   └── parser.py
├── layers/
│   ├── l1_headers.py
│   ├── l2_urls.py
│   ├── l3_nlp.py
│   └── l4_intel.py
├── scoring/
│   └── composite.py
├── storage/
│   └── cache.py            # SQLite: whois, intel, sandbox results
├── api/
│   └── app.py              # FastAPI
├── ui/                     # React + Vite
├── models/                 # persisted .joblib, gitignored
├── data/
│   ├── corpus/             # nazario/, enron/, tycoon_synth/, hard_negatives/
│   └── labels.csv
├── eval/
│   ├── run_eval.py
│   └── ablation.py
├── training/
│   ├── train_urgency.py
│   └── train_fusion.py
├── config/weights.yaml
├── docker-compose.yml
└── tests/
```

---

## 8. API surface

```
POST /auth/login              → OAuth redirect
GET  /messages?limit=25       → inbox list w/ cached verdicts
POST /analyze/{message_id}    → run pipeline, return RiskAssessment
GET  /analyze/{message_id}    → cached assessment
POST /analyze/upload          → .eml file upload (demo path, no Gmail needed)
GET  /health                  → per-layer readiness incl. GPT-2 load state
```

`/analyze/upload` matters more than it looks: it makes the whole system
demonstrable without OAuth, which is what you want when showing it to
someone on a laptop that isn't yours.

---

## 9. Frontend

React + Vite + Tailwind. Two views only.

1. **Inbox** — sender, subject, risk badge (green/amber/red), score.
2. **Detail** — the signal evidence table: every signal, fired or not,
   grouped by layer, with its `evidence` string and contribution to the score.
   Degraded layers explicitly marked "not completed," never blank.

The detail view is the product. Build it before polishing anything else.

---

## 10. Failure modes to handle explicitly

| Failure | Handling |
|---|---|
| WHOIS rate limit | Cache + backoff; signal abstains with `error` set |
| Playwright hangs on attacker page | Hard timeout, kill context, `completed=False` |
| GPT-2 OOM / slow cold start | Load at startup, `/health` gates readiness |
| Gmail token expiry | Refresh flow; 401 → re-auth prompt, never silent failure |
| Body too short for perplexity | Abstain with reason, do not emit 0.0 |
| Intel feed 5xx | Cached last-good; `completed=False` if unavailable |

---

## 11. Non-goals

- Not a Chrome extension. The Gmail DOM does not expose raw MIME, and
  Layer 1 is entirely header-based. The extension form factor cannot support
  the design.
- No public release, no Google OAuth verification, no multi-tenant use.
- No real-time inbox streaming. Pull-based, on demand.
- No remediation actions. Detection and reporting only; read-only scope
  throughout.
