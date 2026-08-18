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
   scoring/composite.py  → RiskAssessment(score, verdict, signals[])
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
depends on nothing.

```python
@dataclass(frozen=True)
class Signal:
    name: str            # stable id, e.g. "l1.domain_age_lt_7d"
    layer: int           # 1..4
    fired: bool          # binary detection outcome
    value: float         # 0.0–1.0 normalized strength
    weight: float        # assigned pre-fit; overwritten by fitted model
    evidence: str        # human-readable, shown in UI. REQUIRED.
    error: str | None = None   # set when the signal could not be computed


@dataclass
class LayerResult:
    layer: int
    signals: list[Signal]
    completed: bool      # False on timeout/exception → renormalization
    duration_ms: int


@dataclass
class ParsedEmail:
    message_id: str
    raw: bytes
    headers: dict[str, list[str]]   # multi-valued, case-normalized keys
    from_addr: str
    from_display: str
    reply_to: str | None
    subject: str
    body_text: str       # plaintext part, or html→text fallback
    body_html: str | None
    urls: list[str]      # deduped, from html hrefs + text extraction
    attachments: list[Attachment]
    inline_images: list[Attachment]   # cid: referenced
    received_chain: list[str]


@dataclass
class RiskAssessment:
    message_id: str
    score: float                  # 0.0–1.0
    verdict: Literal["deliver", "warn", "block"]
    signals: list[Signal]
    layers_completed: list[int]   # for renormalization audit
    scored_at: datetime
```

**Rule:** `evidence` is never empty on a fired signal. The evidence strings
are the demo. A bare score demonstrates nothing.

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
  `LayerResult(completed=False, signals=[])`. Its weight is redistributed
  proportionally across completed layers. A timeout must never silently
  become "clean."
- `layers_completed` is recorded on every assessment so degraded results are
  auditable after the fact.

L1/L3 are CPU-bound; wrap in `run_in_executor` so they don't block the loop.

---

## 6. Scoring

`scoring/composite.py`

**Phase A — pre-evaluation (hand-assigned):**

```
L1 0.30 | L2 0.30 | L3 0.20 | L4 0.20
thresholds: deliver < 0.35 ≤ warn < 0.65 ≤ block
```

**Phase B — post-evaluation (fitted):** LogisticRegression over the full
signal vector on the labelled corpus. Thresholds re-derived from the ROC
curve, not guessed.

Choose the operating point with **false positives weighted more heavily than
false negatives**. A blocked legitimate invoice is a business incident; a
missed phish is one more item for the next control to catch. State the chosen
operating point and the reasoning in the README.

Weights live in `config/weights.yaml`, never hardcoded.

---

## 7. Repo layout

```
tycoon-detect/
├── core/
│   ├── models.py           # Signal, LayerResult, ParsedEmail, RiskAssessment
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
