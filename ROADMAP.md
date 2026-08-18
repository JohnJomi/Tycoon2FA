# ROADMAP.md

Build plan for the Tycoon 2FA detection pipeline. ~3 weeks solo.

Each task has an **acceptance criterion** — a concrete, checkable outcome.
Do not mark a task done without it. Read `ARCHITECTURE.md` first; it defines
the contracts every task depends on.

**Ordering principle:** plumbing before detection, evaluation before polish.
An end-to-end skeleton on day 2 is worth more than a perfect layer on day 6.

---

## Phase 0 — Do today, before any code

- [ ] **Google Cloud project + Gmail API enabled**
- [ ] **OAuth consent screen configured, app left in Testing mode**
- [ ] **Own Google account added as a test user**
- [ ] Credentials JSON downloaded to `.credentials/`, added to `.gitignore`

> This is the only task with an external dependency and it gates everything
> downstream. `gmail.readonly` is a restricted scope — Testing mode (100
> users, no review) is the intended path. Discovering an OAuth problem on
> day 6 costs the project; discovering it today costs an hour.

- [ ] `git init`, Python 3.11+ venv, `requirements.txt`
- [ ] Docker installed and running (needed from Phase 3)
- [ ] Decide egress for Layer 2: cheap VPS or VPN. **Not your home IP.**

---

## Phase 1 — Skeleton (Days 1–2)

Goal: one `.eml` in, a signal table out. Detection logic can be fake.

- [ ] `core/models.py` — Signal, LayerResult, ParsedEmail, RiskAssessment
  - *Accept:* imported by a test that constructs each dataclass.
- [ ] `ingest/gmail_client.py` — OAuth flow, token cache, `format='raw'` fetch
  - *Accept:* CLI pulls last 20 message IDs and prints From/Subject for each.
- [ ] `ingest/parser.py` — RFC-822 → ParsedEmail
  - *Accept:* on a real message, dumps headers, both body parts, extracted
    URLs, attachment names.
- [ ] `core/orchestrator.py` — asyncio.gather, timeouts, renormalization
  - *Accept:* with all four layers stubbed (L1 returns one hardcoded fired
    signal, L2–L4 return empty), pipeline runs end to end and prints a score.
- [ ] `scoring/composite.py` — weighted sum, renormalization on incomplete
  layers, verdict bands from `config/weights.yaml`
  - *Accept:* unit test proves a forced L2 timeout redistributes weight and
    does not lower the score toward "clean."
- [ ] `storage/cache.py` — SQLite, TTL-aware get/set
  - *Accept:* second identical lookup within TTL performs no network call.

**Phase 1 exit:** `python -m cli analyze sample.eml` prints a signal table.
Do not proceed until this works.

---

## Phase 2 — Layer 1 + Layer 3 core (Days 3–5)

Offline-first layers. Layer 3 is the project centerpiece — budget two days.

### Layer 1 (half a day)
- [ ] Parse `Authentication-Results` for spf/dkim/dmarc verdicts
  - *Accept:* correctly reports pass/fail on 5 real messages. Do **not**
    re-verify cryptographically.
- [ ] Reply-To vs From registrable-domain comparison (`tldextract`)
- [ ] WHOIS domain age, cached, 7-day TTL, negative caching
  - *Accept:* handles WHOIS failure by abstaining with `error` set, never by
    returning `fired=False` silently.
- [ ] Display-name brand impersonation check

### Layer 3 (two days) ★
- [ ] Zero-width scan — `[\u200b\u200c\u200d\u2060\ufeff]`
  - *Accept:* fires on a crafted sample, silent on 50 Enron ham messages.
- [ ] `training/train_urgency.py` — download Nazario + Enron, TF-IDF
      (word 1–2gram + char 3–5gram) → LogisticRegression, persist joblib
  - *Accept:* held-out accuracy printed and written to `eval/results/`.
    Whatever the number is, record it.
- [ ] GPT-2 perplexity — singleton load at startup, per-sentence PPL, 512-token
      truncation, abstain under 40 tokens
  - *Accept:* scores a message in < 2s warm; `/health` reports model state;
    a 20-token body returns `error`, not a fabricated score.
- [ ] Burstiness — std-dev of per-sentence PPL and of sentence length
  - *Accept:* reuses the per-sentence PPL vector; no second forward pass.
- [ ] `training/train_fusion.py` — LR over the four L3 features
  - *Accept:* `coef_` printed and saved. These numbers are a deliverable.

**Phase 2 exit:** Layers 1 and 3 produce real signals with evidence strings
on real mail.

---

## Phase 3 — Layer 2 (Days 6–8)

Network layer. Highest risk, so it is sandboxed from the first commit.

- [ ] Docker container for the fetcher: restricted network, no host mounts,
      `accept_downloads=False`
  - *Accept:* container cannot reach the host; verified, not assumed.
- [ ] Redirect chain follower — `allow_redirects=False`, 5s/hop, 8-hop cap
  - *Accept:* correctly reports depth on a known 4-hop test chain.
- [ ] Base64 email-param detection — regex, decode, validate as address
  - *Accept:* fires on a synthesized Tycoon-style URL, silent on ordinary
    tracking URLs with base64 in them (test this specifically — it's the
    obvious false-positive source).
- [ ] Playwright render of terminal URL; Turnstile / hCaptcha / reCAPTCHA
      detection in DOM
  - *Accept:* detects Turnstile on a live demo page.
- [ ] QR decoding — `pyzbar` over image attachments and inline `cid:` images;
      extracted URLs re-enter the URL signal set
  - *Accept:* decodes a QR-in-email test sample end to end and the extracted
    URL is analyzed by the other L2 signals.

> QR is the standout feature. Give it a dedicated demo sample you can show.

**Phase 3 exit:** Layer 2 analyzes a live redirect chain from sandboxed
egress without touching your own IP.

---

## Phase 4 — Layer 4 (Day 9, short)

- [ ] URLhaus + OpenPhish + PhishTank clients, cached 6h
- [ ] ASN lookup vs. static bulletproof-hosting list
- [ ] **Correct the research doc:** Proofpoint/MSTIC feeds are not publicly
      accessible. Reframe as a STIX/TAXII interface where commercial feeds
      would slot in.
  - *Accept:* the writeup no longer claims access to feeds you don't have.

---

## Phase 5 — Corpus & evaluation (Days 10–13)

**The highest-value phase. Do not let it get squeezed.** This is what turns
"I built a perplexity detector" into "perplexity separated LLM-written lures
from human business mail at AUC 0.74, weakest under 40 tokens."

- [ ] Assemble corpus in `data/corpus/`:
  - Nazario phishing + Enron ham (already downloaded in Phase 2)
  - 60–80 synthesized Tycoon-pattern samples built from published IOCs
  - [ ] **Hard negatives: 40+ genuine Microsoft/Google security-alert emails**
- [ ] `data/labels.csv`

> Hard negatives matter most. Real M365 alerts are urgent, branded, and
> link-heavy — precisely the profile your detector will false-positive on.
> Find that out here, not during the demo.

- [ ] `eval/run_eval.py` — per-signal precision/recall/F1, confusion matrix
  - *Accept:* results written to `eval/results/`, reproducible from a clean
    checkout with one command.
- [ ] `eval/ablation.py` — score with each layer disabled in turn
  - *Accept:* a table showing each layer's marginal contribution. **This is
    the single strongest artifact in the project.**
- [ ] ROC curves per layer and composite; derive thresholds from the curve
- [ ] Refit fusion + composite weights on the labelled set; write
      `config/weights.yaml`
  - *Accept:* README states weights are fitted, with the corpus described
    honestly including its synthetic portion.

**Expected outcome, so it isn't alarming:** zero-width will fire on almost
nothing real (high precision, low recall) and urgency will fire on plenty of
legitimate mail (low precision, high recall). That is the normal shape of
this problem and it is exactly what fitting resolves.

---

## Phase 6 — Product (Days 14–17)

- [ ] FastAPI endpoints per `ARCHITECTURE.md` §8, including
      `POST /analyze/upload`
  - *Accept:* full pipeline demonstrable from a `.eml` upload with no OAuth.
- [ ] React inbox view — sender, subject, risk badge, score
- [ ] **React detail view — the signal evidence table.** Every signal grouped
      by layer, fired or not, with evidence string and score contribution.
      Degraded layers marked "not completed," never blank.
  - *Accept:* a stranger can look at the screen and say why the email was
    flagged. This view is the product; build it before any polish.
- [ ] `docker-compose.yml` — API, fetcher, UI
  - *Accept:* `docker compose up` on a clean machine produces a working demo.

---

## Phase 7 — Writeup (Days 18–21)

- [ ] README with real numbers from `eval/results/`, including the weak ones
- [ ] Limitations section: synthetic corpus, perplexity weakness on short
      text and its documented bias against non-native English writers,
      free-feed-only intel
- [ ] Update the research paper: fix the Proofpoint/MSTIC claim, replace
      asserted weights with fitted ones, add the ablation table
- [ ] Demo script — 3 samples: QR-based, redirect-chain-based, and a hard
      negative that correctly passes. **Show the true negative.** Demonstrating
      that it doesn't flag everything is more convincing than three catches.

---

## Resume line (write it now, verify it at the end)

> Built a four-layer AiTM phishing detection pipeline (Python, FastAPI,
> Gmail API) analyzing raw MIME for header-auth failures, redirect-chain
> anomalies, QR-embedded URLs, and Unicode obfuscation. Layer 3 combines a
> TF-IDF/logistic-regression urgency classifier with GPT-2 perplexity and
> burstiness scoring for LLM-authored-text detection; async layer execution
> with graceful degradation on timeout.

Every clause must be something you can open a terminal and demonstrate. At
the end of Phase 7, go through it clause by clause and cut anything you
can't. A shorter true line beats a longer one you can't defend — the
interview question is *"what perplexity range did you actually see?"* and
Phase 5 is where you get the answer.

---

## Cut list, if time runs short

Cut in this order:
1. React frontend → Streamlit (saves 2 days, demo still works)
2. Playwright rendering → static HTML string matching (loses CAPTCHA detection)
3. PhishTank (keep URLhaus + OpenPhish)
4. ASN / bulletproof-hosting signal

**Never cut:** Phase 5. Without evaluation the project has no defensible
claims, and a project with no numbers is a liability on a resume rather than
an asset.
