"""Calibration experiment: how does Walter Writes score real human email?

    python scripts/calibrate_walter_writes.py

Companion to `probe_walter_writes.py`, which established that the endpoint
works. This one asks the question that actually gates Layer 3: does
`ai_score` separate human-written text from anything, or does it call
everything AI-generated? The probe's human sample scored 0.9157, and one
sample is not a finding.

Three groups:
  PERSONAL-*      the operator's own Sent mail - authorship is certain
  KAGGLE-PHISH-*  phishing rows from the Kaggle corpus
  KAGGLE-LEGIT-*  legitimate rows from the same corpus

Reuses `probe_walter_writes.detect` rather than building a second HTTP path.

Privacy: message bodies are held in memory only. Nothing writes an email body,
subject, address or message id to disk or to the output - the results carry
word counts and scores. The API key is read from the environment and never
printed.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

_spec = importlib.util.spec_from_file_location(
    "probe_walter_writes", _ROOT / "scripts" / "probe_walter_writes.py"
)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

MIN_WORDS = probe.MIN_WORDS
PER_GROUP = 5
KAGGLE_CSV = os.environ.get("KAGGLE_CSV", "")

# The API throttles at roughly five requests per minute and says so only in
# the 429 body ("Expected available in 57 seconds") - there is no Retry-After
# header and no documented limit. Pace requests rather than discovering it
# again, and honour the stated wait once if it still fires.
DELAY_SECONDS = float(os.environ.get("DELAY_SECONDS", "15"))
GROUPS = {g.strip() for g in os.environ.get("GROUPS", "personal,phishing").split(",")}

AUTOMATED_MARKERS = ("no-reply", "noreply", "donotreply", "notifications")


# ---------------------------------------------------------------- gathering


def _is_automated(email) -> bool:
    headers = {k.lower() for k in email.headers}
    if "list-unsubscribe" in headers:
        return True
    return any(m in (email.from_addr or "").lower() for m in AUTOMATED_MARKERS)


def personal_samples() -> list[tuple[str, str, int]]:
    """(id, text, words) from the operator's own Sent mail.

    Sent mail, not received: the inbox held no human-written message at all,
    and authorship of one's own sent mail is the only ground truth available
    here that does not require trusting a guess.
    """
    from ingest.gmail_client import GmailClient
    from ingest.pipeline import GmailIngestor

    ingestor = GmailIngestor(GmailClient())
    out: list[tuple[str, str, int]] = []
    for message in ingestor.ingest("in:sent", max_results=40):
        text = (message.email.body_text or "").strip()
        words = len(text.split())
        if words < MIN_WORDS or _is_automated(message.email):
            continue
        out.append((f"PERSONAL-{len(out) + 1:02d}", text, words))
        if len(out) == PER_GROUP:
            break
    return out


def _strip_dataset_noise(text: str) -> str:
    """Drop the corpus's own generation artifact.

    Some phishing rows end with a literal `Keywords: ...` trailer listing the
    phrases the row was generated from. That is dataset formatting, not email
    text, and sending it would score the annotation rather than the message.
    """
    return text.split("\nKeywords:")[0].strip()


def kaggle_samples() -> tuple[list[tuple[str, str, int]], list[tuple[str, int]]]:
    """(sendable phishing, skipped legitimate) - the corpus's legit rows are short."""
    csv.field_size_limit(10**7)
    with open(KAGGLE_CSV, newline="", encoding="utf-8", errors="replace") as handle:
        rows = list(csv.DictReader(handle))

    # One row per distinct `phishing_type`, deduplicated by text. Taking the
    # first five qualifying rows instead returns near-duplicates - the corpus
    # is templated, and a first attempt drew five texts that were 97-100%
    # similar, two of them byte-identical. Five copies of one message is n=1
    # dressed as n=5.
    phish: list[tuple[str, str, int]] = []
    seen_types: set[str] = set()
    seen_texts: set[str] = set()
    for row in rows:
        if row["label"] != "1":
            continue
        text = _strip_dataset_noise(row["text"] or "")
        words = len(text.split())
        if words < MIN_WORDS or text in seen_texts:
            continue
        # Type diversity is not available: every row that reaches 50 words is
        # `social_engineering_advanced`, and those 500 rows hold 6 unique
        # texts between them. Deduplicating by text is the most independence
        # this corpus can offer.
        seen_types.add(row["phishing_type"])
        seen_texts.add(text)
        phish.append((f"KAGGLE-PHISH-{len(phish) + 1:02d}", text, words))
        if len(phish) == PER_GROUP:
            break

    legit = sorted(
        (len(_strip_dataset_noise(r["text"] or "").split()) for r in rows if r["label"] == "0"),
        reverse=True,
    )[:PER_GROUP]
    return phish, [(f"KAGGLE-LEGIT-{i:02d}", w) for i, w in enumerate(legit, 1)]


# ----------------------------------------------------------------- running


def _retry_after_seconds(body: object) -> float | None:
    """The wait the 429 body states, since no Retry-After header is sent."""
    if not isinstance(body, dict):
        return None
    match = re.search(r"available in (\d+) seconds", str(body.get("error", "")))
    return float(match.group(1)) + 2 if match else None


def measure(client: httpx.Client, sample_id: str, source: str, cls: str,
            text: str, words: int) -> dict:
    started = time.perf_counter()
    status, body = probe.detect(client, text)

    if status == 429:
        wait = _retry_after_seconds(body)
        if wait is not None:
            print(f"    throttled; waiting {wait:.0f}s and retrying once")
            time.sleep(wait)
            started = time.perf_counter()
            status, body = probe.detect(client, text)

    latency_ms = round((time.perf_counter() - started) * 1000)

    record = {
        "id": sample_id, "source": source, "class": cls, "words": words,
        "http_status": status, "latency_ms": latency_ms, "succeeded": False,
        "ai_score": None, "result": None, "items": None,
        "sentence_min": None, "sentence_max": None, "sentence_mean": None,
        "credits_remaining": None, "error": None,
    }
    if not isinstance(body, dict):
        record["error"] = "non-JSON response"
        return record
    if not (200 <= status < 300):
        record["error"] = f"{body.get('code') or 'http_error'}: {body.get('error')}"
        return record

    try:
        record["ai_score"] = float(body["ai_score"])
        record["result"] = body.get("result")
        record["credits_remaining"] = body.get("credits_remaining")
        scores = [float(i["ai_score"]) for i in body.get("items") or []]
        if scores:
            record["items"] = len(scores)
            record["sentence_min"] = min(scores)
            record["sentence_max"] = max(scores)
            record["sentence_mean"] = statistics.mean(scores)
        record["succeeded"] = True
    except (KeyError, TypeError, ValueError) as exc:
        record["error"] = f"malformed response ({type(exc).__name__})"
    return record


def main() -> int:
    load_dotenv()
    if not os.environ.get(probe.API_KEY_ENV):
        print(f"error: {probe.API_KEY_ENV} is not set", file=sys.stderr)
        return 1
    if not KAGGLE_CSV or not Path(KAGGLE_CSV).is_file():
        print("error: set KAGGLE_CSV to the dataset path", file=sys.stderr)
        return 1

    personal = personal_samples() if "personal" in GROUPS else []
    phish, skipped_legit = kaggle_samples()
    batch = (
        [(i, "personal", "personal", t, w) for i, t, w in personal
         if "personal" in GROUPS]
        + [(i, "kaggle", "phishing", t, w) for i, t, w in phish
           if "phishing" in GROUPS]
    )

    print(f"personal={len(personal)} phishing={len(phish)} "
          f"legit_skipped={len(skipped_legit)}")

    headers = {"X-API-Key": os.environ[probe.API_KEY_ENV],
               "Content-Type": "application/json"}
    results: list[dict] = []
    with httpx.Client(headers=headers, timeout=probe.TIMEOUT_SECONDS) as client:
        for index, (sample_id, source, cls, text, words) in enumerate(batch):
            if index:
                time.sleep(DELAY_SECONDS)
            record = measure(client, sample_id, source, cls, text, words)
            results.append(record)
            score = "----" if record["ai_score"] is None else f"{record['ai_score']:.4f}"
            print(f"  {record['id']:<18} {words:>4}w  HTTP {record['http_status']}  "
                  f"ai_score={score}  result={record['result']}  "
                  f"{record['latency_ms']}ms")

    payload = {
        "results": results,
        "skipped_below_50_words": [
            {"id": i, "words": w, "reason": "skipped_below_50_words"}
            for i, w in skipped_legit
        ],
        "total_words_sent": sum(r["words"] for r in results),
    }
    out = Path(os.environ.get("CALIBRATION_OUT", "calibration_results.json"))
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwords sent: {payload['total_words_sent']}  ->  {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
