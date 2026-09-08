"""One-off probe: does the Walter Writes AI-detection API work from here?

    python scripts/probe_walter_writes.py

Not Layer 3, and not a step toward it. This is a throwaway connectivity and
contract check that answers three questions before any production code is
written: does the endpoint accept our key, what does a success response
actually look like, and how does it behave on the two ends of the range we
care about (clearly machine-written vs clearly human-written text).

Nothing here is imported by the package. The provider abstraction, the signal
and the orchestrator wiring come later, once this has confirmed the contract.

The API key is read from the environment (WALTER_WRITES_API_KEY, loaded from
the gitignored .env if present) and is never printed, logged, or included in
any output - not even in an error path.
"""

from __future__ import annotations

import json
import os
import sys

import httpx
from dotenv import load_dotenv

ENDPOINT = "https://developer-portal.walterwrites.ai/api/detector/"
API_KEY_ENV = "WALTER_WRITES_API_KEY"
TIMEOUT_SECONDS = 60.0

# The API rejects anything shorter, with a 400. Recorded here because it is a
# real constraint on Layer 3: plenty of phishing bodies are under 50 words.
MIN_WORDS = 50

SAMPLES: dict[str, str] = {
    "machine-written": (
        "In today's rapidly evolving digital landscape, organizations must "
        "leverage cutting-edge artificial intelligence solutions to remain "
        "competitive. By implementing robust machine learning frameworks, "
        "businesses can unlock unprecedented value from their data assets. "
        "Furthermore, a comprehensive approach to digital transformation "
        "enables stakeholders to optimize operational efficiency while "
        "simultaneously enhancing customer engagement. It is important to note "
        "that successful adoption requires careful consideration of both "
        "technical infrastructure and organizational readiness across every "
        "department within the wider enterprise."
    ),
    "human-written": (
        "ok so I finally got the bike fixed yesterday, took way longer than it "
        "should have. the guy at the shop said the chain was basically toast "
        "and he had to order a new one, which took like four days. anyway it "
        "rides fine now, though the brakes still squeak a bit when it rains. "
        "gonna take it out to the lake this weekend if the weather holds up, "
        "assuming I can find my helmet. let me know if you want to come along."
    ),
}


def detect(client: httpx.Client, text: str) -> tuple[int, object]:
    """POST one sample. Returns (status, parsed body or raw text)."""
    response = client.post(ENDPOINT, json={"content": text})
    try:
        return response.status_code, response.json()
    except json.JSONDecodeError:
        # A non-JSON body is itself a finding - show enough to identify it
        # (a WAF challenge page, an HTML error) without dumping the whole page.
        return response.status_code, f"<non-JSON body: {response.text[:200]!r}>"


def report(label: str, text: str, client: httpx.Client) -> bool:
    words = len(text.split())
    print(f"\n--- {label} ({words} words) ---")
    if words < MIN_WORDS:
        print(f"  SKIPPED: under the API's {MIN_WORDS}-word minimum")
        return False

    try:
        status, body = detect(client, text)
    except httpx.TimeoutException:
        print(f"  FAILED: no response within {TIMEOUT_SECONDS:g}s")
        return False
    except httpx.HTTPError as exc:
        # Type only. A transport error's message can carry the request URL,
        # and the key must never reach output by any route.
        print(f"  FAILED: transport error ({type(exc).__name__})")
        return False

    print(f"  HTTP {status}")
    if isinstance(body, dict):
        print("  response:")
        print("    " + json.dumps(body, indent=2, sort_keys=True).replace("\n", "\n    "))
    else:
        print(f"  response: {body}")
    return 200 <= status < 300


def main() -> int:
    load_dotenv()
    if not os.environ.get(API_KEY_ENV):
        print(
            f"error: {API_KEY_ENV} is not set. Put it in .env (gitignored) or "
            f"export it in your shell.",
            file=sys.stderr,
        )
        return 1

    print(f"POST {ENDPOINT}")
    print("auth: X-API-Key header (value not shown)")

    headers = {
        "X-API-Key": os.environ[API_KEY_ENV],
        "Content-Type": "application/json",
    }
    ok = True
    with httpx.Client(headers=headers, timeout=TIMEOUT_SECONDS) as client:
        for label, text in SAMPLES.items():
            ok = report(label, text, client) and ok

    print("\n" + ("probe OK" if ok else "probe FAILED - see above"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
