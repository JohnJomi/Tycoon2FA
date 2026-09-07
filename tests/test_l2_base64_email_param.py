"""Unit tests for the l2.base64_email_param signal in layers/l2_urls.py.

Entirely offline: the signal reads parsed URL parts and decodes strings. An
autouse guard forbids every socket and DNS entry point for the whole file, so a
test that somehow reached the network would fail rather than pass quietly.
"""

from __future__ import annotations

import base64

import pytest

from core.models import (
    DetectionLayer,
    ExtractedURL,
    ParsedEmail,
    RiskLevel,
    URLSource,
)
from layers.l2_urls import (
    BASE64_EMAIL_PARAM_SCORE,
    analyze_base64_email_param,
    decode_base64_email,
)

VICTIM = "victim@corp-invoices.com"
ENCODED = base64.b64encode(VICTIM.encode()).decode()  # 'dmljdGltQGNvcnAtaW52b2ljZXMuY29t'


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("the base64_email_param suite touched the network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket, "gethostbyname", forbidden)


def email(*urls: str) -> ParsedEmail:
    return ParsedEmail(
        message_id="<m@example.com>",
        from_addr="sender@example.com",
        subject="Invoice",
        body_text="See the portal.",
        urls=[ExtractedURL(url=u, source=URLSource.ANCHOR_HREF) for u in urls],
    )


def b64(value: str) -> str:
    return base64.b64encode(value.encode()).decode()


# --- 1 & 2. findings in query and fragment --------------------------------


def test_an_encoded_address_in_a_query_parameter_is_a_finding() -> None:
    url = f"https://evil-phish.com/login?u={ENCODED}"
    signal = analyze_base64_email_param(email(url))

    assert signal.layer is DetectionLayer.L2
    assert signal.name == "base64_email_param"
    assert signal.score == BASE64_EMAIL_PARAM_SCORE
    assert signal.severity is RiskLevel.HIGH
    assert signal.error is None
    assert signal.metadata["fired"] is True
    assert signal.metadata["matches"] == [
        {
            "url": url,
            "location": "query",
            "parameter": "u",
            "encoded": ENCODED,
            "email": VICTIM,
        }
    ]
    assert VICTIM in signal.evidence


def test_an_encoded_address_in_a_fragment_parameter_is_a_finding() -> None:
    url = f"https://evil-phish.com/login#u={ENCODED}"
    signal = analyze_base64_email_param(email(url))

    assert signal.score == BASE64_EMAIL_PARAM_SCORE
    match = signal.metadata["matches"][0]
    assert match["location"] == "fragment"
    assert match["parameter"] == "u"
    assert match["email"] == VICTIM


def test_a_bare_fragment_carrying_the_address_is_a_finding() -> None:
    """A fragment with no `=` is not a parameter list, but it is a fragment."""
    url = f"https://evil-phish.com/login#{ENCODED}"
    signal = analyze_base64_email_param(email(url))

    assert signal.score == BASE64_EMAIL_PARAM_SCORE
    match = signal.metadata["matches"][0]
    assert match["location"] == "fragment"
    assert match["parameter"] is None


def test_percent_encoded_padding_still_decodes() -> None:
    """`=` arrives as %3D in a real URL and is decoded by the foundation."""
    padded = b64("a.victim@corp-invoices.com")
    assert padded.endswith("=")
    url = f"https://evil-phish.com/x?e={padded.replace('=', '%3D')}"

    signal = analyze_base64_email_param(email(url))

    assert signal.metadata["matches"][0]["email"] == "a.victim@corp-invoices.com"


# --- 3, 4, 5. things that must not fire -----------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "not-base64!!!",
        "dmljdGlt QGNvcnA=",       # whitespace inside
        "dmljdGltQGNvcnAtaW52b2",  # truncated, invalid length
        "****************",
    ],
)
def test_invalid_base64_is_not_a_finding(value: str) -> None:
    signal = analyze_base64_email_param(email(f"https://example.com/x?u={value}"))

    assert signal.score == 0.0
    assert signal.error is None          # a clean result, not an abstention
    assert signal.metadata["matches"] == []


@pytest.mark.parametrize(
    "plaintext",
    [
        "this is just some ordinary text",
        "0123456789012345678901234567890123456789",
        "session-token-for-the-current-user",
        "no-at-sign-here.example.com",
    ],
)
def test_valid_base64_that_is_not_an_email_is_not_a_finding(plaintext: str) -> None:
    """Arbitrary base64 is ordinary in URLs and must never fire on its own."""
    signal = analyze_base64_email_param(
        email(f"https://example.com/x?token={b64(plaintext)}")
    )

    assert signal.score == 0.0
    assert signal.metadata["matches"] == []


def test_a_plain_unencoded_email_in_a_parameter_is_not_this_signal() -> None:
    """This signal is about *encoded* addresses; a bare one is another question."""
    signal = analyze_base64_email_param(
        email(f"https://example.com/x?u={VICTIM.replace('@', '%40')}")
    )

    assert signal.score == 0.0
    assert signal.metadata["matches"] == []


def test_an_address_on_a_domain_without_a_public_suffix_does_not_count() -> None:
    """The same bar Layer 1 applies to an address anywhere else."""
    assert decode_base64_email(b64("root@localhost")) is None
    assert decode_base64_email(b64("a@b")) is None


def test_a_decoded_sentence_containing_an_at_sign_is_not_an_address() -> None:
    assert decode_base64_email(b64("meet me @ the cafe at noon")) is None


def test_non_utf8_bytes_decode_to_nothing() -> None:
    encoded = base64.b64encode(b"\xff\xfe\xfa\xfb\xfc\xfd\xf0\xf1\xf2").decode()

    assert decode_base64_email(encoded) is None


# --- 6. empty and blank values --------------------------------------------


def test_a_blank_parameter_value_is_visible_and_yields_nothing() -> None:
    signal = analyze_base64_email_param(email("https://example.com/x?u=&v="))

    assert signal.score == 0.0
    assert signal.metadata["matches"] == []
    assert signal.error is None


def test_a_url_with_no_query_or_fragment_is_clean() -> None:
    signal = analyze_base64_email_param(email("https://example.com/plain"))

    assert signal.score == 0.0
    assert signal.metadata["urls_checked"] == 1
    assert "None of the 1 URL(s)" in signal.evidence


def test_a_message_with_no_urls_is_a_genuine_negative() -> None:
    signal = analyze_base64_email_param(email())

    assert signal.score == 0.0
    assert signal.error is None
    assert signal.metadata["urls_checked"] == 0
    assert "no analysable URLs" in signal.evidence


# --- 7. malformed candidates ----------------------------------------------


def test_a_malformed_url_does_not_stop_the_others_being_analysed() -> None:
    signal = analyze_base64_email_param(
        email(
            "javascript:alert(1)",
            "https://example.com:99999/x",
            f"https://evil-phish.com/login?u={ENCODED}",
        )
    )

    assert signal.score == BASE64_EMAIL_PARAM_SCORE
    assert signal.metadata["urls_in_message"] == 3
    assert signal.metadata["urls_checked"] == 1   # only the parseable one
    assert signal.metadata["matches"][0]["email"] == VICTIM


@pytest.mark.parametrize(
    "value", [None, 12345, "", "   ", "=", "====", "a", "\x00\x01"]
)
def test_decoding_never_raises_on_hostile_input(value) -> None:
    assert decode_base64_email(value) is None


# --- 8, 9, 10. several parameters, URLs, duplicates ------------------------


def test_every_parameter_is_examined() -> None:
    other = b64("second.victim@corp-invoices.com")
    url = f"https://evil-phish.com/x?a=plain&u={ENCODED}&t={b64('opaque token')}&z={other}"

    signal = analyze_base64_email_param(email(url))

    assert [m["parameter"] for m in signal.metadata["matches"]] == ["u", "z"]
    assert [m["email"] for m in signal.metadata["matches"]] == [
        VICTIM,
        "second.victim@corp-invoices.com",
    ]


def test_query_and_fragment_are_both_examined_on_one_url() -> None:
    url = f"https://evil-phish.com/x?u={ENCODED}#e={b64('other@corp-invoices.com')}"

    signal = analyze_base64_email_param(email(url))

    assert [m["location"] for m in signal.metadata["matches"]] == ["query", "fragment"]


def test_multiple_urls_are_analysed_independently() -> None:
    signal = analyze_base64_email_param(
        email(
            "https://good.example/clean",
            f"https://evil-phish.com/a?u={ENCODED}",
            f"https://other-phish.com/b#{b64('third@corp-invoices.com')}",
        )
    )

    assert signal.metadata["urls_checked"] == 3
    assert len(signal.metadata["matches"]) == 2
    assert [m["url"] for m in signal.metadata["matches"]] == [
        f"https://evil-phish.com/a?u={ENCODED}",
        f"https://other-phish.com/b#{b64('third@corp-invoices.com')}",
    ]


def test_duplicate_canonical_urls_are_analysed_once() -> None:
    url = f"https://evil-phish.com/a?u={ENCODED}"
    message = ParsedEmail(
        message_id="<m@example.com>",
        from_addr="sender@example.com",
        urls=[
            ExtractedURL(url=url, source=URLSource.ANCHOR_HREF),
            ExtractedURL(url=url.replace("https://evil", "HTTPS://Evil"), source=URLSource.PLAIN_TEXT),
            ExtractedURL(url=url, source=URLSource.IMG_SRC),
        ],
    )

    signal = analyze_base64_email_param(message)

    assert signal.metadata["urls_in_message"] == 3
    assert signal.metadata["urls_checked"] == 1
    assert len(signal.metadata["matches"]) == 1


# --- 11. raw URL in evidence ----------------------------------------------


def test_the_raw_url_is_quoted_not_the_canonical_form() -> None:
    messy = f"HTTPS://Evil-Phish.com:443/Login?u={ENCODED}"
    signal = analyze_base64_email_param(email(messy))

    assert messy in signal.evidence
    assert signal.metadata["matches"][0]["url"] == messy
    assert "https://evil-phish.com/Login" not in signal.evidence


def test_metadata_carries_only_what_the_finding_needs() -> None:
    """Other parameters may hold session tokens and are not this signal's business."""
    url = f"https://evil-phish.com/x?u={ENCODED}&session=SUPERSECRETVALUE"
    signal = analyze_base64_email_param(email(url))

    match = signal.metadata["matches"][0]
    assert set(match) == {"url", "location", "parameter", "encoded", "email"}
    assert "SUPERSECRETVALUE" not in str(match["encoded"])
    assert set(signal.metadata) == {"urls_in_message", "urls_checked", "matches", "fired"}


# --- 12. determinism -------------------------------------------------------


def test_repeated_analysis_is_deterministic() -> None:
    message = email(
        f"https://evil-phish.com/a?u={ENCODED}",
        f"https://other-phish.com/b#e={b64('third@corp-invoices.com')}",
    )

    results = {
        (
            analyze_base64_email_param(message).score,
            analyze_base64_email_param(message).evidence,
        )
        for _ in range(5)
    }

    assert len(results) == 1


def test_the_parsed_message_is_not_mutated() -> None:
    message = email(f"https://evil-phish.com/a?u={ENCODED}")

    analyze_base64_email_param(message)

    assert message.urls[0].redirect_chain == []
    assert message.urls[0].final_url is None


# --- QR re-entry ----------------------------------------------------------


def test_a_qr_decoded_url_is_analysed_too() -> None:
    decoded = [
        ExtractedURL(url=f"https://evil-phish.com/qr?u={ENCODED}", source=URLSource.QR_CODE)
    ]

    signal = analyze_base64_email_param(email(), extra=decoded)

    assert signal.score == BASE64_EMAIL_PARAM_SCORE
    assert signal.metadata["matches"][0]["email"] == VICTIM
