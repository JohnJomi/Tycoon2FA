"""Unit tests for the l2.domain_mismatch_brand signal in layers/l2_urls.py.

Offline: a fixed brand table and exact token matching. An autouse guard forbids
every socket and DNS entry point for the whole file.

The brand corpus is `l1_headers.BRAND_DOMAINS`, reused rather than duplicated,
so the boundary tests below assert against that table's actual contents.
"""

from __future__ import annotations

import pytest

from core.models import (
    DetectionLayer,
    ExtractedURL,
    ParsedEmail,
    RiskLevel,
    URLSource,
)
from layers.l1_headers import BRAND_DOMAINS
from layers.l2_urls import (
    BRAND_MISMATCH_SCORE,
    analyze_domain_mismatch_brand,
    brands_named_in,
)

PHISH = "https://secure-login.tk/verify"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("the domain_mismatch_brand suite touched the network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket, "gethostbyname", forbidden)


def link(url: str, anchor: str | None) -> ExtractedURL:
    return ExtractedURL(url=url, source=URLSource.ANCHOR_HREF, anchor_text=anchor)


def email(*links: ExtractedURL) -> ParsedEmail:
    return ParsedEmail(
        message_id="<m@example.com>",
        from_addr="sender@example.com",
        subject="Invoice",
        body_text="See the portal.",
        urls=list(links),
    )


# --- the corpus -----------------------------------------------------------


def test_the_brand_corpus_is_layer_ones_and_is_not_duplicated() -> None:
    """A second list would drift; the two signals must agree on who a brand is."""
    import layers.l2_urls as module

    assert module.BRAND_DOMAINS is BRAND_DOMAINS
    assert "microsoft" in BRAND_DOMAINS
    assert "microsoftonline.com" in BRAND_DOMAINS["microsoft"]


# --- 1. findings ----------------------------------------------------------


def test_a_branded_anchor_pointing_off_brand_is_a_finding() -> None:
    signal = analyze_domain_mismatch_brand(email(link(PHISH, "Sign in to Microsoft")))

    assert signal.layer is DetectionLayer.L2
    assert signal.name == "domain_mismatch_brand"
    assert signal.score == BRAND_MISMATCH_SCORE
    assert signal.severity is RiskLevel.MEDIUM
    assert signal.error is None
    assert signal.metadata["fired"] is True

    match = signal.metadata["matches"][0]
    assert match["brands"] == ["microsoft"]
    assert match["destination"] == "secure-login.tk"
    assert match["anchor_text"] == "Sign in to Microsoft"
    # Evidence must establish the brand claimed, the domain reached, and why.
    assert "microsoft" in signal.evidence
    assert "secure-login.tk" in signal.evidence
    assert PHISH in signal.evidence


def test_a_lookalike_subdomain_does_not_launder_the_destination() -> None:
    """`microsoft.com.evil.example` is registrable-domain `evil.example`."""
    url = "https://microsoft.com.login.evil-phish.com/verify"
    signal = analyze_domain_mismatch_brand(email(link(url, "Microsoft 365")))

    assert signal.score == BRAND_MISMATCH_SCORE
    assert signal.metadata["matches"][0]["destination"] == "evil-phish.com"


# --- 2 & 4. legitimate destinations ---------------------------------------


def test_a_branded_anchor_pointing_at_the_brands_domain_is_clean() -> None:
    signal = analyze_domain_mismatch_brand(
        email(link("https://login.microsoftonline.com/", "Sign in to Microsoft"))
    )

    assert signal.score == 0.0
    assert signal.error is None
    assert signal.metadata["matches"] == []
    assert signal.metadata["legitimate"][0]["matched_brand"] == "microsoft"


def test_a_subdomain_of_a_listed_brand_domain_is_clean() -> None:
    """Registrable-domain comparison, so any subdomain of a listed domain passes."""
    signal = analyze_domain_mismatch_brand(
        email(link("https://account.live.com/settings", "Your Microsoft account"))
    )

    assert signal.score == 0.0
    assert signal.metadata["legitimate"]


def test_an_unlisted_domain_carrying_the_brand_token_degrades_to_silence() -> None:
    """The Layer 1 rule: a plausibly legitimate unlisted domain is not accused."""
    signal = analyze_domain_mismatch_brand(
        email(link("https://microsoft-partner.co.uk/portal", "Microsoft Partner"))
    )

    assert signal.score == 0.0
    assert signal.metadata["matches"] == []
    assert signal.metadata["legitimate"]


# --- 3. case and token normalization --------------------------------------


@pytest.mark.parametrize(
    "anchor", ["MICROSOFT", "microsoft", "MiCrOsOfT", "  Microsoft  ", "Microsoft-365"]
)
def test_brand_matching_is_case_and_punctuation_insensitive(anchor: str) -> None:
    assert brands_named_in(anchor) == ["microsoft"]


def test_split_spellings_are_caught() -> None:
    """"Pay Pal" and "Micro Soft" are the spellings attackers actually send."""
    assert brands_named_in("Pay Pal Security") == ["paypal"]
    assert brands_named_in("Micro Soft Team") == ["microsoft"]


def test_the_destination_domain_comparison_is_case_insensitive() -> None:
    signal = analyze_domain_mismatch_brand(
        email(link("HTTPS://LOGIN.MICROSOFTONLINE.COM/x", "Microsoft"))
    )

    assert signal.score == 0.0


# --- 5, 6, 7. anchors that make no claim ----------------------------------


def test_missing_anchor_text_is_clean() -> None:
    signal = analyze_domain_mismatch_brand(email(link(PHISH, None)))

    assert signal.score == 0.0
    assert signal.error is None
    assert signal.metadata["branded_anchors"] == 0
    assert "No link text in this message names a known brand" in signal.evidence


@pytest.mark.parametrize("anchor", ["", "   ", "\n\t"])
def test_empty_anchor_text_is_clean(anchor: str) -> None:
    signal = analyze_domain_mismatch_brand(email(link(PHISH, anchor)))

    assert signal.score == 0.0
    assert signal.error is None
    assert brands_named_in(anchor) == []


@pytest.mark.parametrize(
    "anchor",
    ["Click here", "View your invoice", "https://secure-login.tk/verify", "Unsubscribe"],
)
def test_unrelated_anchor_text_is_clean(anchor: str) -> None:
    signal = analyze_domain_mismatch_brand(email(link(PHISH, anchor)))

    assert signal.score == 0.0
    assert signal.metadata["matches"] == []


def test_a_brand_name_inside_a_longer_word_does_not_match() -> None:
    """Exact tokens only - no substring search, no edit distance."""
    assert brands_named_in("Microsoftware Solutions") == []
    assert brands_named_in("Applesauce Recipes") == []


def test_two_unrelated_words_are_not_joined_into_a_brand() -> None:
    """Only adjacent pairs join, so distant words cannot collude."""
    assert brands_named_in("pay your bill at the pal cafe") == []


# --- the boundary in the existing brand data ------------------------------


def test_an_alias_brand_resolves_to_its_own_legitimate_set() -> None:
    """`onedrive` is its own key with its own domains in the L1 table."""
    assert brands_named_in("OneDrive") == ["onedrive"]

    clean = analyze_domain_mismatch_brand(
        email(link("https://onedrive.live.com/x", "OneDrive"))
    )
    dirty = analyze_domain_mismatch_brand(email(link(PHISH, "OneDrive")))

    assert clean.score == 0.0
    assert dirty.score == BRAND_MISMATCH_SCORE


def test_a_brand_not_in_the_corpus_is_not_matched() -> None:
    assert "spotify" not in BRAND_DOMAINS
    assert brands_named_in("Spotify Premium") == []


def test_an_anchor_naming_two_brands_reports_both() -> None:
    signal = analyze_domain_mismatch_brand(
        email(link(PHISH, "Microsoft and PayPal billing"))
    )

    assert signal.metadata["matches"][0]["brands"] == ["microsoft", "paypal"]


def test_a_destination_legitimate_for_the_second_named_brand_is_clean() -> None:
    signal = analyze_domain_mismatch_brand(
        email(link("https://www.paypal.com/uk", "Microsoft and PayPal billing"))
    )

    assert signal.score == 0.0
    assert signal.metadata["legitimate"][0]["matched_brand"] == "paypal"


# --- 8, 9, 10. several URLs, duplicates, malformed ------------------------


def test_multiple_urls_are_judged_independently() -> None:
    signal = analyze_domain_mismatch_brand(
        email(
            link("https://login.microsoftonline.com/", "Sign in to Microsoft"),
            link(PHISH, "Your PayPal account"),
            link("https://example.com/x", "Click here"),
        )
    )

    assert signal.score == BRAND_MISMATCH_SCORE
    assert [m["brands"] for m in signal.metadata["matches"]] == [["paypal"]]
    assert len(signal.metadata["legitimate"]) == 1
    assert signal.metadata["urls_checked"] == 3


def test_duplicate_canonical_urls_are_judged_once() -> None:
    message = email(
        link(PHISH, "Sign in to Microsoft"),
        link("HTTPS://Secure-Login.tk:443/verify", "Sign in to Microsoft"),
        link(PHISH, "Sign in to Microsoft"),
    )

    signal = analyze_domain_mismatch_brand(message)

    assert signal.metadata["urls_in_message"] == 3
    assert signal.metadata["urls_checked"] == 1
    assert len(signal.metadata["matches"]) == 1


def test_a_malformed_url_does_not_stop_the_others() -> None:
    signal = analyze_domain_mismatch_brand(
        email(
            link("javascript:alert(1)", "Microsoft"),
            link("https://example.com:99999/x", "Microsoft"),
            link(PHISH, "Sign in to Microsoft"),
        )
    )

    assert signal.score == BRAND_MISMATCH_SCORE
    assert signal.metadata["urls_in_message"] == 3
    assert signal.metadata["urls_checked"] == 1


def test_a_branded_link_to_an_ip_literal_abstains() -> None:
    """No registrable domain to compare: unanswerable, not clean."""
    signal = analyze_domain_mismatch_brand(
        email(link("http://192.0.2.10/login", "Sign in to Microsoft"))
    )

    assert signal.score == 0.0
    assert signal.metadata["fired"] is False
    assert signal.error is not None
    assert "absence of information, not a clean result" in signal.evidence
    assert signal.metadata["uncomparable"][0]["destination"] is None


def test_a_finding_elsewhere_outranks_an_uncomparable_link() -> None:
    signal = analyze_domain_mismatch_brand(
        email(
            link("http://192.0.2.10/login", "Microsoft"),
            link(PHISH, "Your PayPal account"),
        )
    )

    assert signal.score == BRAND_MISMATCH_SCORE
    assert signal.error is None
    assert signal.metadata["uncomparable"]


def test_a_message_with_no_urls_is_a_genuine_negative() -> None:
    signal = analyze_domain_mismatch_brand(email())

    assert signal.score == 0.0
    assert signal.error is None
    assert "no analysable URLs" in signal.evidence


# --- 11, 12, 14. evidence, determinism, purity ----------------------------


def test_the_raw_url_is_quoted_not_the_canonical_form() -> None:
    messy = "HTTPS://Secure-Login.tk:443/Verify?session=SECRETTOKEN"
    signal = analyze_domain_mismatch_brand(email(link(messy, "Microsoft")))

    assert messy in signal.evidence
    assert signal.metadata["matches"][0]["url"] == messy


def test_evidence_and_metadata_expose_no_unrelated_url_data() -> None:
    url = "https://secure-login.tk/verify?session=SUPERSECRET&u=other"
    signal = analyze_domain_mismatch_brand(email(link(url, "Microsoft")))

    match = signal.metadata["matches"][0]
    assert set(match) == {"url", "anchor_text", "brands", "destination"}
    assert set(signal.metadata) == {
        "urls_in_message",
        "urls_checked",
        "branded_anchors",
        "matches",
        "legitimate",
        "uncomparable",
        "fired",
    }


def test_repeated_analysis_is_deterministic() -> None:
    message = email(
        link(PHISH, "Sign in to Microsoft"),
        link("https://other.tk/x", "Your PayPal account"),
    )

    results = {
        (
            analyze_domain_mismatch_brand(message).score,
            analyze_domain_mismatch_brand(message).evidence,
        )
        for _ in range(5)
    }

    assert len(results) == 1


def test_the_parsed_message_is_not_mutated() -> None:
    message = email(link(PHISH, "Sign in to Microsoft"))

    analyze_domain_mismatch_brand(message)

    assert message.urls[0].anchor_text == "Sign in to Microsoft"
    assert message.urls[0].redirect_chain == []
    assert message.urls[0].final_url is None


def test_a_qr_decoded_url_carries_no_anchor_and_is_clean() -> None:
    decoded = [ExtractedURL(url=PHISH, source=URLSource.QR_CODE)]

    signal = analyze_domain_mismatch_brand(email(), extra=decoded)

    assert signal.score == 0.0
    assert signal.metadata["branded_anchors"] == 0
