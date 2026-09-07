"""Unit tests for the Layer 2 foundation in layers/l2_urls.py.

Parsing, canonicalization and candidate assembly only - no signal exists yet.
Everything here is pure string work, so no test opens a socket, resolves a
name, follows a redirect or renders anything.
"""

from __future__ import annotations

import pytest

from core.models import ExtractedURL, ParsedEmail, URLSource
from layers.l2_urls import (
    DEFAULT_PORTS,
    URLCandidate,
    candidate_urls,
    canonical_form,
    parse_url,
)


def email(*urls: ExtractedURL) -> ParsedEmail:
    return ParsedEmail(
        message_id="<m@example.com>",
        from_addr="sender@example.com",
        subject="Invoice",
        body_text="See the portal.",
        urls=list(urls),
    )


def link(url: str, source: URLSource = URLSource.ANCHOR_HREF, anchor: str | None = None):
    return ExtractedURL(url=url, source=source, anchor_text=anchor)


# --- parse_url ------------------------------------------------------------


def test_parse_splits_a_url_into_the_parts_the_signals_need() -> None:
    parsed = parse_url("HTTPS://Login.Evil-Phish.com:8443/verify?token=abc#frag")

    assert parsed == ("https", "login.evil-phish.com", 8443, "/verify", "token=abc", "frag")


def test_parse_lowercases_the_host_but_not_the_path() -> None:
    _, host, _, path, _, _ = parse_url("https://EXAMPLE.com/Case/Sensitive")

    assert host == "example.com"
    assert path == "/Case/Sensitive"


def test_parse_strips_a_trailing_dot_on_the_host() -> None:
    assert parse_url("https://example.com./x")[1] == "example.com"


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "mailto:someone@example.com",
        "cid:image001.png",
        "data:text/html;base64,PHNjcmlwdD4=",
        "javascript:alert(1)",
        "ftp://files.example.com/x",
        "/relative/path",
        "//scheme-relative.example/x",
        "https://",
        "http:///nohost",
        "https://example.com:99999/x",
        "https://example.com:notaport/x",
        "https://[unterminated/x",
    ],
)
def test_unparseable_input_returns_none_rather_than_raising(value: str) -> None:
    assert parse_url(value) is None


def test_a_non_string_is_not_a_url() -> None:
    assert parse_url(None) is None  # type: ignore[arg-type]
    assert parse_url(12345) is None  # type: ignore[arg-type]


# --- canonical_form -------------------------------------------------------


@pytest.mark.parametrize(
    "left, right",
    [
        ("https://Example.com/x", "https://example.com/x"),
        ("https://example.com:443/x", "https://example.com/x"),
        ("http://example.com:80/x", "http://example.com/x"),
        ("https://example.com.", "https://example.com/"),
        ("https://example.com", "https://example.com/"),
        ("  https://example.com/x  ", "https://example.com/x"),
    ],
)
def test_equivalent_spellings_share_a_canonical_form(left: str, right: str) -> None:
    assert canonical_form(left) == canonical_form(right)


@pytest.mark.parametrize(
    "left, right",
    [
        ("https://example.com/A", "https://example.com/a"),          # path case matters
        ("https://example.com/x?b=1&a=2", "https://example.com/x?a=2&b=1"),  # order kept
        ("https://example.com/x#one", "https://example.com/x#two"),  # fragment kept
        ("https://example.com/x", "http://example.com/x"),           # scheme matters
        ("https://example.com:8443/x", "https://example.com/x"),     # non-default port kept
    ],
)
def test_different_urls_keep_different_canonical_forms(left: str, right: str) -> None:
    assert canonical_form(left) != canonical_form(right)


def test_percent_encoding_is_left_alone() -> None:
    """Unescaping could change which resource the URL names."""
    assert canonical_form("https://example.com/a%2Fb") == "https://example.com/a%2Fb"


def test_canonical_form_of_an_unparseable_url_is_none() -> None:
    assert canonical_form("javascript:alert(1)") is None


def test_the_default_port_table_covers_both_analysable_schemes() -> None:
    assert DEFAULT_PORTS == {"http": 80, "https": 443}


# --- candidate_urls -------------------------------------------------------


def test_a_candidate_carries_the_parts_and_the_provenance() -> None:
    message = email(
        link("https://login.evil-phish.com/verify?u=dGVzdA%3D%3D#f=1", anchor="Acme Support")
    )

    (candidate,) = candidate_urls(message)

    assert isinstance(candidate, URLCandidate)
    assert candidate.raw == "https://login.evil-phish.com/verify?u=dGVzdA%3D%3D#f=1"
    assert candidate.host == "login.evil-phish.com"
    assert candidate.registrable == "evil-phish.com"
    assert candidate.path == "/verify"
    assert candidate.query_params == (("u", "dGVzdA=="),)
    assert candidate.fragment_params == (("f", "1"),)
    assert candidate.source is URLSource.ANCHOR_HREF
    assert candidate.anchor_text == "Acme Support"


def test_the_raw_string_is_preserved_for_evidence() -> None:
    """Evidence must quote what the message contained, not a rewritten form."""
    (candidate,) = candidate_urls(email(link("HTTPS://Example.com:443/X")))

    assert candidate.raw == "HTTPS://Example.com:443/X"
    assert candidate.canonical == "https://example.com/X"


def test_a_blank_query_value_is_visible_rather_than_dropped() -> None:
    (candidate,) = candidate_urls(email(link("https://example.com/x?redirect=")))

    assert candidate.query_params == (("redirect", ""),)


def test_a_fragment_that_is_not_key_value_yields_no_params() -> None:
    (candidate,) = candidate_urls(email(link("https://example.com/x#section-2")))

    assert candidate.fragment == "section-2"
    assert candidate.fragment_params == ()


def test_multiple_urls_keep_first_seen_order() -> None:
    message = email(
        link("https://a.example/1"), link("https://b.example/2"), link("https://c.example/3")
    )

    assert [c.host for c in candidate_urls(message)] == ["a.example", "b.example", "c.example"]


def test_the_same_place_seen_three_ways_is_one_candidate() -> None:
    message = email(
        link("https://example.com/x", URLSource.ANCHOR_HREF, anchor="Click here"),
        link("https://EXAMPLE.com:443/x", URLSource.PLAIN_TEXT),
        link("https://example.com/x", URLSource.IMG_SRC),
    )

    candidates = candidate_urls(message)

    assert len(candidates) == 1
    # The first observation wins, so its provenance and anchor text survive.
    assert candidates[0].source is URLSource.ANCHOR_HREF
    assert candidates[0].anchor_text == "Click here"


def test_different_paths_on_one_host_are_separate_candidates() -> None:
    message = email(link("https://example.com/a"), link("https://example.com/b"))

    assert len(candidate_urls(message)) == 2


def test_unparseable_urls_are_dropped_not_carried_half_understood() -> None:
    message = email(
        link("https://good.example/x"),
        link("https://example.com:99999/x"),
        link("javascript:alert(1)"),
    )

    candidates = candidate_urls(message)

    assert [c.host for c in candidates] == ["good.example"]


def test_a_message_with_no_urls_yields_no_candidates() -> None:
    assert candidate_urls(email()) == []


def test_an_ip_literal_host_is_distinguishable_from_a_broken_one() -> None:
    (candidate,) = candidate_urls(email(link("http://192.0.2.10/login")))

    assert candidate.registrable is None
    assert candidate.is_ip_literal is True


def test_a_multi_part_public_suffix_uses_the_vendored_list() -> None:
    """registrable_domain is reused, not reimplemented."""
    (candidate,) = candidate_urls(email(link("https://shop.example.co.uk/x")))

    assert candidate.registrable == "example.co.uk"


# --- the QR re-entry seam -------------------------------------------------


def test_decoded_urls_re_enter_the_candidate_set() -> None:
    """ARCHITECTURE.md section 4: a QR-decoded URL re-enters the URL signal set."""
    message = email(link("https://a.example/1"))
    decoded = [ExtractedURL(url="https://evil-phish.com/qr", source=URLSource.QR_CODE)]

    candidates = candidate_urls(message, extra=decoded)

    assert [c.host for c in candidates] == ["a.example", "evil-phish.com"]
    assert candidates[1].source is URLSource.QR_CODE


def test_a_decoded_url_already_in_the_body_is_not_duplicated() -> None:
    message = email(link("https://evil-phish.com/qr"))
    decoded = [ExtractedURL(url="https://evil-phish.com/qr", source=URLSource.QR_CODE)]

    candidates = candidate_urls(message, extra=decoded)

    assert len(candidates) == 1
    assert candidates[0].source is URLSource.ANCHOR_HREF  # the body observation came first


def test_a_decoded_string_that_is_not_a_url_is_dropped() -> None:
    """A QR code carries arbitrary text and has had no upstream filtering."""
    decoded = [ExtractedURL(url="WIFI:S=guest;T=WPA;", source=URLSource.QR_CODE)]

    assert candidate_urls(email(), extra=decoded) == []


# --- offline --------------------------------------------------------------


def test_the_foundation_opens_no_socket_and_resolves_no_name(monkeypatch) -> None:
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("layer 2 foundation attempted network access")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket, "gethostbyname", forbidden)

    candidates = candidate_urls(
        email(link("https://login.evil-phish.com/verify?u=1"), link("http://192.0.2.10/x"))
    )

    assert len(candidates) == 2


def test_candidates_are_frozen_records() -> None:
    (candidate,) = candidate_urls(email(link("https://example.com/x")))

    with pytest.raises(Exception):
        candidate.host = "changed"  # type: ignore[misc]


def test_repeated_assembly_is_deterministic() -> None:
    message = email(link("https://a.example/1"), link("https://b.example/2"))

    assert [c.canonical for c in candidate_urls(message)] == [
        c.canonical for c in candidate_urls(message)
    ]
