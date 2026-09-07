"""Unit tests for the l4.hosting_flag signal in layers/l4_intel.py.

Every ASN and every hosting-list entry here is a **fabricated placeholder** in
the AS64496-AS64511 range RFC 5398 reserves for documentation. No real network
operator is named, no real dataset is shipped, and the repository carries no
bulletproof-hosting list - each test writes a temporary one.

Offline and deterministic: the resolver is an injected fake, so no test opens a
socket, resolves a name, or contacts an analysed host.
"""

from __future__ import annotations

import json

import pytest

from core.models import DetectionLayer, ExtractedURL, ParsedEmail, RiskLevel, URLSource
from layers.l4_intel import (
    BULLETPROOF_LIST_PATH,
    HOSTING_FLAG_SCORE,
    HostingDataUnavailable,
    analyze_hosting_flag,
    default_asn_lookup,
    hosts_of,
    load_hosting_list,
)
from layers.threat_intel import ASNInfo, ASNUnavailable

# RFC 5398 documentation ASNs. Placeholders, not accusations.
FLAGGED_ASN = 64500
CLEAN_ASN = 64501

BAD_HOST = "login.evil-phish.com"
GOOD_HOST = "www.example.com"


@pytest.fixture
def list_file(tmp_path):
    """Write a temporary hosting list and return its path."""

    def write(entries=None, *, payload=None, name="bulletproof_asns.json"):
        path = tmp_path / name
        if payload is None:
            payload = {
                "source": "test fixture",
                "version": "2026-09-01",
                "entries": entries
                if entries is not None
                else [{"asn": FLAGGED_ASN, "name": "Placeholder BP", "reason": "fixture"}],
            }
        path.write_text(
            payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8"
        )
        return path

    return write


class FakeResolver:
    """Deterministic host -> ASN map. Records every host it was asked about."""

    def __init__(self, mapping: dict[str, int], *, names: dict[int, str] | None = None) -> None:
        self.mapping = mapping
        self.names = names or {FLAGGED_ASN: "Placeholder BP", CLEAN_ASN: "Placeholder Clean"}
        self.calls: list[str] = []

    def asn_for(self, host: str) -> ASNInfo:
        self.calls.append(host)
        if host not in self.mapping:
            raise ASNUnavailable(f"no route data for {host}")
        asn = self.mapping[host]
        return ASNInfo(asn=asn, name=self.names.get(asn), prefix="192.0.2.0/24")


def email(*urls: str) -> ParsedEmail:
    return ParsedEmail(
        message_id="<m@example.com>",
        from_addr="sender@example.com",
        subject="Invoice",
        body_text="See attached.",
        urls=[ExtractedURL(url=u, source=URLSource.ANCHOR_HREF) for u in urls],
    )


# --- the repository ships no dataset --------------------------------------


def test_no_bulletproof_list_is_committed_to_the_repository() -> None:
    """Inventing one would mean accusing whoever holds the invented ASNs."""
    assert not BULLETPROOF_LIST_PATH.exists()


def test_there_is_no_default_asn_resolver() -> None:
    assert default_asn_lookup() is None


# --- findings -------------------------------------------------------------


def test_a_host_on_a_flagged_asn_is_a_finding(list_file) -> None:
    resolver = FakeResolver({BAD_HOST: FLAGGED_ASN})

    signal = analyze_hosting_flag(
        email(f"https://{BAD_HOST}/x"),
        asn_lookup=resolver,
        hosting_list_path=list_file(),
    )

    assert signal.layer is DetectionLayer.L4
    assert signal.name == "hosting_flag"
    assert signal.score == HOSTING_FLAG_SCORE
    assert signal.severity is RiskLevel.MEDIUM
    assert signal.error is None
    assert signal.metadata["fired"] is True
    assert signal.metadata["matches"] == [
        {
            "host": BAD_HOST,
            "asn": FLAGGED_ASN,
            "network": "Placeholder BP",
            "entry": f"AS{FLAGGED_ASN} Placeholder BP",
            "reason": "fixture",
        }
    ]
    assert BAD_HOST in signal.evidence
    assert f"AS{FLAGGED_ASN}" in signal.evidence
    assert "test fixture" in signal.evidence


def test_the_hosting_score_is_the_weakest_layer_four_signal() -> None:
    from layers.l4_intel import DOMAIN_IOC_SCORE, URL_IOC_SCORE

    assert HOSTING_FLAG_SCORE < DOMAIN_IOC_SCORE < URL_IOC_SCORE


def test_a_host_on_an_unflagged_asn_is_clean(list_file) -> None:
    resolver = FakeResolver({GOOD_HOST: CLEAN_ASN})

    signal = analyze_hosting_flag(
        email(f"https://{GOOD_HOST}/x"),
        asn_lookup=resolver,
        hosting_list_path=list_file(),
    )

    assert signal.score == 0.0
    assert signal.error is None
    assert signal.metadata["matches"] == []
    assert signal.metadata["hosts_checked"] == [
        {"host": GOOD_HOST, "asn": CLEAN_ASN, "network": "Placeholder Clean"}
    ]
    assert signal.metadata["list_version"] == "2026-09-01"


def test_multiple_hosts_are_each_resolved(list_file) -> None:
    resolver = FakeResolver(
        {BAD_HOST: FLAGGED_ASN, GOOD_HOST: CLEAN_ASN, "cdn.example.net": CLEAN_ASN}
    )

    signal = analyze_hosting_flag(
        email(f"https://{BAD_HOST}/a", f"https://{GOOD_HOST}/b", "https://cdn.example.net/c"),
        asn_lookup=resolver,
        hosting_list_path=list_file(),
    )

    assert resolver.calls == [BAD_HOST, GOOD_HOST, "cdn.example.net"]
    assert signal.metadata["hosts_in_message"] == 3
    assert len(signal.metadata["matches"]) == 1


def test_duplicate_hosts_are_resolved_once(list_file) -> None:
    resolver = FakeResolver({BAD_HOST: FLAGGED_ASN})

    analyze_hosting_flag(
        email(f"https://{BAD_HOST}/a", f"https://{BAD_HOST}/b", f"HTTPS://{BAD_HOST.upper()}/c"),
        asn_lookup=resolver,
        hosting_list_path=list_file(),
    )

    assert resolver.calls == [BAD_HOST]


def test_hosts_are_not_collapsed_to_registrable_domains() -> None:
    """Different hosts can sit on different networks, so each is its own question."""
    assert hosts_of(email("https://a.example.com/x", "https://b.example.com/y")) == [
        "a.example.com",
        "b.example.com",
    ]


# --- resolver unavailable -------------------------------------------------


def test_an_unresolvable_host_abstains_rather_than_reading_as_clean(list_file) -> None:
    resolver = FakeResolver({})  # knows nothing

    signal = analyze_hosting_flag(
        email(f"https://{GOOD_HOST}/x"),
        asn_lookup=resolver,
        hosting_list_path=list_file(),
    )

    assert signal.score == 0.0
    assert signal.metadata["fired"] is False
    assert signal.error is not None and "ASN unavailable" in signal.error
    assert GOOD_HOST in signal.metadata["hosts_unresolved"]
    assert "absence of information, not a clean result" in signal.evidence


def test_a_partial_resolution_failure_still_abstains(list_file) -> None:
    """A clean verdict may not be issued over hosts nobody looked at."""
    resolver = FakeResolver({GOOD_HOST: CLEAN_ASN})

    signal = analyze_hosting_flag(
        email(f"https://{GOOD_HOST}/a", "https://unknown.example.org/b"),
        asn_lookup=resolver,
        hosting_list_path=list_file(),
    )

    assert signal.error is not None
    assert "unknown.example.org" in signal.metadata["hosts_unresolved"]


def test_a_finding_survives_an_unresolvable_sibling_host(list_file) -> None:
    resolver = FakeResolver({BAD_HOST: FLAGGED_ASN})

    signal = analyze_hosting_flag(
        email(f"https://{BAD_HOST}/a", "https://unknown.example.org/b"),
        asn_lookup=resolver,
        hosting_list_path=list_file(),
    )

    assert signal.score == HOSTING_FLAG_SCORE
    assert signal.error is None
    assert "unknown.example.org" in signal.metadata["hosts_unresolved"]


def test_no_resolver_configured_abstains(list_file) -> None:
    signal = analyze_hosting_flag(
        email(f"https://{GOOD_HOST}/x"), hosting_list_path=list_file()
    )

    assert signal.score == 0.0
    assert signal.error == "no ASN resolver configured"
    assert "not a clean result" in signal.evidence


def test_a_resolver_raising_an_unexpected_error_abstains(list_file) -> None:
    class Broken:
        def asn_for(self, host):
            raise ValueError("routing table on fire")

    signal = analyze_hosting_flag(
        email(f"https://{GOOD_HOST}/x"), asn_lookup=Broken(), hosting_list_path=list_file()
    )

    assert signal.score == 0.0
    assert signal.error is not None
    assert "ValueError" in signal.metadata["hosts_unresolved"][GOOD_HOST]


def test_a_resolver_returning_the_wrong_type_abstains(list_file) -> None:
    class Wrong:
        def asn_for(self, host):
            return 64500  # an int, not an ASNInfo

    signal = analyze_hosting_flag(
        email(f"https://{GOOD_HOST}/x"), asn_lookup=Wrong(), hosting_list_path=list_file()
    )

    assert signal.error is not None
    assert "not an ASNInfo" in signal.metadata["hosts_unresolved"][GOOD_HOST]


# --- dataset problems -----------------------------------------------------


def test_a_missing_dataset_abstains(tmp_path) -> None:
    signal = analyze_hosting_flag(
        email(f"https://{BAD_HOST}/x"),
        asn_lookup=FakeResolver({BAD_HOST: FLAGGED_ASN}),
        hosting_list_path=tmp_path / "absent.json",
    )

    assert signal.score == 0.0
    assert signal.metadata["fired"] is False
    assert signal.error is not None and "absent.json" in signal.error
    assert "absence of information, not a clean result" in signal.evidence


def test_a_missing_dataset_abstains_even_with_no_urls(tmp_path) -> None:
    """Requirement 15: no URLs is a genuine negative only if the list exists."""
    signal = analyze_hosting_flag(email(), hosting_list_path=tmp_path / "absent.json")

    assert signal.error is not None


@pytest.mark.parametrize(
    "payload",
    [
        "{",                                      # not JSON
        json.dumps([]),                           # not an object
        json.dumps({"version": "1"}),             # no entries array
        json.dumps({"entries": "nope"}),          # entries not a list
        json.dumps({"entries": []}),              # empty: a failed download, not a claim
        json.dumps({"entries": [{"name": "no asn"}]}),   # no usable rows
        json.dumps({"entries": [{"asn": "64500"}]}),     # asn not an int
        json.dumps({"entries": [{"asn": True}]}),        # bool is not an ASN
    ],
)
def test_a_malformed_dataset_abstains(list_file, payload: str) -> None:
    path = list_file(payload=payload)

    with pytest.raises(HostingDataUnavailable):
        load_hosting_list(path)

    signal = analyze_hosting_flag(
        email(f"https://{BAD_HOST}/x"),
        asn_lookup=FakeResolver({BAD_HOST: FLAGGED_ASN}),
        hosting_list_path=path,
    )
    assert signal.score == 0.0
    assert signal.error is not None


def test_a_malformed_row_is_skipped_but_a_valid_one_still_loads(list_file) -> None:
    path = list_file(entries=[{"asn": "bad"}, {"asn": FLAGGED_ASN, "name": "Placeholder BP"}])
    listing = load_hosting_list(path)

    assert list(listing.entries) == [FLAGGED_ASN]


# --- empty message --------------------------------------------------------


def test_no_urls_is_a_genuine_negative_when_the_list_exists(list_file) -> None:
    resolver = FakeResolver({})

    signal = analyze_hosting_flag(
        email(), asn_lookup=resolver, hosting_list_path=list_file()
    )

    assert signal.score == 0.0
    assert signal.error is None          # a negative, not an abstention
    assert signal.metadata["hosts_in_message"] == 0
    assert resolver.calls == []


def test_a_url_with_no_host_yields_nothing_to_check(list_file) -> None:
    signal = analyze_hosting_flag(
        email("mailto:someone@example.com"),
        asn_lookup=FakeResolver({}),
        hosting_list_path=list_file(),
    )

    assert signal.error is None or "no ASN" not in signal.error


# --- determinism and no network -------------------------------------------


def test_repeated_analysis_is_deterministic(list_file) -> None:
    path = list_file()
    message = email(f"https://{BAD_HOST}/a", f"https://{GOOD_HOST}/b")
    resolver = FakeResolver({BAD_HOST: FLAGGED_ASN, GOOD_HOST: CLEAN_ASN})

    seen = {
        (
            analyze_hosting_flag(message, asn_lookup=resolver, hosting_list_path=path).score,
            json.dumps(
                analyze_hosting_flag(
                    message, asn_lookup=resolver, hosting_list_path=path
                ).metadata["matches"],
                sort_keys=True,
            ),
        )
        for _ in range(5)
    }

    assert len(seen) == 1


def test_analysis_opens_no_socket_and_resolves_no_name(list_file, monkeypatch) -> None:
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("layer 4 attempted network access")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket, "gethostbyname", forbidden)

    signal = analyze_hosting_flag(
        email(f"https://{BAD_HOST}/x"),
        asn_lookup=FakeResolver({BAD_HOST: FLAGGED_ASN}),
        hosting_list_path=list_file(),
    )

    assert signal.score == HOSTING_FLAG_SCORE


def test_the_host_is_handed_to_the_resolver_as_data(list_file) -> None:
    resolver = FakeResolver({BAD_HOST: FLAGGED_ASN})

    analyze_hosting_flag(
        email(f"https://{BAD_HOST}/deep/path?q=1#f"),
        asn_lookup=resolver,
        hosting_list_path=list_file(),
    )

    assert resolver.calls == [BAD_HOST]  # the host alone, never the full URL
