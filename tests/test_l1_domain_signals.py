"""Unit tests for the Layer 1 domain signals in layers/l1_headers.py.

Covers Reply-To vs From, WHOIS domain age, and display-name brand
impersonation.

Deterministic and offline. `tldextract` is configured in the module under test
with its bundled suffix-list snapshot and never fetches, WHOIS is injected as a
fake, and `now` is passed explicitly - so no test opens a socket and no test
depends on today's date.
"""

from __future__ import annotations

import sys
import types
from datetime import datetime, timedelta, timezone

import pytest

from core.models import DetectionLayer, ParsedEmail, RiskLevel
from ingest.parser import parse_email
from layers import l1_headers
from layers.l1_headers import (
    PUBLIC_SUFFIX_LIST_PATH,
    WHOIS_NEGATIVE_TTL_SECONDS,
    WHOIS_TTL_SECONDS,
    DomainAge,
    PythonWhoisLookup,
    WhoisUnavailable,
    analyze,
    analyze_display_name_impersonation,
    analyze_domain_age,
    analyze_reply_to_mismatch,
    registrable_domain,
)
from storage.cache import Cache

NOW = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)


def email(
    from_header: str = "Sender <sender@example.com>",
    *,
    reply_to: str | None = None,
) -> ParsedEmail:
    """A parsed message with the given From and optional Reply-To."""
    headers = f"From: {from_header}\n"
    if reply_to is not None:
        headers += f"Reply-To: {reply_to}\n"
    raw = (
        headers.encode()
        + b"To: john@example.com\n"
        b"Subject: Test message\n"
        b"Message-ID: <l1-domain-test@example.invalid>\n"
        b'Content-Type: text/plain; charset="utf-8"\n'
        b"\n"
        b"Body text.\n"
    )
    return parse_email(raw)


# --------------------------------------------------------------------------
# registrable_domain
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("sender@example.com", "example.com"),
        ("sender@mail.corp.co.uk", "corp.co.uk"),
        ("Sender <a@b.example.com>", "example.com"),
        ("example.com", "example.com"),
        ("MAIL.EXAMPLE.COM", "example.com"),
        ("a@sub.domain.github.io", "domain.github.io"),
    ],
)
def test_registrable_domain_uses_the_public_suffix_list(value, expected):
    assert registrable_domain(value) == expected


@pytest.mark.parametrize(
    "value", ["", "   ", None, "localhost", "a@localhost", "@", "a@", "not a domain",
              "a@192.168.1.1", 42, "a@@b"]
)
def test_registrable_domain_returns_none_when_there_is_no_domain(value):
    """None means "no domain to compare", never "domains differ"."""
    assert registrable_domain(value) is None


# --------------------------------------------------------------------------
# The vendored Public Suffix List
# --------------------------------------------------------------------------
#
# Regression cover for a real-world miss found by the 10-message Gmail
# validation: tldextract's *bundled* snapshot predates several delegated
# second-level `.in` namespaces, so every host under one of them collapsed to
# the namespace itself. `cx.federal.bank.in` reduced to `bank.in` - the
# namespace every Indian bank shares - which is both a wrong From domain to
# report and, worse, a domain two unrelated banks appear to have in common.
#
# These tests pin the *calculation*, not one bank: the vendored list is the
# input, and the delegated namespaces are read from it rather than special
# cased anywhere in the layer.


def test_the_vendored_public_suffix_list_ships_with_the_project():
    """The extractor's input is a file in the repo, not a library snapshot."""
    assert PUBLIC_SUFFIX_LIST_PATH.is_file()
    assert PUBLIC_SUFFIX_LIST_PATH.stat().st_size > 0


@pytest.mark.parametrize(
    "host,expected",
    [
        # Delegated second-level `.in` namespaces. None of these are special
        # cased in the layer; they come from the vendored suffix list.
        ("cx.federal.bank.in", "federal.bank.in"),
        ("somebank.bank.in", "somebank.bank.in"),
        ("mail.a-school.school.in", "a-school.school.in"),
        ("x.some-firm.fin.in", "some-firm.fin.in"),
        # The namespace itself is a public suffix, so there is nothing
        # registrable to compare - "unknown", never a bare match.
        ("bank.in", None),
    ],
)
def test_delegated_second_level_in_namespaces_are_not_collapsed(host, expected):
    assert registrable_domain(host) == expected
    assert registrable_domain(f"someone@{host}") == expected


@pytest.mark.parametrize(
    "host,expected",
    [
        ("example.com", "example.com"),
        ("sub.example.com", "example.com"),
        ("example.co.in", "example.co.in"),
        ("mail.example.co.in", "example.co.in"),
        ("corp.co.uk", "corp.co.uk"),
        ("attacker.github.io", "attacker.github.io"),
    ],
)
def test_the_vendored_list_preserves_the_established_splits(host, expected):
    """Refreshing the suffix list must not move any domain already handled."""
    assert registrable_domain(host) == expected


def test_a_missing_vendored_list_falls_back_instead_of_breaking_the_layer(
    monkeypatch, tmp_path
):
    """An absent file degrades to the library snapshot - the previous behaviour."""
    monkeypatch.setattr(
        l1_headers, "PUBLIC_SUFFIX_LIST_PATH", tmp_path / "absent.dat"
    )

    extractor = l1_headers._build_extractor()

    assert extractor("sub.example.com").top_domain_under_public_suffix == "example.com"


# --------------------------------------------------------------------------
# Reply-To vs From
# --------------------------------------------------------------------------


def test_matching_domains_do_not_fire():
    signal = analyze_reply_to_mismatch(
        email("a@example.com", reply_to="b@example.com")
    )

    assert signal.layer is DetectionLayer.L1
    assert signal.name == "replyto_mismatch"
    assert signal.metadata["fired"] is False
    assert signal.score == 0.0
    assert signal.error is None


def test_different_registrable_domains_fire():
    signal = analyze_reply_to_mismatch(
        email("Finance <ceo@corp.com>", reply_to="attacker@evil.xyz")
    )

    assert signal.metadata["fired"] is True
    assert signal.score > 0.0
    assert signal.severity is RiskLevel.MEDIUM
    assert signal.metadata["from_domain"] == "corp.com"
    assert signal.metadata["reply_to_domain"] == "evil.xyz"
    assert "evil.xyz" in signal.evidence and "corp.com" in signal.evidence


def test_subdomains_of_one_organization_do_not_fire():
    """billing.corp.com replying to corp.com is one organization, not a mismatch."""
    signal = analyze_reply_to_mismatch(
        email("a@billing.corp.com", reply_to="b@support.corp.com")
    )

    assert signal.metadata["fired"] is False
    assert signal.metadata["from_domain"] == "corp.com"


def test_a_multi_part_suffix_is_not_confused_with_a_different_domain():
    """A naive last-two-labels rule would call these both "co.uk"."""
    signal = analyze_reply_to_mismatch(
        email("a@corp.co.uk", reply_to="b@evil.co.uk")
    )

    assert signal.metadata["fired"] is True


def test_two_organizations_in_one_delegated_namespace_are_not_one_organization():
    """The false *negative* the stale suffix list caused.

    Under the bundled snapshot both sides reduced to `bank.in`, so a reply
    redirected from one bank to an entirely different one read as "one
    organization talking to itself" and the signal stayed silent.
    """
    signal = analyze_reply_to_mismatch(
        email("Alpha Bank <alerts@cx.alpha.bank.in>", reply_to="a@beta.bank.in")
    )

    assert signal.metadata["from_domain"] == "alpha.bank.in"
    assert signal.metadata["reply_to_domain"] == "beta.bank.in"
    assert signal.metadata["fired"] is True


def test_subdomains_within_one_delegated_namespace_registrant_do_not_fire():
    """`alpha.bank.in` is one registrant; its own subdomains are one org."""
    signal = analyze_reply_to_mismatch(
        email("a@cx.alpha.bank.in", reply_to="b@care.alpha.bank.in")
    )

    assert signal.metadata["from_domain"] == "alpha.bank.in"
    assert signal.metadata["reply_to_domain"] == "alpha.bank.in"
    assert signal.metadata["fired"] is False


def test_the_real_world_bank_in_case_reports_the_correct_from_domain():
    """The message from the Gmail validation, reduced to its two headers.

    The defect this pins is the *domain calculation*: `from_domain` must be the
    sender's own registrable domain under the delegated namespace, not the
    namespace itself.

    The signal still fires, and that is the layer's documented semantics rather
    than an oversight: a `.bank.in` registrant and a `.co.in` registrant are two
    registrable domains, and `analyze_reply_to_mismatch` compares exactly that.
    Recognising them as one organisation would need evidence this signal does
    not consult - DMARC alignment, say - and inventing it here is out of scope.
    """
    signal = analyze_reply_to_mismatch(
        email("Some Bank <no-reply@cx.somebank.bank.in>", reply_to="a@somebank.co.in")
    )

    assert signal.metadata["from_domain"] == "somebank.bank.in"
    assert signal.metadata["reply_to_domain"] == "somebank.co.in"
    assert signal.metadata["fired"] is True
    assert signal.score == pytest.approx(0.55)


def test_a_missing_reply_to_is_a_genuine_negative_not_an_abstention():
    signal = analyze_reply_to_mismatch(email("a@example.com"))

    assert signal.metadata["fired"] is False
    assert signal.error is None
    assert signal.metadata["reply_to_domain"] is None
    assert "No Reply-To" in signal.evidence


def test_an_empty_reply_to_header_is_treated_as_missing():
    signal = analyze_reply_to_mismatch(email("a@example.com", reply_to="   "))

    assert signal.metadata["fired"] is False
    assert signal.error is None


@pytest.mark.parametrize(
    "from_header,reply_to",
    [
        ("a@example.com", "not-an-address"),
        ("malformed", "b@example.com"),
        ("", "b@example.com"),
        ("a@example.com", "sender@localhost"),
        ("a@localhost", "b@example.com"),
    ],
)
def test_malformed_addresses_abstain_rather_than_crash_or_accuse(from_header, reply_to):
    signal = analyze_reply_to_mismatch(email(from_header, reply_to=reply_to))

    assert signal.metadata["fired"] is False
    assert signal.score == 0.0
    assert signal.error is not None  # could not compare, so it does not claim to have


# --------------------------------------------------------------------------
# WHOIS domain age
# --------------------------------------------------------------------------


class FakeWhois:
    """A WhoisLookup that answers from a table, and counts its calls."""

    def __init__(self, *, created=None, error=None):
        self._created = created
        self._error = error
        self.calls: list[str] = []

    def creation_date(self, domain):
        self.calls.append(domain)
        if self._error is not None:
            raise self._error
        return DomainAge(domain=domain, created_at=self._created)


@pytest.fixture
def cache(tmp_path):
    with Cache(tmp_path / "l1.sqlite") as c:
        yield c


# --------------------------------------------------------------------------
# PythonWhoisLookup: the client is held to the domain it was asked about
# --------------------------------------------------------------------------
#
# `python-whois` parses public suffixes with its own bundled list before it
# queries anything, and that list lags the vendored one. Real validation caught
# it answering a query for a bank's registrant under a delegated `.in`
# namespace with the *namespace's* 2005 registration rather than the
# registrant's 2025 one - a twenty-year-old creation date attached to a domain
# that could have been registered yesterday.
#
# Every test here injects a fake `whois` module into `sys.modules`, which is
# where `PythonWhoisLookup` imports it from. Nothing opens a socket.

# The real-world case, as data. It is the regression scenario, not production
# logic: nothing in layers/ knows these strings.
DELEGATED_CHILD = "federal.bank.in"
PARENT_NAMESPACE = "bank.in"
PARENT_CREATED = datetime(2005, 2, 17, 5, 28, 51, tzinfo=timezone.utc)
CHILD_CREATED = datetime(2025, 7, 18, 6, 12, 24, tzinfo=timezone.utc)


class FakeRecord(dict):
    """A stand-in for `whois.parser.WhoisEntry`, which is also a dict."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - mirrors the real class
            raise AttributeError(name) from exc


class FakeWhoisModule:
    """A fake `python-whois`, including its own suffix-list behaviour.

    `shortens_to` models the defect: a mapping from the domain asked for to the
    (shorter) domain this client's own suffix list would reduce it to. When
    `honours_seeding` is set - as the real client does, since seeding replaces
    the list its extraction reads - the reduction stops happening once
    `suffixes` has been populated from the vendored file.
    """

    def __init__(self, records, *, shortens_to=None, honours_seeding=True,
                 has_extract=True):
        self._records = records
        self._shortens_to = dict(shortens_to or {})
        self._honours_seeding = honours_seeding
        self.suffixes = None
        self.queried: list[str] = []
        self.exceptions = types.SimpleNamespace(
            WhoisDomainNotFoundError=_FakeNotFound
        )
        if not has_extract:
            # A client that does not expose the helper: only the record's own
            # identity can then give a mismatch away.
            self.extract_domain = None

    def _reduce(self, domain):
        if self.suffixes and self._honours_seeding:
            return domain
        return self._shortens_to.get(domain, domain)

    def extract_domain(self, domain):
        return self._reduce(domain)

    def whois(self, domain):
        looked_up = self._reduce(domain)
        self.queried.append(looked_up)
        try:
            return self._records[looked_up]
        except KeyError:
            raise _FakeNotFound(looked_up) from None


class _FakeNotFound(Exception):
    """Stands in for `whois.exceptions.WhoisDomainNotFoundError`."""


def install_whois(monkeypatch, module):
    monkeypatch.setitem(sys.modules, "whois", module)
    return module


def record(domain, created):
    return FakeRecord(domain_name=domain, creation_date=created)


def test_a_plain_com_domain_is_looked_up_and_returned(monkeypatch):
    module = install_whois(
        monkeypatch,
        FakeWhoisModule({"example.com": record("example.com", PARENT_CREATED)}),
    )

    age = PythonWhoisLookup().creation_date("example.com")

    assert module.queried == ["example.com"]
    assert age.domain == "example.com"
    assert age.created_at == PARENT_CREATED


def test_a_multi_label_suffix_domain_is_looked_up_and_returned(monkeypatch):
    """`example.co.in` is three labels too, and must keep working."""
    module = install_whois(
        monkeypatch,
        FakeWhoisModule({"example.co.in": record("EXAMPLE.CO.IN", PARENT_CREATED)}),
    )

    age = PythonWhoisLookup().creation_date("example.co.in")

    assert module.queried == ["example.co.in"]
    assert age.created_at == PARENT_CREATED


def test_the_client_is_seeded_so_the_delegated_child_is_the_domain_queried(
    monkeypatch,
):
    """The fix that keeps the signal working: query the right domain."""
    module = install_whois(
        monkeypatch,
        FakeWhoisModule(
            {
                DELEGATED_CHILD: record(DELEGATED_CHILD, CHILD_CREATED),
                PARENT_NAMESPACE: record(PARENT_NAMESPACE, PARENT_CREATED),
            },
            shortens_to={DELEGATED_CHILD: PARENT_NAMESPACE},
        ),
    )

    age = PythonWhoisLookup().creation_date(DELEGATED_CHILD)

    assert module.queried == [DELEGATED_CHILD]
    assert age.created_at == CHILD_CREATED
    assert age.created_at != PARENT_CREATED


def test_a_client_that_shortens_the_query_is_refused_before_it_is_called(
    monkeypatch,
):
    """Seeding did not take - an older or changed client. Abstain, do not guess."""
    module = install_whois(
        monkeypatch,
        FakeWhoisModule(
            {PARENT_NAMESPACE: record(PARENT_NAMESPACE, PARENT_CREATED)},
            shortens_to={DELEGATED_CHILD: PARENT_NAMESPACE},
            honours_seeding=False,
        ),
    )

    with pytest.raises(WhoisUnavailable):
        PythonWhoisLookup().creation_date(DELEGATED_CHILD)

    assert module.queried == []


def test_a_record_describing_the_parent_namespace_is_refused(monkeypatch):
    """The exact observed defect: asked about the child, answered about the parent.

    The client reports success, so only the record's own identity gives it
    away. `2005-02-17` must never come back as the child's creation date.
    """
    module = FakeWhoisModule(
        {DELEGATED_CHILD: record(PARENT_NAMESPACE, PARENT_CREATED)},
        has_extract=False,
    )
    install_whois(monkeypatch, module)

    with pytest.raises(WhoisUnavailable) as raised:
        PythonWhoisLookup().creation_date(DELEGATED_CHILD)

    assert PARENT_NAMESPACE in str(raised.value)


def test_the_parent_namespace_itself_is_still_answerable(monkeypatch):
    """Refusing the mismatch must not refuse a genuine query for the parent."""
    install_whois(
        monkeypatch,
        FakeWhoisModule({PARENT_NAMESPACE: record(PARENT_NAMESPACE, PARENT_CREATED)}),
    )

    age = PythonWhoisLookup().creation_date(PARENT_NAMESPACE)

    assert age.created_at == PARENT_CREATED


def test_a_record_naming_several_spellings_matches_on_any_of_them(monkeypatch):
    install_whois(
        monkeypatch,
        FakeWhoisModule(
            {
                DELEGATED_CHILD: FakeRecord(
                    domain_name=[DELEGATED_CHILD.upper(), DELEGATED_CHILD],
                    creation_date=CHILD_CREATED,
                )
            }
        ),
    )

    age = PythonWhoisLookup().creation_date(DELEGATED_CHILD)

    assert age.created_at == CHILD_CREATED


def test_a_record_naming_no_domain_at_all_is_still_accepted(monkeypatch):
    """Many registries' parsers record no domain_name; that is not a mismatch."""
    install_whois(
        monkeypatch,
        FakeWhoisModule(
            {"example.com": FakeRecord(domain_name=None, creation_date=PARENT_CREATED)}
        ),
    )

    age = PythonWhoisLookup().creation_date("example.com")

    assert age.created_at == PARENT_CREATED


def test_a_registry_no_match_stays_a_genuine_negative(monkeypatch):
    """Unchanged behaviour: "no such domain" is an answer, not a failure."""
    install_whois(monkeypatch, FakeWhoisModule({}))

    age = PythonWhoisLookup().creation_date("example.com")

    assert age.domain == "example.com"
    assert age.created_at is None


def test_a_transport_failure_is_still_unavailable_not_a_negative(monkeypatch):
    class Exploding(FakeWhoisModule):
        def whois(self, domain):
            raise OSError("connection reset")

    install_whois(monkeypatch, Exploding({}))

    with pytest.raises(WhoisUnavailable):
        PythonWhoisLookup().creation_date("example.com")


def test_a_record_without_a_creation_date_is_a_genuine_negative(monkeypatch):
    install_whois(
        monkeypatch,
        FakeWhoisModule(
            {"example.com": FakeRecord(domain_name="example.com", creation_date=None)}
        ),
    )

    age = PythonWhoisLookup().creation_date("example.com")

    assert age.created_at is None


def test_a_malformed_creation_date_is_a_genuine_negative(monkeypatch):
    install_whois(
        monkeypatch,
        FakeWhoisModule(
            {
                "example.com": FakeRecord(
                    domain_name="example.com", creation_date="not a date"
                )
            }
        ),
    )

    age = PythonWhoisLookup().creation_date("example.com")

    assert age.created_at is None


def test_the_vendored_suffix_seed_is_parsed_from_the_project_list():
    """Seeding uses the vendored PSL, not a list of namespaces in the code."""
    seed = l1_headers._vendored_suffix_seed()

    assert seed is not None
    assert b"com" in seed and b"co.in" in seed
    # Present because the vendored list has it, not because anything names it.
    assert PARENT_NAMESPACE.encode() in seed


# --------------------------------------------------------------------------
# The delegated-namespace case, end to end through analyze_domain_age
# --------------------------------------------------------------------------


def test_a_young_delegated_domain_fires_rather_than_inheriting_the_parent_age():
    """The false negative this whole fix exists to close."""
    lookup = FakeWhois(created=NOW - timedelta(days=1))

    signal = analyze_domain_age(
        email(f"alerts@cx.evil.{PARENT_NAMESPACE}"), lookup=lookup, now=NOW
    )

    assert lookup.calls == [f"evil.{PARENT_NAMESPACE}"]
    assert signal.metadata["fired"] is True
    assert signal.metadata["domain"] == f"evil.{PARENT_NAMESPACE}"


def test_an_old_delegated_domain_stays_silent():
    lookup = FakeWhois(created=NOW - timedelta(days=400))

    signal = analyze_domain_age(
        email(f"a@cx.{DELEGATED_CHILD}"), lookup=lookup, now=NOW
    )

    assert signal.metadata["domain"] == DELEGATED_CHILD
    assert signal.metadata["fired"] is False
    assert signal.error is None


def test_a_refused_identity_abstains_rather_than_reporting_the_domain_old():
    """An unusable WHOIS identity must not read as "checked, and it is old"."""
    lookup = FakeWhois(error=WhoisUnavailable("record was for the parent namespace"))

    signal = analyze_domain_age(
        email(f"a@{DELEGATED_CHILD}"), lookup=lookup, now=NOW
    )

    assert signal.score == 0.0
    assert signal.metadata["fired"] is False
    assert signal.error is not None
    assert signal.metadata["authoritative"] is False


def test_a_domain_registered_days_ago_fires():
    lookup = FakeWhois(created=NOW - timedelta(days=2))

    signal = analyze_domain_age(email("a@fresh-phish.top"), lookup=lookup, now=NOW)

    assert signal.name == "domain_age_lt_7d"
    assert signal.metadata["fired"] is True
    assert signal.score > 0.0
    assert signal.error is None
    assert signal.metadata["domain"] == "fresh-phish.top"
    assert signal.metadata["age_days"] == pytest.approx(2.0)
    assert "fresh-phish.top" in signal.evidence


def test_an_established_domain_does_not_fire():
    lookup = FakeWhois(created=NOW - timedelta(days=4000))

    signal = analyze_domain_age(email("a@example.com"), lookup=lookup, now=NOW)

    assert signal.metadata["fired"] is False
    assert signal.score == 0.0
    assert signal.error is None  # checked, and genuinely old


def test_the_threshold_is_seven_days():
    old_enough = FakeWhois(created=NOW - timedelta(days=7, seconds=1))
    just_young = FakeWhois(created=NOW - timedelta(days=6, hours=23))

    assert analyze_domain_age(
        email("a@x.com"), lookup=old_enough, now=NOW
    ).metadata["fired"] is False
    assert analyze_domain_age(
        email("a@x.com"), lookup=just_young, now=NOW
    ).metadata["fired"] is True


def test_a_successful_lookup_is_cached_and_not_repeated(cache):
    lookup = FakeWhois(created=NOW - timedelta(days=1))
    msg = email("a@fresh-phish.top")

    first = analyze_domain_age(msg, lookup=lookup, cache=cache, now=NOW)
    second = analyze_domain_age(msg, lookup=lookup, cache=cache, now=NOW)

    assert lookup.calls == ["fresh-phish.top"]  # one lookup, two answers
    assert first.metadata["cached"] is False
    assert second.metadata["cached"] is True
    assert second.metadata["fired"] is True
    assert second.metadata["age_days"] == first.metadata["age_days"]


def test_the_positive_cache_ttl_is_seven_days(cache, monkeypatch):
    recorded: list[float] = []
    original = cache.set

    def spy(key, value, ttl_seconds):
        recorded.append(ttl_seconds)
        original(key, value, ttl_seconds)

    monkeypatch.setattr(cache, "set", spy)
    analyze_domain_age(
        email("a@example.com"),
        lookup=FakeWhois(created=NOW - timedelta(days=900)),
        cache=cache,
        now=NOW,
    )

    assert recorded == [WHOIS_TTL_SECONDS]
    assert WHOIS_TTL_SECONDS == 7 * 24 * 3600


def test_an_expired_cache_entry_causes_a_fresh_lookup(cache, monkeypatch):
    lookup = FakeWhois(created=NOW - timedelta(days=1))
    msg = email("a@fresh-phish.top")

    analyze_domain_age(msg, lookup=lookup, cache=cache, now=NOW)
    assert lookup.calls == ["fresh-phish.top"]

    # Step time past the 7-day TTL. `Cache` expires against wall-clock time,
    # so the clock is what has to move, not `now`.
    import storage.cache as cache_module

    real_time = cache_module.time.time
    monkeypatch.setattr(
        cache_module.time, "time", lambda: real_time() + WHOIS_TTL_SECONDS + 1
    )

    analyze_domain_age(msg, lookup=lookup, cache=cache, now=NOW)

    assert lookup.calls == ["fresh-phish.top", "fresh-phish.top"]


def test_a_lookup_failure_abstains_with_an_error():
    """Never a silent not-fired: an unchecked domain is not an old domain."""
    lookup = FakeWhois(error=ConnectionError("whois.verisign-grs.com timed out"))

    signal = analyze_domain_age(email("a@unknown-domain.com"), lookup=lookup, now=NOW)

    assert signal.error is not None
    assert signal.metadata["fired"] is False
    assert signal.score == 0.0
    assert "abstains" in signal.evidence


def test_a_failure_message_does_not_leak_the_registry_response():
    lookup = FakeWhois(error=ConnectionError("RAW REGISTRY BLOB: secret-ish"))

    signal = analyze_domain_age(email("a@unknown-domain.com"), lookup=lookup, now=NOW)

    assert "secret-ish" not in signal.evidence
    assert "secret-ish" not in (signal.error or "")
    assert "ConnectionError" in signal.error


def test_a_genuine_negative_is_negatively_cached_so_it_is_not_retried(cache):
    """The registry answered and recorded no date. That answer is worth keeping."""
    lookup = FakeWhois(created=None)
    msg = email("a@no-creation-date.com")

    first = analyze_domain_age(msg, lookup=lookup, cache=cache, now=NOW)
    second = analyze_domain_age(msg, lookup=lookup, cache=cache, now=NOW)

    assert lookup.calls == ["no-creation-date.com"]  # the second call was suppressed
    assert first.error is not None and second.error is not None
    assert second.metadata["fired"] is False
    assert second.metadata["cached"] is True
    assert second.metadata["authoritative"] is True


def test_the_negative_cache_ttl_is_shorter_than_the_positive_one(cache, monkeypatch):
    recorded: list[float] = []
    original = cache.set
    monkeypatch.setattr(
        cache, "set",
        lambda k, v, ttl: (recorded.append(ttl), original(k, v, ttl))[1],
    )

    analyze_domain_age(
        email("a@no-creation-date.com"),
        lookup=FakeWhois(created=None),
        cache=cache,
        now=NOW,
    )

    assert recorded == [WHOIS_NEGATIVE_TTL_SECONDS]
    assert WHOIS_NEGATIVE_TTL_SECONDS < WHOIS_TTL_SECONDS


def test_a_negative_cache_entry_expires_and_the_lookup_is_retried(cache, monkeypatch):
    lookup = FakeWhois(error=TimeoutError("rate limited"))
    msg = email("a@ratelimited-domain.com")

    analyze_domain_age(msg, lookup=lookup, cache=cache, now=NOW)

    import storage.cache as cache_module

    real_time = cache_module.time.time
    monkeypatch.setattr(
        cache_module.time,
        "time",
        lambda: real_time() + WHOIS_NEGATIVE_TTL_SECONDS + 1,
    )

    analyze_domain_age(msg, lookup=lookup, cache=cache, now=NOW)

    assert len(lookup.calls) == 2


def test_whois_answering_with_no_creation_date_abstains():
    """An answer without a date is an answer - and still not evidence of age."""
    signal = analyze_domain_age(
        email("a@example.com"), lookup=FakeWhois(created=None), now=NOW
    )

    assert signal.error is not None
    assert signal.metadata["fired"] is False


@pytest.mark.parametrize("from_header", ["", "malformed", "a@localhost", "a@"])
def test_a_missing_or_malformed_domain_abstains(from_header):
    lookup = FakeWhois(created=NOW - timedelta(days=1))

    signal = analyze_domain_age(email(from_header), lookup=lookup, now=NOW)

    assert signal.error is not None
    assert signal.metadata["fired"] is False
    assert lookup.calls == []  # nothing to look up, so nothing was looked up


def test_no_configured_lookup_abstains_rather_than_reporting_clean():
    signal = analyze_domain_age(email("a@example.com"), lookup=None, now=NOW)

    assert signal.error == "no WHOIS lookup configured"
    assert signal.metadata["fired"] is False


def test_a_broken_cache_does_not_fail_the_signal():
    class BrokenCache:
        def get(self, key):
            raise RuntimeError("disk gone")

        def set(self, key, value, ttl_seconds):
            raise RuntimeError("disk gone")

    signal = analyze_domain_age(
        email("a@fresh-phish.top"),
        lookup=FakeWhois(created=NOW - timedelta(days=1)),
        cache=BrokenCache(),
        now=NOW,
    )

    assert signal.metadata["fired"] is True


def test_a_naive_whois_datetime_is_treated_as_utc():
    lookup = FakeWhois(created=datetime(2026, 2, 28, 12, 0))  # no tzinfo

    signal = analyze_domain_age(email("a@x.com"), lookup=lookup, now=NOW)

    assert signal.metadata["age_days"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Display-name brand impersonation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "from_header,brand",
    [
        ('"Microsoft Account Team" <security@random-vps.tk>', "microsoft"),
        ('"PayPal Service" <service@paypal-secure.top>', "paypal"),
        ('"DocuSign" <no-reply@docu-sign-alerts.info>', "docusign"),
        ('"Apple Support" <appleid@icloud-verify.ru>', "apple"),
        ('"Chase Online" <alerts@chase-secure-login.co>', "chase"),
        ('"Amazon" <auto-confirm@amazon-billing.xyz>', "amazon"),
    ],
)
def test_a_brand_name_from_an_unrelated_domain_fires(from_header, brand):
    signal = analyze_display_name_impersonation(email(from_header))

    assert signal.name == "display_name_impersonation"
    assert signal.metadata["fired"] is True
    assert signal.score > 0.0
    assert brand in signal.metadata["impersonated_brands"]
    assert brand in signal.evidence
    assert signal.metadata["from_domain"] in signal.evidence


@pytest.mark.parametrize(
    "from_header",
    [
        '"Microsoft Account Team" <account-security-noreply@microsoftonline.com>',
        '"Microsoft" <noreply@office365.com>',
        '"Google" <no-reply@accounts.google.com>',
        '"PayPal" <service@paypal.com>',
        '"Apple" <no_reply@icloud.com>',
        '"Amazon.com" <shipment@amazon.co.uk>',
    ],
)
def test_a_brand_sending_from_its_own_domain_does_not_fire(from_header):
    signal = analyze_display_name_impersonation(email(from_header))

    assert signal.metadata["fired"] is False
    assert signal.score == 0.0
    assert signal.error is None


@pytest.mark.parametrize(
    "from_header",
    [
        '"John Jomi" <john@example.com>',
        '"Apple Valley Dental" <front-desk@avdental.com>',
        '"Accounts Payable" <ap@corp.co.uk>',
        '"Support Team" <help@saas-product.io>',
        '"Bank of Somewhere" <alerts@bankofsomewhere.com>',
    ],
)
def test_ordinary_display_names_do_not_fire(from_header):
    """The false-positive floor: no fuzzy matching means no fuzzy accusations."""
    signal = analyze_display_name_impersonation(email(from_header))

    assert signal.metadata["fired"] is False


@pytest.mark.parametrize(
    "display",
    ["Micros0ft Security", "PayPa1 Service", "Amaz0n Billing", "M\u200bicrosoft Team",
     "MICROSOFT ACCOUNT", "micro-soft support", "Micro Soft Team"],
)
def test_confusable_and_split_spellings_still_match(display):
    """A fixed substitution table, not an edit-distance search."""
    signal = analyze_display_name_impersonation(
        email(f'"{display}" <x@totally-unrelated.tk>')
    )

    assert signal.metadata["fired"] is True


def test_a_name_that_mentions_a_brand_without_claiming_to_be_it_does_not_fire():
    """The false-positive guard: mentioning a brand is not impersonating it.

    "Microsoft Azure" carries no service vocabulary and is not just the brand,
    so it stays silent - deliberately erring towards precision.
    """
    signal = analyze_display_name_impersonation(
        email('"Microsoft Azure" <alerts@some-reseller.com>')
    )

    assert signal.metadata["fired"] is False
    assert "does not present itself" in signal.evidence


def test_a_missing_display_name_does_not_fire():
    signal = analyze_display_name_impersonation(email("security@random-vps.tk"))

    assert signal.metadata["fired"] is False
    assert signal.error is None
    assert signal.metadata["display_name"] is None


def test_a_brand_name_with_an_unparseable_from_domain_abstains():
    signal = analyze_display_name_impersonation(email('"PayPal" <malformed>'))

    assert signal.metadata["fired"] is False
    assert signal.error is not None


def test_the_evidence_names_the_brands_real_domains():
    signal = analyze_display_name_impersonation(
        email('"PayPal Service" <service@paypal-secure.top>')
    )

    assert "paypal.com" in signal.evidence
    assert "paypal-secure.top" in signal.evidence


# --------------------------------------------------------------------------
# The layer as a whole
# --------------------------------------------------------------------------


def test_analyze_emits_every_layer_one_signal():
    signals = analyze(email("a@example.com"), now=NOW)

    assert [s.name for s in signals] == [
        "spf_fail",
        "dkim_fail",
        "dmarc_fail",
        "replyto_mismatch",
        "domain_age_lt_7d",
        "display_name_impersonation",
    ]
    assert all(s.layer is DetectionLayer.L1 for s in signals)


def test_an_unavailable_whois_lookup_does_not_sink_the_other_signals():
    """Graceful degradation is per signal, not per layer."""
    signals = analyze(
        email('"Microsoft" <x@evil.tk>', reply_to="y@other.tk"),
        whois_lookup=FakeWhois(error=TimeoutError("down")),
        now=NOW,
    )
    by_name = {s.name: s for s in signals}

    assert by_name["domain_age_lt_7d"].error is not None
    assert by_name["replyto_mismatch"].metadata["fired"] is True
    assert by_name["display_name_impersonation"].metadata["fired"] is True


def test_analyze_accepts_an_ingested_message():
    """IngestedMessage is the ingestion boundary's own type."""
    from ingest.pipeline import IngestedMessage

    parsed = parse_email(
        b'From: "Microsoft" <x@evil.tk>\n'
        b"To: a@b.com\nSubject: s\nMessage-ID: <m@x>\n"
        b"Authentication-Results: mx.google.com; spf=fail; dmarc=fail\n"
        b"\nbody\n"
    )
    message = IngestedMessage(gmail_id="g1", thread_id="t1", email=parsed)

    by_name = {s.name: s for s in analyze(message, now=NOW)}

    assert by_name["spf_fail"].metadata["fired"] is True
    assert by_name["dmarc_fail"].metadata["fired"] is True
    assert by_name["display_name_impersonation"].metadata["fired"] is True


def test_a_signal_source_of_the_wrong_type_is_rejected_loudly():
    with pytest.raises(TypeError):
        analyze_reply_to_mismatch("not a message")
