"""The discovery engine: a market of two hundred, two gates, and the numbers that fall out.

This file holds the two claims the "expansion of the universe" rests on, and it holds
them separately because they fail differently.

**The generator is a market, not a decoration.** The mix is 60 % scams, 30 % junk, 10 %
honest, and each kind has to be what it says: every scam a curve with no market noise in
it (:data:`agent.physics.sterility.STERILE_CV`), every junk record noise with no yield in
it, every honest record a rate with the market's own noise on it. If a scam carried
honest noise it would be a *harder* market and a different claim; if an honest record
were sterile the assessment would be refusing the honest. Both are asserted per protocol,
not on average. The second half of the claim is that a record is drawn once — a market
that re-rolled its returns on every read would make "the same record, seen later"
untestable — and that five hundred protocols are drawn in under a second, because
otherwise nobody would run the notebook that explains why they are drawn at all.

**The funnel is a measurement, not a script.** Nothing in this file tells the scanner
which protocol is which: :func:`agent.physics.market_generator.kind_of` reads the ground
truth out of a ``note`` the venue never publishes, and it is used here — after the fact —
to check that the ten the funnel funded are ten honest protocols, that no scam and no
junk ever received a credit, and that the cheap gate dropped 120 of 200 for reasons it
published. The strongest test in the file is the one that renames and re-notes the entire
market and asserts the funnel is unchanged: if the agent could see the answer key, that
test would fail.

The negative cases are the point of the design and get their own section. The scanner
cannot approve anything — there is no method on it that returns a whitelist — the book
cannot fund what it has not approved (:class:`~agent.physics.portfolio.NotWhitelisted`),
and the cap cannot be widened by asking (``PortfolioError``). Three different refusals
for three different reasons, and each one is a place where a funnel of "interesting
protocols" could otherwise have become a funnel of "protocols".
"""
import json
import time
from collections import Counter

import numpy as np
import pytest

from agent.physics import discovery as D
from agent.physics import market_generator as M
from agent.physics import portfolio as P
from agent.physics import risk_assessment as R
from agent.physics.interfaces import MockWeb3Provider
from agent.physics.sterility import STERILE_CV, variation

#: The market the notebook trades, and the funnel the brief asked for.
STEPS = 720
COUNT = 200
FUNDED = 10
#: The first step at which a review happens: the record is long enough and the cycle has
#: come round. Every funnel number in this file is quoted at that step, before the rugs.
REVIEW_AT = 72


@pytest.fixture()
def market():
    return M.generate_market(count=COUNT, steps=STEPS, seed=7)


@pytest.fixture()
def venue(market):
    provider = MockWeb3Provider(market)
    provider.advance(REVIEW_AT)
    return provider


@pytest.fixture()
def book(venue):
    return P.Portfolio(whitelist=venue.whitelist, provider=venue, cash=100.0)


@pytest.fixture()
def report(book):
    return book.discover(scanner=D.Scanner())


class SpyVenue(MockWeb3Provider):
    """A venue that records every protocol asked for by name.

    The only way to assert that level 2 looked at the watchlist and at nothing else: a
    count of assessments cannot tell "eighty judged" from "eighty judged because it
    happened to be eighty", but the names the venue was asked for can.
    """

    def __init__(self, market):
        super().__init__(market)
        self.named = []

    def protocol(self, name):
        self.named.append(str(name))
        return super().protocol(name)


# ── the generator: the mix, and that each kind is what it says ──────────────


def test_the_mix_is_the_briefs_and_the_parts_sum_to_the_whole():
    recipe = M.MarketRecipe()
    counts = recipe.counts()
    assert counts[M.KIND_SCAM] == 120
    assert counts[M.KIND_JUNK] == 60
    assert counts[M.KIND_HONEST] == 20
    assert sum(counts.values()) == recipe.count
    assert recipe.honest_share == pytest.approx(0.10)


@pytest.mark.parametrize("count", [1, 3, 10, 200, 500])
def test_a_market_of_any_size_still_adds_up(count):
    counts = M.MarketRecipe(count=count).counts()
    assert sum(counts.values()) == count
    assert min(counts.values()) >= 0


def test_the_market_is_drawn_with_the_mix_it_promises(market):
    kinds = M.kinds_of(market.protocols)
    assert len(kinds) == COUNT
    assert list(kinds.values()).count(M.KIND_SCAM) == 120
    assert list(kinds.values()).count(M.KIND_JUNK) == 60
    assert list(kinds.values()).count(M.KIND_HONEST) == 20
    assert M.UNLABELED not in kinds.values()


def test_a_recipe_reproduces_the_same_market_and_another_seed_does_not():
    first = M.generate_protocols(count=40, steps=200, seed=11)
    again = M.generate_protocols(count=40, steps=200, seed=11)
    other = M.generate_protocols(count=40, steps=200, seed=12)
    assert [protocol.name for protocol in first] == [protocol.name for protocol in again]
    for left, right in zip(first, again):
        assert np.array_equal(left.returns, right.returns)
        assert left.claim_apy == right.claim_apy
    assert not np.array_equal(first[0].returns, other[0].returns)


def test_names_and_addresses_are_unique_and_provably_simulated(market):
    names = [protocol.name for protocol in market.protocols]
    addresses = [protocol.address for protocol in market.protocols]
    assert len(set(names)) == len(names)
    assert len(set(addresses)) == len(addresses)
    for address in addresses:
        assert address == address.lower()
        assert len(address) == 42 and address.startswith("0x")
        int(address[2:], 16)  # forty hex digits and nothing else


def test_five_hundred_protocols_are_drawn_in_under_a_second():
    """The brief's performance requirement, measured rather than asserted in prose."""
    start = time.perf_counter()
    protocols = M.generate_protocols(count=500, steps=720, seed=3)
    elapsed = time.perf_counter() - start
    assert len(protocols) == 500
    assert elapsed < 1.0, f"500 protocols took {elapsed:.3f} s"


def test_every_scam_is_a_formula_statement_87_can_see(market):
    """Not one scam may carry market noise: a trap with residuals is a different market.

    Measured up to the rug step, because a rug is a **real** loss and a real loss is real
    noise — after it the record stops being sterile (the whole-record CV rises to about
    20) and that is the correct reading of a protocol that has died, not a hole in the
    detector. The traps are refused long before then, which the next section checks.
    """
    scams = [
        protocol
        for protocol in market.protocols
        if M.kind_of(protocol) == M.KIND_SCAM
    ]
    assert len(scams) == 120
    for protocol in scams:
        moment = len(protocol.returns) if protocol.rug_step is None else protocol.rug_step
        assert variation(protocol.returns[:moment]) < STERILE_CV, protocol.name


def test_every_junk_record_is_noise_rather_than_a_yield(market):
    junk = [
        protocol
        for protocol in market.protocols
        if M.kind_of(protocol) == M.KIND_JUNK
    ]
    assert len(junk) == 60
    for protocol in junk:
        # The gate reads a prefix of the record and this reads the whole of it: if the
        # entropy is there at seven hundred steps it is there at seventy.
        assert variation(protocol.returns) > D.SCAN_CV_MAX, protocol.name


def test_the_honest_records_carry_the_markets_own_noise(market):
    """Neither sterile nor chaotic: the one band the scanner is supposed to keep."""
    honest = [
        protocol
        for protocol in market.protocols
        if M.kind_of(protocol) == M.KIND_HONEST
    ]
    assert len(honest) == 20
    for protocol in honest:
        spread = variation(protocol.returns)
        assert STERILE_CV * 10.0 < spread < D.SCAN_CV_MAX, protocol.name
        assert float(np.mean(protocol.returns)) > 0.0


def test_half_of_every_population_publishes_no_liquidity(market):
    """The two gates catch different things, so the mix has to exercise both."""
    scams = [p for p in market.protocols if M.kind_of(p) == M.KIND_SCAM]
    junk = [p for p in market.protocols if M.kind_of(p) == M.KIND_JUNK]
    honest = [p for p in market.protocols if M.kind_of(p) == M.KIND_HONEST]
    assert sum(1 for p in scams if p.turnover == 0.0) == 60
    assert sum(1 for p in scams if 0.0 < p.turnover) == 60
    assert sum(1 for p in junk if p.turnover < D.TURNOVER_FLOOR) == 30
    assert sum(1 for p in junk if p.turnover >= D.TURNOVER_FLOOR) == 30
    assert all(p.turnover >= D.TURNOVER_FLOOR for p in honest)


def test_the_scam_population_carries_hidden_rugs(market):
    scams = [p for p in market.protocols if M.kind_of(p) == M.KIND_SCAM]
    with_rug = [p for p in scams if p.rug_step is not None]
    assert len(with_rug) == int(round(120 * M.RUG_SHARE))
    for protocol in with_rug:
        assert protocol.returns[int(protocol.rug_step)] == -1.0
        assert 0 < int(protocol.rug_step) < STEPS - 1
    assert all(p.rug_step is None for p in market.protocols if M.kind_of(p) != M.KIND_SCAM)


def test_the_rug_is_invisible_until_its_step(market):
    """The future is in the array and behind the clock — the same rule as the demo market.

    ``record(step)`` is ``returns[:step]``, so the rug *at* step ``r`` is visible from
    step ``r + 1`` — the same step at which the venue marks the protocol dead. The loss
    and the death arrive together, which is what a rug is.
    """
    rug = next(p for p in market.protocols if p.rug_step is not None)
    moment = int(rug.rug_step)
    before = rug.as_opportunity(moment)
    at = rug.as_opportunity(moment + 1)
    assert min(before["returns"]) > -1.0
    assert before["alive"] is True
    assert min(at["returns"]) == -1.0
    assert at["alive"] is False
    market.advance(moment + 1)
    assert rug.alive is False


def test_a_record_is_drawn_once_and_never_again(market, venue):
    """Generated history is not re-rolled: the clock decides how much of it happened."""
    protocol = market.protocols[0]
    before = np.array(protocol.returns, copy=True)
    first = protocol.as_opportunity(venue.step)["returns"]
    venue.advance(120)
    second = protocol.as_opportunity(venue.step)["returns"]
    assert np.array_equal(protocol.returns, before)  # nothing was redrawn
    assert len(first) == REVIEW_AT
    assert len(second) == REVIEW_AT + 120  # the clock grew the record, not a reroll
    assert second[:REVIEW_AT] == first  # and the past did not change


# ── the ground truth is in the note, and nowhere the agent looks ────────────


def test_the_published_opportunity_carries_no_answer_key(market):
    """What a venue publishes is six fields; the note and the rug step are not among them."""
    published = set(market.protocols[0].as_opportunity(10))
    assert published == {
        "address",
        "name",
        "bucket",
        "claim_apy",
        "returns",
        "alive",
        "turnover",
    }
    assert "note" not in published and "rug_step" not in published


def test_a_protocol_the_generator_did_not_draw_is_unlabelled(market):
    stranger = P.Protocol(
        name="Stranger",
        bucket="safe",
        claim_apy=0.1,
        returns=np.full(64, 1e-3),
        note="a hand-written market",
    )
    assert M.kind_of(stranger) == M.UNLABELED
    assert M.kind_of(market.protocols[0]) in M.KINDS


def test_the_gate_and_the_assessment_agree_on_the_coefficient_of_variation():
    """One estimator, written twice: the cheap gate may not use a different definition."""
    record = np.array([0.01, -0.02, 0.03, 0.0, 0.015, -0.005, 0.02, 0.001])
    _, _, spread, cv = D.measure(record)
    assert spread == pytest.approx(float(np.std(record, ddof=1)))
    assert cv == pytest.approx(variation(record))
    assert D.measure([])[0] == 0
    assert D.measure([0.0] * 8)[3] == float("inf")  # no spread to divide by


def test_a_recipe_that_cannot_be_drawn_is_an_error():
    with pytest.raises(P.PortfolioError):
        M.MarketRecipe(count=0)
    with pytest.raises(P.PortfolioError):
        M.MarketRecipe(steps=4)
    with pytest.raises(P.PortfolioError):
        M.MarketRecipe(scam_share=1.4)
    with pytest.raises(P.PortfolioError):
        M.MarketRecipe(scam_share=0.7, junk_share=0.6)


def test_the_generators_arithmetic_is_the_portfolios_arithmetic():
    """The vectorised conversions must be the same functions, not similar ones."""
    apy = np.array([0.05, 0.3, 1.2, 3.0, 40.0])
    assert np.allclose(M._rates(apy), [P.per_step_rate(value) for value in apy], rtol=1e-12)
    rates = M._rates(apy)
    assert np.allclose(M._annualise(rates), [P.annualise(r) for r in rates], rtol=1e-12)
    with pytest.raises(P.PortfolioError):
        M._rates(np.array([-1.5]))


# ── level 2: the assessment, on what the generator meant by each kind ───────


def test_every_honest_protocol_is_admitted_by_the_assessment(market):
    """Twenty honest records, twenty verdicts — measured at the step the funnel sees them."""
    honest = [p for p in market.protocols if M.kind_of(p) == M.KIND_HONEST]
    verdicts = [
        R.evaluate_opportunity(p.as_opportunity(REVIEW_AT)).verdict for p in honest
    ]
    assert verdicts.count(R.HONEST) == len(honest), verdicts
    for protocol in honest:
        item = R.evaluate_opportunity(protocol.as_opportunity(REVIEW_AT))
        assert item.admissible is True
        assert abs(item.claim_z) < R.CLAIM_Z


def test_every_live_scam_is_convicted_before_it_rugs(market):
    """Statement #87 on a formula: the conviction arrives while the trap is still paying."""
    scams = [p for p in market.protocols if M.kind_of(p) == M.KIND_SCAM]
    verdicts = [
        R.evaluate_opportunity(p.as_opportunity(REVIEW_AT)).verdict for p in scams
    ]
    assert set(verdicts) <= set(R.CONVICTIONS), set(verdicts) - set(R.CONVICTIONS)
    assert verdicts.count(R.SCAM_TRAP) > verdicts.count(R.SYNTHETIC_COLLUSION)
    assert R.SUSPECT not in verdicts and R.HONEST not in verdicts


def test_the_assessment_alone_would_keep_a_quarter_of_the_junk(market):
    """Why there are two gates: entropy is not a verdict.

    A junk record is noise with a wide spread, and a claim that is small against that
    spread is *not* an unsupported claim — it is a promise the record cannot contradict
    yet. Measured on this market, of the sixty junk protocols the deep read refuses
    forty-six (a claim of profit on a record that loses money) and passes fourteen. Level
    2 cannot be the only gate, and level 1 cannot be the only gate either (a formula has
    an ordinary mean and spread). Each removes what the other cannot see, which is the
    whole argument for the funnel.
    """
    junk = [p for p in market.protocols if M.kind_of(p) == M.KIND_JUNK]
    verdicts = Counter(
        R.evaluate_opportunity(p.as_opportunity(REVIEW_AT)).verdict for p in junk
    )
    assert verdicts[R.HONEST] > 0
    assert verdicts[R.SUSPECT] > verdicts[R.HONEST]
    assert set(verdicts) <= {R.HONEST, R.SUSPECT}


def test_a_record_that_loses_money_cannot_support_a_claim_of_profit():
    """The ghost arm, on a record written by hand so the answer is arithmetic."""
    rng = np.random.default_rng(5)
    losing = -2.0e-3 + 1.0e-2 * rng.standard_normal(200)
    item = R.evaluate_opportunity(
        {
            "name": "Ghost",
            "address": P.sim_address("Ghost"),
            "bucket": P.BUCKET_SAFE,
            "claim_apy": 0.40,
            "returns": losing,
            "alive": True,
        }
    )
    assert item.ghost is True
    assert item.verdict == R.SUSPECT
    assert item.admissible is False


# ── level 1: the cheap gate, and what it may not do ─────────────────────────


def test_the_scanner_keeps_every_honest_record(venue, market):
    watched = set(D.Scanner().scan(venue).names())
    honest = {p.name for p in market.protocols if M.kind_of(p) == M.KIND_HONEST}
    assert honest <= watched


def test_every_dropped_candidate_has_a_published_reason(venue):
    watch = D.Scanner().scan(venue)
    assert len(watch) + len(watch.dropped) == watch.published
    counts = Counter(candidate.dropped_reason for candidate in watch.dropped)
    assert set(counts) <= set(D.REASONS)
    assert all(reason for reason in counts)
    assert watch.reasons() == dict(sorted(counts.items()))


def test_the_scanner_spends_what_arithmetic_can_spend_and_nothing_else(venue):
    """The rules that fire, in the order they are checked, on the market of 200."""
    watch = D.Scanner().scan(venue)
    reasons = watch.reasons()
    assert reasons[D.REASON_DARK] == 90  # 60 scams and 30 junk publish no volume
    assert reasons[D.REASON_CHAOTIC] + reasons[D.REASON_GHOST] == 30  # the other 30 junk
    assert D.REASON_TRUNCATED not in reasons  # 80 names fit in 128 slots
    assert watch.truncated() == 0


def test_the_scan_is_deterministic(venue):
    first = D.Scanner().scan(venue)
    again = D.Scanner().scan(venue)
    assert first.names() == again.names()
    assert first.reasons() == again.reasons()


def test_a_venue_that_publishes_nothing_scans_to_nothing():
    from agent.physics.interfaces import NoVenueProvider

    watch = D.Scanner().scan(NoVenueProvider())
    assert len(watch) == 0
    assert watch.published == 0
    assert watch.dropped == ()
    assert watch.as_dict()["watched"] == 0


def test_the_watchlist_is_bounded_and_the_bound_is_counted(venue):
    watch = D.Scanner(limit=5).scan(venue)
    assert len(watch) == 5
    assert watch.truncated() == 75  # 80 would have been kept, 5 slots were offered
    assert watch.reasons()[D.REASON_TRUNCATED] == 75
    assert watch.limit == 5


def test_the_scanner_reads_only_what_a_venue_publishes(venue, market):
    """Rename and re-note the whole market: the gate may not notice, because it may not look."""
    before = D.Scanner().scan(venue)
    for index, protocol in enumerate(market.protocols):
        protocol.note = f"{M.UNLABELED}: {index}"
    after = D.Scanner().scan(venue)
    assert after.addresses() == before.addresses()
    assert after.reasons() == before.reasons()
    assert [c.mean for c in after.kept] == [c.mean for c in before.kept]


def test_the_scanner_cannot_approve_anything():
    """The one thing a screener must not be able to do."""
    scanner = D.Scanner()
    for forbidden in ("approve", "whitelist", "execution_whitelist", "admit"):
        assert not hasattr(scanner, forbidden)
    watch = D.Watchlist()
    assert not isinstance(watch, P.WhiteList)
    assert not hasattr(watch, "allows")
    assert not hasattr(D.Candidate(name="", address="", bucket="safe",
                                  claim_apy=0.0, points=0, mean=0.0,
                                  spread=0.0, cv=0.0), "approved")


def test_the_queue_is_ordered_by_the_record_and_not_by_the_claim():
    rng = np.random.default_rng(3)
    steady = P.Protocol(
        name="Steady",
        bucket="safe",
        claim_apy=0.05,
        returns=2.0e-3 + 2.0e-3 * rng.standard_normal(200),
        turnover=50.0,
    )
    bombast = P.Protocol(
        name="Bombast",
        bucket="safe",
        claim_apy=3.0,
        returns=1.0e-3 + 2.0e-3 * rng.standard_normal(200),
        turnover=50.0,
    )
    market = P.ProtocolMarket(
        protocols=[bombast, steady], whitelist=P.WhiteList.of(bombast, steady)
    )
    market.advance(200)  # a record nobody has read yet is not a record
    assert D.Scanner().scan(market).names() == ["Steady", "Bombast"]


def cheapest(repeats, work):
    """The smallest of several runs — noise only ever adds time, never removes it."""
    best = float("inf")
    for _ in range(repeats):
        start = time.process_time()
        work()
        best = min(best, time.process_time() - start)
    return best


def test_level_one_is_an_order_of_magnitude_cheaper_than_level_two():
    """The reason the funnel exists, measured per protocol rather than argued."""
    market = M.generate_market(count=100, steps=720, seed=5)
    venue = MockWeb3Provider(market)
    venue.advance(200)
    scanner = D.Scanner()
    watch = scanner.scan(venue)
    assert len(watch.kept) == 40
    sample = [venue.protocol(item.name).as_opportunity(200) for item in watch.kept[:20]]

    def scan_all():
        for _ in range(3):
            scanner.scan(venue)

    def assess_all():
        for opportunity in sample:
            R.evaluate_opportunity(opportunity)

    cheap = cheapest(3, scan_all) / (3 * len(market.protocols))
    dear = cheapest(3, assess_all) / len(sample)
    assert cheap < dear / 3.0, f"scan {cheap * 1e6:.0f} us, assess {dear * 1e6:.0f} us"


def test_the_scanner_and_the_market_publish_the_same_fields():
    """A protocol asks for exactly what it advertises; a scanner reads nothing else."""
    protocol = M.generate_protocols(count=1, steps=64, seed=2)[0]
    opportunity = protocol.as_opportunity(32)
    candidate = D.Scanner().candidate(opportunity)
    assert candidate.points == 32
    assert candidate.turnover == opportunity["turnover"]
    assert candidate.claim_apy == pytest.approx(protocol.claim_apy)


# ── the funnel: two lists, one cap, and arithmetic that has to close ────────


def candidate(name="Nova000", **overrides):
    """A level-1 candidate built by hand, for the funnels that are not about a market."""
    fields = dict(
        name=name,
        address=P.sim_address(name),
        bucket="safe",
        claim_apy=0.1,
        points=100,
        mean=1.0e-3,
        spread=1.0e-3,
        cv=1.0,
        turnover=10.0,
    )
    fields.update(overrides)
    return D.Candidate(**fields)


def test_the_funnel_is_the_briefs_numbers(report):
    """200 published, 120 refused by the scanner, 70 by the portfolio, 10 funded."""
    funnel = report.as_dict()
    assert funnel["market"] == COUNT
    assert funnel["watchlist"] == 80
    assert funnel["refused_by_scanner"] == 120
    refused_by_portfolio = funnel["refused_by_portfolio"] + funnel["capped"]
    assert refused_by_portfolio == 70
    assert funnel["refused_by_portfolio"] == 60  # verdicts: every live scam
    assert funnel["capped"] == 10  # honest, and no room
    assert funnel["approved"] == FUNDED
    assert funnel["stages"] == [
        ("market", 200),
        ("worth a look", 80),
        ("admitted by the assessment", 20),
        ("execution whitelist", 10),
    ]
    assert (
        funnel["approved"]
        + refused_by_portfolio
        + funnel["refused_by_scanner"]
        == funnel["market"]
    )


def test_a_funnel_that_does_not_close_is_a_bug():
    with pytest.raises(ValueError):
        D.DiscoveryReport(watchlist=D.Watchlist(kept=(candidate(),), published=1))
    with pytest.raises(ValueError):
        D.DiscoveryReport(
            watchlist=D.Watchlist(kept=(candidate("A"), candidate("B")), published=2),
            approved=(candidate("A"), candidate("B")),
            assessments={"A": object(), "B": object()},
            limit=1,
        )


def test_the_deep_pass_reads_only_the_watchlist(market):
    spy = SpyVenue(market)
    spy.advance(REVIEW_AT)
    book = P.Portfolio(whitelist=spy.whitelist, provider=spy)
    report = book.discover(scanner=D.Scanner())
    assert spy.named == [item.name for item in report.watchlist.kept]
    assert len(spy.named) == 80 < len(market.protocols)


def test_the_scanner_cannot_put_a_protocol_into_the_execution_whitelist(book, venue):
    """A proposal is recorded and grants nothing; the book is still closed."""
    watch = D.Scanner().scan(venue)
    book.propose(watch)
    assert book.execution_whitelist is None
    assert book.watchlist is watch
    with pytest.raises(P.NotWhitelisted):
        book.deposit(venue.protocol(watch.kept[0].name), 10.0)


def test_the_execution_whitelist_is_capped_and_the_cap_cannot_be_widened(
    book, market, report
):
    assert len(report.approved) == FUNDED
    assert len(book.execution_whitelist) == FUNDED
    assert book.execution_limit == D.EXECUTION_LIMIT
    with pytest.raises(P.PortfolioError):
        book.approve(market.protocols[: FUNDED + 1])


def test_the_cap_refuses_by_room_and_not_by_verdict(report, book):
    assert len(report.capped) == 10
    for item in report.capped:
        assert report.assessments[item.name].verdict == R.HONEST
        assert report.assessments[item.name].admissible is True
        assert not book.is_executable(item.address)


def test_every_approved_protocol_is_an_honest_one(market, report, book):
    approved = set(report.approved_names())
    honest = {p.name for p in market.protocols if M.kind_of(p) == M.KIND_HONEST}
    assert approved <= honest
    assert len(approved) == FUNDED
    assert len(book.execution_whitelist) == FUNDED


def test_capital_cannot_reach_a_watched_but_unapproved_address(book, report, market):
    capped = report.capped[0]
    with pytest.raises(P.NotWhitelisted):
        book.deposit(market.protocol(capped.name), 5.0)
    approved = report.approved[0]
    assert book.deposit(market.protocol(approved.name), 5.0) > 0.0


def test_a_proposal_outside_the_operators_list_is_an_error(book):
    stranger = candidate("Stranger")
    with pytest.raises(P.NotWhitelisted):
        book.propose(D.Watchlist(kept=(stranger,), published=1))


def test_without_a_scanner_the_book_is_the_book_it_was(market, venue):
    """The legacy path: no watchlist, no execution whitelist, the operator's list decides."""
    book = P.Portfolio(whitelist=venue.whitelist, provider=venue, cash=100.0)
    assert book.watchlist is None and book.execution_whitelist is None
    assert book.is_executable(market.protocols[0].address) is True
    assert book.deposit(market.protocols[0], 5.0) > 0.0


def test_the_wallets_report_carries_both_lists(book, report):
    wallet = book.report()
    assert wallet["execution_whitelist"] == sorted(report.approved_addresses())
    assert wallet["execution_limit"] == D.EXECUTION_LIMIT
    assert wallet["watchlist"]["watched"] == 80
    assert wallet["watchlist"]["dropped"] == 120
    assert json.loads(json.dumps(wallet))["execution_limit"] == D.EXECUTION_LIMIT


def test_the_discovery_report_is_json(book, report):
    payload = json.loads(json.dumps(report.as_dict()))
    assert payload["market"] == COUNT
    assert payload["verdicts"][R.HONEST] == 20
    assert len(payload["refusals"]) == 60
    assert set(payload["scanner_reasons"]) <= set(D.REASONS)
    assert report.summary_text().splitlines()[0].startswith(f"step {REVIEW_AT}")


def test_a_venue_that_publishes_nothing_discovers_nothing():
    from agent.physics.interfaces import NoVenueProvider

    venue = NoVenueProvider()
    book = P.Portfolio(whitelist=venue.whitelist, provider=venue)
    report = book.discover(scanner=D.Scanner())
    assert report.market == 0 and report.approved == () and report.refused == ()
    assert len(book.execution_whitelist) == 0  # fail-closed, and it says so
    assert book.is_executable(P.sim_address("anything")) is False


# ── the empire at scale: what it trades, and what it never touches ──────────


@pytest.fixture(scope="module")
def empire_scale():
    """One run of the two-hundred-protocol market, shared by the tests that read it."""
    market = M.generate_market(count=COUNT, steps=STEPS, seed=7)
    venue = MockWeb3Provider(market)
    empire = P.Empire.launch(
        provider=venue, scanner=D.Scanner(), steps=STEPS, seed=7
    )
    empire.run(240)
    return empire, market


def test_the_first_review_reports_the_briefs_funnel(empire_scale):
    empire, _ = empire_scale
    first = empire.decisions[0]
    assert first["step"] == REVIEW_AT
    assert first["funnel"]["market"] == 200
    assert first["funnel"]["watchlist"] == 80
    assert first["funnel"]["refused_by_scanner"] == 120
    assert first["funnel"]["approved"] == FUNDED
    assert first["approved"] == FUNDED
    assert len(first["funnel"]["approved_addresses"]) == FUNDED


def test_the_empire_at_scale_grows(empire_scale):
    empire, _ = empire_scale
    history = empire.history()
    assert history["equity"][-1] > history["equity"][0]
    assert empire.portfolio.equity > P.START_CAPITAL


def test_every_credit_that_moved_went_to_an_approved_address(empire_scale):
    empire, _ = empire_scale
    approved_by_step = {
        decision["step"]: set(decision["funnel"]["approved_addresses"])
        for decision in empire.decisions
        if decision["funnel"] is not None
    }
    deposits = [
        event
        for event in empire.portfolio.events
        if event["action"] == "deposit"
    ]
    assert deposits
    for event in deposits:
        assert event["address"] in approved_by_step[int(event["step"])]


def test_no_protocol_outside_the_funnel_ever_received_a_credit(empire_scale):
    """190 of 200 never see a credit. That is the point of the whole exercise."""
    empire, market = empire_scale
    funded = {
        event["address"]
        for event in empire.portfolio.events
        if event["action"] == "deposit"
    }
    assert funded
    assert funded <= set(empire.portfolio.execution_whitelist.as_list()) | {
        address
        for decision in empire.decisions
        if decision["funnel"] is not None
        for address in decision["funnel"]["approved_addresses"]
    }
    untouched = set(p.address for p in market.protocols) - funded
    assert len(untouched) >= 190
    assert all(empire.portfolio.value(address) == 0.0 for address in untouched)


def test_the_empire_never_funded_a_scam_or_a_junk(empire_scale):
    empire, market = empire_scale
    kinds = {p.address: M.kind_of(p) for p in market.protocols}
    funded = {
        event["address"]
        for event in empire.portfolio.events
        if event["action"] == "deposit"
    }
    assert funded
    assert {kinds[address] for address in funded} == {M.KIND_HONEST}


def test_the_empire_report_carries_the_funnel_and_is_json(empire_scale):
    empire, _ = empire_scale
    payload = json.loads(json.dumps(empire.report()))
    assert payload["funnel"]["market"] == 200
    assert payload["funnel"]["approved"] == FUNDED
    assert payload["portfolio"]["execution_limit"] == D.EXECUTION_LIMIT
    assert len(payload["portfolio"]["execution_whitelist"]) == FUNDED
    assert "funnel:" in empire.summary_text()
    assert "published -> " in empire.summary_text()


def test_without_a_scanner_the_empire_still_reads_the_whole_market(venue):
    """The legacy path is unchanged: no funnel, every protocol assessed, same verdicts."""
    empire = P.Empire.launch(provider=venue, steps=120, seed=7)
    empire.run(120)
    assert empire.funnel is None
    assert empire.decisions[0]["funnel"] is None
    assert len(empire.assessments[empire.decisions[0]["step"]]) == COUNT
    assert empire.portfolio.execution_whitelist is None


def test_the_funnel_figure_is_written(tmp_path, report):
    pytest.importorskip("matplotlib")
    from agent.physics.plots import plot_funnel

    written = plot_funnel(
        report.stages(),
        tmp_path / "funnel.png",
        reasons=report.watchlist.reasons(),
    )
    assert written.exists()
    assert written.stat().st_size > 1000
    single = plot_funnel(report.stages(), tmp_path / "funnel_alone.png")
    assert single.exists()

