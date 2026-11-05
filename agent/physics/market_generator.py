"""Two hundred protocols, drawn in one pass — and the mix the investor asked for.

The declared market in :mod:`agent.physics.portfolio` is seven protocols written out by
hand: two honest of each kind, two traps, two crashes, every number chosen so that a
reader can check it with a pencil. That is the right market for an *argument* and the
wrong one for a *capacity test*: seven protocols cannot show whether a funnel holds at
two hundred, cannot show what the scanner costs, and cannot fill a chart with the shape
of a real search space. This module is the other market — many, generated, and
deliberately hostile in a way the declared one is not: **most of the universe is not
worth buying.**

The mix, and why those proportions
==================================

====================  ====================================================
60 % **scams**        a curve with no market noise in it at all, an APY
                      between two hundred percent and four thousand, a
                      hidden rug step, and — for half of them — not a
                      single credit of volume: a formula wearing a
                      protocol's clothes
30 % **junk**         high entropy around a *negative* mean, thin
                      turnover, chaotic noise: something is happening here
                      and none of it is a yield
10 % **honest**       a real rate with the market's own noise on it, an
                      APY inside the bucket's declared ceiling, a claim
                      that is exactly what it pays
====================  ====================================================

The proportions are the brief's and they are also the *result* of the modelling: a
market where most of the yield is a promise is the market the assessment was written for,
and a generator that made honesty the majority would be flattering the detector. What the
mix buys, measured, is the funnel of :mod:`agent.physics.discovery` — 120 of 200 dropped
by arithmetic, 60 convicted by Statement #87, 10 funded — with the numbers falling out of
the mix, the thresholds and the cap rather than being written down anywhere.

Vectorised on purpose, and only once
====================================

A record is drawn in **one** call per kind: a two-dimensional ``rng.standard_normal((n,
steps))``, scaled per row, added to a per-row rate. That is the whole implementation and
it is not an optimisation detail — it is the difference between a market a notebook can
build while the reader is still looking at the cell and a market that teaches the reader
to skip the cell. Measured on this machine, five hundred protocols of seven hundred and
twenty steps are drawn in a few milliseconds, and the test suite holds that under a
second so that a slower machine still passes.

The second half of the same rule is that a record is generated **once**: the history is
in :attr:`Protocol.returns` from the moment the market exists, the clock in
:class:`~agent.physics.portfolio.ProtocolMarket` is the only thing that decides how much
of it has happened, and no step of the simulation ever draws a number. A market that
re-rolled its returns on every access would make the assessment's most important property
— *the same record, seen at a later step* — untestable, and would quietly turn every
"earlier" decision in the log into a fiction.

What the generator knows and the agent cannot
=============================================

A generated protocol carries two facts the agent never sees: :attr:`Protocol.rug_step`
(the step at which it stops paying) and the ``note``, which begins with the protocol's
kind. Neither is published: :meth:`Protocol.as_opportunity` — the one thing the scanner
and the assessment read — carries the address, the name, the bucket, the claim, the
record up to the clock, the alive flag and the turnover, and nothing else. So the ground
truth is available to the *reader* of a notebook and to a *test* that wants to check
whether the funnel found the right ten, and unavailable to everything that decides. That
asymmetry is what makes the demo a measurement instead of a script, and the test suite
checks it by renaming and re-noting the whole market and asserting the funnel is
unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional

import numpy as np

from agent.physics.portfolio import (
    BUCKET_ACTIVE,
    BUCKET_EXPERIMENTAL,
    BUCKET_SAFE,
    BUCKETS,
    STEPS_PER_YEAR,
    PortfolioError,
    Protocol,
    ProtocolMarket,
    WhiteList,
)

__all__ = [
    "DARK_SHARE",
    "HONEST_SHARE",
    "HONEST_APY",
    "HONEST_BUCKETS",
    "HONEST_CV",
    "HONEST_TURNOVER",
    "JUNK_SHARE",
    "KINDS",
    "KIND_HONEST",
    "KIND_JUNK",
    "KIND_SCAM",
    "MarketRecipe",
    "RUG_SHARE",
    "SCAM_SHARE",
    "UNLABELED",
    "generate_market",
    "generate_protocols",
    "kind_of",
    "kinds_of",
]


# ── the three kinds, and what distinguishes them ────────────────────────────

KIND_SCAM = "scam"
KIND_JUNK = "junk"
KIND_HONEST = "honest"
KINDS = (KIND_SCAM, KIND_JUNK, KIND_HONEST)
#: What :func:`kind_of` answers for a protocol this module did not generate: a market
#: built by hand, a testnet's protocols, anything. Unknown is a value, not an exception.
UNLABELED = "unlabeled"

#: The brief's mix, as shares of the market.
SCAM_SHARE = 0.60
JUNK_SHARE = 0.30
HONEST_SHARE = 0.10

#: The share of scams that publishes no turnover at all. They are the half of the scam
#: population that the *cheap* gate can remove without measuring a curve, and keeping
#: them separate is what makes the funnel's two stages answer different questions: a
#: formula with no exit is a liquidity fact, a formula with volume is a lie about noise.
DARK_SHARE = 0.50

#: The share of scams that actually stops paying. A rug is what makes the *simulation*
#: dangerous — it is the only way to lose a hundred percent in one step — and a scam that
#: never rugs would be a trap with no teeth. The rest simply keep paying their formula,
#: which is the more common ending in the world and the harder one to notice.
RUG_SHARE = 0.75

#: A scam's claim, in APY: two hundred percent to four thousand. Chosen so that it is
#: above every bucket's declared ceiling (:data:`agent.physics.risk_assessment.PLAUSIBLE_APY`
#: tops out at four hundred percent), which is what makes most of them `SCAM_TRAP` rather
#: than merely `SYNTHETIC_COLLUSION` — and so that the handful that land under the
#: experimental ceiling are convicted by the *record* alone, which is the arm that has to
#: work on a market that lies carefully.
SCAM_APY_LOW = 2.0
SCAM_APY_HIGH = 40.0

#: A scam's relative spread around its own rate: one hundred-thousandth. Statement #87's
#: floor for a coefficient of variation is ``0.005``, so a curve at ``1e-5`` is two and a
#: half orders of magnitude below the line — "no residuals at all", spelled exactly.
SCAM_NOISE = 1.0e-5

#: A scam's published turnover, in credits per step, for the ones that publish any. High,
#: because volume is the advertisement: the trap that everybody trades is the one that
#: looks like a market.
SCAM_TURNOVER_LOW = 20.0
SCAM_TURNOVER_HIGH = 200.0

#: Where the scams sit among the buckets. Skewed towards the risky end, because a claim
#: of a thousand percent belongs there — and not *entirely* there, because a trap that
#: only ever appears in the experimental bucket is a trap that a bucketed agent can dodge
#: by looking at the label, which is not a measurement.
SCAM_BUCKETS = (0.20, 0.40, 0.40)

#: Junk's per-step mean, negative: it bleeds. The magnitude is chosen *with* the spread
#: below rather than for realism, because the two together decide whether the scanner can
#: see it at all: a record of ``n`` steps has a mean that wobbles by ``sigma/sqrt(n)``, so
#: a drift that is small against its own sampling error is a drift the gate cannot see
#: until the record is long. At ``sigma/|mean| = 12.5`` or more, ``std/|mean|`` stays
#: above five at every record length the market is read at, which is the property the
#: mixture needs to be a *test* of the funnel and not a coin toss.
JUNK_MEAN_LOW = 2.0e-3
JUNK_MEAN_HIGH = 4.0e-3

#: Junk's spread per step: five to nine percent. That is entropy, not a yield — a record
#: that moves by a twentieth of a credit for every credit staked, every step, and still
#: loses money on average, carries no information about the future at all.
JUNK_SIGMA_LOW = 5.0e-2
JUNK_SIGMA_HIGH = 9.0e-2

#: Junk's claim, in APY: thirty percent to three hundred. It advertises, which is why the
#: deep pass would also convict it (:class:`agent.physics.risk_assessment.SUSPECT` — a
#: promise of profit on a record that loses money). The scanner simply gets there first
#: and for the price of a standard deviation.
JUNK_CLAIM_LOW = 0.30
JUNK_CLAIM_HIGH = 3.00

#: Junk's turnover: half of it below the floor in
#: :data:`agent.physics.discovery.TURNOVER_FLOOR` (no exit), half of it above (an exit,
#: and still noise). Two different reasons, both correct, both reported separately.
JUNK_TURNOVER_LOW = 0.0
JUNK_TURNOVER_HIGH = 0.9
JUNK_TURNOVER_LIVE = (1.0, 3.0)

#: An honest protocol's claim, per bucket, in APY. Every ceiling here is *inside*
#: :data:`agent.physics.risk_assessment.PLAUSIBLE_APY` for its bucket, so an honest
#: protocol is never refused for promising too much — the only thing wrong with the
#: dishonest ones is that they promise what no market pays.
HONEST_APY = {
    BUCKET_SAFE: (0.06, 0.30),
    BUCKET_ACTIVE: (0.35, 1.20),
    BUCKET_EXPERIMENTAL: (1.00, 3.00),
}

#: An honest protocol's spread as a *multiple* of its own rate: ``0.9 … 2.4``. That is
#: the number Statement #87 is measured against — the declared market's staking vault
#: reads ``CV = 3.02`` — and it is what makes an honest record categorically different
#: from a formula: residuals of the order of the yield, not of the order of the float.
HONEST_CV = (0.9, 2.4)

#: An honest protocol's published turnover: a real book, tens to hundreds of credits a
#: step. It exists in this module because liquidity is the one fact about a protocol that
#: its *record* cannot show, and a market that did not publish it would make the scanner's
#: cheapest rule untestable.
HONEST_TURNOVER = (8.0, 120.0)

#: Where the honest protocols sit among the buckets. Skewed towards the safe end, and not
#: entirely there: an honest opportunity in the experimental bucket is the reason a
#: bucketed cap is worth having.
HONEST_BUCKETS = (0.50, 0.35, 0.15)

#: The name stems. One pool for all three kinds on purpose: a name is a brand, and a
#: generator whose honest protocols were called ``honest-001`` would leak the ground truth
#: into the one field the agent reads and print as a decision.
STEMS = ("Nova", "Atlas", "Lumen", "Verta", "Orbis")


# ── the recipe: the market as a value ───────────────────────────────────────

#: The shortest record a market may be drawn with. Not :data:`agent.physics.portfolio.MIN_RECORD`
#: (sixty-four, the length an *assessment* needs): a short market is a legitimate thing to
#: generate — a young venue, a test of what happens before anybody may have an opinion —
#: but sixteen steps is the floor below which a rug step and a standard deviation have
#: nowhere to live.
MIN_STEPS = 16


def _check_share(name: str, value: float) -> float:
    """A share is a number in ``[0, 1]`` or it is a bug — never a silent clamp."""
    share = float(value)
    if not 0.0 <= share <= 1.0:
        raise PortfolioError(f"{name} must be a share in [0, 1], got {value!r}")
    return share


@dataclass(frozen=True)
class MarketRecipe:
    """The market as a recipe: how many, how long, and what the mix is.

    A recipe is a *value*, so that a market can be described, repeated and checked: two
    runs of the same recipe produce the same protocols down to the last float, and a test
    can move one ingredient — the mix, the seed, the share of scams that are dark — and
    hold every other one fixed. Everything the generator needs is here, which is why the
    functions below take a recipe and nothing else.
    """

    count: int = 200
    steps: int = 720
    seed: int = 7
    scam_share: float = SCAM_SHARE
    junk_share: float = JUNK_SHARE
    dark_share: float = DARK_SHARE
    rug_share: float = RUG_SHARE

    def __post_init__(self) -> None:
        if int(self.count) < 1:
            raise PortfolioError(f"a market of {self.count!r} protocols is not one")
        if int(self.steps) < MIN_STEPS:
            raise PortfolioError(
                f"{MIN_STEPS} steps is the shortest record a market may be drawn with, "
                f"got {self.steps!r}"
            )
        _check_share("scam_share", self.scam_share)
        _check_share("junk_share", self.junk_share)
        _check_share("dark_share", self.dark_share)
        _check_share("rug_share", self.rug_share)
        if float(self.scam_share) + float(self.junk_share) > 1.0:
            raise PortfolioError(
                "the scams and the junk cannot outnumber the market: "
                f"{self.scam_share!r} + {self.junk_share!r} > 1"
            )

    @property
    def honest_share(self) -> float:
        """Whatever is left: the mix is a partition, so honesty is the remainder."""
        return float(1.0 - float(self.scam_share) - float(self.junk_share))

    def counts(self) -> Dict[str, int]:
        """How many protocols of each kind the recipe asks for, summing to ``count``.

        The honest count is the remainder rather than another rounding, so the three
        always add up to the market — a generator whose parts do not sum to its whole
        would put an off-by-one into every funnel drawn from it.
        """
        count = int(self.count)
        scam = int(round(count * float(self.scam_share)))
        junk = int(round(count * float(self.junk_share)))
        honest = count - scam - junk
        if honest < 0:
            raise PortfolioError(
                f"{scam} scams + {junk} junk is more than the {count} protocols asked for"
            )
        return {KIND_SCAM: scam, KIND_JUNK: junk, KIND_HONEST: honest}

    def describe(self) -> Dict[str, Any]:
        """The recipe as a report line — what a reader needs to reproduce the market."""
        return {
            "count": int(self.count),
            "steps": int(self.steps),
            "seed": int(self.seed),
            "mix": {kind: share for kind, share in (
                (KIND_SCAM, float(self.scam_share)),
                (KIND_JUNK, float(self.junk_share)),
                (KIND_HONEST, self.honest_share),
            )},
            "counts": self.counts(),
            "dark_share": float(self.dark_share),
            "rug_share": float(self.rug_share),
        }


# ── the arithmetic between an APY and a step, on arrays ─────────────────────


def _rates(apy: Any) -> np.ndarray:
    """Per-step rates that compound to ``apy`` — :func:`portfolio.per_step_rate`, on arrays.

    The same function, not a similar one: the generator has to quote an APY that is
    *exactly* the rate it pays, or every honest protocol would be a liar by one rounding,
    and a test holds the two implementations to ``1e-12`` for that reason.
    """
    growth = 1.0 + np.asarray(apy, dtype=float)
    if bool(np.any(growth <= 0.0)):
        raise PortfolioError("an APY of -100% or less is not a growth rate")
    return np.power(growth, 1.0 / float(STEPS_PER_YEAR)) - 1.0


def _annualise(rate: Any) -> np.ndarray:
    """The APY a per-step rate compounds to — :func:`portfolio.annualise`, on arrays."""
    return np.power(1.0 + np.asarray(rate, dtype=float), float(STEPS_PER_YEAR)) - 1.0


def _name(position: int) -> str:
    """A brand, not a label: the same pool of stems for every kind of protocol."""
    return f"{STEMS[int(position) % len(STEMS)]}{int(position):03d}"


def _rug_steps(rng: Any, count: int, steps: int, share: float) -> Dict[int, int]:
    """Which rows stop paying, and when: after a third of the record and before the end.

    A rug on the last step would be invisible (the record would end before it), and a rug
    on the first would make the protocol an obvious corpse; both would be a generator
    that makes the demo easier than the market. Rows are drawn without replacement, so
    one rug per protocol and never two.
    """
    wanted = int(round(int(count) * float(share)))
    if wanted <= 0:
        return {}
    rows = rng.choice(int(count), size=min(wanted, int(count)), replace=False)
    low = max(int(steps) // 3, 1)
    high = max(int(steps) - 1, low + 1)
    moments = rng.integers(low, high, size=rows.size)
    return {int(row): int(moment) for row, moment in zip(rows, moments)}


# ── the three kinds, one draw each ──────────────────────────────────────────


def _scam_group(
    rng: Any, count: int, steps: int, recipe: MarketRecipe
) -> List[Dict[str, Any]]:
    """Draw the scams: a formula, an impossible claim, and a hidden rug.

    One call for the noise of the whole group, one for the rates, one for the turnover —
    that is the entire cost, and it is what "vectorised" means here: the per-protocol
    work is a slice and a dataclass, never a loop over steps.

    The claim is the curve the protocol actually pays, scaled by ``0.95 … 1.05``. That is
    deliberate and it is the sharpest version of the trap: the number is not a lie about
    the *rate*, it is a lie about the *record* — a promise paid exactly, on a curve with
    no market in it at all, which is what :data:`PLAUSIBLE_APY` and Statement #87 each
    refuse from a different side.
    """
    if count <= 0:
        return []
    apy = rng.uniform(SCAM_APY_LOW, SCAM_APY_HIGH, count)
    rate = _rates(apy)
    noise = 1.0 + SCAM_NOISE * rng.standard_normal((count, int(steps)))
    returns = rate[:, None] * noise
    buckets = rng.choice(BUCKETS, size=count, p=list(SCAM_BUCKETS))
    turnover = rng.uniform(SCAM_TURNOVER_LOW, SCAM_TURNOVER_HIGH, count)
    dark = int(round(count * float(recipe.dark_share)))
    turnover[:dark] = 0.0
    claimed = _annualise(rate) * rng.uniform(0.95, 1.05, count)
    rugs = _rug_steps(rng, count, int(steps), recipe.rug_share)
    for row, moment in rugs.items():
        returns[row, moment] = -1.0
    drafts: List[Dict[str, Any]] = []
    for row in range(count):
        moment = rugs.get(row)
        darker = " and no turnover at all" if turnover[row] <= 0.0 else ""
        coffin = f"; it stops paying at step {moment}" if moment is not None else ""
        drafts.append(
            {
                "bucket": str(buckets[row]),
                "claim_apy": float(claimed[row]),
                "returns": returns[row],
                "turnover": float(turnover[row]),
                "rug_step": None if moment is None else int(moment),
                "note": (
                    f"{KIND_SCAM}: a curve with no market noise in it, claiming "
                    f"{float(claimed[row]):.0%} a year{darker}{coffin}"
                ),
            }
        )
    return drafts


def _junk_group(
    rng: Any, count: int, steps: int, recipe: MarketRecipe
) -> List[Dict[str, Any]]:
    """Draw the junk: a negative mean, one to nine percent of noise, thin turnover.

    The spread is set absolutely rather than as a multiple of the mean, and that is the
    point: junk is a record where the *size* of the moves has nothing to do with the
    yield, so ``std/|mean|`` lands in the tens — an order of magnitude above the scanner's
    ceiling — without anybody choosing a threshold to make it so.
    """
    if count <= 0:
        return []
    rate = -rng.uniform(JUNK_MEAN_LOW, JUNK_MEAN_HIGH, count)
    sigma = rng.uniform(JUNK_SIGMA_LOW, JUNK_SIGMA_HIGH, count)
    returns = np.clip(
        rate[:, None] + sigma[:, None] * rng.standard_normal((count, int(steps))),
        -0.95,
        None,
    )
    split = count // 2
    turnover = rng.uniform(JUNK_TURNOVER_LOW, JUNK_TURNOVER_HIGH, count)
    turnover[split:] = rng.uniform(*JUNK_TURNOVER_LIVE, size=count - split)
    buckets = rng.choice(BUCKETS, size=count, p=[0.40, 0.40, 0.20])
    claimed = rng.uniform(JUNK_CLAIM_LOW, JUNK_CLAIM_HIGH, count)
    return [
        {
            "bucket": str(buckets[row]),
            "claim_apy": float(claimed[row]),
            "returns": returns[row],
            "turnover": float(turnover[row]),
            "rug_step": None,
            "note": (
                f"{KIND_JUNK}: chaotic noise around a negative mean, claiming "
                f"{float(claimed[row]):.0%} a year, {float(turnover[row]):.2f} credits "
                "a step of turnover"
            ),
        }
        for row in range(count)
    ]


def _honest_group(
    rng: Any, count: int, steps: int, recipe: MarketRecipe
) -> List[Dict[str, Any]]:
    """Draw the honest protocols: a real rate, the market's noise, a claim that is paid.

    The claim is :func:`_annualise` of the very rate the record was drawn from, so an
    honest protocol is honest *to the last float*: measuring it against its own ninety
    days gives a ``z`` of order one, which is what
    :func:`agent.physics.risk_assessment.evaluate_opportunity` needs in order to say
    ``HONEST`` without being generous.
    """
    if count <= 0:
        return []
    buckets = rng.choice(BUCKETS, size=count, p=list(HONEST_BUCKETS))
    index = np.array([BUCKETS.index(str(bucket)) for bucket in buckets])
    low = np.array([HONEST_APY[bucket][0] for bucket in BUCKETS])
    high = np.array([HONEST_APY[bucket][1] for bucket in BUCKETS])
    apy = rng.uniform(low[index], high[index])
    rate = _rates(apy)
    sigma = rate * rng.uniform(HONEST_CV[0], HONEST_CV[1], count)
    returns = np.clip(
        rate[:, None] + sigma[:, None] * rng.standard_normal((count, int(steps))),
        -0.95,
        None,
    )
    turnover = rng.uniform(*HONEST_TURNOVER, size=count)
    claimed = _annualise(rate)
    return [
        {
            "bucket": str(buckets[row]),
            "claim_apy": float(claimed[row]),
            "returns": returns[row],
            "turnover": float(turnover[row]),
            "rug_step": None,
            "note": (
                f"{KIND_HONEST}: a rate with the market's noise on it, "
                f"{float(claimed[row]):.0%} a year as claimed and as paid, "
                f"CV {float(sigma[row] / rate[row]):.2f}"
            ),
        }
        for row in range(count)
    ]


# ── the two functions a caller uses ─────────────────────────────────────────


def generate_protocols(
    recipe: Optional[MarketRecipe] = None, **overrides: Any
) -> List[Protocol]:
    """Draw the whole market and return it in publication order, shuffled.

    ``recipe`` defaults to :class:`MarketRecipe` and any field may be overridden as a
    keyword (``generate_protocols(count=500)``), which is what the tests and the notebook
    do — one ingredient at a time, everything else held fixed.

    The groups are drawn in one pass each and then **shuffled**, so that the publication
    order carries no information: a market whose first hundred and twenty entries were
    its scams would be a market where "position in the list" is a feature, and a funnel
    that used it would look like a detector and be a sort. Names are assigned after the
    shuffle, for the same reason.
    """
    plan = MarketRecipe() if recipe is None else recipe
    if overrides:
        plan = replace(plan, **overrides)
    counts = plan.counts()
    rng = np.random.default_rng(int(plan.seed))
    drafts: List[Dict[str, Any]] = []
    drafts.extend(_scam_group(rng, counts[KIND_SCAM], int(plan.steps), plan))
    drafts.extend(_junk_group(rng, counts[KIND_JUNK], int(plan.steps), plan))
    drafts.extend(_honest_group(rng, counts[KIND_HONEST], int(plan.steps), plan))
    order = rng.permutation(len(drafts))
    return [
        Protocol(name=_name(position), **drafts[int(index)])
        for position, index in enumerate(order)
    ]


def generate_market(
    recipe: Optional[MarketRecipe] = None, **overrides: Any
) -> ProtocolMarket:
    """The same market, wrapped in the venue the rest of the layer trades.

    The whitelist is the operator's list and it approves **every** generated address —
    including the scams, which is the whole point of the demo: being on the whitelist is
    permission, not a verdict, and the funnel is what has to sort them out.
    """
    protocols = generate_protocols(recipe, **overrides)
    return ProtocolMarket(protocols=protocols, whitelist=WhiteList.of(*protocols))


# ── the ground truth, for a report or a test, never for a decision ──────────


def kind_of(protocol: Any) -> str:
    """What the generator meant a protocol to be — read from its own note.

    Ground truth, and named so that it cannot be mistaken for evidence: nothing in
    :mod:`agent.physics.discovery` or :mod:`agent.physics.risk_assessment` calls this,
    and nothing they read (:meth:`agent.physics.portfolio.Protocol.as_opportunity`)
    carries the note. It exists so that a notebook can say "the ten the funnel funded are
    the ten that were honest" and a test can hold the funnel to it.
    """
    note = str(getattr(protocol, "note", "") or "")
    prefix = note.split(":", 1)[0].strip().lower()
    return prefix if prefix in KINDS else UNLABELED


def kinds_of(protocols: Any) -> Dict[str, str]:
    """Name to kind for a whole market — the answer key, for checking and nothing else."""
    return {str(protocol.name): kind_of(protocol) for protocol in protocols}

