"""Two gates before the expensive question: what a mean and a spread can rule out.

The assessment in :mod:`agent.physics.risk_assessment` is the *deep* reading of a
record: an Anderson–Darling statistic, a reducibility survey through the body's own
posture, the mirror's reality score, the claim measured against ninety steps in standard
errors. It is the right question and it costs — measured on this machine, of the order of
a millisecond per protocol without a body and of a hundred times that with one. Seven
protocols is nothing; five hundred protocols reviewed every cycle is a run that spends
its life inside its own risk department.

So this module is the *cheap* gate in front of it, and the word cheap is a requirement
rather than a description: level 1 reads one published record per protocol, computes a
mean, a standard deviation and a coefficient of variation, and answers **"is this even
worth a look"** in microseconds. What a mean and a spread can see, they see completely:

=========================  =====================================================
``alive``                  a protocol that has stopped paying is not an
                           opportunity; the venue already said so
``points``                 a record shorter than a spread can be estimated on has
                           no statistics to argue about yet
``turnover``               a protocol nobody trades is a protocol with no exit, and
                           that is a fact about liquidity, not about yield
``CV``                     ``std/|mean|``: above the ceiling the record is not a
                           yield with noise on it, it is noise with a mean on it
``mean``                   a record that loses money while claiming a profit is a
                           ghost, and one subtraction is enough to see it
=========================  =====================================================

What the gate deliberately does **not** do is judge a *curve*. A synthetic curve with a
competent noise term has a perfectly ordinary mean and CV — that is the whole reason
Statement #87 exists — so the scanner cannot convict one, and does not try. The division
of labour is the argument of this file: **level 1 removes what arithmetic can remove and
nothing else; level 2 measures what only a measurement can see.** On the two-hundred
protocol market of :mod:`agent.physics.market_generator` the split is measured rather
than asserted: 120 protocols are dropped in level 1, 80 reach the assessment, 70 of those
are refused or find no room, and 10 — the honest ones — reach the wallet.

Two lists, and why they are not the same list
=============================================

:class:`Watchlist` is what the scanner proposes: a ranked, bounded set of names worth the
expensive reading. ``Portfolio.execution_whitelist`` is what the *book* has approved:
addresses that passed level 2 and for which there is room under the cap. They are
different objects produced by different code, and the scanner has no way to write the
second one — there is no method here that returns a ``WhiteList``, and no candidate
carries an approval. That is not a convention but a property of this module's surface: an
adapter that could widen its own permissions would make the whitelist worthless, and the
same argument holds one level down.

The cap is the second half of the funnel, and it is a *policy* rather than a
measurement: ten addresses is a decision about how thinly the empire will spread its
attention, and the report counts what the cap cost (``capped``) separately from what the
assessment refused, so a reader can see which gate did the work. A protocol refused for
want of room is not a protocol refused for want of evidence, and a report that blurred
the two would be lying about the quality of the market.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "EXECUTION_LIMIT",
    "MIN_SCANNED",
    "REASONS",
    "REASON_CHAOTIC",
    "REASON_DARK",
    "REASON_DEAD",
    "REASON_GHOST",
    "REASON_SHORT",
    "REASON_TRUNCATED",
    "REASON_UNPAID",
    "SCAN_CV_MAX",
    "TURNOVER_FLOOR",
    "WATCHLIST_LIMIT",
    "Candidate",
    "DiscoveryReport",
    "Scanner",
    "Watchlist",
    "measure",
]

# ── the numbers the gate is built on ────────────────────────────────────────

#: The largest coefficient of variation a watchlist will carry. Measured, not tuned, and
#: measured on the generated market of :mod:`agent.physics.market_generator`: at seven
#: hundred steps the honest records read ``0.9 … 2.3`` and the junk ``7.5 … 1000``, so the
#: threshold sits in the middle of an empty gap. At seventy steps the two distributions
#: *touch* — honest up to ``3.2``, junk from ``3.0`` — because a spread estimated on
#: seventy numbers is not yet a settled number, and no threshold can fix that. Five is
#: chosen for the long record and the gap is documented rather than papered over: what
#: slips through at the short end is refused one stage later by the ghost rule (a claim
#: of profit on a record that loses money), which is the whole reason level 2 exists.
SCAN_CV_MAX = 5.0

#: Credits per step below which a protocol counts as not traded. Zero is the interesting
#: value — a protocol that publishes no volume at all has no exit — and the floor is one
#: credit a step, because a market maker's rounding error is not liquidity.
TURNOVER_FLOOR = 1.0

#: The fewest numbers from which a spread can be estimated at all. Eight is the floor
#: :func:`agent.physics.risk_assessment.claim_statistics` already uses for the same
#: reason: below it a standard deviation is a rumour.
MIN_SCANNED = 8

#: The most names one watchlist may carry. Level 2 is the expensive stage, so the queue
#: in front of it is bounded; what the bound costs is reported as ``truncated`` and never
#: folded into the assessment's refusals, because a protocol that was never read was not
#: judged.
WATCHLIST_LIMIT = 128

#: The most addresses the execution whitelist may carry at once. Ten is a policy about
#: attention and about dilution: three buckets can only ever hold three positions, so ten
#: approved addresses is a bench three times the team, and a hundred would be a book that
#: believes it can watch a hundred things at once.
EXECUTION_LIMIT = 10

# ── why a candidate was dropped: one phrase, because a report is a table ────

REASON_DEAD = "stopped paying"
REASON_SHORT = "record too short"
REASON_DARK = "no turnover"
REASON_CHAOTIC = "noise, not a yield"
REASON_GHOST = "promises a profit on a loss"
REASON_UNPAID = "the record pays nothing"
REASON_TRUNCATED = "no room on the watchlist"

#: Every reason a candidate may carry, in the order the rules are checked.
REASONS = (
    REASON_DEAD,
    REASON_SHORT,
    REASON_DARK,
    REASON_CHAOTIC,
    REASON_GHOST,
    REASON_UNPAID,
    REASON_TRUNCATED,
)


def measure(record: Sequence[float]) -> Tuple[int, float, float, float]:
    """``(points, mean, spread, cv)`` of a published record — the whole of level 1.

    ``cv`` is ``spread/|mean|``, the same estimator
    :func:`agent.physics.sterility.variation` uses, including its answer for a record
    whose mean is exactly zero: ``inf``. Copying the definition instead of importing it
    keeps this module free of the statistics layer, and the two are held together by a
    test rather than by an import: a gate that could not be imported without the
    assessment would not be a gate in front of it.

    ``spread`` is ``ddof=1``, because a record is a sample of a rate and not the rate.
    """
    values = np.asarray(record, dtype=float).ravel()
    points = int(values.size)
    if points == 0:
        return 0, 0.0, 0.0, math.inf
    mean = float(np.mean(values))
    spread = float(np.std(values, ddof=1)) if points > 1 else 0.0
    if mean == 0.0:
        return points, mean, spread, math.inf
    return points, mean, spread, float(spread / abs(mean))


@dataclass(frozen=True)
class Candidate:
    """One protocol as the *cheap* gate saw it: five numbers and a verdict of worth.

    Everything here is a function of what the venue published — the record up to the
    clock, the claim, the turnover and the alive flag. ``rug_step``, ``note`` and every
    other local truth are deliberately absent: the ground truth about a market is for
    checking the funnel after the fact, never for steering it, and a candidate that
    carried it would make the demo a script.
    """

    name: str
    address: str
    bucket: str
    claim_apy: float
    points: int
    mean: float
    spread: float
    cv: float
    turnover: Optional[float] = None
    alive: bool = True
    dropped_reason: str = ""

    @property
    def watched(self) -> bool:
        """True when this candidate earned the expensive reading."""
        return not self.dropped_reason

    @property
    def score(self) -> float:
        """Record per unit of noise — the cheapest ordering that is not the claim.

        The claim is what a protocol *says*, and ranking by it would reproduce exactly
        the credulity the assessment exists to correct. What is left to rank by, for
        free, is the record's own mean divided by its own spread: a Sharpe ratio with no
        risk-free rate and no pretence of being one. It is a *queue order*, not a
        measurement of quality — level 2 decides, and it decides on every candidate
        regardless of where in the queue it stood.
        """
        if self.mean <= 0.0:
            return float("-inf")
        return float(self.mean / max(self.spread, 1.0e-12))

    def as_dict(self) -> Dict[str, Any]:
        """The candidate as a report row: rounded, JSON-able, ``inf`` as ``None``."""
        return {
            "name": self.name,
            "address": self.address,
            "bucket": self.bucket,
            "claim_apy": round(float(self.claim_apy), 6),
            "points": int(self.points),
            "mean": round(float(self.mean), 9),
            "spread": round(float(self.spread), 9),
            "cv": None if math.isinf(self.cv) else round(float(self.cv), 6),
            "turnover": None if self.turnover is None else round(self.turnover, 6),
            "alive": bool(self.alive),
            "watched": bool(self.watched),
            "reason": self.dropped_reason,
            "score": None if math.isinf(self.score) else round(self.score, 6),
        }


@dataclass(frozen=True)
class Watchlist:
    """What the scan produced: the queue, the refusals, and the count of each reason.

    A value, not a handle: nothing here can be added to, and nothing here is an approval.
    :meth:`addresses` exists so that a book can *record* what was proposed and refuse
    everything else, which is the only direction this information is allowed to travel.
    """

    kept: Tuple[Candidate, ...] = ()
    dropped: Tuple[Candidate, ...] = ()
    step: int = 0
    published: int = 0
    limit: int = WATCHLIST_LIMIT

    def __len__(self) -> int:
        return len(self.kept)

    def __iter__(self):
        return iter(self.kept)

    def names(self) -> List[str]:
        return [candidate.name for candidate in self.kept]

    def addresses(self) -> List[str]:
        return [candidate.address for candidate in self.kept]

    def reasons(self) -> Dict[str, int]:
        """How many candidates each rule removed, in the order the rules are checked."""
        counts: Dict[str, int] = {}
        for candidate in self.dropped:
            key = candidate.dropped_reason
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def truncated(self) -> int:
        """How many candidates were kept by the rules and lost to the bound."""
        return int(self.reasons().get(REASON_TRUNCATED, 0))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "step": int(self.step),
            "published": int(self.published),
            "watched": len(self.kept),
            "dropped": len(self.dropped),
            "limit": int(self.limit),
            "truncated": int(self.truncated()),
            "reasons": self.reasons(),
            "names": self.names(),
        }


def _published(source: Any) -> List[Any]:
    """What a venue publishes: the interface's list, or a market's same list.

    Two shapes and one reader, exactly as :func:`agent.physics.risk_assessment.review`
    reads them: a provider answers ``get_protocol_list()`` and a
    :class:`agent.physics.portfolio.ProtocolMarket` holds ``protocols``. Nothing else is
    asked of either — not a clock, not an address book, not a network.
    """
    getter = getattr(source, "get_protocol_list", None)
    if callable(getter):
        return list(getter())
    return list(getattr(source, "protocols", ()))


def _opportunity(item: Any, step: int) -> Mapping[str, Any]:
    """The published fields of one protocol, as a mapping.

    A protocol object is asked for exactly what it advertises
    (:meth:`agent.physics.portfolio.Protocol.as_opportunity`), so the gate and the
    assessment cannot disagree about what a venue said; a mapping is taken as it comes.
    """
    if isinstance(item, Mapping):
        return item
    publish = getattr(item, "as_opportunity", None)
    if callable(publish):
        return publish(int(step))
    return item


@dataclass(frozen=True)
class Scanner:
    """Level 1: the cheap gate, and the only thing in this file that touches a record.

    A scanner is *configured*, never *trusted*: every threshold is a field with a
    published default (:data:`SCAN_CV_MAX`, :data:`TURNOVER_FLOOR`,
    :data:`MIN_SCANNED`), so a run can say which gate it used and a test can move one
    number without moving the others. What it cannot do is approve anything — there is
    no method on this class that returns a whitelist, and :class:`Candidate` has no field
    that means "approved". The book is the only thing that writes the execution list.
    """

    cv_max: float = SCAN_CV_MAX
    turnover_floor: float = TURNOVER_FLOOR
    min_points: int = MIN_SCANNED
    limit: int = WATCHLIST_LIMIT

    # -- one protocol, one verdict of worth -------------------------------- #

    def refusal(self, candidate: Candidate) -> str:
        """The first rule that drops ``candidate``, or ``""`` when it is worth a look.

        The rules are ordered by what they cost to check and by how certain they are, not
        by how interesting they are: a venue's own ``alive`` flag is free and final, a
        spread needs the record, and the mean is already computed by then. The order is
        also why the report's reasons are worth reading — the first rule to fire is the
        one that explains the drop, and a protocol dropped for liquidity tells you
        something different about the market than one dropped for noise.
        """
        if not candidate.alive:
            return REASON_DEAD
        if int(candidate.points) < int(self.min_points):
            return REASON_SHORT
        if candidate.turnover is not None:
            if float(candidate.turnover) < float(self.turnover_floor):
                return REASON_DARK
        if math.isnan(candidate.cv) or float(candidate.cv) > float(self.cv_max):
            return REASON_CHAOTIC
        if candidate.mean <= 0.0:
            return REASON_GHOST if candidate.claim_apy > 0.0 else REASON_UNPAID
        return ""

    def candidate(self, opportunity: Mapping[str, Any]) -> Candidate:
        """Measure one published opportunity: four numbers and a reason, no judgement."""
        get = opportunity.get
        record = get("returns", None)
        if record is None:
            record = get("record", ())
        points, mean, spread, cv = measure(record)
        turnover = get("turnover", None)
        draft = Candidate(
            name=str(get("name", "") or get("address", "") or "unnamed"),
            address=str(get("address", "") or "").lower(),
            bucket=str(get("bucket", "") or "safe"),
            claim_apy=float(get("claim_apy", 0.0) or 0.0),
            points=points,
            mean=mean,
            spread=spread,
            cv=cv,
            turnover=None if turnover is None else float(turnover),
            alive=bool(get("alive", True)),
        )
        reason = self.refusal(draft)
        return draft if not reason else replace(draft, dropped_reason=reason)

    # -- the whole market, one pass ---------------------------------------- #

    def scan(self, source: Any, *, step: Optional[int] = None) -> Watchlist:
        """Read every published record once and return the queue in front of level 2.

        One pass over the venue's list, one measurement per protocol, then a sort and a
        cut: no statistics beyond a mean and a spread, no oracle, no body, no second look.
        The queue is ordered by :attr:`Candidate.score` and truncated at ``limit``; what
        the truncation cost is reported as :data:`REASON_TRUNCATED` so that a full market
        never quietly looks like a small one.
        """
        clock = int(getattr(source, "step", 0)) if step is None else int(step)
        items = _published(source)
        candidates = [self.candidate(_opportunity(item, clock)) for item in items]
        kept = sorted(
            (candidate for candidate in candidates if candidate.watched),
            key=lambda candidate: (-candidate.score, candidate.name),
        )
        room = max(int(self.limit), 0)
        over = kept[room:]
        kept = kept[:room]
        dropped = [candidate for candidate in candidates if not candidate.watched]
        dropped.extend(
            replace(candidate, dropped_reason=REASON_TRUNCATED) for candidate in over
        )
        return Watchlist(
            kept=tuple(kept),
            dropped=tuple(dropped),
            step=clock,
            published=len(candidates),
            limit=room,
        )


# ── the funnel, as a value that has to add up ───────────────────────────────


def _verdict_of(item: Any) -> str:
    """The verdict of an assessment, whatever shape the caller kept it in."""
    if isinstance(item, Mapping):
        return str(item.get("verdict", ""))
    return str(getattr(item, "verdict", ""))


@dataclass(frozen=True)
class DiscoveryReport:
    """One trip through both gates: what was published, kept, judged, and funded.

    The class exists to make two claims checkable rather than rhetorical. The first is
    the *division of labour*: ``refused`` holds verdicts, ``capped`` holds protocols that
    passed and found no room, and they are different tuples because they are different
    facts — muddling them would let a cap look like a measurement. The second is that the
    funnel **closes**: every published protocol is exactly one of kept, refused by the
    scanner, refused by the assessment, capped, or approved, and
    :meth:`__post_init__` refuses to build a report where that is not true. A demo whose
    numbers add up because nobody checked is the kind of demo this file is written
    against.
    """

    step: int = 0
    watchlist: Watchlist = field(default_factory=Watchlist)
    approved: Tuple[Candidate, ...] = ()
    capped: Tuple[Candidate, ...] = ()
    refused: Tuple[Mapping[str, Any], ...] = ()
    assessments: Mapping[str, Any] = field(default_factory=dict)
    limit: int = EXECUTION_LIMIT

    def __post_init__(self) -> None:
        watched = len(self.watchlist.kept)
        judged = len(self.assessments)
        if judged != watched:
            raise ValueError(
                f"the deep pass judged {judged} of {watched} watched protocols: level 2 "
                "may only look at the watchlist"
            )
        accounted = len(self.approved) + len(self.capped) + len(self.refused)
        if accounted != judged:
            raise ValueError(
                f"the funnel does not close: {judged} judged, {accounted} accounted"
            )
        if len(self.approved) > int(self.limit):
            raise ValueError(
                f"{len(self.approved)} approved against a cap of {int(self.limit)}"
            )

    # -- the counts a funnel is drawn from --------------------------------- #

    @property
    def market(self) -> int:
        """Everything the venue published — the top of the funnel."""
        return int(self.watchlist.published)

    @property
    def watched(self) -> int:
        """What the cheap gate kept, and therefore what level 2 may read."""
        return len(self.watchlist.kept)

    @property
    def assessed(self) -> int:
        return len(self.assessments)

    def verdicts(self) -> Dict[str, int]:
        """How many of each verdict the deep pass reached."""
        counts: Dict[str, int] = {}
        for item in self.assessments.values():
            verdict = _verdict_of(item)
            counts[verdict] = counts.get(verdict, 0) + 1
        return dict(sorted(counts.items()))

    def stages(self) -> List[Tuple[str, int]]:
        """The funnel as surviving widths, in the order a chart draws them.

        Four rows, because there are four things a run can say: what the market offered,
        what was worth a look, what the assessment admitted, and what the cap left room
        for. The differences between consecutive rows are the two gates plus the policy,
        and a reader can subtract them without being told which one did what.
        """
        return [
            ("market", self.market),
            ("worth a look", self.watched),
            ("admitted by the assessment", len(self.approved) + len(self.capped)),
            ("execution whitelist", len(self.approved)),
        ]

    def approved_addresses(self) -> List[str]:
        return [candidate.address for candidate in self.approved]

    def approved_names(self) -> List[str]:
        return [candidate.name for candidate in self.approved]

    def as_dict(self) -> Dict[str, Any]:
        """The whole trip, JSON-able — what a log line, a table and a figure need."""
        return {
            "step": int(self.step),
            "limit": int(self.limit),
            "market": self.market,
            "watchlist": self.watched,
            "refused_by_scanner": len(self.watchlist.dropped),
            "assessed": self.assessed,
            "refused_by_portfolio": len(self.refused),
            "capped": len(self.capped),
            "approved": len(self.approved),
            "approved_names": self.approved_names(),
            "approved_addresses": self.approved_addresses(),
            "scanner_reasons": self.watchlist.reasons(),
            "truncated": int(self.watchlist.truncated()),
            "verdicts": self.verdicts(),
            "stages": self.stages(),
            "refusals": [dict(row) for row in self.refused],
        }

    def summary_text(self) -> str:
        """Four lines in the order of the funnel, because that is the argument."""
        reasons = ", ".join(
            f"{reason} {count}"
            for reason, count in self.watchlist.reasons().items()
        )
        verdicts = ", ".join(
            f"{verdict} {count}" for verdict, count in self.verdicts().items()
        )
        return "\n".join(
            [
                f"step {int(self.step)}: the venue publishes {self.market} protocols",
                f"  level 1 (scanner): kept {self.watched}, dropped "
                f"{len(self.watchlist.dropped)}" + (f" ({reasons})" if reasons else ""),
                f"  level 2 (entrain): judged {self.assessed}, refused "
                f"{len(self.refused)}, no room {len(self.capped)}"
                + (f" ({verdicts})" if verdicts else ""),
                f"  execution whitelist: {len(self.approved)} of a cap of "
                f"{int(self.limit)}",
            ]
        )

