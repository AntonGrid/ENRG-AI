"""``python -m agent.backtest.train`` — fit the policy once, on a record that ends before the test.

:mod:`agent.backtest.walkforward` answers "would this have worked"; this file answers the
narrower, shippable question: *what weights do I write down?* The two are not the same object
and pretending they are is how a backtest becomes a sales pitch. The walk-forward refits inside
every fold and measures; this file fits on the head of the record, checks itself on the tail it
was not allowed to see, and only then — with the check's number printed above the weights —
fits once more on everything and writes a JSON document.

The check is the point
======================

Three readings on the held-out tail, and each one can fail on its own:

``auc``
    The probability that a bar that paid scores above a bar that did not, estimated by rank
    (Mann–Whitney, no library). ``0.5`` is a coin. Under ``0.55`` the features carry so
    little that no threshold is worth tuning.
``precision``
    Of the bars the policy would allow, the share that actually paid — read *next to*
    ``base_rate``, because a policy that allows everything has a precision equal to the base
    rate and has learned nothing, and a policy that allows one lucky bar has a precision of
    one and has learned nothing either.
``brier``
    The mean squared error of the probability itself. This is the reading calibration buys:
    a model can rank perfectly and still say ``0.9`` about a coin.

A check that fails does not stop the file from writing weights — it prints the sentence and,
under ``--strict``, exits non-zero so a pipeline can refuse to ship them. Deciding that a
model is good enough is an operator's call; this file's job is to make the call informed.

Why the final fit uses the whole record
=======================================

Because the alternative — shipping the weights fitted on 70 % of the data — throws away
information the *model* is entitled to, while the *estimate* of its quality keeps the tail it
was denied. The two are separate objects: the check is honest because it was measured on data
the fitting never saw, and the weights are as good as the record allows because they were
fitted after the check was taken. The JSON records ``samples`` and ``positives`` so a reader
can see how much history stands behind them.

No key, no RPC, no state file. The output is a small JSON document that ``agent.fed`` can sign
and any reader can diff.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agent.backtest.data import (
    DEFAULT_PERIOD,
    DEFAULT_TIMEOUT,
    HTX_ENDPOINTS,
    HTX_TICKERS_ENDPOINTS,
    PERIOD_SECONDS,
    CandleSeries,
    HtxError,
    fetch_klines,
    universe,
)
from agent.backtest.admission import add_admission_arguments, admission_from_args
from agent.backtest.exits import add_exit_arguments, exits_from_args
from agent.backtest.engine import Settings
from agent.backtest.features import POLICY_FEATURES
from agent.backtest.labels import DEFAULT_COST
from agent.backtest.policy import DEFAULT_L2, LogisticPolicy
from agent.backtest.run import EXIT_FEED, EXIT_USAGE
from agent.backtest.walkforward import Sample, panel

__all__ = [
    "DEFAULT_POLICY",
    "DEFAULT_REPORT",
    "EXIT_GATE",
    "Check",
    "holdout_check",
    "main",
    "rank_auc",
    "train",
]

#: Where the weights land unless ``--out`` says otherwise. Under ``data/`` for the same reason
#: everything else in this package is: it is scratch, and it is never the live state file.
DEFAULT_POLICY = "data/policy.json"
DEFAULT_REPORT = "data/policy_report.json"

#: What ``--strict`` returns when the check failed: measured, informed, and refused. It is not
#: 1 — that is reserved for a crash — and it is not 2, which means the caller made a mistake.
EXIT_GATE = 5

#: The check's own bar. AUC is the only one that is a *statistic*; the other two depend on a
#: threshold that the caller may move, so they are printed and judged second.
MIN_AUC = 0.55


@dataclass(frozen=True)
class Check:
    """The held-out tail's verdict on the fitted model, as numbers and as sentences.

    ``labels`` counts the rows the check could use: a bar whose future is not in the record has
    no label and is not a zero, so it is excluded here exactly as it is excluded at fit time.
    """

    train_samples: int
    train_positives: int
    test_samples: int
    test_positives: int
    auc: float
    base_rate: float
    precision: float
    recall: float
    brier: float
    threshold: float
    passed: bool
    failures: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "train_samples": int(self.train_samples),
            "train_positives": int(self.train_positives),
            "test_samples": int(self.test_samples),
            "test_positives": int(self.test_positives),
            "auc": round(float(self.auc), 6),
            "base_rate": round(float(self.base_rate), 6),
            "precision": round(float(self.precision), 6),
            "recall": round(float(self.recall), 6),
            "brier": round(float(self.brier), 6),
            "threshold": round(float(self.threshold), 6),
            "passed": bool(self.passed),
            "failures": list(self.failures),
        }

    def describe(self) -> str:
        """Two lines: what the model did, and what that means for shipping it."""
        head = (
            f"AUC {self.auc:.3f} on {self.test_samples} held-out bars "
            f"({self.test_positives} paid, base rate {self.base_rate:.1%}) · "
            f"precision {self.precision:.1%} at p ≥ {self.threshold:.2f} · "
            f"recall {self.recall:.1%} · Brier {self.brier:.4f}"
        )
        tail = "pass" if self.passed else "fail: " + "; ".join(self.failures)
        return f"{head}\n   {tail}"


def rank_auc(scores: Sequence[float], labels: Sequence[float]) -> float:
    """The Mann–Whitney reading of a ranking: ``P(score of a winner > score of a loser)``.

    Written by hand — sorting, ranks, ties averaged — because it is twenty lines and because a
    dependency that could change its own tie-handling between versions is the last thing a
    number this load-bearing should rest on. ``0.5`` is chance; ``0.0`` is a model that is
    perfectly and uselessly inverted, which the caller will want to see rather than clip.
    """
    pairs = [
        (float(score), 1.0 if float(label) > 0.5 else 0.0)
        for score, label in zip(scores, labels)
    ]
    positives = sum(1 for _, label in pairs if label > 0.5)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    ordered = sorted(pairs, key=lambda pair: pair[0])
    ranks: List[float] = [0.0] * len(ordered)
    index = 0
    while index < len(ordered):
        stop = index
        while stop + 1 < len(ordered) and ordered[stop + 1][0] == ordered[index][0]:
            stop += 1
        average = (index + stop) / 2.0 + 1.0
        for position in range(index, stop + 1):
            ranks[position] = average
        index = stop + 1
    rank_sum = sum(rank for rank, (_, label) in zip(ranks, ordered) if label > 0.5)
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)



def holdout_check(
    policy: LogisticPolicy,
    samples: Sequence[Sample],
    *,
    threshold: Optional[float] = None,
    min_auc: float = MIN_AUC,
) -> Check:
    """Score the tail the fitting never saw, and say in words whether it is worth shipping.

    ``precision`` is judged against ``base_rate`` and not against zero: a policy whose
    threshold lets everything through has the base rate's precision and deserves no credit for
    it, so the failure sentence is "precision is the base rate — the threshold separated
    nothing", which is a sentence an operator can act on.
    """
    rows = [sample for sample in samples if sample.label is not None]
    if not rows:
        return Check(
            train_samples=int(policy.samples),
            train_positives=int(policy.positives),
            test_samples=0,
            test_positives=0,
            auc=0.5,
            base_rate=0.0,
            precision=0.0,
            recall=0.0,
            brier=0.0,
            threshold=float(policy.threshold if threshold is None else threshold),
            passed=False,
            failures=("no held-out bar had a label: the tail is too short to check",),
        )
    cut = float(policy.threshold if threshold is None else threshold)
    scores = [float(policy.score(sample.case)) for sample in rows]
    labels = [float(sample.label or 0.0) for sample in rows]
    positives = sum(1 for label in labels if label > 0.5)
    base_rate = positives / float(len(labels))
    auc = rank_auc(scores, labels)
    allowed = [label for score, label in zip(scores, labels) if score >= cut]
    allowed_hits = sum(1 for label in allowed if label > 0.5)
    precision = (allowed_hits / float(len(allowed))) if allowed else 0.0
    recall = (allowed_hits / float(positives)) if positives else 0.0
    brier = float(statistics.fmean((score - label) ** 2 for score, label in zip(scores, labels)))
    failures: List[str] = []
    if auc < float(min_auc):
        failures.append(f"AUC {auc:.3f} below {float(min_auc):.2f} — the ranking is near a coin")
    if allowed and precision <= base_rate:
        failures.append(
            f"precision {precision:.1%} is the base rate {base_rate:.1%} — the threshold "
            "separated nothing"
        )
    if not allowed:
        failures.append(f"nothing clears p ≥ {cut:.2f} — the policy would veto every entry")
    return Check(
        train_samples=int(policy.samples),
        train_positives=int(policy.positives),
        test_samples=len(labels),
        test_positives=positives,
        auc=float(auc),
        base_rate=float(base_rate),
        precision=float(precision),
        recall=float(recall),
        brier=float(brier),
        threshold=float(cut),
        passed=not failures,
        failures=tuple(failures),
    )



# ── the fit ────────────────────────────────────────────────────────────────


def train(
    series: Sequence[CandleSeries],
    *,
    settings: Optional[Settings] = None,
    benchmark: Optional[CandleSeries] = None,
    horizon: int = 3,
    cost: float = DEFAULT_COST,
    label_threshold: float = 0.0,
    l2: float = DEFAULT_L2,
    holdout: float = 0.3,
    features: Sequence[str] = POLICY_FEATURES,
    min_train: int = 40,
    min_test: int = 10,
) -> Tuple[LogisticPolicy, Check]:
    """Fit on the head of the record, check on its tail, then refit on all of it.

    ``series`` is a sequence of pairs and not one pair, because a policy fitted on a single
    instrument learns that instrument's ATR and calls it a rule: the fold of five years of
    BTC panics is not a sample of the market, it is a sample of BTC. Panels are concatenated
    bar by bar and the split is drawn *by time across the pool* — the last ``holdout`` share of
    each pair's own bars is the check, so no pair contributes only to one side.

    The two refusals are deliberate and both raise rather than return a policy nobody should
    ship: fewer than ``min_train`` labelled panics is not a fit, and fewer than ``min_test`` is
    not a check. A caller that wants a policy anyway can lower the numbers and own the choice.
    """
    config = settings or Settings()
    panels = [
        panel(
            pair,
            config,
            benchmark=benchmark,
            horizon=int(horizon),
            cost=float(cost),
            threshold=float(label_threshold),
        )
        for pair in series
    ]
    train_rows: List[Sample] = []
    test_rows: List[Sample] = []
    share = min(max(float(holdout), 0.05), 0.95)
    for pair, samples in zip(series, panels):
        bars = max(len(pair.rows), 1)
        cut = int(bars * (1.0 - share))
        for sample in samples:
            if sample.label is None:
                continue
            (test_rows if int(sample.bar) >= cut else train_rows).append(sample)
    if len(train_rows) < int(min_train):
        raise ValueError(
            f"{len(train_rows)} labelled panics is not a fit: --min-train is {int(min_train)}"
        )
    if len(test_rows) < int(min_test):
        raise ValueError(
            f"{len(test_rows)} held-out panics is not a check: --min-test is {int(min_test)}"
        )
    if all(sample.label == train_rows[0].label for sample in train_rows):
        raise ValueError(
            "every training label is the same value: the sample cannot teach a ranking, and "
            "a model fitted on it would be a constant wearing fifteen weights"
        )
    head = LogisticPolicy.fit(
        ((sample.case, sample.label) for sample in train_rows),
        features=features,
        l2=float(l2),
        horizon=int(horizon),
        cost=float(cost),
    )
    check = holdout_check(head, test_rows)
    final = LogisticPolicy.fit(
        ((sample.case, sample.label) for sample in train_rows + test_rows),
        features=features,
        l2=float(l2),
        horizon=int(horizon),
        cost=float(cost),
    )
    return final, check


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agent.backtest.train",
        description=(
            "Fit the policy on a pool of pairs, check it on a tail it never saw, and write the "
            "weights as JSON. Reads candles, writes a document, signs nothing."
        ),
    )
    parser.add_argument("--symbols", default="", help="pairs separated by commas; empty means --top")
    parser.add_argument("--top", type=int, default=5, help="pairs to take from the venue's ranking")
    parser.add_argument("--quote", default="usdt", help="the quote currency of the ranking")
    parser.add_argument("--exchange", default="htx", choices=("htx", "huobi"), help="which endpoint")
    parser.add_argument(
        "--period", default=DEFAULT_PERIOD, choices=sorted(PERIOD_SECONDS), help="the bar"
    )
    parser.add_argument("--days", type=float, default=2000.0, help="how far back to read, in days")
    parser.add_argument("--benchmark", default="btcusdt", help="the regime's benchmark; '' disables")
    parser.add_argument("--horizon", type=int, default=3, help="bars the label looks ahead")
    parser.add_argument("--cost", type=float, default=DEFAULT_COST, help="the round trip paid by the label")
    parser.add_argument("--label-threshold", type=float, default=0.0, help="the net return a label must beat")
    parser.add_argument("--l2", type=float, default=DEFAULT_L2, help="the ridge")
    parser.add_argument("--holdout", type=float, default=0.3, help="the share of each pair's bars kept back")
    parser.add_argument("--min-train", type=int, default=40, help="labelled panics a fit needs")
    parser.add_argument("--min-test", type=int, default=10, help="held-out panics a check needs")
    parser.add_argument("--confidence", type=float, default=0.0, help="the floor a PANIC must reach")
    parser.add_argument(
        "--min-depth",
        type=float,
        default=0.0,
        help="turnover floor for the hunter's THIN arm, USDT; 0 is off",
    )
    parser.add_argument("--fee", type=float, default=0.001, help="the fee per side")
    parser.add_argument("--slippage", type=float, default=0.001, help="slippage per fill")
    parser.add_argument("--latency-bars", type=int, default=1, help="bars from decision to fill")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="seconds per request")
    parser.add_argument("--out", default=DEFAULT_POLICY, help="where the weights are written")
    parser.add_argument("--report", default=DEFAULT_REPORT, help="where the check is written")
    parser.add_argument("--strict", action="store_true", help=f"exit {EXIT_GATE} when the check fails")
    add_admission_arguments(parser)
    add_exit_arguments(parser)
    return parser



def main(argv: Optional[Sequence[str]] = None) -> int:
    """Read a pool of pairs, fit, check, write — and mean the exit code.

    Exit 0 when the weights were written, whether or not the check passed (unless ``--strict``
    is on, and then a failed check is 5). 2 is a caller's mistake, 3 a feed that did not
    answer. A fit that could not be made is a 4: the record is too young for the question.
    """
    args = _parser().parse_args(argv)
    try:
        settings = Settings(
            fee=float(args.fee),
            slippage=float(args.slippage),
            latency_bars=int(args.latency_bars),
            confidence=float(args.confidence),
            min_depth=float(args.min_depth),
            admission=admission_from_args(args),
            **exits_from_args(args),
        )
    except ValueError as exc:
        print(f"train: {exc}", file=sys.stderr)
        return EXIT_USAGE

    symbols: List[str] = []
    if str(args.symbols).strip():
        symbols = [name.strip().lower() for name in str(args.symbols).split(",") if name.strip()]
    else:
        try:
            listed = universe(
                str(args.quote),
                int(args.top),
                timeout=float(args.timeout),
                endpoint=HTX_TICKERS_ENDPOINTS[1 if str(args.exchange) == "huobi" else 0],
            )
        except HtxError as exc:
            print(f"train: {exc}", file=sys.stderr)
            return EXIT_FEED
        symbols = [pair.symbol for pair in listed]
    if not symbols:
        print("train: the universe came back empty; pass --symbols", file=sys.stderr)
        return EXIT_USAGE

    endpoint = HTX_ENDPOINTS[1 if str(args.exchange) == "huobi" else 0]
    pairs: List[CandleSeries] = []
    benchmark: Optional[CandleSeries] = None
    try:
        for symbol in symbols:
            pairs.append(
                fetch_klines(
                    symbol,
                    period=str(args.period),
                    days=float(args.days),
                    timeout=float(args.timeout),
                    endpoint=endpoint,
                )
            )
        if str(args.benchmark).strip():
            benchmark = fetch_klines(
                str(args.benchmark).strip().lower(),
                period=str(args.period),
                days=float(args.days),
                timeout=float(args.timeout),
                endpoint=endpoint,
            )
    except HtxError as exc:
        print(f"train: {exc}", file=sys.stderr)
        return EXIT_FEED

    print(f"── train · {len(pairs)} pairs · {args.period} · {float(args.days):g} days asked · "
          f"holdout {float(args.holdout):.0%} · label beats {float(args.label_threshold):+.2f}% net")
    try:
        policy, check = train(
            pairs,
            settings=settings,
            benchmark=benchmark,
            horizon=int(args.horizon),
            cost=float(args.cost),
            label_threshold=float(args.label_threshold),
            l2=float(args.l2),
            holdout=float(args.holdout),
            min_train=int(args.min_train),
            min_test=int(args.min_test),
        )
    except ValueError as exc:
        print(f"train: {exc}", file=sys.stderr)
        return 4

    target = policy.save(args.out)
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            {
                "check": check.as_dict(),
                "policy": policy.to_dict(),
                "pairs": [pair.symbol for pair in pairs],
                "period": str(args.period),
                "days": float(args.days),
                "holdout": float(args.holdout),
                "benchmark": benchmark.symbol if benchmark is not None else "",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"   {policy.describe()}")
    print(f"   {check.describe()}")
    print(f"   wrote {target}")
    print(f"   wrote {report}")
    if args.strict and not check.passed:
        print(f"train: --strict and the check failed; the weights are on disk but not to be shipped",
              file=sys.stderr)
        return EXIT_GATE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

