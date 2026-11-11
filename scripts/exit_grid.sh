#!/usr/bin/env bash
#
# The exit grid: ten books, one command, and a hash a second machine can be held to.
#
# Why this file exists
# ====================
#
# The gate in ``agent/backtest/walkforward.py`` is priced in trades per year, and on HTX
# history a four-hour book produces about eight of them — far too few to resolve a 30 bps
# gate either way. So the honest question the grid asks is not "which exit passes" but
# "which exit narrows the spread enough to ever be read". That answer only means something
# if a second machine reproduces it exactly, which is why this script prints a sha256 of
# every table it writes: unequal hashes mean the two runs did not read the same candles,
# and no amount of agreement about the conclusions makes up for that.
#
# Usage
# =====
#
#     bash scripts/exit_grid.sh                 # all ten, on ~16 cores, about 25 minutes
#     bash scripts/exit_grid.sh base atr2       # only the named directories
#     OUT=data/grid SKIP=d-base bash scripts/exit_grid.sh
#
# Reproducibility
# ===============
#
# A live run is deterministic *given its candles*, and the venue's candles move: one new bar
# closes and every fold boundary shifts by one. Two runs of the same command an hour apart are
# therefore two experiments, which is why "the other machine got the same numbers" is not a
# checkable sentence about a live run. It is about a frozen one. ``data/frozen/`` holds one
# pair's bars exactly as a run read them, and each run writes the sha256 of what it read into
# its ``criteria.json``; replaying the same directory on another machine must print the same
# hash for the table, or the two machines did not read the same bars:
#
#     bash scripts/exit_grid.sh --replay data/frozen/4hour 4hour
#
# ``--days`` is bounded by ``MAX_BARS = 2000`` in the provider: ``4hour`` therefore reaches
# about 333 days, which is exactly one 2026 test year after the 65-bar minimum train window,
# and ``1day`` reaches the whole 2000-bar history, which is three expanding folds. Asking for
# more days does not buy more history; it buys the same history and a longer wait.
#
# Nothing here signs anything. The backtest reads ``/market/history/kline`` and
# ``/market/tickers``, both unauthenticated, so a machine with no keys can run the whole grid.
#
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
OUT=${OUT:-data/grid}
TIMEOUT=${TIMEOUT:-3600}

# Replay mode: one frozen directory, one book, and a hash that another machine has to match.
# It runs the ATR-2 book, because that is the row the grid's conclusion rests on.
if [ "${1:-}" = "--replay" ]; then
    directory=${2:?usage: exit_grid.sh --replay data/frozen/4hour [period]}
    period=${3:-}
    period_args=()
    [ -n "$period" ] && period_args=(--period "$period")
    name=replay-$(basename "$directory")${period:+-$period}
    echo "replaying $directory as $name"
    "$PY" -m agent.backtest.walkforward \
        --candles "$directory" ${period_args[@]+"${period_args[@]}"} \
        --stop-mode atr --atr-multiple 2 --risk-per-trade 0.02 \
        --out "$OUT/$name"
    echo
    sha256sum "$OUT/$name/walkforward.csv"
    exit 0
fi

# The four-hour book: one test year, five folds' worth of train, about eight trades.
C=(--top 5 --period 4hour --days 340)
# The daily book: the whole history the venue will serve, three expanding folds.
D=(--top 5 --period 1day --days 2000)

run() {
    local name=$1
    shift
    echo "starting $name"
    ( timeout "$TIMEOUT" "$PY" -m agent.backtest.walkforward "$@" --out "$OUT/$name" \
        > "/tmp/exit-grid-$name.log" 2>&1 ) &
}

# A name given on the command line filters the grid; with no arguments every book runs.
SELECTED=("$@")
should() {
    [ "${#SELECTED[@]}" -eq 0 ] && return 0
    local name=$1 candidate
    for candidate in "${SELECTED[@]}"; do
        [ "$candidate" = "$name" ] && return 0
    done
    return 1
}

mkdir -p "$OUT"

# The controls, first: every later book is read against these two.
should base && run base "${C[@]}"
should atr2 && run atr2 "${C[@]}" --stop-mode atr --atr-multiple 2
should atr2-risk2 && run atr2-risk2 "${C[@]}" --stop-mode atr --atr-multiple 2 --risk-per-trade 0.02
# The R book: a take-profit at 2R, then the same with a 1R trail behind it.
should atr2-risk2-tp2 && run atr2-risk2-tp2 "${C[@]}" --stop-mode atr --atr-multiple 2 --risk-per-trade 0.02 --take-profit-r 2
should atr2-risk2-tp2-trail1 && run atr2-risk2-tp2-trail1 "${C[@]}" --stop-mode atr --atr-multiple 2 --risk-per-trade 0.02 --take-profit-r 2 --trail-r 1
# The clock book: unchanged entries and stops, closed after 48 four-hour bars (eight days).
should atr2-risk2-clock48 && run atr2-risk2-clock48 "${C[@]}" --stop-mode atr --atr-multiple 2 --risk-per-trade 0.02 --time-stop-bars 48

# The same three questions asked of the daily book, where the history is long enough that a
# fold is a year rather than a season.
should d-base && run d-base "${D[@]}"
should d-atr2-risk2 && run d-atr2-risk2 "${D[@]}" --stop-mode atr --atr-multiple 2 --risk-per-trade 0.02
should d-atr2-risk2-tp2 && run d-atr2-risk2-tp2 "${D[@]}" --stop-mode atr --atr-multiple 2 --risk-per-trade 0.02 --take-profit-r 2

wait

echo
echo "=== tables and their hashes, $(date -u +%Y-%m-%dT%H:%M:%SZ)"
for directory in "$OUT"/*/; do
    table="$directory/walkforward.csv"
    [ -f "$table" ] || continue
    sha256sum "$table"
done
