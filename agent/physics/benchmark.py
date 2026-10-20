"""The experiment: does the physics catch what was injected, and what does MAD see?

Three detectors are run on the same 168-hour park (``synthetic.build_scenario``)
and compared producer by producer:

1. ``coherence`` — this layer: silence, resonance, phase, written memory;
2. ``spike_mad`` — ``axis_core.ai.anomaly.detect_anomaly``, the verifier's
   spike rule (ratio against the median of the previous values), fed the same
   buckets the oracle would see;
3. ``trend_mad`` — ``agent.signals.generation_signals``, the live AI anomaly
   signal (MAD of Holt residuals, z = 3).

A producer is *caught* when its detector flags it at least once inside the
reading window. Precision and recall are computed against the declared truth,
not against a detector's own opinion, and silence is not rewarded: a detector
that never speaks has recall zero, and its precision is reported next to it.

The benchmark is not a leaderboard: it is the same discipline as the book's
journal — a criterion declared before the run, and the false positives named.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from agent.forecast.energy import ProofSeries
from agent.physics.park import ParkConfig, observe_park
from agent.physics.synthetic import CLEAN, Scenario, build_scenario
from agent.signals import generation_signals


@dataclass
class DetectorRun:
    """One detector's opinion about every producer."""

    name: str
    flags: Dict[str, bool]
    notes: List[str] = field(default_factory=list)

    def flagged(self) -> List[str]:
        return [device for device, flag in self.flags.items() if flag]


def bucket_minutes_of(scenario: Scenario) -> int:
    if len(scenario.starts) < 2:
        raise ValueError("a scenario needs at least two buckets")
    return int((scenario.starts[1] - scenario.starts[0]).total_seconds() // 60)


def _reading_window(scenario: Scenario, config: Optional[ParkConfig]) -> int:
    cfg = config or ParkConfig(bucket_minutes=bucket_minutes_of(scenario))
    return min(cfg.baseline_buckets, len(scenario.starts) - 1)


def coherence_flags(scenario: Scenario, config: Optional[ParkConfig] = None) -> tuple:
    """Run this layer: one reading of the park, flags for every producer."""
    observation = observe_park(
        scenario.energy, scenario.device_ids, scenario.starts, config
    )
    flags = {row["device_id"]: row["is_anomaly"] for row in observation.rows()}
    return DetectorRun("coherence", flags), observation


def _detect_anomaly():
    """The verifier's rule, imported lazily (see ``spike_mad_flags``)."""
    try:
        from axis_core.ai.anomaly import detect_anomaly
    except ImportError as exc:  # pragma: no cover — environment, not logic
        raise RuntimeError(
            "axis_core is not importable: run this benchmark from the AXIS "
            "workspace (Axis-core installed) or pip install -e Axis-core"
        ) from exc
    return detect_anomaly


def spike_mad_flags(scenario: Scenario, threshold: float = 0.5) -> DetectorRun:
    """The verifier's spike rule (Axis-core), bucket by bucket.

    This is how the rule is called today: the latest report against the median
    of the history. On a series with a daily cycle it fires at every noon for
    every producer, whatever was injected — the rule assumes a stationary
    series. The benchmark keeps it, because that is the behaviour being
    compared, not an opinion about it.
    """
    detect_anomaly = _detect_anomaly()
    start = _reading_window(scenario, None)
    flags: Dict[str, bool] = {}
    for row, device_id in enumerate(scenario.device_ids):
        values = scenario.energy[row]
        flagged = False
        for t in range(start, len(values)):
            history = [
                {"energy_wh": float(values[k]), "nonce": k} for k in range(t + 1)
            ]
            if detect_anomaly(device_id, history, threshold).is_anomaly:
                flagged = True
                break
        flags[device_id] = flagged
    return DetectorRun(
        "spike_mad",
        flags,
        ["ratio >= 2 against the median of the history, bucket by bucket"],
    )


def spike_mad_daily_flags(
    scenario: Scenario, threshold: float = 0.5, day_buckets: Optional[int] = None
) -> DetectorRun:
    """The same rule where a level rule belongs: on 24-hour totals.

    Daily totals remove the cycle, so the rule compares like with like. It sees
    a level change above its own cut (ratio >= 2, ``SPIKE_FACTOR``) and nothing
    else: a 60 % change stays under it, and a shift in time never moves a total.
    """
    detect_anomaly = _detect_anomaly()
    minutes = bucket_minutes_of(scenario)
    day = int(day_buckets or round(24 * 60 / minutes))
    start = _reading_window(scenario, None)
    flags: Dict[str, bool] = {}
    for row, device_id in enumerate(scenario.device_ids):
        values = scenario.energy[row]
        totals = [
            float(np.sum(values[offset : offset + day]))
            for offset in range(0, len(values), day)
        ]
        flagged = False
        for index in range(len(totals)):
            if index * day < start or index + 1 < 4:
                continue
            history = [
                {"energy_wh": total, "nonce": position}
                for position, total in enumerate(totals[: index + 1])
            ]
            if detect_anomaly(device_id, history, threshold).is_anomaly:
                flagged = True
                break
        flags[device_id] = flagged
    return DetectorRun(
        "spike_mad_daily",
        flags,
        ["the same rule on 24 h totals (ratio >= 2 against the median)"],
    )



def trend_mad_flags(scenario: Scenario, z: float = 3.0) -> DetectorRun:
    """The live AI anomaly signal (``agent.signals``), bucket by bucket."""
    start = _reading_window(scenario, None)
    minutes = bucket_minutes_of(scenario)
    flags: Dict[str, bool] = {}
    for row, device_id in enumerate(scenario.device_ids):
        flagged = False
        for t in range(start, len(scenario.starts)):
            series = ProofSeries(
                bucket_minutes=minutes,
                starts=list(scenario.starts[: t + 1]),
                values=np.asarray(scenario.energy[row, : t + 1], dtype=float),
                device_id=device_id,
                source="offline",
                raw_proof_count=t + 1,
            )
            signals = generation_signals(series, horizon_steps=1, anomaly_z=z)
            if any(signal.kind == "generation_anomaly" for signal in signals):
                flagged = True
                break
        flags[device_id] = flagged
    return DetectorRun(
        "trend_mad",
        flags,
        [f"MAD of Holt residuals, z = {z} (agent.signals.generation_anomaly)"],
    )


def evaluate(scenario: Scenario, config: Optional[ParkConfig] = None) -> Dict[str, Any]:
    """Run all three detectors and score them against the declared truth."""
    coherence, observation = coherence_flags(scenario, config)
    detectors = [
        coherence,
        spike_mad_flags(scenario),
        spike_mad_daily_flags(scenario),
        trend_mad_flags(scenario),
    ]
    rows: List[Dict[str, Any]] = []
    for index, device_id in enumerate(scenario.device_ids):
        truth = scenario.truth_of(device_id)
        rows.append(
            {
                "device_id": device_id,
                "truth": truth,
                "positive": truth != CLEAN,
                "flags": {run.name: bool(run.flags.get(device_id)) for run in detectors},
                "score": round(float(observation.scores[index]), 4),
                "reasons": observation.reasons(index),
            }
        )

    summary: Dict[str, Any] = {}
    for run in detectors:
        caught = [r["device_id"] for r in rows if r["positive"] and r["flags"][run.name]]
        missed = [
            r["device_id"] for r in rows if r["positive"] and not r["flags"][run.name]
        ]
        false_alarms = [
            r["device_id"] for r in rows if not r["positive"] and r["flags"][run.name]
        ]
        quiet = [
            r["device_id"] for r in rows if not r["positive"] and not r["flags"][run.name]
        ]
        claims = len(caught) + len(false_alarms)
        positives = len(caught) + len(missed)
        summary[run.name] = {
            "caught": caught,
            "missed": missed,
            "false_alarms": false_alarms,
            "quiet": quiet,
            # Nothing claimed is not the same as perfect: a detector that never
            # speaks has no precision, and a park with nothing wrong has no
            # recall. Both are reported as None (printed "n/a"), not as 0 or 1.
            "precision": round(len(caught) / claims, 3) if claims else None,
            "recall": round(len(caught) / positives, 3) if positives else None,
            "notes": run.notes,
        }
    return {
        "scenario": scenario.name,
        "seed": scenario.seed,
        "detectors": summary,
        "devices": rows,
        "park": observation.park_reading(),
        "warnings": observation.warnings,
    }


def format_table(result: Dict[str, Any]) -> str:
    """The matrix as a fixed-width table: truth against every detector."""
    names = list(result["detectors"])
    header = f"{'device':10} {'truth':13} " + " ".join(f"{name:>10}" for name in names)
    lines = [header, "-" * len(header)]
    for row in result["devices"]:
        cells = " ".join(f"{'X' if row['flags'][name] else '.':>10}" for name in names)
        lines.append(f"{row['device_id']:10} {row['truth']:13} {cells}")
    lines.append("")
    for name, data in result["detectors"].items():
        caught = len(data["caught"])
        positives = caught + len(data["missed"])
        precision = "n/a" if data["precision"] is None else f"{data['precision']:.2f}"
        recall = "n/a" if data["recall"] is None else f"{data['recall']:.2f}"
        lines.append(
            f"{name:15} caught {caught}/{positives}"
            f"  precision {precision}  recall {recall}"
            f"  false alarms {len(data['false_alarms'])}"
        )
    park = result["park"]
    lines.append(
        f"park coherence {park['coherence_mean']:.3f} (min {park['coherence_min']:.3f}),"
        f" silent {park['silent_fraction']:.3f}"
    )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scenario", choices=["mixed", "clean"], default="mixed")
    args = parser.parse_args(argv)
    scenario = build_scenario(seed=args.seed, scenario=args.scenario)
    print(format_table(evaluate(scenario)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
