"""Turn collected feed series into a training matrix (domain-agnostic)."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def build_matrix(samples: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
    """Build a chronological ``(T, K)`` matrix from feed observations.

    ``samples`` is a list of dicts with ``ts`` and ``metrics`` (e.g. produced
    by ``agent.digital_feeds.FeedResult.to_dict``). Columns are all metric
    names, in first-seen order. Missing values are forward-filled per column
    (last observed value), zeros for leading gaps.
    """
    if not samples:
        return np.zeros((0, 0)), []

    rows: Dict[str, Dict[str, float]] = {}
    columns: List[str] = []

    for sample in sorted(samples, key=lambda s: s.get("ts", "")):
        metrics = sample.get("metrics") or {}
        for key in metrics:
            if key not in columns:
                columns.append(key)
        rows.setdefault(sample.get("ts", ""), {}).update(
            {k: float(v) for k, v in metrics.items()}
        )

    timestamps = sorted(rows)
    matrix = np.zeros((len(timestamps), len(columns)))

    last: Dict[str, float] = {}
    for i, ts in enumerate(timestamps):
        for j, col in enumerate(columns):
            value = rows[ts].get(col)
            if value is None:
                matrix[i, j] = last.get(col, 0.0)
            else:
                matrix[i, j] = value
                last[col] = value

    return matrix, columns
