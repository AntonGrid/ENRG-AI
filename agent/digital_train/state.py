"""Persistent state for the digital self-training loop.

Holds the accumulated feed history (so online mode trains on real, growing
data across runs) plus the last trained model snapshot and metrics.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.digital_train.model import TimeSeriesModel


@dataclass
class DigitalState:
    """Serializable state of one digital training node."""

    history: List[Dict[str, Any]] = field(default_factory=list)
    model: Optional[Dict[str, Any]] = None
    round: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def trained(self) -> bool:
        return self.model is not None

    def load_model(self) -> Optional[TimeSeriesModel]:
        if self.model is None:
            return None
        return TimeSeriesModel.from_dict(self.model)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "history": self.history,
            "model": self.model,
            "round": self.round,
            "metrics": self.metrics,
        }


def save(path: str, state: DigitalState) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)


def load(path: str) -> DigitalState:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return DigitalState()
    return DigitalState(
        history=data.get("history", []),
        model=data.get("model"),
        round=data.get("round", 0),
        metrics=data.get("metrics", {}),
    )
