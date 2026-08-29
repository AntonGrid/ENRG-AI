"""Energy generation forecasting for the DePIN pilot (ENRG-AI).

Lives on the real proof stream produced by the ESP32 counter and persisted by
the ENRG oracle (ADR-0010 data bridge, ``/api/v1/proofs``). Aggregates proofs
into fixed-size buckets and fits a lightweight state-space trend model
(Holt's method, numpy only) with horizon-widening prediction intervals.

For heavier zero-shot forecasting on a beefier machine, the TimesFM skill in
``../../skills/timesfm-forecasting`` documents the drop-in alternative
(``scripts/forecast_csv.py``); the CLI here writes the same CSV contract.
"""

from agent.forecast.energy import (
    Proof,
    ProofSeries,
    aggregate_proofs,
    fetch_oracle_proofs,
)
from agent.forecast.model import (
    ForecastResult,
    HoltTrend,
    forecast_energy,
)

__all__ = [
    "Proof",
    "ProofSeries",
    "ForecastResult",
    "HoltTrend",
    "aggregate_proofs",
    "fetch_oracle_proofs",
    "forecast_energy",
]
