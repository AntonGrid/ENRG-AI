"""Domain-agnostic time-series model (digital layer).

A lightweight multi-domain model that works on *any* normalized numeric
series (weather, finance, news activity, chain metrics, demography…):

- **Trends**: for every column (domain metric) it learns an AR model with
  lags of *all* columns — so cross-domain links are baked into the weights;
- **Anomalies**: MAD-based detection on one-step-ahead residuals;
- **Cross-domain links**: the correlation matrix of the series.

Implementation is dependency-light (numpy only) and fully serializable, so
its weights can be shipped as a signed federated contribution
(``agent.fed.protocol``).
"""
from __future__ import annotations

from statistics import median
from typing import Any, Dict, List, Optional

import numpy as np


class TimeSeriesModel:
    """Multi-column linear AR model with lags and ridge regularization."""

    def __init__(self, lag: int = 3, ridge: float = 1e-3) -> None:
        self.lag = lag
        self.ridge = ridge
        self.columns: List[str] = []
        self.means: np.ndarray = np.array([])
        self.stds: np.ndarray = np.array([])
        # column name -> {"w": [...], "intercept": float}
        self.weights: Dict[str, Dict[str, Any]] = {}
        # column name -> in-sample one-step residuals (for anomaly MAD)
        self._residuals: Dict[str, List[float]] = {}

    # ── normalization ───────────────────────────────────────────────────────

    def _normalize(self, matrix: np.ndarray) -> np.ndarray:
        if self.means.size == 0:
            self.means = matrix.mean(axis=0)
            self.stds = np.maximum(matrix.std(axis=0), 1e-9)
        return (matrix - self.means) / self.stds

    def _denormalize(self, matrix: np.ndarray) -> np.ndarray:
        return matrix * self.stds + self.means

    # ── fitting ─────────────────────────────────────────────────────────────

    def _ridge_solve(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        eye = np.eye(a.shape[1]) * self.ridge
        return np.linalg.solve(a.T @ a + eye, a.T @ b)

    def fit(
        self,
        matrix: np.ndarray,
        columns: Optional[List[str]] = None,
    ) -> "TimeSeriesModel":
        """Fit one AR model per column over lagged features of all columns.

        Features for column ``d`` at step ``t`` are the flattened window of
        the last ``lag`` rows across every column, plus a bias term — so
        cross-domain influence is learned as ordinary regression weights.
        """
        matrix = np.asarray(matrix, dtype=float)
        k = matrix.shape[1]
        if columns is None:
            columns = [f"col_{i}" for i in range(k)]
        elif len(columns) != k:
            raise ValueError("columns length must match matrix width")

        self.columns = list(columns)
        x_norm = self._normalize(matrix)
        t, _ = x_norm.shape
        n_features = self.lag * k + 1  # +1 for the bias term

        for d in range(k):
            x_rows: List[np.ndarray] = []
            y_vals: List[float] = []
            for t0 in range(self.lag, t - 1):
                feat = np.append(
                    x_norm[t0 - self.lag + 1 : t0 + 1].flatten(),
                    1.0,  # bias
                )
                x_rows.append(feat)
                y_vals.append(float(x_norm[t0 + 1, d]))

            if len(y_vals) < 2:
                self.weights[self.columns[d]] = {
                    "w": [0.0] * (n_features - 1),
                    "intercept": 0.0,
                }
                self._residuals[self.columns[d]] = []
                continue

            w = self._ridge_solve(np.array(x_rows), np.array(y_vals))
            residuals = np.array(y_vals) - np.array(x_rows) @ w
            self.weights[self.columns[d]] = {
                "w": [float(v) for v in w[: n_features - 1]],
                "intercept": float(w[n_features - 1]),
            }
            self._residuals[self.columns[d]] = [float(v) for v in residuals]

        return self

    # ── inference ───────────────────────────────────────────────────────────

    def _predict_step(self, x_norm: np.ndarray) -> np.ndarray:
        """One-step forecast for the next row given normalized history."""
        window = np.append(x_norm[-self.lag :].flatten(), 1.0)
        out = np.zeros(len(self.columns))
        for d, col in enumerate(self.columns):
            spec = self.weights[col]
            out[d] = float(np.asarray(spec["w"]) @ window[:-1]) + spec["intercept"]
        return out

    def predict(self, matrix: np.ndarray, steps: int = 1) -> np.ndarray:
        """Forecast ``steps`` rows ahead (recursive, denormalized)."""
        x_norm = self._normalize(np.asarray(matrix, dtype=float))
        for _ in range(steps):
            nxt = self._predict_step(x_norm)
            x_norm = np.vstack([x_norm, nxt])
        return self._denormalize(x_norm[-steps:])

    def predict_next(self, matrix: np.ndarray) -> np.ndarray:
        """Alias: one-step-ahead forecast for the row after ``matrix``."""
        return self.predict(matrix, steps=1)[0]

    # ── anomalies & cross-domain links ─────────────────────────────────────

    def detect_anomaly(
        self,
        matrix: np.ndarray,
        z_threshold: float = 3.0,
    ) -> Dict[str, float]:
        """Flag the last row's columns whose one-step residual is an outlier.

        Returns ``{column: residual}`` for anomalous columns only.
        """
        x_norm = self._normalize(np.asarray(matrix, dtype=float))
        if x_norm.shape[0] <= self.lag:
            return {}
        pred = self._predict_step(x_norm[:-1])
        actual = x_norm[-1]
        anomalies: Dict[str, float] = {}
        for d, col in enumerate(self.columns):
            res = float(actual[d] - pred[d])
            resid = self._residuals.get(col, [])
            if len(resid) < 3:
                continue
            mad = median([abs(v) for v in resid])
            scale = 1.4826 * mad if mad > 0 else 1.4826 * 1e-6
            if abs(res) > z_threshold * scale:
                anomalies[col] = round(res, 6)
        return anomalies

    def last_step_mse(self, matrix: np.ndarray) -> float:
        """Normalized one-step-ahead MSE on the last row.

        Computed in z-score space so the metric is comparable across domains
        of wildly different scales (GDP vs. temperature).
        """
        x_norm = self._normalize(np.asarray(matrix, dtype=float))
        if x_norm.shape[0] <= self.lag + 1:
            return 0.0
        pred = self._predict_step(x_norm[:-1])
        return float(np.mean((pred - x_norm[-1]) ** 2))

    def correlations(self, matrix: np.ndarray) -> np.ndarray:
        """Pearson correlation matrix over the *first differences*.

        Differencing removes shared monotonic trends that would otherwise
        produce spurious 1.0 correlations (GDP vs. block height). Columns
        with zero variance produce ``NaN``; those are zeroed.
        """
        x_norm = self._normalize(np.asarray(matrix, dtype=float))
        diff = np.diff(x_norm, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(diff.T)
        return np.nan_to_num(corr, nan=0.0)

    # ── federated contribution support ────────────────────────────────────

    def flatten_weights(self) -> List[float]:
        """Flat weight vector for a federated contribution (all columns)."""
        flat: List[float] = []
        for col in self.columns:
            spec = self.weights.get(col, {"w": [], "intercept": 0.0})
            flat.extend(spec["w"])
            flat.append(spec["intercept"])
        return [round(float(v), 6) for v in flat]

    # ── serialization ──────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lag": self.lag,
            "ridge": self.ridge,
            "columns": list(self.columns),
            "means": [float(v) for v in self.means],
            "stds": [float(v) for v in self.stds],
            "weights": {
                col: {"w": spec["w"], "intercept": spec["intercept"]}
                for col, spec in self.weights.items()
            },
            "residuals": {col: list(v) for col, v in self._residuals.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimeSeriesModel":
        model = cls(lag=data.get("lag", 3), ridge=data.get("ridge", 1e-3))
        model.columns = list(data.get("columns", []))
        model.means = np.array(data.get("means", []))
        model.stds = np.array(data.get("stds", []))
        model.weights = data.get("weights", {})
        model._residuals = data.get("residuals", {})
        return model

