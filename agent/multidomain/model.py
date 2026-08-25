"""Multi-domain model — shared backbone + per-domain heads.

Architecture (GLOBAL_AI_ARCHITECTURE §9.3), siamese-style:

- **Backbone** (shared, all domains): one hidden layer over the *lagged
  window of a single series* — it learns temporal regularities
  (autocorrelation, seasonality) that generalize across any domain;
- **Domain heads**: each domain has a small linear head that maps the shared
  temporal features to that domain's columns and scale.

This gives true transfer learning: a new domain reuses the trained backbone
and only needs to fit its head (few-shot), and failure isolation — each
column predicts from its own window through the shared backbone, so a broken
feed does not corrupt other domains.

Implementation: numpy-only, one tanh hidden layer, SGD.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np


class MultiDomainModel:
    """Shared temporal backbone + per-domain linear heads (numpy SGD)."""

    def __init__(
        self,
        lag: int = 3,
        hidden: int = 8,
        lr: float = 0.01,
        epochs: int = 200,
        seed: int = 0,
    ) -> None:
        self.lag = lag
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.rng = np.random.default_rng(seed)

        self.columns: List[str] = []
        self.domain_of: List[str] = []
        self.domains: List[str] = []
        self.column_indices: Dict[str, List[int]] = {}
        self.means: np.ndarray = np.array([])
        self.stds: np.ndarray = np.array([])

        # backbone: (hidden, lag + 1); head[d]: (n_domain_cols, hidden)
        self.Wb: np.ndarray = np.array([])
        self.heads: Dict[str, np.ndarray] = {}
        self.head_biases: Dict[str, np.ndarray] = {}

    # ── normalization ───────────────────────────────────────────────────────

    def _normalize(self, matrix: np.ndarray) -> np.ndarray:
        if self.means.size == 0:
            self.means = matrix.mean(axis=0)
            self.stds = np.maximum(matrix.std(axis=0), 1e-9)
        return (matrix - self.means) / self.stds

    def _denormalize(self, matrix: np.ndarray) -> np.ndarray:
        return matrix * self.stds + self.means

    # ── dataset & forward ───────────────────────────────────────────────────

    def _dataset(self, x_norm: np.ndarray):
        """Per-column lag windows -> next row. Returns (Xw, Y).

        ``Xw`` has shape (n, lag, K): window ``t-lag+1..t`` for each column.
        """
        t, k = x_norm.shape
        xs = []
        ys = []
        for t0 in range(self.lag, t - 1):
            xs.append(x_norm[t0 - self.lag + 1 : t0 + 1])  # (lag, K)
            ys.append(x_norm[t0 + 1])
        if not xs:
            return np.zeros((0, self.lag, k)), np.zeros((0, k))
        return np.array(xs), np.array(ys)

    def _predict_batch(self, xw: np.ndarray) -> np.ndarray:
        """Predict next row for a batch of lag windows ``(n, lag, K)``."""
        n = xw.shape[0]
        pred = np.zeros((n, len(self.columns)))
        for j, col in enumerate(self.columns):
            window = xw[:, :, j]  # (n, lag)
            xb = np.concatenate([window, np.ones((n, 1))], axis=1)
            h = np.tanh(xb @ self.Wb.T)  # (n, hidden)
            d = self.domain_of[j]
            row = self.column_indices[d].index(j)
            pred[:, j] = h @ self.heads[d][row] + self.head_biases[d][row]
        return pred

    # ── fitting ─────────────────────────────────────────────────────────────

    def _init_params(self) -> None:
        self.Wb = self.rng.normal(0.0, 0.1, (self.hidden, self.lag + 1))
        for domain, idxs in self.column_indices.items():
            self.heads[domain] = self.rng.normal(0.0, 0.1, (len(idxs), self.hidden))
            self.head_biases[domain] = np.zeros(len(idxs))

    def _sgd_step(self, xw: np.ndarray, yb: np.ndarray) -> None:
        """One SGD step over a batch (shared backbone, per-domain heads)."""
        n = xw.shape[0]
        pred = self._predict_batch(xw)
        err = pred - yb  # (n, K)
        d_pred = 2.0 * err / max(1, n)

        for j, col in enumerate(self.columns):
            d = self.domain_of[j]
            row = self.column_indices[d].index(j)

            window = xw[:, :, j]
            xb = np.concatenate([window, np.ones((n, 1))], axis=1)
            h = np.tanh(xb @ self.Wb.T)  # (n, hidden)

            # head gradient
            self.heads[d][row] -= self.lr * (h.T @ d_pred[:, j])
            self.head_biases[d][row] -= self.lr * d_pred[:, j].sum()

            # backbone gradient (shared across all columns)
            w_row = self.heads[d][row]  # (hidden,)
            dh = (d_pred[:, j, None] * w_row[None, :]) * (1.0 - h ** 2)  # (n, hidden)
            self.Wb -= self.lr * (dh.T @ xb)

    def fit(
        self,
        matrix: np.ndarray,
        columns: Sequence[str],
        domain_of: Sequence[str],
        epochs: Optional[int] = None,
        batch_size: int = 16,
    ) -> float:
        """Train the shared backbone + all domain heads on the whole matrix."""
        matrix = np.asarray(matrix, dtype=float)
        self.columns = list(columns)
        self.domain_of = list(domain_of)
        self.domains = sorted(set(self.domain_of))
        self.column_indices = {d: [] for d in self.domains}
        for i, d in enumerate(self.domain_of):
            self.column_indices[d].append(i)

        x_norm = self._normalize(matrix)
        xw_all, yb_all = self._dataset(x_norm)
        if xw_all.shape[0] < 2:
            return 0.0

        self._init_params()
        epochs = epochs or self.epochs
        n = xw_all.shape[0]
        for _ in range(epochs):
            order = self.rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                self._sgd_step(xw_all[idx], yb_all[idx])

        self.train_loss = float(np.mean((self._predict_batch(xw_all) - yb_all) ** 2))
        return self

    def fit_domain(
        self,
        matrix: np.ndarray,
        columns: Sequence[str],
        domain: str,
        epochs: Optional[int] = None,
        batch_size: int = 16,
    ) -> float:
        """Few-shot: train only this domain's head, backbone frozen.

        The new domain inherits the temporal regularities the backbone learned
        from the other domains — transfer learning on small data.
        """
        matrix = np.asarray(matrix, dtype=float)
        self.columns = list(columns)
        self.domain_of = [domain] * len(columns)
        self.domains = [domain]
        self.column_indices = {domain: list(range(len(columns)))}

        # The new domain has its own scale: re-normalize by its statistics.
        self.means = np.array([])
        self.stds = np.array([])

        x_norm = self._normalize(matrix)
        xw_all, yb_all = self._dataset(x_norm)
        if xw_all.shape[0] < 2:
            return 0.0

        self.heads[domain] = self.rng.normal(0.0, 0.1, (len(columns), self.hidden))
        self.head_biases[domain] = np.zeros(len(columns))

        epochs = epochs or self.epochs
        n = xw_all.shape[0]
        for _ in range(epochs):
            order = self.rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                # Head-only update: the backbone is frozen.
                n_batch = idx.shape[0]
                pred = self._predict_batch(xw_all[idx])
                err = pred - yb_all[idx]
                d_pred = 2.0 * err / max(1, n_batch)
                for j, col in enumerate(self.columns):
                    window = xw_all[idx, :, j]
                    xb = np.concatenate([window, np.ones((n_batch, 1))], axis=1)
                    h = np.tanh(xb @ self.Wb.T)
                    self.heads[domain][j] -= self.lr * (h.T @ d_pred[:, j])
                    self.head_biases[domain][j] -= self.lr * d_pred[:, j].sum()

        self.train_loss = float(np.mean((self._predict_batch(xw_all) - yb_all) ** 2))
        return self

    # ── inference ───────────────────────────────────────────────────────────

    def predict_domain(self, domain: str, matrix: np.ndarray) -> np.ndarray:
        """One-step-ahead forecast (denormalized) for one domain."""
        x_norm = self._normalize(np.asarray(matrix, dtype=float))
        pred_norm = np.zeros(len(self.column_indices[domain]))
        for j, col in enumerate(self.column_indices[domain]):
            window = x_norm[-self.lag :, col]
            xb = np.append(window, 1.0)
            h = np.tanh(xb @ self.Wb.T)
            pred_norm[j] = self.heads[domain][j] @ h + self.head_biases[domain][j]
        return pred_norm * self.stds[self.column_indices[domain]] + self.means[self.column_indices[domain]]

    # ── serialization ───────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lag": self.lag,
            "hidden": self.hidden,
            "columns": self.columns,
            "domain_of": self.domain_of,
            "means": [float(v) for v in self.means],
            "stds": [float(v) for v in self.stds],
            "Wb": self.Wb.tolist(),
            "heads": {d: w.tolist() for d, w in self.heads.items()},
            "head_biases": {d: b.tolist() for d, b in self.head_biases.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiDomainModel":
        model = cls(lag=data.get("lag", 3), hidden=data.get("hidden", 8))
        model.columns = data.get("columns", [])
        model.domain_of = data.get("domain_of", [])
        model.domains = sorted(set(model.domain_of))
        model.column_indices = {d: [] for d in model.domains}
        for i, d in enumerate(model.domain_of):
            model.column_indices[d].append(i)
        model.means = np.array(data.get("means", []))
        model.stds = np.array(data.get("stds", []))
        model.Wb = np.array(data.get("Wb", []))
        model.heads = {d: np.array(w) for d, w in data.get("heads", {}).items()}
        model.head_biases = {d: np.array(b) for d, b in data.get("head_biases", {}).items()}
        return model


