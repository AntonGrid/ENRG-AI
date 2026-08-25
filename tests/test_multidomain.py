"""Multi-domain backbone + heads tests (agent.multidomain)."""

import numpy as np

from agent.multidomain import MultiDomainModel


def _series(n, base, a, seasonal, noise=0.2, seed=0):
    """AR(1) series with a shared daily seasonality (the transferable part)."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = base + seasonal * np.sin(2 * np.pi * i / 24) + a * y[i - 1] + rng.normal(0, noise)
    return y


def _domain_mse(model, domain, series):
    errs = []
    for t in range(model.lag + 1, len(series)):
        pred = model.predict_domain(domain, series[:t].reshape(-1, 1))[0]
        errs.append((pred - series[t]) ** 2)
    return float(np.mean(errs))


def test_fit_and_predict_each_domain():
    a = _series(240, 10.0, 0.6, 2.0, seed=1)
    b = _series(240, 20.0, 0.7, 2.0, seed=2)
    matrix = np.column_stack([a, b])

    model = MultiDomainModel(lag=3, hidden=8, epochs=300).fit(
        matrix, ["alpha", "beta"], ["alpha", "beta"]
    )

    pred_a = model.predict_domain("alpha", matrix[:-1])[0]
    assert abs(pred_a - matrix[-1, 0]) / max(abs(matrix[-1, 0]), 1e-9) < 0.25

    pred_b = model.predict_domain("beta", matrix[:-1])[0]
    assert abs(pred_b - matrix[-1, 1]) / max(abs(matrix[-1, 1]), 1e-9) < 0.25


def test_domain_heads_are_independent():
    a = _series(200, 10.0, 0.6, 2.0, seed=1)
    b = _series(200, 20.0, 0.7, 2.0, seed=2)
    matrix = np.column_stack([a, b])
    model = MultiDomainModel(lag=3, hidden=8, epochs=200).fit(
        matrix, ["a", "b"], ["alpha", "beta"]
    )
    # Predictions for one domain must not depend on the other domain's head.
    saved = model.heads["beta"].copy()
    model.heads["beta"] *= 0.0
    pred_alpha = model.predict_domain("alpha", matrix[:-1])
    model.heads["beta"] = saved
    assert np.allclose(pred_alpha, model.predict_domain("alpha", matrix[:-1]))


def test_transfer_beats_scratch_on_few_shots():
    # Two source domains build the backbone.
    a = _series(240, 10.0, 0.6, 2.0, seed=1)
    b = _series(240, 20.0, 0.7, 2.0, seed=2)
    backbone = MultiDomainModel(lag=3, hidden=8, epochs=300).fit(
        np.column_stack([a, b]), ["a", "b"], ["alpha", "beta"]
    )

    # A new domain, similar dynamics, only 30 training points.
    c_train = _series(30, 15.0, 0.65, 2.0, seed=3)
    c_test = _series(60, 15.0, 0.65, 2.0, seed=4)

    transfer = MultiDomainModel.from_dict(backbone.to_dict())
    transfer.fit_domain(c_train.reshape(-1, 1), ["c"], "gamma", epochs=300)
    transfer_mse = _domain_mse(transfer, "gamma", c_test)

    scratch = MultiDomainModel(lag=3, hidden=8, epochs=300)
    scratch.fit(c_train.reshape(-1, 1), ["c"], ["gamma"])
    scratch_mse = _domain_mse(scratch, "gamma", c_test)

    assert transfer_mse < scratch_mse, (
        f"transfer {transfer_mse:.4f} vs scratch {scratch_mse:.4f}"
    )


def test_serialization_roundtrip():
    a = _series(120, 10.0, 0.6, 2.0, seed=1)
    b = _series(120, 20.0, 0.7, 2.0, seed=2)
    model = MultiDomainModel(lag=3, hidden=6, epochs=100).fit(
        np.column_stack([a, b]), ["a", "b"], ["alpha", "beta"]
    )
    restored = MultiDomainModel.from_dict(model.to_dict())
    assert restored.columns == ["a", "b"]
    assert restored.domains == ["alpha", "beta"]
    assert np.allclose(restored.Wb, model.Wb)
    assert np.allclose(restored.predict_domain("alpha", np.column_stack([a, b])[:-1]),
                       model.predict_domain("alpha", np.column_stack([a, b])[:-1]))
