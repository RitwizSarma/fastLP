"""Independent regression checks for Astra findings 11--13."""

import numpy as np
import pandas as pd
import pytest

from fastlp import LocalProjection


def panel():
    rng = np.random.default_rng(91)
    return pd.DataFrame({
        "unit": np.repeat(np.arange(5), 16),
        "time": np.tile(np.arange(16), 5),
        "x": rng.normal(size=80),
        "y": rng.normal(size=80),
    })


@pytest.mark.parametrize("gapped", [False, True])
@pytest.mark.parametrize("sample", ["common", "per_horizon"])
@pytest.mark.parametrize("batch_size", [32, 1])
def test_cumulative_excludes_unrelated_large_history(gapped, sample, batch_size, monkeypatch):
    monkeypatch.setattr("fastlp.estimator._HORIZON_BATCH_SIZE", batch_size)
    data = panel()
    data.loc[data.time == 0, "y"] = 1e18
    if gapped:
        data = data.loc[~((data.unit == 1) & (data.time == 8))]
    fitted = LocalProjection(
        horizons=3, covariance="hc0", response="cumulative",
        sample=sample,
    ).fit(data, outcome="y", shock="x", shock_lags=1, unit="unit", time="time")
    lookup = data.set_index(["unit", "time"])
    for h in range(4):
        x_rows, y_rows, row_ids = [], [], []
        for row in data.itertuples():
            required_horizon = 3 if sample == "common" else h
            if (row.unit, row.time - 1) not in lookup.index or any(
                (row.unit, row.time + step) not in lookup.index
                for step in range(required_horizon + 1)
            ):
                continue
            row_ids.append(row.Index)
            x_rows.append([1, row.x, lookup.loc[(row.unit, row.time - 1), "x"]])
            y_rows.append(sum(
                lookup.loc[(row.unit, row.time + step), "y"] for step in range(h + 1)
            ))
        x, y = np.array(x_rows), np.array(y_rows)
        inverse = np.linalg.pinv(x)
        beta = inverse @ y
        residual = y - x @ beta
        covariance = (inverse * residual) @ (inverse * residual).T
        np.testing.assert_array_equal(fitted.sample_index_by_horizon_[h], row_ids)
        np.testing.assert_allclose(fitted.coef_[h], beta, atol=1e-12)
        np.testing.assert_allclose(fitted.covariance_[h], covariance, atol=1e-12)


@pytest.mark.parametrize("covariance", [
    "homoskedastic", "hc0", "hc1", "hc2", "hc3", "cluster", "hac", "driscoll_kraay",
])
def test_nonfinite_covariance_is_rejected(covariance):
    data = panel()
    data.y *= 1e200
    model = LocalProjection(horizons=0, covariance=covariance, few_cluster_threshold=None)
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="numerical range exceeded"):
            model.fit(
                data, outcome="y", shock="x", unit="unit", time="time",
                cluster="unit" if covariance == "cluster" else None,
            )
    assert not hasattr(model, "coef_")


def test_overflow_in_cumulative_outcomes_is_rejected():
    data = panel()
    data.y = 1e308
    model = LocalProjection(horizons=3, covariance="hc0", response="cumulative")
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="computing regression outcomes"):
            model.fit(data, outcome="y", shock="x", unit="unit", time="time")
    assert not hasattr(model, "coef_")


@pytest.mark.parametrize("categorical", ["categorical", "enum"])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("gapped", [False, True])
def test_polars_categorical_lags_preserve_keys(categorical, lazy, gapped):
    pl = pytest.importorskip("polars")
    data = panel()
    if gapped:
        data = data.loc[data.time != 5]
    data = data.copy()
    data.unit = data.unit.astype(str)
    dtype = pl.Categorical if categorical == "categorical" else pl.Enum([str(i) for i in range(5)])
    alternate = pl.DataFrame(data.to_dict("list")).with_columns(pl.col("unit").cast(dtype))
    if lazy:
        alternate = alternate.lazy()
    options = dict(outcome="y", shock="x", outcome_lags=1, unit="unit", time="time")
    expected = LocalProjection(horizons=2, covariance="hc0").fit(data, **options)
    actual = LocalProjection(horizons=2, covariance="hc0").fit(alternate, **options)
    np.testing.assert_allclose(actual.coef_, expected.coef_, atol=1e-12)
    np.testing.assert_allclose(actual.covariance_, expected.covariance_, atol=1e-12)
    np.testing.assert_array_equal(actual.n_obs_by_horizon_, expected.n_obs_by_horizon_)
