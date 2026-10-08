"""Exact identifier preservation and allocation-budget contract regressions."""

import numpy as np
import pandas as pd
import pytest

from fastlp import LocalProjection
from fastlp._frame import prepare_panel
from fastlp._covariance import _lag_pairs


def panel():
    rng = np.random.default_rng(1519)
    return pd.DataFrame({
        "unit": np.repeat(np.arange(4), 12), "time": np.tile(np.arange(12), 4),
        "y": rng.normal(size=48), "x": rng.normal(size=48),
    })


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_overlapping_model_columns_preserve_exact_clock_and_categories(backend):
    data = panel()
    data.unit += 2**53
    data.time += 2**53
    # Numeric-looking identifiers with distinct spellings must stay distinct.
    data["category"] = np.where(data.index % 2 == 0, "01", "1")
    source = data
    if backend == "polars":
        pl = pytest.importorskip("polars")
        source = pl.DataFrame(data.to_dict("list"))
    frame, _ = prepare_panel(
        source, ("y", "unit", "time", "x", "category"),
        ("y", "unit", "time", "x", "category"),
    )
    np.testing.assert_array_equal(frame.column("unit"), data.unit)
    np.testing.assert_array_equal(frame.column("time"), data.time)
    assert frame.factorize(("unit",))[1] == 4
    assert frame.factorize(("category",))[1] == 2
    leads, _ = frame.lead_positions("unit", "time", 1)
    np.testing.assert_array_equal(leads[:11, 1], np.arange(1, 12))
    assert leads[11, 1] == -1
    numerical = frame.matrix(("unit", "time", "category"), dtype=np.float64)
    assert numerical.dtype == np.float64
    np.testing.assert_array_equal(numerical[:, 2], np.ones(len(data)))


def test_tiny_budget_fails_before_panel_copy_or_alignment(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("panel preparation must not run")
    monkeypatch.setattr("fastlp.estimator.prepare_panel", forbidden)
    with pytest.raises(ValueError, match="memory_budget=.*planned minimum"):
        LocalProjection(horizons=10000, covariance="hc0", memory_budget=1024).fit(
            panel(), outcome="y", shock="x", unit="unit", time="time",
        )


@pytest.mark.parametrize("covariance", ["hc0", "hac"])
def test_budget_boundary_and_streaming_preserve_estimates(covariance):
    data = panel().drop(index=[5, 27])
    options = dict(outcome="y", shock="x", shock_lags=1, unit="unit", time="time")
    config = dict(horizons=3, covariance=covariance, response="cumulative")
    baseline = LocalProjection(**config).fit(data, **options)
    planned = LocalProjection(**config, memory_budget="10MB").fit(data, **options)
    minimum = planned.memory_diagnostics_["minimum_bytes"]
    actual = LocalProjection(**config, memory_budget=minimum).fit(data, **options)
    assert actual.memory_diagnostics_["batch_size"] == 1
    assert actual.memory_diagnostics_["planned_peak_bytes"] == minimum
    np.testing.assert_allclose(actual.coef_, baseline.coef_, atol=1e-12)
    np.testing.assert_allclose(actual.covariance_, baseline.covariance_, atol=1e-12)
    with pytest.raises(ValueError, match="planned minimum"):
        LocalProjection(**config, memory_budget=minimum - 1).fit(data, **options)


def test_lazy_polars_budget_and_string_model_columns():
    pl = pytest.importorskip("polars")
    data = panel()
    data.x = data.x.astype(str)
    options = dict(outcome="y", shock="x", shock_lags=1, unit="unit", time="time")
    expected = LocalProjection(horizons=2, covariance="hc0").fit(data, **options)
    for source in (pl.DataFrame(data.to_dict("list")), pl.DataFrame(data.to_dict("list")).lazy()):
        actual = LocalProjection(horizons=2, covariance="hc0", memory_budget="1MB").fit(
            source, **options,
        )
        np.testing.assert_allclose(actual.coef_, expected.coef_, atol=1e-12)
        assert actual.memory_diagnostics_["planned_peak_bytes"] <= 1024**2


def test_hac_skips_impossible_lags_without_changing_bandwidth():
    data = panel()
    pairs = _lag_pairs(data.unit.to_numpy(), data.time.to_numpy(), 10**9)
    assert len(pairs) == 12
    model = LocalProjection(
        horizons=0, covariance="hac", hac_lags=10**9,
        hac_debias=False, memory_budget="1MB",
    ).fit(data, outcome="y", shock="x", unit="unit", time="time")
    assert model.covariance_diagnostics_[0]["bandwidth"] == 10**9
    x = np.column_stack([np.ones(len(data)), data.x])
    residual = data.y.to_numpy() - x @ np.linalg.lstsq(x, data.y, rcond=None)[0]
    scores = x * residual[:, None]
    meat = np.zeros((2, 2))
    for i in range(len(data)):
        for j in range(len(data)):
            if data.unit.iloc[i] == data.unit.iloc[j]:
                weight = 1 - abs(int(data.time.iloc[i]) - int(data.time.iloc[j])) / (10**9 + 1)
                meat += weight * np.outer(scores[i], scores[j])
    inverse = np.linalg.inv(x.T @ x)
    np.testing.assert_allclose(model.covariance_[0], inverse @ meat @ inverse, atol=1e-12)
