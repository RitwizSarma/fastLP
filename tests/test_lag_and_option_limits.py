"""Public boundary checks for Astra review findings 17 and 18."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, t

from fastlp import LocalProjection
from fastlp._frame import prepare_panel


def panel():
    rng = np.random.default_rng(1718)
    return pd.DataFrame({
        "unit": np.repeat(np.arange(5), 8),
        "time": np.tile(np.arange(8), 5),
        "y": rng.normal(size=40),
        "x": rng.normal(size=40),
    })


def test_lag_limits_fail_before_panel_preparation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid lags must be rejected before preparing data")

    monkeypatch.setattr("fastlp.estimator.prepare_panel", forbidden)
    for specification in (2**63, [2**63], [2**65], 10_001):
        with pytest.raises(ValueError, match="lag limit|scalar lag count"):
            LocalProjection(horizons=0, covariance="hc0").fit(
                panel(), outcome="y", shock="x", outcome_lags=specification,
                unit="unit", time="time",
            )


def test_largest_supported_sparse_lag_matches_exact_int64_dates():
    smallest = np.iinfo(np.int64).min
    largest_lag = np.iinfo(np.int64).max
    data = pd.DataFrame({
        "unit": [0, 0, 1, 1],
        "time": np.array([smallest, -1, smallest, -1], dtype=np.int64),
        "y": [2.0, 3.0, 5.0, 7.0],
        "x": [1.0, 2.0, 3.0, 4.0],
    })
    sources = [data]
    try:
        import polars as pl
    except ImportError:
        pass
    else:
        sources.append(pl.DataFrame(data.to_dict("list")))
    for source in sources:
        frame, _ = prepare_panel(
            source, ("y", "unit", "time", "x"), ("y", "x")
        )
        lagged, names, valid, path = frame.add_lags(
            "unit", "time", [("y", largest_lag)]
        )
        assert names == (f"y_lag{largest_lag}",)
        assert path == "indexed"
        np.testing.assert_array_equal(valid, [False, True, False, True])
        np.testing.assert_allclose(
            lagged.column(names[0]), [np.nan, 2.0, np.nan, 5.0],
            equal_nan=True,
        )
        with pytest.raises(ValueError, match="int64 lag limit"):
            frame.add_lags("unit", "time", [("y", 2**63)])

    dense = panel().iloc[:16].copy()
    dense_sources = [dense]
    if len(sources) == 2:
        dense_sources.append(pl.DataFrame(dense.to_dict("list")))
    for source in dense_sources:
        frame, _ = prepare_panel(
            source, ("y", "unit", "time", "x"), ("y", "x")
        )
        lagged, names, valid, path = frame.add_lags(
            "unit", "time", [("y", largest_lag)]
        )
        assert path == "consecutive_shift"
        assert not valid.any()
        assert np.isnan(lagged.column(names[0])).all()


def test_invalid_numerical_options_fail_at_construction():
    invalid = (
        ("horizons", True), ("horizons", 1.5),
        ("hac_lags", True), ("hac_lags", 1.5),
        ("max_iter", True), ("max_iter", 1.5),
        ("demean_tol", float("nan")), ("demean_tol", float("inf")),
        ("demean_tol", 0), ("demean_tol", True),
        ("alpha", np.nextafter(0.0, 1.0)),
    )
    for option, value in invalid:
        with pytest.raises(ValueError, match=option):
            LocalProjection(**{"horizons": 0, option: value})
    accepted = LocalProjection(
        horizons=np.int64(1), hac_lags=np.int64(0), max_iter=np.int64(10)
    )
    assert (accepted.horizons, accepted.hac_lags, accepted.max_iter) == (1, 0, 10)


def test_tiny_alpha_matches_independent_normal_and_t_tail_quantiles():
    data = panel()
    alpha = 1e-20
    options = dict(outcome="y", shock="x", unit="unit", time="time")
    normal = LocalProjection(horizons=0, covariance="hc0", alpha=alpha).fit(
        data, **options
    )
    clustered = LocalProjection(
        horizons=0, covariance="cluster", alpha=alpha,
        few_cluster_threshold=None,
    ).fit(data, cluster="unit", **options)
    expected = (norm.isf(alpha / 2), t.isf(alpha / 2, 4))
    np.testing.assert_allclose(
        [normal.critical_values_[0], clustered.critical_values_[0]], expected
    )
    for result in (normal, clustered):
        assert np.isfinite(result.conf_int_).all()
        assert result.conf_int_[0, 1, 1] > result.coef_[0, 1]
