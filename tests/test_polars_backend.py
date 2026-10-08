"""Parity and contract tests for the optional Polars input backend."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fastlp import LocalProjection

pl = pytest.importorskip("polars")


def panel(*, unbalanced: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(20260806)
    rows = []
    for unit in range(8):
        for time in range(12):
            rows.append(
                {
                    "unit": unit,
                    "time": time,
                    "region": f"r{unit % 3}",
                    "industry": unit % 2,
                    "y": 0.4 * time + 0.2 * unit + rng.normal(),
                    "shock": rng.normal() + 0.01 * unit * time**2,
                    "control": rng.normal() + 0.02 * unit**2 * time,
                }
            )
    result = pd.DataFrame(rows)
    if unbalanced:
        result = result.loc[~((result.unit == 1) & (result.time == 4))]
        result = result.loc[~((result.unit == 6) & (result.time == 8))]
    return result


def as_polars(frame: pd.DataFrame) -> pl.DataFrame:
    """Convert without making PyArrow a test or package dependency."""
    return pl.DataFrame({column: frame[column].to_numpy() for column in frame.columns})


def fit(data, *, covariance: str = "hc1", sample: str = "common") -> LocalProjection:
    return LocalProjection(horizons=3, covariance=covariance, sample=sample).fit(
        data,
        outcome="y",
        shock="shock",
        controls=["control"],
        outcome_lags=1,
        shock_lags=[1, 2],
        unit="unit",
        time="time",
        fixed_effects=["unit", "time"],
        cluster=["unit", "region"] if covariance == "cluster" else None,
    )


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("unbalanced", [False, True])
def test_polars_matches_pandas_across_frame_and_panel_kinds(
    lazy: bool, unbalanced: bool
) -> None:
    pandas_data = panel(unbalanced=unbalanced).sample(frac=1, random_state=31)
    polars_data = as_polars(pandas_data.reset_index(drop=True))
    if lazy:
        polars_data = polars_data.lazy()
    sample = "per_horizon" if unbalanced else "common"
    expected = fit(pandas_data, sample=sample)
    actual = fit(polars_data, sample=sample)

    np.testing.assert_allclose(actual.coef_, expected.coef_, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(actual.stderr_, expected.stderr_, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(actual.covariance_, expected.covariance_, rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(actual.n_obs_by_horizon_, expected.n_obs_by_horizon_)
    assert actual.feature_names_in_.tolist() == expected.feature_names_in_.tolist()
    assert actual.demeaning_diagnostics_["input_backend"] == "polars"
    assert actual.demeaning_diagnostics_["input_was_sorted"] is False
    assert isinstance(actual.sample_index_, np.ndarray)
    assert isinstance(actual.to_frame(), pd.DataFrame)


def test_polars_original_row_positions_survive_sorting_and_pruning() -> None:
    data = panel(unbalanced=True).sample(frac=1, random_state=7).reset_index(drop=True)
    fitted = LocalProjection(horizons=2, covariance="hc1", sample="per_horizon").fit(
        as_polars(data),
        outcome="y",
        shock="shock",
        unit="unit",
        time="time",
        fixed_effects=["unit", "time"],
    )
    for horizon, positions in enumerate(fitted.sample_index_by_horizon_):
        retained = data.iloc[positions]
        keys = set(zip(data.unit, data.time, strict=True))
        assert all((row.unit, row.time + horizon) in keys for row in retained.itertuples())


def test_polars_cluster_and_cumulative_paths_match_pandas() -> None:
    pandas_data = panel()
    polars_data = as_polars(pandas_data)
    with pytest.warns(UserWarning):
        expected = fit(pandas_data, covariance="cluster")
    with pytest.warns(UserWarning):
        actual = fit(polars_data, covariance="cluster")
    np.testing.assert_allclose(actual.coef_, expected.coef_, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(actual.stderr_, expected.stderr_, rtol=1e-10, atol=1e-10)

    expected_cumulative = LocalProjection(horizons=2, covariance="hc1", response="cumulative").fit(
        pandas_data, outcome="y", shock="shock", unit="unit", time="time"
    )
    actual_cumulative = LocalProjection(horizons=2, covariance="hc1", response="cumulative").fit(
        polars_data.lazy(), outcome="y", shock="shock", unit="unit", time="time"
    )
    np.testing.assert_allclose(actual_cumulative.coef_, expected_cumulative.coef_)


@pytest.mark.parametrize("covariance", ["hac", "driscoll_kraay"])
def test_polars_time_covariances_match_pandas(covariance: str) -> None:
    pandas_data = panel()
    options = dict(outcome="y", shock="shock", unit="unit", time="time")
    expected = LocalProjection(horizons=2, covariance=covariance).fit(pandas_data, **options)
    actual = LocalProjection(horizons=2, covariance=covariance).fit(
        as_polars(pandas_data), **options
    )
    np.testing.assert_allclose(actual.coef_, expected.coef_)
    np.testing.assert_allclose(actual.stderr_, expected.stderr_, rtol=1e-11, atol=1e-11)


def test_polars_integer_string_time_matches_pandas_canonical_clock() -> None:
    pandas_data = panel().query("not (unit == 0 and time == 3)").copy()
    pandas_data["time"] = pandas_data["time"].astype(str)
    options = dict(outcome="y", shock="shock", unit="unit", time="time")
    expected = LocalProjection(
        horizons=2, covariance="hac", hac_lags=1, sample="per_horizon"
    ).fit(pandas_data, **options)
    actual = LocalProjection(
        horizons=2, covariance="hac", hac_lags=1, sample="per_horizon"
    ).fit(as_polars(pandas_data), **options)

    np.testing.assert_allclose(actual.coef_, expected.coef_)
    np.testing.assert_allclose(actual.covariance_, expected.covariance_)
    assert actual.n_obs_by_horizon_.tolist() == expected.n_obs_by_horizon_.tolist()


def test_polars_interaction_cluster_matches_pandas() -> None:
    pandas_data = panel()
    options = dict(
        outcome="y",
        shock="shock",
        unit="unit",
        time="time",
        cluster=("region", "industry"),
    )
    with pytest.warns(UserWarning):
        expected = LocalProjection(horizons=2, covariance="cluster").fit(
            pandas_data, **options
        )
    with pytest.warns(UserWarning):
        actual = LocalProjection(horizons=2, covariance="cluster").fit(
            as_polars(pandas_data), **options
        )
    np.testing.assert_allclose(actual.stderr_, expected.stderr_, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.drop("shock"), "missing required columns"),
        (lambda frame: frame.with_columns(pl.lit(None).alias("shock")), "complete data"),
        (lambda frame: frame.with_columns(pl.lit("bad").alias("shock")), "must be numeric"),
        (lambda frame: frame.with_columns(pl.lit(float("inf")).alias("shock")), "must be finite"),
        (lambda frame: pl.concat([frame, frame.head(1)]), "uniquely identify"),
    ],
)
def test_polars_validation_errors_match_public_contract(mutate, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        fit(mutate(as_polars(panel())))


def test_polars_internal_row_name_does_not_conflict_with_user_column() -> None:
    data = as_polars(panel()).with_columns(
        pl.col("control").alias("__fastlp_original_row__")
    )
    fitted = LocalProjection(horizons=1, covariance="hc1").fit(
        data,
        outcome="y",
        shock="shock",
        controls=["__fastlp_original_row__"],
        unit="unit",
        time="time",
    )
    assert fitted.demeaning_diagnostics_["input_backend"] == "polars"
