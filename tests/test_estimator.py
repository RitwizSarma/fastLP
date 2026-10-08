from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest
import warnings

from fastlp import FewClustersWarning, LocalProjection, __version__
from fastlp._demean import Residualizer, factorize_effects


def balanced_panel() -> pd.DataFrame:
    rows = []
    for unit in range(4):
        for time in range(8):
            shock = (unit - 1.5) * 0.4 + time * 0.2
            control = unit * 0.3 - time * 0.1
            outcome = 1.2 * shock - 0.5 * control + unit * 0.7 + time * 0.4
            rows.append(
                {"unit": unit, "time": time, "y": outcome, "shock": shock, "control": control}
            )
    return pd.DataFrame(rows)


def test_package_exposes_installed_version() -> None:
    assert __version__ == version("fastlp-py")


def test_numpy_array_input_matches_pandas() -> None:
    data = balanced_panel()
    options = {
        "outcome": "y",
        "shock": "shock",
        "controls": ["control"],
        "unit": "unit",
        "time": "time",
    }
    expected = LocalProjection(horizons=2, covariance="hc1").fit(data, **options)
    actual = LocalProjection(horizons=2, covariance="hc1").fit(
        data.to_numpy(), column_names=data.columns.tolist(), **options
    )

    np.testing.assert_allclose(actual.coef_, expected.coef_)
    np.testing.assert_allclose(actual.covariance_, expected.covariance_)
    assert actual.feature_names_in_.tolist() == expected.feature_names_in_.tolist()


@pytest.mark.parametrize(
    ("data", "column_names", "message"),
    [
        (np.ones(3), ["y"], "two-dimensional"),
        (np.ones((3, 2)), None, "must be supplied"),
        (np.ones((3, 2)), ["y"], "one name for each array column"),
        (np.ones((3, 2)), ["y", "y"], "must not contain duplicates"),
    ],
)
def test_numpy_array_input_requires_valid_column_names(
    data: np.ndarray, column_names: list[str] | None, message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        LocalProjection(horizons=0).fit(
            data,
            column_names=column_names,
            outcome="y",
            shock="shock",
            unit="unit",
            time="time",
        )


def test_dataframe_input_rejects_column_names() -> None:
    with pytest.raises(ValueError, match="only valid when data is a NumPy array"):
        LocalProjection(horizons=0).fit(
            balanced_panel(),
            column_names=["y", "shock", "control", "unit", "time"],
            outcome="y",
            shock="shock",
            unit="unit",
            time="time",
        )


def test_cached_no_fe_matches_separate_ols() -> None:
    data = balanced_panel()
    fitted = LocalProjection(horizons=2, covariance="hc1").fit(
        data, outcome="y", shock="shock", controls=["control"], unit="unit", time="time"
    )
    expected = []
    for horizon in range(3):
        anchors = data[data.time <= data.time.max() - 2].sort_values(["unit", "time"])
        y = np.concatenate(
            [group.y.to_numpy()[horizon : horizon + 6] for _, group in data.groupby("unit")]
        )
        x = np.column_stack((np.ones(len(anchors)), anchors[["shock", "control"]]))
        expected.append(np.linalg.lstsq(x, y, rcond=None)[0])
    np.testing.assert_allclose(fitted.coef_, np.asarray(expected), atol=1e-10)
    assert fitted.n_obs_ == 24
    assert list(fitted.to_frame().columns) == [
        "horizon", "sample", "response", "n_obs", "coefficient", "estimate", "std_error", "ci_low", "ci_high"
    ]


def test_fixed_effects_remove_intercept_and_converge() -> None:
    data = balanced_panel()
    data["shock"] += 0.03 * data["unit"] * data["time"]
    data["control"] += 0.01 * data["unit"] * data["time"] ** 2
    fitted = LocalProjection(horizons=1, covariance="cluster").fit(
        data,
        outcome="y",
        shock="shock",
        controls=["control"],
        unit="unit",
        time="time",
        fixed_effects=["unit", "time"],
        cluster="unit",
    )
    assert list(fitted.feature_names_in_) == ["shock", "control"]
    assert fitted.demeaning_diagnostics_["backend"] in {"numpy", "rust"}
    assert fitted.demeaning_diagnostics_["method"] in {
        "cyclic_kaczmarz", "symmetric_kaczmarz", "exact_balanced_two_way"
    }
    assert np.all(fitted.demeaning_diagnostics_["x_iterations"] > 0)
    assert fitted.covariance_.shape == (2, 2, 2)


def test_scaled_cholesky_preserves_original_feature_units() -> None:
    data = covariance_panel()
    baseline = LocalProjection(horizons=2, covariance="hc1").fit(
        data,
        outcome="y",
        shock="shock",
        controls=["control"],
        unit="unit",
        time="time",
    )
    rescaled = data.copy()
    rescaled["shock"] *= 1e9
    rescaled["control"] *= 1e-9
    fitted = LocalProjection(horizons=2, covariance="hc1").fit(
        rescaled,
        outcome="y",
        shock="shock",
        controls=["control"],
        unit="unit",
        time="time",
    )

    coefficient_map = np.diag([1.0, 1e9, 1e-9])
    np.testing.assert_allclose(
        fitted.coef_ @ coefficient_map,
        baseline.coef_,
        rtol=1e-10,
        atol=1e-10,
    )
    for horizon in range(3):
        np.testing.assert_allclose(
            coefficient_map @ fitted.covariance_[horizon] @ coefficient_map,
            baseline.covariance_[horizon],
            rtol=1e-9,
            atol=1e-10,
        )
    diagnostics = fitted.linear_algebra_diagnostics_
    assert diagnostics["solver"] == "scaled_cholesky"
    group = diagnostics["by_cache_group"][0]
    assert group["rank"] == 3
    assert np.isfinite(group["scaled_design_condition_number"])
    assert group["column_scales"][1] > group["column_scales"][2] * 1e15


def test_scaled_cholesky_reports_absorbed_and_collinear_designs() -> None:
    absorbed = balanced_panel()
    with pytest.raises(ValueError, match="negligible within variation.*shock"):
        LocalProjection(horizons=0, covariance="hc1").fit(
            absorbed,
            outcome="y",
            shock="shock",
            unit="unit",
            time="time",
            fixed_effects=["unit", "time"],
        )

    collinear = covariance_panel().copy()
    collinear["duplicate"] = collinear["shock"]
    with pytest.raises(ValueError, match="numerically rank deficient.*rank 2 of 3"):
        LocalProjection(horizons=0, covariance="hc1").fit(
            collinear,
            outcome="y",
            shock="shock",
            controls=["duplicate"],
            unit="unit",
            time="time",
        )


def test_within_variation_check_is_invariant_to_absorbed_levels() -> None:
    rng = np.random.default_rng(1729)
    data = pd.DataFrame(
        {
            "unit": np.repeat(np.arange(20), 3),
            "time": np.tile(np.arange(3), 20),
            "shock": rng.normal(size=60),
            "y": rng.normal(size=60),
        }
    )
    options = {
        "outcome": "y",
        "shock": "shock",
        "unit": "unit",
        "time": "time",
        "fixed_effects": ["unit"],
    }
    baseline = LocalProjection(horizons=0, covariance="hc0").fit(data, **options)
    shifted = data.copy()
    shifted["shock"] += 1e8
    actual = LocalProjection(horizons=0, covariance="hc0").fit(shifted, **options)

    np.testing.assert_allclose(actual.coef_, baseline.coef_, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        actual.covariance_, baseline.covariance_, rtol=1e-8, atol=1e-10
    )
    diagnostics = actual.linear_algebra_diagnostics_["by_cache_group"][0]
    assert (
        diagnostics["normalized_within_norms"][0] > diagnostics["within_signal_floor"]
    )


def test_iterative_within_certificate_rejects_absorbed_dust_but_keeps_signal() -> None:
    rng = np.random.default_rng(24680)
    n_obs = 500
    data = pd.DataFrame(
        {
            "unit": np.arange(n_obs),
            "time": np.zeros(n_obs, dtype=int),
            "first": rng.integers(0, 35, n_obs),
            "second": rng.integers(0, 25, n_obs),
            "shock": rng.normal(size=n_obs),
            "y": rng.normal(size=n_obs),
        }
    )
    options = {
        "outcome": "y",
        "shock": "shock",
        "unit": "unit",
        "time": "time",
        "fixed_effects": ["first", "second"],
    }
    baseline = LocalProjection(horizons=0, covariance="hc0").fit(data, **options)
    shifted = data.copy()
    shifted["shock"] += 1e8
    actual = LocalProjection(horizons=0, covariance="hc0").fit(shifted, **options)
    np.testing.assert_allclose(actual.coef_, baseline.coef_, rtol=2e-7, atol=1e-9)

    absorbed = data.copy()
    absorbed["shock"] = absorbed["first"].astype(float) + 1e8
    with pytest.raises(ValueError, match="negligible within variation.*shock"):
        LocalProjection(horizons=0, covariance="hc0").fit(absorbed, **options)


def test_cholesky_failure_falls_back_to_qr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fastlp.estimator as estimator

    def fail_cholesky(unused: np.ndarray) -> np.ndarray:
        raise np.linalg.LinAlgError("forced failure")

    monkeypatch.setattr(estimator.np.linalg, "cholesky", fail_cholesky)
    data = covariance_panel()
    fitted = LocalProjection(horizons=0, covariance="hc1").fit(
        data, outcome="y", shock="shock", controls=["control"], unit="unit", time="time"
    )
    expected = np.linalg.lstsq(
        np.column_stack([np.ones(len(data)), data[["shock", "control"]]]), data.y, rcond=None
    )[0]
    np.testing.assert_allclose(fitted.coef_[0], expected, atol=1e-12)
    assert fitted.linear_algebra_diagnostics_["solver"] == "scaled_qr"
    assert fitted.linear_algebra_diagnostics_["by_cache_group"][0]["fallback_reason"] == "cholesky_failed"


def test_prepared_residualizer_reuses_topology_and_matches_group_projection() -> None:
    data = balanced_panel()
    codes, counts = factorize_effects(data, ("unit", "time"))
    plan = Residualizer(codes, counts)
    values = data[["y", "shock", "control"]].to_numpy()
    first, first_diag = plan.transform(values, tol=1e-10, max_iter=10_000)
    second, second_diag = plan.transform(values[:, :1], tol=1e-10, max_iter=10_000)

    np.testing.assert_allclose(first[:, :1], second, atol=1e-12)
    for dimension in range(codes.shape[1]):
        for group in range(counts[dimension]):
            np.testing.assert_allclose(
                first[codes[:, dimension] == group].mean(axis=0), 0.0, atol=1e-10
            )
    assert first_diag.backend == second_diag.backend


def test_fit_reports_sorted_fast_path() -> None:
    data = balanced_panel()
    sorted_fit = LocalProjection(horizons=1, covariance="hc1").fit(
        data, outcome="y", shock="shock", unit="unit", time="time"
    )
    shuffled_fit = LocalProjection(horizons=1, covariance="hc1").fit(
        data.sample(frac=1, random_state=42),
        outcome="y", shock="shock", unit="unit", time="time"
    )
    assert sorted_fit.demeaning_diagnostics_["input_was_sorted"] is True
    assert shuffled_fit.demeaning_diagnostics_["input_was_sorted"] is False
    np.testing.assert_allclose(sorted_fit.coef_, shuffled_fit.coef_, atol=1e-12)


def test_horizon_batches_preserve_results(monkeypatch: pytest.MonkeyPatch) -> None:
    import fastlp.estimator as estimator

    data = covariance_panel()
    unbatched = LocalProjection(horizons=3, covariance="cluster").fit(
        data, outcome="y", shock="shock", controls=["control"],
        unit="unit", time="time", fixed_effects=["unit"], cluster="unit"
    )
    monkeypatch.setattr(estimator, "_HORIZON_BATCH_SIZE", 2)
    batched = LocalProjection(horizons=3, covariance="cluster").fit(
        data, outcome="y", shock="shock", controls=["control"],
        unit="unit", time="time", fixed_effects=["unit"], cluster="unit"
    )
    np.testing.assert_allclose(batched.coef_, unbatched.coef_, atol=1e-12)
    np.testing.assert_allclose(batched.covariance_, unbatched.covariance_, atol=1e-12)


def test_common_sample_accepts_unbalanced_panel_and_reports_shared_rows() -> None:
    data = balanced_panel().query("not (unit == 0 and time == 7)")
    fitted = LocalProjection(horizons=1, covariance="hc1").fit(
        data, outcome="y", shock="shock", unit="unit", time="time"
    )
    assert fitted.sample_ == "common"
    np.testing.assert_array_equal(fitted.n_obs_by_horizon_, [27, 27])
    assert fitted.cache_group_by_horizon_.tolist() == [0, 0]
    assert fitted.sample_index_by_horizon_[0].equals(fitted.sample_index_by_horizon_[1])


def test_common_sample_matches_separate_ols_on_its_shared_mask() -> None:
    data = balanced_panel()
    data["shock"] += 0.013 * data["unit"] * data["time"] ** 2
    data["control"] += 0.017 * data["unit"] ** 2 * data["time"]
    data = data.query("not ((unit == 0 and time == 2) or (unit == 1 and time == 6))")
    fitted = LocalProjection(horizons=2, covariance="hc1").fit(
        data, outcome="y", shock="shock", controls=["control"], unit="unit", time="time"
    )

    indexed_y = data.set_index(["unit", "time"])["y"]
    mask = np.ones(len(data), dtype=bool)
    for horizon in range(3):
        keys = pd.MultiIndex.from_arrays((data.unit, data.time + horizon))
        mask &= keys.isin(indexed_y.index)
    expected = []
    for horizon in range(3):
        sample = data.loc[mask]
        keys = pd.MultiIndex.from_arrays((sample.unit, sample.time + horizon))
        y = indexed_y.reindex(keys).to_numpy()
        x = np.column_stack((np.ones(len(sample)), sample[["shock", "control"]]))
        expected.append(np.linalg.lstsq(x, y, rcond=None)[0])

    np.testing.assert_allclose(fitted.coef_, expected, atol=1e-10)
    tidy_samples = fitted.to_frame().loc[:, ["horizon", "sample", "n_obs"]].drop_duplicates()
    assert tidy_samples["sample"].tolist() == ["common", "common", "common"]
    assert tidy_samples["n_obs"].tolist() == [mask.sum()] * 3


@pytest.mark.parametrize("fixed_effects", [(), ("unit",), ("unit", "time")])
def test_unbalanced_matches_separate_horizon_ols(fixed_effects: tuple[str, ...]) -> None:
    data = balanced_panel()
    data["shock"] += 0.013 * data["unit"] * data["time"] ** 2
    data["control"] += 0.017 * data["unit"] ** 2 * data["time"]
    data = data.query(
        "not ((unit == 0 and time == 2) or (unit == 1 and time == 6) or (unit == 3 and time == 4))"
    )
    fitted = LocalProjection(horizons=2, covariance="hc1", sample="per_horizon").fit(
        data,
        outcome="y",
        shock="shock",
        controls=["control"],
        unit="unit",
        time="time",
        fixed_effects=fixed_effects,
    )

    indexed_y = data.set_index(["unit", "time"])["y"]
    expected = []
    expected_n = []
    for horizon in range(3):
        sample = data.copy()
        keys = pd.MultiIndex.from_arrays((sample.unit, sample.time + horizon))
        valid = keys.isin(indexed_y.index)
        sample = sample.loc[valid].copy()
        sample["lead_y"] = indexed_y.reindex(keys[valid]).to_numpy()
        x = sample[["shock", "control"]].to_numpy(dtype=float)
        y = sample["lead_y"].to_numpy(dtype=float)
        if not fixed_effects:
            x = np.column_stack((np.ones(len(x)), x))
        else:
            from fastlp._demean import demean, factorize_effects

            codes, counts = factorize_effects(sample, fixed_effects)
            x, _ = demean(x, codes, counts, tol=1e-10, max_iter=10_000)
            y = demean(y[:, None], codes, counts, tol=1e-10, max_iter=10_000)[0][:, 0]
        expected.append(np.linalg.lstsq(x, y, rcond=None)[0])
        expected_n.append(len(sample))

    np.testing.assert_allclose(fitted.coef_, expected, atol=1e-10)
    np.testing.assert_array_equal(fitted.n_obs_by_horizon_, expected_n)
    assert fitted.sample_ == "per_horizon"
    assert fitted.demeaning_diagnostics_["cache_mode"] == "mask_grouped"
    assert fitted.cache_group_by_horizon_.tolist() == [0, 1, 2]


def test_cluster_covariance_requires_cluster_column() -> None:
    with pytest.raises(ValueError, match="cluster must be supplied"):
        LocalProjection(horizons=1).fit(
            balanced_panel(), outcome="y", shock="shock", unit="unit", time="time"
        )


def covariance_panel() -> pd.DataFrame:
    """Non-collinear panel with categorical cluster dimensions."""
    rng = np.random.default_rng(9182)
    rows = []
    for unit in range(8):
        for time in range(14):
            shock = rng.normal()
            control = rng.normal()
            rows.append(
                {
                    "unit": unit,
                    "time": time,
                    "industry": unit % 3,
                    "region": unit % 2,
                    "y": 0.7 * shock - 0.2 * control + rng.normal(),
                    "shock": shock,
                    "control": control,
                }
            )
    return pd.DataFrame(rows)


def test_multiway_and_interaction_clustering_are_explicit() -> None:
    data = covariance_panel()
    multiway = LocalProjection(horizons=2, covariance="cluster").fit(
        data,
        outcome="y",
        shock="shock",
        controls=["control"],
        unit="unit",
        time="time",
        cluster=["unit", "time"],
    )
    interaction = LocalProjection(horizons=2, covariance="cluster").fit(
        data,
        outcome="y",
        shock="shock",
        controls=["control"],
        unit="unit",
        time="time",
        cluster=("industry", "time"),
    )
    assert multiway.covariance_config_["cluster_terms"] == ("unit", "time")
    assert interaction.covariance_config_["cluster_terms"] == ("industry#time",)
    assert len(multiway.covariance_diagnostics_[0]["components"]) == 3
    assert np.isfinite(interaction.stderr_).all()


def test_negative_cluster_variance_rejection_is_scale_invariant() -> None:
    rng = np.random.default_rng(0)
    n_obs = 40
    data = pd.DataFrame(
        {
            "unit": np.repeat(np.arange(10), 4),
            "time": np.tile(np.arange(4), 10),
            "first_cluster": rng.integers(0, 4, n_obs),
            "second_cluster": rng.integers(0, 4, n_obs),
            "shock": rng.normal(size=n_obs),
            "y": rng.normal(size=n_obs),
        }
    )
    for scale in (1.0, 1e9):
        scaled = data.copy()
        scaled["shock"] *= scale
        with pytest.raises(
            ValueError, match="cluster covariance has a negative variance"
        ):
            LocalProjection(
                horizons=0,
                covariance="cluster",
                few_cluster_threshold=None,
            ).fit(
                scaled,
                outcome="y",
                shock="shock",
                unit="unit",
                time="time",
                fixed_effects=["unit"],
                cluster=["first_cluster", "second_cluster"],
            )


def test_roundoff_negative_variance_cleanup_is_coherent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastlp import estimator

    def roundoff_negative(x, residuals, *args, **kwargs):
        covariance = np.zeros((residuals.shape[1], x.shape[1], x.shape[1]))
        covariance[:, 0, 0] = -np.finfo(float).eps
        scale = np.ones_like(covariance) * 1e6
        diagnostics = tuple({"terms": ("unit",)} for _ in range(1))
        return covariance, diagnostics, scale

    monkeypatch.setattr(estimator, "cluster_covariance", roundoff_negative)
    fitted = LocalProjection(
        horizons=0,
        covariance="cluster",
        few_cluster_threshold=None,
    ).fit(
        covariance_panel(),
        outcome="y",
        shock="shock",
        unit="unit",
        time="time",
        fixed_effects=["unit"],
        cluster="unit",
    )

    assert fitted.covariance_[0, 0, 0] == 0.0
    assert fitted.stderr_[0, 0] == 0.0
    assert fitted.covariance_diagnostics_[0]["negative_variance_cleanup"] == (0,)


def test_cluster_validation_limits_terms_and_rejects_hac_cluster_mix() -> None:
    data = covariance_panel()
    with pytest.raises(ValueError, match="at most four"):
        LocalProjection(horizons=1, covariance="cluster").fit(
            data,
            outcome="y",
            shock="shock",
            unit="unit",
            time="time",
            cluster=["unit", "time", "industry", "region", ("unit", "region")],
        )
    with pytest.raises(ValueError, match="only valid"):
        LocalProjection(horizons=1, covariance="hac").fit(
            data, outcome="y", shock="shock", unit="unit", time="time", cluster="unit"
        )


def test_rejects_unknown_sample_policy() -> None:
    with pytest.raises(ValueError, match="sample must be"):
        LocalProjection(horizons=1, covariance="hc1", sample="largest")


def test_fit_generates_nonconsecutive_lags_without_dropping_future_outcomes() -> None:
    rng = np.random.default_rng(4321)
    rows = []
    for unit in range(5):
        for time in range(10):
            rows.append(
                {
                    "unit": unit,
                    "time": time,
                    "y": rng.normal(),
                    "shock": rng.normal(),
                    "control": rng.normal(),
                }
            )
    data = pd.DataFrame(rows)
    fitted = LocalProjection(horizons=2, covariance="hc1").fit(
        data,
        outcome="y",
        shock="shock",
        controls=["control"],
        outcome_lags=[1, 3],
        shock_lags={"shock": [1, 3]},
        control_lags=[2],
        unit="unit",
        time="time",
    )

    expected = []
    for horizon in range(3):
        x_rows = []
        y_rows = []
        for _, panel in data.groupby("unit", sort=True):
            panel = panel.sort_values("time").reset_index(drop=True)
            for time in range(3, 8):
                row = panel.iloc[time]
                x_rows.append(
                    [
                        1.0,
                        row.shock,
                        row.control,
                        panel.iloc[time - 1].y,
                        panel.iloc[time - 3].y,
                        panel.iloc[time - 1].shock,
                        panel.iloc[time - 3].shock,
                        panel.iloc[time - 2].control,
                    ]
                )
                y_rows.append(panel.iloc[time + horizon].y)
        expected.append(np.linalg.lstsq(np.asarray(x_rows), y_rows, rcond=None)[0])

    np.testing.assert_allclose(fitted.coef_, expected, atol=1e-10)
    assert list(fitted.feature_names_in_) == [
        "Intercept",
        "shock",
        "control",
        "y_lag1",
        "y_lag3",
        "shock_lag1",
        "shock_lag3",
        "control_lag2",
    ]
    assert fitted.n_obs_ == 25
    assert fitted.demeaning_diagnostics_["lag_path"] == "consecutive_shift"


def test_gapped_lagged_design_uses_exact_calendar_matches() -> None:
    rng = np.random.default_rng(9876)
    data = pd.DataFrame(
        [
            {"unit": unit, "time": time, "y": rng.normal(), "shock": rng.normal()}
            for unit in range(4)
            for time in range(9)
            if (unit, time) != (1, 4)
        ]
    )
    fitted = LocalProjection(horizons=2, covariance="hc1", sample="per_horizon").fit(
        data,
        outcome="y",
        shock="shock",
        outcome_lags=[1, 3],
        shock_lags=2,
        unit="unit",
        time="time",
    )

    expected = []
    for horizon in range(3):
        x_rows = []
        y_rows = []
        for _, panel in data.groupby("unit", sort=True):
            panel = panel.sort_values("time").reset_index(drop=True)
            indexed_y = panel.set_index("time").y
            indexed_shock = panel.set_index("time").shock
            for row in panel.itertuples():
                required_times = (row.time - 1, row.time - 2, row.time - 3)
                if (
                    any(date not in indexed_y.index for date in required_times)
                    or row.time + horizon not in indexed_y.index
                ):
                    continue
                x_rows.append(
                    [
                        1.0,
                        row.shock,
                        indexed_y.loc[row.time - 1],
                        indexed_y.loc[row.time - 3],
                        indexed_shock.loc[row.time - 1],
                        indexed_shock.loc[row.time - 2],
                    ]
                )
                y_rows.append(indexed_y.loc[row.time + horizon])
        expected.append(np.linalg.lstsq(np.asarray(x_rows), y_rows, rcond=None)[0])

    np.testing.assert_allclose(fitted.coef_, expected, atol=1e-10)
    assert fitted.demeaning_diagnostics_["lag_path"] == "indexed"


def test_cumulative_response_matches_manual_regressions() -> None:
    data = covariance_panel()
    fitted = LocalProjection(horizons=3, covariance="hc1", response="cumulative").fit(
        data,
        outcome="y",
        shock="shock",
        controls=["control"],
        unit="unit",
        time="time",
    )
    expected = []
    for horizon in range(4):
        rows = []
        outcomes = []
        for _, panel in data.groupby("unit", sort=True):
            panel = panel.sort_values("time").reset_index(drop=True)
            for anchor in range(len(panel) - 3):
                row = panel.iloc[anchor]
                rows.append([1.0, row.shock, row.control])
                outcomes.append(panel.y.iloc[anchor : anchor + horizon + 1].sum())
        expected.append(np.linalg.lstsq(np.asarray(rows), outcomes, rcond=None)[0])
    np.testing.assert_allclose(fitted.coef_, expected, atol=1e-10)
    assert fitted.response_ == "cumulative"
    assert set(fitted.to_frame()["response"]) == {"cumulative"}


def test_cumulative_per_horizon_requires_every_intermediate_outcome() -> None:
    data = balanced_panel().query("not (unit == 0 and time == 3)")
    fitted = LocalProjection(
        horizons=2, covariance="hc1", response="cumulative", sample="per_horizon"
    ).fit(data, outcome="y", shock="shock", unit="unit", time="time")
    assert fitted.n_obs_by_horizon_[2] < fitted.n_obs_by_horizon_[0]
    indexed = data.set_index(["unit", "time"])["y"]
    for source_index in fitted.sample_index_by_horizon_[2]:
        row = data.loc[source_index]
        assert all((row.unit, row.time + step) in indexed.index for step in range(3))


def test_balanced_fast_path_matches_general_demeaning() -> None:
    data = covariance_panel()
    fitted = LocalProjection(horizons=2, covariance="hc1").fit(
        data,
        outcome="y",
        shock="shock",
        controls=["control"],
        unit="unit",
        time="time",
        fixed_effects=["unit", "time"],
    )
    assert fitted.demeaning_diagnostics_["lead_path"] == "balanced_arithmetic"
    assert fitted.demeaning_diagnostics_["method"] == "exact_balanced_two_way"
    anchors = data.query("time <= 11").copy()
    codes, counts = factorize_effects(anchors, ("unit", "time"))
    residualizer = Residualizer(codes, counts)
    x = anchors[["shock", "control"]].to_numpy()
    x_tilde, _ = residualizer.transform(x, tol=1e-12, max_iter=10_000)
    expected = []
    for horizon in range(3):
        y = np.concatenate(
            [panel.y.to_numpy()[horizon : horizon + 12] for _, panel in data.groupby("unit")]
        )
        y_tilde, _ = residualizer.transform(y[:, None], tol=1e-12, max_iter=10_000)
        expected.append(np.linalg.lstsq(x_tilde, y_tilde[:, 0], rcond=None)[0])
    np.testing.assert_allclose(fitted.coef_, expected, atol=1e-10)


def test_memory_budget_and_residual_retention() -> None:
    options = dict(
        outcome="y", shock="shock", unit="unit", time="time", cluster="unit",
    )
    baseline = LocalProjection(
        horizons=3, covariance="cluster", retain_residuals=False,
        memory_budget="1MB", few_cluster_threshold=None,
    ).fit(covariance_panel(), **options)
    fitted = LocalProjection(
        horizons=3,
        covariance="cluster",
        retain_residuals=False,
        memory_budget=baseline.memory_diagnostics_["minimum_bytes"],
        few_cluster_threshold=None,
    ).fit(covariance_panel(), **options)
    assert fitted.residuals_ is None
    assert fitted.demeaning_diagnostics_["batch_size"] == 1
    assert fitted.memory_diagnostics_["planned_peak_bytes"] <= fitted.memory_budget
    np.testing.assert_allclose(fitted.coef_, baseline.coef_)
    np.testing.assert_allclose(fitted.covariance_, baseline.covariance_)


def test_new_option_validation() -> None:
    with pytest.raises(ValueError, match="response must be"):
        LocalProjection(horizons=1, response="ratio")
    with pytest.raises(ValueError, match="memory_budget"):
        LocalProjection(horizons=1, memory_budget="plenty")
    with pytest.raises(ValueError, match="retain_residuals"):
        LocalProjection(horizons=1, retain_residuals=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="demean_acceleration"):
        LocalProjection(horizons=1, demean_acceleration="turbo")
    with pytest.raises(ValueError, match="few_cluster_threshold"):
        LocalProjection(horizons=1, few_cluster_threshold=1)
    with pytest.raises(ValueError, match="singleton_policy"):
        LocalProjection(horizons=1, singleton_policy="sometimes")


def test_unbalanced_single_effect_uses_exact_group_transform() -> None:
    data = covariance_panel().query("not (unit == 0 and time in [3, 7])")
    fitted = LocalProjection(horizons=1, covariance="hc1", sample="per_horizon").fit(
        data,
        outcome="y",
        shock="shock",
        unit="unit",
        time="time",
        fixed_effects=["unit"],
    )
    assert set(fitted.demeaning_diagnostics_["method_by_cache_group"]) == {
        "exact_group_unit"
    }


def test_balanced_demeaning_keeps_numpy_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    import fastlp._demean as demeaning

    rng = np.random.default_rng(2468)
    values = rng.normal(size=(5 * 7, 3))
    native, native_backend = demeaning.balanced_demean(
        values, 5, 7, unit_effect=True, time_effect=True
    )
    monkeypatch.setattr(demeaning, "_rust_demean_balanced", None)
    fallback, fallback_backend = demeaning.balanced_demean(
        values, 5, 7, unit_effect=True, time_effect=True
    )

    assert native_backend == "rust"
    assert fallback_backend == "numpy"
    np.testing.assert_allclose(fallback, native, atol=1e-12)


def test_accelerated_three_way_absorption_matches_reference() -> None:
    rng = np.random.default_rng(13579)
    n_obs = 1_000
    effects = pd.DataFrame(
        {
            "first": rng.integers(0, 70, n_obs),
            "second": rng.integers(0, 45, n_obs),
            "third": rng.integers(0, 30, n_obs),
        }
    )
    codes, counts = factorize_effects(effects, ("first", "second", "third"))
    values = rng.normal(size=(n_obs, 4))
    plan = Residualizer(codes, counts)
    aitken, aitken_diag = plan.transform(
        values, tol=1e-10, max_iter=10_000, acceleration="aitken"
    )
    accelerated, accelerated_diag = plan.transform(
        values, tol=1e-10, max_iter=10_000, acceleration="irons_tuck"
    )
    reference, reference_diag = plan.transform(
        values, tol=1e-10, max_iter=10_000, acceleration="none"
    )
    plan._native = None
    numpy_accelerated, numpy_diag = plan.transform(
        values, tol=1e-10, max_iter=10_000, acceleration="irons_tuck"
    )

    np.testing.assert_allclose(aitken, reference, atol=5e-10)
    np.testing.assert_allclose(accelerated, reference, atol=5e-10)
    np.testing.assert_allclose(numpy_accelerated, reference, atol=5e-10)
    assert aitken_diag.method == "symmetric_kaczmarz_aitken"
    assert accelerated_diag.method == "symmetric_kaczmarz_irons_tuck"
    assert accelerated_diag.acceleration_accepted.sum() > 0
    assert reference_diag.acceleration_accepted.sum() == 0
    assert numpy_diag.backend == "numpy"
    assert numpy_diag.acceleration_accepted.sum() > 0
    for dimension, n_groups in enumerate(counts):
        for group in range(n_groups):
            np.testing.assert_allclose(
                accelerated[codes[:, dimension] == group].mean(axis=0), 0.0, atol=1e-10
            )


@pytest.mark.parametrize("use_native", [True, False])
@pytest.mark.parametrize("scale", [1e-12, 1e-8, 1e8, 1e12])
def test_iterative_absorption_is_scale_invariant(use_native: bool, scale: float) -> None:
    rng = np.random.default_rng(992)
    n_obs = 500
    effects = pd.DataFrame(
        {
            "first": rng.integers(0, 35, n_obs),
            "second": rng.integers(0, 25, n_obs),
            "third": rng.integers(0, 15, n_obs),
        }
    )
    codes, counts = factorize_effects(effects, ("first", "second", "third"))
    values = rng.normal(size=(n_obs, 3))
    plan = Residualizer(codes, counts)
    if use_native and plan._native is None:
        pytest.skip("native demeaning extension is unavailable")
    if not use_native:
        plan._native = None

    baseline, baseline_diagnostics = plan.transform(
        values, tol=1e-10, max_iter=10_000, acceleration="irons_tuck"
    )
    rescaled, rescaled_diagnostics = plan.transform(
        values * scale, tol=1e-10, max_iter=10_000, acceleration="irons_tuck"
    )

    np.testing.assert_allclose(rescaled / scale, baseline, rtol=1e-11, atol=1e-11)
    np.testing.assert_array_equal(
        rescaled_diagnostics.iterations, baseline_diagnostics.iterations
    )
    for dimension, n_groups in enumerate(counts):
        for group in range(n_groups):
            group_mean = (rescaled[codes[:, dimension] == group] / scale).mean(axis=0)
            np.testing.assert_allclose(group_mean, 0.0, atol=1e-10)


def test_hc2_hc3_with_fixed_effects_fail_fast_and_recommend_hc0_hc1() -> None:
    data = balanced_panel()
    for covariance in ("hc2", "hc3"):
        with pytest.raises(
            ValueError,
            match=r"use covariance='hc0' or covariance='hc1'",
        ):
            LocalProjection(horizons=0, covariance=covariance).fit(
                data,
                outcome="y",
                shock="shock",
                unit="unit",
                time="time",
                fixed_effects=["unit"],
            )


def test_few_cluster_warning_and_opt_out() -> None:
    data = covariance_panel()
    with pytest.warns(FewClustersWarning, match="only 8 clusters"):
        LocalProjection(
            horizons=1, covariance="cluster", few_cluster_threshold=50
        ).fit(
            data,
            outcome="y",
            shock="shock",
            unit="unit",
            time="time",
            cluster="unit",
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error", FewClustersWarning)
        LocalProjection(
            horizons=1, covariance="cluster", few_cluster_threshold=None
        ).fit(
            data,
            outcome="y",
            shock="shock",
            unit="unit",
            time="time",
            cluster="unit",
        )


def test_recursive_singleton_pruning_reaches_stable_sample() -> None:
    edges = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (0, 0),  # Positive residual df in the retained core, even with a slope.
        (1, 2),
        (2, 2),
        (2, 3),
        (3, 3),
    ]
    rng = np.random.default_rng(8642)
    data = pd.DataFrame(
        {
            "unit": np.arange(len(edges)),
            "time": np.zeros(len(edges), dtype=int),
            "first_fe": [edge[0] for edge in edges],
            "second_fe": [edge[1] for edge in edges],
            "y": rng.normal(size=len(edges)),
            "shock": rng.normal(size=len(edges)),
        }
    )
    dropped = LocalProjection(
        horizons=0, covariance="hc1", singleton_policy="drop"
    ).fit(
        data,
        outcome="y",
        shock="shock",
        unit="unit",
        time="time",
        fixed_effects=["first_fe", "second_fe"],
    )
    kept = LocalProjection(
        horizons=0, covariance="hc1", singleton_policy="keep"
    ).fit(
        data,
        outcome="y",
        shock="shock",
        unit="unit",
        time="time",
        fixed_effects=["first_fe", "second_fe"],
    )

    assert dropped.n_obs_ == 5
    assert dropped.sample_index_.tolist() == [0, 1, 2, 3, 4]
    assert dropped.singleton_diagnostics_["dropped_by_horizon"].tolist() == [4]
    assert dropped.singleton_diagnostics_["rounds_by_horizon"].tolist() == [4]
    assert kept.n_obs_ == 9
    assert kept.singleton_diagnostics_["dropped_by_horizon"].tolist() == [0]


def test_per_horizon_singletons_are_pruned_before_cache_grouping() -> None:
    rng = np.random.default_rng(97531)
    data = pd.DataFrame(
        [
            {
                "unit": unit,
                "time": time,
                "y": rng.normal(),
                "shock": rng.normal(),
            }
            for unit, periods in enumerate((3, 4, 5))
            for time in range(periods)
        ]
    )
    fitted = LocalProjection(
        horizons=2, covariance="hc1", sample="per_horizon"
    ).fit(
        data,
        outcome="y",
        shock="shock",
        unit="unit",
        time="time",
        fixed_effects=["unit"],
    )

    assert fitted.n_obs_by_horizon_.tolist() == [12, 9, 5]
    assert fitted.singleton_diagnostics_["dropped_by_horizon"].tolist() == [0, 0, 1]
    assert fitted.singleton_diagnostics_["rounds_by_horizon"].tolist() == [0, 0, 1]


def test_irf_plot_is_cached_styled_and_selectable() -> None:
    import matplotlib.pyplot as plt

    data = covariance_panel().copy()
    data["shock_2"] = np.roll(data["shock"].to_numpy(), 1)
    fitted = LocalProjection(horizons=3, covariance="hc1").fit(
        data,
        outcome="y",
        shock=["shock", "shock_2"],
        controls=["control"],
        unit="unit",
        time="time",
    )

    default = fitted.irfplot
    assert default is fitted.irfplot
    assert default.get_facecolor() == (1.0, 1.0, 1.0, 1.0)
    assert default.get_xlabel() == "Horizon"
    assert "shock" in default.get_title(loc="left")
    assert len(default.collections) == 1
    selected = fitted.plot_irf("shock_2", show_zero_line=False, title="Second shock")
    assert selected.get_title(loc="left") == "Second shock"
    assert len(selected.lines) == 1
    with pytest.raises(ValueError, match="not a fitted shock"):
        fitted.plot_irf("control")
    plt.close(default.figure)
    plt.close(selected.figure)


def test_fractional_time_is_rejected_before_lead_construction() -> None:
    rows = []
    for unit in range(4):
        for time in (0.0, 1.0, 2.000000001, 3.0):
            rows.append(
                {
                    "unit": unit,
                    "time": time,
                    "y": unit + 0.5 * time,
                    "shock": (unit + 1) * (time + 1),
                }
            )
    data = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="integer period identifiers"):
        LocalProjection(horizons=1, covariance="hc1", sample="per_horizon").fit(
            data,
            outcome="y",
            shock="shock",
            unit="unit",
            time="time",
        )


def test_integer_string_time_matches_integer_time_on_unbalanced_panel() -> None:
    data = covariance_panel().query("not (unit == 0 and time == 3)").copy()
    options = dict(
        outcome="y",
        shock="shock",
        controls=["control"],
        unit="unit",
        time="time",
    )
    expected = LocalProjection(
        horizons=2, covariance="hac", hac_lags=1, sample="per_horizon"
    ).fit(data, **options)
    string_time = data.copy()
    string_time["time"] = string_time["time"].astype(str)
    actual = LocalProjection(
        horizons=2, covariance="hac", hac_lags=1, sample="per_horizon"
    ).fit(string_time, **options)

    np.testing.assert_allclose(actual.coef_, expected.coef_)
    np.testing.assert_allclose(actual.covariance_, expected.covariance_)
    assert actual.n_obs_by_horizon_.tolist() == expected.n_obs_by_horizon_.tolist()
    assert actual.demeaning_diagnostics_["lead_path"] == "indexed"
    np.testing.assert_array_equal(
        actual.sample_index_by_horizon_[0], string_time.index.to_numpy()
    )


def test_large_native_integer_time_preserves_exact_leads() -> None:
    data = balanced_panel()
    data["time"] = data["time"].astype(np.int64) + (2**53 + 10)
    fitted = LocalProjection(
        horizons=1, covariance="hc1", sample="per_horizon"
    ).fit(
        data,
        outcome="y",
        shock="shock",
        unit="unit",
        time="time",
    )

    assert fitted.n_obs_by_horizon_.tolist() == [32, 28]
    assert fitted.demeaning_diagnostics_["lead_path"] == "balanced_arithmetic"


@pytest.mark.parametrize(
    "values",
    [
        [0.0, 1.0, float(2**53)],
        ["0", "1", str(2**63)],
        pd.date_range("2020-01-01", periods=3),
    ],
)
def test_unsupported_time_representations_fail_clearly(values) -> None:
    data = balanced_panel().query("unit == 0 and time < 3").copy()
    data["time"] = values
    with pytest.raises(ValueError, match="integer period identifiers"):
        LocalProjection(horizons=0, covariance="hc1").fit(
            data,
            outcome="y",
            shock="shock",
            unit="unit",
            time="time",
        )


def test_time_conversion_collision_is_rejected() -> None:
    data = balanced_panel().query("unit == 0 and time < 2").copy()
    data["time"] = ["01", "1"]
    with pytest.raises(ValueError, match="uniquely identify"):
        LocalProjection(horizons=0, covariance="hc1").fit(
            data,
            outcome="y",
            shock="shock",
            unit="unit",
            time="time",
        )
