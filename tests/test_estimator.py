import numpy as np
import pandas as pd
import pytest
import warnings

from fastlp import FewClustersWarning, LocalProjection
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


@pytest.mark.parametrize("covariance", ["homoskedastic", "hc0", "hc1", "hc2", "hc3", "hac", "driscoll_kraay"])
def test_extended_covariances_produce_finite_cached_results(covariance: str) -> None:
    fitted = LocalProjection(horizons=2, covariance=covariance).fit(
        covariance_panel(),
        outcome="y",
        shock="shock",
        controls=["control"],
        unit="unit",
        time="time",
    )
    assert np.isfinite(fitted.stderr_).all()
    assert fitted.covariance_config_["kind"] == covariance
    assert len(fitted.covariance_diagnostics_) == 3


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


def test_unbalanced_lagged_design_matches_separate_ols() -> None:
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
            for row_index, row in panel.iterrows():
                if row_index < 3 or row.time + horizon not in indexed_y.index:
                    continue
                x_rows.append(
                    [
                        1.0,
                        row.shock,
                        panel.iloc[row_index - 1].y,
                        panel.iloc[row_index - 3].y,
                        panel.iloc[row_index - 1].shock,
                        panel.iloc[row_index - 2].shock,
                    ]
                )
                y_rows.append(indexed_y.loc[row.time + horizon])
        expected.append(np.linalg.lstsq(np.asarray(x_rows), y_rows, rcond=None)[0])

    np.testing.assert_allclose(fitted.coef_, expected, atol=1e-10)


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
    fitted = LocalProjection(
        horizons=3,
        covariance="cluster",
        retain_residuals=False,
        memory_budget="1KB",
    ).fit(
        covariance_panel(),
        outcome="y",
        shock="shock",
        unit="unit",
        time="time",
        cluster="unit",
    )
    assert fitted.residuals_ is None
    assert fitted.demeaning_diagnostics_["batch_size"] == 1


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

    assert dropped.n_obs_ == 4
    assert dropped.sample_index_.tolist() == [0, 1, 2, 3]
    assert dropped.singleton_diagnostics_["dropped_by_horizon"].tolist() == [4]
    assert dropped.singleton_diagnostics_["rounds_by_horizon"].tolist() == [4]
    assert kept.n_obs_ == 8
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


def test_nearly_unit_spaced_time_uses_exact_indexed_leads() -> None:
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
    fitted = LocalProjection(
        horizons=1, covariance="hc1", sample="per_horizon"
    ).fit(
        data,
        outcome="y",
        shock="shock",
        unit="unit",
        time="time",
    )

    assert fitted.demeaning_diagnostics_["lead_path"] == "indexed"
    assert fitted.n_obs_by_horizon_.tolist() == [16, 4]
    expected_horizon_one = data.index[data["time"] == 0.0].tolist()
    assert fitted.sample_index_by_horizon_[1].tolist() == expected_horizon_one
