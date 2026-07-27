import numpy as np
import pandas as pd
import pytest

from fastlp import LocalProjection


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
        "horizon", "coefficient", "estimate", "std_error", "ci_low", "ci_high"
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
    assert np.all(fitted.demeaning_diagnostics_["x_iterations"] > 0)
    assert fitted.covariance_.shape == (2, 2, 2)


def test_rejects_unbalanced_panel() -> None:
    data = balanced_panel().query("not (unit == 0 and time == 7)")
    with pytest.raises(ValueError, match="same ordered time labels"):
        LocalProjection(horizons=1, covariance="hc1").fit(
            data, outcome="y", shock="shock", unit="unit", time="time"
        )


@pytest.mark.parametrize("fixed_effects", [(), ("unit",), ("unit", "time")])
def test_unbalanced_matches_separate_horizon_ols(fixed_effects: tuple[str, ...]) -> None:
    data = balanced_panel()
    data["shock"] += 0.013 * data["unit"] * data["time"] ** 2
    data["control"] += 0.017 * data["unit"] ** 2 * data["time"]
    data = data.query(
        "not ((unit == 0 and time == 2) or (unit == 1 and time == 6) or (unit == 3 and time == 4))"
    )
    fitted = LocalProjection(horizons=2, covariance="hc1", allow_unbalanced=True).fit(
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
    modes = fitted.demeaning_diagnostics_["gram_cache_mode"]
    expected_mode = {
        0: "rank_update", 1: "group_sufficient_statistics", 2: "alternating_projections"
    }[len(fixed_effects)]
    assert set(modes) == {expected_mode}


def test_cluster_covariance_requires_cluster_column() -> None:
    with pytest.raises(ValueError, match="cluster must be supplied"):
        LocalProjection(horizons=1).fit(
            balanced_panel(), outcome="y", shock="shock", unit="unit", time="time"
        )
