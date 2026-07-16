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


def test_cluster_covariance_requires_cluster_column() -> None:
    with pytest.raises(ValueError, match="cluster must be supplied"):
        LocalProjection(horizons=1).fit(
            balanced_panel(), outcome="y", shock="shock", unit="unit", time="time"
        )
