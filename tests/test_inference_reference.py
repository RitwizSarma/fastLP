"""Independent statsmodels references for FE rank and inference conventions."""

from itertools import combinations, permutations

import numpy as np
import pandas as pd
import pytest
from linearmodels.iv.covariance import (
    KernelCovariance,
    kernel_weight_quadratic_spectral,
)
from scipy.linalg import qr
from scipy.stats import norm, t
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import (
    cov_cluster,
    cov_nw_groupsum,
    cov_nw_panel,
)

from fastlp import LocalProjection
from fastlp._covariance import _kernel_weight
from fastlp._degrees_of_freedom import effect_rank
from fastlp._demean import factorize_effects


def panel():
    rng = np.random.default_rng(881)
    n = 240
    data = pd.DataFrame(
        {
            "unit": np.repeat(np.arange(20), 12),
            "time": np.tile(np.arange(12), 20),
            "x": rng.normal(size=n),
            "z": rng.normal(size=n),
            "y": rng.normal(size=n),
            "third": rng.integers(0, 5, n),
        }
    )
    data["state"] = data.unit // 4
    data["duplicate"] = data.unit
    return data


def reference(data, effects, y=None):
    """Explicit dummies; remove redundant columns independently before OLS.

    statsmodels' cluster correction uses column count, so supply a full-rank
    design rather than redundant dummy columns. Slopes remain the first two.
    """
    dummy = np.column_stack([pd.get_dummies(data[c], dtype=float) for c in effects])
    _, _, pivots = qr(dummy, mode="economic", pivoting=True)
    dummy = dummy[:, pivots[: np.linalg.matrix_rank(dummy)]]
    design = np.column_stack([data[["x", "z"]], dummy])
    return OLS(data.y if y is None else y, design).fit()


@pytest.mark.parametrize(
    "effects",
    [
        ["unit"],
        ["unit", "time"],
        ["unit", "state"],
        ["unit", "time", "third"],
        ["unit", "duplicate", "state", "time"],
    ],
)
@pytest.mark.parametrize("covariance", ["homoskedastic", "hc0", "hc1"])
@pytest.mark.parametrize("sample", ["common", "per_horizon"])
def test_full_dummy_ols_rank_residuals_and_covariance(effects, covariance, sample):
    data = panel().drop(index=[3, 51, 178])
    fitted = LocalProjection(horizons=2, covariance=covariance, sample=sample).fit(
        data,
        outcome="y",
        shock="x",
        controls=["z"],
        unit="unit",
        time="time",
        fixed_effects=effects,
    )
    lookup = data.set_index(["unit", "time"]).y
    for h, rows in enumerate(fitted.sample_index_by_horizon_):
        anchors = data.loc[rows]
        y = lookup.reindex(
            pd.MultiIndex.from_arrays([anchors.unit, anchors.time + h])
        ).to_numpy()
        expected = reference(anchors, effects, y)
        covariance_expected = (
            expected.cov_params()
            if covariance == "homoskedastic"
            else expected.get_robustcov_results(
                cov_type=covariance.upper()
            ).cov_params()
        )
        np.testing.assert_allclose(fitted.coef_[h], expected.params[:2], atol=1e-10)
        np.testing.assert_allclose(
            fitted.covariance_[h], np.asarray(covariance_expected)[:2, :2], atol=1e-10
        )
        residual = (
            fitted.residuals_[:, h] if sample == "common" else fitted.residuals_[h]
        )
        np.testing.assert_allclose(residual, expected.resid, atol=1e-9)
        assert fitted.df_resid_by_horizon_[h] == expected.df_resid


@pytest.mark.parametrize(
    ("cluster", "adjustment"),
    [
        ("unit", "min"),
        ("state", "min"),
        ("third", "min"),
        (("unit", "time"), "min"),
        # Only multiway clustering distinguishes the group-adjustment rules.
        (["unit", "time"], "min"),
        (["unit", "time"], "component"),
    ],
)
@pytest.mark.parametrize("policy", ["nonnested", "full", "none"])
def test_cluster_covariance_and_intervals_against_statsmodels(
    cluster, policy, adjustment
):
    data = panel()
    effects = ["unit", "time"]
    fitted = LocalProjection(
        horizons=0,
        covariance="cluster",
        cluster_df=policy,
        cluster_group_adjustment=adjustment,
        few_cluster_threshold=None,
    ).fit(
        data,
        outcome="y",
        shock="x",
        controls=["z"],
        unit="unit",
        time="time",
        fixed_effects=effects,
        cluster=cluster,
    )
    expected = reference(data, effects)
    terms = cluster if isinstance(cluster, list) else [cluster]
    groups = [
        pd.factorize(
            pd.MultiIndex.from_frame(
                data[list(term) if isinstance(term, tuple) else [term]]
            )
        )[0]
        for term in terms
    ]
    n_groups = [len(np.unique(group)) for group in groups]
    # Known nesting for this fixture, independently specified.
    if policy == "full":
        correction_k = 2 + 20 + 12 - 1
    elif policy == "none":
        correction_k = 2
    elif cluster in ("unit", "state"):
        correction_k = 2 + 12
    elif isinstance(cluster, list):
        correction_k = 2 + 1
    else:
        correction_k = 2 + 20 + 12 - 1
    result = np.zeros((2, 2))
    for size in range(1, len(groups) + 1):
        for subset in combinations(range(len(groups)), size):
            joint = pd.factorize(
                pd.MultiIndex.from_arrays([groups[index] for index in subset])
            )[0]
            g = min(n_groups) if adjustment == "min" else len(np.unique(joint))
            raw = cov_cluster(expected, joint, use_correction=False)[:2, :2]
            result += (
                (-1) ** (size + 1)
                * raw
                * g
                / (g - 1)
                * (len(data) - 1)
                / (len(data) - correction_k)
            )
    np.testing.assert_allclose(fitted.covariance_[0], result, rtol=1e-10, atol=1e-12)
    assert fitted.correction_rank_by_horizon_[0] == correction_k
    assert fitted.df_resid_by_horizon_[0] == expected.df_resid
    assert fitted.inference_df_by_horizon_[0] == min(n_groups) - 1
    half_width = t.ppf(0.975, min(n_groups) - 1) * np.sqrt(np.diag(result))
    np.testing.assert_allclose(fitted.conf_int_[0, :, 1] - fitted.coef_[0], half_width)
    if policy == "full" and len(groups) == 1:
        np.testing.assert_allclose(result, cov_cluster(expected, groups[0])[:2, :2])


def test_cr0_and_normal_reference_are_independent_options():
    data = panel()
    fitted = LocalProjection(
        horizons=0,
        covariance="cluster",
        cluster_correction="cr0",
        cluster_inference="normal",
        few_cluster_threshold=None,
    ).fit(
        data,
        outcome="y",
        shock="x",
        controls=["z"],
        unit="unit",
        time="time",
        fixed_effects=["unit"],
        cluster="unit",
    )
    expected = reference(data, ["unit"])
    np.testing.assert_allclose(
        fitted.covariance_[0],
        cov_cluster(expected, data.unit, use_correction=False)[:2, :2],
    )
    np.testing.assert_allclose(fitted.critical_values_, norm.ppf(0.975))


def test_cluster_nesting_is_recomputed_on_each_retained_sample():
    data = panel()
    data.loc[data.time == 11, "state"] = (data.loc[data.time == 11, "state"] + 1) % 5
    fitted = LocalProjection(
        horizons=1,
        covariance="cluster",
        sample="per_horizon",
        few_cluster_threshold=None,
    ).fit(
        data,
        outcome="y",
        shock="x",
        controls=["z"],
        unit="unit",
        time="time",
        fixed_effects=["unit"],
        cluster="state",
    )
    assert fitted.correction_rank_by_horizon_.tolist() == [22, 3]
    groups = fitted.degrees_of_freedom_diagnostics_["by_cache_group"]
    assert groups[0]["nested_effects"] == ()
    assert groups[1]["nested_effects"] == ("unit",)
    lookup = data.set_index(["unit", "time"]).y
    for h, rows in enumerate(fitted.sample_index_by_horizon_):
        anchor = data.loc[rows]
        y = lookup.reindex(
            pd.MultiIndex.from_arrays([anchor.unit, anchor.time + h])
        ).to_numpy()
        expected = reference(anchor, ["unit"], y)
        raw = cov_cluster(expected, anchor.state, use_correction=False)[:2, :2]
        correction = 5 / 4 * (len(anchor) - 1) / (len(anchor) - [22, 3][h])
        np.testing.assert_allclose(fitted.covariance_[h], correction * raw, atol=1e-11)


@pytest.mark.parametrize("covariance", ["hac", "driscoll_kraay"])
def test_hac_full_model_debias_matches_statsmodels(covariance):
    data = panel()
    fitted = LocalProjection(horizons=0, covariance=covariance, hac_lags=2).fit(
        data,
        outcome="y",
        shock="x",
        controls=["z"],
        unit="unit",
        time="time",
        fixed_effects=["unit", "time"],
    )
    expected = reference(data, ["unit", "time"])
    if covariance == "hac":
        result = cov_nw_panel(
            expected,
            2,
            [(i * 12, (i + 1) * 12) for i in range(20)],
            use_correction="hac",
        )
    else:
        result = cov_nw_groupsum(
            expected, 2, data.time.to_numpy(), use_correction="hac"
        )
    np.testing.assert_allclose(fitted.covariance_[0], result[:2, :2], atol=1e-11)
    assert np.isfinite(fitted.stderr_).all()
    assert fitted.covariance_config_["kind"] == covariance
    assert len(fitted.covariance_diagnostics_) == len(fitted.horizons_)


def test_quadratic_spectral_weights_match_linearmodels():
    bandwidth = 3
    expected = kernel_weight_quadratic_spectral(bandwidth, 24)
    actual = np.asarray(
        [_kernel_weight("quadratic_spectral", lag, bandwidth) for lag in range(25)]
    )
    np.testing.assert_allclose(actual, expected, atol=1e-15)
    assert actual[bandwidth + 1] != 0.0
    assert _kernel_weight("quadratic_spectral", 0, 0) == 1.0
    assert _kernel_weight("quadratic_spectral", 1, 0) == 0.0


@pytest.mark.parametrize("debiased", [False, True])
def test_quadratic_spectral_hac_matches_linearmodels(debiased):
    rng = np.random.default_rng(903)
    n_obs = 80
    data = pd.DataFrame(
        {
            "unit": np.zeros(n_obs, dtype=int),
            "time": np.arange(n_obs),
            "x": rng.normal(size=n_obs),
            "z": rng.normal(size=n_obs),
            "y": rng.normal(size=n_obs),
        }
    )
    fitted = LocalProjection(
        horizons=0,
        covariance="hac",
        hac_kernel="quadratic_spectral",
        hac_lags=3,
        hac_debias=debiased,
    ).fit(
        data,
        outcome="y",
        shock="x",
        controls=["z"],
        unit="unit",
        time="time",
    )
    design = np.column_stack([np.ones(n_obs), data[["x", "z"]]])
    parameters = np.linalg.lstsq(design, data.y.to_numpy(), rcond=None)[0]
    expected = KernelCovariance(
        design,
        data.y.to_numpy()[:, None],
        design,
        parameters[:, None],
        kernel="quadratic-spectral",
        bandwidth=3,
        debiased=debiased,
    ).cov
    np.testing.assert_allclose(fitted.covariance_[0], expected, atol=1e-12)
    assert fitted.covariance_diagnostics_[0]["lag_limit"] is None


@pytest.mark.parametrize("covariance", ["hac", "driscoll_kraay"])
@pytest.mark.parametrize("debiased", [False, True])
def test_panel_quadratic_spectral_matches_all_pair_reference(covariance, debiased):
    data = panel().drop(index=[3, 51, 178])
    data["time"] *= 2
    fitted = LocalProjection(
        horizons=0,
        covariance=covariance,
        hac_kernel="quadratic_spectral",
        hac_lags=3,
        hac_debias=debiased,
    ).fit(
        data,
        outcome="y",
        shock="x",
        controls=["z"],
        unit="unit",
        time="time",
    )
    design = np.column_stack([np.ones(len(data)), data[["x", "z"]]])
    residuals = data.y.to_numpy() - design @ fitted.coef_[0]
    scores = design * residuals[:, None]
    if covariance == "driscoll_kraay":
        scores = pd.DataFrame(scores).groupby(data.time.to_numpy()).sum().to_numpy()
        times = np.sort(data.time.unique())
        units = np.zeros(len(times), dtype=int)
    else:
        times = data.time.to_numpy()
        units = data.unit.to_numpy()
    meat = np.zeros((design.shape[1], design.shape[1]))
    for left in range(len(scores)):
        for right in range(len(scores)):
            if units[left] != units[right]:
                continue
            lag = abs(int(times[left]) - int(times[right]))
            weight = _kernel_weight("quadratic_spectral", lag, 3)
            meat += weight * np.outer(scores[left], scores[right])
    if debiased:
        meat *= len(data) / (len(data) - design.shape[1])
    bread = np.linalg.inv(design.T @ design)
    expected = bread @ meat @ bread
    np.testing.assert_allclose(fitted.covariance_[0], expected, atol=1e-12)


def test_zero_bandwidth_quadratic_spectral_hac_equals_hc0():
    data = panel()
    options = {
        "outcome": "y",
        "shock": "x",
        "controls": ["z"],
        "unit": "unit",
        "time": "time",
    }
    actual = LocalProjection(
        horizons=0,
        covariance="hac",
        hac_kernel="quadratic_spectral",
        hac_lags=0,
        hac_debias=False,
    ).fit(data, **options)
    expected = LocalProjection(horizons=0, covariance="hc0").fit(data, **options)
    np.testing.assert_allclose(actual.covariance_, expected.covariance_, atol=1e-12)


def test_disconnected_nested_duplicate_and_multiway_rank():
    data = panel()
    data = data.loc[
        ((data.unit < 10) & (data.time < 6)) | ((data.unit >= 10) & (data.time >= 6))
    ]
    for effects in [
        ["unit", "time"],
        ["unit", "time", "state"],
        ["unit", "time", "third"],
    ]:
        for order in permutations(effects):
            codes, counts = factorize_effects(data, order)
            dummy = np.column_stack(
                [pd.get_dummies(data[column], dtype=float) for column in order]
            )
            actual = effect_rank(codes, counts)
            assert actual.exact
            assert actual.rank == np.linalg.matrix_rank(dummy)
    codes, counts = factorize_effects(data, ("unit", "time"))
    assert effect_rank(codes, counts).rank == 20 + 12 - 2


def test_three_way_rank_budget_is_explicit(monkeypatch):
    import fastlp._degrees_of_freedom as dof

    data = panel()
    codes, counts = factorize_effects(data, ("unit", "time", "third"))
    expected = effect_rank(codes, counts)
    monkeypatch.setattr(dof, "_MAX_DUMMY_BYTES", 1)
    with pytest.raises(ValueError, match="rank budget|numerical-rank budget"):
        effect_rank(codes, counts)
    bound = effect_rank(codes, counts, mode="conservative")
    assert not bound.exact
    assert bound.rank >= expected.rank
    fitted = LocalProjection(horizons=0, covariance="hc1", fe_dof="conservative").fit(
        data,
        outcome="y",
        shock="x",
        unit="unit",
        time="time",
        fixed_effects=["unit", "time", "third"],
    )
    assert fitted.degrees_of_freedom_diagnostics_["by_cache_group"][0]["exact"] is False


def test_three_way_rank_budget_fails_before_absorption(monkeypatch):
    import fastlp._degrees_of_freedom as dof
    from fastlp import estimator

    data = panel()
    monkeypatch.setattr(dof, "_MAX_DUMMY_BYTES", 1)

    def absorption_must_not_run(*args, **kwargs):
        raise AssertionError("fixed-effect absorption ran before the rank-budget check")

    monkeypatch.setattr(estimator.Residualizer, "transform", absorption_must_not_run)
    with pytest.raises(ValueError, match="numerical-rank budget"):
        LocalProjection(horizons=0, covariance="hc1").fit(
            data,
            outcome="y",
            shock="x",
            unit="unit",
            time="time",
            fixed_effects=["unit", "time", "third"],
        )


def test_saturated_fixed_effect_model_rejects_inference():
    data = pd.DataFrame(
        {
            "unit": [0, 0, 1, 1],
            "time": [0, 1, 0, 1],
            "x": [0.0, 1.0, 2.0, 4.0],
            "y": [2.0, 4.0, 1.0, 6.0],
        }
    )
    with pytest.raises(
        ValueError, match="residual degrees of freedom must be positive"
    ):
        LocalProjection(horizons=0, covariance="homoskedastic").fit(
            data,
            outcome="y",
            shock="x",
            unit="unit",
            time="time",
            fixed_effects=["unit", "time"],
        )


@pytest.mark.parametrize(
    "option,value",
    [
        ("fe_dof", "guess"),
        ("cluster_df", "guess"),
        ("cluster_group_adjustment", "guess"),
        ("cluster_inference", "guess"),
    ],
)
def test_new_inference_options_validate(option, value):
    with pytest.raises(ValueError, match=option):
        LocalProjection(horizons=0, **{option: value})
