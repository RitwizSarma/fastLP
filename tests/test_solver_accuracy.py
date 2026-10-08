"""Stress the actual cached solver against independent statsmodels OLS."""

import numpy as np
import pandas as pd
import pytest
from statsmodels.regression.linear_model import OLS

from fastlp import LocalProjection
from fastlp._covariance import hc
from fastlp._linalg import RegressionDesign


@pytest.mark.parametrize(
    ("k", "condition"),
    # A one-column design has condition one regardless of the requested ratio.
    [(1, 1)] + [
        (k, condition)
        for k in (2, 3, 4, 5, 6, 7, 16)
        for condition in (1, 100, 500, 2000, 1e5, 1e7)
    ],
)
def test_solver_coefficients_residuals_and_covariance(k, condition):
    rng = np.random.default_rng(314)
    n = 160
    u, _ = np.linalg.qr(rng.normal(size=(n, k)))
    v, _ = np.linalg.qr(rng.normal(size=(k, k)))
    x = (u * np.geomspace(1.0, 1.0 / condition, k)) @ v.T
    x *= np.geomspace(1e-3, 1e3, k)
    y = rng.normal(size=(n, 3))
    design = RegressionDesign(x, horizon=0)
    beta, residuals = design.solve(y)
    working_cov = hc(design.x, residuals, design.bread, "hc0")
    covariance = design.covariance_in_original_units(working_cov)
    # Scale columns for the reference too: avoid statsmodels dropping a column
    # solely because physical units span a large range.
    scales = np.linalg.norm(x, axis=0)
    for column in range(y.shape[1]):
        expected = OLS(y[:, column], x / scales).fit()
        expected_beta = expected.params / scales
        expected_cov = expected.cov_HC0 / scales[:, None] / scales[None, :]
        np.testing.assert_allclose(beta[:, column], expected_beta, rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(residuals[:, column], expected.resid, atol=1e-7)
        np.testing.assert_allclose(
            covariance[column], expected_cov, rtol=1e-7, atol=1e-8
        )
    actual_condition = np.linalg.cond(x / scales)
    expected_method = "scaled_cholesky" if actual_condition <= 1000 else "scaled_qr"
    assert design.diagnostics["solver"] == expected_method


@pytest.mark.parametrize(
    "covariance", ["homoskedastic", "hc0", "hc1", "hc2", "hc3", "cluster"]
)
def test_public_qr_path_matches_statsmodels_and_batches(covariance, monkeypatch):
    monkeypatch.setattr("fastlp.estimator._HORIZON_BATCH_SIZE", 1)
    rng = np.random.default_rng(61)
    n = 200
    x = rng.normal(size=n)
    z = x + 1e-7 * rng.normal(size=n)
    data = pd.DataFrame(
        {
            "unit": np.repeat(np.arange(10), 20),
            "time": np.tile(np.arange(20), 10),
            "x": x,
            "z": z,
            "y": x - z + 0.01 * rng.normal(size=n),
        }
    )
    options = {
        "outcome": "y",
        "shock": "x",
        "controls": ["z"],
        "unit": "unit",
        "time": "time",
        "cluster": "unit" if covariance == "cluster" else None,
    }
    fitted = LocalProjection(
        horizons=2, covariance=covariance, few_cluster_threshold=None
    ).fit(data, **options)
    assert fitted.linear_algebra_diagnostics_["solver"] == "scaled_qr"
    assert np.isfinite(fitted.stderr_).all()
    assert fitted.covariance_config_["kind"] == covariance
    assert len(fitted.covariance_diagnostics_) == 3
    lookup = data.set_index(["unit", "time"]).y
    for h, rows in enumerate(fitted.sample_index_by_horizon_):
        anchor = data.loc[rows]
        y = lookup.reindex(
            pd.MultiIndex.from_arrays([anchor.unit, anchor.time + h])
        ).to_numpy()
        X = np.column_stack([np.ones(len(anchor)), anchor[["x", "z"]]])
        expected = OLS(y, X).fit()
        if covariance == "homoskedastic":
            cov = expected.cov_params()
        elif covariance == "cluster":
            cov = expected.get_robustcov_results(
                cov_type="cluster", groups=anchor.unit
            ).cov_params()
        else:
            cov = expected.get_robustcov_results(
                cov_type=covariance.upper()
            ).cov_params()
        np.testing.assert_allclose(
            fitted.coef_[h], expected.params, rtol=2e-7, atol=1e-8
        )
        # statsmodels' HC2/HC3 leverage and cluster covariance also use unstable
        # intermediates on ill-conditioned designs; compare those cases through
        # a well-conditioned orthogonal reference basis below instead.
        if covariance not in {"cluster", "hc2", "hc3"}:
            np.testing.assert_allclose(fitted.covariance_[h], cov, rtol=2e-7, atol=1e-8)
        else:
            q, r = np.linalg.qr(X)
            orthogonal_fit = OLS(y, q).fit()
            orthogonal = (
                orthogonal_fit.get_robustcov_results(
                    cov_type="cluster", groups=anchor.unit
                )
                if covariance == "cluster"
                else orthogonal_fit.get_robustcov_results(cov_type=covariance.upper())
            )
            inverse_r = np.linalg.solve(r, np.eye(X.shape[1]))
            stable_cov = inverse_r @ orthogonal.cov_params() @ inverse_r.T
            np.testing.assert_allclose(
                fitted.covariance_[h], stable_cov, rtol=2e-7, atol=1e-8
            )


def test_exact_collinearity_is_rejected_by_qr_rank_check():
    rng = np.random.default_rng(712)
    x = rng.normal(size=(200, 2))
    with pytest.raises(ValueError, match="rank 2 of 3"):
        RegressionDesign(np.column_stack([x, x[:, 0] + x[:, 1]]), horizon=4)


@pytest.mark.parametrize("k", [2, 4, 7])
@pytest.mark.parametrize("structure", ["polynomial", "concentrated", "near_duplicate"])
@pytest.mark.parametrize("outcome_scale", [1e-12, 1e12])
def test_solver_on_structured_and_rescaled_designs(k, structure, outcome_scale):
    rng = np.random.default_rng(614)
    n = 400
    x = rng.normal(size=(n, k))
    if structure == "polynomial":
        x = np.vander(np.linspace(0.1, 1.0, n), N=k, increasing=True)
    elif structure == "concentrated":
        x[: n // 2] *= 1e-5
        x[:, -1] = x[:, 0] + 1e-4 * x[:, -1]
    else:
        x[:, -1] = x[:, 0] + 1e-7 * x[:, -1]
    y = rng.normal(size=(n, 1)) * outcome_scale
    scales = np.geomspace(1e-9, 1e9, k)
    design = RegressionDesign(x * scales, horizon=0)
    actual, residuals = design.solve(y)
    expected = OLS(y[:, 0], x).fit()
    np.testing.assert_allclose(
        actual[:, 0] * scales / outcome_scale,
        expected.params / outcome_scale,
        rtol=2e-7,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        residuals[:, 0] / outcome_scale, expected.resid / outcome_scale, atol=1e-7
    )
