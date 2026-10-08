"""Uncached explicit-dummy statsmodels reference for small validation panels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import qr
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_cluster


@dataclass(frozen=True)
class ReferenceResult:
    coef: np.ndarray
    stderr: np.ndarray


def _dummy_basis(anchor: pd.DataFrame, columns) -> np.ndarray:
    if not columns:
        return np.ones((len(anchor), 1))
    width = sum(anchor[column].nunique() for column in columns)
    if len(anchor) * width * 8 > 64 * 1024**2 or min(len(anchor), width) > 2000:
        raise MemoryError(
            "explicit-dummy statsmodels reference exceeds its rank/memory budget"
        )
    dummy = np.column_stack(
        [pd.get_dummies(anchor[column], dtype=float) for column in columns]
    )
    _, _, pivots = qr(dummy, mode="economic", pivoting=True)
    return dummy[:, pivots[: np.linalg.matrix_rank(dummy)]]


def fit_independent_horizons(
    data: pd.DataFrame,
    *,
    horizons: int,
    covariance: str,
    fixed_effects: tuple[str, ...],
    cluster: str | None,
    controls: tuple[str, ...],
) -> ReferenceResult:
    """Match common samples and fastLP's default nonnested one-way CR1 policy.

    Explicit dummies intentionally keep this oracle independent of fastLP's
    absorption and rank helpers. Callers should treat MemoryError as an
    unavailable reference, not an estimation failure or a successful check.
    """
    frame = data.sort_values(["unit", "time"], kind="stable")
    panels = [group for _, group in frame.groupby("unit", sort=True, observed=True)]
    labels = panels[0]["time"].tolist()
    if any(panel["time"].tolist() != labels for panel in panels[1:]):
        raise ValueError("reference implementation requires a balanced panel")
    n_anchor = len(labels) - horizons
    if n_anchor <= 0:
        raise ValueError("horizons must be smaller than the number of periods")
    coefficients, errors = [], []
    feature_columns = ("shock", *controls)
    for horizon in range(horizons + 1):
        anchor = pd.concat(
            [panel.iloc[:n_anchor] for panel in panels], ignore_index=False
        )
        x = anchor.loc[:, list(feature_columns)].to_numpy(dtype=np.float64)

        if fixed_effects:
            design = np.column_stack((x, _dummy_basis(anchor, fixed_effects)))
            n_slopes = x.shape[1]
        else:
            design = np.column_stack((np.ones(len(anchor)), x))
            n_slopes = design.shape[1]
        y = np.concatenate(
            [
                panel["outcome"].to_numpy()[horizon : horizon + n_anchor]
                for panel in panels
            ]
        )
        fitted = OLS(y, design).fit()
        beta = fitted.params[:n_slopes]
        if covariance == "hc1":
            cov = fitted.cov_HC1[:n_slopes, :n_slopes]
        else:
            if cluster is None:
                raise ValueError("cluster is required for clustered covariance")
            nonnested = [
                column
                for column in fixed_effects
                if anchor.groupby(column, observed=True)[cluster].nunique().max() > 1
            ]
            correction_k = (
                n_slopes + _dummy_basis(anchor, nonnested).shape[1]
                if fixed_effects
                else n_slopes
            )
            n_clusters = anchor[cluster].nunique()
            cov = cov_cluster(fitted, anchor[cluster], use_correction=False)[
                :n_slopes, :n_slopes
            ]
            cov *= (
                n_clusters
                / (n_clusters - 1)
                * (len(anchor) - 1)
                / (len(anchor) - correction_k)
            )
        coefficients.append(beta)
        errors.append(np.sqrt(np.maximum(np.diag(cov), 0.0)))
    return ReferenceResult(np.asarray(coefficients), np.asarray(errors))
