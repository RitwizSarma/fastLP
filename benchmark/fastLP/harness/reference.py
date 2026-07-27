"""Deliberately uncached balanced-panel reference used only for validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fastlp._covariance import cluster_cr1, hc1
from fastlp._demean import demean, factorize_effects


@dataclass(frozen=True)
class ReferenceResult:
    coef: np.ndarray
    stderr: np.ndarray


def fit_independent_horizons(
    data: pd.DataFrame, *, horizons: int, covariance: str, fixed_effects: tuple[str, ...], cluster: str | None,
    controls: tuple[str, ...],
) -> ReferenceResult:
    """Estimate every balanced horizon from scratch; do not reuse its design."""
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
        anchor = pd.concat([panel.iloc[:n_anchor] for panel in panels], ignore_index=False)
        x = anchor.loc[:, list(feature_columns)].to_numpy(dtype=np.float64)
        if not fixed_effects:
            x = np.column_stack((np.ones(len(anchor)), x))
        y = np.concatenate([panel["outcome"].to_numpy()[horizon : horizon + n_anchor] for panel in panels])[:, None]
        codes, counts = factorize_effects(anchor, fixed_effects)
        xt, _ = demean(x, codes, counts, tol=1e-10, max_iter=10_000)
        yt, _ = demean(y, codes, counts, tol=1e-10, max_iter=10_000)
        gram = xt.T @ xt
        chol = np.linalg.cholesky(gram)
        bread = np.linalg.solve(chol.T, np.linalg.solve(chol, np.eye(xt.shape[1])))
        beta = np.linalg.solve(chol.T, np.linalg.solve(chol, xt.T @ yt))[:, 0]
        residual = yt - xt @ beta[:, None]
        if covariance == "hc1":
            cov = hc1(xt, residual, bread)[0]
        else:
            if cluster is None:
                raise ValueError("cluster is required for clustered covariance")
            cluster_codes, _ = pd.factorize(anchor[cluster], sort=True)
            cov = cluster_cr1(xt, residual, cluster_codes.astype(np.int64), bread)[0]
        coefficients.append(beta)
        errors.append(np.sqrt(np.maximum(np.diag(cov), 0.0)))
    return ReferenceResult(np.asarray(coefficients), np.asarray(errors))
