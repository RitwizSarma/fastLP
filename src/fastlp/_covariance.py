"""Covariance estimators for residualized linear regressions."""

from __future__ import annotations

import numpy as np


def hc1(x: np.ndarray, residuals: np.ndarray, bread: np.ndarray) -> np.ndarray:
    """Return one HC1 covariance matrix for each residual column."""
    n_obs, rank = x.shape
    output = np.empty((residuals.shape[1], rank, rank), dtype=np.float64)
    correction = n_obs / (n_obs - rank)
    for horizon, residual in enumerate(residuals.T):
        meat = (x * residual[:, None]).T @ (x * residual[:, None])
        output[horizon] = correction * bread @ meat @ bread
    return output


def cluster_cr1(
    x: np.ndarray, residuals: np.ndarray, codes: np.ndarray, bread: np.ndarray
) -> np.ndarray:
    """Return one one-way CR1 covariance matrix for each residual column."""
    n_obs, rank = x.shape
    n_clusters = int(codes.max()) + 1
    if n_clusters < 2:
        raise ValueError("cluster covariance requires at least two clusters")
    output = np.empty((residuals.shape[1], rank, rank), dtype=np.float64)
    correction = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - rank))
    for horizon, residual in enumerate(residuals.T):
        scores = np.empty((n_clusters, rank), dtype=np.float64)
        for feature in range(rank):
            scores[:, feature] = np.bincount(
                codes, weights=x[:, feature] * residual, minlength=n_clusters
            )
        output[horizon] = correction * bread @ (scores.T @ scores) @ bread
    return output
