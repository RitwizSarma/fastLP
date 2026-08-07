"""Covariance estimators for cached, residualized panel regressions.

All functions accept every horizon's residuals at once.  Point estimation is
therefore independent of the covariance choice and keeps the shared-design
cache valid.
"""

from __future__ import annotations

from itertools import combinations
from math import floor, pi, sin, cos
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

try:  # Built by maturin when the optional native backend is available.
    from ._fastlp_rust import cluster_meat as _rust_cluster_meat
except ImportError:  # pragma: no cover - exercised only in native builds.
    _rust_cluster_meat = None

Kernel = Literal["bartlett", "parzen", "quadratic_spectral"]


def _symmetrize(covariance: np.ndarray) -> np.ndarray:
    return (covariance + covariance.swapaxes(-1, -2)) / 2


def _sandwich(meat: np.ndarray, bread: np.ndarray) -> np.ndarray:
    """Apply a shared bread matrix to one meat per horizon."""
    return np.asarray([bread @ item @ bread for item in meat])


def homoskedastic(x: np.ndarray, residuals: np.ndarray, bread: np.ndarray) -> np.ndarray:
    """Classical residual-degrees-of-freedom covariance for every horizon."""
    n_obs, rank = x.shape
    sigma2 = np.sum(residuals**2, axis=0) / (n_obs - rank)
    return sigma2[:, None, None] * bread


def hc(x: np.ndarray, residuals: np.ndarray, bread: np.ndarray, kind: str) -> np.ndarray:
    """Return HC0--HC3 covariance matrices for all residual columns."""
    n_obs, rank = x.shape
    if kind not in {"hc0", "hc1", "hc2", "hc3"}:
        raise ValueError(f"unknown heteroskedastic covariance {kind!r}")
    weights = residuals**2
    if kind == "hc1":
        weights = weights * (n_obs / (n_obs - rank))
    elif kind in {"hc2", "hc3"}:
        leverage = np.einsum("ij,jk,ik->i", x, bread, x, optimize=True)
        remaining = 1.0 - leverage
        if np.any(remaining <= np.finfo(float).eps):
            raise ValueError("HC2/HC3 is undefined because a leverage value is one")
        power = 1 if kind == "hc2" else 2
        weights = weights / remaining[:, None] ** power
    meat = np.einsum("ni,nj,nh->hij", x, x, weights, optimize=True)
    return _symmetrize(_sandwich(meat, bread))


def hc1(x: np.ndarray, residuals: np.ndarray, bread: np.ndarray) -> np.ndarray:
    """Backward-compatible HC1 spelling."""
    return hc(x, residuals, bread, "hc1")


def factorize_cluster_terms(frame, terms: tuple[tuple[str, ...], ...]) -> tuple[list[np.ndarray], list[str]]:
    """Factorize categorical cluster terms, including interaction terms."""
    codes: list[np.ndarray] = []
    labels: list[str] = []
    for term in terms:
        encoded, n_categories = frame.factorize(term)
        if (encoded < 0).any():
            raise ValueError(f"cluster term {'#'.join(term)!r} contains missing values")
        if n_categories < 2:
            raise ValueError(f"cluster term {'#'.join(term)!r} requires at least two clusters")
        codes.append(np.asarray(encoded, dtype=np.int64))
        labels.append("#".join(term))
    return codes, labels


def _intersection_codes(codes: list[np.ndarray], subset: tuple[int, ...]) -> np.ndarray:
    if len(subset) == 1:
        return codes[subset[0]]
    stacked = np.column_stack([codes[index] for index in subset])
    encoded, _ = pd.factorize(pd.MultiIndex.from_arrays(stacked.T), sort=True)
    return np.asarray(encoded, dtype=np.int64)


def _cluster_meat(x: np.ndarray, residuals: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Sum cluster score outer products, batching horizons to cap memory."""
    n_clusters = int(codes.max()) + 1
    if _rust_cluster_meat is not None:
        return np.asarray(
            _rust_cluster_meat(
                np.ascontiguousarray(x, dtype=np.float64),
                np.ascontiguousarray(residuals, dtype=np.float64),
                np.ascontiguousarray(codes, dtype=np.int64),
                n_clusters,
            )
        )
    n_obs, rank = x.shape
    del n_obs
    n_horizons = residuals.shape[1]
    output = np.empty((n_horizons, rank, rank), dtype=np.float64)
    # A group-score block has G * K * B doubles.  Keep it below about 64 MiB.
    batch = max(1, (64 * 1024 * 1024) // max(8 * n_clusters * rank, 1))
    for first in range(0, n_horizons, batch):
        last = min(first + batch, n_horizons)
        width = last - first
        scores = np.zeros((n_clusters, rank, width), dtype=np.float64)
        for feature in range(rank):
            np.add.at(scores[:, feature, :], codes, x[:, feature, None] * residuals[:, first:last])
        output[first:last] = np.einsum("gib,gjb->bij", scores, scores, optimize=True)
    return output


def cluster(
    x: np.ndarray,
    residuals: np.ndarray,
    codes: list[np.ndarray],
    labels: list[str],
    bread: np.ndarray,
    correction: Literal["cr0", "cr1"],
) -> tuple[np.ndarray, tuple[dict[str, object], ...]]:
    """Cameron--Gelbach--Miller multiway CR0/CR1 covariance.

    ``codes`` holds one array per requested cluster term.  Every non-empty
    intersection is included with alternating sign, so interaction terms are
    first-class cluster dimensions rather than parsed special cases.
    """
    if not codes:
        raise ValueError("cluster covariance requires at least one cluster term")
    n_obs, rank = x.shape
    meat = np.zeros((residuals.shape[1], rank, rank), dtype=np.float64)
    diagnostics: list[dict[str, object]] = []
    for width in range(1, len(codes) + 1):
        sign = 1 if width % 2 else -1
        for subset in combinations(range(len(codes)), width):
            combined = _intersection_codes(codes, subset)
            n_clusters = int(combined.max()) + 1
            if n_clusters < 2:
                joined = "#".join(labels[index] for index in subset)
                raise ValueError(f"cluster intersection {joined!r} requires at least two clusters")
            component = _cluster_meat(x, residuals, combined)
            if correction == "cr1":
                component *= (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - rank))
            meat += sign * component
            diagnostics.append(
                {"terms": tuple(labels[index] for index in subset), "n_clusters": n_clusters, "sign": sign}
            )
    return _symmetrize(_sandwich(meat, bread)), tuple(diagnostics)


def cluster_cr1(x: np.ndarray, residuals: np.ndarray, codes: np.ndarray, bread: np.ndarray) -> np.ndarray:
    """Backward-compatible one-way CR1 covariance spelling."""
    result, _ = cluster(x, residuals, [codes], ["cluster"], bread, "cr1")
    return result


def _kernel_weight(kernel: Kernel, lag: int, bandwidth: int) -> float:
    if lag == 0:
        return 1.0
    if bandwidth == 0 or lag > bandwidth:
        return 0.0
    ratio = lag / (bandwidth + 1)
    if kernel == "bartlett":
        return 1.0 - ratio
    if kernel == "parzen":
        if ratio <= 0.5:
            return 1.0 - 6.0 * ratio**2 + 6.0 * ratio**3
        return 2.0 * (1.0 - ratio) ** 3
    # Quadratic spectral kernel, using the conventional Andrews scaling.
    z = 6.0 * pi * ratio / 5.0
    return 25.0 / (12.0 * pi**2 * ratio**2) * (sin(z) / z - cos(z))


def _integer_time(values: np.ndarray) -> np.ndarray:
    try:
        numeric = np.asarray(values).astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("HAC and Driscoll-Kraay require numeric integral time labels") from error
    if not np.isfinite(numeric).all() or not np.allclose(numeric, np.rint(numeric)):
        raise ValueError("HAC and Driscoll-Kraay require numeric integral time labels")
    return np.rint(numeric).astype(np.int64)


def _lag_pairs(unit: np.ndarray | None, time: np.ndarray, max_lag: int) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return current/lagged row positions for exact calendar lags."""
    if unit is None:
        keys = {(int(t),): index for index, t in enumerate(time)}
        prefix = lambda index: ()  # noqa: E731
    else:
        unit_codes, _ = pd.factorize(np.asarray(unit), sort=True)
        keys = {(int(unit_codes[index]), int(t)): index for index, t in enumerate(time)}
        prefix = lambda index: (int(unit_codes[index]),)  # noqa: E731
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for lag in range(max_lag + 1):
        current: list[int] = []
        previous: list[int] = []
        for index, value in enumerate(time):
            other = keys.get((*prefix(index), int(value - lag)))
            if other is not None:
                current.append(index)
                previous.append(other)
        pairs.append((np.asarray(current, dtype=np.intp), np.asarray(previous, dtype=np.intp)))
    return tuple(pairs)


def default_bandwidth(horizon: int, n_time: int) -> int:
    """Horizon-aware Newey--West default on the effective time dimension."""
    return max(horizon, floor(4 * (n_time / 100) ** (2 / 9)))


def _hac_meat(scores: np.ndarray, pairs: tuple[tuple[np.ndarray, np.ndarray], ...], bandwidth: int, kernel: Kernel) -> np.ndarray:
    rank = scores.shape[1]
    meat = np.zeros((rank, rank), dtype=np.float64)
    for lag in range(bandwidth + 1):
        current, previous = pairs[lag]
        if not len(current):
            continue
        gamma = scores[current].T @ scores[previous]
        if lag == 0:
            meat += gamma
        else:
            meat += _kernel_weight(kernel, lag, bandwidth) * (gamma + gamma.T)
    return meat


def hac(
    x: np.ndarray,
    residuals: np.ndarray,
    unit: np.ndarray,
    time: np.ndarray,
    horizons: Sequence[int],
    bread: np.ndarray,
    *,
    max_lags: int | None,
    kernel: Kernel,
    debias: bool,
    driscoll_kraay: bool,
) -> tuple[np.ndarray, tuple[dict[str, object], ...]]:
    """Panel Newey--West or Driscoll--Kraay covariance for each horizon."""
    numeric_time = _integer_time(time)
    n_obs, rank = x.shape
    n_time = len(np.unique(numeric_time))
    bandwidths = [max_lags if max_lags is not None else default_bandwidth(h, n_time) for h in horizons]
    if any(bandwidth < 0 for bandwidth in bandwidths):
        raise ValueError("hac_lags must be non-negative")
    max_bandwidth = max(bandwidths, default=0)
    if driscoll_kraay:
        time_codes, unique_time = pd.factorize(numeric_time, sort=True)
        n_groups = len(unique_time)
        # The aggregate score series has one observation for each calendar date.
        pairs = _lag_pairs(None, np.asarray(unique_time, dtype=np.int64), max_bandwidth)
    else:
        time_codes = None
        n_groups = None
        pairs = _lag_pairs(unit, numeric_time, max_bandwidth)

    output = np.empty((residuals.shape[1], rank, rank), dtype=np.float64)
    diagnostics: list[dict[str, object]] = []
    correction = n_obs / (n_obs - rank) if debias else 1.0
    for column, bandwidth in enumerate(bandwidths):
        scores = x * residuals[:, column, None]
        if driscoll_kraay:
            aggregate = np.zeros((int(n_groups), rank), dtype=np.float64)
            np.add.at(aggregate, time_codes, scores)
            scores = aggregate
        meat = _hac_meat(scores, pairs, bandwidth, kernel)
        output[column] = correction * bread @ meat @ bread
        diagnostics.append(
            {
                "bandwidth": bandwidth,
                "kernel": kernel,
                "debiased": debias,
                "time_periods": n_time,
                "kind": "driscoll_kraay" if driscoll_kraay else "hac",
            }
        )
    return _symmetrize(output), tuple(diagnostics)
