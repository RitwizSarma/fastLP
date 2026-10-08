"""Covariance estimators for cached, residualized panel regressions.

All functions accept every horizon's residuals at once.  Point estimation is
therefore independent of the covariance choice and keeps the shared-design
cache valid.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import combinations
from math import cos, floor, pi, sin
from typing import Literal

import numpy as np
import pandas as pd

from ._time import canonical_time

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


def homoskedastic(
    x: np.ndarray,
    residuals: np.ndarray,
    bread: np.ndarray,
    *,
    model_rank: int | None = None,
) -> np.ndarray:
    """Classical residual-degrees-of-freedom covariance for every horizon."""
    n_obs, rank = x.shape
    rank = rank if model_rank is None else model_rank
    sigma2 = np.sum(residuals**2, axis=0) / (n_obs - rank)
    return sigma2[:, None, None] * bread


def hc(
    x: np.ndarray,
    residuals: np.ndarray,
    bread: np.ndarray,
    kind: str,
    *,
    model_rank: int | None = None,
) -> np.ndarray:
    """Return HC0--HC3 covariance matrices for all residual columns."""
    n_obs, rank = x.shape
    if kind not in {"hc0", "hc1", "hc2", "hc3"}:
        raise ValueError(f"unknown heteroskedastic covariance {kind!r}")
    weights = residuals**2
    if kind == "hc1":
        correction_rank = rank if model_rank is None else model_rank
        weights = weights * (n_obs / (n_obs - correction_rank))
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


def factorize_cluster_terms(
    frame, terms: tuple[tuple[str, ...], ...]
) -> tuple[list[np.ndarray], list[str]]:
    """Factorize categorical cluster terms, including interaction terms."""
    codes: list[np.ndarray] = []
    labels: list[str] = []
    for term in terms:
        encoded, n_categories = frame.factorize(term)
        if (encoded < 0).any():
            raise ValueError(f"cluster term {'#'.join(term)!r} contains missing values")
        if n_categories < 2:
            raise ValueError(
                f"cluster term {'#'.join(term)!r} requires at least two clusters"
            )
        codes.append(np.asarray(encoded, dtype=np.int64))
        labels.append("#".join(term))
    return codes, labels


def _intersection_codes(codes: list[np.ndarray], subset: tuple[int, ...]) -> np.ndarray:
    if len(subset) == 1:
        return codes[subset[0]]
    stacked = np.column_stack([codes[index] for index in subset])
    encoded, _ = pd.factorize(pd.MultiIndex.from_arrays(stacked.T), sort=True)
    return np.asarray(encoded, dtype=np.int64)


def _cluster_meat(
    x: np.ndarray, residuals: np.ndarray, codes: np.ndarray
) -> np.ndarray:
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
            np.add.at(
                scores[:, feature, :],
                codes,
                x[:, feature, None] * residuals[:, first:last],
            )
        output[first:last] = np.einsum("gib,gjb->bij", scores, scores, optimize=True)
    return output


def cluster(
    x: np.ndarray,
    residuals: np.ndarray,
    codes: list[np.ndarray],
    labels: list[str],
    bread: np.ndarray,
    correction: Literal["cr0", "cr1"],
    *,
    correction_rank: int | None = None,
    group_adjustment: str = "component",
    return_roundoff_scale: bool = False,
) -> (
    tuple[np.ndarray, tuple[dict[str, object], ...]]
    | tuple[np.ndarray, tuple[dict[str, object], ...], np.ndarray]
):
    """Cameron--Gelbach--Miller multiway CR0/CR1 covariance.

    ``codes`` holds one array per requested cluster term.  Every non-empty
    intersection is included with alternating sign, so interaction terms are
    first-class cluster dimensions rather than parsed special cases.
    """
    if not codes:
        raise ValueError("cluster covariance requires at least one cluster term")
    n_obs, rank = x.shape
    correction_rank = rank if correction_rank is None else correction_rank
    minimum_clusters = min(int(item.max()) + 1 for item in codes)
    meat = np.zeros((residuals.shape[1], rank, rank), dtype=np.float64)
    roundoff_scale = np.zeros_like(meat) if return_roundoff_scale else None
    diagnostics: list[dict[str, object]] = []
    for width in range(1, len(codes) + 1):
        sign = 1 if width % 2 else -1
        for subset in combinations(range(len(codes)), width):
            combined = _intersection_codes(codes, subset)
            n_clusters = int(combined.max()) + 1
            if n_clusters < 2:
                joined = "#".join(labels[index] for index in subset)
                raise ValueError(
                    f"cluster intersection {joined!r} requires at least two clusters"
                )
            component = _cluster_meat(x, residuals, combined)
            if correction == "cr1":
                adjustment_clusters = (
                    minimum_clusters if group_adjustment == "min" else n_clusters
                )
                component *= (adjustment_clusters / (adjustment_clusters - 1)) * (
                    (n_obs - 1) / (n_obs - correction_rank)
                )
            meat += sign * component
            if roundoff_scale is not None:
                roundoff_scale += np.abs(_sandwich(component, bread))
            diagnostics.append(
                {
                    "terms": tuple(labels[index] for index in subset),
                    "n_clusters": n_clusters,
                    "sign": sign,
                    "adjustment_clusters": (
                        minimum_clusters if group_adjustment == "min" else n_clusters
                    )
                    if correction == "cr1"
                    else None,
                }
            )
    covariance = _symmetrize(_sandwich(meat, bread))
    if roundoff_scale is not None:
        return covariance, tuple(diagnostics), _symmetrize(roundoff_scale)
    return covariance, tuple(diagnostics)


def cluster_cr1(
    x: np.ndarray, residuals: np.ndarray, codes: np.ndarray, bread: np.ndarray
) -> np.ndarray:
    """Backward-compatible one-way CR1 covariance spelling."""
    result, _ = cluster(x, residuals, [codes], ["cluster"], bread, "cr1")
    return result


def _quadratic_spectral_weight(z: float | np.ndarray) -> float | np.ndarray:
    """Evaluate QS weights without cancellation near zero."""
    def series(value):
        square = value * value
        return 1.0 + square * (
            -1.0 / 10.0 + square * (
                1.0 / 280.0 + square * (-1.0 / 15120.0 + square / 1330560.0)
            )
        )

    # At |z| < 0.1 the omitted z**10 term is below 6e-19.
    if np.isscalar(z):
        if abs(z) < 0.1:
            return series(z)
        return 3.0 * (sin(z) / z - cos(z)) / z**2
    weights = np.empty_like(z)
    small = np.abs(z) < 0.1
    weights[small] = series(z[small])
    large_z = z[~small]
    weights[~small] = (
        3.0 * (np.sin(large_z) / large_z - np.cos(large_z)) / large_z**2
    )
    return weights


def _kernel_weight(kernel: Kernel, lag: int, bandwidth: int) -> float:
    if lag == 0:
        return 1.0
    if bandwidth == 0:
        return 0.0
    if kernel == "quadratic_spectral":
        ratio = lag / bandwidth
        z = 6.0 * pi * ratio / 5.0
        return float(_quadratic_spectral_weight(z))
    if lag > bandwidth:
        return 0.0
    ratio = lag / (bandwidth + 1)
    if kernel == "bartlett":
        return 1.0 - ratio
    if kernel == "parzen":
        if ratio <= 0.5:
            return 1.0 - 6.0 * ratio**2 + 6.0 * ratio**3
        return 2.0 * (1.0 - ratio) ** 3
    raise ValueError(f"unknown HAC kernel {kernel!r}")


def _integer_time(values: np.ndarray) -> np.ndarray:
    return canonical_time(values)


def _lag_pairs(
    unit: np.ndarray | None, time: np.ndarray, max_lag: int
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return current/lagged row positions for exact calendar lags."""
    return tuple(_iter_lag_pairs(unit, time, max_lag))


def _iter_lag_pairs(unit: np.ndarray | None, time: np.ndarray, max_lag: int):
    """Yield one exact pair vector at a time to bound budgeted HAC storage."""
    if unit is None:
        keys = {(int(t),): index for index, t in enumerate(time)}
        prefix = lambda index: ()
    else:
        unit_codes, _ = pd.factorize(np.asarray(unit), sort=True)
        keys = {(int(unit_codes[index]), int(t)): index for index, t in enumerate(time)}
        prefix = lambda index: (int(unit_codes[index]),)
    span = int(time.max()) - int(time.min()) if len(time) else 0
    for lag in range(min(max_lag, span) + 1):
        current: list[int] = []
        previous: list[int] = []
        for index, value in enumerate(time):
            other = keys.get((*prefix(index), int(value) - lag))
            if other is not None:
                current.append(index)
                previous.append(other)
        yield (
            np.asarray(current, dtype=np.intp), np.asarray(previous, dtype=np.intp)
        )


def default_bandwidth(horizon: int, n_time: int) -> int:
    """Horizon-aware Newey--West default on the effective time dimension."""
    return max(horizon, floor(4 * (n_time / 100) ** (2 / 9)))


def _hac_meat(
    scores: np.ndarray,
    pairs: Iterable[tuple[np.ndarray, np.ndarray]],
    bandwidth: int,
    kernel: Kernel,
) -> np.ndarray:
    rank = scores.shape[1]
    meat = np.zeros((rank, rank), dtype=np.float64)
    for lag, (current, previous) in enumerate(pairs):
        if lag > bandwidth:
            break
        if not len(current):
            continue
        gamma = scores[current].T @ scores[previous]
        if lag == 0:
            meat += gamma
        else:
            meat += _kernel_weight(kernel, lag, bandwidth) * (gamma + gamma.T)
    return meat


_QS_BLOCK_ELEMENTS = 1_000_000


def _dense_quadratic_spectral_meat(
    scores: np.ndarray,
    group_codes: np.ndarray,
    time: np.ndarray,
    bandwidth: int,
) -> np.ndarray | None:
    """Use FFT cross-correlations when every group shares one dense time grid."""
    order = np.lexsort((time, group_codes))
    ordered_codes = group_codes[order]
    boundaries = np.flatnonzero(np.diff(ordered_codes)) + 1
    counts = np.diff(np.concatenate(([0], boundaries, [len(order)])))
    if not len(counts) or np.any(counts != counts[0]):
        return None
    n_periods = int(counts[0])
    group_time = time[order].reshape(len(counts), n_periods)
    spacing = np.diff(group_time[0])
    if np.any(group_time != group_time[0]) or (
        n_periods > 1 and (spacing[0] <= 0 or np.any(spacing != spacing[0]))
    ):
        return None
    time_step = int(spacing[0]) if n_periods > 1 else 1

    score_cube = scores[order].reshape(len(counts), n_periods, scores.shape[1])
    fft_size = 1 << (2 * n_periods - 1).bit_length()
    transforms = np.fft.rfft(score_cube, n=fft_size, axis=1)
    cross_products = np.einsum(
        "ufi,ufj->fij", transforms, transforms.conj(), optimize=True
    )
    correlations = np.fft.irfft(cross_products, n=fft_size, axis=0)[:n_periods]
    meat = correlations[0].copy()
    for lag in range(1, n_periods):
        gamma = correlations[lag]
        meat += _kernel_weight("quadratic_spectral", lag * time_step, bandwidth) * (
            gamma + gamma.T
        )
    return meat


def _quadratic_spectral_meat(
    scores: np.ndarray,
    unit: np.ndarray | None,
    time: np.ndarray,
    bandwidth: int,
) -> np.ndarray:
    """All-lag QS score covariance, blocked to bound temporary memory."""
    if bandwidth == 0:
        # A zero bandwidth means no serial correlation adjustment, consistently
        # with the other kernels and with the public hac_lags=0 contract.
        return scores.T @ scores

    if unit is None:
        group_codes = np.zeros(len(scores), dtype=np.int64)
    else:
        group_codes, _ = pd.factorize(np.asarray(unit), sort=True)
    dense = _dense_quadratic_spectral_meat(scores, group_codes, time, bandwidth)
    if dense is not None:
        return dense
    order = np.argsort(group_codes, kind="stable")
    ordered_codes = group_codes[order]
    boundaries = np.flatnonzero(np.diff(ordered_codes)) + 1
    groups = np.split(order, boundaries)
    meat = np.zeros((scores.shape[1], scores.shape[1]), dtype=np.float64)
    for positions in groups:
        group_scores = scores[positions]
        group_time = time[positions]
        block_rows = max(1, min(len(positions), _QS_BLOCK_ELEMENTS // len(positions)))
        for start in range(0, len(positions), block_rows):
            stop = min(start + block_rows, len(positions))
            differences = np.abs(group_time[start:stop, None] - group_time[None, :])
            z = 6.0 * pi * differences.astype(np.float64) / (5.0 * bandwidth)
            weights = _quadratic_spectral_weight(z)
            meat += group_scores[start:stop].T @ weights @ group_scores
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
    model_rank: int | None = None,
    stream_pairs: bool = False,
) -> tuple[np.ndarray, tuple[dict[str, object], ...]]:
    """Panel Newey--West or Driscoll--Kraay covariance for each horizon."""
    numeric_time = _integer_time(time)
    if (
        len(numeric_time)
        and int(numeric_time.max()) - int(numeric_time.min()) > np.iinfo(np.int64).max
    ):
        raise ValueError("time span exceeds the supported int64 lag range")
    n_obs, rank = x.shape
    n_time = len(np.unique(numeric_time))
    bandwidths = [
        max_lags if max_lags is not None else default_bandwidth(h, n_time)
        for h in horizons
    ]
    if any(bandwidth < 0 for bandwidth in bandwidths):
        raise ValueError("hac_lags must be non-negative")
    max_bandwidth = max(bandwidths, default=0)
    if driscoll_kraay:
        time_codes, unique_time = pd.factorize(numeric_time, sort=True)
        n_groups = len(unique_time)
        score_time = np.asarray(unique_time, dtype=np.int64)
        score_unit = None
    else:
        time_codes = None
        n_groups = None
        score_time = numeric_time
        score_unit = unit
    pairs = (
        None
        if kernel == "quadratic_spectral" or stream_pairs
        else _lag_pairs(score_unit, score_time, max_bandwidth)
    )

    output = np.empty((residuals.shape[1], rank, rank), dtype=np.float64)
    diagnostics: list[dict[str, object]] = []
    correction_rank = rank if model_rank is None else model_rank
    correction = n_obs / (n_obs - correction_rank) if debias else 1.0
    for column, bandwidth in enumerate(bandwidths):
        scores = x * residuals[:, column, None]
        if driscoll_kraay:
            aggregate = np.zeros((int(n_groups), rank), dtype=np.float64)
            np.add.at(aggregate, time_codes, scores)
            scores = aggregate
        if kernel == "quadratic_spectral":
            meat = _quadratic_spectral_meat(scores, score_unit, score_time, bandwidth)
        else:
            current_pairs = (
                _iter_lag_pairs(score_unit, score_time, bandwidth)
                if stream_pairs else pairs
            )
            assert current_pairs is not None
            meat = _hac_meat(scores, current_pairs, bandwidth, kernel)
        output[column] = correction * bread @ meat @ bread
        diagnostics.append(
            {
                "bandwidth": bandwidth,
                "kernel": kernel,
                "lag_limit": None if kernel == "quadratic_spectral" else bandwidth,
                "debiased": debias,
                "time_periods": n_time,
                "kind": "driscoll_kraay" if driscoll_kraay else "hac",
            }
        )
    return _symmetrize(output), tuple(diagnostics)
