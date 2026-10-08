"""Conservative planning for estimator-owned frames and numerical arrays.

This is an allocation estimate, not an operating-system RSS limit. Caller-owned
input, Python object overhead, allocator caches and private library workspaces
are outside the contract.
"""

from dataclasses import dataclass
from math import floor

import pandas as pd


def input_size(data, required) -> tuple[int, int]:
    """Measure selected input storage before preparing/copying the panel."""
    names = list(dict.fromkeys(required))
    if isinstance(data, pd.DataFrame):
        size = int(data.index.memory_usage(deep=True))
        size += sum(
            int(data[name].memory_usage(index=False, deep=True))
            for name in names if name in data.columns
        )
        return len(data), size
    try:
        import polars as pl
    except ImportError as error:
        raise TypeError("unsupported panel input for memory planning") from error

    if isinstance(data, pl.LazyFrame):
        # Aggregate lengths, without collecting the whole lazy input. Strings
        # need their byte lengths as well as offsets in the prepared frame.
        schema = data.collect_schema()
        selected = [name for name in names if name in schema]
        sizes = [
            pl.col(name).str.len_bytes().sum().cast(pl.UInt64)
            + pl.len().cast(pl.UInt64) * 8
            if schema[name] == pl.String else pl.len().cast(pl.UInt64) * 16
            for name in selected
        ]
        count, size = data.select(
            pl.len().alias("__rows__"),
            (pl.sum_horizontal(sizes) if sizes else pl.lit(0)).alias("__bytes__"),
        ).collect().row(0)
        return int(count), int(size)
    if isinstance(data, pl.DataFrame):
        selected = data.select([name for name in names if name in data.columns])
        return len(data), int(selected.estimated_size())
    raise TypeError("unsupported panel input for memory planning")


@dataclass(frozen=True)
class MemoryPlan:
    budget: int
    reserved: int
    per_horizon: int
    stream_pairs: bool = False
    cached_pair_bytes: int = 0

    @property
    def minimum(self) -> int:
        return self.reserved - self.cached_pair_bytes + self.per_horizon

    def batch_size(self, maximum: int) -> int:
        if self.minimum > self.budget:
            raise ValueError(
                f"memory_budget={self.budget} bytes is below the planned minimum "
                f"of {self.minimum} bytes for panel storage and one horizon; "
                "increase memory_budget, reduce horizons/features, or disable "
                "retain_residuals"
            )
        return min(maximum, (self.budget - self.reserved) // self.per_horizon)

    def diagnostics(self, batch: int) -> dict:
        return {
            "budget_bytes": self.budget,
            "reserved_bytes": self.reserved,
            "bytes_per_batch_horizon": self.per_horizon,
            "minimum_bytes": self.minimum,
            "planned_peak_bytes": self.reserved + batch * self.per_horizon,
            "batch_size": batch,
            "stream_hac_pairs": self.stream_pairs,
            "cached_hac_pair_bytes": self.cached_pair_bytes,
            "scope": "estimated estimator-owned frames and arrays; not process RSS",
        }


def plan_memory(
    *, budget: int, rows: int, frame_bytes: int, features: int, horizons: int,
    lag_features: int, effects: int, cluster_terms: int, retain_residuals: bool,
    covariance: str, kernel: str, exact_rank: bool,
    hac_lags: int | None = None,
) -> MemoryPlan:
    """Reserve worst-sample-size storage before large estimator allocations.

    Coefficients/results, alignment, masks, retained sample IDs, lag columns and
    design/absorption buffers are reserved independently of outcome batch size.
    Headroom includes overlapping old/new buffers between groups and batches.
    """
    n, k, h = int(rows), int(features), int(horizons) + 1
    # Up to three selected-frame copies plus numeric conversion/lag buffers.
    frames = 3 * frame_bytes + 24 * n * lag_features
    # Lead positions, masks/pruning caches, packed keys and retained row IDs.
    alignment = 32 * n * h
    # Common-sample residual assembly temporarily retains both representations.
    residuals = 16 * n * h if retain_residuals else 0
    # Result arrays and per-mask solver diagnostics (at most h masks).
    results = h * (64 * k * k + 128 * k + 256)
    # Design, factorization, absorption, grouping, sorting and gather buffers.
    workspace = n * (256 * k + 128 * effects + 128 * cluster_terms + 512)
    workspace += 128 * k * k
    if effects >= 3 and exact_rank:
        # Dummy matrix is already capped at 64 MiB; allow overlapping rank work.
        workspace += 4 * min(64 * 1024**2, 8 * n * n * effects)
    if covariance in {"hac", "driscoll_kraay"}:
        if kernel == "quadratic_spectral":
            # FFT buffers or one bounded pair block, whichever is larger.
            workspace += max(
                128 * n * (k + k * k),
                128 * min(n * n, max(1_000_000, n)),
            )
        else:
            # Budgeted HAC streams one pair vector instead of caching all lags.
            workspace += 64 * n
    # Outcomes, transformed outcomes, residuals, absorber temporaries, cluster
    # scores, and covariance batches; includes transition allocation headroom.
    per_horizon = max(1, n * (128 + 32 * k) + 128 * k * k)
    reserved = frames + alignment + residuals + results + workspace
    stream_pairs = covariance in {"hac", "driscoll_kraay"} and kernel != "quadratic_spectral"
    cached_pair_bytes = 0
    if stream_pairs:
        bandwidth = (
            hac_lags if hac_lags is not None
            else max(horizons, floor(4 * (n / 100) ** (2 / 9)))
        )
        cached_pairs = 16 * n * min(bandwidth + 1, n)
        if reserved + cached_pairs + per_horizon * min(h, 32) <= budget:
            reserved += cached_pairs
            cached_pair_bytes = cached_pairs
            stream_pairs = False
    return MemoryPlan(budget, reserved, per_horizon, stream_pairs, cached_pair_bytes)
