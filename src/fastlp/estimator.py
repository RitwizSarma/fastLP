"""Scikit-learn-style cached panel local projections estimator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Integral, Real
from typing import Any
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm as normal_dist
from scipy.stats import t as student_t

from ._covariance import (
    Kernel,
    cluster as cluster_covariance,
    factorize_cluster_terms,
    hac,
    hc,
    homoskedastic,
)
from ._demean import DemeanDiagnostics, Residualizer, balanced_demean, factorize_effects
from ._degrees_of_freedom import cluster_parameter_count, effect_rank
from ._frame import PanelFrame, prepare_panel
from ._linalg import RegressionDesign
from ._memory import input_size, plan_memory


def _as_columns(value: str | Sequence[str] | None, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    columns = tuple(value)
    if not all(isinstance(column, str) for column in columns):
        raise ValueError(f"{name} must contain only column names")
    return columns


def _with_column_names(data: Any, column_names: Sequence[str] | None) -> Any:
    """Convert a named two-dimensional array to the standard dataframe input."""
    if not isinstance(data, np.ndarray):
        if column_names is None:
            return data
        raise ValueError("column_names is only valid when data is a NumPy array")
    if column_names is None:
        raise ValueError("column_names must be supplied when data is a NumPy array")
    if data.ndim != 2:
        raise ValueError("NumPy array data must be two-dimensional")
    if isinstance(column_names, str):
        raise ValueError("column_names must be a sequence of column names")
    names = tuple(column_names)
    if not all(isinstance(name, str) for name in names):
        raise ValueError("column_names must contain only strings")
    if len(set(names)) != len(names):
        raise ValueError("column_names must not contain duplicates")
    if len(names) != data.shape[1]:
        raise ValueError("column_names must contain one name for each array column")
    return pd.DataFrame(data, columns=names)


ClusterTerm = str | tuple[str, ...]
_HORIZON_BATCH_SIZE = 32
_MAX_SUPPORTED_LAG = int(np.iinfo(np.int64).max)
_MAX_SCALAR_LAG_COUNT = 10_000


def _require_finite(values: np.ndarray, stage: str) -> None:
    """Reject numerical-range failures before publishing regression results."""
    if not np.isfinite(values).all():
        raise ValueError(
            f"numerical range exceeded while computing {stage}; "
            "nonfinite values encountered, rescale the model variables"
        )


def _cumulative_batch(
    outcome: np.ndarray,
    future_positions: np.ndarray,
    positions: np.ndarray,
    horizons: Sequence[int],
    running: np.ndarray,
    next_step: int,
) -> tuple[np.ndarray, int]:
    """Accumulate forward from each anchor, preserving state across batches."""
    result = np.empty((len(positions), len(horizons)), dtype=np.float64)
    for column, horizon in enumerate(horizons):
        for step in range(next_step, horizon + 1):
            running += outcome[future_positions[positions, step]]
        result[:, column] = running
        next_step = horizon + 1
    return result, next_step


class FewClustersWarning(UserWarning):
    """Cluster-robust inference may be unreliable with few clusters."""


def _memory_bytes(value: int | str | None) -> int | None:
    """Normalize a byte count or a compact human-readable memory budget."""
    if value is None:
        return None
    if isinstance(value, Integral) and not isinstance(value, bool):
        if value < 1:
            raise ValueError("memory_budget must be positive")
        return int(value)
    if not isinstance(value, str):
        raise ValueError("memory_budget must be bytes or a string such as '4GB'")
    normalized = value.strip().upper().replace(" ", "")
    units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
    for suffix in ("TB", "GB", "MB", "KB", "B"):
        if normalized.endswith(suffix):
            try:
                amount = float(normalized[: -len(suffix)])
            except ValueError as error:
                raise ValueError("memory_budget must be bytes or a string such as '4GB'") from error
            result = int(amount * units[suffix])
            if result < 1:
                raise ValueError("memory_budget must be positive")
            return result
    raise ValueError("memory_budget must be bytes or a string such as '4GB'")


def _balanced_panel_shape(frame: PanelFrame, unit: str, time: str) -> tuple[int, int] | None:
    """Return the dense panel shape when every unit has the same time labels."""
    return frame.balanced_shape(unit, time)


def _prune_singletons(
    mask: np.ndarray, effect_codes: np.ndarray, group_counts: np.ndarray
) -> tuple[np.ndarray, int, int]:
    """Recursively remove rows singleton in any fixed-effect dimension."""
    retained = mask.copy()
    initial = int(retained.sum())
    rounds = 0
    while retained.any():
        positions = np.flatnonzero(retained)
        drop = np.zeros(len(positions), dtype=bool)
        for dimension, n_groups in enumerate(group_counts):
            codes = effect_codes[positions, dimension]
            counts = np.bincount(codes, minlength=int(n_groups))
            drop |= counts[codes] == 1
        if not drop.any():
            break
        retained[positions[drop]] = False
        rounds += 1
    return retained, initial - int(retained.sum()), rounds


def _within_signal_floor(
    n_obs: int,
    effect_codes: np.ndarray,
    effect_counts: np.ndarray,
    *,
    iterative: bool,
    tolerance: float,
) -> float:
    """Normalized L2 signal below which FE absorption is numerically unresolved."""
    if not effect_counts.size:
        return 0.0
    largest_group = max(
        int(np.bincount(effect_codes[:, dimension]).max())
        for dimension in range(effect_codes.shape[1])
    )
    accumulation_length = n_obs if len(effect_counts) > 1 else largest_group
    roundoff = accumulation_length * np.finfo(np.float64).eps
    if roundoff >= 1.0:
        return np.inf
    roundoff /= 1.0 - roundoff
    per_observation_error = (len(effect_counts) + 1) * roundoff
    if iterative:
        per_observation_error += 10.0 * tolerance
    return float(np.sqrt(n_obs) * per_observation_error)


def _covariance_roundoff_multiplier(n_obs: int, n_features: int) -> float:
    """Conservative gamma bound for score accumulation and sandwich products."""
    operations = 2 * n_obs + 4 * n_features
    product = operations * np.finfo(np.float64).eps
    return np.inf if product >= 1.0 else product / (1.0 - product)


def _exact_panel_demean(
    values: np.ndarray,
    frame: PanelFrame,
    effects: tuple[str, ...],
    *,
    unit: str,
    time: str,
) -> tuple[np.ndarray, DemeanDiagnostics] | None:
    """Apply exact one-/two-way within transforms on a dense panel."""
    if (
        not effects
        or len(set(effects)) != len(effects)
        or not set(effects).issubset({unit, time})
    ):
        return None
    values = np.asarray(values, dtype=np.float64)
    values = values - values.mean(axis=0)
    shape = _balanced_panel_shape(frame, unit, time)
    if len(effects) == 1 and shape is None:
        codes, n_groups = frame.factorize((effects[0],))
        counts = np.bincount(codes, minlength=n_groups).astype(np.float64)
        transformed = values.copy()
        for column in range(transformed.shape[1]):
            sums = np.bincount(codes, weights=transformed[:, column], minlength=n_groups)
            transformed[:, column] -= sums[codes] / counts[codes]
        method = "exact_group_unit" if effects == (unit,) else "exact_group_time"
        return transformed, DemeanDiagnostics(
            np.ones(values.shape[1], dtype=int),
            "numpy",
            method,
            np.zeros(values.shape[1], dtype=int),
        )
    if shape is None:
        return None
    n_units, n_periods = shape
    effect_set = set(effects)
    transformed, backend = balanced_demean(
        values,
        n_units,
        n_periods,
        unit_effect=unit in effect_set,
        time_effect=time in effect_set,
    )
    if set(effects) == {unit, time}:
        method = "exact_balanced_two_way"
    elif effects == (unit,):
        method = "exact_balanced_unit"
    else:
        method = "exact_balanced_time"
    iterations = np.ones(values.shape[1], dtype=int)
    return transformed, DemeanDiagnostics(
        iterations, backend, method, np.zeros(values.shape[1], dtype=int)
    )


def _as_cluster_terms(value: ClusterTerm | Sequence[ClusterTerm] | None) -> tuple[tuple[str, ...], ...]:
    """Normalize cluster terms; a bare tuple denotes one interaction term."""
    if value is None:
        return ()
    if isinstance(value, str):
        raw_terms: Sequence[ClusterTerm] = (value,)
    elif isinstance(value, tuple) and all(isinstance(column, str) for column in value):
        raw_terms = (value,)
    else:
        if isinstance(value, (str, bytes)):
            raise ValueError("cluster must be a column, an interaction tuple, or a sequence of terms")
        try:
            raw_terms = tuple(value)
        except TypeError as error:
            raise ValueError("cluster must be a column, an interaction tuple, or a sequence of terms") from error

    terms: list[tuple[str, ...]] = []
    for item in raw_terms:
        if isinstance(item, str):
            term = (item,)
        elif isinstance(item, tuple) and item and all(isinstance(column, str) for column in item):
            if len(set(item)) != len(item):
                raise ValueError("a cluster interaction must not repeat a column")
            term = tuple(sorted(item))
        else:
            raise ValueError("cluster terms must be column names or non-empty tuples of column names")
        terms.append(term)
    if not terms:
        raise ValueError("cluster must contain at least one term")
    if len(terms) > 4:
        raise ValueError("cluster supports at most four terms because multiway covariance is exponential")
    if len(set(terms)) != len(terms):
        raise ValueError("cluster must not contain duplicate terms")
    return tuple(terms)


def _as_lag_numbers(value: int | Sequence[int], name: str) -> tuple[int, ...]:
    """Normalize a lag count or an explicit positive lag grid."""
    if isinstance(value, Integral) and not isinstance(value, bool):
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
        if value > _MAX_SUPPORTED_LAG:
            raise ValueError(f"{name} must not exceed the int64 lag limit")
        if value > _MAX_SCALAR_LAG_COUNT:
            raise ValueError(
                f"{name} scalar lag count must not exceed {_MAX_SCALAR_LAG_COUNT}; "
                "use a sequence for sparse, larger lag numbers"
            )
        return tuple(range(1, int(value) + 1))
    if isinstance(value, str):
        raise ValueError(f"{name} must be an integer or a sequence of positive integers")
    try:
        lags = tuple(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer or a sequence of positive integers") from error
    if any(not isinstance(lag, Integral) or isinstance(lag, bool) or lag < 1 for lag in lags):
        raise ValueError(f"{name} must contain only positive integers")
    if any(lag > _MAX_SUPPORTED_LAG for lag in lags):
        raise ValueError(f"{name} must not exceed the int64 lag limit")
    if len(set(lags)) != len(lags):
        raise ValueError(f"{name} must not contain duplicate lags")
    return tuple(int(lag) for lag in lags)


def _lag_requests(
    value: int | Sequence[int] | Mapping[str, int | Sequence[int]],
    columns: tuple[str, ...],
    name: str,
) -> tuple[tuple[str, int], ...]:
    """Expand one lag specification into ordered ``(source, lag)`` pairs."""
    if isinstance(value, Mapping):
        unknown = sorted(set(value).difference(columns))
        if unknown:
            raise ValueError(f"{name} contains columns not in this specification: {unknown}")
        requests = [
            (column, lag)
            for column in columns
            for lag in _as_lag_numbers(value.get(column, ()), f"{name}[{column!r}]")
        ]
    else:
        lags = _as_lag_numbers(value, name)
        requests = [(column, lag) for column in columns for lag in lags]
    return tuple(requests)


def _add_lags(
    frame: PanelFrame, unit: str, time: str, requests: Sequence[tuple[str, int]]
) -> tuple[PanelFrame, tuple[str, ...], np.ndarray, str]:
    """Generate exact within-unit calendar lags and valid LP anchors."""
    return frame.add_lags(unit, time, requests)


class LocalProjection:
    """Fast local projections estimator class.

    Estimate the impulse response at each horizon.
    """

    def __init__(
        self,
        *,
        horizons: int,
        covariance: str = "cluster",
        hac_lags: int | None = None,
        hac_kernel: Kernel = "bartlett",
        hac_debias: bool = True,
        cluster_correction: str = "cr1",
        alpha: float = 0.05,
        demean_tol: float = 1e-10,
        max_iter: int = 10_000,
        sample: str = "common",
        response: str = "level",
        retain_residuals: bool = True,
        memory_budget: int | str | None = None,
        demean_acceleration: str = "irons_tuck",
        few_cluster_threshold: int | None = 50,
        singleton_policy: str = "drop",
        fe_dof: str = "exact",
        cluster_df: str = "nonnested",
        cluster_group_adjustment: str = "min",
        cluster_inference: str = "t",
    ) -> None:
        if not isinstance(horizons, Integral) or isinstance(horizons, bool) or horizons < 0:
            raise ValueError("horizons must be a non-negative integer")
        allowed_covariance = {
            "homoskedastic",
            "hc0",
            "hc1",
            "hc2",
            "hc3",
            "cluster",
            "hac",
            "driscoll_kraay",
        }
        if covariance not in allowed_covariance:
            raise ValueError(f"covariance must be one of {sorted(allowed_covariance)}")
        if hac_lags is not None and (
            not isinstance(hac_lags, Integral)
            or isinstance(hac_lags, bool)
            or hac_lags < 0
        ):
            raise ValueError("hac_lags must be a non-negative integer or None")
        if hac_kernel not in {"bartlett", "parzen", "quadratic_spectral"}:
            raise ValueError("hac_kernel must be 'bartlett', 'parzen', or 'quadratic_spectral'")
        if not isinstance(hac_debias, bool):
            raise ValueError("hac_debias must be a boolean")
        if cluster_correction not in {"cr0", "cr1"}:
            raise ValueError("cluster_correction must be 'cr0' or 'cr1'")
        if not isinstance(alpha, Real) or isinstance(alpha, bool) or not 0 < alpha < 1:
            raise ValueError("alpha must be between zero and one")
        if float(alpha) / 2 == 0:
            raise ValueError("alpha is too small for float64 tail probabilities")
        if (
            not isinstance(demean_tol, Real)
            or isinstance(demean_tol, bool)
            or not isfinite(demean_tol)
            or demean_tol <= 0
        ):
            raise ValueError("demean_tol must be finite and positive")
        if not isinstance(max_iter, Integral) or isinstance(max_iter, bool) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if sample not in {"common", "per_horizon"}:
            raise ValueError("sample must be 'common' or 'per_horizon'")
        if response not in {"level", "cumulative"}:
            raise ValueError("response must be 'level' or 'cumulative'")
        if not isinstance(retain_residuals, bool):
            raise ValueError("retain_residuals must be a boolean")
        if demean_acceleration not in {"none", "aitken", "irons_tuck"}:
            raise ValueError(
                "demean_acceleration must be 'none', 'aitken', or 'irons_tuck'"
            )
        if few_cluster_threshold is not None and (
            not isinstance(few_cluster_threshold, Integral)
            or isinstance(few_cluster_threshold, bool)
            or few_cluster_threshold < 2
        ):
            raise ValueError("few_cluster_threshold must be at least two or None")
        if singleton_policy not in {"drop", "keep"}:
            raise ValueError("singleton_policy must be 'drop' or 'keep'")
        if fe_dof not in {"exact", "conservative"}:
            raise ValueError("fe_dof must be 'exact' or 'conservative'")
        if cluster_df not in {"nonnested", "full", "none"}:
            raise ValueError("cluster_df must be 'nonnested', 'full', or 'none'")
        if cluster_group_adjustment not in {"min", "component"}:
            raise ValueError("cluster_group_adjustment must be 'min' or 'component'")
        if cluster_inference not in {"t", "normal"}:
            raise ValueError("cluster_inference must be 't' or 'normal'")
        self.horizons = int(horizons)
        self.covariance = covariance
        self.hac_lags = None if hac_lags is None else int(hac_lags)
        self.hac_kernel = hac_kernel
        self.hac_debias = hac_debias
        self.cluster_correction = cluster_correction
        self.alpha = float(alpha)
        self.demean_tol = float(demean_tol)
        self.max_iter = int(max_iter)
        self.sample = sample
        self.response = response
        self.retain_residuals = retain_residuals
        self.memory_budget = _memory_bytes(memory_budget)
        self.demean_acceleration = demean_acceleration
        self.few_cluster_threshold = (
            None if few_cluster_threshold is None else int(few_cluster_threshold)
        )
        self.singleton_policy = singleton_policy
        self.fe_dof = fe_dof
        self.cluster_df = cluster_df
        self.cluster_group_adjustment = cluster_group_adjustment
        self.cluster_inference = cluster_inference

    def fit(
        self,
        data: Any,
        *,
        column_names: Sequence[str] | None = None,
        outcome: str,
        shock: str | Sequence[str],
        controls: Sequence[str] = (),
        outcome_lags: int | Sequence[int] = 0,
        shock_lags: int | Sequence[int] | Mapping[str, int | Sequence[int]] = 0,
        control_lags: int | Sequence[int] | Mapping[str, int | Sequence[int]] = 0,
        unit: str,
        time: str,
        fixed_effects: Sequence[str] = (),
        cluster: ClusterTerm | Sequence[ClusterTerm] | None = None,
    ) -> "LocalProjection":
        """Fit local projections.

        NumPy input must be a two-dimensional array accompanied by one unique
        name per column through ``column_names``. It is converted to pandas at
        the input boundary; dataframe inputs must not supply ``column_names``.

        A scalar lag setting includes all lags from one through that value;
        sequences select an arbitrary positive lag grid.  For shocks and
        controls, a mapping can assign a separate setting to each column.
        Lag numbers must fit in signed int64; scalar lag counts are limited
        to 10,000 to avoid expanding an impractically large feature grid.
        Lags use exact within-unit calendar matches at ``time - lag``. Rows
        without every requested lag are not used as LP anchors, but remain
        available as future outcomes.

        ``memory_budget`` bounds a conservative allocation plan for owned
        frames, alignment, results and numerical workspaces. Too-small budgets
        fail before panel preparation; remaining space determines batch size.
        It is not a process-RSS limit: caller input, Python/allocator overhead
        and private library workspaces are excluded.
        """
        data = _with_column_names(data, column_names)
        shocks = _as_columns(shock, "shock")
        if not shocks:
            raise ValueError("shock must be a column name or a non-empty sequence of column names")
        controls = _as_columns(controls, "controls")
        effects = _as_columns(fixed_effects, "fixed_effects")
        if effects and self.covariance in {"hc2", "hc3"}:
            raise ValueError(
                "HC2 and HC3 are not supported with fixed effects because exact "
                "full-model leverage is not yet available; use covariance='hc0' "
                "or covariance='hc1'"
            )
        cluster_terms = _as_cluster_terms(cluster) if cluster is not None else ()
        required = (outcome, unit, time, *shocks, *controls, *effects)
        if self.covariance == "cluster":
            if not cluster_terms:
                raise ValueError("cluster must be supplied when covariance='cluster'")
            required = (*required, *(column for term in cluster_terms for column in term))
        elif cluster_terms:
            raise ValueError("cluster is only valid when covariance='cluster'")
        if len(set((*shocks, *controls))) != len((*shocks, *controls)):
            raise ValueError("shock and controls must not contain duplicate columns")

        lag_requests = (
            *_lag_requests(outcome_lags, (outcome,), "outcome_lags"),
            *_lag_requests(shock_lags, shocks, "shock_lags"),
            *_lag_requests(control_lags, controls, "control_lags"),
        )

        numeric_columns = (outcome, *shocks, *controls)
        memory_plan = None
        planned_batch = _HORIZON_BATCH_SIZE
        if self.memory_budget is not None:
            rows, frame_bytes = input_size(data, required)
            memory_plan = plan_memory(
                budget=self.memory_budget, rows=rows, frame_bytes=frame_bytes,
                features=len(shocks) + len(controls) + len(set(lag_requests)) + int(not effects),
                horizons=self.horizons, lag_features=len(set(lag_requests)),
                effects=len(effects), cluster_terms=len(cluster_terms),
                retain_residuals=self.retain_residuals, covariance=self.covariance,
                kernel=self.hac_kernel, exact_rank=self.fe_dof == "exact",
                hac_lags=self.hac_lags,
            )
            planned_batch = memory_plan.batch_size(
                min(_HORIZON_BATCH_SIZE, self.horizons + 1)
            )
        frame, input_was_sorted = prepare_panel(data, required, numeric_columns)
        frame, lag_feature_names, valid_lag_rows, lag_path = _add_lags(
            frame, unit, time, lag_requests
        )
        if not len(frame):
            raise ValueError("data must contain at least one panel unit")
        future_positions, lead_path = self._lead_positions(frame, unit=unit, time=time)
        future_valid = future_positions >= 0
        if self.response == "cumulative":
            future_valid = np.logical_and.accumulate(future_valid, axis=1)
        masks = valid_lag_rows[:, None] & future_valid
        if self.sample == "common":
            masks = np.repeat(masks.all(axis=1, keepdims=True), self.horizons + 1, axis=1)
        singleton_dropped = np.zeros(self.horizons + 1, dtype=int)
        singleton_rounds = np.zeros(self.horizons + 1, dtype=int)
        if effects and self.singleton_policy == "drop":
            pruning_codes, pruning_counts = factorize_effects(frame, effects)
            pruned_by_key: dict[bytes, tuple[np.ndarray, int, int]] = {}
            for horizon in range(self.horizons + 1):
                key = np.packbits(masks[:, horizon]).tobytes()
                result = pruned_by_key.get(key)
                if result is None:
                    result = _prune_singletons(
                        masks[:, horizon], pruning_codes, pruning_counts
                    )
                    pruned_by_key[key] = result
                masks[:, horizon], singleton_dropped[horizon], singleton_rounds[horizon] = result

        base_features = (*shocks, *controls, *lag_feature_names)
        base_x = frame.matrix(base_features, dtype=np.float64)
        if not effects:
            base_x = np.column_stack((np.ones(len(frame), dtype=np.float64), base_x))
        feature_names = ("Intercept", *base_features) if not effects else base_features
        outcome_values = frame.column(outcome).astype(np.float64, copy=False)
        groups: dict[bytes, list[int]] = {}
        group_masks: list[np.ndarray] = []
        for horizon in range(self.horizons + 1):
            mask = masks[:, horizon]
            key = np.packbits(mask).tobytes()
            if key not in groups:
                groups[key] = []
                group_masks.append(mask)
            groups[key].append(horizon)

        n_features = base_x.shape[1]
        n_horizons = self.horizons + 1
        coefficients = np.empty((n_horizons, n_features), dtype=np.float64)
        standard_errors = np.empty_like(coefficients)
        covariances = np.empty((n_horizons, n_features, n_features), dtype=np.float64)
        residuals_by_horizon: list[np.ndarray | None] | None = (
            [None] * n_horizons if self.retain_residuals else None
        )
        covariance_diagnostics_by_horizon: list[dict[str, object] | None] = [None] * n_horizons
        y_iterations = np.empty(n_horizons, dtype=int)
        y_acceleration_accepted = np.empty(n_horizons, dtype=int)
        cache_group_by_horizon = np.empty(n_horizons, dtype=int)
        x_iterations_by_group: list[np.ndarray] = []
        acceleration_accepted_by_group: list[np.ndarray] = []
        linear_algebra_by_group: list[dict[str, object]] = []
        degrees_of_freedom_by_group: list[dict[str, object]] = []
        model_ranks = np.empty(n_horizons, dtype=int)
        correction_ranks = np.empty(n_horizons, dtype=int)
        inference_df = np.full(n_horizons, np.inf)
        backends: list[str] = []
        methods: list[str] = []
        warned_cluster_terms: set[str] = set()

        for group_id, (mask, horizons) in enumerate(zip(group_masks, groups.values(), strict=True)):
            if not mask.any():
                raise ValueError(f"no valid observations at horizon {horizons[0]}")
            positions = np.flatnonzero(mask)
            anchor = frame.take(positions)
            x = base_x[mask]
            effect_codes, effect_counts = factorize_effects(anchor, effects)
            # Rank depends on the final retained sample, but not on residualized
            # values. Resolve it before potentially expensive FE absorption so
            # an over-budget exact multiway calculation fails immediately.
            absorbed = effect_rank(effect_codes, effect_counts, mode=self.fe_dof)
            model_rank = n_features + absorbed.rank
            if len(anchor) <= model_rank:
                qualification = "" if absorbed.exact else " under the conservative rank bound"
                raise ValueError(
                    f"residual degrees of freedom must be positive{qualification} "
                    f"at horizon {horizons[0]} (n={len(anchor)}, model rank={model_rank})"
                )
            correction_rank = model_rank
            nested_effects: tuple[int, ...] = ()
            correction_effect_rank = absorbed
            cluster_data = None
            if self.covariance == "cluster":
                if cluster_terms == ((unit,),):
                    unit_sizes = anchor.group_sizes(unit)
                    cluster_codes = [
                        np.repeat(np.arange(len(unit_sizes), dtype=np.int64), unit_sizes)
                    ]
                    labels = [unit]
                else:
                    cluster_codes, labels = factorize_cluster_terms(anchor, cluster_terms)
                cluster_data = (cluster_codes, labels)
                correction_rank, nested_effects, correction_effect_rank = cluster_parameter_count(
                    n_features, effect_codes, effect_counts, cluster_codes,
                    policy=self.cluster_df, full_rank=absorbed, mode=self.fe_dof,
                )
                if self.cluster_correction == "cr1" and len(anchor) <= correction_rank:
                    raise ValueError(
                        "cluster correction requires positive degrees of freedom "
                        "under the selected parameter-count policy"
                    )
                if self.cluster_inference == "t":
                    inference_df[horizons] = min(int(codes.max()) for codes in cluster_codes)
                if self.few_cluster_threshold is not None:
                    for codes, label in zip(cluster_codes, labels, strict=True):
                        n_clusters = int(codes.max()) + 1
                        if (
                            n_clusters < self.few_cluster_threshold
                            and label not in warned_cluster_terms
                        ):
                            warnings.warn(
                                f"cluster term {label!r} has only {n_clusters} clusters; "
                                "cluster-robust inference may be unreliable, and CR1 does "
                                "not eliminate the few-cluster problem",
                                FewClustersWarning,
                                stacklevel=2,
                            )
                            warned_cluster_terms.add(label)

            model_ranks[horizons] = model_rank
            correction_ranks[horizons] = correction_rank
            degrees_of_freedom_by_group.append(
                {
                    "horizons": tuple(horizons),
                    "n_obs": len(anchor),
                    "absorbed_rank": absorbed.rank,
                    "model_rank": model_rank,
                    "df_resid": len(anchor) - model_rank,
                    "exact": absorbed.exact,
                    "method": absorbed.method,
                    "correction_rank": correction_rank,
                    "correction_rank_exact": correction_effect_rank.exact,
                    "correction_rank_method": correction_effect_rank.method,
                    "nested_effects": tuple(effects[index] for index in nested_effects),
                }
            )

            exact_x = _exact_panel_demean(x, anchor, effects, unit=unit, time=time)
            if exact_x is None:
                residualizer = Residualizer(effect_codes, effect_counts)
                x_tilde, x_diag = residualizer.transform(
                    x,
                    tol=self.demean_tol,
                    max_iter=self.max_iter,
                    acceleration=self.demean_acceleration,
                )
            else:
                residualizer = None
                x_tilde, x_diag = exact_x
            column_scales = np.linalg.norm(x_tilde, axis=0)
            centered_x = x - x.mean(axis=0)
            input_scales = np.max(np.abs(centered_x), axis=0)
            safe_input_scales = np.where(input_scales == 0.0, 1.0, input_scales)
            normalized_within_norms = np.linalg.norm(
                x_tilde / safe_input_scales, axis=0
            )
            within_signal_floor = _within_signal_floor(
                len(anchor),
                effect_codes,
                effect_counts,
                iterative=residualizer is not None and bool(effects),
                tolerance=self.demean_tol,
            )
            weak_columns = (
                (~np.isfinite(column_scales))
                | (~np.isfinite(normalized_within_norms))
                | (column_scales == 0.0)
                | (normalized_within_norms <= within_signal_floor)
            )
            if weak_columns.any():
                names = [feature_names[index] for index in np.flatnonzero(weak_columns)]
                raise ValueError(
                    f"residualized design has zero or negligible within variation at "
                    f"horizon {horizons[0]} for columns {names}"
                )

            design = RegressionDesign(x_tilde, horizon=horizons[0], column_scales=column_scales)
            design.diagnostics["normalized_within_norms"] = normalized_within_norms.copy()
            design.diagnostics["within_signal_floor"] = within_signal_floor
            x_tilde, bread = design.x, design.bread
            linear_algebra_by_group.append(design.diagnostics)

            batch_size = min(planned_batch, len(horizons))
            running_outcome = (
                np.zeros(len(anchor), dtype=np.float64)
                if self.response == "cumulative" else None
            )
            next_step = 0
            for batch_start in range(0, len(horizons), batch_size):
                batch_horizons = horizons[batch_start : batch_start + batch_size]
                if self.response == "level":
                    y = np.column_stack(
                        [outcome_values[future_positions[mask, horizon]] for horizon in batch_horizons]
                    )
                else:
                    assert running_outcome is not None
                    y, next_step = _cumulative_batch(
                        outcome_values, future_positions, positions,
                        batch_horizons, running_outcome, next_step,
                    )
                _require_finite(y, "regression outcomes")
                if residualizer is None:
                    exact_y = _exact_panel_demean(y, anchor, effects, unit=unit, time=time)
                    assert exact_y is not None
                    y_tilde, y_diag = exact_y
                else:
                    y_tilde, y_diag = residualizer.transform(
                        y,
                        tol=self.demean_tol,
                        max_iter=self.max_iter,
                        acceleration=self.demean_acceleration,
                    )
                _require_finite(y_tilde, "transformed outcomes")
                coef, residuals = design.solve(y_tilde)
                _require_finite(coef, "coefficients")
                _require_finite(residuals, "residuals")
                if self.covariance == "homoskedastic":
                    covariance = homoskedastic(x_tilde, residuals, bread, model_rank=model_rank)
                    covariance_diagnostics = tuple(
                        {"kind": "homoskedastic"} for _ in batch_horizons
                    )
                elif self.covariance in {"hc0", "hc1", "hc2", "hc3"}:
                    covariance = hc(x_tilde, residuals, bread, self.covariance, model_rank=model_rank)
                    covariance_diagnostics = tuple(
                        {"kind": self.covariance} for _ in batch_horizons
                    )
                elif self.covariance == "cluster":
                    assert cluster_data is not None
                    (
                        covariance,
                        cluster_diagnostics,
                        covariance_roundoff_scale,
                    ) = cluster_covariance(
                        x_tilde,
                        residuals,
                        *cluster_data,
                        bread,
                        self.cluster_correction,
                        correction_rank=correction_rank,
                        group_adjustment=self.cluster_group_adjustment,
                        return_roundoff_scale=True,
                    )
                    covariance_diagnostics = tuple(
                        {
                            "kind": "cluster",
                            "correction": self.cluster_correction,
                            "correction_rank": correction_rank,
                            "parameter_count": self.cluster_df,
                            "group_adjustment": self.cluster_group_adjustment,
                            "components": cluster_diagnostics,
                        }
                        for _ in batch_horizons
                    )
                else:
                    covariance, covariance_diagnostics = hac(
                        x_tilde,
                        residuals,
                        anchor.column(unit),
                        anchor.column(time),
                        batch_horizons,
                        bread,
                        max_lags=self.hac_lags, kernel=self.hac_kernel, debias=self.hac_debias,
                        driscoll_kraay=self.covariance == "driscoll_kraay",
                        model_rank=model_rank,
                        stream_pairs=memory_plan.stream_pairs if memory_plan is not None else False,
                    )
                covariance = design.covariance_in_scaled_feature_units(covariance)
                covariance = (covariance + covariance.transpose(0, 2, 1)) / 2
                _require_finite(covariance, "normalized covariance")
                diagonal = np.diagonal(covariance, axis1=1, axis2=2)
                if self.covariance == "cluster":
                    covariance_roundoff_scale = (
                        design.covariance_error_in_scaled_feature_units(
                            covariance_roundoff_scale
                        )
                    )
                    diagonal_scale = np.diagonal(
                        covariance_roundoff_scale, axis1=1, axis2=2
                    )
                else:
                    diagonal_scale = np.max(np.abs(covariance), axis=(1, 2))[:, None]
                negative_tolerance = _covariance_roundoff_multiplier(
                    len(anchor), n_features
                ) * diagonal_scale
                materially_negative = diagonal < -negative_tolerance
                if materially_negative.any():
                    local_horizon, feature = np.argwhere(materially_negative)[0]
                    raise ValueError(
                        f"{self.covariance} covariance has a negative variance at horizon "
                        f"{batch_horizons[int(local_horizon)]}, feature {int(feature)}"
                    )
                for column in range(len(batch_horizons)):
                    features = tuple(np.flatnonzero(diagonal[column] < 0.0).tolist())
                    if features:
                        covariance[column, features, features] = 0.0
                    covariance_diagnostics[column]["negative_variance_cleanup"] = (
                        features
                    )
                covariance = design.covariance_from_scaled_feature_units(covariance)
                covariance = (covariance + covariance.transpose(0, 2, 1)) / 2
                diagonal = np.diagonal(covariance, axis1=1, axis2=2)
                stderr = np.sqrt(diagonal)
                _require_finite(covariance, "covariance in original units")
                _require_finite(stderr, "standard errors")
                coefficients[batch_horizons] = coef.T
                standard_errors[batch_horizons] = stderr
                covariances[batch_horizons] = covariance
                for column, horizon in enumerate(batch_horizons):
                    if residuals_by_horizon is not None:
                        residuals_by_horizon[horizon] = residuals[:, column].copy()
                    y_iterations[horizon] = y_diag.iterations[column]
                    y_acceleration_accepted[horizon] = y_diag.acceleration_accepted[column]
                    cache_group_by_horizon[horizon] = group_id
                    covariance_diagnostics_by_horizon[horizon] = covariance_diagnostics[column]
            x_iterations_by_group.append(x_diag.iterations)
            acceleration_accepted_by_group.append(x_diag.acceleration_accepted)
            backends.append(x_diag.backend)
            methods.append(x_diag.method)

        tail_probability = self.alpha / 2
        critical_values = np.full(n_horizons, normal_dist.isf(tail_probability))
        finite_df = np.isfinite(inference_df)
        critical_values[finite_df] = student_t.isf(
            tail_probability, inference_df[finite_df]
        )
        _require_finite(critical_values, "confidence critical values")
        confidence_intervals = np.stack(
            (coefficients - critical_values[:, None] * standard_errors,
             coefficients + critical_values[:, None] * standard_errors), axis=-1
        )
        _require_finite(confidence_intervals, "confidence intervals")
        sample_indices = tuple(frame.sample_ids(mask) for mask in masks.T)
        self.coef_ = coefficients
        self.stderr_ = standard_errors
        self.covariance_ = covariances
        self.conf_int_ = confidence_intervals
        self.horizons_ = np.arange(n_horizons)
        self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        self.shock_names_in_ = np.asarray(shocks, dtype=object)
        self.sample_ = self.sample
        self.response_ = self.response
        self.n_obs_by_horizon_ = masks.sum(axis=0, dtype=int)
        self.model_rank_by_horizon_ = model_ranks
        self.df_resid_by_horizon_ = self.n_obs_by_horizon_ - model_ranks
        self.correction_rank_by_horizon_ = correction_ranks
        self.inference_df_by_horizon_ = inference_df
        self.critical_values_ = critical_values
        self.degrees_of_freedom_diagnostics_ = {
            "policy": self.fe_dof,
            "by_cache_group": tuple(degrees_of_freedom_by_group),
        }
        self.memory_diagnostics_ = (
            memory_plan.diagnostics(planned_batch) if memory_plan is not None
            else {"budget_bytes": None, "scope": "unbudgeted"}
        )
        self.n_obs_ = int(self.n_obs_by_horizon_.max())
        self.n_units_ = frame.nunique(unit)
        self.n_periods_ = frame.nunique(time)
        self.sample_index_by_horizon_ = sample_indices
        self.sample_index_ = sample_indices[0]
        if residuals_by_horizon is None:
            self.residuals_ = None
        else:
            self.residuals_ = (
                np.column_stack(residuals_by_horizon)
                if self.sample == "common"
                else tuple(residuals_by_horizon)
            )
        self.cache_group_by_horizon_ = cache_group_by_horizon
        self.covariance_config_ = {
            "kind": self.covariance,
            "cluster_terms": tuple("#".join(term) for term in cluster_terms),
            "cluster_correction": self.cluster_correction if self.covariance == "cluster" else None,
            "cluster_df": self.cluster_df if self.covariance == "cluster" else None,
            "cluster_group_adjustment": self.cluster_group_adjustment if self.covariance == "cluster" else None,
            "cluster_inference": self.cluster_inference if self.covariance == "cluster" else None,
            "few_cluster_threshold": self.few_cluster_threshold if self.covariance == "cluster" else None,
            "hac_lags": self.hac_lags if self.covariance in {"hac", "driscoll_kraay"} else None,
            "hac_kernel": self.hac_kernel if self.covariance in {"hac", "driscoll_kraay"} else None,
            "hac_debias": self.hac_debias if self.covariance in {"hac", "driscoll_kraay"} else None,
        }
        self.covariance_diagnostics_ = tuple(covariance_diagnostics_by_horizon)
        self.singleton_diagnostics_ = {
            "policy": self.singleton_policy,
            "dropped_by_horizon": singleton_dropped,
            "rounds_by_horizon": singleton_rounds,
        }
        self.linear_algebra_diagnostics_ = {
            "solver": linear_algebra_by_group[0]["solver"] if len({
                item["solver"] for item in linear_algebra_by_group
            }) == 1 else "mixed",
            "by_cache_group": tuple(linear_algebra_by_group),
        }
        self._irfplot = None
        self.demeaning_diagnostics_ = {
            "x_iterations": x_iterations_by_group[0]
            if len(x_iterations_by_group) == 1
            else tuple(x_iterations_by_group),
            "x_iterations_by_cache_group": tuple(x_iterations_by_group),
            "x_acceleration_accepted_by_cache_group": tuple(acceleration_accepted_by_group),
            "y_iterations": y_iterations,
            "y_acceleration_accepted": y_acceleration_accepted,
            "backend": backends[0],
            "method": methods[0],
            "method_by_cache_group": tuple(methods),
            "cache_mode": "shared_design" if len(group_masks) == 1 else "mask_grouped",
            "input_was_sorted": input_was_sorted,
            "input_backend": frame.backend,
            "lag_path": lag_path,
            "lead_path": lead_path,
            "batch_size": batch_size,
            "acceleration": self.demean_acceleration,
        }
        return self

    def plot_irf(self, coefficient: str | None = None, **kwargs: object):
        """Return a publication-oriented Matplotlib impulse-response plot."""
        from .plotting import plot_irf

        return plot_irf(self, coefficient=coefficient, **kwargs)

    @property
    def irfplot(self):
        """Lazily create and cache the default impulse-response plot axes."""
        self._require_fitted()
        if self._irfplot is None:
            self._irfplot = self.plot_irf()
        return self._irfplot

    def _lead_positions(
        self, frame: PanelFrame, *, unit: str, time: str
    ) -> tuple[np.ndarray, str]:
        """Resolve exact ``time + horizon`` outcome rows for every anchor."""
        return frame.lead_positions(unit, time, self.horizons)

    def to_frame(self) -> pd.DataFrame:
        """Return coefficient paths in tidy long form."""
        self._require_fitted()
        records = []
        for h_index, horizon in enumerate(self.horizons_):
            for feature_index, feature in enumerate(self.feature_names_in_):
                records.append(
                    {
                        "horizon": horizon,
                        "sample": self.sample_,
                        "response": self.response_,
                        "n_obs": self.n_obs_by_horizon_[h_index],
                        "coefficient": feature,
                        "estimate": self.coef_[h_index, feature_index],
                        "std_error": self.stderr_[h_index, feature_index],
                        "ci_low": self.conf_int_[h_index, feature_index, 0],
                        "ci_high": self.conf_int_[h_index, feature_index, 1],
                    }
                )
        return pd.DataFrame.from_records(records)

    def summary(self) -> str:
        """Return a compact textual summary."""
        self._require_fitted()
        return (
            f"LocalProjection(horizons=0..{self.horizons}, response={self.response_!r}, "
            f"sample={self.sample_!r}, n_obs={self.n_obs_}, "
            f"covariance={self.covariance}, backend={self.demeaning_diagnostics_['backend']})"
        )

    def _require_fitted(self) -> None:
        if not hasattr(self, "coef_"):
            raise RuntimeError("fit must be called before requesting results")
