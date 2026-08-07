"""Scikit-learn-style cached panel local projections estimator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from statistics import NormalDist
from typing import Any
import warnings

import numpy as np
import pandas as pd

from ._covariance import (
    Kernel,
    cluster as cluster_covariance,
    factorize_cluster_terms,
    hac,
    hc,
    homoskedastic,
)
from ._demean import DemeanDiagnostics, Residualizer, balanced_demean, factorize_effects
from ._frame import PanelFrame, prepare_panel


def _as_columns(value: str | Sequence[str] | None, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    columns = tuple(value)
    if not all(isinstance(column, str) for column in columns):
        raise ValueError(f"{name} must contain only column names")
    return columns


ClusterTerm = str | tuple[str, ...]
_HORIZON_BATCH_SIZE = 32


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


def _exact_panel_demean(
    values: np.ndarray,
    frame: PanelFrame,
    effects: tuple[str, ...],
    *,
    unit: str,
    time: str,
) -> tuple[np.ndarray, DemeanDiagnostics] | None:
    """Apply exact one-/two-way within transforms on a dense panel."""
    if not effects or len(set(effects)) != len(effects) or not set(effects).issubset({unit, time}):
        return None
    shape = _balanced_panel_shape(frame, unit, time)
    if len(effects) == 1 and shape is None:
        codes, n_groups = frame.factorize((effects[0],))
        counts = np.bincount(codes, minlength=n_groups).astype(np.float64)
        transformed = np.asarray(values, dtype=np.float64).copy()
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
        return tuple(range(1, int(value) + 1))
    if isinstance(value, str):
        raise ValueError(f"{name} must be an integer or a sequence of positive integers")
    try:
        lags = tuple(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer or a sequence of positive integers") from error
    if any(not isinstance(lag, Integral) or isinstance(lag, bool) or lag < 1 for lag in lags):
        raise ValueError(f"{name} must contain only positive integers")
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
    frame: PanelFrame, unit: str, requests: Sequence[tuple[str, int]]
) -> tuple[PanelFrame, tuple[str, ...], np.ndarray]:
    """Generate within-unit row lags and identify rows valid as LP anchors."""
    return frame.add_lags(unit, requests)


class LocalProjection:
    """Fast local projections with an explicit horizon-sample policy.

    ``sample="common"`` uses one anchor sample at every horizon, so the
    residualized RHS and its factorization are shared. ``sample="per_horizon"``
    retains every valid anchor at each horizon and shares a cache only among
    horizons with an identical retained-row mask.
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
    ) -> None:
        if not isinstance(horizons, int) or horizons < 0:
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
        if hac_lags is not None and (not isinstance(hac_lags, int) or hac_lags < 0):
            raise ValueError("hac_lags must be a non-negative integer or None")
        if hac_kernel not in {"bartlett", "parzen", "quadratic_spectral"}:
            raise ValueError("hac_kernel must be 'bartlett', 'parzen', or 'quadratic_spectral'")
        if not isinstance(hac_debias, bool):
            raise ValueError("hac_debias must be a boolean")
        if cluster_correction not in {"cr0", "cr1"}:
            raise ValueError("cluster_correction must be 'cr0' or 'cr1'")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between zero and one")
        if demean_tol <= 0 or max_iter < 1:
            raise ValueError("demean_tol must be positive and max_iter must be at least one")
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
        self.horizons = horizons
        self.covariance = covariance
        self.hac_lags = hac_lags
        self.hac_kernel = hac_kernel
        self.hac_debias = hac_debias
        self.cluster_correction = cluster_correction
        self.alpha = alpha
        self.demean_tol = demean_tol
        self.max_iter = max_iter
        self.sample = sample
        self.response = response
        self.retain_residuals = retain_residuals
        self.memory_budget = _memory_bytes(memory_budget)
        self.demean_acceleration = demean_acceleration
        self.few_cluster_threshold = (
            None if few_cluster_threshold is None else int(few_cluster_threshold)
        )
        self.singleton_policy = singleton_policy

    def fit(
        self,
        data: Any,
        *,
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
        """Fit local projections according to the configured sample policy.

        A scalar lag setting includes all lags from one through that value;
        sequences select an arbitrary positive lag grid.  For shocks and
        controls, a mapping can assign a separate setting to each column.
        Lags are generated automatically from preceding, sorted observations
        within each unit. Rows without every requested lag are not used as LP
        anchors, but remain available as future outcomes.
        """
        shocks = _as_columns(shock, "shock")
        if not shocks:
            raise ValueError("shock must be a column name or a non-empty sequence of column names")
        controls = _as_columns(controls, "controls")
        effects = _as_columns(fixed_effects, "fixed_effects")
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
        frame, input_was_sorted = prepare_panel(data, required, numeric_columns)
        frame, lag_feature_names, valid_lag_rows = _add_lags(frame, unit, lag_requests)
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
        outcome_prefix = None
        dense_shape = _balanced_panel_shape(frame, unit, time)
        if self.response == "cumulative" and lead_path == "balanced_arithmetic":
            assert dense_shape is not None
            outcome_grid = outcome_values.reshape(dense_shape)
            outcome_prefix = np.column_stack(
                (np.zeros(dense_shape[0], dtype=np.float64), np.cumsum(outcome_grid, axis=1))
            )
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
        backends: list[str] = []
        methods: list[str] = []
        warned_cluster_terms: set[str] = set()

        for group_id, (mask, horizons) in enumerate(zip(group_masks, groups.values(), strict=True)):
            if not mask.any():
                raise ValueError(f"no valid observations at horizon {horizons[0]}")
            positions = np.flatnonzero(mask)
            anchor = frame.take(positions)
            x = base_x[mask]
            exact_x = _exact_panel_demean(x, anchor, effects, unit=unit, time=time)
            if exact_x is None:
                effect_codes, effect_counts = factorize_effects(anchor, effects)
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
            raw_column_scales = np.linalg.norm(x, axis=0)
            scale_floor = np.sqrt(np.finfo(np.float64).eps) * raw_column_scales
            weak_columns = (~np.isfinite(column_scales)) | (column_scales <= scale_floor)
            if weak_columns.any():
                names = [feature_names[index] for index in np.flatnonzero(weak_columns)]
                raise ValueError(
                    f"residualized design has zero or negligible within variation at "
                    f"horizon {horizons[0]} for columns {names}"
                )

            # Equilibrate the design before forming cross-products.  All
            # reported coefficients and covariance matrices are transformed
            # back to the original feature units below.
            x_tilde /= column_scales
            gram = x_tilde.T @ x_tilde
            gram = (gram + gram.T) / 2
            try:
                eigenvalues = np.linalg.eigvalsh(gram)
            except np.linalg.LinAlgError as error:
                raise ValueError(
                    f"could not assess the scaled design at horizon {horizons[0]}; "
                    "the small Gram-matrix eigendecomposition did not converge"
                ) from error
            largest_eigenvalue = float(eigenvalues[-1])
            rank_tolerance = (
                np.finfo(np.float64).eps * gram.shape[0] * largest_eigenvalue
            )
            rank = int(np.count_nonzero(eigenvalues > rank_tolerance))
            smallest_eigenvalue = float(eigenvalues[0])
            gram_condition = (
                largest_eigenvalue / smallest_eigenvalue
                if smallest_eigenvalue > 0.0
                else np.inf
            )
            design_condition = float(np.sqrt(gram_condition))
            linear_algebra_by_group.append(
                {
                    "horizon": int(horizons[0]),
                    "rank": rank,
                    "n_features": int(x_tilde.shape[1]),
                    "rank_tolerance": float(rank_tolerance),
                    "scaled_gram_condition_number": float(gram_condition),
                    "scaled_design_condition_number": design_condition,
                    "column_scales": column_scales.copy(),
                }
            )
            if rank != x_tilde.shape[1]:
                raise ValueError(
                    f"residualized design is numerically rank deficient at horizon "
                    f"{horizons[0]} (rank {rank} of {x_tilde.shape[1]}; scaled-design "
                    f"condition number {design_condition:.3g})"
                )
            if len(anchor) <= rank:
                raise ValueError("residual degrees of freedom must be positive")
            try:
                chol = np.linalg.cholesky(gram)
            except np.linalg.LinAlgError as error:
                raise ValueError(
                    f"Cholesky factorization of the scaled Gram matrix failed at "
                    f"horizon {horizons[0]} despite an estimated rank of {rank}; "
                    f"the residualized design may be ill-conditioned "
                    f"(estimated scaled-design condition number "
                    f"{design_condition:.3g})"
                ) from error
            bread = np.linalg.solve(chol.T, np.linalg.solve(chol, np.eye(rank)))
            cluster_data = None
            if self.covariance == "cluster":
                if cluster_terms == ((unit,),):
                    unit_sizes = anchor.group_sizes(unit)
                    cluster_codes = [np.repeat(np.arange(len(unit_sizes), dtype=np.int64), unit_sizes)]
                    labels = [unit]
                else:
                    cluster_codes, labels = factorize_cluster_terms(anchor, cluster_terms)
                cluster_data = (cluster_codes, labels)
                if self.few_cluster_threshold is not None:
                    for codes, label in zip(cluster_codes, labels, strict=True):
                        n_clusters = int(codes.max()) + 1
                        if n_clusters < self.few_cluster_threshold and label not in warned_cluster_terms:
                            warnings.warn(
                                f"cluster term {label!r} has only {n_clusters} clusters; "
                                "cluster-robust inference may be unreliable, and CR1 does "
                                "not eliminate the few-cluster problem",
                                FewClustersWarning,
                                stacklevel=2,
                            )
                            warned_cluster_terms.add(label)

            # Budget for raw outcomes, transformed outcomes, and residuals.
            bytes_per_horizon = max(1, len(anchor) * np.dtype(np.float64).itemsize * 3)
            budget_batch = (
                len(horizons) if self.memory_budget is None
                else max(1, self.memory_budget // bytes_per_horizon)
            )
            batch_size = max(1, min(_HORIZON_BATCH_SIZE, int(budget_batch)))
            for batch_start in range(0, len(horizons), batch_size):
                batch_horizons = horizons[batch_start : batch_start + batch_size]
                if self.response == "level":
                    y = np.column_stack(
                        [outcome_values[future_positions[mask, horizon]] for horizon in batch_horizons]
                    )
                else:
                    if outcome_prefix is not None and dense_shape is not None:
                        anchor_positions = positions
                        anchor_units = anchor_positions // dense_shape[1]
                        anchor_times = anchor_positions % dense_shape[1]
                        y = np.column_stack(
                            [
                                outcome_prefix[anchor_units, anchor_times + horizon + 1]
                                - outcome_prefix[anchor_units, anchor_times]
                                for horizon in batch_horizons
                            ]
                        )
                    else:
                        y = np.column_stack(
                            [
                                np.sum(
                                    np.column_stack(
                                        [outcome_values[future_positions[mask, step]] for step in range(horizon + 1)]
                                    ),
                                    axis=1,
                                )
                                for horizon in batch_horizons
                            ]
                        )
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
                scaled_coef = np.linalg.solve(
                    chol.T, np.linalg.solve(chol, x_tilde.T @ y_tilde)
                )
                residuals = y_tilde - x_tilde @ scaled_coef
                if self.covariance == "homoskedastic":
                    covariance = homoskedastic(x_tilde, residuals, bread)
                    covariance_diagnostics = tuple(
                        {"kind": "homoskedastic"} for _ in batch_horizons
                    )
                elif self.covariance in {"hc0", "hc1", "hc2", "hc3"}:
                    covariance = hc(x_tilde, residuals, bread, self.covariance)
                    covariance_diagnostics = tuple(
                        {"kind": self.covariance} for _ in batch_horizons
                    )
                elif self.covariance == "cluster":
                    assert cluster_data is not None
                    covariance, cluster_diagnostics = cluster_covariance(
                        x_tilde, residuals, *cluster_data, bread, self.cluster_correction
                    )
                    covariance_diagnostics = tuple(
                        {
                            "kind": "cluster",
                            "correction": self.cluster_correction,
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
                    )
                inverse_scales = 1.0 / column_scales
                coef = scaled_coef * inverse_scales[:, None]
                covariance *= inverse_scales[None, :, None]
                covariance *= inverse_scales[None, None, :]
                covariance = (covariance + covariance.transpose(0, 2, 1)) / 2
                diagonal = np.diagonal(covariance, axis1=1, axis2=2)
                covariance_scale = np.maximum(
                    np.max(np.abs(covariance), axis=(1, 2)), 1.0
                )
                materially_negative = (
                    diagonal
                    < -np.finfo(float).eps * covariance_scale[:, None] * 100
                )
                if materially_negative.any():
                    local_horizon, feature = np.argwhere(materially_negative)[0]
                    raise ValueError(
                        f"{self.covariance} covariance has a negative variance at horizon "
                        f"{batch_horizons[int(local_horizon)]}, feature {int(feature)}"
                    )
                stderr = np.sqrt(np.maximum(diagonal, 0.0))
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

        z_value = NormalDist().inv_cdf(1 - self.alpha / 2)
        sample_indices = tuple(frame.sample_ids(mask) for mask in masks.T)
        self.coef_ = coefficients
        self.stderr_ = standard_errors
        self.covariance_ = covariances
        self.conf_int_ = np.stack(
            (coefficients - z_value * standard_errors, coefficients + z_value * standard_errors), axis=-1
        )
        self.horizons_ = np.arange(n_horizons)
        self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        self.shock_names_in_ = np.asarray(shocks, dtype=object)
        self.sample_ = self.sample
        self.response_ = self.response
        self.n_obs_by_horizon_ = masks.sum(axis=0, dtype=int)
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
            "solver": "scaled_cholesky",
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
