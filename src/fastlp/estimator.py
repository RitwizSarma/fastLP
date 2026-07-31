"""Scikit-learn-style cached panel local projections estimator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from statistics import NormalDist

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
from ._demean import Residualizer, factorize_effects


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
    frame: pd.DataFrame, unit: str, requests: Sequence[tuple[str, int]]
) -> tuple[pd.DataFrame, tuple[str, ...], np.ndarray]:
    """Generate within-unit row lags and identify rows valid as LP anchors."""
    output = frame.copy()
    feature_names: list[str] = []
    seen: set[tuple[str, int]] = set()
    for source, lag in requests:
        key = (source, lag)
        if key in seen:
            continue
        seen.add(key)
        feature = f"{source}_lag{lag}"
        if feature in output.columns:
            raise ValueError(f"generated lag feature {feature!r} conflicts with a supplied model column")
        output[feature] = output.groupby(unit, sort=False, observed=True)[source].shift(lag)
        feature_names.append(feature)
    if not feature_names:
        return output, (), np.ones(len(output), dtype=bool)
    valid = output.loc[:, feature_names].notna().all(axis=1).to_numpy()
    return output, tuple(feature_names), valid


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

    def fit(
        self,
        data: pd.DataFrame,
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
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
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
        missing = sorted(set(required).difference(data.columns))
        if missing:
            raise ValueError(f"data is missing required columns: {missing}")
        if len(set((*shocks, *controls))) != len((*shocks, *controls)):
            raise ValueError("shock and controls must not contain duplicate columns")

        lag_requests = (
            *_lag_requests(outcome_lags, (outcome,), "outcome_lags"),
            *_lag_requests(shock_lags, shocks, "shock_lags"),
            *_lag_requests(control_lags, controls, "control_lags"),
        )

        frame = data.loc[:, list(dict.fromkeys(required))].copy()
        if frame.isna().any().any():
            raise ValueError("v0.1 requires complete data in all model columns")
        numeric_columns = (outcome, *shocks, *controls)
        try:
            for column in dict.fromkeys(numeric_columns):
                frame[column] = pd.to_numeric(frame[column], errors="raise").astype(np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("outcome, shock, and controls must be numeric") from error
        if not np.isfinite(frame.loc[:, numeric_columns].to_numpy(dtype=float)).all():
            raise ValueError("outcome, shock, and controls must be finite")
        if frame.duplicated([unit, time]).any():
            raise ValueError("unit and time must uniquely identify panel observations")

        panel_order = pd.MultiIndex.from_frame(frame[[unit, time]])
        input_was_sorted = panel_order.is_monotonic_increasing
        if not input_was_sorted:
            frame = frame.sort_values([unit, time], kind="stable")
        frame, lag_feature_names, valid_lag_rows = _add_lags(frame, unit, lag_requests)
        if frame.empty:
            raise ValueError("data must contain at least one panel unit")
        future_positions = self._lead_positions(frame, unit=unit, time=time)
        masks = valid_lag_rows[:, None] & (future_positions >= 0)
        if self.sample == "common":
            masks = np.repeat(masks.all(axis=1, keepdims=True), self.horizons + 1, axis=1)

        base_features = (*shocks, *controls, *lag_feature_names)
        base_x = frame.loc[:, list(base_features)].to_numpy(dtype=np.float64)
        if not effects:
            base_x = np.column_stack((np.ones(len(frame), dtype=np.float64), base_x))
        feature_names = ("Intercept", *base_features) if not effects else base_features
        outcome_values = frame[outcome].to_numpy(dtype=np.float64)
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
        residuals_by_horizon: list[np.ndarray | None] = [None] * n_horizons
        covariance_diagnostics_by_horizon: list[dict[str, object] | None] = [None] * n_horizons
        y_iterations = np.empty(n_horizons, dtype=int)
        cache_group_by_horizon = np.empty(n_horizons, dtype=int)
        x_iterations_by_group: list[np.ndarray] = []
        backends: list[str] = []
        methods: list[str] = []

        for group_id, (mask, horizons) in enumerate(zip(group_masks, groups.values(), strict=True)):
            if not mask.any():
                raise ValueError(f"no valid observations at horizon {horizons[0]}")
            positions = np.flatnonzero(mask)
            anchor = frame.iloc[positions]
            x = base_x[mask]
            effect_codes, effect_counts = factorize_effects(anchor, effects)
            residualizer = Residualizer(effect_codes, effect_counts)
            x_tilde, x_diag = residualizer.transform(
                x, tol=self.demean_tol, max_iter=self.max_iter
            )
            gram = (x_tilde.T @ x_tilde)
            gram = (gram + gram.T) / 2
            rank = int(np.linalg.matrix_rank(x_tilde))
            if rank != x_tilde.shape[1]:
                raise ValueError(f"residualized design is rank deficient at horizon {horizons[0]}")
            if len(anchor) <= rank:
                raise ValueError("residual degrees of freedom must be positive")
            try:
                chol = np.linalg.cholesky(gram)
            except np.linalg.LinAlgError as error:
                raise ValueError("residualized design is not positive definite") from error
            bread = np.linalg.solve(chol.T, np.linalg.solve(chol, np.eye(rank)))
            cluster_data = None
            if self.covariance == "cluster":
                cluster_codes, labels = factorize_cluster_terms(anchor, cluster_terms)
                cluster_data = (cluster_codes, labels)

            for batch_start in range(0, len(horizons), _HORIZON_BATCH_SIZE):
                batch_horizons = horizons[batch_start : batch_start + _HORIZON_BATCH_SIZE]
                y = np.column_stack(
                    [outcome_values[future_positions[mask, horizon]] for horizon in batch_horizons]
                )
                y_tilde, y_diag = residualizer.transform(
                    y, tol=self.demean_tol, max_iter=self.max_iter
                )
                coef = np.linalg.solve(chol.T, np.linalg.solve(chol, x_tilde.T @ y_tilde))
                residuals = y_tilde - x_tilde @ coef
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
                        x_tilde, residuals, anchor[unit], anchor[time], batch_horizons, bread,
                        max_lags=self.hac_lags, kernel=self.hac_kernel, debias=self.hac_debias,
                        driscoll_kraay=self.covariance == "driscoll_kraay",
                    )
                covariance = (covariance + covariance.transpose(0, 2, 1)) / 2
                diagonal = np.diagonal(covariance, axis1=1, axis2=2)
                scale = np.maximum(np.max(np.abs(covariance), axis=(1, 2)), 1.0)
                materially_negative = diagonal < -np.finfo(float).eps * scale[:, None] * 100
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
                    residuals_by_horizon[horizon] = residuals[:, column]
                    y_iterations[horizon] = y_diag.iterations[column]
                    cache_group_by_horizon[horizon] = group_id
                    covariance_diagnostics_by_horizon[horizon] = covariance_diagnostics[column]
            x_iterations_by_group.append(x_diag.iterations)
            backends.append(x_diag.backend)
            methods.append(x_diag.method)

        z_value = NormalDist().inv_cdf(1 - self.alpha / 2)
        sample_indices = tuple(frame.index[mask].copy() for mask in masks.T)
        self.coef_ = coefficients
        self.stderr_ = standard_errors
        self.covariance_ = covariances
        self.conf_int_ = np.stack(
            (coefficients - z_value * standard_errors, coefficients + z_value * standard_errors), axis=-1
        )
        self.horizons_ = np.arange(n_horizons)
        self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        self.sample_ = self.sample
        self.n_obs_by_horizon_ = masks.sum(axis=0, dtype=int)
        self.n_obs_ = int(self.n_obs_by_horizon_.max())
        self.n_units_ = frame[unit].nunique()
        self.n_periods_ = frame[time].nunique()
        self.sample_index_by_horizon_ = sample_indices
        self.sample_index_ = sample_indices[0]
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
            "hac_lags": self.hac_lags if self.covariance in {"hac", "driscoll_kraay"} else None,
            "hac_kernel": self.hac_kernel if self.covariance in {"hac", "driscoll_kraay"} else None,
            "hac_debias": self.hac_debias if self.covariance in {"hac", "driscoll_kraay"} else None,
        }
        self.covariance_diagnostics_ = tuple(covariance_diagnostics_by_horizon)
        self.demeaning_diagnostics_ = {
            "x_iterations": x_iterations_by_group[0]
            if len(x_iterations_by_group) == 1
            else tuple(x_iterations_by_group),
            "x_iterations_by_cache_group": tuple(x_iterations_by_group),
            "y_iterations": y_iterations,
            "backend": backends[0],
            "method": methods[0],
            "method_by_cache_group": tuple(methods),
            "cache_mode": "shared_design" if len(group_masks) == 1 else "mask_grouped",
            "input_was_sorted": input_was_sorted,
        }
        return self

    def _lead_positions(self, frame: pd.DataFrame, *, unit: str, time: str) -> np.ndarray:
        """Resolve exact ``time + horizon`` outcome rows for every anchor."""
        try:
            numeric_time = pd.to_numeric(frame[time], errors="raise").to_numpy()
        except (TypeError, ValueError):
            panels = [group for _, group in frame.groupby(unit, sort=True, observed=True)]
            time_index = panels[0][time].tolist()
            if any(panel[time].tolist() != time_index for panel in panels[1:]):
                raise ValueError(
                    "unbalanced panels require numeric time labels for exact time + horizon alignment"
                ) from None
            positions = np.full((len(frame), self.horizons + 1), -1, dtype=int)
            for panel_positions in frame.groupby(unit, sort=True, observed=True).indices.values():
                panel_positions = np.asarray(panel_positions)
                for horizon in range(self.horizons + 1):
                    n_valid = len(panel_positions) - horizon
                    if n_valid > 0:
                        positions[panel_positions[:n_valid], horizon] = panel_positions[horizon:]
            return positions

        panel_index = pd.MultiIndex.from_frame(frame[[unit, time]])
        units = frame[unit].to_numpy()
        positions = np.empty((len(frame), self.horizons + 1), dtype=int)
        for horizon in range(self.horizons + 1):
            positions[:, horizon] = panel_index.get_indexer(
                pd.MultiIndex.from_arrays((units, numeric_time + horizon))
            )
        return positions

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
            f"LocalProjection(horizons=0..{self.horizons}, sample={self.sample_!r}, n_obs={self.n_obs_}, "
            f"covariance={self.covariance}, backend={self.demeaning_diagnostics_['backend']})"
        )

    def _require_fitted(self) -> None:
        if not hasattr(self, "coef_"):
            raise RuntimeError("fit must be called before requesting results")
