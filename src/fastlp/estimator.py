"""Scikit-learn-style balanced-panel local projections estimator."""

from __future__ import annotations

from collections.abc import Sequence
from statistics import NormalDist

import numpy as np
import pandas as pd

from ._covariance import cluster_cr1, hc1
from ._demean import demean, factorize_effects


def _as_columns(value: str | Sequence[str] | None, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    columns = tuple(value)
    if not all(isinstance(column, str) for column in columns):
        raise ValueError(f"{name} must contain only column names")
    return columns


class LocalProjection:
    """Fast local projections for a strictly balanced panel.

    Every horizon uses the common anchor periods ``0..T-horizons``. This is
    intentional: it guarantees that the transformed RHS is identical at every
    horizon and can be cached safely.
    """

    def __init__(
        self,
        *,
        horizons: int,
        covariance: str = "cluster",
        alpha: float = 0.05,
        demean_tol: float = 1e-10,
        max_iter: int = 10_000,
    ) -> None:
        if not isinstance(horizons, int) or horizons < 0:
            raise ValueError("horizons must be a non-negative integer")
        if covariance not in {"hc1", "cluster"}:
            raise ValueError("covariance must be 'hc1' or 'cluster'")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between zero and one")
        if demean_tol <= 0 or max_iter < 1:
            raise ValueError("demean_tol must be positive and max_iter must be at least one")
        self.horizons = horizons
        self.covariance = covariance
        self.alpha = alpha
        self.demean_tol = demean_tol
        self.max_iter = max_iter

    def fit(
        self,
        data: pd.DataFrame,
        *,
        outcome: str,
        shock: str | Sequence[str],
        controls: Sequence[str] = (),
        unit: str,
        time: str,
        fixed_effects: Sequence[str] = (),
        cluster: str | None = None,
    ) -> "LocalProjection":
        """Fit local projections and cache all shared-design calculations."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        shocks = _as_columns(shock, "shock")
        if not shocks:
            raise ValueError("shock must be a column name or a non-empty sequence of column names")
        controls = _as_columns(controls, "controls")
        effects = _as_columns(fixed_effects, "fixed_effects")
        required = (outcome, unit, time, *shocks, *controls, *effects)
        if self.covariance == "cluster":
            if cluster is None:
                raise ValueError("cluster must be supplied when covariance='cluster'")
            required = (*required, cluster)
        missing = sorted(set(required).difference(data.columns))
        if missing:
            raise ValueError(f"data is missing required columns: {missing}")
        if len(set((*shocks, *controls))) != len((*shocks, *controls)):
            raise ValueError("shock and controls must not contain duplicate columns")

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

        frame = frame.sort_values([unit, time], kind="stable")
        panels = [group for _, group in frame.groupby(unit, sort=True, observed=True)]
        if len(panels) < 1:
            raise ValueError("data must contain at least one panel unit")
        time_index = panels[0][time].tolist()
        if any(panel[time].tolist() != time_index for panel in panels[1:]):
            raise ValueError("v0.1 requires every unit to share the same ordered time labels")
        n_periods = len(time_index)
        if n_periods <= self.horizons:
            raise ValueError("each unit must have more periods than horizons")

        n_anchor = n_periods - self.horizons
        anchor = pd.concat([panel.iloc[:n_anchor] for panel in panels], ignore_index=False)
        y = np.column_stack(
            [np.concatenate([panel[outcome].to_numpy()[h : h + n_anchor] for panel in panels]) for h in range(self.horizons + 1)]
        )
        x_parts = [anchor.loc[:, list((*shocks, *controls))].to_numpy(dtype=np.float64)]
        feature_names = list((*shocks, *controls))
        if not effects:
            x_parts.insert(0, np.ones((len(anchor), 1), dtype=np.float64))
            feature_names.insert(0, "Intercept")
        x = np.column_stack(x_parts)

        effect_codes, effect_counts = factorize_effects(anchor, effects)
        x_tilde, x_diag = demean(x, effect_codes, effect_counts, tol=self.demean_tol, max_iter=self.max_iter)
        y_tilde, y_diag = demean(y, effect_codes, effect_counts, tol=self.demean_tol, max_iter=self.max_iter)
        gram = x_tilde.T @ x_tilde
        rank = int(np.linalg.matrix_rank(x_tilde))
        if rank != x_tilde.shape[1]:
            raise ValueError("residualized design is rank deficient")
        if len(anchor) <= rank:
            raise ValueError("residual degrees of freedom must be positive")
        try:
            chol = np.linalg.cholesky(gram)
        except np.linalg.LinAlgError as error:
            raise ValueError("residualized design is not positive definite") from error
        rhs = x_tilde.T @ y_tilde
        coef = np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))
        residuals = y_tilde - x_tilde @ coef
        bread = np.linalg.solve(chol.T, np.linalg.solve(chol, np.eye(rank)))
        if self.covariance == "hc1":
            covariance = hc1(x_tilde, residuals, bread)
        else:
            cluster_codes, _ = pd.factorize(anchor[cluster], sort=True)
            covariance = cluster_cr1(x_tilde, residuals, cluster_codes.astype(np.int64), bread)
        covariance = (covariance + covariance.transpose(0, 2, 1)) / 2
        stderr = np.sqrt(np.maximum(np.diagonal(covariance, axis1=1, axis2=2), 0.0))
        z_value = NormalDist().inv_cdf(1 - self.alpha / 2)
        interval = np.stack((coef.T - z_value * stderr, coef.T + z_value * stderr), axis=-1)

        self.coef_ = coef.T
        self.stderr_ = stderr
        self.covariance_ = covariance
        self.conf_int_ = interval
        self.horizons_ = np.arange(self.horizons + 1)
        self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        self.n_obs_ = len(anchor)
        self.n_units_ = len(panels)
        self.n_periods_ = n_periods
        self.sample_index_ = anchor.index.copy()
        self.residuals_ = residuals
        self.demeaning_diagnostics_ = {
            "x_iterations": x_diag.iterations,
            "y_iterations": y_diag.iterations,
            "backend": x_diag.backend,
        }
        return self

    def to_frame(self) -> pd.DataFrame:
        """Return coefficient paths in tidy long form."""
        self._require_fitted()
        records = []
        for h_index, horizon in enumerate(self.horizons_):
            for feature_index, feature in enumerate(self.feature_names_in_):
                records.append(
                    {
                        "horizon": horizon,
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
            f"LocalProjection(horizons=0..{self.horizons}, n_obs={self.n_obs_}, "
            f"covariance={self.covariance}, backend={self.demeaning_diagnostics_['backend']})"
        )

    def _require_fitted(self) -> None:
        if not hasattr(self, "coef_"):
            raise RuntimeError("fit must be called before requesting results")
