"""Cached regression factorization with a tested Cholesky/QR switch."""

import numpy as np
from scipy.linalg import solve_triangular

# Scaled DESIGN condition, not Gram condition. See docs/review/solver_stress.py.
# Leave headroom below the 1e-8 coefficient/covariance error target in that grid.
CHOLESKY_CONDITION_LIMIT = 1000.0


class RegressionDesign:
    """Factor a demeaned design once and reuse it across outcome batches.

    On the QR path, inference uses Q and identity bread. This avoids sandwiching
    nearly cancelling score products with an ill-conditioned Gram inverse.
    coefficient_map converts either working basis to original feature units.
    """

    def __init__(
        self, x: np.ndarray, *, horizon: int, column_scales: np.ndarray | None = None
    ) -> None:
        n, k = x.shape
        self.scales = (
            np.linalg.norm(x, axis=0) if column_scales is None else column_scales
        )
        scaled = x / self.scales
        gram = scaled.T @ scaled
        gram = (gram + gram.T) / 2
        fallback_reason = None
        try:
            eigenvalues = np.linalg.eigvalsh(gram)
            condition = (
                float(np.sqrt(eigenvalues[-1] / eigenvalues[0]))
                if eigenvalues[0] > 0
                else np.inf
            )
        except np.linalg.LinAlgError:
            condition = np.inf
            fallback_reason = "gram_eigendecomposition_failed"
        self.chol = None
        self.r = None
        if condition <= CHOLESKY_CONDITION_LIMIT:
            try:
                self.chol = np.linalg.cholesky(gram)
            except np.linalg.LinAlgError:
                fallback_reason = "cholesky_failed"

        if self.chol is not None:
            self.x = scaled
            inverse_chol = solve_triangular(self.chol, np.eye(k), lower=True)
            self.bread = inverse_chol.T @ inverse_chol
            self.scaled_feature_map = np.eye(k)
            self.coefficient_map = np.diag(1.0 / self.scales)
            method = "scaled_cholesky"
            rank = k
            rank_tolerance = np.finfo(float).eps * max(n, k) * np.sqrt(eigenvalues[-1])
        else:
            try:
                q, r = np.linalg.qr(scaled, mode="reduced")
                singular_values = np.linalg.svd(r, compute_uv=False)
            except np.linalg.LinAlgError as error:
                raise ValueError(
                    f"QR rank assessment failed at horizon {horizon}"
                ) from error
            rank_tolerance = np.finfo(float).eps * max(n, k) * singular_values[0]
            rank = int(np.count_nonzero(singular_values > rank_tolerance))
            condition = (
                float(singular_values[0] / singular_values[-1])
                if singular_values[-1] > 0 and n >= k
                else np.inf
            )
            if rank != k:
                raise ValueError(
                    f"residualized design is numerically rank deficient at horizon "
                    f"{horizon} (rank {rank} of {k}; scaled-design "
                    f"condition number {condition:.3g})"
                )
            self.x = q
            self.r = r
            self.bread = np.eye(k)
            self.scaled_feature_map = solve_triangular(r, np.eye(k))
            self.coefficient_map = self.scaled_feature_map / self.scales[:, None]
            method = "scaled_qr"
            fallback_reason = fallback_reason or "condition_limit"
        self.diagnostics = {
            "horizon": int(horizon),
            "rank": rank,
            "n_features": k,
            "rank_tolerance": float(rank_tolerance),
            "rank_tolerance_scale": "design_singular_values",
            "scaled_design_condition_number": condition,
            "scaled_gram_condition_number": condition**2,
            "column_scales": self.scales.copy(),
            "solver": method,
            "cholesky_condition_limit": CHOLESKY_CONDITION_LIMIT,
            "fallback_reason": fallback_reason,
        }

    def solve(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coordinates = self.x.T @ y
        if self.chol is not None:
            coordinates = solve_triangular(
                self.chol.T,
                solve_triangular(self.chol, coordinates, lower=True),
            )
        residuals = y - self.x @ coordinates
        coefficients = (
            solve_triangular(self.r, coordinates) / self.scales[:, None]
            if self.r is not None
            else self.coefficient_map @ coordinates
        )
        return coefficients, residuals

    def covariance_in_original_units(self, covariance: np.ndarray) -> np.ndarray:
        return self.coefficient_map @ covariance @ self.coefficient_map.T

    def covariance_in_scaled_feature_units(self, covariance: np.ndarray) -> np.ndarray:
        """Map working-basis covariance to the normalized original features."""
        return self.scaled_feature_map @ covariance @ self.scaled_feature_map.T

    def covariance_error_in_scaled_feature_units(
        self, error_scale: np.ndarray
    ) -> np.ndarray:
        """Conservatively map componentwise absolute covariance error scales."""
        absolute_map = np.abs(self.scaled_feature_map)
        return absolute_map @ error_scale @ absolute_map.T

    def covariance_from_scaled_feature_units(
        self, covariance: np.ndarray
    ) -> np.ndarray:
        """Restore normalized-feature covariance to user feature units."""
        return covariance / self.scales[:, None] / self.scales[None, :]
