"""Fixed-effect absorption kernels.

The optional Rust extension mirrors the NumPy implementation below. Keeping the
reference implementation here makes source installs usable and provides a
direct correctness oracle for the compiled kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # Built by maturin when the optional native backend is available.
    from ._fastlp_rust import HdfePlan as _RustHdfePlan
except ImportError:  # pragma: no cover - exercised only in native builds.
    _RustHdfePlan = None

try:
    from ._fastlp_rust import demean_balanced as _rust_demean_balanced
except ImportError:  # An older locally built extension can still supply HdfePlan.
    _rust_demean_balanced = None


def balanced_demean(
    values: np.ndarray,
    n_units: int,
    n_periods: int,
    *,
    unit_effect: bool,
    time_effect: bool,
) -> tuple[np.ndarray, str]:
    """Exact dense-panel within transformation, using Rust when available."""
    values = np.ascontiguousarray(values, dtype=np.float64)
    if _rust_demean_balanced is not None:
        return (
            np.asarray(
                _rust_demean_balanced(values, n_units, n_periods, unit_effect, time_effect)
            ),
            "rust",
        )
    cube = values.reshape(n_units, n_periods, -1)
    transformed = cube.copy()
    if unit_effect and time_effect:
        transformed -= cube.mean(axis=1, keepdims=True)
        transformed -= cube.mean(axis=0, keepdims=True)
        transformed += cube.mean(axis=(0, 1), keepdims=True)
    elif unit_effect:
        transformed -= cube.mean(axis=1, keepdims=True)
    else:
        transformed -= cube.mean(axis=0, keepdims=True)
    return transformed.reshape(values.shape), "numpy"


@dataclass(frozen=True)
class DemeanDiagnostics:
    iterations: np.ndarray
    backend: str
    method: str
    acceleration_accepted: np.ndarray


class Residualizer:
    """Prepared fixed-effect projection with reusable encoded topology."""

    def __init__(self, codes: np.ndarray, group_counts: np.ndarray) -> None:
        self.codes = np.ascontiguousarray(codes, dtype=np.int64)
        self.group_counts = np.asarray(group_counts, dtype=np.int64)
        self._native = (
            _RustHdfePlan(self.codes, self.group_counts.tolist())
            if _RustHdfePlan is not None and self.codes.shape[1]
            else None
        )

    def transform(
        self,
        values: np.ndarray,
        *,
        tol: float,
        max_iter: int,
        acceleration: str = "irons_tuck",
    ) -> tuple[np.ndarray, DemeanDiagnostics]:
        if acceleration not in {"none", "aitken", "irons_tuck"}:
            raise ValueError("acceleration must be 'none', 'aitken', or 'irons_tuck'")
        values = np.ascontiguousarray(values, dtype=np.float64)
        if self.codes.shape[1] == 0:
            return values.copy(), DemeanDiagnostics(
                np.zeros(values.shape[1], dtype=int),
                "none",
                "none",
                np.zeros(values.shape[1], dtype=int),
            )
        if self._native is not None:
            acceleration_code = {"none": 0, "aitken": 1, "irons_tuck": 2}[acceleration]
            transformed, iterations, accepted = self._native.transform(
                values, tol, max_iter, acceleration_code
            )
            iterations = np.asarray(iterations, dtype=int)
            if (iterations < 0).any():
                raise RuntimeError("fixed-effect demeaning did not converge")
            return np.asarray(transformed), DemeanDiagnostics(
                iterations,
                "rust",
                f"symmetric_kaczmarz_{acceleration}"
                if acceleration != "none"
                else "symmetric_kaczmarz",
                np.asarray(accepted, dtype=int),
            )

        transformed = values.copy()
        iterations = np.zeros(values.shape[1], dtype=int)
        accepted = np.zeros(values.shape[1], dtype=int)
        for column in range(transformed.shape[1]):
            vector = transformed[:, column]
            older = vector.copy()
            for iteration in range(1, max_iter + 1):
                previous = vector.copy()
                for dimension, n_groups in enumerate(self.group_counts):
                    group_codes = self.codes[:, dimension]
                    sums = np.bincount(group_codes, weights=vector, minlength=int(n_groups))
                    sizes = np.bincount(group_codes, minlength=int(n_groups))
                    vector -= sums[group_codes] / sizes[group_codes]
                if acceleration != "none" and iteration >= 2 and iteration % 3 == 0:
                    first_delta = previous - older
                    next_delta = vector - previous
                    second_delta = next_delta - first_delta
                    denominator = float(second_delta @ second_delta)
                    if denominator > np.finfo(float).eps:
                        if acceleration == "irons_tuck":
                            factor = float(next_delta @ second_delta) / denominator
                            candidate = vector - factor * next_delta
                        else:
                            factor = float(first_delta @ second_delta) / denominator
                            candidate = older - factor * first_delta
                        if np.isfinite(candidate).all() and self._projection_error(candidate) < self._projection_error(vector):
                            vector[:] = candidate
                            accepted[column] += 1
                if np.max(np.abs(vector - previous)) < tol:
                    iterations[column] = iteration
                    break
                older[:] = previous
            else:
                raise RuntimeError("fixed-effect demeaning did not converge")
        method = (
            f"cyclic_kaczmarz_{acceleration}"
            if acceleration != "none"
            else "cyclic_kaczmarz"
        )
        return transformed, DemeanDiagnostics(iterations, "numpy", method, accepted)

    def _projection_error(self, vector: np.ndarray) -> float:
        """Largest remaining absolute group mean over every FE dimension."""
        error = 0.0
        for dimension, n_groups in enumerate(self.group_counts):
            group_codes = self.codes[:, dimension]
            sums = np.bincount(group_codes, weights=vector, minlength=int(n_groups))
            sizes = np.bincount(group_codes, minlength=int(n_groups))
            error = max(error, float(np.max(np.abs(sums / sizes))))
        return error


def factorize_effects(frame, columns: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Return dense categorical codes and group counts for each FE column."""
    if not columns:
        return np.empty((len(frame), 0), dtype=np.int64), np.empty(0, dtype=np.int64)

    codes = np.empty((len(frame), len(columns)), dtype=np.int64)
    counts = np.empty(len(columns), dtype=np.int64)
    for index, column in enumerate(columns):
        encoded, uniques = __import__("pandas").factorize(frame[column], sort=True)
        if (encoded < 0).any():
            raise ValueError(f"fixed-effect column {column!r} contains missing values")
        codes[:, index] = encoded
        counts[index] = len(uniques)
    return codes, counts


def demean(
    values: np.ndarray,
    codes: np.ndarray,
    group_counts: np.ndarray,
    *,
    tol: float,
    max_iter: int,
    acceleration: str = "irons_tuck",
) -> tuple[np.ndarray, DemeanDiagnostics]:
    """Residualize each column of ``values`` against all categorical effects."""
    return Residualizer(codes, group_counts).transform(
        values, tol=tol, max_iter=max_iter, acceleration=acceleration
    )
