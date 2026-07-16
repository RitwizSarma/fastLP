"""Fixed-effect absorption kernels.

The optional Rust extension mirrors the NumPy implementation below. Keeping the
reference implementation here makes source installs usable and provides a
direct correctness oracle for the compiled kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # Built by maturin when the optional native backend is available.
    from ._fastlp_rust import demean_map as _rust_demean_map
except ImportError:  # pragma: no cover - exercised only in native builds.
    _rust_demean_map = None


@dataclass(frozen=True)
class DemeanDiagnostics:
    iterations: np.ndarray
    backend: str


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
) -> tuple[np.ndarray, DemeanDiagnostics]:
    """Residualize each column of ``values`` against all categorical effects."""
    values = np.ascontiguousarray(values, dtype=np.float64)
    if codes.shape[1] == 0:
        return values.copy(), DemeanDiagnostics(np.zeros(values.shape[1], dtype=int), "none")

    if _rust_demean_map is not None:
        transformed, iterations = _rust_demean_map(values, codes, group_counts.tolist(), tol, max_iter)
        iterations = np.asarray(iterations, dtype=int)
        if (iterations < 0).any():
            raise RuntimeError("fixed-effect demeaning did not converge")
        return np.asarray(transformed), DemeanDiagnostics(iterations, "rust")

    transformed = values.copy()
    iterations = np.zeros(values.shape[1], dtype=int)
    for column in range(transformed.shape[1]):
        vector = transformed[:, column]
        for iteration in range(1, max_iter + 1):
            previous = vector.copy()
            for dimension, n_groups in enumerate(group_counts):
                group_codes = codes[:, dimension]
                sums = np.bincount(group_codes, weights=vector, minlength=int(n_groups))
                sizes = np.bincount(group_codes, minlength=int(n_groups))
                vector -= sums[group_codes] / sizes[group_codes]
            if np.max(np.abs(vector - previous)) < tol:
                iterations[column] = iteration
                break
        else:
            raise RuntimeError("fixed-effect demeaning did not converge")
    return transformed, DemeanDiagnostics(iterations, "numpy")
