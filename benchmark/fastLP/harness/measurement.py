"""Low-overhead, dependency-free resource and stage instrumentation."""

from __future__ import annotations

import contextlib
import os
import platform
import resource
import sys
import time
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any


def peak_rss_bytes() -> int | None:
    """Return process peak RSS where the host exposes it.

    Linux reports KiB from ``getrusage``; macOS reports bytes.  It is a
    process-lifetime high-water mark, so a fresh subprocess is recommended
    when comparing cases with very different sizes.
    """
    try:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (AttributeError, OSError):
        return None
    return int(value if sys.platform == "darwin" else value * 1024)


def runtime_environment() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "rayon_threads": os.environ.get("RAYON_NUM_THREADS"),
        "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "mkl_threads": os.environ.get("MKL_NUM_THREADS"),
        "omp_threads": os.environ.get("OMP_NUM_THREADS"),
    }


@dataclass
class Measurement:
    wall_seconds: float
    cpu_seconds: float
    cpu_utilization_percent: float | None
    peak_rss_bytes: int | None
    stages_seconds: dict[str, float]


class _StageHooks:
    """Time fit's stable computational boundaries without changing fastLP.

    The hooks are deliberately process-local and restore all patched functions
    before returning.  ``other`` is the remaining fit time (validation, data
    preparation, cross-products, residual construction, and result assembly).
    """

    def __init__(self) -> None:
        self.times: defaultdict[str, float] = defaultdict(float)
        self._saved: list[tuple[object, str, object]] = []

    def _wrap(self, owner: object, attribute: str, stage: str) -> None:
        original = getattr(owner, attribute)

        def timed(*args: object, **kwargs: object) -> object:
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.times[stage] += time.perf_counter() - start

        self._saved.append((owner, attribute, original))
        setattr(owner, attribute, timed)

    def __enter__(self) -> "_StageHooks":
        # Imported here so this module remains usable for any future benchmark.
        import fastlp.estimator as estimator

        self._wrap(estimator, "factorize_effects", "fe_encoding")
        self._wrap(estimator, "demean", "demeaning")
        self._wrap(estimator, "homoskedastic", "covariance")
        self._wrap(estimator, "hc", "covariance")
        self._wrap(estimator, "cluster_covariance", "covariance")
        self._wrap(estimator, "hac", "covariance")
        self._wrap(estimator.np.linalg, "matrix_rank", "rank_check")
        self._wrap(estimator.np.linalg, "cholesky", "factorization")
        self._wrap(estimator.np.linalg, "solve", "linear_solves")
        return self

    def __exit__(self, *unused: object) -> None:
        for owner, attribute, original in reversed(self._saved):
            setattr(owner, attribute, original)


def measure(callable_: Any, *, stage_timing: bool) -> Measurement:
    before_rss = peak_rss_bytes()
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    if stage_timing:
        with _StageHooks() as hooks:
            callable_()
        stages = dict(hooks.times)
    else:
        callable_()
        stages = {}
    wall = time.perf_counter() - wall_start
    cpu = time.process_time() - cpu_start
    if stage_timing:
        stages["other"] = max(0.0, wall - sum(stages.values()))
    peak = peak_rss_bytes()
    # ru_maxrss cannot decrease, so retain the high-water mark rather than a
    # misleading delta.  before_rss is intentionally read for documentation.
    del before_rss
    utilization = (100 * cpu / wall) if wall else None
    return Measurement(wall, cpu, utilization, peak, stages)


@contextlib.contextmanager
def temporary_thread_policy(threads: int | None) -> Iterator[None]:
    """Apply one coordinated BLAS/Rayon policy for the duration of a run."""
    if threads is None:
        yield
        return
    names = ("RAYON_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = str(threads)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
