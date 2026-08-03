"""Low-overhead, dependency-free resource and stage instrumentation."""

from __future__ import annotations

import contextlib
import os
import platform
import resource
import sys
import time
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
class StageMeasurement:
    """Elapsed and process-resource use attributed to one fit boundary."""

    wall_seconds: float
    user_seconds: float
    system_seconds: float

    @property
    def cpu_seconds(self) -> float:
        return self.user_seconds + self.system_seconds

    @property
    def cpu_utilization_percent(self) -> float | None:
        return 100 * self.cpu_seconds / self.wall_seconds if self.wall_seconds else None


@dataclass
class Measurement:
    wall_seconds: float
    user_seconds: float
    system_seconds: float
    cpu_seconds: float
    cpu_utilization_percent: float | None
    peak_rss_bytes: int | None
    stages: dict[str, StageMeasurement]

    @property
    def stages_seconds(self) -> dict[str, float]:
        """Compatibility view used by older consumers of the harness output."""
        return {stage: timing.wall_seconds for stage, timing in self.stages.items()}


def _process_times() -> tuple[float, float]:
    """Return user and kernel CPU seconds for this process only."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime, usage.ru_stime


class _StageHooks:
    """Time fit's stable computational boundaries without changing fastLP.

    The hooks are deliberately process-local and restore all patched functions
    before returning.  ``other`` is the remaining fit time (validation, data
    preparation, cross-products, residual construction, and result assembly).
    """

    def __init__(self) -> None:
        self.times: dict[str, StageMeasurement] = {}
        self._saved: list[tuple[object, str, object]] = []

    def _wrap(self, owner: object, attribute: str, stage: str) -> None:
        original = getattr(owner, attribute)

        def timed(*args: object, **kwargs: object) -> object:
            wall_start = time.perf_counter()
            user_start, system_start = _process_times()
            try:
                return original(*args, **kwargs)
            finally:
                user_end, system_end = _process_times()
                elapsed = self.times.get(stage, StageMeasurement(0.0, 0.0, 0.0))
                self.times[stage] = StageMeasurement(
                    wall_seconds=elapsed.wall_seconds + time.perf_counter() - wall_start,
                    user_seconds=elapsed.user_seconds + user_end - user_start,
                    system_seconds=elapsed.system_seconds + system_end - system_start,
                )

        self._saved.append((owner, attribute, original))
        setattr(owner, attribute, timed)

    def __enter__(self) -> "_StageHooks":
        # Imported here so this module remains usable for any future benchmark.
        import fastlp.estimator as estimator

        self._wrap(estimator, "_add_lags", "lag_construction")
        self._wrap(estimator.LocalProjection, "_lead_positions", "lead_construction")
        self._wrap(estimator, "_prune_singletons", "singleton_pruning")
        self._wrap(estimator, "factorize_effects", "fe_encoding")
        # Balanced panels use this exact transform, whereas unbalanced panels
        # use Residualizer below.  Both must count as demeaning.
        self._wrap(estimator, "balanced_demean", "demeaning")
        self._wrap(estimator.Residualizer, "transform", "demeaning")
        self._wrap(estimator, "homoskedastic", "covariance")
        self._wrap(estimator, "hc", "covariance")
        self._wrap(estimator, "cluster_covariance", "covariance")
        self._wrap(estimator, "hac", "covariance")
        self._wrap(estimator.np.linalg, "eigvalsh", "rank_check")
        self._wrap(estimator.np.linalg, "cholesky", "factorization")
        self._wrap(estimator.np.linalg, "solve", "linear_solves")
        return self

    def __exit__(self, *unused: object) -> None:
        for owner, attribute, original in reversed(self._saved):
            setattr(owner, attribute, original)


def measure(callable_: Any, *, stage_timing: bool) -> Measurement:
    before_rss = peak_rss_bytes()
    user_start, system_start = _process_times()
    wall_start = time.perf_counter()
    if stage_timing:
        with _StageHooks() as hooks:
            callable_()
        stages = dict(hooks.times)
    else:
        callable_()
        stages = {}
    wall = time.perf_counter() - wall_start
    user_end, system_end = _process_times()
    user = user_end - user_start
    system = system_end - system_start
    cpu = user + system
    if stage_timing:
        stages["other"] = StageMeasurement(
            wall_seconds=max(0.0, wall - sum(item.wall_seconds for item in stages.values())),
            user_seconds=max(0.0, user - sum(item.user_seconds for item in stages.values())),
            system_seconds=max(0.0, system - sum(item.system_seconds for item in stages.values())),
        )
    peak = peak_rss_bytes()
    # ru_maxrss cannot decrease, so retain the high-water mark rather than a
    # misleading delta.  before_rss is intentionally read for documentation.
    del before_rss
    utilization = (100 * cpu / wall) if wall else None
    return Measurement(wall, user, system, cpu, utilization, peak, stages)


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
