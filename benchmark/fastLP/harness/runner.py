#!/usr/bin/env python3
"""Run reproducible fastLP benchmarks and write machine-readable artifacts.

Examples (from the repository root)::

    uv run python benchmark/fastLP/harness/runner.py --scenario smoke --validate
    uv run python benchmark/fastLP/harness/runner.py --csv benchmark/data/small_balanced.csv --repetitions 5
"""

from __future__ import annotations

import argparse
import cProfile
import csv
import gc
import json
import pstats
import sys
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Support direct execution as documented, without making the parent benchmark
# directory an importable package.
if __package__:
    from .measurement import Measurement, measure, runtime_environment, temporary_thread_policy
    from .reference import fit_independent_horizons
    from .scenarios import SCENARIOS, make_panel
else:
    from measurement import Measurement, measure, runtime_environment, temporary_thread_policy
    from reference import fit_independent_horizons
    from scenarios import SCENARIOS, make_panel

from fastlp import LocalProjection


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _controls(data: Any) -> tuple[str, ...]:
    controls = [column for column in data.columns if column == "control" or column.startswith("control_")]
    return tuple(sorted(controls))


def _column_array(data: Any, column: str) -> np.ndarray:
    if isinstance(data, pd.DataFrame):
        return data[column].to_numpy()
    return data.get_column(column).to_numpy()


def _balanced(data: Any) -> bool:
    # Exact label sequences, rather than merely equal row counts, match fastLP's
    # definition of a balanced panel.
    unit = _column_array(data, "unit")
    time = _column_array(data, "time")
    if not len(unit):
        return False
    order = np.lexsort((time, unit))
    unit = unit[order]
    time = time[order]
    starts = np.flatnonzero(np.r_[True, unit[1:] != unit[:-1]])
    sizes = np.diff(np.r_[starts, len(unit)])
    if not np.all(sizes == sizes[0]):
        return False
    labels = time.reshape(len(starts), int(sizes[0]))
    return bool(np.all(labels == labels[0]))


def _as_pandas(data: Any) -> pd.DataFrame:
    """Create a small/reference pandas view without requiring PyArrow."""
    if isinstance(data, pd.DataFrame):
        return data
    return pd.DataFrame({column: data.get_column(column).to_numpy() for column in data.columns})


def _as_polars(data: pd.DataFrame) -> Any:
    try:
        import polars as pl
    except ImportError as error:
        raise SystemExit("--backend polars requires: uv run --extra polars ...") from error
    return pl.DataFrame({column: data[column].to_numpy() for column in data.columns})


def _dimensions(data: Any, horizons: int, fixed_effects: tuple[str, ...], controls: tuple[str, ...]) -> dict[str, Any]:
    return {
        "n_rows": len(data),
        "n_units": int(len(np.unique(_column_array(data, "unit")))),
        "n_periods": int(len(np.unique(_column_array(data, "time")))),
        "horizons": horizons,
        "n_horizons": horizons + 1,
        "n_regressors": 1 + len(controls),
        "n_fixed_effects": len(fixed_effects),
        "fixed_effects": list(fixed_effects),
    }


def _agreement(fitted: LocalProjection, reference: Any, atol: float, rtol: float) -> dict[str, Any]:
    coef_error = np.abs(fitted.coef_ - reference.coef)
    stderr_error = np.abs(fitted.stderr_ - reference.stderr)
    passed = bool(np.allclose(fitted.coef_, reference.coef, atol=atol, rtol=rtol) and np.allclose(
        fitted.stderr_, reference.stderr, atol=atol, rtol=rtol
    ))
    return {
        "available": True,
        "passed": passed,
        "atol": atol,
        "rtol": rtol,
        "max_abs_coef_error": float(coef_error.max(initial=0.0)),
        "max_abs_stderr_error": float(stderr_error.max(initial=0.0)),
    }


def _stage_medians(measurements: list[Measurement]) -> dict[str, dict[str, float | None]]:
    """Return per-stage medians for every clock recorded by ``measure``."""
    stages = sorted({stage for measurement in measurements for stage in measurement.stages})
    result: dict[str, dict[str, float | None]] = {}
    for stage in stages:
        values = [measurement.stages.get(stage) for measurement in measurements]
        present = [value for value in values if value is not None]
        wall = float(np.median([value.wall_seconds for value in present]))
        user = float(np.median([value.user_seconds for value in present]))
        system = float(np.median([value.system_seconds for value in present]))
        cpu = user + system
        result[stage] = {
            "wall_seconds": wall,
            "user_seconds": user,
            "system_seconds": system,
            "cpu_seconds": cpu,
            "cpu_utilization_percent": 100 * cpu / wall if wall else None,
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--scenario", choices=SCENARIOS, default="smoke", help="deterministic generated design")
    source.add_argument("--csv", type=Path, help="existing panel CSV; reading happens before measurement")
    source.add_argument("--parquet", type=Path, help="existing panel Parquet file; reading happens before measurement")
    parser.add_argument("--backend", choices=("pandas", "polars"), default="pandas")
    parser.add_argument("--n-units", type=int, help="override a generated scenario's number of panel units")
    parser.add_argument("--n-periods", type=int, help="override a generated scenario's number of periods per unit")
    parser.add_argument("--n-controls", type=int, help="override a generated scenario's number of controls")
    parser.add_argument("--unbalanced-share", type=float, help="override a generated scenario's missing-row share")
    parser.add_argument("--seed", type=int, help="override a generated scenario's random seed")
    parser.add_argument("--horizons", type=int, help="override a scenario's horizon count")
    parser.add_argument("--repetitions", type=int, default=3, help="measured fits (default: 3)")
    parser.add_argument("--warmups", type=int, default=1, help="unreported warm-up fits (default: 1)")
    parser.add_argument(
        "--covariance",
        choices=("homoskedastic", "hc0", "hc1", "hc2", "hc3", "cluster", "hac", "driscoll_kraay"),
        default="cluster",
    )
    parser.add_argument("--fixed-effects", nargs="*", default=["unit", "time"])
    parser.add_argument("--threads", type=int, help="set coordinated Rayon/BLAS thread counts")
    parser.add_argument("--response", choices=("level", "cumulative"), default="level")
    parser.add_argument("--memory-budget", help="working-memory budget, for example 4GB")
    parser.add_argument(
        "--discard-residuals", action="store_true", help="do not retain residual arrays"
    )
    parser.add_argument("--validate", action="store_true", help="compare with independent-horizon reference if balanced")
    parser.add_argument("--baseline", action="store_true", help="also time the independent-horizon reference")
    parser.add_argument("--profile", action="store_true", help="write a cProfile report for one fastLP fit")
    parser.add_argument("--no-stage-timing", action="store_true", help="only measure end-to-end fit time")
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--rtol", type=float, default=1e-7)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark/results"))
    arguments = parser.parse_args()
    if arguments.repetitions < 1 or arguments.warmups < 0:
        parser.error("repetitions must be at least one and warmups cannot be negative")
    if arguments.threads is not None and arguments.threads < 1:
        parser.error("threads must be at least one")
    if any(value is not None for value in (arguments.n_units, arguments.n_periods, arguments.n_controls, arguments.unbalanced_share, arguments.seed)):
        if arguments.csv or arguments.parquet:
            parser.error("generated-panel overrides cannot be combined with file input")
        if arguments.n_units is not None and arguments.n_units < 1:
            parser.error("--n-units must be positive")
        if arguments.n_periods is not None and arguments.n_periods < 2:
            parser.error("--n-periods must be at least two")
        if arguments.n_controls is not None and arguments.n_controls < 0:
            parser.error("--n-controls cannot be negative")
        if arguments.unbalanced_share is not None and not 0 <= arguments.unbalanced_share < 1:
            parser.error("--unbalanced-share must be in [0, 1)")
    return arguments


def main() -> None:
    args = parse_args()
    input_path = args.csv or args.parquet
    if input_path:
        if not input_path.is_file():
            raise SystemExit(f"input file not found: {input_path}")
        if args.horizons is None:
            raise SystemExit("--horizons is required with file input (it cannot be inferred safely)")
        file_format = "csv" if args.csv else "parquet"
        if args.backend == "pandas":
            try:
                data = pd.read_csv(input_path) if args.csv else pd.read_parquet(input_path)
            except ImportError as error:
                raise SystemExit(
                    "pandas Parquet input requires a pandas Parquet engine such as pyarrow"
                ) from error
        else:
            try:
                import polars as pl
            except ImportError as error:
                raise SystemExit("--backend polars requires: uv run --extra polars ...") from error
            data = pl.read_csv(input_path) if args.csv else pl.read_parquet(input_path)
        source = {
            "kind": file_format,
            "path": str(input_path.resolve()),
            "backend": args.backend,
        }
        horizons = args.horizons
    else:
        scenario = SCENARIOS[args.scenario]
        overrides = {
            name: value
            for name, value in {
                "n_units": args.n_units,
                "n_periods": args.n_periods,
                "n_controls": args.n_controls,
                "unbalanced_share": args.unbalanced_share,
                "seed": args.seed,
            }.items()
            if value is not None
        }
        if overrides:
            scenario = replace(scenario, name="custom", **overrides)
        data = make_panel(scenario)
        if args.backend == "polars":
            data = _as_polars(data)
        source = {"kind": "generated", **scenario.metadata()}
        source["backend"] = args.backend
        horizons = args.horizons if args.horizons is not None else scenario.horizons
    required = {"unit", "time", "outcome", "shock"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise SystemExit(f"input is missing required benchmark columns: {missing}")
    controls = _controls(data)
    effects = tuple(args.fixed_effects)
    missing_effects = sorted(set(effects).difference(data.columns))
    if missing_effects:
        raise SystemExit(f"input is missing fixed-effect columns: {missing_effects}")
    balanced = _balanced(data)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir / f"fastlp_{run_id}"
    output.mkdir(parents=True, exist_ok=False)
    cluster = "unit" if args.covariance == "cluster" else None
    fit_options = dict(
        outcome="outcome", shock="shock", controls=controls, unit="unit", time="time", fixed_effects=effects, cluster=cluster
    )

    def fit_once() -> LocalProjection:
        model = LocalProjection(
            horizons=horizons,
            covariance=args.covariance,
            sample="common" if balanced else "per_horizon",
            response=args.response,
            memory_budget=args.memory_budget,
            retain_residuals=not args.discard_residuals,
        )
        return model.fit(data, **fit_options)

    with temporary_thread_policy(args.threads):
        environment = runtime_environment()
        for _ in range(args.warmups):
            fit_once()
        measurements: list[Measurement] = []
        fitted: LocalProjection | None = None
        for _ in range(args.repetitions):
            gc.collect()
            holder: dict[str, LocalProjection] = {}
            result = measure(lambda: holder.setdefault("fit", fit_once()), stage_timing=not args.no_stage_timing)
            measurements.append(result)
            fitted = holder["fit"]

        reference = None
        agreement: dict[str, Any]
        if args.validate or args.baseline:
            if balanced and args.response == "level" and args.covariance in {"cluster", "hc1"}:
                reference_data = _as_pandas(data)
                reference_call = lambda: fit_independent_horizons(
                    reference_data, horizons=horizons, covariance=args.covariance, fixed_effects=effects, cluster=cluster, controls=controls
                )
                if args.baseline:
                    baseline = measure(reference_call, stage_timing=False)
                else:
                    baseline = None
                reference = reference_call()
                agreement = _agreement(fitted, reference, args.atol, args.rtol)  # type: ignore[arg-type]
            else:
                baseline = None
                agreement = {
                    "available": False,
                    "reason": "reference currently supports only balanced level-response hc1 and one-way cluster covariance",
                }
        else:
            baseline = None
            agreement = {"available": False, "reason": "pass --validate to run numerical agreement"}

        if args.profile:
            profile = cProfile.Profile()
            profile.runcall(fit_once)
            profile.dump_stats(output / "fit.pstats")
            with (output / "profile.txt").open("w", encoding="utf-8") as stream:
                pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats()

    observation_rows = []
    stage_rows = []
    for index, result in enumerate(measurements, start=1):
        serialized = asdict(result)
        stages = serialized.pop("stages")
        observation_rows.append(
            {
                "run": index,
                **serialized,
                "stages_seconds": json.dumps(result.stages_seconds, sort_keys=True),
                "stages": json.dumps(stages, sort_keys=True),
            }
        )
        for stage, timing in result.stages.items():
            stage_rows.append(
                {
                    "run": index,
                    "stage": stage,
                    "wall_seconds": timing.wall_seconds,
                    "user_seconds": timing.user_seconds,
                    "system_seconds": timing.system_seconds,
                    "cpu_seconds": timing.cpu_seconds,
                    "cpu_utilization_percent": timing.cpu_utilization_percent,
                }
            )
    stages_median = _stage_medians(measurements)
    summary = {
        "run_id": run_id,
        "source": source,
        "environment": environment,
        "dimensions": _dimensions(data, horizons, effects, controls),
        "settings": {
            "covariance": args.covariance,
            "sample": "common" if balanced else "per_horizon",
            "balanced": balanced,
            "response": args.response,
            "memory_budget": args.memory_budget,
            "retain_residuals": not args.discard_residuals,
            "input_backend": args.backend,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
        },
        "wall_seconds": {"min": min(item.wall_seconds for item in measurements), "median": float(np.median([item.wall_seconds for item in measurements])), "max": max(item.wall_seconds for item in measurements)},
        "peak_rss_bytes": {
            "max": max(
                (item.peak_rss_bytes for item in measurements if item.peak_rss_bytes is not None),
                default=None,
            )
        },
        # Keep this wall-only key for existing dashboards while exposing all
        # three clocks below.
        "stages_seconds_median": {stage: values["wall_seconds"] for stage, values in stages_median.items()},
        "stages_median": stages_median,
        "agreement": agreement,
    }
    if baseline is not None:
        summary["independent_horizon_baseline"] = asdict(baseline)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "agreement.json").write_text(json.dumps(agreement, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output / "observations.csv", observation_rows)
    _write_csv(output / "stages.csv", stage_rows)
    if not stage_rows:
        (output / "stages.csv").write_text(
            "run,stage,wall_seconds,user_seconds,system_seconds,cpu_seconds,cpu_utilization_percent\n",
            encoding="utf-8",
        )
    print(f"Results: {output}")
    print(f"fastLP median fit: {summary['wall_seconds']['median']:.6f} s ({args.repetitions} measured run(s))")
    if agreement.get("available"):
        print(f"Numerical agreement: {'PASS' if agreement['passed'] else 'FAIL'}")
        if not agreement["passed"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
