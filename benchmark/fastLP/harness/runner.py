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
from dataclasses import asdict
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


def _controls(data: pd.DataFrame) -> tuple[str, ...]:
    controls = [column for column in data.columns if column == "control" or column.startswith("control_")]
    return tuple(sorted(controls))


def _balanced(data: pd.DataFrame) -> bool:
    # Exact label sequences, rather than merely equal row counts, match fastLP's
    # definition of a balanced panel.
    sequences = data.sort_values(["unit", "time"], kind="stable").groupby("unit", sort=False)["time"]
    first: tuple[object, ...] | None = None
    for _, labels in sequences:
        current = tuple(labels.tolist())
        if first is None:
            first = current
        elif current != first:
            return False
    return first is not None


def _dimensions(data: pd.DataFrame, horizons: int, fixed_effects: tuple[str, ...], controls: tuple[str, ...]) -> dict[str, Any]:
    return {
        "n_rows": len(data),
        "n_units": int(data["unit"].nunique()),
        "n_periods": int(data["time"].nunique()),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--scenario", choices=SCENARIOS, default="smoke", help="deterministic generated design")
    source.add_argument("--csv", type=Path, help="existing panel CSV; reading happens before measurement")
    parser.add_argument("--horizons", type=int, help="override a scenario's horizon count")
    parser.add_argument("--repetitions", type=int, default=3, help="measured fits (default: 3)")
    parser.add_argument("--warmups", type=int, default=1, help="unreported warm-up fits (default: 1)")
    parser.add_argument("--covariance", choices=("cluster", "hc1"), default="cluster")
    parser.add_argument("--fixed-effects", nargs="*", default=["unit", "time"])
    parser.add_argument("--threads", type=int, help="set coordinated Rayon/BLAS thread counts")
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
    return arguments


def main() -> None:
    args = parse_args()
    if args.csv:
        if not args.csv.is_file():
            raise SystemExit(f"CSV not found: {args.csv}")
        if args.horizons is None:
            raise SystemExit("--horizons is required with --csv (it cannot be inferred safely)")
        data = pd.read_csv(args.csv)
        source: dict[str, Any] = {"kind": "csv", "path": str(args.csv.resolve())}
        horizons = args.horizons
    else:
        scenario = SCENARIOS[args.scenario]
        data = make_panel(scenario)
        source = {"kind": "generated", **scenario.metadata()}
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
        model = LocalProjection(horizons=horizons, covariance=args.covariance, allow_unbalanced=not balanced)
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
            if balanced:
                reference_call = lambda: fit_independent_horizons(
                    data, horizons=horizons, covariance=args.covariance, fixed_effects=effects, cluster=cluster, controls=controls
                )
                if args.baseline:
                    baseline = measure(reference_call, stage_timing=False)
                else:
                    baseline = None
                reference = reference_call()
                agreement = _agreement(fitted, reference, args.atol, args.rtol)  # type: ignore[arg-type]
            else:
                baseline = None
                agreement = {"available": False, "reason": "reference currently supports balanced panels only"}
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
        observation_rows.append({"run": index, **asdict(result), "stages_seconds": json.dumps(result.stages_seconds, sort_keys=True)})
        stage_rows.extend({"run": index, "stage": stage, "seconds": seconds} for stage, seconds in result.stages_seconds.items())
    summary = {
        "run_id": run_id,
        "source": source,
        "environment": environment,
        "dimensions": _dimensions(data, horizons, effects, controls),
        "settings": {"covariance": args.covariance, "allow_unbalanced": not balanced, "balanced": balanced, "warmups": args.warmups, "repetitions": args.repetitions},
        "wall_seconds": {"min": min(item.wall_seconds for item in measurements), "median": float(np.median([item.wall_seconds for item in measurements])), "max": max(item.wall_seconds for item in measurements)},
        "agreement": agreement,
    }
    if baseline is not None:
        summary["independent_horizon_baseline"] = asdict(baseline)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "agreement.json").write_text(json.dumps(agreement, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output / "observations.csv", observation_rows)
    _write_csv(output / "stages.csv", stage_rows)
    if not stage_rows:
        (output / "stages.csv").write_text("run,stage,seconds\n", encoding="utf-8")
    print(f"Results: {output}")
    print(f"fastLP median fit: {summary['wall_seconds']['median']:.6f} s ({args.repetitions} measured run(s))")
    if agreement.get("available"):
        print(f"Numerical agreement: {'PASS' if agreement['passed'] else 'FAIL'}")
        if not agreement["passed"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
