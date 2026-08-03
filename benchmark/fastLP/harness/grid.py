#!/usr/bin/env python3
"""Benchmark a reproducible panel grid and collect comparable stage timings.

The process executes every cell in a fresh Python subprocess.  That makes
peak RSS meaningful and prevents one cell's allocator high-water mark or
native thread pool from contaminating another cell.

Example (from the project root)::

    uv run python benchmark/fastLP/harness/grid.py \
        --n-units 312,1250,5000 --n-controls 1,8,32 \
        --n-periods 32 --horizons 12 --threads 1
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "runner.py"


def _positive_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("all values must be positive")
    return values


def _shares(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated decimal shares") from error
    if not values or any(not 0 <= item < 1 for item in values):
        raise argparse.ArgumentTypeError("shares must be in [0, 1)")
    return values


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-units", type=_positive_ints, default=(312, 1250, 5000))
    parser.add_argument("--n-controls", type=lambda value: tuple(int(item) for item in value.split(",")), default=(1, 8, 32))
    parser.add_argument("--unbalanced-shares", type=_shares, default=(0.0,))
    parser.add_argument("--n-periods", type=int, default=32)
    parser.add_argument("--horizons", type=int, default=12)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--covariance", default="cluster", choices=("homoskedastic", "hc0", "hc1", "hc2", "hc3", "cluster", "hac", "driscoll_kraay"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark/results"))
    args = parser.parse_args()
    if args.n_periods < 2 or args.horizons < 0:
        parser.error("--n-periods must be at least two and --horizons cannot be negative")
    if args.horizons >= args.n_periods:
        parser.error("--horizons must be smaller than --n-periods")
    if args.repetitions < 1 or args.warmups < 0 or args.threads < 1:
        parser.error("invalid repetitions, warmups, or thread count")
    if any(value < 0 for value in args.n_controls):
        parser.error("--n-controls cannot contain negative values")
    return args


def main() -> None:
    args = parse_args()
    run_id = datetime.now(UTC).strftime("panel_grid_%Y%m%dT%H%M%SZ")
    output = args.output_dir / run_id
    output.mkdir(parents=True, exist_ok=False)
    case_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []

    for unbalanced_share in args.unbalanced_shares:
        for n_units in args.n_units:
            for n_controls in args.n_controls:
                label = f"u{n_units}_t{args.n_periods}_c{n_controls}_missing{unbalanced_share:g}"
                case_dir = output / label
                command = [
                    sys.executable,
                    str(RUNNER),
                    "--scenario", "smoke",
                    "--n-units", str(n_units),
                    "--n-periods", str(args.n_periods),
                    "--n-controls", str(n_controls),
                    "--unbalanced-share", str(unbalanced_share),
                    "--horizons", str(args.horizons),
                    "--covariance", args.covariance,
                    "--repetitions", str(args.repetitions),
                    "--warmups", str(args.warmups),
                    "--threads", str(args.threads),
                    "--output-dir", str(case_dir),
                ]
                print(f"Running {label}...", flush=True)
                subprocess.run(command, check=True)
                summaries = list(case_dir.glob("fastlp_*/summary.json"))
                if len(summaries) != 1:
                    raise RuntimeError(f"expected exactly one summary for {label}, found {len(summaries)}")
                summary_path = summaries[0]
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                dimensions = summary["dimensions"]
                wall = summary["wall_seconds"]
                case = {
                    "case": label,
                    "summary_path": str(summary_path.relative_to(output)),
                    "balanced": summary["settings"]["balanced"],
                    "unbalanced_share": unbalanced_share,
                    "n_rows": dimensions["n_rows"],
                    "n_units": dimensions["n_units"],
                    "n_periods": dimensions["n_periods"],
                    # K includes shock plus user-requested controls, matching
                    # the regressors actually residualized by fastLP.
                    "k_regressors": dimensions["n_regressors"],
                    "n_controls": n_controls,
                    "horizons": dimensions["horizons"],
                    "wall_seconds_min": wall["min"],
                    "wall_seconds_median": wall["median"],
                    "wall_seconds_max": wall["max"],
                    "peak_rss_bytes": summary["peak_rss_bytes"]["max"],
                }
                case_rows.append(case)
                for stage, timing in summary["stages_median"].items():
                    stage_rows.append({**case, "stage": stage, **timing})

    _write_csv(output / "cases.csv", case_rows)
    _write_csv(output / "stages.csv", stage_rows)
    metadata = {
        "run_id": run_id,
        "n_units": args.n_units,
        "n_controls": args.n_controls,
        "unbalanced_shares": args.unbalanced_shares,
        "n_periods": args.n_periods,
        "horizons": args.horizons,
        "repetitions": args.repetitions,
        "warmups": args.warmups,
        "threads": args.threads,
        "covariance": args.covariance,
    }
    (output / "grid.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Grid results: {output}")
    print(f"Cases: {len(case_rows)}; stage rows: {len(stage_rows)}")


if __name__ == "__main__":
    main()
