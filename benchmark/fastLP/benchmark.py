#!/usr/bin/env python3
"""Run repeated fastLP estimations on one benchmark panel."""

from __future__ import annotations

import argparse
import resource
import statistics
import time
from pathlib import Path

import pandas as pd

from fastlp import LocalProjection


CONFIG = {
    "n5000_t40": {"horizons": 12},
    "n10000_t40": {"horizons": 12},
}


def _summary(values: list[float]) -> dict[str, float]:
    series = pd.Series(values)
    return {
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values),
        "median": statistics.median(values),
        "q1": float(series.quantile(0.25)),
        "q3": float(series.quantile(0.75)),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=CONFIG, help="dataset name from benchmark/data/")
    parser.add_argument(
        "--output-dir", type=Path, help="directory for the timing CSV (default: benchmark/)"
    )
    parser.add_argument("--repetitions", type=int, default=200)
    args = parser.parse_args()
    if args.repetitions < 2:
        parser.error("--repetitions must be at least 2")

    benchmark_dir = Path(__file__).resolve().parents[1]
    input_file = benchmark_dir / "data" / f"{args.dataset}.csv"
    if not input_file.exists():
        parser.error(f"dataset not found: {input_file}; run benchmark/data/generate_data.py first")

    config = CONFIG[args.dataset]
    data = pd.read_csv(input_file)
    timings: list[dict[str, float | int | str]] = []
    estimates = None
    for repetition in range(1, args.repetitions + 1):
        estimator = LocalProjection(
            horizons=config["horizons"], covariance="cluster", sample="common"
        )
        usage_start = resource.getrusage(resource.RUSAGE_SELF)
        wall_start = time.perf_counter()
        estimator.fit(
            data,
            outcome="outcome",
            shock="shock",
            controls=["control"],
            unit="unit",
            time="time",
            fixed_effects=["unit", "time"],
            cluster="unit",
        )
        usage_end = resource.getrusage(resource.RUSAGE_SELF)
        timings.append(
            {
                "dataset": args.dataset,
                "repetition": repetition,
                "wall_seconds": time.perf_counter() - wall_start,
                "user_seconds": usage_end.ru_utime - usage_start.ru_utime,
                "system_seconds": usage_end.ru_stime - usage_start.ru_stime,
            }
        )
        if repetition == 1:
            estimates = (
                estimator.to_frame()
                .loc[lambda frame: frame["coefficient"].isin(["shock", "control"])]
                .pivot(index="horizon", columns="coefficient", values="estimate")
                .rename(columns={"shock": "estimate_shock", "control": "estimate_control"})
                .reset_index()
            )

    output_dir = args.output_dir if args.output_dir is not None else benchmark_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"results_fastlp_{args.dataset}.csv"
    pd.DataFrame(timings).to_csv(output_file, index=False)
    assert estimates is not None
    estimates.insert(0, "dataset", args.dataset)
    estimates.to_csv(output_dir / f"estimates_fastlp_{args.dataset}.csv", index=False)

    summaries = {
        "wall": _summary([row["wall_seconds"] for row in timings]),
        "user": _summary([row["user_seconds"] for row in timings]),
        "system": _summary([row["system_seconds"] for row in timings]),
    }
    print(f"Dataset: {args.dataset}")
    print(f"Rows: {len(data):,} | N={data['unit'].nunique():,} | T={data['time'].nunique():,}")
    print(f"Repetitions: {args.repetitions}")
    print("Timing summary (seconds):")
    print(pd.DataFrame(summaries).T.to_string(float_format=lambda value: f"{value:.6f}"))
    print(f"Results written to: {output_file}")


if __name__ == "__main__":
    main()
