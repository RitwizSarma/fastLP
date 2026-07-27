#!/usr/bin/env python3
"""Run fastLP on one of the synthetic benchmark panels."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from fastlp import LocalProjection


CONFIG = {
    "small_balanced": {"horizons": 12, "unbalanced": False},
    "small_unbalanced": {"horizons": 12, "unbalanced": True},
    "large_balanced": {"horizons": 2, "unbalanced": False},
    "large_unbalanced": {"horizons": 2, "unbalanced": True},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=CONFIG, help="dataset name from benchmark/data/")
    args = parser.parse_args()

    benchmark_dir = Path(__file__).resolve().parents[1]
    input_file = benchmark_dir / "data" / f"{args.dataset}.csv"
    if not input_file.exists():
        parser.error(f"dataset not found: {input_file}; run benchmark/data/generate_data.py first")

    config = CONFIG[args.dataset]
    data = pd.read_csv(input_file)
    estimator = LocalProjection(
        horizons=config["horizons"],
        covariance="cluster",
        allow_unbalanced=config["unbalanced"],
    )

    # Keep CSV loading outside the timed region, matching the R benchmark.
    start = time.perf_counter()
    fitted = estimator.fit(
        data,
        outcome="outcome",
        shock="shock",
        controls=["control"],
        unit="unit",
        time="time",
        fixed_effects=["unit", "time"],
        cluster="unit",
    )
    elapsed = time.perf_counter() - start

    print(f"Dataset: {args.dataset}")
    print(f"Rows loaded: {len(data):,}")
    print(f"Estimation time: {elapsed:.3f} seconds")
    print("\nFinal estimates:")
    print(fitted.to_frame().to_string(index=False))


if __name__ == "__main__":
    main()
