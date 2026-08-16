#!/usr/bin/env python3
"""Bootstrap timing summaries and create a cross-library comparison plot."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LIBRARIES = {
    "fastLP": "results_fastlp_{dataset}.csv",
    "R: fixest": "results_fixest_{dataset}.csv",
    "Stata: reghdfe": "results_reghdfe_{dataset}.csv",
}

DATASET_NAMES = {
    "n5000_t40": "N=5,000, T=40",
    "n10000_t40": "N=10,000, T=40",
}


def bootstrap_statistic(
    values: np.ndarray,
    statistic: str,
    replications: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Return the observed statistic and its percentile bootstrap 95% CI."""
    if statistic == "median":
        calculate = np.median
    else:  # pragma: no cover - guarded by the caller
        raise ValueError(f"unknown statistic: {statistic}")

    bootstrapped = np.empty(replications)
    for draw in range(replications):
        sample = rng.choice(values, size=values.size, replace=True)
        bootstrapped[draw] = calculate(sample)
    lower, upper = np.quantile(bootstrapped, [0.025, 0.975])
    return float(calculate(values)), float(lower), float(upper)


def load_timings(input_dir: Path, dataset: str) -> dict[str, np.ndarray]:
    timings: dict[str, np.ndarray] = {}
    for library, filename_template in LIBRARIES.items():
        path = input_dir / filename_template.format(dataset=dataset)
        if not path.exists():
            raise FileNotFoundError(f"timing CSV not found: {path}")
        frame = pd.read_csv(path)
        if "wall_seconds" not in frame:
            raise ValueError(f"{path} does not contain a wall_seconds column")
        values = pd.to_numeric(frame["wall_seconds"], errors="coerce").dropna().to_numpy()
        if values.size < 2:
            raise ValueError(f"{path} must contain at least two valid timings")
        timings[library] = values
    return timings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="dataset suffix, for example n5000_t40")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="directory containing the timing CSVs (default: benchmark/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="directory for the PNG and summary CSV (default: --input-dir)",
    )
    parser.add_argument("--replications", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()
    if args.replications < 1:
        parser.error("--replications must be positive")

    output_dir = args.output_dir or args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    timings = load_timings(args.input_dir, args.dataset)
    rng = np.random.default_rng(args.seed)

    rows: list[dict[str, float | int | str]] = []
    for library, values in timings.items():
        estimate, ci_low, ci_high = bootstrap_statistic(
            values, "median", args.replications, rng
        )
        rows.append(
            {
                "dataset": args.dataset,
                "library": library,
                "statistic": "median",
                "estimate_seconds": estimate,
                "ci_low_seconds": ci_low,
                "ci_high_seconds": ci_high,
                "n_timings": values.size,
                "bootstrap_replications": args.replications,
                "seed": args.seed,
            }
        )

    summary = pd.DataFrame(rows)
    summary_file = output_dir / f"timing_summary_{args.dataset}.csv"
    summary.to_csv(summary_file, index=False)

    plot_data = summary.set_index("library").reindex(list(LIBRARIES))
    y = np.arange(len(LIBRARIES))
    estimates = plot_data["estimate_seconds"].to_numpy()
    lower = plot_data["ci_low_seconds"].to_numpy()
    upper = plot_data["ci_high_seconds"].to_numpy()
    yerr = np.vstack((estimates - lower, upper - estimates))

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh(
        y,
        estimates,
        height=0.52,
        color="#0f4ca8",
        alpha=0.88,
        edgecolor="#1e3a8a",
        linewidth=0.8,
        xerr=yerr,
        error_kw={"ecolor": "#000000", "capsize": 5, "elinewidth": 1.5},
    )
    for position, estimate in zip(y, estimates):
        ax.text(
            estimate,
            position,
            f"  {estimate:.3f}s",
            va="center",
            ha="left",
            color="#111827",
            fontsize=10,
        )

    ax.set_yticks(y, list(LIBRARIES))
    ax.invert_yaxis()
    ax.set_xlabel("Median wall-clock time (seconds)")
    fig.suptitle(
        f"Panel LP time comparison on dataset with {DATASET_NAMES.get(args.dataset)}",
        x=0.125,
        y=0.98,
        ha="left",
        va="top",
        fontsize=15,
    )
    fig.text(
        0.125,
        0.925,
        f"Bars show medians; whiskers show 95% bootstrap CIs (n={args.replications:,} replications)",
        ha="left",
        va="top",
        fontsize=10,
        color="#4b5563",
    )
    ax.grid(axis="x", alpha=0.28)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.margins(x=0.08)
    fig.subplots_adjust(top=0.84)
    fig.tight_layout()
    plot_file = output_dir / f"timing_comparison_{args.dataset}.png"
    fig.savefig(plot_file, dpi=200, transparent=True)
    plt.close(fig)

    print(f"Summary written to: {summary_file}")
    print(f"Plot written to: {plot_file}")


if __name__ == "__main__":
    main()
