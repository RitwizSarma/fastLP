# Cross-library LP benchmark

This suite runs the same complete-panel CSV through fastLP, R/fixest, and
Stata/reghdfe. Each runner estimates all horizons 200 times and writes one CSV 
row for every repetition. The CSVs hold the raw wall-clock timings and, where
the runtime exposes them, user-CPU and system-CPU timings. A separate
`estimates_<library>_<dataset>.csv` records the first repetition's shock and
 control estimates by horizon.

The two datasets are:

| Dataset | Units (N) | Periods (T) | Rows | Max horizon |
| --- | ---: | ---: | ---: | ---: |
| `n5000_t40` | 5,000 | 40 | 200,000 | 12 |
| `n10000_t40` | 10,000 | 40 | 400,000 | 12 |

Generate the inputs from the project root:

```bash
uv run python benchmark/data/generate_data.py
```

Run fastLP:

```bash
uv run python benchmark/fastLP/benchmark.py n5000_t40
uv run python benchmark/fastLP/benchmark.py n10000_t40
```

Run R/fixest by opening `R/benchmark_fixest.R`, setting `dataset` at the top,
and sourcing the script in RStudio or the R GUI:

```r
dataset <- "n5000_t40"
source("benchmark/R/benchmark_fixest.R")
```

The Stata do-file is in `benchmark/Stata`.


CSV loading and output writing are outside the estimation timers. We time 
the complete LP regression over horizons 0 through 12. After the timing CSVs 
have been generated, create 95% percentile bootstrap confidence intervals 
and a comparison plot with:

```bash
uv run python benchmark/compare_timings.py n5000_t40
```

By default, we do 5,000 bootstrap replications and produce
`timing_summary_n5000_t40.csv` and `timing_comparison_n5000_t40.png` in
`benchmark/`. Use `--seed`, `--replications`, or `--output-dir` to customize
the analysis.
