# LP benchmark inputs and runners

`data/generate_data.py` creates four CSV inputs using the DGP in
`src/notebooks/synthetic_benchmark.ipynb`:

| Dataset | Units × periods before omissions | Rows written | Max horizon |
| --- | ---: | ---: | ---: |
| `small_balanced` | 200 × 40 | 8,000 | 12 |
| `small_unbalanced` | 200 × 40 | 7,200 | 12 |
| `large_balanced` | 1,250,000 × 8 | 10,000,000 | 2 |
| `large_unbalanced` | 1,250,000 × 8 | 8,750,000 | 2 |

The unbalanced versions delete a reproducible random 10% of each unit's time
periods after generating the notebook DGP. Thus they retain all units but do
not treat observations separated by a missing time as adjacent.

Generate (or regenerate) all inputs from the project root:

```bash
UV_CACHE_DIR=/tmp/fastlp-uv-cache uv run python benchmark/data/generate_data.py
```

To run the R benchmark, set `dataset` at the top of
`R/benchmark_fixest.R`, install `fixest`, then run:

```bash
Rscript benchmark/R/benchmark_fixest.R
```

It writes `results_fixest_<dataset>.csv` in this directory. Its reported
calculation time covers the complete set of horizon regressions, but excludes
CSV loading and result-file writing.

To run Stata, set `dataset` at the top of either
`Stata/benchmark_locproj.do` or `Stata/benchmark_reghdfe.do` and execute the
do-file with `benchmark/Stata` as the working directory. The direct reghdfe
runner constructs the exact-time common sample and then estimates every
horizon with unit and time fixed effects and unit-clustered covariance. Its
timer includes sample construction, matching fastLP's in-memory fit boundary.

The R runner uses the same common sample. Cross-language speed claims are valid
only when the sample, regressors, FE structure, covariance correction, thread
policy, and timed boundary agree. Report cold and warmed medians on the same
machine; warm-up runs are measurement hygiene, not estimator caching.

To run fastLP, choose a dataset on the command line from the project root:

```bash
uv run python benchmark/fastLP/benchmark.py small_balanced
```

The runner accepts all four dataset names. It excludes CSV loading from the
reported estimation time and prints the final estimates for every horizon.
