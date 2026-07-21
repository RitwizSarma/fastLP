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

To run Stata, set `dataset` at the top of `Stata/benchmark_locproj.do`, ensure
`locproj` is installed, and execute the do-file with `benchmark/Stata` as the
working directory. It uses `fe` for unit fixed effects, `i.time` for time fixed
effects, and clusters at the unit level. The timer covers `locproj` only.
