# fastLP benchmark harness

This is the extensible benchmark runner.  It measures the complete in-memory
fit (CSV I/O is deliberately outside the timer), records a process peak-RSS
high-water mark, CPU time/utilization, active thread-policy environment,
dimensions, and timing for stable fit boundaries.  It also writes results in
formats suitable for a dashboard or a regression gate.

```bash
# Fast correctness and instrumentation smoke test
uv run python benchmark/fastLP/harness/runner.py --scenario smoke --validate

# Repeat a realistic generated case and compare the independent-horizon baseline
uv run python benchmark/fastLP/harness/runner.py --scenario small --repetitions 5 --baseline --validate

# Reuse the existing CSV data; loading remains outside the reported fit time
uv run python benchmark/fastLP/harness/runner.py --csv benchmark/data/n5000_t40.csv --horizons 12 --profile

# Use Polars input or a Parquet source; loading remains outside the fit timer
uv run --extra polars python benchmark/fastLP/harness/runner.py \
  --csv benchmark/data/n5000_t40.csv --horizons 12 --backend polars
uv run python benchmark/fastLP/harness/runner.py \
  --parquet benchmark/data/n5000_t40.parquet --horizons 12 --backend pandas

# Vary N (rows, via units x periods) and K (shock plus controls) in fresh processes
uv run python benchmark/fastLP/harness/grid.py \
  --n-units 312,1250,5000 --n-controls 1,8,32 \
  --n-periods 32 --horizons 12 --threads 1

# Repeat selected complete-panel cases while varying N and K
uv run python benchmark/fastLP/harness/grid.py \
  --n-units 5000,10000 --n-controls 8 --n-periods 40 \
  --horizons 12 --threads 1
```

Each invocation creates `benchmark/results/fastlp_<UTC timestamp>/` with:

- `summary.json`: run configuration, dimensions, environment, aggregate timing,
  optional baseline, and agreement result.
- `observations.csv`: one row per timed fit, including wall time, CPU time,
  CPU utilization, peak RSS, and serialized stage timing.
- `stages.csv`: tidy timing rows, with wall-clock, user CPU, system CPU, and
  total CPU time for every stage.
- `agreement.json`: coefficient and standard-error error bounds when requested.
- `fit.pstats` and `profile.txt` when `--profile` is supplied.

`--baseline` times a deliberately independent-horizon implementation. It is
intended for manageable complete panels; use it to find the horizon count at
which cache construction pays off. `--validate` applies the same reference as
a numerical regression check.

The stage rows partition total fit time into lag construction, lead
construction, singleton pruning, FE encoding, demeaning, rank checking,
factorization, linear solves, covariance, and `other`. `other` contains input
validation, pandas preparation, matrix products, residual formation, and
result assembly. The clocks are process-local: `system_seconds` is CPU time in
the OS kernel, while `user_seconds` is CPU time in user space. Peak RSS is a
process lifetime high-water mark on Linux and macOS; the grid runner therefore
uses a fresh subprocess for every cell.

The grid produces `cases.csv` (one median end-to-end row per N/K/dataset
combination), `stages.csv` (one median stage row per combination), and the
per-case harness artifacts. In this harness N is the realized row count and K
is the number of regressors after adding the shock (and before fixed effects).

When using `--csv` or `--parquet`, always supply `--horizons`: the correct value
cannot be derived safely from an arbitrary file. File loading remains outside
the measured fit boundary; the input backend and format are recorded in
`summary.json`. Use `--threads N` to set one coordinated `RAYON_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and `OMP_NUM_THREADS` policy for a
run. This makes oversubscription experiments reproducible. The chosen policy
and any existing thread settings are stored in `summary.json`.

Pandas Parquet input requires a pandas-compatible Parquet engine such as
PyArrow. Polars Parquet input is available through the `polars` extra without
an additional PyArrow dependency.

Use `--response cumulative` to exercise cumulative responses,
`--memory-budget 4GB` to test adaptive horizon batching, and
`--discard-residuals` for the low-memory result path. The independent-horizon
validation fixture currently covers level responses only.
