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
uv run python benchmark/fastLP/harness/runner.py --csv benchmark/data/small_balanced.csv --horizons 12 --profile
```

Each invocation creates `benchmark/results/fastlp_<UTC timestamp>/` with:

- `summary.json`: run configuration, dimensions, environment, aggregate timing,
  optional baseline, and agreement result.
- `observations.csv`: one row per timed fit, including wall time, CPU time,
  CPU utilization, peak RSS, and serialized stage timing.
- `stages.csv`: tidy timing rows, making comparisons across versions easy.
- `agreement.json`: coefficient and standard-error error bounds when requested.
- `fit.pstats` and `profile.txt` when `--profile` is supplied.

`--baseline` times a deliberately independent-horizon implementation.  It is
only intended for small, balanced inputs; use it to find the horizon count at
which cache construction pays off. `--validate` applies the same reference as
a numerical regression check. The reference is intentionally unavailable for
unbalanced designs, whose samples vary by horizon.

The stage rows partition total fit time into `fe_encoding`, `demeaning`, rank
checking, factorization, linear solves, covariance, and `other`. `other`
contains validation, pandas preparation, matrix products, residual formation,
and result assembly. Peak RSS is process lifetime high-water mark on Linux and
macOS; start a fresh command for clean cross-case memory comparisons.

When using `--csv`, always supply `--horizons`: the correct value cannot be
derived safely from an arbitrary file. Use `--threads N` to set one coordinated `RAYON_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and `OMP_NUM_THREADS` policy for a
run. This makes oversubscription experiments reproducible. The chosen policy
and any existing thread settings are stored in `summary.json`.
