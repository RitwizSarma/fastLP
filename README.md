# fastLP

`fastLP` estimates linear local projections for balanced and unbalanced panel data.
It uses one common anchor sample across all horizons, residualizes the
right-hand-side design once, and reuses its factorization for every horizon.

```python
from fastlp import LocalProjection

lp = LocalProjection(horizons=20, covariance="cluster", sample="common")
lp.fit(
    data,
    outcome="log_output",
    shock="monetary_shock",
    controls=["output_l1", "inflation_l1"],
    outcome_lags=[1, 2, 4],
    shock_lags=2,
    control_lags={"output_l1": [1, 3], "inflation_l1": 1},
    unit="country",
    time="quarter",
    fixed_effects=["country", "quarter"],
    cluster="country",
)
print(lp.to_frame())
```

fastLP supports classical, HC0--HC3, one- through four-way CR0/CR1 clustering,
within-unit Newey--West HAC, and Driscoll--Kraay covariance. A cluster term is
a column name; a tuple is one interaction term, so
`cluster=["country", "quarter", ("industry", "quarter")]` requests three
cluster dimensions. `sample="common"`
(the default) retains only anchors with valid outcomes at every requested
horizon. It residualizes the RHS and factors its Gram matrix once, yielding
the fastest, directly comparable coefficient paths even for unbalanced panels.
Use `sample="per_horizon"` to retain every anchor with an exact
`time + horizon` outcome; this can change the sample at each horizon, so it
groups only horizons with identical retained rows for cache reuse. Results
report the selected `sample`, `n_obs_by_horizon_`, and
`sample_index_by_horizon_`. HAC and Driscoll--Kraay require integral numeric
time labels; missing periods are skipped rather than treated as adjacent.
Use `hac_lags`, `hac_kernel` (`"bartlett"`, `"parzen"`, or
`"quadratic_spectral"`), and `hac_debias` to configure those estimators.
`covariance_config_` and `covariance_diagnostics_` record every inference
choice and its per-horizon effective bandwidth.

`outcome_lags`, `shock_lags`, and `control_lags` create within-unit lags from
the ordered panel automatically. A non-negative integer requests all lags up
to that number (`2` creates lags 1 and 2); an explicit sequence permits a
non-consecutive grid. For shocks and controls, a mapping may select lags per
column. Rows without all requested lags are excluded as regression anchors but
are retained as possible future outcomes. These generated lags are ordinary,
time-invariant anchor-date regressors, so they use the cached design path.

The optional native demeaning backend is built with:

```bash
uv run maturin develop --manifest-path rust/Cargo.toml
```

Without Rust, the package uses its equivalent NumPy implementation.
The native backend prepares fixed-effect topology once per distinct horizon
sample, reuses it for the RHS and outcome transforms, and applies symmetric
Kaczmarz sweeps with reusable iteration buffers. The selected method and
whether the input used the already-sorted fast path are recorded in
`demeaning_diagnostics_`.
