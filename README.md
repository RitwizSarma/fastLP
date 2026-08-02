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

Every fitted model exposes a lazily created default IRF plot:

```python
ax = lp.irfplot
ax.figure.savefig("irf.pdf", bbox_inches="tight")

# Customize the plot or select another shock in a multi-shock model.
ax = lp.plot_irf(
    "monetary_shock",
    title="Output response to a monetary shock",
    ylabel="Log points",
    show=True,
)
```

The plot uses a white publication-style background, a light-blue pointwise
confidence band, a zero reference line, and the confidence level configured on
the estimator. `lp.irfplot` caches the default Matplotlib axes; `plot_irf()`
creates a new customizable axes and also accepts an existing `ax`.

Set `response="cumulative"` to regress the cumulative outcome
`y[t] + ... + y[t+h]` at each horizon. This is a cumulative response, not an
integral multiplier. Its standard errors are calculated from the cumulative
regression itself. On a balanced panel, cumulative outcomes use within-unit
prefix sums rather than repeated summation.

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

Clustered fits warn when any requested cluster term has fewer than 50 groups.
This is a diagnostic threshold, not a validity cutoff: CR1 does not remove the
few-cluster problem. Configure it with `few_cluster_threshold=30`, or set it to
`None` to disable the warning. The exported `FewClustersWarning` class allows
applications to filter or promote the warning explicitly.

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

Balanced panels with unit fixed effects, time fixed effects, or both use an
exact dense-panel within transformation. The two-way transform is
`x - unit_mean - time_mean + grand_mean`; it avoids iterative convergence and
uses the Rust backend when installed. Exact unit-spaced numeric panels also use
arithmetic lead positions instead of horizon-by-horizon index joins. Other FE
structures and unbalanced samples retain the general mask-grouped alternating-
projection path.

The general multiway-FE residualizer uses guarded Irons--Tuck acceleration by
default. Every proposed extrapolation is accepted only when it is finite and
reduces the largest remaining absolute FE-group mean; otherwise ordinary
Kaczmarz continues. Set `demean_acceleration="aitken"` to use the simpler
vector Aitken method or `demean_acceleration="none"` for the unaccelerated
reference path. Diagnostics report the selected method and accepted
acceleration steps. Exact balanced and one-way transforms bypass iteration and
therefore do not use acceleration.

Fixed-effect singletons are recursively removed by default before a sample's
absorber and factorization are prepared. Removing a singleton in one FE can
create a new singleton in another, so pruning continues until every retained
FE level has at least two observations. In common-sample mode the shared mask
is pruned once; in per-horizon mode each distinct mask is pruned independently
and horizons are regrouped afterward. Set `singleton_policy="keep"` to retain
them. `singleton_diagnostics_` reports the number of dropped rows and pruning
rounds at every horizon.

Use `memory_budget="4GB"` (or an integer byte count) to cap the approximate
working space used for raw, transformed, and residualized horizon outcomes.
The estimator reduces its batch size accordingly. Set
`retain_residuals=False` when only estimates and inference are needed; the
fitted object then exposes `residuals_ = None` and avoids retaining an
observation-by-horizon result.
