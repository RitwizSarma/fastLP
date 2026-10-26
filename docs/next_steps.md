# fastLP Next Steps

## Priority plan: `sugar` terms and fit progress

**Status:** implemented in the current working tree; the specification and
acceptance criteria below remain the reference for follow-up work. This
feature must work for arbitrary panel data and specifications. The
flood-adaptation example motivates the API but must not be hard-coded into
term expansion, identification checks, or plotting. Preserve the existing
string-based API and numerical results when the new features are unused.

### Public API and meaning

Put the immutable term specifications and helpers in `src/fastlp/sugar.py`.
Expose the module from `fastlp` so one import statement is sufficient:

```python
from fastlp import LocalProjection, sugar as s

lp = LocalProjection(horizons=24, covariance="cluster")
lp.fit(
    data,
    outcome="y",
    shock=s.interact(
        "event", "treated",
        by=s.C("prior_exposure_group", levels=["0", "1", "2", "3+"]),
        name="event_by_experience",
    ),
    controls=[s.interact("x1", "x2", "x3", "x4", "x5")],
    shock_lags=6,
    unit="unit_id",
    time="period",
    fixed_effects=[s.fe("location", "period", "product"), "firm_fe"],
    cluster="location",
    progress=True,
)
lp.plot_irf("event_by_experience")
```

- `s.interact(*columns, name=None)` represents their numeric product and
  accepts any nonempty set of distinct factors; a one-factor form is useful
  with `by`. Validate the expanded feature count and memory requirement rather
  than impose an arbitrary interaction-order limit. An ordinary control
  interaction requests its product only, so a five-way control does not
  silently generate all 31 nonempty subsets. A shock interaction needs its
  identifiable lower-order terms as nuisance regressors; generate and dedupe
  those automatically. Document this context-dependent expansion clearly.
- `s.C(column, levels=None)` marks a categorical moderator. `levels` fixes
  category order and plot labels; validate missing, duplicate, and unobserved
  levels. `by=s.C(...)` makes one shock basis column and one IRF per category.
  Keep every category-specific shock coefficient, rather than choosing a
  reference category for the focal shock. Reference-code categorical nuisance
  terms to avoid a dummy trap. Guard against accidental very-high-cardinality
  expansions with a clear error and advice to group levels explicitly.
- `by="numeric_column"` specifies a continuous linear moderator. Include the
  unmoderated shock and its interaction with the moderator. Compute responses
  at explicitly requested values, for example
  `lp.plot_irf("named_shock", at=[0, 1, 3])`. This estimates a shared linear
  slope, unlike `by=s.C(...)`, which estimates separate level responses.
- `s.fe(*columns)` denotes **one** fixed-effect partition formed from the
  joint keys, including three-, four-, and five-way keys. It is not a numeric
  product. Existing fixed-effect strings remain valid, and users with
  precomputed integer FE IDs may continue passing those strings for speed.
- Keep `shock` as a string, term, or sequence of either; allow strings and
  terms together in `controls` and `fixed_effects`. Give generated columns
  deterministic, collision-checked names. A `name` labels a shock family for
  plotting; it does not replace unique internal coefficient names. Scalar
  `shock_lags` applies to every shock basis. In a lag mapping, a named shock
  family's key applies the same lags to all its bases; a named control term's
  key similarly identifies its completed product. Reject ambiguous or unknown
  keys. `shock_lags` must shift each **completed** shock basis column, so the
  lagged interaction and moderator refer to their values at the lagged date.

For a categorical moderator with shock factors `A` and `B` and categories
`D_k`, the focal columns are `A * B * D_k` for every `k`. Add the proper
lower-order terms needed to compare these paths, including category-specific
`A` and `B` effects where identifiable. Do not add a redundant unsplit
`A * B` alongside every `A * B * D_k`. For a continuous moderator `Z`, the
plotted response at `z` is the base-shock coefficient plus `z` times the
moderation coefficient; use the full horizon-specific covariance matrix for
its standard error and confidence interval. Do not interpret the interaction
coefficient alone as the response at `z`.

### Identification and fixed-effect absorption

Handle absorption from the *actual retained sample at each horizon*, without
special cases for particular column names or FE combinations:

1. Generate candidate shock, explicit-control, and automatically required
   lower-order columns. Track each column's source and whether it is a focal
   shock, an explicit user control, or generated nuisance term.
2. Apply the existing FE within transform and numerical within-signal check to
   candidate columns. Omit an absorbed **generated nuisance** term and record
   its name, reason, and affected horizon group. An absorbed requested shock
   basis column must raise an actionable identification error; never draw an
   IRF for it. Preserve the current error behavior for an absorbed explicit
   control. Deduplicate structurally identical generated terms first.
3. Calculate model rank, residual degrees of freedom, and clustered correction
   rank using the columns actually retained. The current `fit()` calculates
   these before demeaning, so its ordering must change. FE rank itself can be
   prepared earlier. Keep the numerical rank-deficiency check for dependencies
   that FE absorption alone does not explain; do not silently choose a
   different estimand by dropping an arbitrary jointly collinear regressor.
4. Keep a consistent coefficient layout across horizon cache groups. If a
   requested level is unsupported at any horizon, or generated-nuisance
   absorption differs across groups in a way that changes the model, report
   the affected horizons and terms. Do not silently change the model between
   horizons. Check category support after lag, lead, and singleton filtering.

Use reference coding for categorical nuisance terms and report level-specific
sample/support counts. Do not infer support from the unfiltered input. Preserve
the exact one-/two-way unit/time FE shortcut only when the normalized FE terms
really are the plain unit/time columns; grouped `s.fe(...)` terms use the
general factorization path.

### Performance and memory requirements

The syntax must not impose a performance penalty on fits that use only strings
and leave progress disabled. Resolve term specifications once, before the
existing preparation/lag/design path, and keep the parsed specification small.

- Expand source-column names and the **maximum** generated feature count
  before calling the existing memory planner. Include shock bases, requested
  lags, categorical nuisance columns, covariance/result matrices, and grouped
  FE workspaces. The budget must fail before expensive materialization when
  the planned minimum is too large. Preserve the planner's conservative
  estimator-owned-memory semantics.
- Select only needed source columns from pandas/Polars inputs. Keep Polars
  native; do not convert the entire input to pandas. Generate lagged shock
  basis columns on the full prepared panel because exact calendar lags need
  past rows. Generate nonlagged products on retained rows where practical.
  Avoid constructing a dense all-category indicator matrix or making repeated
  full-frame copies merely to name terms.
- Factorize joint FE keys directly with `PanelFrame.factorize(tuple_of_keys)`;
  both frame adapters already support this. Adapt `factorize_effects()` to
  pass each `s.fe(...)` key tuple as one FE dimension. Reuse FE codes and one
  residualizer across all RHS columns and horizon outcomes within a cache
  group. Do not split the native RHS transform into tiny calls for progress:
  the Rust path parallelizes across columns.
- Preserve horizon-mask grouping, the cached residualized design, and bounded
  outcome batches. Compute all paths from one fitted covariance matrix per
  horizon. Do not refit the LP once per category or requested `at` value.
  Do not shrink the numerical batch size merely to refresh progress.
- Benchmark time by preparation, term generation, FE encoding, RHS absorption,
  solve, and covariance. Record peak RSS and numerical equivalence to a
  manually constructed design on representative synthetic panels. Test both
  categorical and continuous moderation and the no-sugar/no-progress path.

The column count is a real cost, not a parser cost: `K` categorical levels
with `L` shock lags create `K * (L + 1)` focal/lag columns before nuisance
terms. Each `float64` design column costs 8 bytes per retained row per copy;
matrix work and covariance also grow with feature count. Offer level-count
validation and diagnostics rather than claiming large interactions are free.

### Progress reporting

Add `progress: bool = False` and an optional
`progress_callback: Callable[[ProgressEvent], None] | None = None` to `fit()`.
With `progress=True`, show concise stage names, elapsed time, and genuine
completed/total counts when available. Use a small internal reporter in
`src/fastlp/_progress.py`; avoid a mandatory terminal/notebook UI dependency.
Make `ProgressEvent` an immutable public type with stage, completed, total,
elapsed seconds, and message fields. Emit callback events on the fitting
thread at stage/batch boundaries; the built-in renderer alone may use a
background heartbeat during a long blocking call.

Instrument these stages in the existing code: input read/validation/sort;
interaction and lag generation; lead alignment and sample selection; FE
factorization/rank; RHS absorption and identification; solver preparation;
outcome/solve/covariance horizon batches; and result assembly. Report real
cache-group, column, or horizon-batch completion at boundaries. Panel sorting,
some NumPy operations, and the Rust absorption/covariance calls are opaque
while running: show the active stage plus a throttled elapsed-time heartbeat,
not an invented percentage. The current Rust calls release the GIL, allowing
a lightweight heartbeat while they run. Do not add per-row callbacks or locks
inside numerical loops.

Default-off reporting should amount to inexpensive conditional checks and no
timer thread. With progress enabled, throttle updates (for example every
5-10 seconds), use a display that works in terminals and notebooks, and clean
up the reporter in `try/finally` on success or failure. Do not interleave
background UI writes with user callbacks; dispatch structured completion
events on the fitting thread. Progress output must not change the fit result,
sample, batching, or numerical path.

### Implementation sequence and acceptance checks

1. Add `sugar.py` and export `sugar` in `__init__.py`. Unit-test validation,
   naming, deduplication, categorical basis construction, continuous bases,
   and products with one through five factors plus a higher-order case before
   integrating with `fit()`.
2. Extend `estimator.py`, `_frame.py`, `_demean.py`, and `_memory.py` in that
   order: parse raw dependencies; budget the expanded design; generate only
   needed columns; encode grouped FEs; absorb and classify terms; then compute
   rank and solve. Keep original plain-string results unchanged.
3. Store named shock-family metadata. Extend `plotting.py` and add a tidy
   `irf_frame()` for level/`at` responses, estimates, standard errors, and
   intervals. Keep `to_frame()` as the coefficient-level table for backward
   compatibility. Verify categorical direct-coefficient and continuous
   covariance-based intervals against manual calculations.
4. Add the progress reporter and fit hooks. Test silent default, ordered
   stages, truthful counts, throttling, exception cleanup, and unchanged
   estimates with progress on and off. Test both pandas and Polars inputs.
5. Document the API in README and quickstart. Add generic small-panel manual
   equivalence tests for grouped FEs, lags of completed interactions, absorbed
   lower-order terms, unsupported shock levels, and `sample="per_horizon"`.
   Run the focused suite, then the existing suite, and benchmark opt-in larger
   cases without adding brittle wall-clock assertions to unit tests.

## Basic and obvious features

1. `numpy` arrays as input. Duh.
2. Validate that the `time` column contains finite integer-valued period
   indices. Its scale must advance by one for each underlying period because
   horizon `h` is defined by matching `time + h`. Gaps may remain in an
   unbalanced panel, but labels such as quarterly decimals must be rejected in
   favor of a sequential period index.

## Common-sample attrition diagnostics

Using one common sample across horizons enables fastLP's shared-design
optimization and makes horizon estimates directly comparable, but the sample
is selected by having valid outcomes through the maximum horizon. In an
unbalanced panel or with nonrandom missingness, that restriction may leave a
systematically different estimation population and therefore change the
estimand. Descriptive balance checks cannot establish or rule out selection
bias, but they should make potentially important sample changes visible.

Whenever `sample="common"` is used, report diagnostics comparing the original
eligible sample with the retained common sample. At minimum, show:

- observations, units, and periods retained, with counts and percentages;
- retention rates by unit and time period;
- means, standard deviations, medians, and selected quantiles for the outcome,
  shock, and controls in both samples; and
- differences in means and standardized mean differences, with a warning when
  attrition is substantial or covariate balance changes materially.

Store the full comparison in fitted-result metadata and expose a tidy table for
programmatic use. The ordinary fit summary should display a compact retention
and balance summary so users do not have to opt into discovering that the
common-sample restriction materially changed their data.

## Inference: broaden error calculations without narrowing LP specifications

The estimator now provides classical, HC0--HC3, one- through four-way CR0/CR1
clustered, within-unit Newey--West HAC, and Driscoll--Kraay covariance, all
with normal critical values. LP users still need a broader inference menu over
time, because few clusters and nonstandard sampling distributions are common.

Possible additions are:

1. **Bootstrap and leverage-adjusted clustering.** Add CR2/CR3, wild-cluster,
   and bootstrap procedures for applications with few clusters or nonstandard
   sampling distributions.
2. **Combined dependence models.** Define and validate any future
   cluster-HAC estimator explicitly; the current HAC and Driscoll--Kraay paths
   intentionally do not combine arbitrary cluster terms.
3. **Small-sample critical values.** Make the interval distribution explicit:
   support normal and Student-*t* critical values, use residual degrees of
   freedom for non-clustered estimates, and provide a cluster-based degrees-of-
   freedom convention for clustered estimates. Record the chosen critical value,
   degrees of freedom, and correction in result metadata.

All covariance choices should be available independently of the sample policy.
They may use different cache paths internally, but selecting HAC, multiway
clustering, or a small-sample critical value must never change the requested LP
regressors, lags, fixed effects, or horizon sample without reporting it.

### Cumulative responses are not multipliers

When fastLP gains cumulative or IV specifications, expose the estimand
explicitly rather than using one ambiguous boolean:

- **Level response:** regress \(y_{t+h}\) on the contemporaneous shock.
- **Cumulative response:** regress \(\sum_{j=0}^{h} y_{t+j}\) on the
  contemporaneous shock. This is a cumulative impulse response, not a fiscal
  or integral multiplier.
- **Integral multiplier:** regress cumulative outcome on cumulative impulse;
  with an endogenous impulse, estimate this as one LP-IV regression. Its
  reported standard error must be for that coefficient itself, not a ratio of
  separately estimated IRFs or a cumulative sum of level-response standard
  errors.

Use named modes (for example, `cumulation="none"`, `"outcome"`, and
`"both"`) and separate result labels. For LP-IV, return a horizon-specific
weak-instrument diagnostic with the result. This makes a common substantive
mistake—calling a cumulative response a multiplier—hard to express in the API.

### Validation additions

Add an external-reference and statistical-validation layer alongside the
existing slow-equivalence tests:

1. Create versioned golden fixtures from independent implementations:
   statsmodels for OLS/HAC LPs and linearmodels for LP-IV. Pin point estimates,
   standard errors, effective sample sizes, covariance settings, and bandwidth
   conventions with documented numerical tolerances.
2. Add deterministic Monte Carlo property tests for coverage, especially under
   persistent and near-unit-root data-generating processes. Compare the
   coverage of any lag-augmented HC1 option with HAC rather than validating
   only point estimates.
3. Test cumulative estimands directly: cumulative-response point estimates may
   approximately track sums of level IRFs, but their standard errors must come
   from the cumulative regression. Test multiplier identities using the
   one-step estimator, including weak-instrument warnings.
4. Keep the fastLP-specific reference suite for fixed effects, clustering,
   unbalanced panels, and common versus per-horizon samples; tsecon's
   single-series tests do not cover those cases.

### Provenance

These additions were prompted by a review of the vendored
[`others/tsecon`](../others/tsecon) Rust implementation, especially
`crates/tsecon-lp/src/spec.rs`, `level.rs`, and its golden/property tests.
Its API distinguishes cumulative IRFs from integral multipliers, and its test
fixtures compare HAC OLS to statsmodels and LP-IV to linearmodels. The
lag-augmented inference idea follows Montiel Olea and Plagborg-Møller (2021);
the cumulative-multiplier distinction follows Ramey and Zubairy (2018). These
are ideas to adapt to fastLP's panel setting, not a proposal to replace the
shared-design panel engine with tsecon's per-horizon single-series engine.

## Current Assessment

The v0.1 algorithm has the correct high-level optimization for the balanced,
common-sample setting: residualize the RHS once, factor its Gram matrix once,
and solve every horizon against that cached design. This avoids the principal
redundancy in a naive local-projection loop.

The current implementation is nevertheless an early performance baseline, not
yet a production-scale general HDFE engine. It now has a specialized balanced-
panel path with arithmetic lead construction, exact unit/time demeaning,
bounded horizon batches, and optional residual retention. Arbitrary unbalanced
multiway FE problems still need substantial work before claiming parity with
mature implementations such as `fixest` or `reghdfe`.

## Memory Management: Current Rating 6/10

### What the implementation does well

- Reuses factorized FE codes and the residualized RHS within a fit.
- Stores only small coefficient and covariance outputs after fitting.
- Uses `float64` arrays consistently for predictable numerical behavior.

### Current costs

The fit path creates several full-size representations of the data:

1. a copy of selected DataFrame columns;
2. a sorted DataFrame;
3. the common-anchor DataFrame;
4. raw RHS matrix `X`;
5. horizon outcome matrix `Y`;
6. transformed outcome matrix `Y_tilde`;
7. residual matrix; and
8. FE-code matrices and Rust working/output buffers.

The Rust kernels still own separate input/output column buffers and pandas
preparation can create large copies. Outcome work is now batched under an
optional memory budget, and residual retention can be disabled, but input
streaming and memory mapping are not implemented.

For the notebook's 10 million-row configuration with fixed effects and two
horizons, peak memory will likely be several GiB. Exact usage depends on pandas,
NumPy, allocator behavior, and the number of active Rayon/BLAS workers; it has
not yet been measured. A machine with less than roughly 8-16 GiB of free memory
may struggle.

### Priority memory work

1. **Remove avoidable pandas copies.** Validate already-sorted data where
   possible; use array views/contiguous arrays after one controlled conversion
   rather than keeping multiple DataFrame copies alive.
2. **Measure production-scale peak RSS.** The harness records process peak RSS;
   run it for every large benchmark case,
   including outcome construction, demeaning, linear algebra, and covariance.
3. **Stream input arrays.** Avoid the full future-position matrix on general
   unbalanced panels and consider Arrow-style or memory-mapped inputs.
4. **Consider memory-mapped outputs.** For very large horizon grids, allow
   temporary outcome/residual batches to be backed by disk rather than RAM.

## Processor Efficiency: Current Rating 5/10 Overall

The core cached solve is approximately 7/10: it performs the right reuse and
uses BLAS-backed matrix products. End-to-end efficiency is lower because FE
residualization, DataFrame preparation, and covariance construction dominate
large multi-FE workloads.

### What the implementation does well

- Residualizes `X` once, then reuses it for every horizon.
- Computes all cached-design cross-products with matrix multiplication.
- Uses one Cholesky factorization for the shared design.
- Releases the GIL in the Rust residualizer and parallelizes across columns.
- Reuses FE and cluster integer codes and the covariance bread matrix.

### Current bottlenecks

- Unbalanced multi-way FEs use guarded Irons--Tuck-accelerated alternating
  projections. Balanced unit/time FEs and arbitrary one-way FEs use exact
  transforms, and singleton observations are recursively pruned. There is still
  no graph reduction, Krylov solver, or preconditioner.
- The Rust kernel reconstructs FE code vectors for each residualization call.
- It allocates a previous-value vector and group sums/counts during each
  iteration, rather than reusing thread-local buffers.
- Column-level Rayon parallelism is limited when few horizons are estimated.
- Cluster covariance loops over horizon and feature pairs, making repeated
  `np.bincount` calls from Python.
- Sorting and pandas materialization are substantial costs at 10 million rows.
- Rayon workers and multithreaded BLAS can oversubscribe CPU cores unless their
  thread counts are coordinated.

### Priority processor work

1. **Profile before further algorithm changes.** Break benchmark time into
   preparation, FE encoding, demeaning, factorization, point estimation, and
   covariance stages.
2. **Reuse residualizer scratch.** Keep FE codes, denominators, group-sum
   buffers, and previous-value buffers in a prepared native residualizer.
3. **Batch and parallelize output work.** Process horizon batches in parallel
   only when this does not compete with BLAS; expose a single thread-control
   policy for Rayon and BLAS.
4. **Move cluster scores native.** Implement grouped score accumulation in Rust
   and parallelize independent horizon score calculations.
5. **Evaluate stronger HDFE algorithms.** Benchmark guarded Irons--Tuck and
   Aitken, and eventually evaluate LSMR/LSQR or graph-based approaches against
   unaccelerated symmetric MAP.

## Benchmark Plan

The existing synthetic notebook provides a small case and an opt-in 10 million
observation case. Extend it into a reproducible benchmark harness that records:

- wall-clock time per pipeline stage;
- peak resident memory;
- CPU utilization and active Rayon/BLAS thread counts;
- dimensions: observations, units, periods, regressors, FE dimensions, and
  horizons;
- covariance type; and
- numerical agreement with the slow reference implementation on smaller cases.

Benchmark against a naive independent-horizon loop first. When an equivalent
specification is available, add a mature HDFE baseline. Report the break-even
number of horizons, because cache construction is not always worthwhile for
small `H` or designs with tiny RHS matrices.

## Recommended Implementation Order

1. Add timing and peak-RSS instrumentation to the benchmark harness.
2. Refactor fitting around a prepared design/residualizer and horizon batches.
3. Rework the Rust MAP kernel to retain codes and reusable scratch buffers.
4. Move one-way clustered covariance score accumulation to the native backend.
5. Benchmark thread policies and prevent BLAS/Rayon oversubscription.
6. Add algorithmic HDFE improvements only after profiling identifies demeaning
   as the dominant remaining cost.

These changes preserve the v0.1 estimator's key invariant: reuse must occur
only when every horizon shares the same validated transformed RHS design.
