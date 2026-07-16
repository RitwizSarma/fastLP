# fastLP Implementation Plan

## 1. Objective

Build `fastlp`, a Python library for estimating linear local projections (LPs)
efficiently when horizon regressions share a common right-hand-side design and
fixed-effect (FE) structure. The library must return the same point estimates
and supported covariance estimates as estimating the equivalent OLS/FWL
regression separately at each horizon, within documented numerical tolerances.

The performance strategy is to turn the usual sequence of horizon regressions
into a multiple-right-hand-side least-squares problem:

\[
Y = [y_0, \ldots, y_H], \qquad
\hat B = (\widetilde X'\widetilde X)^{-1}\widetilde X'\widetilde Y.
\]

For a valid shared design, the library will:

1. construct and validate the estimation sample once;
2. encode fixed effects, clusters, and regressors once;
3. residualize `X` once with respect to the FEs;
4. factor `X_tilde.T @ X_tilde` once;
5. residualize the matrix of horizon outcomes in batches;
6. solve all horizon coefficient vectors from the cached factorization; and
7. compute horizon-specific residuals and inference using cached design-side
   objects where possible.

The first release should prioritize correctness, transparent assumptions, and
strong equivalence tests. Optimization beyond the shared-design algorithm is a
later concern.

## 2. Econometric Contract and Scope

### Initial supported model

For observation `t` and horizon `h`, estimate:

\[
y_{t+h} = a_h + b_h s_t + C_h' w_t + F_t' d_h + u_{t+h},
\]

where `s_t` is one or more shocks, `w_t` are contemporaneous controls and
lags, and `F_t` represents one or more absorbed fixed effects. The primary
output is the coefficient path for selected shock terms across horizons.

Version 0.1 should support:

- Linear, balanced-panel or time-series LPs indexed by an explicit time column.
- A single outcome, one or more shocks, controls, and deterministic terms.
- Horizons `0..H`.
- No FEs, one FE, and multi-way categorical FEs.
- Homoskedastic and HC1 covariance estimators.
- One-way cluster-robust covariance with an explicit finite-sample correction.
- A common estimation sample across every horizon.
- Data supplied as pandas `DataFrame`; internally operate on contiguous NumPy
  arrays.

The public result must include coefficients, standard errors, confidence
intervals, residual degrees of freedom, observation counts, horizon labels,
sample diagnostics, convergence diagnostics, and enough metadata to reproduce
the fitted specification.

### Explicit non-goals for the first release

- Nonlinear/state-dependent LPs, IV LPs, quantile LPs, and panel GMM.
- Two-way or multi-way clustered covariance.
- Driscoll-Kraay/HAC/Newey-West inference, wild bootstrap, and simultaneous
  confidence bands.
- Unbalanced horizon-specific samples under the accelerated shared-plan path.
- Recovery or reporting of individual FE coefficients.
- Full formula-language compatibility with R or Stata packages.

These are valuable extensions but should not delay a verifiable core.

## 3. Resolve the Key Rebuttals Before Coding

The source documents correctly identify reusable computation, but their
"only the dependent variable changes" premise is conditional. The library
design must make those conditions explicit rather than silently applying an
invalid cache.

### 3.1 Horizon samples often differ

With the usual LP construction, observations near the end of a time series do
not have `y[t+h]`. Missing outcomes can also differ by horizon. Then both the
rows of `X` and the FE/cluster memberships can change, so `X_tilde.T @ X_tilde`
cannot be reused unchanged.

**Decision:** default to `sample="common"`: retain only anchor observations
with valid outcomes for every requested horizon and valid RHS/FE/cluster data.
This makes the cache mathematically valid and makes estimates comparable across
horizons. Offer `sample="per_horizon"` only as a correctness-first fallback;
it creates a separate cached plan for each distinct row mask and must not claim
the full shared-design speedup.

### 3.2 Controls and fixed effects must be time-aligned consistently

The estimator reuses a design only if controls, lags, trends, interactions, FE
variables, weights, and clustering assignments are all defined on the same
anchor rows at every horizon. Horizon-varying controls or FEs imply separate
plans.

**Decision:** expose a fixed `DesignSpec` built from anchor-time columns. Reject
horizon-varying design inputs in `fit_lp`; add a future grouped-plan API for
users who deliberately need them.

### 3.3 Singleton handling can change the valid sample

Iterative singleton removal is part of many HDFE implementations. If it is
performed separately for each horizon, it can produce different samples and
invalidate a shared residualizer.

**Decision:** in common-sample mode, prune singleton observations once from the
common sample until stable, then use that same retained mask for all horizons.
Record dropped counts and group counts. In per-horizon mode, prune separately
inside each plan.

### 3.4 Alternating demeaning is approximate

Multi-way alternating projections need tolerances and iteration limits;
different convergence states can contaminate comparisons and covariance
calculations.

**Decision:** use a documented convergence criterion based on the largest
absolute update relative to vector scale, expose `atol`, `rtol`, and
`max_iter`, and fail clearly when convergence is not reached. Test the
residualized output against an explicit dummy-variable projection on small
problems.

### 3.5 A matrix inverse should not be cached

The documents use `(X'X)^(-1)` for exposition, but explicitly forming an
inverse is less stable and usually slower than solving a factored system.

**Decision:** cache a rank-revealing QR factorization of `X_tilde` initially,
or a Cholesky factorization of its Gram matrix only after a rank/conditioning
check. Use triangular solves, never a stored inverse. Define and test a clear
policy for collinear columns: raise by default; an opt-in rank-revealing mode
may drop and report aliased columns later.

### 3.6 Clustered inference still has substantial horizon work

The bread matrix is reusable, but the cluster score meat depends on each
horizon's residual vector. Caching cannot eliminate those residual-dependent
operations.

**Decision:** cache cluster integer codes, observation ordering by cluster, the
design factorization, and the bread. Compute residuals and cluster score sums
per horizon or in bounded batches. Benchmark point estimation and covariance
separately.

### 3.7 Intercepts are absorbed by fixed effects

An intercept is collinear with any complete categorical FE set after the
within transformation.

**Decision:** automatically omit the intercept when FEs are absorbed and
report this in the result metadata. Without FEs, include it by default unless
disabled.

## 4. Public API and Data Model

Keep the first API small and typed. The exact names can evolve, but its
separation between specification, prepared plan, fitting, and results should
remain.

```python
from fastlp import LPDesign, LPOptions, prepare_lp, fit_lp

design = LPDesign(
    outcome="log_output",
    time="quarter",
    entity=None,
    shocks=["monetary_shock"],
    controls=["inflation_l1", "output_l1", "output_l2"],
    fixed_effects=["country", "quarter"],
    cluster="country",
)

options = LPOptions(
    horizons=range(0, 21),
    sample="common",
    covariance="cluster",
    singleton_policy="drop",
    demean_rtol=1e-10,
)

plan = prepare_lp(data, design, options)
result = fit_lp(plan)
result.tidy(coefficient="monetary_shock")
```

### Proposed objects

| Object | Responsibility |
| --- | --- |
| `LPDesign` | Immutable column names and model structure; validates no duplicate or conflicting terms. |
| `LPOptions` | Horizons, sample policy, demeaning controls, covariance choice, batching, and numerical policy. |
| `PreparedLP` | Internal immutable cache: retained rows, outcome matrix, encoded groups, `X_tilde`, factorization, and diagnostics. |
| `LPResult` | Per-horizon estimates and inference with plotting- and table-friendly methods. |
| `fit_lp` | Convenience wrapper that calls preparation then fitting. |

Use a narrow column-name API first. Formula parsing, Polars input, and a richer
design-matrix language can be added after the numerical API is stable.

## 5. Package Architecture

Create the source layout below.

```text
fastlp/
  __init__.py
  api.py                 # public functions and re-exports
  specification.py       # frozen public dataclasses and validation
  sample.py              # time alignment, common/per-horizon masks, singleton pruning
  encoding.py            # stable factorization of FE and cluster labels
  design.py              # RHS matrix construction and column metadata
  residualize.py         # one- and multi-way within transformations
  linalg.py              # factorization, rank checks, batched solves
  covariance.py          # homoskedastic, HC1, one-way cluster covariance
  engine.py              # PreparedLP construction and block fitting
  results.py             # result containers, tidy export, intervals
  exceptions.py          # focused domain errors
tests/
  unit/
  integration/
  reference/
  performance/
docs/
```

Use `src/fastlp/` instead if packaging is configured from scratch; choose one
layout before writing implementation code and configure `pyproject.toml`
accordingly. Keep compiled kernels behind small array-only interfaces so their
behavior can be tested independently of pandas.

## 6. Core Algorithms

### 6.1 Sample construction and horizon outcome matrix

1. Validate that time is unique within entity when an entity key is supplied;
   sort deterministically by entity and time without mutating caller data.
2. Construct `y_h` by an entity-safe forward shift of the outcome, not by raw
   array shifting across panel boundaries.
3. Build the common mask from every required outcome, RHS term, FE, and cluster
   variable. Reject infinite numeric values; define the missing-data policy
   explicitly.
4. In common mode, apply recursive singleton pruning after the mask is built.
5. Produce contiguous `float64` arrays `X: (n, k)` and `Y: (n, H + 1)` and
   integer factor codes for each categorical variable.
6. Preserve original row identifiers and report every exclusion reason.

### 6.2 Fixed-effect residualization

For one FE, transform each numeric column by subtracting its group mean. For
multiple FEs, apply cyclic alternating projections to a two-dimensional block
`A = [X, Y_batch]`:

```text
repeat until converged:
    for FE codes in a deterministic order:
        subtract each group mean from every column of A
```

Block residualizing `Y` amortizes group traversal and avoids repeated Python
dispatch. `X` is residualized exactly once and cached. The initial
implementation should use NumPy group reductions with integer codes; measure
before adding Numba. Do not build dense dummy or projection matrices.

Implementation details:

- Encode groups with dense zero-based integer codes and maintain group counts.
- Use `np.bincount`/indexed reductions or a tested sorted-segment kernel.
- Perform calculations in `float64`; require finite results after every pass.
- Measure convergence on the whole block and retain iteration count and final
  error for each residualization call.
- Add a residual-orthogonality diagnostic: group sums of transformed columns
  must be near zero for every absorbed FE.

### 6.3 Cached least-squares solve

After obtaining `X_tilde`:

1. Check `n > k` and diagnose numerical rank using pivoted QR or SVD.
2. For a full-rank design, cache `G = X_tilde.T @ X_tilde` and its Cholesky
   factor `L`; cache `bread = solve(G, I)` only when covariance calculation
   needs it and memory cost is acceptable.
3. For a residualized batch `Y_tilde_batch`, compute
   `rhs = X_tilde.T @ Y_tilde_batch` with BLAS matrix multiplication.
4. Obtain all coefficients through two triangular solves.
5. Compute `E_batch = Y_tilde_batch - X_tilde @ B_batch` using matrix
   multiplication.

Use batch size as a memory control, not a public econometric setting. A batch
of all horizons is preferred when it fits comfortably in memory.

### 6.4 Covariance estimators

Implement each estimator against the residualized design and use the same
coefficient-column ordering for every horizon.

- **Classical:** `s2_h * G^{-1}`, with `s2_h = e_h.T @ e_h / (n - rank)`.
- **HC1:** compute `X_tilde.T @ diag(e_h**2) @ X_tilde` in blocks; apply
  `n / (n - rank)`.
- **One-way cluster:** for each horizon, calculate cluster scores
  `s_g = sum_{i in g} x_i * e_i`, then form `sum_g s_g @ s_g.T`. Apply a
  documented finite-sample correction, initially
  `(G_clusters / (G_clusters - 1)) * ((n - 1) / (n - rank))`, and reject fewer
  than two clusters.

For each covariance matrix, symmetrize numerical noise, inspect diagonal signs,
and return standard errors only when the selected coefficient variance is
nonnegative within tolerance. The result should retain full covariance matrices
initially; add a `store_covariance=False` memory option later.

## 7. Correctness and Reference Strategy

Equivalence testing is the primary acceptance criterion. Build a deliberately
slow reference implementation for tests only:

1. Construct a dense dummy design for small samples, drop one level per FE or
   use a numerically stable least-squares routine.
2. Fit each horizon independently using exactly the same common-sample mask.
3. Compare its coefficients, residuals, fitted values, residual degrees of
   freedom, and supported covariance matrices to `fastlp`.

Test matrix:

| Area | Required cases |
| --- | --- |
| Time alignment | time series, balanced panel, unsorted input, missing outcome, panel boundary, `H=0` |
| Design | intercept/no intercept, multiple shocks, controls, collinear columns, `n <= k` |
| FEs | none, one-way, two-way, three-way small reference case, disconnected groups, recursive singletons |
| Inference | classical, HC1, one-way clusters, few-cluster error, cluster with FE overlap |
| Numerics | scale differences, near collinearity, nonconvergence, missing/infinite data |
| API | immutable options, reproducible sorting, tidy output schema, metadata completeness |

Property tests should generate small random panels and compare every supported
configuration to the dummy-variable reference. Add regression fixtures from
known analytical examples so failures are readable.

For external validation, compare point estimates and standard errors on frozen
datasets with a trusted implementation such as `pyfixest` or `statsmodels`
using an explicitly equivalent design and sample. Pin versions in the test
environment and document any differences in degrees-of-freedom conventions.

## 8. Performance Plan

Benchmark against two baselines:

1. a naïve NumPy loop that reconstructs and solves each horizon; and
2. repeated calls to a mature fixed-effects estimator where a comparable setup
   is available.

Benchmark scenarios must separately vary observations, number of horizons,
number of regressors, FE cardinality, number of FE dimensions, and covariance
choice. Include a no-FE case to isolate repeated linear algebra from
residualization. Record wall time, peak memory, cache construction time,
per-batch fit time, and agreement with the reference.

Expected complexity in common-sample mode, omitting iteration counts, is:

- preprocess and residualize RHS once: approximately `O(n * k * D * I)`;
- residualize outcomes in blocks: approximately `O(n * (H + 1) * D * I)`;
- cross-products and solves: `O(n * k * (H + 1) + k^2 * (H + 1))` after one
  `O(n * k^2 + k^3)` design factorization;
- covariance: additional horizon-dependent work, especially for clustering.

Here `D` is the number of FE dimensions and `I` the demeaning iteration count.
The plan must not promise speedups where `H` is very small, `k` is tiny, or
cluster covariance dominates runtime.

Only optimize a measured bottleneck. Likely sequence: vectorized NumPy;
Numba-compiled grouped demeaning and cluster-score kernels; optional Rust/C++;
parallelize independent output batches only after deterministic single-threaded
results and memory behavior are established. Keep parallelism opt-in and avoid
BLAS-thread oversubscription.

## 9. Delivery Milestones

### M0: Project foundation

- Create package layout, strict type checking/linting configuration, pytest,
  and a `uv`-managed development workflow.
- Add NumPy/pandas as core dependencies; add `scipy` for stable factorization
  utilities if its functionality is not implemented directly. Add optional
  benchmark/test dependencies in a development group.
- Establish CI for Python 3.12+, unit tests, formatting, and documentation
  link checks.

**Exit criterion:** an installable package with a passing minimal test suite.

### M1: No-FE shared-design LP

- Implement specification validation, entity-safe horizon alignment, common
  sample construction, `X`/`Y` creation, factorized batched OLS, and results.
- Implement classical covariance and coefficient-path tidy output.
- Add exact/reference tests for no-FE time series and panels.

**Exit criterion:** coefficients and classical standard errors match separate
OLS horizon fits; the common-sample path demonstrably reuses one factorization.

### M2: Fixed-effect preprocessing and residualization

- Implement categorical encoding, one-way demeaning, multi-way alternating
  projections, convergence reporting, and common-sample singleton pruning.
- Cache the residualized RHS and factorization in `PreparedLP`.
- Add dense-dummy FWL equivalence tests for small multi-way FE datasets.

**Exit criterion:** transformed variables satisfy orthogonality diagnostics and
all supported point estimates match the dense reference within tolerance.

### M3: Robust inference

- Implement HC1 and one-way cluster covariance with tested finite-sample
  corrections.
- Add residual, covariance, degrees-of-freedom, and few-cluster diagnostics.

**Exit criterion:** supported covariance matrices match independent reference
calculations on frozen cases.

### M4: Usability and observability

- Add `fit_lp` convenience function, result summaries, tidy pandas output,
  coefficient selection, confidence intervals, and basic impulse-response
  plotting data (not necessarily a plotting dependency).
- Add clear errors for invalid cache assumptions and rich preparation/fitting
  diagnostics.
- Write tutorial notebooks or examples covering time series, panel FEs, and
  clustered inference.

**Exit criterion:** a user can reproduce each documented example without
accessing internal arrays.

### M5: Benchmarking and optimization

- Add reproducible benchmark harness and publish scenario results.
- Profile common large-panel configurations; optimize only demonstrated hot
  paths while preserving the reference suite.
- Consider an optional compiled group-reduction backend only when it provides a
  material, reproducible improvement.

**Exit criterion:** benchmark report shows where the implementation wins and
where it intentionally does not; all equivalence tests still pass.

### M6: Controlled extensions

- Add a grouped-cache implementation for `sample="per_horizon"`, keyed by the
  exact retained-row mask and validated design/FE/cluster encoding.
- Add multi-way clustering, HAC options, weights, formula support, and optional
  Polars input one at a time, each with its own reference tests and benchmark.

**Exit criterion:** every extension preserves the invariant that reuse occurs
only for mathematically identical transformed designs.

## 10. Documentation Requirements

The user-facing documentation must state that common-sample estimation is the
default because it enables valid reuse and changes the estimand relative to
separate horizon-specific samples. It must show how to inspect retained rows,
dropped observations, FE convergence, rank, and covariance corrections.

Document these implementation guarantees:

- no dense FE dummy matrix or FE projector is constructed in production;
- no normal-equation inverse is formed;
- all reused objects are tied to a validated sample/design fingerprint;
- output preserves original coefficient names and horizon labels;
- numerical tolerances, rank policy, singleton policy, and covariance
  corrections are exposed in metadata.

Also document limitations plainly: alternating projections may converge slowly
for difficult FE graphs, cluster-robust inference can be unreliable with few
clusters, and point-estimate acceleration does not make residual-dependent
inference free.

## 11. Definition of Done for the First Stable Release

The first stable release is complete when it:

- fits common-sample linear LPs with zero to multiple categorical FEs;
- returns reproducible coefficients and supported standard errors for every
  requested horizon;
- matches a per-horizon dense-dummy/OLS reference across the test matrix;
- rejects or routes configurations that violate shared-design assumptions;
- exposes enough diagnostics to audit sample construction and numerical
  convergence;
- has benchmark evidence for its claimed reuse benefits; and
- is installable and tested through the repository's `uv` workflow.

This sequence delivers the novel estimator as a reliable cached estimation
plan, rather than as a loop around regressions with an unverified performance
claim.
