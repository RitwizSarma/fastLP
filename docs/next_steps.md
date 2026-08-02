# fastLP Next Steps

## Inference: broaden error calculations without narrowing LP specifications

The estimator now provides classical, HC0--HC3, one- through four-way CR0/CR1
clustered, within-unit Newey--West HAC, and Driscoll--Kraay covariance, all
with normal critical values. LP users still need a broader inference menu over
time, because few clusters and nonstandard sampling distributions are common.

Priority additions are:

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
