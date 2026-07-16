# fastLP Next Steps

## Current Assessment

The v0.1 algorithm has the correct high-level optimization for the balanced,
common-sample setting: residualize the RHS once, factor its Gram matrix once,
and solve every horizon against that cached design. This avoids the principal
redundancy in a naive local-projection loop.

The current implementation is nevertheless an early performance baseline, not
yet a production-scale HDFE engine. Its point-estimation cache is useful, but
its memory behavior and fixed-effect residualizer need substantial work before
claiming parity with mature implementations such as `fixest` or `reghdfe`.

## Memory Management: Current Rating 3/10

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

The Rust kernel currently owns a separate column buffer and a separate output
matrix. It does not residualize user-provided arrays in place. The Python engine
also materializes every requested horizon at once. There is no outcome batching,
streaming, memory mapping, or reusable scratch allocation.

For the notebook's 10 million-row configuration with fixed effects and two
horizons, peak memory will likely be several GiB. Exact usage depends on pandas,
NumPy, allocator behavior, and the number of active Rayon/BLAS workers; it has
not yet been measured. A machine with less than roughly 8-16 GiB of free memory
may struggle.

### Priority memory work

1. **Batch horizons.** Build, residualize, solve, and release a bounded group
   of outcome horizons at a time. Retain only coefficient, covariance, and
   diagnostic outputs.
2. **Remove avoidable pandas copies.** Validate already-sorted data where
   possible; use array views/contiguous arrays after one controlled conversion
   rather than keeping multiple DataFrame copies alive.
3. **Prepare fixed effects once.** Introduce an internal residualizer object
   that owns immutable codes/group counts and reuses per-worker scratch buffers.
4. **Measure peak RSS.** Record peak resident memory for every benchmark case,
   including outcome construction, demeaning, linear algebra, and covariance.
5. **Consider memory-mapped outputs.** For very large horizon grids, allow
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

- Multi-way FEs use unaccelerated alternating projections. There is no graph
  reduction, singleton pruning, warm start, Krylov solver, or preconditioner.
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
5. **Evaluate stronger HDFE algorithms.** Benchmark accelerated symmetric MAP,
   warm starts, singleton removal, and eventually LSMR/LSQR or graph-based
   approaches against the existing kernel.

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
