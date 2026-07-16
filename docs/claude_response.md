# Response: Caching Cross-Products in Local Projections

The original document's central claim is correct: naive local projection (LP) code
re-derives quantities that are, in the textbook case, invariant across horizons. But
the textbook case is a special case. Three features of how LPs are actually estimated
in applied work — unbalanced samples across horizons, HAC-robust inference, and
horizon-varying regressors — break the "compute (X'X)^{-1} once" argument as literally
stated. Below: where it breaks, how to fix it without giving up most of the speed gain,
and how to test empirically whether the fix is worth building.

---

## 1. Where the thesis holds without modification

If every horizon uses (a) the exact same rows of data, (b) the exact same regressor
matrix X, and (c) simple heteroskedasticity- or cluster-robust (not HAC) standard
errors, the caching argument goes through exactly as described: `(X'X)^{-1}` is
computed once, only `X'y_h` varies. This case is common enough to be worth
implementing on its own — but it's narrower than "local projections" as usually
estimated.

---

## 2. Three barriers to generality

### 2.1 Sample imbalance across horizons

LPs regress `y_{t+h}` on information dated `t`. Near the end of the sample,
`y_{t+h}` doesn't exist once `t + h` exceeds the last observed period. So the
usable sample **shrinks as h grows** — a panel with T periods per unit loses h
trailing periods per unit at horizon h.

This matters because fixed-effect demeaning is sample-dependent: group means are
computed only over the rows present. If the horizon-0 regression uses T periods per
unit and the horizon-5 regression uses T-5, the group means differ, so `X̃`
(the demeaned X) is **not actually identical across horizons** — it's a different
matrix each time, even though the underlying raw X hasn't changed. The same is true
of `X'X` and its inverse. Caching one inverse and reusing it across horizons would
be estimating the wrong model at every horizon but h=0.

### 2.2 Standard errors are HAC, not just clustered

LP forecast errors `u_{t+h}` are a moving-average process by construction: an h-step
overlapping forecast error is correlated with forecast errors up to h-1 periods away,
even under correct specification. The standard fix (Jordà 2005) is Newey-West with a
bandwidth that grows with h (commonly h+1 or h). This is not optional — using plain
heteroskedasticity- or cluster-robust SEs at longer horizons understates the true
sampling variance.

The document's variance formula is the clustered-SE case. It doesn't generalize to
HAC without modification, and the HAC "meat" of the sandwich is horizon-dependent in
a way that simple clustering isn't: the number of autocovariance terms being summed
literally changes with h.

### 2.3 Horizon-varying regressors

Several common LP variants don't hold X fixed across horizons at all:

- **State-dependent / smooth-transition LPs** interact regressors with a state
  indicator that can itself be horizon-specific (e.g., the state measured at t,
  or averaged over [t, t+h]).
- Some specifications add **h extra lags of controls** as h grows, to soak up
  additional pre-trend or to satisfy a "local" version of strict exogeneity.

In both cases the premise "X is identical across horizons" is false by construction,
not just due to sample loss.

---

## 3. Solutions

| Barrier | Why the naive cache fails | Proposed fix |
|---|---|---|
| Shrinking sample | Group means / `(X'X)^{-1}` differ by horizon | Warm-started (or downdated) demeaning |
| HAC/Newey-West SEs | Bandwidth and autocovariance terms grow with h | Separate the X-only and residual-only parts of the sandwich |
| Horizon-varying regressors | X itself changes, not just the sample | Partitioned regression / block-inverse update |

### 3.1 Warm-started (or exact downdated) demeaning

**Practical fix.** Alternating-projections demeaning (à la Correia 2016 / `fixest`)
is iterative. Instead of restarting from scratch at each horizon, initialize the
horizon-h demeaning with the converged fixed-effect estimates from horizon h-1. Since
the sample at h differs from h-1 by only a handful of trailing rows per unit, the
starting point is already close to the new fixed point, so convergence typically
takes far fewer iterations. This requires no assumption about the structure of
missingness and is easy to bolt onto an existing alternating-projections routine.

**Exact fix, when applicable.** If the sample at horizon h is a strict, nested
subset of the sample at horizon h-1 (true for a balanced panel where only trailing
periods are lost), you can use rank-k downdating (Sherman–Morrison–Woodbury) to
update `(X'X)^{-1}` and group means by removing the small number of dropped rows,
rather than recomputing from the full data. This is exact and avoids reconvergence
entirely, but only applies under the nesting assumption — check it before relying on
it, since interior missingness (not just trailing) breaks the nesting.

**Fallback.** If neither is worth implementing, forcing a single common (fully
balanced) estimation sample across all H horizons restores the original document's
argument exactly, at the cost of discarding the h=0...H-1 observations that longer
horizons can't use. Worth doing only when H is small relative to T.

### 3.2 Split the HAC sandwich into cacheable and non-cacheable parts

The long-run variance estimator sums weighted autocovariances of the score
`X_t û_{t+h}`. The regressor side of that product, `X_t X_{t-j}'`, doesn't depend on
h at all — only on the lag j. Precompute and cache these cross-lag products of X
once. At each horizon, only the residual-side terms `û_{t+h} û_{t+h-j}` need to be
formed and multiplied through, and only up to that horizon's bandwidth `L_h`. This
turns the HAC computation into a set of weighted sums over cached X-blocks rather
than a full reconstruction per horizon — the same "separate what depends on y from
what doesn't" logic as the original document, extended to the variance rather than
just the point estimate.

### 3.3 Partitioned regression for extra per-horizon regressors

When horizon h adds extra regressors `W_h` on top of the fixed base X, use the
block-matrix-inversion (Woodbury / FWL) identity rather than re-inverting the full
stacked matrix:

```
[X  W_h]'[X  W_h]  inverse, built from:
  - the already-cached (X'X)^{-1}
  - X'W_h            (cheap: W_h is usually low-dimensional)
  - (W_h' M_X W_h)^{-1}   where M_X = I - X(X'X)^{-1}X'
```

This reduces the per-horizon cost from inverting a `(K + k_w) × (K + k_w)` matrix to
inverting a `k_w × k_w` one, reusing the cached base inverse. Worth it whenever `k_w`
(the number of horizon-specific extra regressors) is small relative to K.

---

## 4. Simple tests to confirm the caching implementation is worth it

Build in this order — each test gates the next. There's no point benchmarking speed
before confirming correctness, and no point committing to full HAC-splitting before
knowing where the naive implementation's time actually goes.

**4.1 — Correctness first, always.** Simulate a small panel with known β, run both
the naive per-horizon loop and the cached version, and confirm coefficients and
standard errors agree to numerical tolerance (~1e-8) at every horizon. In R,
`all.equal()`; in Python, `np.allclose()`; in Stata, compare via `mreldif()` on the
coefficient and variance matrices. If they don't match at h=0 (no sample loss yet),
the bug is in the base caching logic; if they match at h=0 but diverge for h>0,
the bug is almost certainly the sample-imbalance issue in §2.1.

**4.2 — Profile the naive implementation before optimizing it.** Time-split a single
naive run into: (a) fixed-effect demeaning, (b) matrix inversion, (c) HAC/cluster SE
construction, (d) everything else (I/O, reshaping). If demeaning convergence — not
the final inversion — dominates runtime (plausible with high-dimensional fixed
effects), then caching `(X'X)^{-1}` alone buys little, and the real payoff is in
§3.1's warm-starting. Tools: `profvis` in R, `cProfile`/`line_profiler` in Python;
in Stata, wrap each stage in separate `timer on/off` blocks since built-in profiling
is limited.

**4.3 — Benchmark across the parameter space that matters, not a toy case.**
Vary N (units), T (periods), K (regressors), number/dimensionality of fixed effects,
and H (horizons) independently, and measure wall-clock time for naive vs. cached.
Use ranges drawn from your actual datasets, not arbitrary small values — the caching
gain is concentrated in the region of large N/K, many FE dimensions, and large H;
for small H (many macro applications run 4–12 horizons) the fixed cost of building
the caching infrastructure may not be repaid. `bench::mark()` in R,
`timeit`/`perf_counter` in Python, `timer` in Stata.

**4.4 — Find the break-even H.** Plot runtime against H for both implementations on
otherwise-identical data and locate the crossover point. If your typical applications
run fewer horizons than the crossover, the caching implementation isn't worth the
added code complexity for you, regardless of what it does asymptotically.

**4.5 — Re-run 4.1–4.3 on an unbalanced (not synthetic-balanced) sample.** This is
the test that actually answers the question posed by §2.1: how much of the
theoretical speedup survives once you handle sample shrinkage honestly, via
warm-starting or downdating, instead of assuming X is literally identical across
horizons? Compare three variants head-to-head: naive loop, cache-with-forced-balance
(fast but discards data), and the warm-start hybrid (§3.1). If the hybrid's gain over
naive is small once real-world imbalance is accounted for, that's the honest answer —
better to know it from a benchmark than discover it after building the
infrastructure.

---

## 5. Bottom line

The redundant-computation critique is real and the caching instinct is right. But
implement it against the version of LPs actually run in practice — unbalanced tails,
HAC-robust SEs, occasional horizon-specific regressors — rather than the balanced,
homoskedastic-clustering special case where the algebra is cleanest. The warm-start
demeaning fix (§3.1) is the highest-value piece to build first: it's the simplest to
implement, doesn't require the nesting assumption to hold exactly, and directly
targets what profiling will likely show as the actual bottleneck (§4.2) in
high-dimensional fixed-effect settings — which is where LPs are typically run in
applied macro and public finance work.
