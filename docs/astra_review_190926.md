# Estimation correctness review — 2026-09-19

The caching architecture is mathematically appropriate for the samples it constructs, and ordinary well-conditioned point estimates pass independent checks. The library is **not ready for an unqualified estimation-correctness sign-off**: fixed-effect inference, the quadratic-spectral kernel, and several numerical and time-index edge cases have reproducible problems. No production implementation was changed during this review.

## Scope and evidence

Reviewed every production Python module and the Rust implementation: input conversion and sorting, both dataframe adapters, leads/lags, common and horizon-specific masks, recursive singleton pruning, exact and iterative absorption, acceleration, rank checks, coefficient solves, all covariance choices, coefficient/covariance rescaling, confidence intervals, result assembly, and plotting's consumption of estimates. Also inspected both test files and the benchmark/reference workflows, including R/Stata specifications. UX and documentation quality were excluded.

The installed native extension was exercised alongside explicitly forced NumPy fallbacks. R/Stata were inspected, not executed. No external econometrics package was installed; numerical reference calculations use explicit dummy designs and NumPy SVD pseudoinverses, or direct score-pair summation, independently of fastLP's numerical helpers.

Results:

- Existing suite: **61 passed** (including Polars tests).
- **256 horizon-level comparisons** of coefficients and HC0 against explicit full-dummy regressions passed. These cover balanced/gapped panels, no/one/two/three effects, common/per-horizon samples, level/cumulative outcomes, native/fallback absorption, and forced one-horizon batches. Maximum coefficient difference: `4.17e-15`.
- Independent no-FE classical and HC0–HC3 calculations passed.
- Independent one- through four-way and interaction cluster calculations passed for CR0/CR1 on native and fallback paths; expected negative-variance rejections were checked too. These checks use no absorbed effects and therefore do not validate FE degrees-of-freedom adjustments.
- **24** direct pairwise Bartlett/Parzen panel-HAC and Driscoll–Kraay comparisons passed, covering bandwidths 0/1/3 and debiasing on/off on a gapped panel without FEs.
- Separate adversarial examples reproduced the findings below. The reproduction script prints discrepancies; successful exit is not a declaration that those discrepancies are acceptable.

Run from the repository root:

```bash
UV_CACHE_DIR=/tmp/fastlp-review-uv uv run --no-sync pytest -q
UV_CACHE_DIR=/tmp/fastlp-review-uv uv run --no-sync python docs/review/independent_checks.py
UV_CACHE_DIR=/tmp/fastlp-review-uv uv run --no-sync python docs/review/estimation_reproductions.py
```

The temporary cache location avoids the environment's read-only default uv cache. `--no-sync` uses the already installed project environment.

## Findings

### 1. P1 — Absorbed parameters are missing from residual degrees of freedom

Locations: `src/fastlp/_covariance.py:35`, `:49`, `:146`, `:262`; `src/fastlp/estimator.py:533`.

All corrections use the number of columns of residualized X as the entire model rank. Absorption removes nuisance parameters from the solve, not from the fitted model's residual degrees of freedom. If D is the dummy matrix, the classical denominator is `n - rank([D, X])`, or `n - rank(D) - rank(M_D X)` when the latter design is full rank.

Reproduction: 20 units, 3 observations each, unit FEs and one slope. The code divides by 59 instead of 39. Slope variance is `0.01207358` versus `0.01826515` for classical covariance; HC1 is `0.01163304` versus `0.01759870`. Both standard errors are about 18.7% too small.

A 2×2 panel with unit/time effects and one identified slope has zero residual df. It nevertheless fits successfully and reports a zero standard error. The model must distinguish an identified coefficient from estimable residual variance.

Fix: compute absorbed rank on each retained sample, account for redundant/disconnected effects, expose residual df, and reject undefined inference. CR1 needs an explicit convention for effects nested in clusters; packages legitimately differ there, so blindly subtracting every FE for every cluster configuration is not the solution. The present blanket omission is still insufficient, especially for nonnested effects. HAC debiasing also needs a defined full-model parameter-count convention. [Reference: linearmodels' FE covariance treatment](https://bashtage.github.io/linearmodels/_modules/linearmodels/panel/covariance.html).

### 2. P1 — HC2/HC3 omit fixed-effect leverage

Location: `src/fastlp/_covariance.py:50`.

The leverage calculation uses only residualized regressors. For the original fitted model, `H = P_D + P_(M_D X)`. Its diagonal includes FE leverage: with one unit effect it includes `1/T_i` in addition to slope leverage. FWL preserves coefficients and HC0 slope covariance, but does not justify discarding this part of HC2/HC3 leverage.

On the same example, HC2 variance is `0.01188618` instead of `0.01819279`; HC3 is `0.01235740` instead of `0.02897225`. HC3 standard errors are about 34.7% too small. Full-model leverage-one observations are also not recognized reliably.

Fix: include the absorbed projection's leverage or reject HC2/HC3 with FEs until it is implemented. Validate against the slope block of explicit-dummy HC2/HC3, including unbalanced and saturated cases. [Reference: statsmodels' HC covariance implementation](https://www.statsmodels.org/dev/_modules/statsmodels/stats/sandwich_covariance.html).

### 3. P1 — Accepted ill-conditioned designs can produce materially inaccurate coefficients

Locations: `src/fastlp/estimator.py:495`, `:527`, `:536` and the batched coefficient solve.

Column equilibration is useful, but the solve and rank diagnosis both use `X'X`, which squares the design's condition number. The code records a condition estimate without selecting a safer solver when that estimate is large. A positive Cholesky factor is not an accuracy certificate.

With 200 rows and regressors `z = x + 1e-7 * noise`, an accepted fit differs from direct SVD least squares by **3.36% in coefficient-vector norm**: approximately `6919.13` versus `7159.55` for the first slope. At perturbation `1e-6`, the error is about `0.0394%`.

Fix: cache QR/SVD for risky designs, and diagnose rank from the design itself when the Gram matrix is unreliable. Rejecting exact collinearity is appropriate; silently returning a materially inaccurate accepted solution is not. Add near-collinearity comparisons of coefficients, residuals, and covariance, not just exact-duplicate tests.

### 4. P1 — Iterative absorption depends on measurement units

Locations: `src/fastlp/_demean.py:138`; `rust/src/lib.rs:166`.

Convergence uses an absolute update threshold on raw columns. Scaling both X and y should leave the slope unchanged, but small-valued columns can terminate after one sweep before their effects are removed. Acceleration's absolute denominator cutoff adds another scale-dependent behavior.

On a 200-row two-effect example, the baseline slope is `0.03413779`. Multiplying both X and y by `1e-12` returns `0.03343635` with Rust and `0.03063720` with NumPy, both reporting convergence after one iteration. The latter is a 10.25% error. Exact balanced-panel transforms are not affected by this iterative stopping rule.

Fix: normalize columns before absorption or use a relative convergence rule with well-defined scale handling, and check remaining projection error as well as update size. Test rescaling on genuinely iterative effect structures. Checking an absolute update alone also gives weak assurance on slowly converging FE graphs.

### 5. P1 — The quadratic-spectral HAC kernel is not the standard QS estimator

Locations: `src/fastlp/_covariance.py:160`, `:216`, `:249`.

QS shares the Bartlett/Parzen hard truncation and uses `lag/(bandwidth+1)`. Standard QS uses `lag/bandwidth` and retains weights at all available lags; its bandwidth is not a cutoff. The lag-pair builder and accumulation loop must change too, not only the weight function.

For an 80-observation time series, bandwidth 3 and no debiasing, reported slope variance is `0.00446966` versus `0.00533423` from direct untruncated QS summation, a 16.2% shortfall. This affects both HAC and Driscoll–Kraay when QS is selected. [Reference and formula: linearmodels QS documentation](https://bashtage.github.io/linearmodels/iv/iv/linearmodels.iv.covariance.kernel_weight_quadratic_spectral.html).

### 6. P2 — HAC time validation silently merges distinct dates

Location: `src/fastlp/_covariance.py:182`.

`np.allclose(time, round(time))` uses a relative tolerance. Thus `[2020.001, 2020.002, 2021.001]` passes an ostensibly integral-label check and becomes `[2020, 2020, 2021]`. Panel HAC's dictionary then overwrites duplicate unit/date entries, including at lag zero; DK aggregates distinct dates together.

Reproduction: with `hac_lags=0, hac_debias=False`, HAC slope variance becomes `0.00595781` while HC0 is `0.00891903`. These must agree for valid unique panel dates at zero lag.

Fix: require genuinely integral, representable labels without a magnitude-dependent tolerance; preserve integer precision and recheck uniqueness if coercion occurs. Add rejection tests for fractional years, near-integers, out-of-range values, and large integers.

### 7. P2 — Lag and lead operators use different clocks on gapped panels

Locations: `src/fastlp/_frame.py:157`, `:266`, and both `lead_positions` implementations.

Lags use preceding observed rows; numeric leads use exact calendar `t+h`. At dates `[0, 2]`, `y_lag1` at date 2 is populated with date 0, even though date 1 is missing. A regression presented as controlling for `y_(t-1)` therefore controls for a variable-duration lag after a gap, changing the conditioning set and potentially the estimand.

This behavior is explicitly described as row lags in the fit docstring and encoded in an existing test; it is a methodological contract problem, not a failure to implement that docstring. For conventional calendar-time LPs, implement exact `t-lag` matching consistent with leads. If row-time lags are desired, expose that as a deliberate alternative and use consistent time semantics.

### 8. P2 — Valid within variation can be rejected solely because of an absorbed level

Location: `src/fastlp/estimator.py:480`.

The absorption threshold is `sqrt(eps) * norm(raw X)`. Adding an absorbed constant increases this threshold without changing identifying within variation. In the reproduction, the original unit-FE fit estimates `0.23175044`; adding `1e8` to X causes a negligible-within-variation error even though the O(1) variation remains numerically resolvable.

Fix: distinguish actual absorption and numerical projection error from a raw-level-relative heuristic. Include additive-shift invariance tests as well as multiplicative scaling tests.

### 9. P2 — Time representation can erase the horizon-zero sample

Location: `src/fastlp/_frame.py:373`.

The pandas indexed path converts target time labels to numbers but leaves the lookup index in its original dtype. A balanced daily `DatetimeIndex` example reports `no valid observations at horizon 0`. Numeric strings work on a consecutive balanced fast path, but dropping one observation makes the indexed path compare integer targets to string keys, again losing the entire horizon-zero sample.

Fix: normalize the lookup and targets together, or explicitly reject unsupported representations before construction. Horizon zero should always map a retained observation to itself. Define the period unit before implementing calendar leads for datetimes, and test representation/fast-path parity across both dataframe backends.

### 10. P2 — Negative variances can become zero standard errors after rescaling

Location: `src/fastlp/estimator.py:659`.

The negative-variance threshold is applied after conversion to feature units and floors its reference scale at 1. This treats every negative variance smaller in magnitude than roughly `2.22e-14` as harmless, regardless of its relative size.

A two-way cluster example correctly raises for a negative variance. Multiplying the shock by `1e9` makes the same model return covariance `-1.96202071e-21` and standard error zero. The covariance remains negative in the public result while its interval collapses to a point.

Fix: assess materiality in normalized coordinates or against a justified componentwise numerical error bound. Genuine indefinite multiway covariance is possible; it must not be disguised as perfect precision by measurement units. Any PSD adjustment must be explicit and coherent with the returned covariance.

## Validation gaps and methodological boundaries

The benchmark's `fit_independent_horizons` imports fastLP's own `demean`, `hc1`, and `cluster_cr1`, and uses another normal-equation solve. It verifies cache equivalence, not independent statistical correctness. Its HC1 comparison therefore repeats the missing-FE-df error. Most extended covariance tests check finiteness or pandas/Polars agreement rather than a separate mathematical oracle. The new review scripts close some of this evidentiary gap, but are not a replacement for permanent regression tests after correction.

The top-level Python benchmark uses `sample='common'`, while the fixest/reghdfe scripts let each horizon use its available sample. For T=40 and H=12 this means 28 observations per unit at every Python horizon versus 40 at horizon zero in the other scripts. Those outputs cannot certify identical estimators until samples are aligned. Their saved estimates also omit covariance/SE comparisons.

Other choices require interpretation rather than being unconditional implementation bugs:

- Confidence intervals always use normal quantiles. They are pointwise asymptotic intervals, not exact small-sample Student-t intervals, few-cluster-adjusted intervals, or simultaneous bands. CR1 scaling alone does not change the reference distribution.
- Panel HAC assumes the required cross-unit independence; DK allows cross-sectional dependence but needs suitable time-series asymptotics. Correct implementation does not make either universally appropriate.
- FE OLS with lagged outcomes can have dynamic-panel bias; neither faster absorption nor sandwich covariance removes coefficient bias. Identification of a supplied shock is an application-level assumption.
- `response='cumulative'` estimates the response of `sum(y[t:t+h+1])`. This is appropriate when that sum is the target (for example, accumulated flows/growth); it is not automatically `y[t+h]-y[t-1]` for a level outcome.
- Common-sample restriction can legitimately change estimates relative to horizon-specific samples. The mask grouping and FE re-estimation on each retained mask were correct in the independent cases tested.
- Recursive singleton removal does not establish positive residual df or full FE-graph connectivity. The saturated 2×2 example has no singletons.

## Correction order

First fix absorbed rank/df and HC2/HC3 leverage, stable solving, scale-aware demeaning, and QS. Then address time normalization/lag semantics and invariant numerical validation. Before treating the library as validated, promote the reproductions into regression tests and require independent full-dummy coefficient, residual, covariance, and interval comparisons across every supported covariance/FE combination. For cluster corrections, lock down the intended nesting convention before claiming parity with another package.

The passing checks support the caching design and the explicitly tested formulas; they do not justify a claim of 100% correctness or finite-sample statistical validity for every application.

## Follow-up: problems 1 and 3 — implementation and verification

The user approved implementing nonnested cluster parameter counting, a common
minimum-cluster CR1 multiplier, and Student-t clustered intervals with
`min(G)-1` reference degrees of freedom. Actual model rank remains separate from
the cluster correction count. Full parameter counting and componentwise cluster
adjustments will remain explicit comparison options. Exact FE rank is the
default; a conservative approximation must be requested explicitly.

For problem 3, choose a Cholesky/QR switching threshold using reproducible
stress tests, emphasizing 1–7 explicit regressors (with larger checks too).
Test coefficients, residuals, and covariance; preserve decomposition reuse per
retained sample. A condition diagnostic alone is insufficient.

Use statsmodels OLS with explicit dummy variables as the independent Python
reference for rank, coefficients, residuals, and classical/HC1/full-count
cluster covariance. For nonnested and multiway-min corrections, compare the
uncorrected independent covariance with the explicitly specified multipliers;
do not confuse statsmodels' default correction with our selected convention.
SciPy supplies stable triangular solves and Student-t quantiles; statsmodels is
a development dependency, not an estimator runtime dependency.

### Problem 3: threshold experiment and implementation

`docs/review/solver_stress.py` ran 774 deterministic designs with 1–7 and 16
regressors, 64/512/4,096 observations, target condition numbers from 1 through
1e10, and three random seeds. Nine additional large designs within that total
use 140,000 observations. Each case compares noisy, nearly orthogonal, and
exact-fit outcomes against a direct SVD solve, plus an HC0 covariance reference.
Condition numbers refer to column-normalized X, not X'X.

| Maximum admitted design condition | Worst relative coefficient error | Worst relative HC0 error |
| ---: | ---: | ---: |
| 100 | 2.30e-12 | 3.58e-12 |
| 300 | 2.70e-11 | 4.11e-11 |
| 1,000 | 4.29e-10 | 6.19e-10 |
| 3,000 | 2.11e-9 | 3.45e-9 |
| 10,000 | 3.67e-8 | 8.36e-8 |
| 100,000 | 2.72e-6 | 3.91e-6 |

Provisional cutoff: **1,000**, leaving headroom below a 1e-8 relative-error
target in this experiment. This is empirical evidence, not a universal error
bound. Well-conditioned fits retain Cholesky; poor conditioning or failed Gram
factorization triggers QR. Rank-deficient designs are diagnosed using singular
values of R rather than rejected solely from a noisy Gram eigendecomposition.
QR computes residuals and sandwich scores in its orthogonal Q basis, then maps
coefficients/covariances back to original units. This also avoids the unstable
Gram-inverse sandwich on the difficult branch. Both paths cache per sample.

### Problem 1: implemented counting and correction policy, validation underway

Added exact observed-category rank for one FE, connected-component rank for
two FEs, and the balanced two-way shortcut. General multiway rank removes
nested/duplicate dimensions before a bounded explicit-dummy SVD (64 MiB dummy
matrix and minimum dimension at most 2,000). Above that budget, exact mode
raises rather than silently approximating. `fe_dof='conservative'` explicitly
selects a pairwise-connectivity upper bound for general 3+ effects; diagnostics
identify the bound as non-exact. Counts are recomputed on each retained sample.

Full model rank now determines classical, HC1, and debiased HAC/DK denominators
and positive-residual-df validation. Cluster correction options are
`cluster_df='nonnested'|'full'|'none'`,
`cluster_group_adjustment='min'|'component'`, and
`cluster_inference='t'|'normal'`. Defaults are the first values. Nonnested
counting removes dimensions wholly nested in any requested primary cluster
partition; retained nonnested dimensions are rank-counted jointly. When none
remain, an intercept is still counted. `none` is a compatibility option that
excludes the entire absorbed span from correction K, including its intercept.
CR0 applies no multiplier. Nonclustered interval reference distributions are
unchanged; clustered intervals now use Student-t with min(G)-1 by default.

The original singleton-pruning fixture retained a saturated 2x2 core. It now
correctly fails the residual-df check. Its pruning test has one extra core
observation, and a separate regression test verifies rejection of the original
saturated model. The old test expecting Cholesky failure to abort was changed
to verify successful, accurate QR fallback.

### Problems 1 and 3: validation results and performance

The original problem-1 reproduction now matches explicit dummy regression:
classical slope variance `0.01826515455344563` versus reference
`0.018265154553445635`, and HC1 `0.017598704167338785` versus
`0.01759870416733879`. The saturated 2x2 example now raises a positive-residual-df
error instead of returning a zero standard error.

The original problem-3 example (`z = x + 1e-7 * noise`) now has relative
coefficient error `7.57e-10` against SVD, down from `0.03358`. The 1,000 cutoff
is retained after additional polynomial, concentrated-row, nearly duplicate,
and measurement-unit stress cases. Tests include 1–7 and 16 regressors and
both small and large outcome units. Extremely ill-conditioned problems still
have intrinsic sensitivity; QR is not a universal forward-accuracy guarantee.

Independent Python validation uses statsmodels 0.15.0, with SciPy 1.18.1 and
NumPy 2.5.1 in this environment. Tests cover full-dummy coefficients, residuals,
classical/HC0/HC1 covariance, full-model HAC/DK debiasing, all three cluster
parameter-count policies, both multiway multipliers, interaction clusters,
CR0, Student-t and normal intervals, and nesting that changes with the
horizon's retained sample. On severely collinear designs, statsmodels' own
HC2/HC3 leverage and cluster sandwich intermediates can lose precision. For
those comparisons, statsmodels fits an equivalent orthogonal basis and the
result is transformed back; no tolerance was loosened to accommodate that
reference instability. This tests the solver, not the still-unfixed omission
of FE leverage in HC2/HC3 (problem 2).

The benchmark reference now uses explicit-dummy statsmodels rather than
fastLP's own absorption/covariance helpers. It matches common samples and our
default nonnested one-way correction. Large explicit-dummy references are
reported unavailable when their rank/memory budget is exceeded. The smoke
benchmark's numerical comparison passes. This replaces the circular reference
described in the original review; the separate R/Stata benchmark sample
mismatch remains outside this change.

Measured solver-stage medians, one BLAS thread, five repetitions after warm-up,
140,000 observations and 13 outcomes (milliseconds; excludes data preparation,
FE absorption, and covariance score accumulation):

| Regressors | Cholesky baseline | Always QR | New automatic solver |
| ---: | ---: | ---: | ---: |
| 2 | 22.55 | 36.21 | 22.34 |
| 4 | 17.65 | 22.95 | 16.14 |
| 7 | 22.10 | 38.77 | 22.55 |

Each timing includes normalization, rank diagnostics, coefficient/residual
calculation, and the small covariance mapping/bread setup. These are local
microbenchmarks, not a claimed end-to-end speedup; differences of a few
milliseconds and nonmonotone times reflect allocation/cache/timing effects.
Well-conditioned cases use Cholesky. QR's extra cost is paid on the problematic
branch and amortized across horizons with the same sample.

For a 132,925-row gapped panel with 5,000 units and 28 periods, the exact
two-way connected-component calculation took **3.59 ms**, and one nesting
scan took **1.80 ms**. Counting one effect with already-encoded categories took
about **0.005 ms**. Encoding itself is not included in these timings; generic
three-plus-effect SVD costs can be much larger. Balanced two-way panels use
the direct category-count shortcut.

Reproduce the threshold grid and timings:

```bash
UV_CACHE_DIR=/tmp/fastlp-review-uv OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-sync python docs/review/solver_stress.py --timings --output /tmp/fastlp_solver_stress.json
```

The original independent review script still passes all 256 coefficient/HC0
comparisons, its cluster checks, and its 24 Bartlett/Parzen HAC/DK checks. Its
cluster checks now explicitly select the componentwise convention they test.
The reproduction script handles the newly rejected saturated example so it
can continue showing the remaining open issues.

Current implementation status: **problems 1 and 3 addressed within the stated
rank-budget and numerical-precision limits**. Problems 2 and 4–10 remain open;
in particular, fixing rank accounting does not fix FE leverage, iterative
absorption scaling, or QS. No other review issue is marked resolved by these
changes. Future steps and results should continue to be appended here.

Final verification for this follow-up: **205 tests passed**, including the
optional Polars backend, using `uv run --frozen --extra polars pytest -q`
(one BLAS thread). The seven warnings are the expected few-cluster warnings in
existing fixtures. Both HC1 and clustered smoke benchmarks pass the new
statsmodels comparison. Ruff checks pass for the new numerical modules,
reference tests, stress script, and benchmark reference; `git diff --check`
passes. Only problems 1 and 3 and the reference infrastructure needed to
verify them were implemented in this follow-up.

### Follow-up: fail-fast exact rank budget

Fixed-effect rank is now resolved immediately after singleton pruning and FE
encoding for each distinct retained sample, before exact or iterative design
absorption, solver factorization, outcome construction, or covariance work.
This is the earliest correct point because rank, connectivity, nesting, and the
exact-budget decision can change when a horizon has a different retained
sample. Nested and duplicate effects are reduced inside this check, so a
nominal three-plus-effect specification that simplifies exactly to one or two
independent dimensions continues without requiring `fe_dof="conservative"`.

A regression test forces the exact dummy-rank budget to one byte and replaces
the iterative absorber with a function that raises if called. The public fit
raises the intended numerical-rank-budget error, proving that absorption is not
entered. Cluster nesting and correction counts are also resolved before
absorption once full rank is available, so invalid cluster correction degrees
of freedom fail at the same early boundary.

Verification after this change: **206 tests passed** with the frozen project
environment and optional Polars backend. The additional test is the fail-fast
absorber guard described above; `git diff --check` also passes.

### Follow-up: proposed resolution of problems 4 and 2

No implementation was changed in this follow-up. The two issues and candidate
solutions were re-evaluated against the current NumPy and Rust paths.

For problem 4, both iterative backends currently compare the largest raw-unit
change between successive iterates with `demean_tol`. The acceleration
denominator is also compared with an absolute machine epsilon. Consequently,
changing a column from dollars to billions of dollars changes the numerical
question being asked, even though it must not change the regression. The
recommended correction is to normalize every input column to a stable scale
before iterative absorption, run the existing algorithm in normalized units,
and restore its original scale afterward. Convergence should require both (a)
a sufficiently small normalized update and (b) a sufficiently small remaining
FE projection error (largest absolute group mean). The projection-error check
can be evaluated only when the update test first claims convergence, avoiding
an extra group scan on every iteration. This also makes the acceleration
epsilon meaningful in normalized coordinates. Zero/non-finite columns need an
explicit path, and NumPy and Rust must share precisely the same scale and
stopping definition.

An equivalent alternative is a mixed absolute/relative stopping rule based on
each column's initial norm, again combined with a projection-error certificate.
It avoids physically rescaling the work vector but is easier for the two
backends to implement differently by accident. Merely changing the update test
to relative error is insufficient: the residualized vector may legitimately
approach zero, and a small iteration-to-iteration change alone does not certify
orthogonality to every FE. Direct sparse least-squares projection would be a
third, substantially more expensive alternative and would give up the main
reason for the alternating-projection implementation. The recommended
normalize-and-certify approach adds two linear column passes for scale/restore
and normally one final FE group scan; its expected cost is small relative to
multiple absorption sweeps. It must be stress-tested over very small and large
units, nearly absorbed columns, poorly connected FE graphs, all acceleration
modes, and both backends.

For problem 2, HC2 and HC3 must use the diagonal of the hat matrix for the
*entire* fitted model. With absorbed effects this decomposes as
`h_full = h_FE + h_within_X`; the current code includes only `h_within_X`.
Thus point estimates and HC0 remain valid, while HC2/HC3 under-correct each
observation's squared residual and can miss leverage-one observations. This is
the same full-design leverage used in the standard HC2/HC3 definitions in
[statsmodels](https://www.statsmodels.org/dev/_modules/statsmodels/stats/sandwich_covariance.html).

There are three defensible implementation strategies. First, compute exact FE
leverage. One FE is cheap (`1 / group_size`), and a complete balanced two-way
panel has the closed form `1/N + 1/T - 1/(N*T)`. General unbalanced two-way
effects require graph/Laplacian leverage calculations; general three-plus-way
effects require the diagonal of a potentially very large dummy projection.
Those general exact calculations can be substantially more expensive than
coefficient absorption. Second, build an explicit independent dummy design and
obtain row leverage from a QR/SVD under a strict memory/rank budget. This is a
clean exact reference and a practical small-problem path, but is not suitable
for large high-dimensional effects. Third, use randomized leverage estimates.
That scales well but is approximate, adds randomness/tuning, and is inconsistent
with the present goal of claiming exact HC2/HC3.

The recommended product-safe sequence is therefore: immediately reject HC2
and HC3 whenever effects are present, rather than continue returning known
incorrect standard errors; then add exact fast paths for one FE and complete
balanced two-way FE, plus a budgeted explicit-QR path for other small designs.
Larger general designs should continue to fail clearly until an exact sparse
leverage algorithm is implemented. HC0, HC1, cluster, HAC, and Driscoll--Kraay
do not use observation leverage and remain available. Tests should compare the
slope block with explicit-dummy statsmodels HC2/HC3 for unbalanced, disconnected,
nested, singleton-pruned, and leverage-one samples. An approximation could be
offered only later as an explicitly named opt-in estimator, never silently as
HC2 or HC3.

### Follow-up: implemented scale-safe absorption and HC2/HC3 guard

Problem 4 is now fixed in the shared `Residualizer` boundary. Every column is
divided by its maximum absolute input value before either the Rust or NumPy
iterative absorber runs, with an all-zero column assigned scale one, and the
transformed result is restored to its original units on return. Consequently,
the convergence tolerance and acceleration denominator operate in normalized
coordinates. Both backends now accept convergence only when the normalized
maximum update is below `demean_tol` *and* the largest remaining FE group mean
is below that tolerance. The projection scan is evaluated only after the cheap
update condition passes because the language-level `and` operations short
circuit. Exact balanced-panel demeaning does not enter this path and is
unchanged.

Regression tests cover three irregular effects, both native Rust and NumPy,
and scales `1e-12`, `1e-8`, `1e8`, and `1e12`. Residualized results agree after
undoing the scale, iteration counts are identical, and all FE group means meet
the normalized tolerance. The original public reproduction now returns the
same coefficient at scales one, `1e-8`, and `1e-12`: `0.0341377923494863` up
to displayed floating-point variation. Both backends take 13 iterations at
all three scales; before this change the `1e-12` cases incorrectly stopped
after one iteration and differed by as much as 10.25%.

For problem 2, public `fit` now rejects HC2 or HC3 as soon as nonempty fixed
effects are parsed. Its error states that exact full-model leverage is not yet
available and explicitly recommends `covariance='hc0'` or `covariance='hc1'`.
This prevents the library from returning the known incorrect FE-adjusted
HC2/HC3 standard errors while preserving HC2/HC3 for regressions without fixed
effects. The reproduction script now records this deliberate rejection beside
the explicit-dummy reference values. Exact FE-aware HC2/HC3 remains a future
capability, not a claimed implementation; the correctness defect is contained
by fail-fast behavior.

The Rust extension was rebuilt locally from the changed source with maturin's
offline, in-place mode. Final verification: **215 tests passed** with the
frozen environment and optional Polars backend; the seven warnings are the
expected few-cluster warnings. The targeted native/fallback and HC guard set
has 10 passing tests. `cargo fmt --check` and `git diff --check` pass. Ruff on
the touched file set reports only pre-existing whole-file findings in the
estimator, estimator tests, and reproduction script; it reports no newly
introduced diagnostic localized to these changes.

### Follow-up: corrected quadratic-spectral HAC (problem 5)

Problem 5 is now fixed. Quadratic-spectral weights use the standard
`lag / bandwidth` argument and
`3 * (sin(z) / z - cos(z)) / z**2`, where
`z = 6 * pi * lag / (5 * bandwidth)`. Unlike Bartlett and Parzen, QS now
includes every available calendar lag rather than treating the bandwidth as a
hard cutoff. This applies to both within-unit panel HAC and time-aggregated
Driscoll--Kraay. Their debiasing and full-model parameter count are unchanged.

Bandwidth zero has an explicit fastLP meaning: retain lag zero and no serial
lag terms. Panel HAC with `hac_lags=0`, QS, and no debiasing therefore equals
HC0. This is consistent with fastLP's existing cross-kernel `hac_lags=0`
contract. It deliberately does not copy linearmodels 7.0's unusual standalone
QS weight-array behavior of returning a zero at lag zero when bandwidth is
zero. For positive bandwidths, fastLP's weights match linearmodels' public
`kernel_weight_quadratic_spectral` through lags well beyond the bandwidth.

The initial exact all-pairs implementation exposed a material performance cost.
The final implementation has two paths:

- Panels whose units share a dense or regularly spaced time grid use FFT
  cross-correlations. The actual calendar spacing is retained when evaluating
  weights, so a grid spaced by two receives weights at lags 2, 4, and so on.
- Unequal or irregular calendars use an exact all-pairs calculation. Work is
  divided into blocks capped by one million pair elements, bounding temporary
  memory rather than constructing the full panel pair matrix.

The FFT result was tested against both linearmodels' full OLS QS covariance and
an independent ordered-pair formula. The irregular fallback was tested on a
gapped, unbalanced panel against that ordered-pair formula. Coverage includes
panel HAC, Driscoll--Kraay, debiasing on/off, positive and zero bandwidth, and
weights beyond the bandwidth. The independent review script now adds 12 QS
HAC/DK comparisons to its existing 24 Bartlett/Parzen comparisons. The
original 80-observation reproduction now reports slope variance
`0.005334232287937206` versus direct reference `0.005334232287937196`; the old
result was `0.004469657867859741`.

`linearmodels==7.0` was added as a development dependency solely for the
independent QS oracle. It is not a runtime dependency.

Benchmark medians below use five measured repetitions after one warm-up, one
BLAS/OpenMP thread, four score columns, and bandwidth four. The comparison is
kernel-meat time only. “Old” is the former bandwidth-truncated amount of work;
it is faster partly because it computes a different, incomplete estimator.

| Covariance path | Time grid | Size | Old truncated | Correct all-lag QS | Ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| Panel HAC | dense | 500 x 20 (10,000 obs.) | 2.12 ms | 5.07 ms | 2.40x |
| Panel HAC | dense | 500 x 80 (40,000 obs.) | 9.56 ms | 18.52 ms | 1.94x |
| Panel HAC | dense | 500 x 200 (100,000 obs.) | 23.88 ms | 29.27 ms | 1.23x |
| Panel HAC | regular gaps | 500 x 80 | 6.00 ms | 8.55 ms | 1.42x |
| Panel HAC | regular gaps | 500 x 200 | 14.47 ms | 26.31 ms | 1.82x |
| Panel HAC | irregular | 500 x 80 | 7.14 ms | 188.44 ms | 26.40x |
| Panel HAC | irregular | 500 x 200 | 19.08 ms | 1,041.27 ms | 54.59x |
| Driscoll--Kraay | dense | 100 periods | 0.09 ms | 0.86 ms | 9.78x |
| Driscoll--Kraay | dense | 500 periods | 0.17 ms | 3.42 ms | 19.64x |
| Driscoll--Kraay | dense | 2,000 periods | 0.55 ms | 12.72 ms | 23.03x |
| Driscoll--Kraay | irregular | 500 periods | 0.15 ms | 12.13 ms | 79.60x |
| Driscoll--Kraay | irregular | 2,000 periods | 0.43 ms | 235.87 ms | 552.97x |

The large ratios for Driscoll--Kraay start from sub-millisecond truncated
baselines; absolute corrected time remains about 13 ms at 2,000 regular periods
and 236 ms at 2,000 irregular periods. General irregular QS is intrinsically
quadratic in observations per unit/time series because every available pair
has a nonzero kernel weight. The optimized regular-grid cases are the expected
common panel configuration. Reproduce the benchmark with:

```bash
UV_CACHE_DIR=/tmp/fastlp-review-uv OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --frozen python docs/review/qs_benchmark.py --repeats 5
```

Final verification: **223 tests passed** with the frozen environment and
optional Polars backend; the seven warnings are the expected few-cluster
warnings. The eight targeted QS reference tests and all 12 independent-script
QS comparisons pass. Ruff check/format pass for the changed covariance module
and benchmark script, and `git diff --check` passes. Problems 1, 3, 4, and 5
are implemented; problem 2 is safely rejected for FE models pending exact
full-model leverage. Problems 6--10 remain open.

### Follow-up: proposed joint resolution of problems 6 and 9

No implementation changed in this follow-up. Problems 6 and 9 have the same
underlying cause: fastLP has no single internal definition of panel time. Lead
construction, balanced-panel detection, pandas joins, Polars joins, and HAC
each convert or interpret the supplied labels independently.

Problem 6 occurs because HAC converts every label through `float64`, accepts
values that are merely *close* to integers using a magnitude-dependent
relative tolerance, rounds them, and casts to `int64`. Distinct dates can then
become the same internal date. This corrupts lag-zero HAC and DK aggregation;
integer labels larger than float64's exact range can also lose information.

Problem 9 occurs because the pandas indexed lead path converts target labels
to numeric values but builds its lookup from the original labels. Numeric
strings therefore compare against integers, and datetimes compare against an
unrelated numeric representation. The balanced path happens to avoid that
join, so changing only panel balance can change whether identical labels work.
Horizon zero is unnecessarily sent through this mismatched lookup instead of
being defined as each row itself.

Three broad policies are possible:

1. Require literal integer period identifiers everywhere. This is the smallest
   and safest contract, but rejects convenient integer-valued floats and
   integer strings and requires users to convert dates before fitting.
2. Build one exact canonical integer-period clock. Native integers are retained
   without passing through float; floats are accepted only when finite, exactly
   integral, in range, and round-trip safe; integer-form strings can be parsed
   exactly. Fractional values and unsupported labels fail before sample or
   covariance construction. This keeps common year/quarter-index workflows
   while giving every backend and estimator one clock.
3. Support calendar types directly. Datetimes and periods require an explicit
   frequency or step (day, month, quarter, and so on), since `horizon + 1` has
   no universal meaning for a timestamp. This is the richest interface but
   expands the API and requires careful pandas/Polars parity work.

Silently factorizing sorted dates into `0, 1, 2, ...` is not a valid general
solution. It erases gaps: dates 2000 and 2002 would become adjacent, changing
both lead availability and HAC lag distance. Likewise, retaining the current
“arbitrary balanced labels mean adjacent rows” fallback makes the estimand
depend on whether one observation happens to be missing.

The recommended core fix is option 2 now, with option 3 as a later explicit
feature. Normalize time exactly once during panel preparation and expose that
canonical `int64` coordinate to lead construction and HAC/DK. Revalidate
uniqueness of `(unit, canonical_time)` after conversion so no accepted labels
can collide. Initialize horizon zero directly with row positions, then perform
all positive-horizon joins as exact `canonical_time + horizon` matches. Use the
same coordinate in pandas and Polars and allow the balanced arithmetic path
only when its shared time grid advances by exactly one canonical period.

For the initial contract, accept signed/unsigned integer values within `int64`,
exact integral finite floats that can be represented safely, and optionally
strict base-10 integer strings. Reject fractional numbers such as `2020.001`,
out-of-range integers, mixed representations, datetimes, and arbitrary labels
with an error asking the user to supply integer period identifiers. Datetime
support should later require a declared frequency and convert to the same
canonical coordinate rather than creating a second lead algorithm.

The expected performance cost is one linear validation/conversion pass and one
eight-byte integer coordinate per observation. Lead joins and HAC already need
a time array, so subsequent work should be unchanged or slightly simpler.
Tests must cover fractional near-integers at small and large magnitudes,
integers around `2**53`, `int64` boundaries, conversion collisions, numeric
strings if retained, unsupported datetimes, horizon-zero self-mapping,
balanced/unbalanced parity, pandas/Polars parity, and HAC0 equality with HC0.

### Follow-up: implemented the canonical panel clock (problems 6 and 9)

Problems 6 and 9 are now fixed within an explicit integer-period contract. A
new shared time-normalization module runs once at the pandas or Polars input
boundary, before uniqueness checks, sorting, lead construction, or covariance
work. Both backends replace their private copy of the supplied time column with
the same canonical `int64` coordinate. HAC/DK uses this same normalizer as a
defensive internal boundary rather than maintaining a separate float-rounding
implementation.

Accepted clocks are native signed integers, unsigned integers no larger than
`int64` maximum, finite exactly integral floats with magnitude no larger than
`2**53 - 1`, and strict base-10 integer strings. String parsing uses Python
integers before the `int64` range check and never passes through float. The
implementation rejects booleans, fractions and near-integers, non-finite
values, larger floats, out-of-range integers/strings, datetimes, timedeltas,
arbitrary labels, and incompatible object mixtures. Datetime support remains a
future explicit-frequency feature; it is not assigned an implicit daily or
row-order meaning.

`(unit, canonical_time)` uniqueness is rechecked after conversion, catching
collisions such as `"01"` and `"1"`. Native integer labels beyond `2**53` are
preserved without float conversion. Positive lead construction checks that
adding the requested maximum horizon cannot overflow `int64`; HAC/DK rejects a
time span too large for safe signed lag subtraction.

All indexed lead matrices now initialize horizon zero directly to each row's
own position and join only positive horizons. Both pandas and Polars match
positive horizons against exact `canonical_time + horizon` keys. The balanced
fast path no longer converts labels to float: it verifies an exact shared
`int64` sequence with period increment one. Integer-string panels therefore
behave identically whether balanced or unbalanced, while arbitrary balanced
labels no longer receive a path-dependent row-order interpretation.

The original adversarial HAC example with `[2020.001, 2020.002, 2021.001]`
now fails at input validation rather than merging dates. Unbalanced numeric
strings now fit successfully through the indexed path; the prior reproduction
lost the complete horizon-zero sample. Datetimes fail with the same explicit
integer-period message in every path rather than sometimes working only on a
balanced panel.

New tests cover exact native integers above `2**53`, both `int64` boundaries,
unsigned and string overflow, fractional and nearly integral floats, the
float consecutive-integer safety boundary, booleans, mixed objects,
conversion collisions, lead-addition overflow, HAC lag-span overflow,
integer-string balanced/unbalanced equivalence, horizon-zero sample identity,
and pandas/Polars coefficient and covariance parity. Direct normalization
microbenchmarks in this environment took about **0.005 ms** for an already
`int64` one-million-row array (zero-copy validation), **5.22 ms** for one
million `float64` values, and **259 ms** for 200,000 strings. Dataframe copying,
assignment, and sorting are excluded; exact string parsing is intentionally
more expensive than unsafe float coercion.

Final verification: **239 tests passed** with the frozen environment and
optional Polars backend; the seven warnings are the expected few-cluster
warnings. All 256 independent full-dummy coefficient/HC0 comparisons, cluster
checks, 24 Bartlett/Parzen HAC/DK comparisons, and 12 QS comparisons still
pass. Ruff check/format pass for the new clock module, changed numerical
modules, and boundary tests (excluding only pre-existing quoted-annotation
findings in the full frame adapter); `git diff --check` passes. Problems 6 and
9 are resolved under the documented supported clock representations.

### Follow-up: proposed resolution of problems 8 and 10

No implementation changed in this follow-up.

Problem 8 is caused by comparing the norm of the residualized regressor with
`sqrt(machine_epsilon) * norm(raw_regressor)`. The raw norm contains levels
that fixed effects are supposed to remove. Adding a large constant or other FE
component therefore raises the rejection threshold without changing the
identified within variation. The current `sqrt(epsilon)` cutoff is also much
looser than the actual configured absorption tolerance in common cases.

Simply deleting the check is one option: exact-zero columns would still fail,
and `RegressionDesign` would diagnose collinearity after scaling. However, a
perfectly absorbed column can leave floating-point dust, which could then be
scaled to unit norm and mistaken for identification. Another option is to
compare against a centered raw norm, but global centering removes only a
constant and remains sensitive to other absorbable FE components. Rebuilding
the complete dummy design and testing rank is exact for small problems but
defeats high-dimensional absorption at scale.

The recommended solution is a method-aware numerical certificate in the same
normalized coordinates used by absorption. Retain each column's normalization
scale and normalized residualized norm. Reject a column only when its within
signal is zero/non-finite or indistinguishable from the absorber's documented
error bound: a tolerance-derived bound for iterative absorption and a
machine-roundoff-derived bound for exact group transforms. Then let the
already scale-equilibrated QR/Cholesky design check handle collinearity among
the surviving columns. This remains sensitive to units only when the within
signal genuinely falls below floating-point or requested absorption accuracy,
not merely because `sqrt(epsilon) * raw norm` is large. A budgeted explicit
dummy/SVD calculation can serve as the oracle in stress tests and, if desired,
as a rare fallback for borderline small designs.

Problem 10 occurs after covariance has already been converted from the
normalized regression basis back to the user's feature units. Its reference
scale is floored at one, so any negative variance smaller than roughly
`2.22e-14` is treated as harmless even if it is 100% of that variance. The
negative value remains in `covariance_`, while `sqrt(max(value, 0))` reports a
zero standard error. Rescaling a regressor can therefore change an error into
apparently perfect precision.

Possible policies are: reject every negative diagonal (simple but vulnerable
to one-ulp roundoff), silently project the covariance to positive semidefinite
(changes all entries and should not be an implicit definition of cluster
covariance), or distinguish roundoff from a materially negative estimate in a
scale-invariant working basis. The recommended policy is the third.

Validate the covariance before mapping it out of `RegressionDesign`'s
normalized coordinates. Use a componentwise roundoff scale with no absolute
floor. For multiway clustering, where large inclusion--exclusion components
can cancel, propagate the sum of the absolute component contributions (or an
equivalent sandwich error bound), rather than comparing only with the small
final matrix. A diagonal below the negative numerical bound must raise. A
negative diagonal within that bound is numerical cleanup: set the returned
diagonal itself to exactly zero, set its standard error to zero, and record the
cleanup in covariance diagnostics. Never return a negative covariance
diagonal paired with a zero standard error. Do not silently PSD-project the
full matrix; an indefinite off-diagonal covariance can be a property of the
finite-sample multiway estimator, while a negative variance makes the reported
standard error undefined.

The problem-8 certificate adds only columnwise scale/error bookkeeping and is
negligible relative to absorption. A small-design SVD fallback would be paid
only for borderline cases. Problem 10's normalized validation is negligible;
tracking multiway cancellation may add a small `k x k` sandwich/error-bound
operation per inclusion--exclusion component (at most 15 with the current
four-term cluster limit), while cluster score accumulation remains the dominant
cost. Tests should cover additive FE shifts, multiplicative units, exactly and
nearly absorbed columns, iterative/exact and pandas/Polars paths, one- through
four-way cluster cancellation, covariance rescaling, and coherence among
`covariance_`, `stderr_`, and confidence intervals.

### Follow-up: problem 8 rechecked after the QR implementation

Problem 8 remains active and is independent of the Cholesky/QR selection added
for problem 3. The within-variation guard executes immediately after FE
absorption and before `RegressionDesign` is constructed. Solver selection,
design rank diagnosis, and QR therefore never see a column rejected by this
guard.

The original 60-observation unit-FE reproduction was rerun on the current
code. The unshifted model estimates `0.23175043664900719` and selects
`scaled_cholesky`. Adding `1e8` to the shock still raises “residualized design
has zero or negligible within variation.” For an explicit ordering check,
`RegressionDesign` was temporarily wrapped with a constructor flag; the flag
remained false when the shifted fit failed. Thus neither the automatic QR
threshold nor forcing a numerically safer solver downstream can address the
false rejection. The proposed absorption-error-based replacement is still
needed.

### Follow-up: implemented problems 8 and 10

Problem 8 is now fixed. The raw-norm `sqrt(epsilon)` heuristic was removed.
Whenever fixed effects are present, the absorber first removes the global mean
(which is always in a nonempty categorical FE span) before choosing its
normalization scale. This makes both exact and iterative absorption invariant
to a large additive constant and improves the accuracy of the transformed
values, not merely the decision to accept them.

After absorption, each column's within norm is evaluated in those centered,
normalized input coordinates. Exact transforms use a floating-point gamma
bound derived from the largest relevant group accumulation (and the full
sample for multi-effect grand-mean work). Iterative transforms add a bound
derived from `demean_tol`; the existing normalized update and projection-error
requirements remain in force. Zero/non-finite or numerically unresolved
columns fail here, while `RegressionDesign` continues to diagnose collinearity
among accepted columns. The normalized within norms and the applied signal
floor are exposed in each cache group's linear-algebra diagnostics.

The original unit-FE example now estimates `0.23175043664900719` before the
shift and differs by only `3.08e-10` after adding `1e8` to the shock (the input
addition itself loses some low floating-point bits). Coefficients and HC0
covariance pass the permanent invariance comparison. A separate irregular
two-effect iterative test accepts meaningful within signal after the same
shift, while a column equal to an FE category plus `1e8` is still rejected as
absorbed numerical dust. Existing exactly absorbed and collinear-design tests
continue to fail at their intended boundaries.

Problem 10 is also fixed. Covariance validation now occurs in normalized
*original-feature* coordinates. On the QR path this first maps out of the Q
basis through `R^{-1}` but does not yet restore user measurement units; on the
Cholesky path it is already in the normalized feature basis. Validation is
therefore invariant to user rescaling without incorrectly treating Q-basis
diagonals as feature variances.

Multiway cluster covariance optionally propagates a componentwise cancellation
scale: each inclusion--exclusion component is sandwiched separately, converted
to an absolute contribution, and accumulated. That scale is conservatively
mapped through `abs(R^{-1})` on QR designs. The negative threshold uses the
standard gamma form `m*eps/(1-m*eps)`, with `m = 2*n + 4*k`, rather than an
absolute floor of one. Noncluster covariance uses the normalized matrix scale
with the same gamma multiplier.

A diagonal below the negative error bound raises. A negative diagonal within
the bound is set to exactly zero *in the returned covariance basis* before
unit restoration; its standard error is therefore coherently zero, and the
affected feature indices are recorded as `negative_variance_cleanup` in the
covariance diagnostics. No full PSD projection is performed. The original
two-way cluster reproduction now raises the same negative-variance error at
shock scales one and `1e9`, instead of returning a negative covariance with a
zero standard error at the latter scale. A controlled roundoff-negative test
verifies that the cleanup path returns covariance zero, standard error zero,
and the diagnostic marker together.

The cancellation bookkeeping has a measurable but limited multiway-cluster
cost. On a synthetic 140,000-observation, seven-regressor, 13-outcome,
three-way cluster calculation (seven inclusion--exclusion components), five
run medians with one thread were **959.1 ms** without the scale and
**1,005.2 ms** with it: **46.1 ms / 4.8%** overhead. One-way clustering adds
only one small sandwich operation; point estimation and noncluster covariance
paths do not pay this component-tracking cost.

Final verification: **243 tests passed** with the frozen environment and
optional Polars backend; the seven warnings are the expected few-cluster
warnings. All 256 independent full-dummy coefficient/HC0 comparisons, cluster
checks, 24 Bartlett/Parzen HAC/DK comparisons, and 12 QS comparisons pass.
Ruff check/format pass for the changed covariance and linear-algebra modules,
and `git diff --check` passes. Problems 8 and 10 are resolved. Of the original
review items, problem 7 remains open, while FE-aware HC2/HC3 remains explicitly
unsupported rather than silently incorrect.

### Follow-up: problem 7 options and performance estimate

No implementation changed in this follow-up. Problem 7 is a clock mismatch:
positive-horizon outcomes are matched at exact canonical time `t + h`, while
requested lags are currently produced by shifting rows within each unit. On
observed dates `[0, 2]`, row lag one at date 2 uses the date-0 value even though
the exact period `t - 1 = 1` is absent. The regression therefore conditions on
a two-period-old value under a one-lag name. Whether this changes the sample
and estimates depends on missing-period patterns, making the issue
methodological rather than merely cosmetic.

Four policies are possible. First, retain row lags and only document them;
this preserves speed but leaves leads and lags on different clocks. Second,
require every unit to have a consecutive clock whenever lags are requested and
reject gapped panels; row shifts are then exact but the restriction is
unnecessarily severe. Third, densify every unit onto a complete time grid and
then shift; this is conceptually simple but can consume extreme memory when
calendar ranges are sparse. Fourth, match every lag through exact
`(unit, canonical_time - lag)` keys, with a row-shift fast path only when the
panel is proven consecutive. This is the recommended default. A deliberately
named row-lag mode could remain as an opt-in if there is a real use case for
“previous observed record” controls, but it must not silently masquerade as a
calendar lag.

The canonical integer clock added for problems 6/9 makes exact lagging
straightforward and backend-consistent. For each distinct requested lag, build
or join one vector of source-row positions, cache it, and reuse it across every
outcome/shock/control requesting that lag. Missing keys mark the anchor invalid,
just as missing exact future keys do. Dense consecutive panels can retain the
current group-shift path because the two definitions are identical there.

A local pandas microbenchmark used 5,000 units, 28 periods, 140,000 rows, and
six requested lagged columns across three distinct lags (1, 2, and 4), with
seven measured repetitions after warm-up and one numerical-library thread.
Caching one exact index lookup per distinct lag produced these medians:

| Panel | Rows | Current row shifts | Exact cached lookup | Extra time |
| --- | ---: | ---: | ---: | ---: |
| Dense | 140,000 | 25.21 ms | 47.97 ms | 22.76 ms (1.90x) |
| Gapped | 137,855 | 24.65 ms | 49.07 ms | 24.42 ms (1.99x) |

The dense result estimates the generic indexed fallback; the recommended
implementation would keep the proven-consecutive shift fast path, so its
expected dense-panel hit is approximately zero beyond the already performed
clock/shape check. The gapped-panel estimate is about **24 ms additional lag
preparation** for this workload. Position-cache memory is one `int64` per row
per distinct lag: about **3.3 MiB** here, plus lagged output columns that both
methods already require. Exact lookup remains `O(n * distinct_lags)` and should
normally be a small fraction of FE absorption and covariance work. Tests must
verify gap-induced row removal, multiple sources sharing cached lag positions,
nonconsecutive lag grids, balanced/indexed and pandas/Polars parity, and an
explicit row-lag alternative if one is retained.

### Final recheck of all original findings

The complete suite, independent references, all adversarial reproductions, and
the 774-case solver grid were rerun after the problem-8/10 changes. No fixed
finding regressed. The final status is:

| Problem | Final status | Recheck evidence |
| ---: | --- | --- |
| 1. Absorbed FE residual degrees of freedom | Fixed | Classical and HC1 match explicit full-dummy OLS; saturated FE model rejects at model rank 4 of 4. |
| 2. FE leverage in HC2/HC3 | Safely contained, not implemented | FE HC2/HC3 fail immediately and recommend HC0/HC1; no-FE HC2/HC3 still match independent references. |
| 3. Normal-equation instability | Fixed | Original near-collinear error remains `7.57e-10`; 774-case stress grid confirms the 1,000 design-condition cutoff. |
| 4. Scale-dependent iterative absorption | Fixed | Rust and NumPy return the same coefficient and 13 iterations at scales one, `1e-8`, and `1e-12`. |
| 5. Incorrect QS estimator | Fixed | Reproduction is `0.005334232287937206` versus `0.005334232287937196`; linearmodels and all-pair tests pass. |
| 6. HAC time coercion collisions | Fixed | Fractional near-integer dates reject before covariance construction; integral strings and integers agree. |
| 7. Row lags versus calendar leads | **Open** | Reproduction still shows date 2 receiving date 0 as lag one when date 1 is absent. A solution is designed and benchmarked but not implemented. |
| 8. Raw-level-dependent within check | Fixed | Adding `1e8` now preserves the unit-FE coefficient (`3.08e-10` difference) and iterative FE signal; absorbed dust still rejects. |
| 9. Time representation erasing horizon zero | Fixed | One canonical clock is used in both backends; unbalanced numeric strings work, unsupported dates reject, and horizon zero self-maps. |
| 10. Unit-dependent negative-variance handling | Fixed | The same two-way cluster design raises at feature scales one and `1e9`; controlled roundoff cleanup is coherent and diagnosed. |

Verification results:

- **243 tests passed** with the frozen environment and optional Polars backend;
  seven expected few-cluster warnings.
- **256** explicit full-dummy coefficient/HC0 comparisons passed; maximum
  coefficient difference `4.08e-15`.
- Independent no-FE classical/HC0--HC3, native/fallback one- through four-way
  cluster, 24 Bartlett/Parzen HAC/DK, and 12 all-pair QS comparisons passed.
- The solver stress grid completed **774 cases**. At the production threshold
  1,000, maximum accepted Cholesky relative coefficient error was `4.29e-10`
  and covariance error `6.19e-10`; QR continues above that threshold.
- Cargo formatting, the lockfile check, Ruff checks for the new numerical and
  reference modules, and `git diff --check` pass.

Therefore the library is not yet “all clear” on every original point: exact
calendar lag construction (problem 7) still needs implementation. FE-aware
HC2/HC3 also remains intentionally unavailable rather than incorrectly
estimated. All other original numerical findings are fixed under the stated
contracts and have passed the final regression audit.

### Follow-up: problem 7 implemented and verified

Problem 7 is now fixed with exact calendar-time lag matching. A requested lag
`L` resolves only the key `(unit, canonical_time - L)`. If that key is absent,
the generated value is missing and the row cannot be an LP anchor. The original
gap reproduction now reports times `[0, 2]`, lag-one values `[nan, nan]`, and
validity `[False, False]`; date 0 is no longer treated as date 2's lag one.

The implementation has two explicit paths in both pandas and Polars:

- `consecutive_shift` is used only after proving that every adjacent row within
  every unit advances by exactly one canonical period. Ordinary shifts are
  mathematically identical to exact matching in that case.
- `indexed` is used for every gapped panel. It constructs one source-position
  vector per **distinct lag**, caches that vector, and reuses it for every
  outcome, shock, or control requesting the same lag. Subtraction at the lower
  `int64` boundary is guarded, so an impossible prior date becomes an unmatched
  key rather than wrapping around.

The chosen path is exposed as `demeaning_diagnostics_["lag_path"]`, including
`none` when no lags are requested. The fit contract now states exact
`time - lag` behavior. The unbalanced-lag regression test was changed from an
old row-position reference to an independent keyed-calendar OLS reference; it
requests several sources and lags, exercises cache reuse, and asserts the
indexed path. Dense-panel tests assert the consecutive fast path. Existing
gapped pandas/Polars comparisons exercise backend parity, and the standalone
problem reproduction was updated to the exact-lag API.

An apples-to-apples local benchmark used 5,000 units, 28 periods, 140,000 rows,
and six lagged features sharing three distinct lags (1, 2, and 4). Medians are
from seven measured repetitions after warm-up:

| Panel | Rows | Previous shifts | Implemented path | New time | Difference |
| --- | ---: | ---: | --- | ---: | ---: |
| Dense | 140,000 | 42.80 ms | consecutive shift | 45.20 ms | +2.40 ms (1.06x) |
| Gapped | 139,850 | 48.99 ms | cached indexed | 49.49 ms | +0.50 ms (1.01x) |

These measurements include output-frame construction in both cases. They
confirm that the proof required for the dense fast path has a small cost and
that caching by distinct lag keeps exact matching competitive on the gapped
case. Timings are environment-specific; the semantic result does not depend on
which path is selected.

Final verification after the fix:

- **243 tests passed**, with the same seven expected few-cluster warnings.
- The original reproduction suite passes and shows exact missing-lag behavior.
- All **256** explicit full-dummy coefficient/HC0 comparisons still pass, with
  maximum coefficient error `4.08e-15`.
- Independent no-FE classical/HC0--HC3, one- through four-way cluster,
  24 Bartlett/Parzen HAC/DK, and 12 all-pair QS comparisons still pass.
- `git diff --check` passes.

Problem 7 is therefore closed. All ten original findings are now either fixed
or, for FE-aware HC2/HC3 in problem 2, fail immediately with the explicit
recommendation to use HC0 or HC1 instead of returning incorrect estimates.

### Second review: adversarial coverage after the original fixes (2026-09-20)

This addendum supersedes any reading of the earlier closing statement as an
all-clear for the library. The ten original findings have dispositions, but a
fresh review found additional issues. Production code was not changed in this
pass. Reproductions are in `docs/review/second_pass_checks.py`; run with
`UV_CACHE_DIR=/tmp/uv-cache uv run python docs/review/second_pass_checks.py`.
That script is an exploratory runner: it prints exceptions and continues, so
its exit status alone is **not** a regression-test success criterion.

Scope: both frame adapters and exact clock conversion; lag/lead construction;
common/per-horizon and cumulative samples; singleton pruning; exact/iterative
absorption and Rust kernels; FE rank and cluster correction counts; Cholesky/QR
solving; classical, HC, cluster, HAC and DK covariance; confidence intervals;
batching and numerical resource use; result assembly. Plotting was inspected
only for how it consumes fitted estimates and intervals. UX, prose quality,
packaging and release engineering are outside this estimation review.

#### 11. High: cumulative prefix subtraction can silently change outcomes

Location: `src/fastlp/estimator.py`, outcome-prefix construction and the dense
cumulative response branch (around lines 496 and 673).

The dense implementation forms a prefix sum over the entire unit history and
subtracts prefixes to obtain a requested window. A large value **outside** the
window can erase smaller subsequent observations in both prefixes. This is a
numerical error in the dependent variable, before OLS even starts.

Reproduction: four units, 20 periods, otherwise ordinary random data; replace
each unit's first outcome with `1e18`, request `shock_lags=1` so those first
observations cannot be anchors, and fit horizon zero. Level and cumulative
responses must then be identical. Instead their coefficients differ by up to
`0.22917467831084332`, and residuals by `2.34587940317256`. The dense cumulative
path has lost the small retained outcomes. This is not unavoidable loss in the
supplied data: those small outcomes remain exactly available in their own rows.

Recommended fix: use the original outcome directly for cumulative horizon
zero, and accumulate requested forward windows without subtracting a prefix
that contains unrelated history. An incremental forward accumulation over
horizons can reuse work; compensated summation may help within a window but
compensated prefixes alone do not guarantee safe prefix subtraction. Test
dense/indexed parity, excluded large early observations, mixed-sign outcomes,
common/per-horizon masks, lags, and batch-size invariance against direct sums.

#### 12. High: nonfinite inference can escape as a successful fit

Locations: squared residuals in `_covariance.py`; transformed covariance,
negative-diagonal validation, and final result assignment in `estimator.py`.

Input finiteness does not imply intermediate or output finiteness. With the
ordinary reproduction data's outcome multiplied by `1e200`, HC0 fitting
returns finite coefficients but nonfinite covariance and confidence intervals.
Runtime warnings report overflow in squaring and invalid matrix products;
the estimator nevertheless publishes fitted results. Comparisons such as
`diagonal < -tolerance` do not catch NaN.

The variance in this particular example exceeds float64 range, so the expected
behavior is a clear numerical-range error, not a promise to represent it.
Recommended fix: explicitly certify finite transformed outcomes, coefficients,
residuals, covariance, standard errors and intervals before publishing a fit;
use scaling where it can prevent avoidable intermediate overflow/underflow.
Add extreme-scale tests for all covariance families and both backends,
distinguishing unrepresentable answers from avoidable intermediate failures.

#### 13. Medium: exact lag joins break categorical Polars unit IDs

Location: `PolarsFrame.add_lags`, construction of the left join frame around
`_frame.py:363`.

The lookup retains the original Polars unit dtype, but the left frame rebuilds
units from a NumPy array. A categorical unit column becomes String on the
left, while remaining Categorical on the right. On a gapped panel with
`outcome_lags=1`, fitting raises `SchemaError: datatypes of join keys don't
match`. This affects a normal panel-ID representation and is a gap in the
problem-7 backend parity coverage.

Recommended fix: construct the left frame from the original unit Series and
attach only the computed target-time Series. Test Categorical and Enum IDs,
String and integer IDs, eager/lazy input, gaps, consecutive panels, and
temporary-column name collisions. Preserve dtype rather than coercing both
keys through a potentially lossy generic representation.

#### 14. Medium: QS weights lose accuracy when lag/bandwidth is very small

Locations: `_kernel_weight` and the blocked `_quadratic_spectral_meat` formula
in `_covariance.py`.

The corrected QS formula still evaluates a cancellation-prone expression,
`3 * (sin(z)/z - cos(z)) / z**2`. At lag one and bandwidth `10**9`, the scalar
helper returns **0.0**. Expanding this expression directly gives
`1 - z**2/10 + z**4/280 - ...`, so this weight should be approximately one.
The blocked implementation uses the same vulnerable formula. The ordinary
bandwidth comparisons in the original review do not exercise this limit.

Recommended fix: share a stable weight implementation with a small-argument
Taylor branch and the existing expression elsewhere. Validate continuity at
the branch threshold, large bandwidths, zero lag, and FFT/blocked agreement
against a higher-precision or analytic-series oracle. This is an extreme
bandwidth case, not evidence that the previous ordinary-bandwidth QS results
are incorrect.

#### 15. Medium: numeric model conversion precedes preservation of panel keys

Locations: `PandasFrame.prepare` and `PolarsFrame.prepare` numeric casts before
canonical time conversion.

A column can serve both as a regressor and as the time identifier. Both
adapters cast model columns to float64 before canonicalizing the time column.
For valid integer dates beginning at `2**53`, using `shock='time'` therefore
rejects the clock after it has been converted to float. The original exact
integers should have remained available for key matching. Similar overlapping
roles for numeric model columns and categorical identifiers require auditing
for category collapse; that broader effect is a code-inspection concern, not
an additional demonstrated silent failure in this pass.

Recommended fix: preserve exact key/category representations independently
from floating numerical design arrays, or explicitly reject unsupported role
overlaps before casting. Regressor conditioning should be evaluated separately
from clock validity. Test large integer clocks and IDs with overlapping roles,
and verify panel uniqueness before and after model preparation.

#### 16. Medium: pandas silently discards imaginary outcome components

Location: `PandasFrame.prepare`, `.astype(np.float64)` numeric conversion.

Give the outcome the values `original_y + 10j`: fitting succeeds after a
`ComplexWarning` and estimates the real-valued outcome. This silently changes
the supplied model (the warning can be filtered). Real-valued regression is a
reasonable contract; silently replacing complex data with its real component
is not reliable enforcement of that contract.

Recommended fix: explicitly reject complex outcomes, shocks and controls
before conversion. Cover pandas, named NumPy arrays, and backend-specific
unsupported-dtype handling, with a clear real-numeric-data error.

#### 17. Low: accepted oversized lags raise implementation-level overflow

Locations: lag validation in `estimator.py`; lag subtraction and shifts in
both frame adapters.

The public lag validator accepts arbitrary positive Python integers. With a
gapped ordinary panel, requesting lag `2**63` or `2**65` reaches NumPy arithmetic
and raises `OverflowError: Python int too large to convert to C long` rather
than producing a missing-match result or a deliberate argument error.
The problem-7 underflow mask does work for ordinary lag magnitudes near the
lower int64 clock boundary, but does not handle an oversized lag operand.

Recommended fix: explicitly define and enforce a lag range, or implement
bounded target construction that handles the full supported clock difference.
Test very large lags in both the consecutive and indexed paths, including a
lag larger than all observed within-unit spans. Do not blindly discard all
lags above int64 maximum if supporting the entire int64 clock span is intended:
two valid dates can differ by more than that amount.

#### 18. Low: numerical option validation and extreme confidence levels

The constructor accepts `demean_tol=nan`, `demean_tol=inf`, `max_iter=1.5`, and
boolean horizons. Infinite tolerance on the reproduced iterative FE panel
eventually raises a misleading negligible-variation error. Backend-specific
conversion or iteration errors are possible for malformed iteration counts.
These are validation gaps, not demonstrated silent coefficient errors here.

Separately, valid `alpha=1e-20` fails after estimation because
`1 - alpha/2` rounds to one and `NormalDist.inv_cdf` raises `StatisticsError`.
Recommended fix: require finite positive tolerance and integer iteration
counts; decide boolean/integer conventions consistently; evaluate upper-tail
critical values through inverse survival functions using `alpha/2` directly,
with explicit supported-range checks. Add constructor and full-fit tests.

#### 19. Resource limits: batching does not bound all estimation allocations

This item is established by code inspection, without deliberately triggering
an out-of-memory event. `memory_budget` determines the outcome batch size;
the full `n_rows * (horizons + 1)` lead-position array, masks, requested lag
columns/cache, and retained results are still allocated independently. The
indexed cumulative branch additionally constructs an `n_anchors * (h+1)`
temporary even for a one-horizon batch. Bartlett/Parzen pair construction loops
through every requested bandwidth, including lags beyond the observed span.
The irregular QS fallback has bounded pair-block storage but quadratic work
within groups. A budget smaller than one outcome workspace is rounded up to
one batch without enforcing that byte limit.

Recommended action: define whether the option is a workspace hint or a strict
memory cap. For a strict cap, budget persistent arrays and temporaries before
allocation, stream/chunk alignment and cumulative construction where needed,
and fail early when unavoidable storage exceeds the limit. Skip impossible
HAC pair lags while preserving the requested bandwidth in the kernel weights.
Benchmark large horizon grids, sparse dates, long irregular groups and many
distinct lags; report both peak memory and time. This is an availability and
scalability issue rather than an established wrong-answer case.

#### Coverage evidence and remaining limits

Current checks still pass:

- **243 tests**, with seven expected few-cluster warnings.
- **256** independent full-dummy coefficient/HC0 comparisons; maximum
  coefficient difference `4.08e-15`.
- Independent no-FE classical/HC0--HC3, one- through four-way cluster, 24
  Bartlett/Parzen HAC/DK and 12 all-pair QS comparisons.
- **288 new lag-feature comparisons** against a Python-integer dictionary
  oracle: randomly gapped and shuffled panels, two sources, three lags,
  pandas/Polars, and clocks near both int64 boundaries. These pass. They do
  not cover categorical Polars IDs or oversized lag operands, which fail as
  documented above.

| Estimation area | Current evidence | Further acceptance coverage needed |
| --- | --- | --- |
| Key preparation and matching | Canonical clock tests; random exact-lag oracle | Categorical/Enum keys, overlapping roles, huge lag operands, unsupported numeric types |
| Sample masks and responses | Existing common/per-horizon, singleton and cumulative references | Large dynamic-range cumulative windows; retained-sample and path invariance |
| FE absorption | Explicit-dummy references; native/fallback checks; prior scale tests | Weakly connected large FE graphs; stronger independent projection-error certification near tolerance |
| Rank and correction counts | Prior exact/full-dummy and nesting tests | More adversarial 3+ FE topology; conservative-bound behavior at practical resource limits |
| Regression solving | Prior 774-case Cholesky/QR stress grid; current reference tests | Extreme exponent ranges and nonfinite-output rejection; this pass did not rerun the 774-case grid |
| Covariance | Current independent classical/HC/cluster/HAC/DK checks | Stable small-argument QS weights; overflow/underflow; large-bandwidth resource limits |
| Inference and returned results | Existing cluster t/normal and reference checks | Extreme alpha, final finiteness, failure-state semantics on refit, parameter mutation after fit |
| Resource use | Inspection of allocations and batching | Measured peak-memory tests and an explicit memory-budget contract |

The method-level limitations from the first review still apply: sandwich
covariance does not establish shock identification or remove dynamic-panel
bias; intervals are pointwise; HC2/HC3 with absorbed FEs remain explicitly
unsupported; the selected cluster conventions and conservative rank mode are
deliberate policies, not universal finite-sample guarantees. Multiway cluster
covariance can be indefinite even when all marginal variances are positive;
the library's negative-diagonal checks do not certify arbitrary joint Wald
inference. No new error in that stated behavior was established here.

Priority: address 11 and 12 first, then the categorical lag regression (13),
followed by stable QS weights and input-role/type preservation (14--16).
Resolve 17--18 with explicit validation contracts and 19 with resource tests.
Promote each reproduction to a permanent regression test when implementing its
fix. This review covers each production estimation component, but does not
claim exhaustive input enumeration, formal verification, or 100% statistical
accuracy for every application.

### Follow-up: problems 11--13 implemented and benchmarked

Problem 11 now constructs cumulative outcomes by adding forward from each
retained anchor. It uses the same calculation for dense and indexed panels,
stores one running-sum vector per sample-mask group, and preserves that vector
across horizon batches. A horizon-zero response is the original outcome;
observations before the anchor never enter the accumulation. Different sample
groups initialize their own sums. This removes both prefix subtraction and the
indexed path's full window matrix temporary. It does not promise exact
arithmetic for extreme cancellation *within* a requested window; compensated
summation is not part of this change.

Problem 12 now checks finiteness of generated outcomes, transformed outcomes,
coefficients, residuals, normalized and original-unit covariance, standard
errors, and confidence intervals. A failure raises a numerical-range error
with the failed stage and a rescaling suggestion. Confidence intervals are
computed and checked before fitted arrays are assigned to the estimator.
These checks reject nonfinite results; they do not introduce higher precision,
rescale all covariance calculations, or certify against finite underflow.
The independent issue of retaining an earlier successful fit after a failed
refit is not changed by this work.

Problem 13 now constructs the Polars lag-join left side using the original unit
column, adding only the calculated target date. Categorical and Enum dtypes
therefore survive and agree with the lookup's join-key dtype.

Verification: **268 tests passed**, including 25 new permanent regressions in
`tests/test_review_followups.py`. They cover direct-sum/independent OLS and HC0
references for large excluded history, dense/gapped samples, common/per-horizon
selection, one-horizon/default batches, overflow rejection for all eight
covariance families and cumulative outcomes, and categorical/Enum Polars IDs
with eager/lazy and dense/gapped input. The 256 full-dummy comparisons and
independent classical/HC, multiway cluster, Bartlett/Parzen and QS checks still
pass. There are seven expected few-cluster warnings. `git diff --check` passes.

#### Measured performance

Reproducible benchmark: `docs/review/cumulative_finite_benchmark.py`. It measures
complete fits on 3,500 units x 40 periods (140,000 rows), horizons 0--12,
cumulative responses, an intercept and one shock, no FEs, HC1, and one
BLAS/OpenMP thread. The gapped case removes 500 observations. Results below
are medians of seven measured fits after warm-up, with variant order randomized.
The old method is reconstructed in a benchmark-only builder: cached dense
prefix subtraction or indexed direct-window matrices. Both old and new use
the current remaining estimator machinery; old checks are disabled. Ordinary
data coefficient agreement is asserted to absolute tolerance `1e-12`.

| Panel/sample | Old cumulative, no new checks | New cumulative plus checks | Change |
| --- | ---: | ---: | ---: |
| Dense/common | 95.49 ms | 104.92 ms | +9.9% |
| Dense/per-horizon | 275.65 ms | 367.58 ms | +33.3% |
| Gapped/common | 257.13 ms | 155.05 ms | -39.7% |
| Gapped/per-horizon | 475.87 ms | 421.13 ms | -11.5% |

The dense per-horizon slowdown is meaningful: changing samples require separate
forward sums for each mask group, while the old, inaccurate prefix method
could answer each window by two lookups. The gapped path benefits from removing
the repeated window-matrix construction. These are workload-specific results,
not a universal performance guarantee, and the chosen benchmark does not
quantify memory peaks or Polars performance.

To isolate problem 12, the benchmark also measures elapsed time inside every
new finiteness check. Their total medians were **2.55, 2.44, 2.07, and 2.41 ms**
respectively, about **0.6--2.4%** of complete new-fit times. Whole-fit medians
with checks disabled were 113.84, 376.89, 152.96, and 436.16 ms; run-to-run noise
is larger than the checks' cost in several comparisons, so negative apparent
overhead must not be interpreted as a speed benefit. Directly timed scans
support the expectation of a small cost on this workload. Problems 11--13 are
resolved under these tested contracts; findings 14--19 remain open.

### Follow-up: problems 14 and 16 implemented

Problem 14 now uses one shared QS weight evaluator for scalar/FFT and blocked
array paths. For `abs(z) < 0.1` it evaluates
`1 - z**2/10 + z**4/280 - z**6/15120 + z**8/1330560` using nested multiplication;
the first omitted term is below `6e-19` in that region. Outside that region it
retains the trigonometric formula. The diagonal weight is exactly one. Lag
one at bandwidth `10**9` now returns **1.0**, instead of zero. The existing
zero-bandwidth contract remains unchanged.

New tests compare scalar and array weights to an independent 60-digit Decimal
series oracle, including zero, tiny arguments and both sides of the branch
threshold. Separate all-pair Decimal-reference tests cover ordinary and very
large bandwidths for the dense FFT and irregular blocked meat calculations.
The existing linearmodels and end-to-end QS references remain passing.

Problem 16 now checks the dtype returned by pandas numeric conversion before
casting to float64. Complex outcomes, shocks and controls are rejected with
`outcome, shock, and controls must be numeric (real values only)`. This covers
native complex columns, object-backed complex values, and named NumPy input.
Polars already rejects unsupported complex Object conversion; its public error
now uses the same real-numeric contract. No per-row complex scan was added.

Performance: the dtype check costs one metadata check per numeric model column.
The stable QS branch introduces a small cost, not a guarantee of zero overhead.
A local microbenchmark measured the complete QS meat calculation on 500 units
x 80 periods, four score columns, bandwidth four, one BLAS/OpenMP thread, and
15 repetitions after warm-up with randomized old/new order. The old weight
formula was substituted into the current calculation for comparison:

| Grid | Old weight formula | Stable weight formula |
| --- | ---: | ---: |
| Dense, FFT | 11.592 ms | 10.861 ms |
| Irregular, blocked pairs | 128.140 ms | 134.591 ms |

The dense difference is consistent with measurement noise, not a claimed
speedup. The irregular result suggests about **5%** additional covariance-kernel
time (**6.45 ms**) for that workload; it is not a full-fit overhead estimate.
The change does not alter asymptotic work or introduce a new full-panel pair
matrix. These local timings are indicative, not a performance guarantee.

Verification: **283 tests passed**, including 15 new regressions in
`tests/test_qs_and_real_inputs.py`, with seven expected few-cluster warnings.
An initial full-suite run exposed a test depending on the phrase "must be
numeric"; that phrase was preserved while adding the real-values requirement,
and the final suite passes. Problems **14 and 16 are resolved**. Problems
**15, 17, 18 and 19 remain open**.

### Follow-up: test-suite size and redundancy assessment

Inspected the collected cases and parameter grids in response to the concern
about suite size. No tests were removed in this assessment. There are **83 test
functions**, expanded by parameterization into **283 cases**, across 2,242 lines
of test code. A timed run passed all cases in **6.81 seconds**. The slowest
individual case was plotting at 0.86 seconds; the other seven reported slowest
cases each took approximately 0.09--0.10 seconds. Runtime does not currently
justify a separate slow-test scheduling system, but redundant cases and weak
checks can be trimmed for maintenance clarity.

Concrete candidates:

- In the solver condition grid, `k=1` produces the same design for all six
  requested condition numbers: a single singular value cannot encode those
  ratios. Keep one case, removing five exact repetitions.
- In the cluster reference grid, `min` and `component` group adjustment give
  the same mathematical correction for a single clustering term. Four of the
  five specifications have a single term (including the interaction tuple).
  Across three parameter-count policies, twelve repeated comparisons can be
  collapsed. Retain both adjustments for the genuine multiway case and a
  targeted one-way equivalence assertion if desired.
- The nine complex-input cases cross three model roles with three input
  representations. A smaller explicit set can exercise all three roles and
  all conversion routes; five cases is a reasonable starting point. This
  reduces combinations, so it should not be described as literally identical
  coverage. Retain the independent Polars rejection test.
- Seven older covariance smoke cases mostly assert finiteness. Their metadata
  assertions should be moved into appropriate stronger reference tests before
  considering removal; public-method smoke coverage should not simply vanish.

Keep independent numerical oracles, every distinct historical bug regression,
native/fallback behavior, and interactions that change the actual computation
(sample masks, cumulative batching, FFT versus blocked QS, and categorical
join versus shift paths). Shared fixtures and grouping tests by subsystem
rather than review number can reduce maintenance without reducing coverage.
Recommendation: a modest targeted cleanup, starting with the seventeen clearly
redundant parameter combinations, rather than targeting an arbitrary total
test count or deleting correctness references to make the suite look smaller.

### Follow-up: targeted test trim implemented

Reduced the suite from **283 to 255 cases** (28 fewer):

- Removed five identical one-regressor solver-condition combinations. All
  multi-regressor condition cases remain.
- Removed twelve redundant one-term cluster adjustment comparisons. Both
  adjustment policies remain tested for genuine multiway clustering, and every
  parameter-count policy and cluster specification remains represented.
- Reduced nine complex-input combinations to five: all three model roles use
  native complex dtype, while the outcome also exercises object and named
  NumPy conversion. The Polars complex-input rejection remains separate.
- Removed seven finiteness-only covariance smoke cases after moving their
  finiteness/configuration/diagnostic-length assertions into existing public
  QR numerical-reference tests and HAC/DK full-model reference tests.

Independent numerical references and historical bug regressions remain. The
complex-input grid deliberately no longer crosses every representation with
every role, and the removed smoke scenarios are replaced by assertions in
stronger existing scenarios rather than preserved as identical input cases.
No production code changed. The trimmed suite passes **255 tests**, with seven
expected warnings, in **6.37 seconds** on this run; the prior 6.81-second run
is only an indicative comparison, not a controlled speed benchmark.
`git diff --check` passes. No new test-run tiers or fixture framework was added.

### Follow-up: exact identifiers (15) and allocation planning (19)

**Problem 15 is fixed.** Both adapters canonicalize time before any floating
numeric validation and preserve the original remaining columns. Numeric model
values are validated through temporary conversions, then cast when extracting
regression arrays. Thus an integer identifier can also be an outcome, shock,
control, FE or cluster column without the floating model conversion overwriting
its grouping/matching representation. Numeric-looking category strings such as
`"01"` and `"1"` also remain separate partitions. Large model values still face
ordinary float64 precision/conditioning limits in the regression itself; exact
identifiers do not imply arbitrary-precision OLS.

**Problem 19 now has conservative preflight planning and bounded-workspace
paths.** `_memory.py` sizes selected inputs and reserves frame copies, lag
columns, alignment/masks/sample IDs, retained residuals/results, and numerical
workspaces before panel preparation. Remaining space determines horizon batch
size. A budget below the planned one-horizon minimum raises a clear error
before lag/lead construction. The allocation plan is exposed through
`memory_diagnostics_`, including minimum/reserved/per-horizon/planned-peak bytes,
batch size and whether HAC pairs are streamed.

The resource changes are:

- Lag construction still builds each distinct lag lookup once and shares it
  among sources, but releases its position vector before the next distinct lag.
- With enough budget, Bartlett/Parzen HAC retain pair caching. Under tight
  budgets they generate one pair vector at a time. Pair generation stops at
  the observed time span; the original bandwidth remains in kernel weights
  and diagnostics. A bandwidth of `10**9` over twelve observed consecutive
  dates therefore only generates twelve lag vectors.
- Cumulative outcomes already use a running vector across batches from the
  problem-11 fix; their old window-matrix allocation is absent.
- Large lead/mask/result arrays are reserved up front. If those do not fit,
  the fit fails early rather than attempting to stream every kind of result.
  `retain_residuals=False` reduces the planned persistent storage.

The contract is intentionally precise: **the budget bounds a conservative
allocation plan, not measured process RSS or a formally certified peak of every
allocation**. Caller-owned input, Python object/allocator overhead and hidden
third-party workspaces are excluded. Numerical reserves include generous
headroom and can reject jobs that would happen to fit in less memory. Lazy
Polars sizing requires an additional aggregate evaluation of its query; an
upstream lazy operation can itself allocate memory. Strict OS-level isolation
and exhaustive allocator instrumentation are not implemented. Irregular QS
still has quadratic computational work, and sparse very-wide calendars can
still make pair enumeration expensive; this budget is not a runtime limit.
These remaining limits must not be described as a universal resource guarantee.

This deliberately changes the previous behavior of `memory_budget=1` and
`"1KB"`, which formerly forced one-horizon batches while ignoring the rest of
the memory footprint. Tests and the independent-check script now request one
batch explicitly when testing batching alone; the memory test uses the actual
planned minimum. The quickstart and fit contract describe the new semantics.

Verification: **262 tests passed**, with six expected few-cluster warnings;
only seven new focused cases were added after the preceding suite trim. Tests
cover exact large identifiers and overlapping numeric categories in both
backends, rejection before panel preparation, the exact minimum/one-byte-below
boundary, tight-budget estimation parity, eager/lazy Polars string-valued model
columns, and HAC large-bandwidth agreement with an independent all-pair
reference. All 256 independent full-dummy checks, independent classical/HC,
cluster, Bartlett/Parzen and QS checks pass. `git diff --check` passes.

#### Performance and memory measurements

Reproducer: `docs/review/key_memory_benchmark.py`; medians of seven complete
fits after warm-up, randomized variant order, one BLAS/OpenMP thread, two shock
lags and no retained residuals. Coefficients and covariance are checked against
the ample-budget result. Times compare the new code with and without a budget;
they do not isolate the identifier-preservation change against the old code.

| Workload | Unbudgeted | Ample budget (1 GiB) | Minimum planned budget |
| --- | ---: | ---: | ---: |
| HC1, 50,000 rows, horizons 0--8 | 55.09 ms | 53.34 ms | 32.61 ms |
| HAC, 10,000 rows, horizons 0--4 | 27.24 ms | 26.81 ms | 71.30 ms |

Ample-budget accounting has no measurable slowdown here. Tight HAC execution
is approximately **2.6x slower**, because one-horizon batches stream/recompute
pairs. Tight HC1 happens to be faster on this shape; small batches can improve
working-set behavior, so performance is not monotonic in allowed memory.

Separate `tracemalloc` runs (not timed runs) recorded:

| Workload | Ample traced peak | Tight traced peak | Tight planned peak |
| --- | ---: | ---: | ---: |
| HC1 | 26.79 MiB | 18.78 MiB | 106.07 MiB |
| HAC | 4.71 MiB | 4.45 MiB | 20.61 MiB |

Traced peaks are instrumentation observations, not total process RSS or proof
of the plan's bound across all inputs/backends. The substantially higher planned
numbers show the deliberate conservatism. Problem 15 is closed; problem 19's
silent budget overrun is addressed at the stated allocation-plan level, with
the strict-RSS and runtime limitations explicitly retained. Problems 17 and
18 remain open.

### Follow-up: oversized lags and numerical options (problems 17 and 18)

**Problem 17 is fixed under an explicit lag range.** Public lag specifications
and both frame adapters now reject lag numbers larger than the signed `int64`
maximum with `ValueError` before handing them to NumPy, pandas or Polars.
Scalar lag settings expand to every lag from one through the supplied count,
so scalar counts above 10,000 also fail early with a suggestion to provide a
sparse lag sequence. That count limit prevents an enormous tuple/feature grid
from being constructed during argument normalization. An explicit sparse lag
as large as `2**63 - 1` remains valid. On a consecutive panel, the shift
operand is capped at the panel row count: every larger shift is identically
missing, and this avoids pandas' own overflow on an otherwise supported lag.
The indexed path still matches exact `(unit, time - lag)` keys at the lower
`int64` date boundary. The possible distance of `2**64 - 1` between the two
most extreme valid dates is outside the newly stated lag range; no claim of
support for that rare pair is made.

**Problem 18 is fixed for the identified option and interval failures.** The
constructor now rejects boolean or fractional `horizons`, `hac_lags` and
`max_iter`; `demean_tol` must be finite and positive. It accepts ordinary
NumPy integer scalar counts. Confidence critical values now use the lower
tail directly: `norm.isf(alpha / 2)` and `t.isf(alpha / 2, df)`. This avoids
rounding `1 - alpha / 2` to one when `alpha` is small. Alpha values whose half
tail underflows to zero in float64 fail at construction with a clear error;
nonfinite critical values from a reference distribution fail before fitted
results are published. At `alpha=1e-20`, both normal and cluster Student-t
intervals match independently evaluated SciPy tail quantiles and are finite.
The usual `alpha=0.05` results remain within their existing test tolerances.

Four focused tests cover early lag rejection, both pandas and Polars at the
largest supported lag on indexed and consecutive panels, constructor options,
and tiny-alpha normal and cluster intervals. The dense-path test caught an
additional pandas overflow on a huge *valid* shift operand; the row-count cap
resolved it. The full suite now passes **266 tests**, with six expected
few-cluster warnings. All 256 independent full-dummy coefficient/HC0 checks,
the independent classical/HC and cluster checks, 24 Bartlett/Parzen HAC/DK
checks, and 12 QS checks pass. The second-pass adversarial runner now prints
the expected clear lag/tolerance errors and finite tiny-alpha intervals;
because it catches exceptions for exploration, its exit code is not treated
as a formal test result. `git diff --check` passes.

The changes add constant-time validation per requested lag/option and a
constant-time shift cap per lagged feature. Tail quantiles are evaluated once
per horizon after estimation. No additional per-observation work was added,
so a material fit-time cost is not expected. No new benchmark was needed to
establish this complexity claim. Problems **17 and 18 are closed**. Across all
19 Astra findings, 15 numerical/contract findings plus these two are fixed;
problem 2 remains safely rejected for FE-aware HC2/HC3, and problem 19 remains
addressed only at the documented allocation-plan level rather than a strict
process-memory cap.
