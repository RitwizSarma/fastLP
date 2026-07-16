# Computational Aspects of Local Projections with High-Dimensional Fixed Effects

## 1. Introduction

Local projections (Jordà, 2005) estimate impulse responses by running a sequence of horizon-specific regressions. While the econometric formulation is straightforward, the computational cost can become substantial when the sample is large and the specification includes high-dimensional fixed effects (HDFE).

Modern estimation packages such as **reghdfe** (Stata), **fixest** (R), and **pyfixest** (Python) achieve remarkable performance by exploiting results from sparse linear algebra, iterative projection methods, and the Frisch–Waugh–Lovell theorem. Understanding these algorithms is useful both for selecting software and for identifying opportunities to accelerate local projections beyond existing implementations.

---

# 2. OLS as a Linear Algebra Problem

Consider

[
y = X\beta + \varepsilon,
]

where

* (X \in \mathbb{R}^{N\times K}),
* (y \in \mathbb{R}^{N}).

The normal equations are

[
X'X\hat\beta = X'y,
]

giving

[
\hat\beta=(X'X)^{-1}X'y.
]

Computationally, OLS consists of three operations:

1. Form (X'X).
2. Solve the linear system.
3. Compute residuals.

For small (K), the dominant cost is usually forming the cross-products, which is (O(NK^2)).

---

# 3. Why Dummy Variables Become Impossible

Suppose we include

* 100,000 firm effects,
* 1,000 time effects,
* 500 industry effects.

The design matrix becomes

[
X =
[X_0; D],
]

where

(D) contains hundreds of thousands of dummy variables.

Even though each row contains only a few nonzero entries, explicitly constructing

[
D'D
]

is computationally and memory intensive.

This is fundamentally a sparse linear algebra problem.

The dummy matrix is extremely sparse:

* each observation belongs to exactly one firm,
* one year,
* one industry.

The matrix therefore contains almost entirely zeros.

Modern HDFE estimators never construct this matrix explicitly.

---

# 4. Frisch–Waugh–Lovell

Instead of estimating

[
y=X\beta+D\alpha+\varepsilon,
]

FWL shows that

[
\hat\beta
=========

(\tilde X'\tilde X)^{-1}
\tilde X'\tilde y,
]

where

[
\tilde X=M_DX,
]

[
\tilde y=M_Dy,
]

and

[
M_D
===

I-D(D'D)^{-1}D'.
]

The matrix

[
M_D
]

is the orthogonal projector onto the space orthogonal to the fixed effects.

Unfortunately,

[
M_D
]

is itself enormous.

The challenge therefore becomes:

> How do we apply (M_D) without ever constructing it?

---

# 5. Alternating Projections

Suppose there are two fixed effects:

* firm
* year

Removing firm effects means

[
x
\leftarrow
x-\bar x_{\text{firm}}.
]

Removing year effects means

[
x
\leftarrow
x-\bar x_{\text{year}}.
]

Doing this once is insufficient because removing year effects generally reintroduces firm means.

Instead we iterate

```text
Firm demeaning

↓

Year demeaning

↓

Firm demeaning

↓

Year demeaning

↓

...
```

until convergence.

This is the **method of alternating projections** (MAP), rooted in the von Neumann alternating projection theorem.

The algorithm converges to the projection of the vector onto the intersection of the orthogonal complements of all fixed-effect spaces.

---

# 6. Kaczmarz Interpretation

The same problem can also be viewed as solving a sparse linear system.

One classical algorithm is the **Kaczmarz method**, introduced in 1937.

Rather than solving

[
Ax=b
]

directly, Kaczmarz repeatedly projects the current iterate onto one equation at a time.

Modern randomized Kaczmarz algorithms have convergence guarantees proportional to the condition number of the system.

Correia (2016) shows that the HDFE problem can be formulated in this framework and proposes symmetric projection schemes with improved convergence properties.

This perspective links fixed-effect estimation directly to numerical linear algebra.

---

# 7. Graph-Theoretic Interpretation

An alternative viewpoint is graph theoretic.

Each observation connects several fixed-effect groups.

For example

```text
Observation

Firm 104

Year 2018

Industry 7
```

defines edges in a bipartite graph.

Estimating fixed effects becomes equivalent to solving a Laplacian system on this graph.

Recent HDFE algorithms exploit nearly-linear-time Laplacian solvers and graph preconditioners to accelerate convergence.

This is one reason modern implementations scale to millions of observations.

---

# 8. reghdfe

`reghdfe` implements the algorithms developed by Sergio Correia.

Its main components include

* iterative within-transformation,
* alternating projections,
* graph simplification,
* singleton removal,
* iterative Krylov solvers (LSMR/LSQR) for difficult problems,
* sparse bookkeeping in Mata.

Rather than estimating dummy coefficients,

it repeatedly residualizes variables until convergence.

The resulting transformed data satisfy

[
\tilde X=M_DX,
]

without ever forming

[
M_D.
]

---

# 9. fixest

`fixest` follows the same econometric principle but uses a different software architecture.

Its implementation emphasizes

* highly optimized C++ code,
* contiguous memory layouts,
* cache-friendly operations,
* OpenMP multithreading,
* efficient storage of factor indices,
* aggressive reuse of intermediate objects.

The demeaning algorithm is based on Laurent Berge's work on efficient multi-way fixed-effect estimation and employs iterative projection techniques similar in spirit to those used by `reghdfe`.

A key practical advantage is that multiple regressions sharing identical fixed effects can reuse substantial preprocessing.

---

# 10. pyfixest

`pyfixest` ports much of the `fixest` philosophy to Python.

The package combines

* NumPy,
* compiled extensions,
* Rust-backed dataframe infrastructure (when used with Polars),
* optimized factor handling.

Although the regression engine is still evolving, the goal is to provide performance comparable to `fixest` while integrating naturally with the Python scientific ecosystem.

---

# 11. Why Local Projections Repeat Work

Suppose we estimate horizons

[
h=0,\ldots,H.
]

Every regression uses

* identical regressors,
* identical fixed effects,
* identical clusters.

Only

[
y_h
]

changes.

Nevertheless, current implementations effectively perform

```text
Residualize X

Residualize y0

OLS

----------------

Residualize X

Residualize y1

OLS

----------------

Residualize X

Residualize y2

OLS
```

The computational bottleneck—residualizing the regressors—is repeated unnecessarily.

---

# 12. Caching Opportunities

After the first regression we already possess

[
\tilde X.
]

Therefore

[
\tilde X'\tilde X
]

is constant across all horizons.

Its Cholesky factorization or inverse can be computed once.

Each additional horizon only requires

[
\tilde X'\tilde y_h.
]

The dependent variable still changes, so

[
\tilde y_h
]

must be recomputed.

However,

* the factorization,
* the regressor residualization,
* the sparsity structure,
* the fixed-effect mappings,

are all reusable.

Current LP software generally exploits little of this redundancy.

---

# 13. Clustered Covariance

Cluster-robust covariance estimation computes

[
(X'X)^{-1}
\left(
\sum_g X_g'e_ge_g'X_g
\right)
(X'X)^{-1}.
]

Again,

[
(X'X)^{-1}
]

is identical across horizons.

Only the residual vectors

[
e_g
]

change.

Thus even covariance computation contains reusable components.

---

# 14. A Potential Next-Generation LP Algorithm

An LP estimator designed specifically for repeated horizons could proceed as follows.

### Stage 1

Construct

* regressor matrix,
* fixed-effect mappings,
* cluster mappings.

### Stage 2

Residualize the regressors once

[
X
\rightarrow
\tilde X.
]

### Stage 3

Compute

[
R=\operatorname{chol}(\tilde X'\tilde X)
]

or another suitable factorization.

### Stage 4

For each horizon

1. construct (y_h),
2. residualize (y_h),
3. compute

[
\tilde X'\tilde y_h,
]

4. solve using the cached factorization,
5. compute residuals,
6. update clustered covariance.

Compared with repeated calls to `feols()` or `reghdfe`, this avoids repeated preprocessing and repeated factorizations.

---

# 15. Beyond Existing Software

Current packages are optimized for **one regression at a time**.

Local projections are different because they involve **many regressions sharing an almost identical design matrix**.

From a numerical linear algebra perspective, the problem is closer to solving

[
AX=B,
]

where the left-hand side is fixed and only the right-hand side changes.

This suggests that block linear algebra methods, cached matrix factorizations, and repeated projection operators could substantially outperform the current regression-by-regression approach.

In other words, the computational structure of local projections is richer than the software interfaces currently expose. Existing HDFE estimators are highly optimized for a single within-transformation, but there remains considerable scope for specialized algorithms that exploit the repeated structure inherent in LP estimation.

---

# References

* Jordà, Ò. (2005). *Estimation and Inference of Impulse Responses by Local Projections*. American Economic Review.
* Correia, S. (2016). *Linear Models with High-Dimensional Fixed Effects: An Efficient and Feasible Estimator*. Working Paper.
* Berge, L. (2018). *Efficient Estimation of Maximum Likelihood Models with Multiple Fixed Effects*. CREA Discussion Paper.
* Fong, D. & Saunders, M. (2011). *LSMR: An Iterative Algorithm for Sparse Least-Squares Problems*. SIAM Journal on Scientific Computing.
* Strohmer, T. & Vershynin, R. (2009). *A Randomized Kaczmarz Algorithm with Exponential Convergence*. Journal of Fourier Analysis and Applications.

