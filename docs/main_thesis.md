# Understanding the Computation Behind Local Projections

## Motivation

Local projections (LPs) are one of the most widely used tools for estimating impulse response functions. Econometrically, they are conceptually simple: estimate one regression for each forecast horizon. Computationally, however, this approach repeats a great deal of identical work.

Understanding where the computation occurs is useful for choosing the fastest software implementation and for designing more efficient algorithms.

---

# 1. Ordinary Least Squares

Consider the linear regression

[
y = X\beta + \varepsilon,
]

where

* (y) is an (N \times 1) vector,
* (X) is an (N \times K) matrix of regressors,
* (\beta) is a (K \times 1) coefficient vector.

The OLS estimator is

[
\hat{\beta} = (X'X)^{-1}X'y.
]

Notice that the estimator depends on only two quantities:

1. the cross-product matrix

[
X'X,
]

2. the cross-product between regressors and the dependent variable

[
X'y.
]

---

# 2. What Happens if Only the Dependent Variable Changes?

Suppose we estimate several regressions with the same regressors but different dependent variables.

Regression 1:

[
y_0 = X\beta_0 + \varepsilon
]

Regression 2:

[
y_1 = X\beta_1 + \varepsilon
]

Regression 3:

[
y_2 = X\beta_2 + \varepsilon.
]

The corresponding estimators are

[
\hat{\beta}_h = (X'X)^{-1}X'y_h.
]

The important observation is that

* (X) is identical,
* (X'X) is identical,
* ((X'X)^{-1}) is identical.

Only (X'y_h) changes.

Therefore, the expensive matrix inverse only needs to be computed once.

---

# 3. Local Projections are Exactly This Problem

For horizon (h), a local projection estimates

[
y_{t+h}
=======

\alpha_h
+
\beta_h x_t
+
\Gamma_h Z_t
+
u_{t+h}.
]

Each horizon corresponds to a different regression:

* Horizon 0 uses (y_t),
* Horizon 1 uses (y_{t+1}),
* Horizon 2 uses (y_{t+2}),
* ...

The regressors

[
X =
\begin{bmatrix}
1 & x & Z
\end{bmatrix}
]

remain exactly the same.

Only the dependent variable changes.

---

# 4. The Naïve Computational Algorithm

Most software effectively performs

```text
for h = 0,...,H

    Construct X

    Compute X'X

    Compute (X'X)^(-1)

    Compute X'y_h

    Estimate β_h
```

This repeats substantial computation.

A more efficient algorithm would instead compute

```text
Construct X

Compute X'X

Compute (X'X)^(-1)

for h = 0,...,H

    Compute X'y_h

    Estimate β_h
```

The econometric estimates are identical.

The computational cost is lower because the expensive matrix inverse is reused.

---

# 5. Introducing Fixed Effects

Suppose the model includes high-dimensional fixed effects:

[
y = X\beta + D\alpha + \varepsilon,
]

where (D) is a large matrix of dummy variables.

For example,

* district fixed effects,
* year fixed effects,
* month fixed effects.

Explicitly constructing (D) would often be computationally impossible.

---

# 6. Frisch–Waugh–Lovell (FWL)

The Frisch–Waugh–Lovell theorem states that

[
y = X\beta + D\alpha + \varepsilon
]

can be estimated by

1. removing the fixed effects from (y),
2. removing the fixed effects from every column of (X),
3. running OLS on the transformed variables.

Formally,

[
\tilde y = M_D y,
]

[
\tilde X = M_D X,
]

where

[
M_D = I - D(D'D)^{-1}D'
]

is the projection matrix that removes the fixed effects.

The coefficient estimate becomes

[
\hat\beta
=========

(\tilde X'\tilde X)^{-1}
\tilde X'\tilde y.
]

This produces exactly the same coefficients as the full dummy-variable regression.

---

# 7. How `reghdfe` and `fixest` Work

Neither package constructs (M_D) explicitly.

Instead, they repeatedly demean variables with respect to each fixed effect until convergence.

Conceptually,

```text
Original X

↓

Subtract district means

↓

Subtract month means

↓

Subtract year means

↓

Repeat until convergence
```

The same procedure is applied to the dependent variable.

The output is a residualized dataset

[
(\tilde X,\tilde y),
]

on which ordinary least squares is performed.

The difference between packages lies mainly in implementation:

* **reghdfe** uses Mata and the alternating-projections algorithm developed by Correia (2016),
* **fixest** implements a closely related approach in highly optimized C++ with multithreading and aggressive caching.

---

# 8. Where Local Projections Waste Computation

Suppose we estimate 21 horizons.

Every regression has

* identical regressors,
* identical fixed effects,
* identical clustering structure.

Only the dependent variable changes.

Nevertheless, a naïve implementation performs

```text
Horizon 0

Residualize X

Residualize y0

OLS

----------------

Horizon 1

Residualize X again

Residualize y1

OLS

----------------

Horizon 2

Residualize X again

Residualize y2

OLS
```

The regressors are residualized repeatedly even though they never change.

---

# 9. What Can Be Cached?

After residualizing the regressors once, we obtain

[
\tilde X.
]

This object is identical across all horizons.

Therefore,

[
\tilde X'\tilde X
]

only needs to be computed once.

Its inverse

[
(\tilde X'\tilde X)^{-1}
]

also only needs to be computed once.

For each horizon, only

[
\tilde X'\tilde y_h
]

must be recomputed.

This eliminates a large amount of redundant work.

---

# 10. Why the Dependent Variable Still Must Be Residualized

Each horizon has a different dependent variable:

[
y_t,;
y_{t+1},;
y_{t+2},;
\ldots
]

Because the values change, each dependent variable must be residualized separately.

However, the fixed-effect structure itself does not change.

A sufficiently sophisticated implementation can therefore reuse information about the fixed effects while applying the transformation to each new dependent variable.

---

# 11. Cluster-Robust Standard Errors

With clustered standard errors,

[
\widehat{\mathrm{Var}}(\hat\beta)
=================================

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

Only the residuals change.

This means that parts of the covariance computation can also be reused, although the residual-dependent terms must still be recomputed for each regression.

---

# 12. Implications for Software

Current LP implementations (e.g., `locproj` in Stata) generally estimate one regression per horizon.

This is computationally straightforward but repeats substantial work.

Packages such as **fixest** reduce runtime by providing a highly optimized fixed-effects estimator, but they still treat each horizon as a separate regression.

A purpose-built LP implementation could go further by

* residualizing the regressors only once,
* caching the cross-product matrices,
* reusing the fixed-effect structure across horizons,
* computing only the horizon-specific components of the estimator.

Such an implementation would produce identical econometric estimates while reducing redundant computation, particularly in applications with many horizons, large datasets, and high-dimensional fixed effects.

---

# Summary

From an econometric perspective, local projections are simply a collection of independent regressions.

From a computational perspective, however, those regressions share almost all of their structure:

* the regressors are identical,
* the fixed effects are identical,
* much of the linear algebra is identical.

Recognizing this distinction explains why implementations based solely on looping over regressions are not computationally optimal, and it highlights opportunities for specialized algorithms that can substantially reduce estimation time without changing the underlying econometric estimator.

