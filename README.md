# fastLP

`fastLP` estimates panel local projections for massive datasets very very
quickly. Beginner-friendly API, Rust core, and helpful functions for
IRF graphs and tables: everything an empirical economist might need.

## Installation

fastLP requires Python 3.12 or later. Install it from PyPI or `uv` with:

```bash
pip install fastlp-py
uv add fastlp-py
```

Polars input support is optional and can be installed with:

```bash
pip install "fastlp-py[polars]"
uv add "fastlp-py[polars]"
```

To work on fastLP itself, clone the repository and use
[`uv`](https://docs.astral.sh/uv/). A Rust toolchain with Cargo is required for
source builds:

```bash
git clone https://github.com/RitwizSarma/fastLP.git
cd fastLP
uv sync --all-groups
```

## Quick start

Get started quickly with the `scikit`-style API:

```python
from fastlp import LocalProjection

lp = LocalProjection(horizons=12, covariance="cluster")
lp.fit(
    data,
    outcome="y",
    shock="shock",
    outcome_lags=range(1, 4),
    shock_lags=2,
    unit="unit",
    time="time",
    fixed_effects=["unit", "time"],
    cluster="unit"
)
```

Make sure you explicitly indicate the fixed effects specification. `fastLP`
does _not_ implicitly use unit- or time-fixed effects. For multi-way fixed
effects, use a tuple like `fixed_effects=["unit", ("quarter", "zipcode")]`.

Retrieve the results and plot the shock response:

<details>
<summary>Code</summary>

```python
print(lp.to_frame())

ax = lp.plot_irf(
    "shock",
    title="Response of y to shock",
    ylabel="Outcome units",
)
ax.figure.savefig("irf.png", bbox_inches="tight")
```

`to_frame()` returns one row per horizon and coefficient, with the estimate,
standard error, and pointwise confidence interval. The fitted estimator also
offers a cached default plot through `lp.irfplot`.

</details>

For a more detailed example, check out the [documentation](https://ritwizsarma.github.io/fastLP/quickstart.html).


<!-- ## Data input flexibility

- `pandas`, Polars (including `LazyFrames`) and `numpy` arrays are all allowed.

- Polars remains native through validation, sorting, lag construction, grouping,
and exact lead alignment. A lazy scan projects the required model columns and
is collected once before the numerical NumPy/Rust estimation core:

```python
import polars as pl

data = pl.scan_parquet("panel.parquet")
lp.fit(data, outcome="y", shock="shock", unit="unit", time="time")
```

- `to_frame()` continues to return pandas for every input backend. Retained
sample identifiers preserve the pandas index for pandas input and use original
zero-based row positions for Polars input. -->

<!-- ## Estimation and inference

fastLP supports level and cumulative responses, generated
within-unit lags, and absorbed fixed effects. It provides homoskedastic,
HC0--HC3, clustered CR0/CR1, within-unit HAC/Newey--West, and
Driscoll--Kraay covariance estimators.

One-way fixed effects and balanced unit/time fixed effects use exact within
transformations. Other multiway fixed effects are absorbed with alternating
group projections: the native Rust backend uses symmetric
forward-and-reverse sweeps, while the NumPy fallback uses cyclic forward
sweeps. Guarded Irons--Tuck acceleration is used by default.

The final regression uses a simple Cholesky solver. Reported
coefficients and covariance matrices remain in the original feature units;
`linear_algebra_diagnostics_` records rank and condition estimates for each
distinct horizon-sample design.

For `covariance="cluster"`, pass `cluster` as a column name, an interaction
tuple, or up to four cluster terms. For example,
`cluster=["country", "quarter", ("industry", "quarter")]` requests three
cluster dimensions. fastLP warns when a requested cluster term has fewer than
50 groups; this is a diagnostic, not a correction for few-cluster inference.

HAC and Driscoll--Kraay estimation require integral numeric time labels. Use
`hac_lags`, `hac_kernel`, and `hac_debias` to configure them. -->

## License

fastLP is licensed under the [GNU General Public License v3.0 or later](LICENSE).
The Rust fixed-effect kernel includes an adaptation of MIT-licensed work; its
attribution notice is retained in [LICENSES/diff-diff-MIT.txt](LICENSES/diff-diff-MIT.txt).
