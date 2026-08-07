# fastLP

`fastLP` estimates panel local projections for massive datasets very very
quickly. Beginner-friendly API, Rust core, and helpful functions for
almost everything an empirical economist might need.

## Installation

For now, install fastLP from a Git clone. You need Python 3.12 or later,
[`uv`](https://docs.astral.sh/uv/), and a Rust toolchain with Cargo: fastLP
builds its native fixed-effect backend during installation.

```bash
git clone git@github.com:RitwizSarma/fastLP.git
cd fastLP
uv sync
```

Polars input support is optional but recommended if you're
working with large datasets:

```bash
uv sync --extra polars
```

## Quick start

Pass a pandas or Polars frame and identify its outcome, shock, panel unit, and
time columns. 

```python
from fastlp import LocalProjection

lp = LocalProjection(horizons=12, covariance="driscoll_kraay")
lp.fit(
    data,
    outcome="log_output",
    shock="monetary_shock",
    controls=["output_l1", "inflation_l1"],
    outcome_lags=range(1,4),
    shock_lags=2,
    control_lags={"output_l1": range(1,3), "inflation_l1": 1},
)
```

The following example estimates a 20-period response with country and
quarter fixed effects and country-clustered standard errors, and 
retrieves the results and impulse response plot.

<details>
<summary>Code</summary>

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

ax = lp.plot_irf(
    "monetary_shock",
    title="Output response to a monetary shock",
    ylabel="Log points",
)
ax.figure.savefig("irf.png", bbox_inches="tight")
```

`to_frame()` returns one row per horizon and coefficient, with the estimate,
standard error, and pointwise confidence interval. The fitted estimator also
offers a cached default plot through `lp.irfplot`.

</details>


## Data requirements

- `data` must be a pandas `DataFrame` or Polars `DataFrame`/`LazyFrame`; 
  NumPy arrays are not accepted currently. Backend selection is automatic 
  from the input type.
- Each `(unit, time)` pair must appear at most once. The panel may be
  unbalanced: units may have different observed periods and periods may be
  missing.
- The outcome, shocks, and controls must be numeric, finite, and non-missing.
  Model columns such as fixed effects and cluster identifiers must also be
  complete.
- Horizons are exact leads. With numeric time labels, a horizon of `h` uses
  the observation at `time + h`; missing periods are not treated as adjacent.

Rows that cannot provide the requested lags or future outcome are excluded
from the relevant regression sample. Use `sample="common"` (the default) to
hold the anchor sample fixed across all horizons, or `sample="per_horizon"`
to retain every valid anchor separately at each horizon.

Polars remains native through validation, sorting, lag construction, grouping,
and exact lead alignment. A lazy scan projects the required model columns and
is collected once before the numerical NumPy/Rust estimation core:

```python
import polars as pl

data = pl.scan_parquet("panel.parquet")
lp.fit(data, outcome="y", shock="shock", unit="unit", time="time")
```

`to_frame()` continues to return pandas for every input backend. Retained
sample identifiers preserve the pandas index for pandas input and use original
zero-based row positions for Polars input.

## Estimation and inference

fastLP supports level and cumulative responses, one or more shocks, generated
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
`hac_lags`, `hac_kernel`, and `hac_debias` to configure them.

## License

fastLP is licensed under the [GNU General Public License v3.0 or later](LICENSE).
The Rust fixed-effect kernel includes an adaptation of MIT-licensed work; its
attribution notice is retained in [LICENSES/diff-diff-MIT.txt](LICENSES/diff-diff-MIT.txt).
