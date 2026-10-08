# Quickstart

This example creates a small panel, estimates the response of `y` to `shock`,
and plots the impulse response.

## Create example data

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n_units = 50
n_periods = 30

data = pd.DataFrame(
    {
        "unit": np.repeat(np.arange(n_units), n_periods),
        "time": np.tile(np.arange(n_periods), n_units),
    }
)
data["shock"] = rng.normal(size=len(data))
data["control"] = rng.normal(size=len(data))
unit_effect = rng.normal(size=n_units)[data["unit"]]
data["y"] = (
    0.8 * data["shock"]
    + 0.3 * data["control"]
    + unit_effect
    + rng.normal(size=len(data))
)
```

Every `(unit, time)` pair must be unique. Use a finite, integer-valued period
index for `time`, with the index advancing by one per underlying period (for
example, `0, 1, 2, ...`). Panels may be unbalanced and therefore may contain
gaps in that index. Horizons are exact time leads: horizon `h` uses the outcome
at `time + h`, so a missing target period is not replaced by the next observed
row. For monthly or quarterly data, construct a sequential period index rather
than using decimal labels such as `2020.1` or `2020.2`.

## Fit local projections

```python
from fastlp import LocalProjection

lp = LocalProjection(
    horizons=8,
    covariance="cluster",
    sample="common",
)

lp.fit(
    data,
    outcome="y",
    shock="shock",
    controls=["control"],
    outcome_lags=2,
    shock_lags=2,
    control_lags=2,
    unit="unit",
    time="time",
    fixed_effects=["unit", "time"],
    cluster="unit",
)
```

A scalar lag value includes every lag from one through that value.

`sample="common"` uses the same anchor observations at every horizon.
`sample="per_horizon"` instead retains all observations available for each
horizon, so sample size may vary across the response path. `common` is 
encouraged for better performance.

## Inspect results

```python
print(lp.summary())
results = lp.to_frame()
print(results.query("coefficient == 'shock'"))
```

`to_frame()` returns one row per horizon and coefficient. It includes the
estimate, standard error, pointwise confidence interval, response type, sample
policy, and number of observations.

Useful fitted attributes include:

- `coef_`, `stderr_`, `covariance_`, and `conf_int_`
- `feature_names_in_` and `shock_names_in_`
- `n_obs_by_horizon_`, `n_units_`, and `n_periods_`
- `demeaning_diagnostics_`, `linear_algebra_diagnostics_`, and
  `covariance_diagnostics_`

## Plot the response

```python
ax = lp.plot_irf(
    "shock",
    title="Response of y to shock",
    ylabel="Outcome units",
)
ax.figure.savefig("irf.png", dpi=150, bbox_inches="tight")
```

The returned object is a Matplotlib `Axes`, so normal Matplotlib customization
remains available. Set `show = False` to use the `Axes` object in your own `Figure`. 

## Choosing inference

Set `covariance` when constructing `LocalProjection`:

| Value | Use |
| --- | --- |
| `"homoskedastic"` | Classical OLS covariance |
| `"hc0"`--`"hc3"` | Heteroskedasticity-robust covariance |
| `"cluster"` | One-way or multiway clustered covariance |
| `"hac"` | Within-unit HAC/Newey--West covariance |
| `"driscoll_kraay"` | Driscoll--Kraay covariance |

Clustered inference requires `cluster=` in `fit()`. HAC and Driscoll--Kraay
require integral numeric time labels. fastLP warns when a clustering dimension
has fewer than 50 groups by default; that warning identifies a limitation of
cluster asymptotics.

## Memory budgeting

`LocalProjection(memory_budget="1GB")` checks a conservative allocation plan
before panel preparation. The plan includes selected-frame copies, lag columns,
lead alignment, sample masks, retained results, and numerical workspaces. The
remaining allowance determines horizon batch size; tight budgets can also
stream HAC lag pairs. Budgets below the planned minimum raise `ValueError`
instead of silently being exceeded. Inspect `lp.memory_diagnostics_` after a
successful fit for the planned minimum, peak, and selected batch size.

This is a bound on the **allocation plan**, not a hard process-memory cap.
Caller-owned input, Python/allocator overhead and private library workspaces
are excluded; conservative estimates may reject a fit that would happen to
use less memory. Lazy Polars inputs require an additional aggregate pass for
sizing. Leave `memory_budget=None` to retain the unbudgeted execution path.

## Other input backends

Polars `DataFrame` and `LazyFrame` inputs use the same column-based interface.
A NumPy array requires one unique name for every column:

```python
array = data[["y", "shock", "control", "unit", "time"]].to_numpy()

lp.fit(
    array,
    column_names=["y", "shock", "control", "unit", "time"],
    outcome="y",
    shock="shock",
    controls=["control"],
    unit="unit",
    time="time",
)
```
