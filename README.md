# fastLP

`fastLP` estimates linear local projections for strictly balanced panel data.
It uses one common anchor sample across all horizons, residualizes the
right-hand-side design once, and reuses its factorization for every horizon.

```python
from fastlp import LocalProjection

lp = LocalProjection(horizons=20, covariance="cluster")
lp.fit(
    data,
    outcome="log_output",
    shock="monetary_shock",
    controls=["output_l1", "inflation_l1"],
    unit="country",
    time="quarter",
    fixed_effects=["country", "quarter"],
    cluster="country",
)
print(lp.to_frame())
```

v0.1 supports HC1 and one-way cluster-robust covariance. It intentionally
rejects unbalanced panels, horizon-varying regressors, and HAC inference.

The optional native demeaning backend is built with:

```bash
uv run maturin develop --manifest-path rust/Cargo.toml
```

Without Rust, the package uses its equivalent NumPy implementation.
