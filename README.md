# fastLP

`fastLP` estimates linear local projections for balanced and unbalanced panel data.
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

v0.1 supports HC1 and one-way cluster-robust covariance. Unbalanced panels are
opt-in with `allow_unbalanced=True`; each horizon uses observations with an
exact `time + horizon` lead. That path caches the initial raw-design Gram and
updates it for rows entering or leaving the sample. With one fixed effect it
uses exact group sufficient statistics; specifications with multiple fixed
effects safely fall back to alternating-projection demeaning. Horizon-varying
regressors and HAC inference are not yet supported.

The optional native demeaning backend is built with:

```bash
uv run maturin develop --manifest-path rust/Cargo.toml
```

Without Rust, the package uses its equivalent NumPy implementation.
