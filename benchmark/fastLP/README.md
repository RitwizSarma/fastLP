# fastLP benchmark runner

Run one of the four datasets with `uv` from the project root:

```bash
uv run python benchmark/fastLP/benchmark.py small_balanced
uv run python benchmark/fastLP/benchmark.py small_unbalanced
uv run python benchmark/fastLP/benchmark.py large_balanced
uv run python benchmark/fastLP/benchmark.py large_unbalanced
```

CSV loading is excluded from the reported estimation time. The runner uses
unit and time fixed effects, clusters standard errors by unit, and prints the
complete coefficient table for every horizon.
