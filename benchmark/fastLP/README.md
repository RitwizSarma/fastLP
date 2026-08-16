# fastLP cross-library runner

Generate the two benchmark panels from the project root, then run:

```bash
uv run python benchmark/data/generate_data.py
uv run python benchmark/fastLP/benchmark.py n5000_t40
uv run python benchmark/fastLP/benchmark.py n10000_t40
```

Each command performs the complete 13-horizon estimation 200 times. It writes
`results_fastlp_<dataset>.csv` with one row per repetition and wall-clock,
user-CPU, and system-CPU timing columns, plus
`estimates_fastlp_<dataset>.csv` with the first fit's coefficient estimates by
horizon. Mean, SD, median, Q1, Q3, minimum, and maximum statistics for all
three timing series are printed to the screen.
