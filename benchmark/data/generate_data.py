"""Generate the complete-panel datasets used by the cross-library benchmark.

Every engine receives the same CSV for a dataset.  The grid varies the number
of units (N) and periods per unit (T); the regression specification and random
data-generating process are otherwise fixed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    n_units: int
    n_periods: int
    horizon: int
    seed: int


DATASETS = (
    DatasetSpec("n5000_t40", n_units=5_000, n_periods=40, horizon=12, seed=2030),
    DatasetSpec("n10000_t40", n_units=10_000, n_periods=40, horizon=12, seed=2031),
)

DATA_DIR = Path(__file__).resolve().parent


def make_panel(spec: DatasetSpec) -> pd.DataFrame:
    """Return one deterministic complete panel for ``spec``."""
    rng = np.random.default_rng(spec.seed)
    n_obs = spec.n_units * spec.n_periods
    unit = np.repeat(np.arange(spec.n_units, dtype=np.int32), spec.n_periods)
    time = np.tile(np.arange(spec.n_periods, dtype=np.int16), spec.n_units)
    unit_effect = rng.normal(scale=0.75, size=spec.n_units)
    time_effect = rng.normal(scale=0.25, size=spec.n_periods)
    shock = rng.normal(size=n_obs)
    control = rng.normal(size=n_obs)
    noise = rng.normal(scale=0.5, size=n_obs)
    outcome = (
        1.0
        + 0.8 * shock
        - 0.35 * control
        + unit_effect[unit]
        + time_effect[time]
        + noise
    )
    return pd.DataFrame(
        {"unit": unit, "time": time, "outcome": outcome, "shock": shock, "control": control}
    )


def write_dataset(spec: DatasetSpec) -> None:
    data = make_panel(spec)
    path = DATA_DIR / f"{spec.name}.csv"
    data.to_csv(path, index=False, float_format="%.10g")
    print(f"wrote {path.name}: {len(data):,} rows (N={spec.n_units}, T={spec.n_periods})")


def main() -> None:
    for spec in DATASETS:
        write_dataset(spec)
    (DATA_DIR / "datasets.json").write_text(
        json.dumps([asdict(spec) for spec in DATASETS], indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
