"""Generate reproducible balanced and unbalanced synthetic LP panels.

The balanced designs match ``src/notebooks/synthetic_benchmark.ipynb``.  The
unbalanced designs use the same DGP, then remove 10% of each unit's periods
(while retaining enough observations for the configured LP horizons).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SEED = 2026
SMALL_UNITS, SMALL_PERIODS, SMALL_HORIZON = 200, 40, 12
LARGE_UNITS, LARGE_PERIODS, LARGE_HORIZON = 1_250_000, 8, 2
UNBALANCED_SHARE = 0.10

DATA_DIR = Path(__file__).resolve().parent


def make_panel(n_units: int, n_periods: int, seed: int) -> pd.DataFrame:
    """Return the notebook's synthetic balanced-panel DGP."""
    rng = np.random.default_rng(seed)
    unit = np.repeat(np.arange(n_units, dtype=np.int32), n_periods)
    time = np.tile(np.arange(n_periods, dtype=np.int16), n_units)
    unit_effect = rng.normal(scale=0.75, size=n_units)
    time_effect = rng.normal(scale=0.25, size=n_periods)
    shock = rng.normal(size=n_units * n_periods)
    control = rng.normal(size=n_units * n_periods)
    noise = rng.normal(scale=0.5, size=n_units * n_periods)
    outcome = 1.0 + 0.8 * shock - 0.35 * control + unit_effect[unit] + time_effect[time] + noise
    return pd.DataFrame(
        {"unit": unit, "time": time, "outcome": outcome, "shock": shock, "control": control}
    )


def make_unbalanced(panel: pd.DataFrame, n_periods: int, horizon: int, seed: int) -> pd.DataFrame:
    """Drop a random 10% of periods within every unit, preserving LP support."""
    n_drop = max(1, round(n_periods * UNBALANCED_SHARE))
    if n_periods - n_drop <= horizon:
        raise ValueError("unbalanced design leaves too few periods for its LP horizon")

    rng = np.random.default_rng(seed)
    n_units = panel["unit"].nunique()
    drop_positions = np.empty((n_units, n_drop), dtype=np.int16)
    for unit in range(n_units):
        drop_positions[unit] = rng.choice(n_periods, size=n_drop, replace=False)
    drop_mask = np.zeros((n_units, n_periods), dtype=bool)
    drop_mask[np.arange(n_units)[:, None], drop_positions] = True
    return panel.loc[~drop_mask.ravel()].reset_index(drop=True)


def write_dataset(name: str, data: pd.DataFrame) -> None:
    path = DATA_DIR / f"{name}.csv"
    data.to_csv(path, index=False, float_format="%.10g")
    print(f"wrote {path.name}: {len(data):,} rows")


def main() -> None:
    small = make_panel(SMALL_UNITS, SMALL_PERIODS, SEED)
    write_dataset("small_balanced", small)
    write_dataset("small_unbalanced", make_unbalanced(small, SMALL_PERIODS, SMALL_HORIZON, SEED + 10))

    large = make_panel(LARGE_UNITS, LARGE_PERIODS, SEED + 1)
    write_dataset("large_balanced", large)
    write_dataset("large_unbalanced", make_unbalanced(large, LARGE_PERIODS, LARGE_HORIZON, SEED + 11))


if __name__ == "__main__":
    main()
