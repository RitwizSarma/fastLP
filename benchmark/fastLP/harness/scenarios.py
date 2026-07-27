"""Reproducible benchmark inputs and their intentionally modest defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Scenario:
    name: str
    n_units: int
    n_periods: int
    horizons: int
    n_controls: int = 1
    unbalanced_share: float = 0.0
    seed: int = 2026

    def metadata(self) -> dict[str, object]:
        return asdict(self)


SCENARIOS = {
    # Kept small enough for CI and for checking numerical agreement every run.
    "smoke": Scenario("smoke", n_units=40, n_periods=24, horizons=6),
    "small": Scenario("small", n_units=200, n_periods=40, horizons=12),
    "small_unbalanced": Scenario(
        "small_unbalanced", n_units=200, n_periods=40, horizons=12, unbalanced_share=0.10
    ),
    # This mirrors benchmark/data's ten-million-row configuration.  It must be
    # explicitly selected; it is never a default.
    "large": Scenario("large", n_units=1_250_000, n_periods=8, horizons=2, seed=2027),
}


def make_panel(scenario: Scenario) -> pd.DataFrame:
    """Create a deterministic DGP without involving CSV parsing in a run."""
    rng = np.random.default_rng(scenario.seed)
    n_obs = scenario.n_units * scenario.n_periods
    unit = np.repeat(np.arange(scenario.n_units, dtype=np.int32), scenario.n_periods)
    time = np.tile(np.arange(scenario.n_periods, dtype=np.int16), scenario.n_units)
    unit_effect = rng.normal(scale=0.75, size=scenario.n_units)
    time_effect = rng.normal(scale=0.25, size=scenario.n_periods)
    shock = rng.normal(size=n_obs)
    controls = rng.normal(size=(n_obs, scenario.n_controls))
    noise = rng.normal(scale=0.5, size=n_obs)
    outcome = 1.0 + 0.8 * shock + unit_effect[unit] + time_effect[time] + noise
    data: dict[str, np.ndarray] = {"unit": unit, "time": time, "outcome": outcome, "shock": shock}
    for index in range(scenario.n_controls):
        name = "control" if index == 0 else f"control_{index + 1}"
        data[name] = controls[:, index]
        outcome -= (0.35 / (index + 1)) * controls[:, index]
    data["outcome"] = outcome
    frame = pd.DataFrame(data)
    if scenario.unbalanced_share:
        keep = rng.random(n_obs) >= scenario.unbalanced_share
        frame = frame.loc[keep].reset_index(drop=True)
    return frame
