"""Exact panel-clock validation and boundary behavior."""

import numpy as np
import pandas as pd
import pytest

from fastlp import LocalProjection
from fastlp._time import canonical_time


def small_panel(times) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit": np.zeros(len(times), dtype=int),
            "time": times,
            "y": np.arange(len(times), dtype=float),
            "x": np.linspace(0.25, 1.25, len(times)),
        }
    )


def test_native_int64_boundaries_are_preserved_without_float_conversion() -> None:
    values = np.asarray([np.iinfo(np.int64).min, np.iinfo(np.int64).max])
    result = canonical_time(values)
    np.testing.assert_array_equal(result, values)


@pytest.mark.parametrize(
    "values",
    [
        np.asarray([0, np.iinfo(np.int64).max + 1], dtype=np.uint64),
        np.asarray([0.0, 1.0000000001]),
        np.asarray([0.0, float(2**53)]),
        np.asarray([0, "1"], dtype=object),
        np.asarray([True, False]),
        np.asarray(["0", str(np.iinfo(np.int64).max + 1)]),
    ],
)
def test_ambiguous_or_out_of_range_time_is_rejected(values) -> None:
    with pytest.raises(ValueError, match="integer period identifiers"):
        canonical_time(values)


def test_lead_addition_overflow_is_rejected() -> None:
    maximum = np.iinfo(np.int64).max
    data = small_panel(np.asarray([maximum - 4, maximum - 2, maximum]))
    with pytest.raises(ValueError, match="requested horizons exceeds"):
        LocalProjection(horizons=1, covariance="hc0").fit(
            data,
            outcome="y",
            shock="x",
            unit="unit",
            time="time",
        )


def test_hac_rejects_time_span_that_cannot_be_subtracted_safely() -> None:
    data = small_panel(np.asarray([np.iinfo(np.int64).min, 0, np.iinfo(np.int64).max]))
    with pytest.raises(ValueError, match="time span exceeds"):
        LocalProjection(
            horizons=0,
            covariance="hac",
            hac_lags=1,
            hac_debias=False,
        ).fit(
            data,
            outcome="y",
            shock="x",
            unit="unit",
            time="time",
        )
