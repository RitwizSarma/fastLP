"""Numerical limit and input-contract regressions for review items 14 and 16."""

from decimal import Decimal, localcontext

import numpy as np
import pandas as pd
import pytest

from fastlp import LocalProjection
from fastlp._covariance import (
    _kernel_weight,
    _quadratic_spectral_meat,
    _quadratic_spectral_weight,
)


def decimal_qs(z):
    with localcontext() as context:
        context.prec = 60
        square = Decimal(str(z)) ** 2
        total = term = Decimal(1)
        for k in range(1, 60):
            term *= -square / Decimal(2 * k * (2 * k + 3))
            total += term
        return float(total)


def test_qs_small_argument_and_threshold_match_decimal_oracle():
    z = np.array([0, 1e-15, 1e-9, 1e-5, 0.05, 0.099999999, 0.1, 0.100000001, 1, 4])
    expected = np.array([decimal_qs(value) for value in z])
    np.testing.assert_allclose(_quadratic_spectral_weight(z), expected, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(
        [_quadratic_spectral_weight(float(value)) for value in z],
        expected, rtol=1e-13, atol=1e-15,
    )
    assert _kernel_weight("quadratic_spectral", 1, 10**9) == 1.0


@pytest.mark.parametrize("gapped", [False, True])
@pytest.mark.parametrize("bandwidth", [3, 10**9])
def test_qs_fft_and_blocked_match_high_precision_pairs(gapped, bandwidth):
    rng = np.random.default_rng(14)
    times = np.tile(np.arange(8), 3)
    units = np.repeat(np.arange(3), 8)
    scores = rng.normal(size=(24, 2))
    if gapped:
        keep = ~((units == 1) & (times == 3))
        times, units, scores = times[keep], units[keep], scores[keep]
    expected = np.zeros((2, 2))
    for i in range(len(times)):
        for j in range(len(times)):
            if units[i] == units[j]:
                z = 6 * np.pi * abs(int(times[i]) - int(times[j])) / (5 * bandwidth)
                expected += decimal_qs(z) * np.outer(scores[i], scores[j])
    actual = _quadratic_spectral_meat(scores, units, times, bandwidth)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("column", "representation"),
    [
        ("y", "complex_dtype"),
        ("x", "complex_dtype"),
        ("control", "complex_dtype"),
        ("y", "object"),
        ("y", "numpy"),
    ],
)
def test_complex_model_columns_are_rejected(column, representation):
    rng = np.random.default_rng(16)
    data = pd.DataFrame({
        "unit": np.repeat(np.arange(3), 8), "time": np.tile(np.arange(8), 3),
        "y": rng.normal(size=24), "x": rng.normal(size=24),
        "control": rng.normal(size=24),
    })
    data[column] = data[column].astype(complex) + 10j
    if representation == "object":
        data[column] = data[column].astype(object)
    kwargs = {}
    if representation == "numpy":
        kwargs["column_names"] = data.columns.tolist()
        data = data.to_numpy(dtype=object)
    with pytest.raises(ValueError, match="real values only"):
        LocalProjection(horizons=0, covariance="hc0").fit(
            data, outcome="y", shock="x", controls=["control"],
            unit="unit", time="time", **kwargs,
        )


def test_polars_complex_object_is_rejected():
    pl = pytest.importorskip("polars")
    data = pl.DataFrame({"unit": [0] * 6, "time": list(range(6)), "x": list(range(6))})
    data = data.with_columns(pl.Series("y", [1 + 2j] * 6, dtype=pl.Object))
    with pytest.raises(ValueError, match="real values only"):
        LocalProjection(horizons=0, covariance="hc0").fit(
            data, outcome="y", shock="x", unit="unit", time="time",
        )
