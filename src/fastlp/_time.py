"""Exact normalization of user-supplied panel period identifiers."""

from __future__ import annotations

import re

import numpy as np

_INT64_INFO = np.iinfo(np.int64)
_MAX_EXACT_FLOAT_INTEGER = 2**53 - 1
_INTEGER_STRING = re.compile(r"[+-]?[0-9]+\Z")
_TIME_ERROR = (
    "time must contain integer period identifiers: use int64-range integers, "
    "exact integral floats no larger than 2**53 - 1 in magnitude, or base-10 "
    "integer strings; datetimes, fractional values, and mixed representations "
    "are not supported"
)


def _checked_integers(values: list[int]) -> np.ndarray:
    if any(value < _INT64_INFO.min or value > _INT64_INFO.max for value in values):
        raise ValueError(_TIME_ERROR)
    return np.asarray(values, dtype=np.int64)


def canonical_time(values: np.ndarray) -> np.ndarray:
    """Return one collision-checkable, exact int64 panel clock."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(_TIME_ERROR)
    if not len(array):
        return np.empty(0, dtype=np.int64)
    if (
        np.issubdtype(array.dtype, np.bool_)
        or np.issubdtype(array.dtype, np.datetime64)
        or np.issubdtype(array.dtype, np.timedelta64)
    ):
        raise ValueError(_TIME_ERROR)
    if np.issubdtype(array.dtype, np.signedinteger):
        return array.astype(np.int64, copy=False)
    if np.issubdtype(array.dtype, np.unsignedinteger):
        if int(array.max()) > _INT64_INFO.max:
            raise ValueError(_TIME_ERROR)
        return array.astype(np.int64)
    if np.issubdtype(array.dtype, np.floating):
        if (
            not np.isfinite(array).all()
            or np.any(np.abs(array) > _MAX_EXACT_FLOAT_INTEGER)
            or np.any(array != np.trunc(array))
        ):
            raise ValueError(_TIME_ERROR)
        return array.astype(np.int64)

    items = array.tolist()
    if all(
        isinstance(value, (int, np.integer)) and not isinstance(value, bool)
        for value in items
    ):
        return _checked_integers([int(value) for value in items])
    if all(isinstance(value, (str, np.str_)) for value in items):
        strings = [str(value) for value in items]
        if not all(_INTEGER_STRING.fullmatch(value) for value in strings):
            raise ValueError(_TIME_ERROR)
        return _checked_integers([int(value, 10) for value in strings])
    raise ValueError(_TIME_ERROR)
