"""Backend-neutral panel-frame preparation.

The estimator works with this small adapter rather than converting alternate
dataframe implementations through pandas.  Each adapter owns dataframe-native
sorting, lag construction, grouping, and exact lead alignment, and exposes
NumPy only at the numerical-kernel boundary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ._time import canonical_time

_ROW_ID = "__fastlp_original_row__"


def _row_id_name(columns: Sequence[str]) -> str:
    name = _ROW_ID
    while name in columns:
        name += "_"
    return name


class PanelFrame(ABC):
    """The dataframe operations needed by the shared estimator."""

    backend: str

    @classmethod
    @abstractmethod
    def prepare(
        cls, data: Any, required: Sequence[str], numeric: Sequence[str]
    ) -> tuple["PanelFrame", bool]: ...

    @property
    @abstractmethod
    def columns(self) -> Sequence[str]: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def column(self, name: str) -> np.ndarray: ...

    def matrix(self, names: Sequence[str], *, dtype: Any = None) -> np.ndarray:
        if not names:
            return np.empty((len(self), 0), dtype=dtype or np.float64)
        return np.column_stack([self.column(name) for name in names]).astype(
            dtype, copy=False
        )

    @abstractmethod
    def take(self, positions: np.ndarray) -> "PanelFrame": ...

    @abstractmethod
    def add_lags(
        self, unit: str, time: str, requests: Sequence[tuple[str, int]]
    ) -> tuple["PanelFrame", tuple[str, ...], np.ndarray, str]: ...

    @abstractmethod
    def factorize(self, columns: tuple[str, ...]) -> tuple[np.ndarray, int]: ...

    @abstractmethod
    def lead_positions(
        self, unit: str, time: str, horizons: int
    ) -> tuple[np.ndarray, str]: ...

    @abstractmethod
    def sample_ids(self, mask: np.ndarray): ...

    def group_sizes(self, column: str) -> np.ndarray:
        codes, count = self.factorize((column,))
        return np.bincount(codes, minlength=count)

    def nunique(self, column: str) -> int:
        return self.factorize((column,))[1]

    def balanced_shape(self, unit: str, time: str) -> tuple[int, int] | None:
        sizes = self.group_sizes(unit)
        if not len(sizes) or not np.all(sizes == sizes[0]):
            return None
        n_periods = int(sizes[0])
        labels = self.column(time).reshape(len(sizes), n_periods)
        if not np.all(labels == labels[0]):
            return None
        return len(sizes), n_periods

    def has_consecutive_time(self, unit: str, time: str) -> bool:
        """Return whether every within-unit row advances by exactly one period."""
        if len(self) < 2:
            return True
        units = self.column(unit)
        labels = self.column(time)
        within_unit = units[1:] == units[:-1]
        previous = labels[:-1]
        safe = previous != np.iinfo(np.int64).max
        expected = previous.copy()
        np.add(previous, 1, out=expected, where=safe)
        return bool(np.all(~within_unit | (safe & (labels[1:] == expected))))


class PandasFrame(PanelFrame):
    backend = "pandas"

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    @classmethod
    def prepare(
        cls, data: pd.DataFrame, required: Sequence[str], numeric: Sequence[str]
    ) -> tuple["PandasFrame", bool]:
        missing = sorted(set(required).difference(data.columns))
        if missing:
            raise ValueError(f"data is missing required columns: {missing}")
        frame = data.loc[:, list(dict.fromkeys(required))].copy()
        if frame.isna().any().any():
            raise ValueError("v0.1 requires complete data in all model columns")
        unit, time = required[1], required[2]
        frame[time] = canonical_time(frame[time].to_numpy())
        try:
            for column in dict.fromkeys(numeric):
                converted = pd.to_numeric(frame[column], errors="raise")
                if pd.api.types.is_complex_dtype(converted.dtype):
                    raise ValueError("complex model columns are unsupported")
                values = converted.to_numpy(dtype=np.float64)
                if not np.isfinite(values).all():
                    raise FloatingPointError
                # Keep identifiers/categories exact even when they also serve
                # as model variables. Numerical matrices are cast on extraction.
        except (TypeError, ValueError) as error:
            raise ValueError(
                "outcome, shock, and controls must be numeric (real values only)"
            ) from error
        except FloatingPointError as error:
            raise ValueError("outcome, shock, and controls must be finite") from error
        if frame.duplicated([unit, time]).any():
            raise ValueError("unit and time must uniquely identify panel observations")
        order = pd.MultiIndex.from_frame(frame[[unit, time]])
        input_was_sorted = order.is_monotonic_increasing
        if not input_was_sorted:
            frame = frame.sort_values([unit, time], kind="stable")
        return cls(frame), input_was_sorted

    @property
    def columns(self) -> Sequence[str]:
        return self.frame.columns

    def __len__(self) -> int:
        return len(self.frame)

    def column(self, name: str) -> np.ndarray:
        return self.frame[name].to_numpy()

    def matrix(self, names: Sequence[str], *, dtype: Any = None) -> np.ndarray:
        return self.frame.loc[:, list(names)].to_numpy(dtype=dtype)

    def take(self, positions: np.ndarray) -> "PandasFrame":
        return PandasFrame(self.frame.iloc[positions])

    def add_lags(
        self, unit: str, time: str, requests: Sequence[tuple[str, int]]
    ) -> tuple["PandasFrame", tuple[str, ...], np.ndarray, str]:
        output = self.frame.copy()
        names: list[str] = []
        seen: set[tuple[str, int]] = set()
        unique_requests: list[tuple[str, int, str]] = []
        for source, lag in requests:
            if lag > np.iinfo(np.int64).max:
                raise ValueError("lag must not exceed the int64 lag limit")
            if (source, lag) in seen:
                continue
            seen.add((source, lag))
            feature = f"{source}_lag{lag}"
            if feature in output.columns:
                raise ValueError(
                    f"generated lag feature {feature!r} conflicts with a supplied model column"
                )
            names.append(feature)
            unique_requests.append((source, lag, feature))

        if not unique_requests:
            return PandasFrame(output), (), np.ones(len(output), dtype=bool), "none"

        if self.has_consecutive_time(unit, time):
            for source, lag, feature in unique_requests:
                output[feature] = output.groupby(unit, sort=False, observed=True)[
                    source
                ].shift(min(lag, len(self)))
            path = "consecutive_shift"
        else:
            panel_index = pd.MultiIndex.from_frame(output[[unit, time]])
            units = output[unit].to_numpy()
            labels = output[time].to_numpy()
            minimum = np.iinfo(np.int64).min
            for lag in dict.fromkeys(lag for _, lag, _ in unique_requests):
                safe = labels >= minimum + lag
                positions = np.full(len(output), -1, dtype=np.int64)
                targets = labels[safe] - lag
                positions[safe] = panel_index.get_indexer(
                    pd.MultiIndex.from_arrays((units[safe], targets))
                )
                matched = positions >= 0
                for source, requested_lag, feature in unique_requests:
                    if requested_lag == lag:
                        values = np.full(len(output), np.nan, dtype=np.float64)
                        values[matched] = output[source].to_numpy()[positions[matched]]
                        output[feature] = values
            path = "indexed"
        valid = (
            output.loc[:, names].notna().all(axis=1).to_numpy()
            if names
            else np.ones(len(output), dtype=bool)
        )
        return PandasFrame(output), tuple(names), valid, path

    def factorize(self, columns: tuple[str, ...]) -> tuple[np.ndarray, int]:
        if len(columns) == 1:
            codes, categories = pd.factorize(self.frame[columns[0]], sort=True)
        else:
            index = pd.MultiIndex.from_frame(self.frame.loc[:, list(columns)])
            codes, categories = pd.factorize(index, sort=True)
        return np.asarray(codes, dtype=np.int64), len(categories)

    def lead_positions(
        self, unit: str, time: str, horizons: int
    ) -> tuple[np.ndarray, str]:
        return _lead_positions_pandas(self.frame, unit, time, horizons)

    def sample_ids(self, mask: np.ndarray):
        return self.frame.index[mask].copy()


class PolarsFrame(PanelFrame):
    backend = "polars"

    def __init__(self, frame: Any, row_id: str = _ROW_ID) -> None:
        self.frame = frame
        self.row_id = row_id

    @classmethod
    def prepare(
        cls, data: Any, required: Sequence[str], numeric: Sequence[str]
    ) -> tuple["PolarsFrame", bool]:
        import polars as pl

        names = list(dict.fromkeys(required))
        if isinstance(data, pl.LazyFrame):
            available = data.collect_schema().names()
            missing = sorted(set(names).difference(available))
            if missing:
                raise ValueError(f"data is missing required columns: {missing}")
            row_id = _row_id_name(available)
            frame = data.with_row_index(row_id).select(row_id, *names).collect()
        else:
            missing = sorted(set(names).difference(data.columns))
            if missing:
                raise ValueError(f"data is missing required columns: {missing}")
            row_id = _row_id_name(data.columns)
            frame = data.with_row_index(row_id).select(row_id, *names)
        if frame.select(pl.any_horizontal(pl.all().is_null())).to_series().any():
            raise ValueError("v0.1 requires complete data in all model columns")
        unit, time = required[1], required[2]
        canonical = canonical_time(frame.get_column(time).to_numpy())
        frame = frame.with_columns(pl.Series(name=time, values=canonical))
        try:
            numeric_frame = frame.select(
                [
                    pl.col(column).cast(pl.Float64, strict=True)
                    for column in dict.fromkeys(numeric)
                ]
            )
        except (pl.exceptions.PolarsError, TypeError) as error:
            raise ValueError(
                "outcome, shock, and controls must be numeric (real values only)"
            ) from error
        if (
            not numeric_frame.select(
                pl.all_horizontal([pl.col(column).is_finite() for column in numeric])
            )
            .to_series()
            .all()
        ):
            raise ValueError("outcome, shock, and controls must be finite")
        if frame.select(pl.struct(unit, time).is_duplicated().any()).item():
            raise ValueError("unit and time must uniquely identify panel observations")
        frame = frame.sort([unit, time], maintain_order=True)
        row_ids = frame.get_column(row_id).to_numpy()
        input_was_sorted = np.array_equal(
            row_ids, np.arange(len(frame), dtype=row_ids.dtype)
        )
        return cls(frame, row_id), input_was_sorted

    @property
    def columns(self) -> Sequence[str]:
        return self.frame.columns

    def __len__(self) -> int:
        return self.frame.height

    def column(self, name: str) -> np.ndarray:
        return self.frame.get_column(name).to_numpy()

    def matrix(self, names: Sequence[str], *, dtype: Any = None) -> np.ndarray:
        if not names:
            return np.empty((len(self), 0), dtype=dtype or np.float64)
        result = self.frame.select(list(names)).to_numpy()
        return result.astype(dtype, copy=False) if dtype is not None else result

    def take(self, positions: np.ndarray) -> "PolarsFrame":
        return PolarsFrame(self.frame[positions], self.row_id)

    def add_lags(
        self, unit: str, time: str, requests: Sequence[tuple[str, int]]
    ) -> tuple["PolarsFrame", tuple[str, ...], np.ndarray, str]:
        import polars as pl

        names: list[str] = []
        seen: set[tuple[str, int]] = set()
        unique_requests: list[tuple[str, int, str]] = []
        for source, lag in requests:
            if lag > np.iinfo(np.int64).max:
                raise ValueError("lag must not exceed the int64 lag limit")
            if (source, lag) in seen:
                continue
            seen.add((source, lag))
            feature = f"{source}_lag{lag}"
            if feature in self.frame.columns:
                raise ValueError(
                    f"generated lag feature {feature!r} conflicts with a supplied model column"
                )
            names.append(feature)
            unique_requests.append((source, lag, feature))

        if not unique_requests:
            return (
                PolarsFrame(self.frame, self.row_id),
                (),
                np.ones(len(self), dtype=bool),
                "none",
            )

        if self.has_consecutive_time(unit, time):
            expressions = [
                pl.col(source).shift(min(lag, len(self))).over(unit).alias(feature)
                for source, lag, feature in unique_requests
            ]
            output = self.frame.with_columns(expressions)
            valid = output.select(
                pl.all_horizontal([pl.col(name).is_not_null() for name in names])
            ).to_series().to_numpy()
            path = "consecutive_shift"
        else:
            source_position = _row_id_name(self.frame.columns)
            target_time = _row_id_name((*self.frame.columns, source_position))
            lookup = self.frame.select(
                pl.col(unit),
                pl.col(time),
                pl.int_range(0, len(self), dtype=pl.Int64).alias(source_position),
            )
            labels = self.column(time)
            columns = []
            valid = np.ones(len(self), dtype=bool)
            minimum = np.iinfo(np.int64).min
            for lag in dict.fromkeys(lag for _, lag, _ in unique_requests):
                safe = labels >= minimum + lag
                targets = np.zeros(len(self), dtype=np.int64)
                targets[safe] = labels[safe] - lag
                left = self.frame.select(unit).with_columns(
                    pl.Series(target_time, targets)
                )
                matched = left.join(
                    lookup,
                    left_on=[unit, target_time],
                    right_on=[unit, time],
                    how="left",
                    maintain_order="left",
                )
                matched = (
                    matched.get_column(source_position)
                    .fill_null(-1)
                    .to_numpy()
                    .astype(np.int64)
                )
                matched[~safe] = -1
                positions = matched
                matched = positions >= 0
                valid &= matched
                for source, requested_lag, feature in unique_requests:
                    if requested_lag == lag:
                        values = np.full(len(self), np.nan, dtype=np.float64)
                        values[matched] = self.column(source)[positions[matched]]
                        columns.append(pl.Series(feature, values))
            output = self.frame.with_columns(columns)
            path = "indexed"
        return (
            PolarsFrame(output, self.row_id),
            tuple(names),
            np.asarray(valid, dtype=bool),
            path,
        )

    def factorize(self, columns: tuple[str, ...]) -> tuple[np.ndarray, int]:
        categories = (
            self.frame.select(list(columns))
            .unique()
            .sort(list(columns))
            .with_row_index("__fastlp_code__")
        )
        coded = self.frame.select(list(columns)).join(
            categories, on=list(columns), how="left", maintain_order="left"
        )
        return coded.get_column("__fastlp_code__").to_numpy().astype(np.int64), len(
            categories
        )

    def lead_positions(
        self, unit: str, time: str, horizons: int
    ) -> tuple[np.ndarray, str]:
        shape = self.balanced_shape(unit, time)
        if shape is not None:
            result = _balanced_leads(self.column(time), shape, horizons)
            if result is not None:
                return result, "balanced_arithmetic"
        import polars as pl

        numeric_time = self.column(time)
        if horizons and int(numeric_time.max()) > np.iinfo(np.int64).max - horizons:
            raise ValueError("time plus requested horizons exceeds the int64 range")
        positions = np.empty((len(self), horizons + 1), dtype=np.int64)
        positions[:, 0] = np.arange(len(self), dtype=np.int64)
        lookup = self.frame.select(
            pl.col(unit),
            pl.col(time),
            pl.int_range(0, len(self), dtype=pl.Int64).alias("__future__"),
        )
        left = self.frame.select(pl.col(unit), pl.col(time))
        for horizon in range(1, horizons + 1):
            matched = left.with_columns(
                (pl.col(time) + horizon).alias("__target__")
            ).join(
                lookup,
                left_on=[unit, "__target__"],
                right_on=[unit, time],
                how="left",
                maintain_order="left",
            )
            positions[:, horizon] = (
                matched.get_column("__future__").fill_null(-1).to_numpy()
            )
        return positions, "indexed"

    def sample_ids(self, mask: np.ndarray) -> np.ndarray:
        return self.column(self.row_id)[mask].copy()


def _balanced_leads(
    labels: np.ndarray, shape: tuple[int, int], horizons: int
) -> np.ndarray | None:
    n_units, n_periods = shape
    numeric = labels.reshape(n_units, n_periods)[0]
    start = int(numeric[0])
    if start > np.iinfo(np.int64).max - (n_periods - 1):
        return None
    expected = start + np.arange(n_periods, dtype=np.int64)
    if not np.array_equal(numeric, expected):
        return None
    positions = np.full((n_units * n_periods, horizons + 1), -1, dtype=np.int64)
    base = np.arange(n_units * n_periods, dtype=np.int64).reshape(n_units, n_periods)
    view = positions.reshape(n_units, n_periods, -1)
    for horizon in range(horizons + 1):
        if horizon < n_periods:
            view[:, : n_periods - horizon, horizon] = base[:, horizon:]
    return positions


def _lead_positions_pandas(
    frame: pd.DataFrame, unit: str, time: str, horizons: int
) -> tuple[np.ndarray, str]:
    wrapped = PandasFrame(frame)
    shape = wrapped.balanced_shape(unit, time)
    if shape is not None:
        result = _balanced_leads(wrapped.column(time), shape, horizons)
        if result is not None:
            return result, "balanced_arithmetic"
    numeric_time = frame[time].to_numpy()
    if horizons and int(numeric_time.max()) > np.iinfo(np.int64).max - horizons:
        raise ValueError("time plus requested horizons exceeds the int64 range")
    panel_index = pd.MultiIndex.from_frame(frame[[unit, time]])
    units = frame[unit].to_numpy()
    positions = np.full((len(frame), horizons + 1), -1, dtype=np.int64)
    positions[:, 0] = np.arange(len(frame), dtype=np.int64)
    for horizon in range(1, horizons + 1):
        positions[:, horizon] = panel_index.get_indexer(
            pd.MultiIndex.from_arrays((units, numeric_time + horizon))
        )
    return positions, "indexed"


def prepare_panel(
    data: Any, required: Sequence[str], numeric: Sequence[str]
) -> tuple[PanelFrame, bool]:
    if isinstance(data, pd.DataFrame):
        return PandasFrame.prepare(data, required, numeric)
    try:
        import polars as pl
    except ImportError:
        pl = None
    if pl is not None and isinstance(data, (pl.DataFrame, pl.LazyFrame)):
        return PolarsFrame.prepare(data, required, numeric)
    module = type(data).__module__.split(".", 1)[0]
    if module == "polars" and pl is None:
        raise ImportError(
            "Polars input requires the optional dependency: install fastlp-py[polars]"
        )
    raise TypeError(
        "data must be a pandas DataFrame, polars DataFrame, or polars LazyFrame"
    )
