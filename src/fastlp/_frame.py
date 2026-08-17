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
        self, unit: str, requests: Sequence[tuple[str, int]]
    ) -> tuple["PanelFrame", tuple[str, ...], np.ndarray]: ...

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
        try:
            for column in dict.fromkeys(numeric):
                frame[column] = pd.to_numeric(frame[column], errors="raise").astype(np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("outcome, shock, and controls must be numeric") from error
        if not np.isfinite(frame.loc[:, list(numeric)].to_numpy(dtype=float)).all():
            raise ValueError("outcome, shock, and controls must be finite")
        unit, time = required[1], required[2]
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
        self, unit: str, requests: Sequence[tuple[str, int]]
    ) -> tuple["PandasFrame", tuple[str, ...], np.ndarray]:
        output = self.frame.copy()
        names: list[str] = []
        seen: set[tuple[str, int]] = set()
        for source, lag in requests:
            if (source, lag) in seen:
                continue
            seen.add((source, lag))
            feature = f"{source}_lag{lag}"
            if feature in output.columns:
                raise ValueError(
                    f"generated lag feature {feature!r} conflicts with a supplied model column"
                )
            output[feature] = output.groupby(unit, sort=False, observed=True)[source].shift(lag)
            names.append(feature)
        valid = (
            output.loc[:, names].notna().all(axis=1).to_numpy()
            if names
            else np.ones(len(output), dtype=bool)
        )
        return PandasFrame(output), tuple(names), valid

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
        try:
            frame = frame.with_columns(
                [pl.col(column).cast(pl.Float64, strict=True) for column in dict.fromkeys(numeric)]
            )
        except (pl.exceptions.PolarsError, TypeError) as error:
            raise ValueError("outcome, shock, and controls must be numeric") from error
        if not frame.select(
            pl.all_horizontal([pl.col(column).is_finite() for column in numeric])
        ).to_series().all():
            raise ValueError("outcome, shock, and controls must be finite")
        unit, time = required[1], required[2]
        if frame.select(pl.struct(unit, time).is_duplicated().any()).item():
            raise ValueError("unit and time must uniquely identify panel observations")
        frame = frame.sort([unit, time], maintain_order=True)
        row_ids = frame.get_column(row_id).to_numpy()
        input_was_sorted = np.array_equal(row_ids, np.arange(len(frame), dtype=row_ids.dtype))
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
        self, unit: str, requests: Sequence[tuple[str, int]]
    ) -> tuple["PolarsFrame", tuple[str, ...], np.ndarray]:
        import polars as pl

        expressions = []
        names: list[str] = []
        seen: set[tuple[str, int]] = set()
        for source, lag in requests:
            if (source, lag) in seen:
                continue
            seen.add((source, lag))
            feature = f"{source}_lag{lag}"
            if feature in self.frame.columns:
                raise ValueError(
                    f"generated lag feature {feature!r} conflicts with a supplied model column"
                )
            expressions.append(pl.col(source).shift(lag).over(unit).alias(feature))
            names.append(feature)
        output = self.frame.with_columns(expressions) if expressions else self.frame
        valid = (
            output.select(pl.all_horizontal([pl.col(name).is_not_null() for name in names]))
            .to_series()
            .to_numpy()
            if names
            else np.ones(len(self), dtype=bool)
        )
        return PolarsFrame(output, self.row_id), tuple(names), np.asarray(valid, dtype=bool)

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
        return coded.get_column("__fastlp_code__").to_numpy().astype(np.int64), len(categories)

    def lead_positions(
        self, unit: str, time: str, horizons: int
    ) -> tuple[np.ndarray, str]:
        shape = self.balanced_shape(unit, time)
        if shape is not None:
            result = _balanced_leads(self.column(time), shape, horizons)
            if result is not None:
                return result, "balanced_arithmetic"
        try:
            numeric_time = self.column(time).astype(np.float64)
        except (TypeError, ValueError):
            return _balanced_label_leads(self, unit, time, horizons)
        if not np.isfinite(numeric_time).all():
            raise ValueError("unbalanced panels require numeric time labels for exact time + horizon alignment")
        import polars as pl

        positions = np.empty((len(self), horizons + 1), dtype=np.int64)
        positions[:, 0] = np.arange(len(self), dtype=np.int64)
        lookup = self.frame.select(
            pl.col(unit), pl.col(time), pl.int_range(0, len(self), dtype=pl.Int64).alias("__future__")
        )
        left = self.frame.select(pl.col(unit), pl.col(time))
        for horizon in range(1, horizons + 1):
            matched = left.with_columns((pl.col(time) + horizon).alias("__target__")).join(
                lookup,
                left_on=[unit, "__target__"],
                right_on=[unit, time],
                how="left",
                maintain_order="left",
            )
            positions[:, horizon] = matched.get_column("__future__").fill_null(-1).to_numpy()
        return positions, "indexed"

    def sample_ids(self, mask: np.ndarray) -> np.ndarray:
        return self.column(self.row_id)[mask].copy()


def _balanced_leads(
    labels: np.ndarray, shape: tuple[int, int], horizons: int
) -> np.ndarray | None:
    n_units, n_periods = shape
    try:
        numeric = labels.reshape(n_units, n_periods)[0].astype(float)
        if not np.array_equal(numeric, numeric[0] + np.arange(n_periods)):
            return None
    except (TypeError, ValueError):
        return None
    positions = np.full((n_units * n_periods, horizons + 1), -1, dtype=np.int64)
    base = np.arange(n_units * n_periods, dtype=np.int64).reshape(n_units, n_periods)
    view = positions.reshape(n_units, n_periods, -1)
    for horizon in range(horizons + 1):
        if horizon < n_periods:
            view[:, : n_periods - horizon, horizon] = base[:, horizon:]
    return positions


def _balanced_label_leads(
    frame: PanelFrame, unit: str, time: str, horizons: int
) -> tuple[np.ndarray, str]:
    shape = frame.balanced_shape(unit, time)
    if shape is None:
        raise ValueError(
            "unbalanced panels require numeric time labels for exact time + horizon alignment"
        )
    n_units, n_periods = shape
    positions = np.full((len(frame), horizons + 1), -1, dtype=np.int64)
    base = np.arange(len(frame), dtype=np.int64).reshape(n_units, n_periods)
    view = positions.reshape(n_units, n_periods, -1)
    for horizon in range(horizons + 1):
        if horizon < n_periods:
            view[:, : n_periods - horizon, horizon] = base[:, horizon:]
    return positions, "balanced_labels"


def _lead_positions_pandas(
    frame: pd.DataFrame, unit: str, time: str, horizons: int
) -> tuple[np.ndarray, str]:
    wrapped = PandasFrame(frame)
    shape = wrapped.balanced_shape(unit, time)
    if shape is not None:
        result = _balanced_leads(wrapped.column(time), shape, horizons)
        if result is not None:
            return result, "balanced_arithmetic"
    try:
        numeric_time = pd.to_numeric(frame[time], errors="raise").to_numpy()
    except (TypeError, ValueError):
        return _balanced_label_leads(wrapped, unit, time, horizons)
    panel_index = pd.MultiIndex.from_frame(frame[[unit, time]])
    units = frame[unit].to_numpy()
    positions = np.empty((len(frame), horizons + 1), dtype=np.int64)
    for horizon in range(horizons + 1):
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
    raise TypeError("data must be a pandas DataFrame, polars DataFrame, or polars LazyFrame")
