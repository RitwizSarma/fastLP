"""Explicit, backend-neutral specifications for interaction terms."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


_MAX_LEVELS = 32


@dataclass(frozen=True)
class Category:
    column: str
    levels: tuple[Any, ...] | None = None


@dataclass(frozen=True)
class Interaction:
    factors: tuple[str, ...]
    by: Category | str | None = None
    name: str | None = None


@dataclass(frozen=True)
class FixedEffect:
    columns: tuple[str, ...]


@dataclass(frozen=True)
class GeneratedTerm:
    name: str
    factors: tuple[str, ...]
    category: Category | None = None
    level: Any = None
    nuisance: bool = False


@dataclass(frozen=True)
class ShockFamily:
    name: str
    kind: str
    basis: tuple[str, ...]
    levels: tuple[Any, ...] = ()
    moderator: str | None = None


def _column(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("term factors must be nonempty column names")
    return value


def C(column: str, *, levels=None) -> Category:
    """Mark a column as a categorical shock moderator."""
    column = _column(column)
    if levels is None:
        return Category(column)
    if isinstance(levels, (str, bytes)):
        raise ValueError("categorical levels must be a sequence")
    chosen = tuple(levels)
    if not chosen or len(chosen) > _MAX_LEVELS:
        raise ValueError(f"categorical levels must contain 1 to {_MAX_LEVELS} values")
    try:
        if len(set(chosen)) != len(chosen):
            raise ValueError("categorical levels must be unique")
    except TypeError as error:
        raise ValueError("categorical levels must be hashable") from error
    if any(pd.isna(level) for level in chosen):
        raise ValueError("categorical levels must not be missing")
    return Category(column, chosen)


def interact(
    *columns: str, by: Category | str | None = None, name: str | None = None
) -> Interaction:
    """Specify a numeric product, optionally moderated by a category or number."""
    if not columns:
        raise ValueError("interact requires at least one factor")
    factors = tuple(_column(column) for column in columns)
    if len(set(factors)) != len(factors):
        raise ValueError("interaction factors must be distinct")
    if by is not None and not isinstance(by, (Category, str)):
        raise ValueError("by must be a column name or C(...) term")
    moderator = by.column if isinstance(by, Category) else by
    if moderator is not None:
        _column(moderator)
        if moderator in factors:
            raise ValueError("interaction moderator must differ from its factors")
    if name is not None:
        _column(name)
    return Interaction(factors, by, name)


def fe(*columns: str) -> FixedEffect:
    """Specify one fixed-effect partition made from joint categorical keys."""
    if not columns:
        raise ValueError("fe requires at least one column")
    keys = tuple(_column(column) for column in columns)
    if len(set(keys)) != len(keys):
        raise ValueError("fixed-effect keys must be distinct")
    return FixedEffect(keys)


def dependencies(terms: tuple[str | Interaction, ...]) -> tuple[str, ...]:
    names: list[str] = []
    for term in terms:
        if isinstance(term, str):
            names.append(term)
        else:
            names.extend(term.factors)
            if term.by is not None:
                names.append(term.by.column if isinstance(term.by, Category) else term.by)
    return tuple(dict.fromkeys(names))


def _levels(frame, category: Category) -> tuple[Any, ...]:
    observed = tuple(pd.unique(frame.column(category.column)))
    if len(observed) > _MAX_LEVELS:
        raise ValueError(
            f"{category.column!r} has {len(observed)} levels; group them into "
            f"at most {_MAX_LEVELS} levels before fitting"
        )
    if category.levels is None:
        return observed
    if set(observed) != set(category.levels):
        raise ValueError(
            f"declared levels for {category.column!r} must match its observed levels"
        )
    return category.levels


def compile_terms(
    frame, shocks: tuple[str | Interaction, ...], controls: tuple[str | Interaction, ...]
):
    """Compile products and required hierarchy without materializing columns."""
    generated: dict[tuple[Any, ...], GeneratedTerm] = {}
    families: list[ShockFamily] = []
    explicit: list[str] = []
    nuisance: list[str] = []
    used_names = set(frame.columns)
    family_names: set[str] = set()
    reference_levels: dict[str, Any] = {}

    def add(
        factors: tuple[str, ...], *, category: Category | None = None,
        level: Any = None, auto: bool = False,
    ) -> str:
        key = (tuple(sorted(factors)), category.column if category else None, level)
        if key in generated:
            name = generated[key].name
            (nuisance if auto else explicit).append(name)
            return name
        name = ":".join(factors)
        if category is not None:
            name = f"{name + ':' if name else ''}{category.column}[{level}]"
        if name in used_names:
            # A single-factor generated term can reuse the input column.
            if category is None and len(factors) == 1 and name == factors[0]:
                (nuisance if auto else explicit).append(name)
                return name
            raise ValueError(f"generated term {name!r} conflicts with an input column")
        used_names.add(name)
        generated[key] = GeneratedTerm(name, factors, category, level, auto)
        (nuisance if auto else explicit).append(name)
        return name

    for term in shocks:
        if isinstance(term, str):
            if term in family_names:
                raise ValueError(f"duplicate shock family {term!r}")
            families.append(ShockFamily(term, "plain", (term,)))
            family_names.add(term)
            explicit.append(term)
            continue
        label = term.name or ":".join(term.factors)
        if label in family_names:
            raise ValueError(f"duplicate shock family {label!r}")
        family_names.add(label)
        factors = term.factors
        subsets = [
            subset for width in range(1, len(factors))
            for subset in combinations(factors, width)
        ]
        for subset in subsets:
            add(subset, auto=True)
        if isinstance(term.by, Category):
            levels = _levels(frame, term.by)
            basis = tuple(add(factors, category=term.by, level=level) for level in levels)
            reference = reference_levels.setdefault(term.by.column, levels[0])
            for level in levels:
                if level == reference:
                    continue
                add((), category=term.by, level=level, auto=True)
                for subset in subsets:
                    add(subset, category=term.by, level=level, auto=True)
            families.append(ShockFamily(label, "categorical", basis, levels, term.by.column))
        elif isinstance(term.by, str):
            base = add(factors)
            moderated = add((*factors, term.by))
            add((term.by,), auto=True)
            for subset in subsets:
                add((*subset, term.by), auto=True)
            families.append(ShockFamily(label, "continuous", (base, moderated), (), term.by))
        else:
            families.append(ShockFamily(label, "plain", (add(factors),)))

    for term in controls:
        if isinstance(term, str):
            explicit.append(term)
        else:
            if term.by is not None:
                raise ValueError("by= is supported for shocks; use explicit controls otherwise")
            explicit.append(add(term.factors))
    names = tuple(dict.fromkeys((*explicit, *nuisance)))
    term_by_name = {term.name: term for term in generated.values()}
    return names, term_by_name, tuple(families), frozenset(set(nuisance) - set(explicit))


def upper_bound_features(shocks, controls) -> int:
    """Conservative feature count for planning before reading a panel."""
    count = 0
    for term in shocks:
        if isinstance(term, str):
            count += 1
        else:
            if isinstance(term.by, Category):
                levels = len(term.by.levels) if term.by.levels is not None else _MAX_LEVELS
                count += levels * 2 ** len(term.factors)
            elif term.by is not None:
                count += 2 ** (len(term.factors) + 1) - 1
            else:
                count += 2 ** len(term.factors) - 1
    count += len(controls)
    return count


def materialize_terms(frame, generated: dict[str, GeneratedTerm]):
    """Add generated regressors using the input frame's native backend."""
    if not generated:
        return frame
    from ._frame import PandasFrame, PolarsFrame

    if isinstance(frame, PolarsFrame):
        import polars as pl

        expressions = []
        for term in generated.values():
            expression = pl.lit(1.0)
            for factor in term.factors:
                expression *= pl.col(factor).cast(pl.Float64)
            if term.category is not None:
                expression *= (
                    pl.col(term.category.column) == pl.lit(term.level)
                ).cast(pl.Float64)
            expressions.append(expression.alias(term.name))
        return PolarsFrame(frame.frame.with_columns(expressions), frame.row_id)

    assert isinstance(frame, PandasFrame)
    # PandasFrame.prepare already owns its selected frame. The ordinary lag
    # path makes its own copy, so copying here would duplicate a large panel.
    output = frame.frame
    for term in generated.values():
        values = np.ones(len(frame), dtype=np.float64)
        for factor in term.factors:
            values *= frame.column(factor).astype(np.float64, copy=False)
        if term.category is not None:
            values *= frame.column(term.category.column) == term.level
        output[term.name] = values
    return PandasFrame(output)


def design_matrix(frame, names: tuple[str, ...], generated: dict[str, GeneratedTerm]) -> np.ndarray:
    """Build one numeric matrix without storing nonlagged terms on the frame."""
    if not generated or all(name in frame.columns for name in names):
        return frame.matrix(names, dtype=np.float64)
    if frame.backend == "polars":
        import polars as pl

        expressions = []
        for name in names:
            term = generated.get(name)
            if name in frame.columns or term is None:
                expressions.append(pl.col(name).cast(pl.Float64))
                continue
            expression = pl.lit(1.0)
            for factor in term.factors:
                expression *= pl.col(factor).cast(pl.Float64)
            if term.category is not None:
                expression *= (
                    pl.col(term.category.column) == pl.lit(term.level)
                ).cast(pl.Float64)
            expressions.append(expression.alias(name))
        return frame.frame.select(expressions).to_numpy().astype(np.float64, copy=False)

    columns = []
    for name in names:
        term = generated.get(name)
        if name in frame.columns or term is None:
            columns.append(frame.column(name).astype(np.float64, copy=False))
            continue
        values = np.ones(len(frame), dtype=np.float64)
        for factor in term.factors:
            values *= frame.column(factor).astype(np.float64, copy=False)
        if term.category is not None:
            values *= frame.column(term.category.column) == term.level
        columns.append(values)
    return np.column_stack(columns)


__all__ = ["C", "Category", "FixedEffect", "Interaction", "fe", "interact"]
