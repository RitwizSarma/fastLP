"""Publication-oriented impulse-response plotting."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .estimator import LocalProjection


def plot_irf(
    fitted: "LocalProjection",
    coefficient: str | None = None,
    *,
    at: Sequence[float] | None = None,
    levels: Sequence[object] | None = None,
    ax: "Axes | None" = None,
    figsize: tuple[float, float] = (8.0, 4.8),
    title: str | None = None,
    xlabel: str = "Horizon",
    ylabel: str | None = None,
    color: str = "#2563eb",
    band_color: str = "#bfdbfe",
    band_alpha: float = 0.55,
    marker: str = "o",
    linewidth: float = 2.0,
    markersize: float = 5.0,
    show_zero_line: bool = True,
    show: bool = False,
) -> "Axes":
    """Plot one fitted impulse response with its pointwise confidence band.

    The estimator's configured confidence level is used. When a model has
    multiple shocks and ``coefficient`` is omitted, the first supplied shock
    is plotted. The returned Matplotlib axes can be further customized or
    exported through ``ax.figure.savefig(...)``.
    """
    fitted._require_fitted()
    import matplotlib.pyplot as plt

    if coefficient is None:
        coefficient = str(fitted.shock_names_in_[0])
    shocks = [str(name) for name in fitted.shock_names_in_]
    if coefficient not in shocks:
        raise ValueError(
            f"coefficient {coefficient!r} is not a fitted shock; choose one of {shocks}"
        )
    if not isinstance(band_alpha, (int, float)) or not 0 <= band_alpha <= 1:
        raise ValueError("band_alpha must be between zero and one")

    responses = fitted.irf_frame(coefficient, at=at, levels=levels)
    paths = list(responses.groupby(["level", "at"], dropna=False, sort=False))

    if ax is None:
        _, ax = plt.subplots(figsize=figsize, facecolor="white")
    figure = ax.figure
    figure.patch.set_facecolor("white")
    ax.set_facecolor("white")

    palette = plt.get_cmap("tab10")
    for index, ((level, at_value), path) in enumerate(paths):
        level = None if pd.isna(level) else level
        at_value = None if pd.isna(at_value) else at_value
        line_color = color if len(paths) == 1 else palette(index % 10)
        fill_color = band_color if len(paths) == 1 else line_color
        label = (
            coefficient if len(paths) == 1 and level is None and at_value is None
            else f"{level}" if level is not None
            else f"at {at_value:g}"
        )
        ax.fill_between(
            path["horizon"].to_numpy(), path["ci_low"].to_numpy(),
            path["ci_high"].to_numpy(), color=fill_color,
            alpha=band_alpha if len(paths) == 1 else min(band_alpha, 0.18),
            linewidth=0,
            label=f"{100 * (1 - fitted.alpha):.0f}% confidence interval" if len(paths) == 1 else None,
            zorder=1,
        )
        ax.plot(
            path["horizon"].to_numpy(), path["estimate"].to_numpy(),
            color=line_color, marker=marker, linewidth=linewidth,
            markersize=markersize, markerfacecolor="white",
            markeredgewidth=1.5, label=label, zorder=3,
        )
    if show_zero_line:
        ax.axhline(0.0, color="#64748b", linewidth=1.0, linestyle="--", zorder=2)

    response_label = "Cumulative response" if fitted.response_ == "cumulative" else "Response"
    ax.set_title(title or f"{response_label} to {coefficient}", loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or response_label)
    ax.set_xticks(fitted.horizons_)
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(colors="#334155")
    ax.legend(frameon=False, loc="best")
    figure.tight_layout()
    if show:
        plt.show()
    return ax
