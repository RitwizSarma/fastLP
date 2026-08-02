"""Cached local-projection estimation for balanced and unbalanced panel data."""

from .estimator import FewClustersWarning, LocalProjection
from .plotting import plot_irf

__all__ = ["FewClustersWarning", "LocalProjection", "plot_irf"]
