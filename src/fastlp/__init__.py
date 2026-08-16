"""Cached local-projection estimation for balanced and unbalanced panel data."""

from importlib.metadata import version

from .estimator import FewClustersWarning, LocalProjection
from .plotting import plot_irf

__version__ = version("fastlp")

__all__ = ["FewClustersWarning", "LocalProjection", "__version__", "plot_irf"]
