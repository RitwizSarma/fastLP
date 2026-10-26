"""Cached local-projection estimation for balanced and unbalanced panel data."""

from importlib.metadata import version

from .estimator import FewClustersWarning, LocalProjection
from ._progress import ProgressEvent
from .plotting import plot_irf
from . import sugar

__version__ = version("fastlp-py")

__all__ = ["FewClustersWarning", "LocalProjection", "ProgressEvent", "__version__", "plot_irf", "sugar"]
