API reference
=============

The public interface consists of the local-projection estimator, its plotting
helper, and the warning emitted for clustering dimensions with few groups.

LocalProjection
---------------

.. autoclass:: fastlp.LocalProjection
   :members: fit, to_frame, summary, plot_irf, irfplot
   :special-members: __init__

Plotting function
-----------------

.. autofunction:: fastlp.plot_irf

Warnings
--------

.. autoclass:: fastlp.FewClustersWarning
