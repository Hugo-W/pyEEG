Statistical Inference
=====================

.. automodule:: pyeeg.stats

The ``pyeeg.stats`` module provides nonparametric statistical inference for
TRF (Temporal Response Function) analysis. It complements the parametric
``tvals_``/``pvals_`` path in :class:`pyeeg.models.TRFEstimator`.

Key features:

- **Permutation test** (``permutation_test_trf``): circular-shift surrogate
  test with FWE correction via the max-statistic. Supports ``stat="zscore"``
  (default: internal pre-lag z-scoring + refit, clean for any solver),
  ``stat="t"`` (OLS only), ``stat="coef"``, and ``stat="perm_norm"``
  (permutation-null normalised).
- **Cluster-based correction** (``cluster_based_permutation_test``):
  Maris & Oostenveld (2007) cluster-level FWE. Positive and negative clusters
  formed separately; adjacency (lag, explicit, sparse, or none).
- **Bootstrap CIs** (``bootstrap_ci_trf``): paired circular block bootstrap
  with boundary drop and auto block-size estimation.
- **Jackknife SE** (``jackknife_se_trf``): leave-one-epoch-out standard error
  and confidence intervals (standalone, not a permutation stat).
- **Cross-subject consistency** (``cross_subject_consistency``): descriptive
  pairwise or leave-one-out reliability (Pearson or cosine).
- **Group-level test** (``group_level_test``): sign-flip permutation test on
  subject coefficient maps (H0: population mean = 0).

No MNE imports occur in this module. Spatial adjacency matrices must be
supplied by the user (e.g. from ``mne.channels.find_ch_adjacency``).
