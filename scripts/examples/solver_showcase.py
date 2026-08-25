"""
Solver showcase: compare all TRF solvers on simulated data with outliers.

This script demonstrates the Solver Pattern abstraction (issue #12).
It simulates a smooth (low-pass filtered) stimulus convolved with a known
Gaussian TRF kernel, then adds Gaussian noise and gross outliers to the
output.  The smooth stimulus makes the design matrix ill-conditioned, so
plain OLS fails — ridge or smoothness regularization is needed.  The
outliers make robust (Cauchy-loss) fitting beneficial.

Each solver is fit on the same data and compared on:
  - wall-clock fit time
  - correlation between recovered and ground-truth kernel
  - a side-by-side plot of the recovered kernel vs ground truth
  - a residual plot highlighting outliers

Solvers compared:
  - LSTSQSolver             (ordinary least squares, no regularization)
  - SVDSolver (ridge)        (SVD ridge regression with L2)
  - SVDSolver + smoothness M  (SVD with quadratic smoothness regularizer)
  - ConjugateGradientSolver  (CG on normal equations with L2)
  - IRLSSolver (Cauchy)       (robust Cauchy-loss IRLS with L2)

Note: ScipyRobustSolver is intentionally omitted — it is a reference
validator (unregularized SciPy nonlinear least-squares) and operates on
the full sample matrix rather than compressed normal equations, making
it ~100x slower than IRLS with no accuracy benefit.

Run with:
    uv run python scripts/examples/solver_showcase.py
"""
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

from pyeeg.models.trf import TRFEstimator
from pyeeg.solvers import (
    ConjugateGradientSolver,
    IRLSSolver,
    LSTSQSolver,
    SVDSolver,
)
from pyeeg.simulate import dummy_trf_kernel, simulate_smooth_input, simulate_trf_output
from pyeeg.utils import lag_matrix

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
SRATE = 100          # Hz
DURATION = 60.0      # seconds
TMIN, TMAX = -0.2, 0.5
ALPHA = 100.0        # ridge strength — needs to be high for smooth X
OUTLIER_FRAC = 0.02  # fraction of samples corrupted by gross outliers
OUTLIER_MAG = 15.0   # outlier magnitude in y-clean std units
NOISE_FRAC = 0.1     # Gaussian noise as fraction of y std
SEED = 42

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Build the ground-truth kernel and stimulus/response
# ---------------------------------------------------------------------------
t_kernel, kernel = dummy_trf_kernel(
    tmin=TMIN, tmax=TMAX, srate=SRATE, tloc=0.15, sigma=0.08,
    kernel_type="gaussian",
)

# Smooth (low-pass filtered) stimulus — makes the lag matrix ill-conditioned
# so that OLS fails and ridge / smoothness regularization is necessary.
_, X = simulate_smooth_input(dur=DURATION, srate=SRATE, fmax=8, seed=SEED)
X = X[:, None]  # (n_samples, 1 feature)

# Generate output by convolving stimulus with the kernel
y_clean = simulate_trf_output(t_kernel, kernel, X[:, 0], srate=SRATE)

# Add Gaussian noise
y = y_clean + rng.standard_normal(len(y_clean)) * NOISE_FRAC * y_clean.std()

# Add gross outliers (heavy-tailed contamination)
n_samples = len(y)
n_outliers = int(n_samples * OUTLIER_FRAC)
outlier_idx = rng.choice(n_samples, size=n_outliers, replace=False)
y[outlier_idx] += rng.choice([-1, 1], size=n_outliers) * OUTLIER_MAG * y_clean.std()

print(f"Simulated {DURATION:.0f}s at {SRATE} Hz ({n_samples} samples)")
print(f"  {len(kernel)} lags ({TMIN}s to {TMAX}s), 1 feature, 1 channel")
print(f"  Noise: {NOISE_FRAC*100:.0f}% of y std, {n_outliers} outliers "
      f"({OUTLIER_FRAC*100:.0f}% @ +/-{OUTLIER_MAG} sigma)")
print()

# ---------------------------------------------------------------------------
# Define the solvers to compare
# ---------------------------------------------------------------------------
# Each entry: (label, solver_instance, extra_trf_kwargs)
solvers = [
    ("LSTSQ (OLS)", LSTSQSolver(),
     {"alpha": 0.0}),
    ("SVD (ridge a=100)", SVDSolver(),
     {"alpha": ALPHA}),
    ("SVD + smoothness M", SVDSolver(),
     {"alpha": ALPHA, "quadratic_reg": "smoothness"}),
    ("CG (ridge a=100)", ConjugateGradientSolver(tol=1e-10),
     {"alpha": ALPHA}),
    ("IRLS Cauchy (a=100)", IRLSSolver(max_iter=50, tol=1e-8),
     {"alpha": ALPHA}),
]

# ---------------------------------------------------------------------------
# Fit with each solver and collect results
# ---------------------------------------------------------------------------
results = []  # (label, trf, elapsed, corr, recovered, y_pred)

print(f"{'Solver':<25} {'Time (ms)':>10} {'Corr':>8}")
print("-" * 47)

for label, solver, extra_kwargs in solvers:
    trf_kwargs = dict(
        tmin=TMIN, tmax=TMAX, srate=SRATE,
        verbose=False, fit_intercept=True,
        solver=solver,
    )
    trf_kwargs.update(extra_kwargs)
    trf = TRFEstimator(**trf_kwargs)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        trf.fit(X, y[:, None], lagged=False, drop=True)
        elapsed = (time.perf_counter() - t0) * 1000

    # Correlation between recovered kernel and ground truth
    recovered = trf.coef_[:, 0, 0]  # (n_lags, n_feats=1, n_chans=1)
    corr = np.corrcoef(recovered, kernel)[0, 1]

    # Compute predictions on the valid samples for residual plot
    X_lag = lag_matrix(X, lags=trf.lags, mode="valid", fill_value=0.0,
                       block_order="lags")
    if trf.fit_intercept:
        X_lag = np.hstack([np.ones((len(X_lag), 1)), X_lag])
    betas_flat = trf._coef_to_beta(trf.coef_)
    if trf.fit_intercept:
        betas_flat = np.vstack([trf.intercept_[None, :], betas_flat])
    y_pred = X_lag @ betas_flat

    results.append((label, trf, elapsed, corr, recovered, y_pred))
    print(f"{label:<25} {elapsed:>10.1f} {corr:>8.3f}")

print()

# ---------------------------------------------------------------------------
# Plot: kernel recovery + residuals
# ---------------------------------------------------------------------------
n_solvers = len(results)
fig, axes = plt.subplots(2, n_solvers, figsize=(3.5 * n_solvers, 7),
                         sharex="col", sharey="row")
if n_solvers == 1:
    axes = axes.reshape(2, 1)

t_lags = np.arange(TMIN, TMAX, 1 / SRATE)

# Use a fixed y-range based on the ground-truth kernel so every panel
# shows the ground truth at the same scale regardless of solver quality.
kernel_ylim = (-kernel.max() * 0.2, kernel.max() * 1.3)

for col, (label, trf, elapsed, corr, recovered, y_pred) in enumerate(results):
    # Top row: recovered kernel vs ground truth
    ax = axes[0, col]
    ax.plot(t_lags, kernel, "k--", lw=2, label="Ground truth")
    ax.plot(t_lags, recovered, "r-", lw=1.5, label="Recovered")
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("Lag (s)")
    ax.set_ylim(kernel_ylim)
    if col == 0:
        ax.set_ylabel("TRF amplitude")
        ax.legend(fontsize=8, loc="upper left")
    ax.text(0.95, 0.95, f"r = {corr:.3f}\n{elapsed:.0f} ms",
            transform=ax.transAxes, va="top", ha="right", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))

    # Bottom row: residuals highlighting outliers
    ax = axes[1, col]
    valid_idx = np.where(trf.valid_samples_)[0]
    residual = y[valid_idx] - y_pred[:, 0]
    ax.plot(residual, lw=0.3, color="steelblue", alpha=0.7)

    # Mark outliers that fall within valid samples
    outlier_mask = np.isin(valid_idx, outlier_idx)
    ax.scatter(np.where(outlier_mask)[0], residual[outlier_mask],
               color="red", s=10, zorder=5, label="Outlier")
    ax.set_xlabel("Valid sample")
    if col == 0:
        ax.set_ylabel("Residual")
        ax.legend(fontsize=8)

fig.suptitle("Solver Showcase: TRF Recovery with Outliers on Smooth Stimulus",
             fontsize=13, y=1.01)
fig.tight_layout()

# ---------------------------------------------------------------------------
# Summary bar chart
# ---------------------------------------------------------------------------
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
labels = [r[0] for r in results]
times = [r[2] for r in results]
corrs = [r[3] for r in results]
colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))

ax1.barh(labels, times, color=colors)
ax1.set_xlabel("Fit time (ms)")
ax1.set_title("Speed comparison (log scale)")
ax1.set_xscale("log")

ax2.barh(labels, corrs, color=colors)
ax2.set_xlabel("Correlation with ground truth")
ax2.set_xlim(0, 1)
ax2.set_title("Accuracy comparison")

fig2.suptitle("Solver Pattern: Speed vs Accuracy", fontsize=13)
fig2.tight_layout()

plt.show()
