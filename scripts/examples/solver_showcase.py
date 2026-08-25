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

    results.append((label, trf, elapsed, corr, recovered))
    print(f"{label:<25} {elapsed:>10.1f} {corr:>8.3f}")

print()

# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 12,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Okabe-Ito / Wong palette — perceptually uniform, print-friendly
SOLVER_COLORS = {
    "LSTSQ (OLS)": "#4E79A7",
    "SVD (ridge a=100)": "#F28E2B",
    "SVD + smoothness M": "#E15759",
    "CG (ridge a=100)": "#76B7B2",
    "IRLS Cauchy (a=100)": "#59A14F",
}

# ---------------------------------------------------------------------------
# Figure 1: Kernel recovery (single row of 5 panels)
# ---------------------------------------------------------------------------
n_solvers = len(results)
fig1, axes = plt.subplots(1, n_solvers, figsize=(16, 3.5), sharey=True)
if n_solvers == 1:
    axes = [axes]

t_lags = np.arange(TMIN, TMAX, 1 / SRATE)
kernel_ylim = (-kernel.max() * 0.2, kernel.max() * 1.3)

for ax, (label, trf, elapsed, corr, recovered) in zip(axes, results):
    ax.plot(t_lags, kernel, "k--", lw=2,
            label="Ground truth" if ax is axes[0] else "")
    ax.plot(t_lags, recovered, color=SOLVER_COLORS[label], lw=2)

    ax.set_title(label, fontweight="bold", fontsize=10, pad=10)
    ax.set_xlabel("Lag (s)")
    ax.set_ylim(kernel_ylim)
    ax.grid(True, alpha=0.3)

    ax.text(0.02, 0.98, f"r = {corr:.3f}\nt = {elapsed:.0f} ms",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

axes[0].set_ylabel("TRF amplitude")
axes[0].legend(loc="upper left", fontsize=8, framealpha=0.8)

fig1.suptitle("TRF Kernel Recovery by Solver", fontweight="bold", y=1.02, fontsize=14)
fig1.tight_layout()

# ---------------------------------------------------------------------------
# Figure 2: Speed vs Accuracy (side-by-side horizontal bars)
# ---------------------------------------------------------------------------
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

labels = [r[0] for r in results]
times = [r[2] for r in results]
corrs = [r[3] for r in results]
colors = [SOLVER_COLORS[l] for l in labels]

ax1.barh(labels, times, color=colors)
ax1.set_xscale("log")
ax1.set_xlabel("Fit time (ms, log scale)")
ax1.set_title("Speed", fontweight="bold")
ax1.grid(True, alpha=0.3, axis="x")
ax1.invert_yaxis()

ax2.barh(labels, corrs, color=colors)
ax2.set_xlabel("Correlation with ground truth")
ax2.set_xlim(0, 1)
ax2.set_title("Accuracy", fontweight="bold")
ax2.grid(True, alpha=0.3, axis="x")
ax2.invert_yaxis()

fig2.suptitle("Solver Performance: Speed vs Accuracy",
              fontweight="bold", y=1.02, fontsize=14)
fig2.tight_layout()

plt.show()
