# natmeeg TRF Explorer

The dashboard is a local web application for interactively fitting and
inspecting Temporal Response Functions (TRFs) with `pyeeg.TRFEstimator`.
It is intended for exploratory analysis, not as a production deployment.

## Install

This project is managed with `uv`. Install the dashboard dependencies from the
repository root:

```bash
uv sync --extra exploratory-trf
```

The extra installs Flask, Werkzeug, and Gunicorn. NumPy, SciPy, and
scikit-learn are core project dependencies.

## Run

From the repository root (or this package's worktree):

```bash
uv run trf-explore
```

The equivalent module invocation is:

```bash
uv run --extra exploratory-trf python -m pyeeg.dashboard.server
```

Open <http://localhost:5000>. The server accepts the following options:

```bash
uv run trf-explore --help
uv run trf-explore --host 127.0.0.1 --port 5000 --debug
```

The `trf-explore` console script is declared in `pyproject.toml` and points to
`pyeeg.dashboard.server:main`.

## Workflow

1. Upload a predictor array (**X**) and a response array (**Y**).
2. Select the regularisation type, solver, sampling frequency, alpha, and lag
   window.
3. Select **Compute TRF**.
4. Inspect the channel-wise coefficient traces in the central plot.

Supported regularisation types are **None**, **Ridge**, and **Smoothness**.
Supported solver choices are **Default**, **Robust**, and **CG**. Alpha is
controlled on a base-10 logarithmic scale from `0.0001` to `10000`.

The plot displays one line per response channel. The previous fit remains as a
faded grey overlay when a new fit is computed, while the newest fit is shown in
green. A vertical line marks zero seconds when the selected lag window crosses
zero.

## Input format

Uploads must be `.npy` or `.npz` files no larger than 30 MB. The first array in
an NPZ archive is used. Arrays are converted to floating-point values before
fitting.

- **X** is expected as `(n_samples, n_features)`; a one-dimensional X is
  reshaped to `(n_samples, 1)`.
- **Y** is expected as `(n_samples, n_channels)`; a one-dimensional Y is
  reshaped to `(n_samples, 1)`.
- A two-dimensional Y supplied as `(n_channels, n_samples)` is transposed when
  its first dimension is smaller than its second.

X and Y must have the same number of samples after normalization. Uploaded files
are stored in a temporary server-side session directory and are cleared with
**Reset session**.

## Development checks

Run the test suite and formatting-independent repository checks with:

```bash
uv run --with pytest pytest tests
uv run python -m compileall -q pyeeg/dashboard
git diff --check
```

The dashboard currently has no browser automation suite; manual verification in
a browser is still recommended after frontend changes.
