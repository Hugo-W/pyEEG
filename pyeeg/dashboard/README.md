# pyEEG Dashboard

A web-based dashboard for interactive TRF (Temporal Response Function) analysis.

## Overview

This dashboard provides a simple and elegant interface for:
- Uploading EEG/MEG data (X) and feature data (Y) as numpy arrays (.npz format)
- Visualizing TRF results
- Selecting solvers
- Adjusting regularization parameters via sliders
- Specifying sampling frequency (Fs) for proper time axis scaling

## Requirements

- Python 3.10+
- Flask (web framework)
- NumPy (for data handling)
- Optional dependencies: see `pyproject.toml` under `[project.optional-dependencies.exploratory-trf]`

## Installation

Install the dashboard with optional dependencies:

```bash
pip install -e ".[exploratory-trf]"
```

## Usage

Start the dashboard server:

```bash
python -m pyeeg.dashboard.server
```

Or use the entry point:

```bash
pyeeg-dashboard
```

The dashboard will be available at `http://localhost:5000` by default.

## File Constraints

- Maximum file size: ~30MB
- Expected format: numpy arrays saved as .npz files
- Typical dimensions: 300 sensors, 20 minutes at 50Hz (float32)

## Features

- **Data Upload**: Drag and drop areas for X (EEG/MEG) and Y (features) data
- **TRF Visualization**: Interactive display of temporal response functions
- **Solver Selection**: Choose from available TRF solvers
- **Regularization Control**: Slider to adjust regularization parameters
- **Sampling Frequency**: Input field for Fs to enable time axis in seconds

## Architecture

The dashboard consists of:
- Frontend: HTML/CSS/JavaScript for the web interface
- Backend: Flask server handling data processing and TRF computation
- Data Flow: Client uploads -> Server processing -> Results to client

## Future Enhancements

See TODO.md for planned features and improvements.
