"""Flask application for interactive TRF exploration."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)
ALLOWED_EXTENSIONS = {"npz", "npy"}
MAX_FILE_SIZE = 30 * 1024 * 1024
SOLVERS = ["default", "robust", "CG"]
REGULARIZATION_TYPES = ["none", "ridge", "smoothness"]


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _load_array(path: Path) -> np.ndarray:
    loaded = np.load(path, allow_pickle=False)
    try:
        if isinstance(loaded, np.lib.npyio.NpzFile):
            if not loaded.files:
                raise ValueError("The NPZ archive does not contain an array")
            return np.asarray(loaded[loaded.files[0]])
        return np.asarray(loaded)
    finally:
        if isinstance(loaded, np.lib.npyio.NpzFile):
            loaded.close()


def _normalise_array(array: np.ndarray, file_type: str) -> np.ndarray:
    """Normalise common single-channel and channel-first array layouts."""
    array = np.asarray(array)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim != 2:
        raise ValueError(f"{file_type} must be a 1-D or 2-D array.")
    # A channel-first response convention is recognisable when channels are
    # fewer than samples. X is intentionally left in its documented layout.
    if file_type == "Y" and array.shape[0] < array.shape[1]:
        return array.T
    return array


def _file_metadata(path: Path, file_type: str) -> dict:
    array = _normalise_array(_load_array(path), file_type)
    return {"filename": path.name, "type": file_type, "shape": list(array.shape),
            "dtype": str(array.dtype), "size": path.stat().st_size}


def create_app(upload_folder: str | os.PathLike[str] | None = None) -> Flask:
    """Create the dashboard application."""
    app = Flask(__name__, template_folder="templates", static_folder="static")
    folder = Path(upload_folder) if upload_folder else Path(tempfile.mkdtemp(prefix="pyeeg_dashboard_"))
    app.config.update(UPLOAD_FOLDER=folder, MAX_CONTENT_LENGTH=MAX_FILE_SIZE)
    folder.mkdir(parents=True, exist_ok=True)

    @app.errorhandler(413)
    def request_too_large(_error):
        return jsonify(error="File is too large. The maximum size is 30 MB."), 413

    @app.route("/")
    def index():
        return render_template("index.html", solvers=SOLVERS, regularization_types=REGULARIZATION_TYPES)

    @app.route("/upload", methods=["POST"])
    def upload_file():
        file, file_type = request.files.get("file"), request.form.get("type", "").upper()
        if file is None or not file.filename:
            return jsonify(error="Choose a file to upload."), 400
        if file_type not in {"X", "Y"}:
            return jsonify(error="File type must be X or Y."), 400
        if not allowed_file(file.filename):
            return jsonify(error="Only .npy and .npz files are supported."), 400
        filename = secure_filename(file.filename)
        if not filename:
            return jsonify(error="The filename is not valid."), 400
        target_dir, path = folder / file_type, folder / file_type / filename
        target_dir.mkdir(exist_ok=True)
        file.save(path)
        try:
            metadata = _file_metadata(path, file_type)
        except (ValueError, OSError, EOFError) as exc:
            path.unlink(missing_ok=True)
            return jsonify(error=f"Could not read NumPy file: {exc}"), 400
        return jsonify(success=True, file_info=metadata)

    @app.route("/list_files")
    def list_files():
        files = []
        for file_type in ("X", "Y"):
            directory = folder / file_type
            if directory.exists():
                for path in sorted(directory.iterdir()):
                    if path.is_file():
                        try:
                            files.append(_file_metadata(path, file_type))
                        except (ValueError, OSError, EOFError):
                            logger.warning("Skipping unreadable upload %s", path)
        return jsonify(files=files)

    @app.route("/compute_trf", methods=["POST"])
    def compute_trf():
        payload = request.get_json(silent=True) or {}
        x_name, y_name = payload.get("x_file"), payload.get("y_file")
        if not x_name or not y_name:
            return jsonify(error="Upload both predictor (X) and response (Y) data."), 400
        x_path, y_path = folder / "X" / Path(x_name).name, folder / "Y" / Path(y_name).name
        if not x_path.is_file() or not y_path.is_file():
            return jsonify(error="One or both selected files are no longer available."), 404
        try:
            x_array = _normalise_array(_load_array(x_path), "X")
            y_array = _normalise_array(_load_array(y_path), "Y")
            if x_array.shape[0] != y_array.shape[0]:
                raise ValueError(f"X and Y must have the same samples after normalisation ({x_array.shape[0]} vs {y_array.shape[0]}).")
            fs = float(payload.get("fs") or 1.0)
            tmin, tmax = float(payload.get("tmin", -0.2)), float(payload.get("tmax", 0.5))
            alpha = float(payload.get("regularization", 1.0))
            solver, reg_type = payload.get("solver", "default"), payload.get("regularization_type", "ridge")
            if solver not in SOLVERS or reg_type not in REGULARIZATION_TYPES or fs <= 0 or tmin >= tmax or alpha < 0:
                raise ValueError("Check solver, regularisation type, sampling frequency, lag range, and alpha.")
            from pyeeg.models.trf import TRFEstimator
            kwargs = {"tmin": tmin, "tmax": tmax, "srate": fs, "alpha": 0 if reg_type == "none" else alpha, "verbose": False}
            if reg_type == "smoothness":
                kwargs["quadratic_reg"] = "smoothness"
            if solver in {"robust", "CG"}:
                kwargs["loss"] = "cauchy"
                kwargs["robust_inner_solver"] = "cg" if solver == "CG" else "svd"
            import time
            model = TRFEstimator(**kwargs)
            started = time.perf_counter()
            model.fit(x_array.astype(float), y_array.astype(float))
            fit_seconds = time.perf_counter() - started
            coef = np.asarray(model.coef_)  # n_lag × n_features × n_channels
            return jsonify(success=True, result={"coef": coef[:, 0, :].tolist(), "time": model.times.tolist(),
                "coef_shape": list(coef.shape), "x_shape": list(x_array.shape), "y_shape": list(y_array.shape),
                "solver": solver, "regularization_type": reg_type, "regularization": alpha, "fs": fs,
                "tmin": tmin, "tmax": tmax, "fit_seconds": fit_seconds})
        except (ValueError, TypeError, FloatingPointError, MemoryError) as exc:
            return jsonify(error=str(exc)), 400
        except Exception:
            logger.exception("TRF computation failed")
            return jsonify(error="TRF computation failed. Check the server log for details."), 500

    @app.route("/clear_uploads", methods=["POST"])
    def clear_uploads():
        for file_type in ("X", "Y"):
            shutil.rmtree(folder / file_type, ignore_errors=True)
        return jsonify(success=True)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
