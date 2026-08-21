"""
Flask application for the pyEEG dashboard.
"""

import os
import tempfile
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'npz', 'npy'}

# Maximum file size (30MB)
MAX_FILE_SIZE = 30 * 1024 * 1024

# Supported solvers (to be populated from pyEEG solvers)
SOLVERS = ['ridge', 'lasso', 'elasticnet']

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__, 
                template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
                static_folder=os.path.join(os.path.dirname(__file__), 'static'))
    
    # Configure upload folder
    app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp(prefix='pyeeg_dashboard_')
    app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
    
    # Ensure upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    
    @app.route('/')
    def index():
        """Render the main dashboard page."""
        return render_template('index.html', solvers=SOLVERS)
    
    @app.route('/upload', methods=['POST'])
    def upload_file():
        """Handle file uploads for X (EEG/MEG) and Y (features)."""
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        file_type = request.form.get('type')  # 'X' or 'Y'
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            
            # Create type-specific subdirectory
            type_dir = os.path.join(app.config['UPLOAD_FOLDER'], file_type)
            os.makedirs(type_dir, exist_ok=True)
            
            filepath = os.path.join(type_dir, filename)
            file.save(filepath)
            
            # Load and validate the file
            try:
                data = np.load(filepath)
                if isinstance(data, np.lib.npyio.NpzFile):
                    # Handle .npz files with multiple arrays
                    keys = list(data.keys())
                    if len(keys) == 1:
                        array_data = data[keys[0]]
                    else:
                        # For multiple arrays, use the first one
                        array_data = data[keys[0]]
                        logger.warning(f"NPZ file has multiple arrays, using first: {keys[0]}")
                else:
                    # Handle .npy files
                    array_data = data
                
                # Check file size constraint
                file_size = os.path.getsize(filepath)
                if file_size > MAX_FILE_SIZE:
                    os.remove(filepath)
                    return jsonify({'error': f'File too large ({file_size} bytes). Max is {MAX_FILE_SIZE} bytes'}), 400
                
                # Store file info
                file_info = {
                    'filename': filename,
                    'type': file_type,
                    'shape': array_data.shape,
                    'dtype': str(array_data.dtype),
                    'size': file_size,
                    'filepath': filepath
                }
                
                logger.info(f"Uploaded {file_type} file: {filename}, shape: {array_data.shape}")
                return jsonify({'success': True, 'file_info': file_info})
                
            except Exception as e:
                logger.error(f"Error loading file: {e}")
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({'error': f'Invalid numpy file: {str(e)}'}), 400
        else:
            return jsonify({'error': 'File type not allowed'}), 400
    
    @app.route('/compute_trf', methods=['POST'])
    def compute_trf():
        """Compute TRF from uploaded X and Y data."""
        try:
            data = request.get_json()
            
            # Get parameters
            x_file = data.get('x_file')
            y_file = data.get('y_file')
            solver = data.get('solver', 'ridge')
            regularization = data.get('regularization', 1.0)
            fs = data.get('fs', None)
            
            if not x_file or not y_file:
                return jsonify({'error': 'Both X and Y files are required'}), 400
            
            # Load data
            x_path = os.path.join(app.config['UPLOAD_FOLDER'], 'X', x_file)
            y_path = os.path.join(app.config['UPLOAD_FOLDER'], 'Y', y_file)
            
            if not os.path.exists(x_path) or not os.path.exists(y_path):
                return jsonify({'error': 'File not found on server'}), 404
            
            x_data = np.load(x_path)
            y_data = np.load(y_path)
            
            # Handle .npz files
            if isinstance(x_data, np.lib.npyio.NpzFile):
                x_array = x_data[x_data.files[0]] if len(x_data.files) > 0 else None
            else:
                x_array = x_data
                
            if isinstance(y_data, np.lib.npyio.NpzFile):
                y_array = y_data[y_data.files[0]] if len(y_data.files) > 0 else None
            else:
                y_array = y_data
            
            if x_array is None or y_array is None:
                return jsonify({'error': 'Could not load data arrays'}), 400
            
            # Simple TRF computation (placeholder - to be replaced with actual pyEEG solvers)
            # For now, return mock results
            trf_result = {
                'trf': np.random.randn(100).tolist(),  # Mock TRF
                'time': np.linspace(0, 1, 100).tolist() if fs else list(range(100)),
                'x_shape': x_array.shape,
                'y_shape': y_array.shape,
                'solver': solver,
                'regularization': regularization,
                'fs': fs
            }
            
            return jsonify({'success': True, 'result': trf_result})
            
        except Exception as e:
            logger.error(f"Error computing TRF: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/files/<path:filename>')
    def uploaded_file(filename):
        """Serve uploaded files."""
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    
    @app.route('/clear_uploads', methods=['POST'])
    def clear_uploads():
        """Clear all uploaded files."""
        try:
            import shutil
            shutil.rmtree(app.config['UPLOAD_FOLDER'])
            app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp(prefix='pyeeg_dashboard_')
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            return jsonify({'success': True})
        except Exception as e:
            logger.error(f"Error clearing uploads: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/list_files')
    def list_files():
        """List all uploaded files."""
        files = []
        
        for file_type in ['X', 'Y']:
            type_dir = os.path.join(app.config['UPLOAD_FOLDER'], file_type)
            if os.path.exists(type_dir):
                for filename in os.listdir(type_dir):
                    filepath = os.path.join(type_dir, filename)
                    if os.path.isfile(filepath):
                        try:
                            data = np.load(filepath)
                            if isinstance(data, np.lib.npyio.NpzFile):
                                keys = list(data.keys())
                                shape = data[keys[0]].shape if keys else None
                            else:
                                shape = data.shape
                            
                            files.append({
                                'type': file_type,
                                'filename': filename,
                                'shape': str(shape) if shape else 'unknown',
                                'size': os.path.getsize(filepath)
                            })
                        except:
                            files.append({
                                'type': file_type,
                                'filename': filename,
                                'shape': 'unknown',
                                'size': os.path.getsize(filepath)
                            })
        
        return jsonify({'files': files})
    
    return app


# Create a default app instance for convenience
app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
