// pyEEG Dashboard JavaScript

// Global state
const uploadedFiles = {
    X: [],
    Y: []
};

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    loadFiles();
    
    // Update slider value display
    const regSlider = document.getElementById('regularization');
    if (regSlider) {
        document.getElementById('regValue').textContent = regSlider.value;
        regSlider.addEventListener('input', function() {
            document.getElementById('regValue').textContent = this.value;
        });
    }
});

// File upload handlers
function handleFileSelect(event, fileType) {
    const file = event.target.files[0];
    if (file) {
        uploadFile(file, fileType);
    }
}

function handleDrop(event, fileType) {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file) {
        uploadFile(file, fileType);
    }
}

function uploadFile(file, fileType) {
    // Validate file size
    const maxSize = 30 * 1024 * 1024; // 30MB
    if (file.size > maxSize) {
        showStatus(`File too large (${(file.size / 1024 / 1024).toFixed(2)}MB). Max is 30MB`, 'error');
        return;
    }
    
    // Validate file extension
    const validExtensions = ['.npz', '.npy'];
    const fileName = file.name.toLowerCase();
    const isValid = validExtensions.some(ext => fileName.endsWith(ext));
    
    if (!isValid) {
        showStatus('Only .npz and .npy files are allowed', 'error');
        return;
    }
    
    // Show loading state
    showStatus(`Uploading ${file.name}...`, 'success');
    
    // Create form data
    const formData = new FormData();
    formData.append('file', file);
    formData.append('type', fileType);
    
    // Upload file
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            const fileInfo = data.file_info;
            uploadedFiles[fileType].push(fileInfo);
            updateFileInfo(fileType, fileInfo);
            updateFileList(fileType);
            showStatus(`Successfully uploaded ${file.name}`, 'success');
        } else {
            showStatus(data.error || 'Upload failed', 'error');
        }
    })
    .catch(error => {
        showStatus(`Upload error: ${error.message}`, 'error');
    });
}

function updateFileInfo(fileType, fileInfo) {
    const infoDiv = document.getElementById(`fileInfo${fileType}`);
    const filenameDiv = document.getElementById(`${fileType.toLowerCase()}Filename`);
    const shapeDiv = document.getElementById(`${fileType.toLowerCase()}Shape`);
    const sizeDiv = document.getElementById(`${fileType.toLowerCase()}Size`);
    
    if (filenameDiv && shapeDiv && sizeDiv) {
        filenameDiv.textContent = `File: ${fileInfo.filename}`;
        shapeDiv.textContent = `Shape: ${fileInfo.shape}`;
        sizeDiv.textContent = `Size: ${(fileInfo.size / 1024 / 1024).toFixed(2)} MB`;
        infoDiv.classList.add('active');
    }
}

function updateFileList(fileType) {
    const fileListDiv = document.getElementById(`${fileType.toLowerCase()}FileList`);
    if (fileListDiv) {
        fileListDiv.innerHTML = '';
        
        uploadedFiles[fileType].forEach((file, index) => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            fileItem.innerHTML = `
                <div class="info">
                    <strong>${file.filename}</strong> - ${file.shape}
                </div>
                <span class="remove" onclick="removeFile('${fileType}', ${index})">✕</span>
            `;
            fileListDiv.appendChild(fileItem);
        });
    }
}

function removeFile(fileType, index) {
    uploadedFiles[fileType].splice(index, 1);
    updateFileList(fileType);
    
    if (uploadedFiles[fileType].length === 0) {
        const infoDiv = document.getElementById(`fileInfo${fileType}`);
        if (infoDiv) {
            infoDiv.classList.remove('active');
        }
    }
    
    // Notify server to delete file
    fetch('/clear_uploads', {
        method: 'POST'
    }).catch(() => {});
}

function loadFiles() {
    fetch('/list_files')
    .then(response => response.json())
    .then(data => {
        uploadedFiles.X = [];
        uploadedFiles.Y = [];
        data.files.forEach(file => {
            uploadedFiles[file.type].push(file);
        });
        
        // Update UI
        if (uploadedFiles.X.length > 0) {
            const lastX = uploadedFiles.X[uploadedFiles.X.length - 1];
            updateFileInfo('X', lastX);
        }
        if (uploadedFiles.Y.length > 0) {
            const lastY = uploadedFiles.Y[uploadedFiles.Y.length - 1];
            updateFileInfo('Y', lastY);
        }
        
        updateFileList('X');
        updateFileList('Y');
    })
    .catch(error => {
        console.error('Error loading files:', error);
    });
}

function clearAllFiles() {
    fetch('/clear_uploads', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            uploadedFiles.X = [];
            uploadedFiles.Y = [];
            const xInfoDiv = document.getElementById('fileInfoX');
            const yInfoDiv = document.getElementById('fileInfoY');
            const xFileList = document.getElementById('xFileList');
            const yFileList = document.getElementById('yFileList');
            
            if (xInfoDiv) xInfoDiv.classList.remove('active');
            if (yInfoDiv) yInfoDiv.classList.remove('active');
            if (xFileList) xFileList.innerHTML = '';
            if (yFileList) yFileList.innerHTML = '';
            
            const plotContainer = document.getElementById('plotContainer');
            if (plotContainer) {
                plotContainer.innerHTML = '<p style="color: #999;">Upload data and compute TRF to see results</p>';
            }
            
            const resultInfo = document.getElementById('resultInfo');
            if (resultInfo) {
                resultInfo.style.display = 'none';
            }
            
            showStatus('All files cleared', 'success');
        } else {
            showStatus(data.error || 'Failed to clear files', 'error');
        }
    })
    .catch(error => {
        showStatus(`Error clearing files: ${error.message}`, 'error');
    });
}

// TRF computation
function computeTRF() {
    const xFile = uploadedFiles.X.length > 0 ? uploadedFiles.X[0].filename : null;
    const yFile = uploadedFiles.Y.length > 0 ? uploadedFiles.Y[0].filename : null;
    
    if (!xFile || !yFile) {
        showStatus('Please upload both X and Y data files', 'error');
        return;
    }
    
    // Show loading
    const loadingDiv = document.getElementById('loading');
    const statusDiv = document.getElementById('status');
    if (loadingDiv) loadingDiv.classList.add('active');
    if (statusDiv) statusDiv.style.display = 'none';
    
    // Get parameters
    const solver = document.getElementById('solver').value;
    const regularization = parseFloat(document.getElementById('regularization').value);
    const fs = document.getElementById('fs').value ? parseFloat(document.getElementById('fs').value) : null;
    const xaxis = document.querySelector('input[name="xaxis"]:checked').value;
    
    // Send computation request
    fetch('/compute_trf', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            x_file: xFile,
            y_file: yFile,
            solver: solver,
            regularization: regularization,
            fs: fs,
            xaxis: xaxis
        })
    })
    .then(response => response.json())
    .then(data => {
        if (loadingDiv) loadingDiv.classList.remove('active');
        
        if (data.success) {
            const result = data.result;
            displayResults(result);
            showStatus('TRF computation successful!', 'success');
        } else {
            showStatus(data.error || 'TRF computation failed', 'error');
        }
    })
    .catch(error => {
        if (loadingDiv) loadingDiv.classList.remove('active');
        showStatus(`Computation error: ${error.message}`, 'error');
    });
}

function displayResults(result) {
    // Update result info
    const resultSolver = document.getElementById('resultSolver');
    const resultRegularization = document.getElementById('resultRegularization');
    const resultFs = document.getElementById('resultFs');
    const resultInfo = document.getElementById('resultInfo');
    
    if (resultSolver && resultRegularization && resultFs && resultInfo) {
        resultSolver.textContent = `Solver: ${result.solver}`;
        resultRegularization.textContent = `Regularization: ${result.regularization}`;
        resultFs.textContent = `Fs: ${result.fs ? result.fs + ' Hz' : 'Not specified (samples)'}`;
        resultInfo.style.display = 'block';
    }
    
    // Create plot
    const plotContainer = document.getElementById('plotContainer');
    if (plotContainer) {
        const canvas = document.createElement('canvas');
        canvas.id = 'trfPlot';
        canvas.width = 800;
        canvas.height = 400;
        
        plotContainer.innerHTML = '';
        plotContainer.appendChild(canvas);
        
        // Simple line plot using canvas
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        const padding = 50;
        
        // Clear canvas
        ctx.fillStyle = '#f8f9fa';
        ctx.fillRect(0, 0, width, height);
        
        // Draw axes
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 2;
        
        // X axis
        ctx.beginPath();
        ctx.moveTo(padding, height - padding);
        ctx.lineTo(width - padding, height - padding);
        ctx.stroke();
        
        // Y axis
        ctx.beginPath();
        ctx.moveTo(padding, padding);
        ctx.lineTo(padding, height - padding);
        ctx.stroke();
        
        // Draw grid
        ctx.strokeStyle = '#e9ecef';
        ctx.lineWidth = 1;
        
        // Horizontal grid lines
        for (let i = 1; i < 5; i++) {
            const y = height - padding - (i * (height - 2 * padding) / 5);
            ctx.beginPath();
            ctx.moveTo(padding, y);
            ctx.lineTo(width - padding, y);
            ctx.stroke();
        }
        
        // Draw TRF line
        const trf = result.trf;
        const time = result.time;
        const plotWidth = width - 2 * padding;
        const plotHeight = height - 2 * padding;
        
        ctx.strokeStyle = '#667eea';
        ctx.lineWidth = 3;
        ctx.beginPath();
        
        for (let i = 0; i < trf.length; i++) {
            const x = padding + (i / (trf.length - 1)) * plotWidth;
            const y = padding + plotHeight - ((trf[i] - Math.min(...trf)) / (Math.max(...trf) - Math.min(...trf))) * plotHeight;
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();
        
        // Draw labels
        ctx.fillStyle = '#333';
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        
        // X axis label
        const xaxisType = document.querySelector('input[name="xaxis"]:checked').value;
        ctx.fillText(xaxisType === 'seconds' ? 'Time (s)' : 'Samples', width / 2, height - 20);
        
        // Y axis label
        ctx.save();
        ctx.translate(20, height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('TRF Amplitude', 0, 0);
        ctx.restore();
        
        // Title
        ctx.font = '16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Temporal Response Function', width / 2, 30);
    }
}

function showStatus(message, type) {
    const statusDiv = document.getElementById('status');
    if (statusDiv) {
        statusDiv.textContent = message;
        statusDiv.className = `status ${type}`;
    }
}

// Export functions for potential module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        handleFileSelect,
        handleDrop,
        uploadFile,
        computeTRF,
        clearAllFiles,
        loadFiles
    };
}
