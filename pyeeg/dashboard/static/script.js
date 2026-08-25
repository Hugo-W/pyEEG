const uploadedFiles = {X: [], Y: []};
let previousResult = null;
const maxSize = 30 * 1024 * 1024;

document.addEventListener('DOMContentLoaded', () => {
  ['X', 'Y'].forEach(type => {
    const zone = document.getElementById(`dropZone${type}`), input = document.getElementById(`fileInput${type}`);
    zone.addEventListener('click', () => input.click());
    zone.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') input.click(); });
    zone.addEventListener('dragover', event => { event.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', event => { event.preventDefault(); zone.classList.remove('drag-over'); uploadFile(event.dataTransfer.files[0], type); });
    input.addEventListener('change', event => uploadFile(event.target.files[0], type));
  });
  const slider = document.getElementById('regularization');
  const updateAlpha = () => document.getElementById('regValue').value = (10 ** Number(slider.value)).toFixed(4);
  slider.addEventListener('input', updateAlpha); updateAlpha();
  loadFiles();
});

async function uploadFile(file, type) {
  if (!file) return;
  if (file.size > maxSize) return showStatus('That file is larger than 30 MB.', 'error');
  if (!/\.(npy|npz)$/i.test(file.name)) return showStatus('Only .npy and .npz files are supported.', 'error');
  const body = new FormData(); body.append('file', file); body.append('type', type);
  showStatus(`Uploading ${file.name}…`);
  try {
    const response = await fetch('/upload', {method: 'POST', body}); const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Upload failed');
    uploadedFiles[type] = [data.file_info]; updateFileList(type); showStatus(`${file.name} is ready.`);
  } catch (error) { showStatus(error.message, 'error'); }
}
function updateFileList(type) {
  const list = document.getElementById(`${type.toLowerCase()}FileList`); list.innerHTML = '';
  uploadedFiles[type].forEach(file => { const item = document.createElement('div'); item.className = 'file-item'; item.innerHTML = `<span><strong>${escapeHtml(file.filename)}</strong> · ${file.shape.join(' × ')}</span><span class="remove" title="Remove">×</span>`; item.querySelector('.remove').onclick = () => { uploadedFiles[type] = []; updateFileList(type); }; list.appendChild(item); });
}
async function loadFiles() { try { const data = await (await fetch('/list_files')).json(); uploadedFiles.X = data.files.filter(f => f.type === 'X'); uploadedFiles.Y = data.files.filter(f => f.type === 'Y'); updateFileList('X'); updateFileList('Y'); } catch (_) {} }
async function clearAllFiles() { await fetch('/clear_uploads', {method: 'POST'}); uploadedFiles.X = []; uploadedFiles.Y = []; updateFileList('X'); updateFileList('Y'); document.getElementById('resultInfo').hidden = true; document.getElementById('plotContainer').innerHTML = '<div class="empty-state"><span>⌁</span><strong>Your TRF will appear here</strong><p>Upload both arrays and run the model to reveal its temporal profile.</p></div>'; showStatus('Session reset.'); }
async function computeTRF() {
  if (!uploadedFiles.X.length || !uploadedFiles.Y.length) return showStatus('Upload both predictor and response arrays first.', 'error');
  const button = document.getElementById('computeButton'); button.disabled = true; button.textContent = 'Computing…'; showStatus('Fitting model…');
  const payload = {x_file: uploadedFiles.X[0].filename, y_file: uploadedFiles.Y[0].filename, solver: document.getElementById('solver').value, regularization: 10 ** Number(document.getElementById('regularization').value), regularization_type: document.getElementById('regularizationType').value, fs: Number(document.getElementById('fs').value), tmin: Number(document.getElementById('tmin').value), tmax: Number(document.getElementById('tmax').value)};
  try { const response = await fetch('/compute_trf', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)}); const data = await response.json(); if (!response.ok) throw new Error(data.error); displayResults(data.result); showStatus('TRF computed successfully.'); } catch (error) { showStatus(error.message || 'Computation failed.', 'error'); } finally { button.disabled = false; button.innerHTML = 'Compute TRF <span>→</span>'; }
}
function displayResults(result) {
  const oldResult = previousResult; previousResult = result;
  document.getElementById('resultInfo').hidden = false; document.getElementById('resultSolver').textContent = `solver: ${result.solver}`; document.getElementById('resultRegularization').textContent = `α: ${result.regularization}`; document.getElementById('resultFs').textContent = `fs: ${result.fs} Hz`; document.getElementById('resultShape').textContent = `coef: ${result.coef_shape.join(' × ')}`; const fitTime = document.createElement('span'); fitTime.textContent = `fit: ${result.fit_seconds.toFixed(3)} s`; document.getElementById('resultInfo').appendChild(fitTime);
  const canvas = document.createElement('canvas'); canvas.width = 1000; canvas.height = 400; document.getElementById('plotContainer').replaceChildren(canvas); const ctx = canvas.getContext('2d'), pad = 52, values = result.coef, oldValues = oldResult?.coef, all = values.flat().concat(oldValues?.flat() || []), min = Math.min(...all), max = Math.max(...all), span = max - min || 1, width = canvas.width - pad * 2, height = canvas.height - pad * 2;
  ctx.clearRect(0,0,canvas.width,canvas.height); ctx.strokeStyle='#dce5df'; ctx.lineWidth=1; ctx.globalAlpha=1; for(let i=0;i<5;i++){const y=pad+i*height/4;ctx.beginPath();ctx.moveTo(pad,y);ctx.lineTo(canvas.width-pad,y);ctx.stroke()} if(result.fs && result.tmin < 0 && result.tmax > 0){const zeroX=pad+(-result.tmin)/(result.tmax-result.tmin)*width;ctx.strokeStyle='#ed7d62';ctx.setLineDash([5,5]);ctx.beginPath();ctx.moveTo(zeroX,pad);ctx.lineTo(zeroX,pad+height);ctx.stroke();ctx.setLineDash([])} for(let i=0;i<5;i++){const y=pad+i*height/4;ctx.beginPath();ctx.moveTo(pad,y);ctx.lineTo(canvas.width-pad,y);ctx.stroke()} const draw = (data, color, opacity) => { ctx.strokeStyle=color; ctx.globalAlpha=opacity; ctx.lineWidth=opacity > .3 ? 2.1 : 1.1; data[0].forEach((_, channel) => { ctx.beginPath(); data.forEach((row, i) => { const x=pad+i*width/(data.length-1||1), y=pad+height-(row[channel]-min)*height/span; i ? ctx.lineTo(x,y) : ctx.moveTo(x,y); }); ctx.stroke(); }); }; if (oldValues) draw(oldValues, '#71817b', .22); draw(values, '#1f6d59', .58); ctx.globalAlpha=1; ctx.fillStyle='#71817b';ctx.font='12px DM Mono';ctx.fillText(String(result.tmin)+' s',pad,canvas.height-18);ctx.textAlign='right';ctx.fillText(String(result.tmax)+' s',canvas.width-pad,canvas.height-18);
}
function showStatus(message, type='') { const status = document.getElementById('status'); status.textContent = message; status.className = `status ${type}`; }
function escapeHtml(value) { const div = document.createElement('div'); div.textContent = value; return div.innerHTML; }
if (typeof module !== 'undefined') module.exports = {uploadFile, computeTRF, clearAllFiles, loadFiles};
