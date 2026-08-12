const statusText = document.getElementById('status-text');
const responseBox = document.getElementById('response-box');
const hud = document.getElementById('vision-hud');
const muteBtn = document.getElementById('mute-btn');
const muteIcon = document.getElementById('mute-icon');

const codePanel = document.getElementById('code-panel');
const codeContent = document.getElementById('code-content');
const codeLang = document.getElementById('code-lang');
const codeCopyBtn = document.getElementById('code-copy-btn');

const searchPanel = document.getElementById('search-panel');
const searchQueryLabel = document.getElementById('search-query-label');
const searchResultsEl = document.getElementById('search-results');
const searchImagesEl = document.getElementById('search-images');

const execPanel = document.getElementById('exec-panel');
const execCode = document.getElementById('exec-code');
const execOutput = document.getElementById('exec-output');

const pageOverlay = document.getElementById('page-render-overlay');
const pageFrame = document.getElementById('page-render-frame');

const screenIndicator = document.getElementById('screen-indicator');
const cameraIndicator = document.getElementById('camera-indicator');

const allPanels = [codePanel, searchPanel, execPanel];

function closeAllPanels() {
  allPanels.forEach(p => p.classList.remove('visible'));
  hud.classList.remove('code-open');
}

function closeVisualPanel() {
  closeAllPanels();
  pageOverlay.classList.remove('visible');
  pageFrame.srcdoc = '';
  hud.classList.remove('page-open');
}

function setStatus(state) {
  statusText.textContent = state;
  if (window.setOrbState) window.setOrbState(state);
}

function showResponse(text) {
  responseBox.textContent = text;
  responseBox.classList.add('visible');
}

function clearResponse() {
  responseBox.classList.remove('visible');
}

function showCode(code, language) {
  closeVisualPanel();
  codeContent.textContent = code;
  codeLang.textContent = language || 'code';
  codePanel.classList.add('visible');
  hud.classList.add('code-open');
}

function showSearchResults(query, results, images) {
  closeVisualPanel();
  searchQueryLabel.textContent = query;
  searchResultsEl.innerHTML = '';
  searchImagesEl.innerHTML = '';

  (images || []).slice(0, 6).forEach(imgUrl => {
    const img = document.createElement('img');
    img.src = imgUrl;
    img.loading = 'lazy';
    img.onerror = () => img.remove();
    searchImagesEl.appendChild(img);
  });

  results.forEach(r => {
    const item = document.createElement('div');
    item.className = 'search-result-item';

    const title = document.createElement('div');
    title.className = 'search-result-title';
    title.textContent = r.title || '';

    const snippet = document.createElement('div');
    snippet.className = 'search-result-snippet';
    snippet.textContent = (r.content || '').slice(0, 200);

    const url = document.createElement('div');
    url.className = 'search-result-url';
    url.textContent = r.url || '';

    item.appendChild(title);
    item.appendChild(snippet);
    item.appendChild(url);
    searchResultsEl.appendChild(item);
  });

  searchPanel.classList.add('visible');
  hud.classList.add('code-open');
}

function showExecutionResult(code, stdout, stderr, success) {
  closeVisualPanel();
  execCode.textContent = code;
  execOutput.textContent = success ? (stdout || '(no output)') : (stderr || 'Error');
  execOutput.className = 'exec-output ' + (success ? 'success' : 'error');

  execPanel.classList.add('visible');
  hud.classList.add('code-open');
}

function renderWebpage(html) {
  closeVisualPanel();
  pageFrame.srcdoc = html;
  pageOverlay.classList.add('visible');
  hud.classList.add('page-open');
}

function setScreenActive(active) {
  screenIndicator.classList.toggle('visible', active);
}

function setCameraActive(active) {
  cameraIndicator.classList.toggle('visible', active);
}

window.updateStatus = setStatus;
window.updateResponse = showResponse;
window.showCode = showCode;
window.showSearchResults = showSearchResults;
window.showExecutionResult = showExecutionResult;
window.renderWebpage = renderWebpage;
window.closeVisualPanel = closeVisualPanel;
window.setScreenActive = setScreenActive;
window.setCameraActive = setCameraActive;

let isMuted = false;

muteBtn.addEventListener('click', async () => {
  isMuted = !isMuted;
  muteBtn.classList.toggle('muted', isMuted);
  muteIcon.textContent = isMuted ? '✕' : '●';
  muteBtn.title = isMuted ? 'Unmute VISION' : 'Mute VISION';

  // Optimistic UI feedback — the backend also sets this on its own status
  // transitions, but this makes the button feel instant rather than
  // waiting for the next natural state change.
  setStatus(isMuted ? 'muted' : 'listening');

  if (window.pywebview && window.pywebview.api && window.pywebview.api.toggle_mute) {
    try {
      await window.pywebview.api.toggle_mute(isMuted);
    } catch (e) {
      console.error('toggle_mute failed:', e);
    }
  }
});

codeCopyBtn.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(codeContent.textContent);
    codeCopyBtn.textContent = 'Copied!';
    codeCopyBtn.classList.add('copied');
    setTimeout(() => {
      codeCopyBtn.textContent = 'Copy';
      codeCopyBtn.classList.remove('copied');
    }, 1500);
  } catch (e) {
    console.error('Copy failed:', e);
  }
});