/* VISION HUD — LiveKit-style agent shell.
 *
 * The Python side (voice/vision_live.py) drives everything through the
 * window.* functions at the bottom of this file. The older calls
 * (updateStatus / updateResponse / showCode / showSearchResults /
 * showExecutionResult / renderWebpage / closeVisualPanel /
 * setScreenActive / setCameraActive) all still work; the newer ones
 * (updateTranscript / setAudioLevel / setToolActivity) are what make the
 * transcript stream and the visualizer move with real audio. */

const el = (id) => document.getElementById(id);

const body          = document.body;
const statePill     = el('state-pill');
const stateLabel    = el('state-label');
const agentHint     = el('agent-hint');
const visualizer    = el('visualizer');
const transcriptEl  = el('transcript');
const feed          = el('transcript-feed');
const emptyState    = el('transcript-empty');
const micBtn        = el('mic-btn');
const micLabel      = el('mic-label');
const micMeterFill  = el('mic-meter-fill');
const chipScreen    = el('chip-screen');
const chipCamera    = el('chip-camera');
const chipTool      = el('chip-tool');
const chipToolName  = el('chip-tool-name');
const clearBtn      = el('clear-btn');
const overlay       = el('page-overlay');
const overlayClose  = el('overlay-close');
const pageFrame     = el('page-frame');

/* ------------------------------------------------------------------ */
/* State                                                               */
/* ------------------------------------------------------------------ */

const STATES = {
  idle:            { label: 'idle',        hint: 'Say “Vision” to start' },
  listening:       { label: 'listening',   hint: 'Listening…' },
  processing:      { label: 'thinking',    hint: 'Working on it…' },
  speaking:        { label: 'speaking',    hint: '' },
  muted:           { label: 'muted',       hint: 'Mic is off — unmute to talk' },
  reconnecting:    { label: 'reconnecting',hint: 'Lost the connection, retrying…' },
  mic_unavailable: { label: 'no mic',      hint: 'No microphone available' },
};

let currentState = 'idle';

function setStatus(state) {
  const info = STATES[state] || { label: state, hint: '' };
  currentState = state;
  body.dataset.state = state;
  stateLabel.textContent = info.label;
  statePill.title = state;
  agentHint.textContent = info.hint;

  const muted = state === 'muted';
  micBtn.classList.toggle('muted', muted);
  micBtn.setAttribute('aria-pressed', String(muted));
  micLabel.textContent = muted ? 'Mic off' : 'Mic on';
  micBtn.title = muted ? 'Unmute VISION' : 'Mute VISION';
  isMuted = muted;
}

/* ------------------------------------------------------------------ */
/* Visualizer                                                          */
/* ------------------------------------------------------------------ */

const BAR_COUNT = 7;
const BARS = [];
// Centre bars run taller than the edges, the way a level meter reads.
const WEIGHTS = [0.45, 0.68, 0.88, 1.0, 0.88, 0.68, 0.45];
const LEVEL_STALE_MS = 220;   // no fresh chunk for this long -> decay to 0

for (let i = 0; i < BAR_COUNT; i++) {
  const bar = document.createElement('div');
  bar.className = 'bar';
  visualizer.appendChild(bar);
  BARS.push({ node: bar, value: 0.08 });
}

const levels = { user: 0, agent: 0 };
const levelSeenAt = { user: 0, agent: 0 };

function setAudioLevel(source, level) {
  if (!(source in levels)) return;
  levels[source] = Math.max(0, Math.min(1, Number(level) || 0));
  levelSeenAt[source] = performance.now();
}

function activeLevel(now) {
  const source = currentState === 'speaking' ? 'agent' : 'user';
  if (currentState === 'muted' || currentState === 'mic_unavailable') return 0;
  if (now - levelSeenAt[source] > LEVEL_STALE_MS) return 0;
  return levels[source];
}

function render(now) {
  const t = now / 1000;
  const level = activeLevel(now);
  const thinking = currentState === 'processing' || currentState === 'reconnecting';

  for (let i = 0; i < BAR_COUNT; i++) {
    let target;

    if (thinking) {
      // A pulse that travels across the bars: motion without pretending
      // there's audio to show.
      const phase = t * 3.0 - i * 0.5;
      target = 0.14 + 0.62 * Math.pow(Math.max(0, Math.sin(phase)), 4);
    } else if (level > 0.01) {
      const wobble = 0.86 + 0.14 * Math.sin(t * 7 + i * 1.9);
      target = 0.08 + level * WEIGHTS[i] * wobble * 1.25;
    } else {
      // Idle breathing.
      target = 0.07 + 0.035 * (0.5 + 0.5 * Math.sin(t * 1.5 + i * 0.5)) * WEIGHTS[i];
    }

    target = Math.max(0.05, Math.min(1, target));

    const b = BARS[i];
    // Rise fast, fall slow — reads as a real level meter rather than a jitter.
    const ease = target > b.value ? 0.45 : 0.14;
    b.value += (target - b.value) * ease;
    b.node.style.transform = 'scaleY(' + b.value.toFixed(3) + ')';
  }

  const micPct = (currentState === 'muted' ? 0 : Math.min(1, levels.user *
    (now - levelSeenAt.user > LEVEL_STALE_MS ? 0 : 1.4))) * 100;
  micMeterFill.style.width = micPct.toFixed(1) + '%';

  requestAnimationFrame(render);
}

requestAnimationFrame(render);

/* ------------------------------------------------------------------ */
/* Transcript                                                          */
/* ------------------------------------------------------------------ */

const ROLE_LABEL = { user: 'You', agent: 'VISION' };
const streaming = { user: null, agent: null };
let lastFinal = { user: '', agent: '' };

function nearBottom() {
  return transcriptEl.scrollHeight - transcriptEl.scrollTop - transcriptEl.clientHeight < 90;
}

function scrollToEnd(force) {
  if (force || nearBottom()) transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

function appendEntry(node) {
  emptyState.classList.add('hidden');
  const stick = nearBottom();
  feed.appendChild(node);
  scrollToEnd(stick);
  return node;
}

function makeTurn(role) {
  const turn = document.createElement('div');
  turn.className = 'turn ' + role + ' streaming';

  const label = document.createElement('div');
  label.className = 'turn-role';
  label.textContent = ROLE_LABEL[role] || role;

  const text = document.createElement('div');
  text.className = 'turn-text';

  turn.appendChild(label);
  turn.appendChild(text);
  turn._text = text;
  return turn;
}

function updateTranscript(role, text, final) {
  if (role !== 'user' && role !== 'agent') return;
  const value = (text || '').trim();
  if (!value) return;

  // The agent's own final line arrives twice on some paths (streamed, then
  // once more via the legacy updateResponse) — don't echo it.
  if (final && lastFinal[role] === value && !streaming[role]) return;

  let turn = streaming[role];
  if (!turn) {
    turn = appendEntry(makeTurn(role));
    streaming[role] = turn;
  }

  turn._text.textContent = value;

  if (final) {
    turn.classList.remove('streaming');
    streaming[role] = null;
    lastFinal[role] = value;
  }
  scrollToEnd(false);
}

/* Legacy: the final assistant line. */
function showResponse(text) {
  updateTranscript('agent', text, true);
}

/* ------------------------------------------------------------------ */
/* Tool cards (inline in the transcript, in the order they happened)    */
/* ------------------------------------------------------------------ */

function makeCard(title, actions) {
  const card = document.createElement('div');
  card.className = 'card';

  const head = document.createElement('div');
  head.className = 'card-head';

  const label = document.createElement('span');
  label.textContent = title;
  head.appendChild(label);
  if (actions) head.appendChild(actions);

  const bodyEl = document.createElement('div');
  bodyEl.className = 'card-body';

  card.appendChild(head);
  card.appendChild(bodyEl);
  card._body = bodyEl;
  return card;
}

function copyButton(getText) {
  const btn = document.createElement('button');
  btn.className = 'copy-btn';
  btn.type = 'button';
  btn.textContent = 'Copy';
  btn.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(getText());
      btn.textContent = 'Copied';
      btn.classList.add('copied');
      setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 1500);
    } catch (e) {
      console.error('Copy failed:', e);
    }
  });
  return btn;
}

function showCode(code, language) {
  const card = makeCard(language || 'code', copyButton(() => code));
  const pre = document.createElement('pre');
  pre.className = 'card-code';
  pre.textContent = code;
  card._body.appendChild(pre);
  appendEntry(card);
}

function showSearchResults(query, results, images) {
  const card = makeCard('search · ' + (query || ''));

  const strip = document.createElement('div');
  strip.className = 'search-images';
  (images || []).slice(0, 6).forEach((url) => {
    const img = document.createElement('img');
    img.src = url;
    img.loading = 'lazy';
    img.onerror = () => img.remove();
    strip.appendChild(img);
  });
  card._body.appendChild(strip);

  (results || []).forEach((r) => {
    const item = document.createElement('div');
    item.className = 'result';

    const title = document.createElement('div');
    title.className = 'result-title';
    title.textContent = r.title || '';

    const snippet = document.createElement('div');
    snippet.className = 'result-snippet';
    snippet.textContent = (r.content || '').slice(0, 200);

    const url = document.createElement('div');
    url.className = 'result-url';
    url.textContent = r.url || '';

    item.appendChild(title);
    item.appendChild(snippet);
    item.appendChild(url);
    card._body.appendChild(item);
  });

  appendEntry(card);
}

function showExecutionResult(code, stdout, stderr, success) {
  const card = makeCard(success ? 'ran python' : 'python failed', copyButton(() => code));

  const pre = document.createElement('pre');
  pre.className = 'card-code';
  pre.textContent = code;

  const out = document.createElement('div');
  out.className = 'card-out ' + (success ? 'success' : 'error');
  out.textContent = success ? (stdout || '(no output)') : (stderr || 'Error');

  card._body.appendChild(pre);
  card._body.appendChild(out);
  appendEntry(card);
}

/* ------------------------------------------------------------------ */
/* Indicators                                                          */
/* ------------------------------------------------------------------ */

function setScreenActive(active) { chipScreen.classList.toggle('active', !!active); }
function setCameraActive(active) { chipCamera.classList.toggle('active', !!active); }

function setToolActivity(name, active) {
  chipToolName.textContent = name || 'tool';
  chipTool.classList.toggle('active', !!active);
}

/* ------------------------------------------------------------------ */
/* Rendered-page overlay                                               */
/* ------------------------------------------------------------------ */

function renderWebpage(html) {
  pageFrame.srcdoc = html;
  overlay.classList.add('visible');
}

function closeVisualPanel() {
  overlay.classList.remove('visible');
  pageFrame.srcdoc = '';
}

overlayClose.addEventListener('click', closeVisualPanel);
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && overlay.classList.contains('visible')) closeVisualPanel();
});

/* ------------------------------------------------------------------ */
/* Controls                                                            */
/* ------------------------------------------------------------------ */

let isMuted = false;

micBtn.addEventListener('click', async () => {
  isMuted = !isMuted;
  // Optimistic — the backend echoes the real state back through updateStatus.
  setStatus(isMuted ? 'muted' : 'listening');

  if (window.pywebview && window.pywebview.api && window.pywebview.api.toggle_mute) {
    try {
      await window.pywebview.api.toggle_mute(isMuted);
    } catch (e) {
      console.error('toggle_mute failed:', e);
    }
  }
});

clearBtn.addEventListener('click', () => {
  feed.innerHTML = '';
  streaming.user = null;
  streaming.agent = null;
  lastFinal = { user: '', agent: '' };
  emptyState.classList.remove('hidden');
});

/* ------------------------------------------------------------------ */
/* Python bridge                                                       */
/* ------------------------------------------------------------------ */

window.updateStatus        = setStatus;
window.updateResponse      = showResponse;
window.updateTranscript    = updateTranscript;
window.setAudioLevel       = setAudioLevel;
window.setToolActivity     = setToolActivity;
window.showCode            = showCode;
window.showSearchResults   = showSearchResults;
window.showExecutionResult = showExecutionResult;
window.renderWebpage       = renderWebpage;
window.closeVisualPanel    = closeVisualPanel;
window.setScreenActive     = setScreenActive;
window.setCameraActive     = setCameraActive;

setStatus('idle');
