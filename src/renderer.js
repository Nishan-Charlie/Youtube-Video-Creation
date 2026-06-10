let BACKEND_URL = 'http://127.0.0.1:5847';
let currentAudioFilename = null;
let audioElement = null;
let audioContext = null;
let analyser = null;
let scriptSource = 'editor'; // 'editor' | 'custom'
let animFrame = null;

// ── Video editor state ─────────────────────────────────────────
let videoSequence = [];   // [{id, type, title, src, thumb, duration, text, style, subtitle, enabled}]
let exportJobId = null;
let exportPollInterval = null;

// ── Backend status ────────────────────────────────────────────
window.electron.onBackendStatus((data) => {
  const dot = document.getElementById('backendDot');
  const label = document.getElementById('backendLabel');
  const status = document.getElementById('statusMessage');

  if (data.status === 'ready') {
    BACKEND_URL = data.url || BACKEND_URL;
    dot.className = 'indicator-dot ready';
    label.textContent = 'Backend ready';
    status.innerHTML = `<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg> Ready`;
    enableControls(true);
    loadSavedKey();
    fetchDeviceInfo();
    loadTTSModels();
    _loadHfToken();          // load HF token once backend is reachable
    const savedKey = localStorage.getItem('gemini_api_key');
    if (savedKey) fetchModels(savedKey);
  } else if (data.status === 'error') {
    dot.className = 'indicator-dot error';
    label.textContent = 'Backend error';
    status.textContent = 'Backend failed — check setup.bat';
    toast('Backend failed to start. Run setup.bat first.', 'error');
  } else {
    dot.className = 'indicator-dot starting';
    label.textContent = 'Starting…';
  }
});

function enableControls(enabled) {
  document.getElementById('generateScriptBtn').disabled = !enabled;
  document.getElementById('generateAudioBtn').disabled = !enabled;
}

// ── Persist API key ────────────────────────────────────────────
function loadSavedKey() {
  const saved = localStorage.getItem('gemini_api_key');
  if (saved) document.getElementById('apiKey').value = saved;
}

document.getElementById('apiKey').addEventListener('change', (e) => {
  localStorage.setItem('gemini_api_key', e.target.value);
  if (e.target.value.trim()) fetchModels(e.target.value.trim());
});

document.getElementById('toggleKey').addEventListener('click', () => {
  const input = document.getElementById('apiKey');
  const icon = document.getElementById('eyeIcon');
  if (input.type === 'password') {
    input.type = 'text';
    icon.innerHTML = '<path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/><line x1="1" y1="1" x2="23" y2="23"/>';
  } else {
    input.type = 'password';
    icon.innerHTML = '<path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="3"/>';
  }
});

// ── Tabs ─────────────────────────────────────────────────────
window.switchTab = function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.getElementById(`tab-${name}`).classList.add('active');
  document.getElementById(`panel-${name}`).classList.add('active');
}

// ── Word / read-time counter ───────────────────────────────────
const scriptEditor = document.getElementById('scriptEditor');
scriptEditor.addEventListener('input', updateStats);

function updateStats() {
  const text = scriptEditor.value.trim();
  const words = text ? text.split(/\s+/).length : 0;
  const mins = Math.ceil(words / 140);
  document.getElementById('wordCount').textContent = words.toLocaleString();
  document.getElementById('readTime').textContent = mins;
}

// ── Range sliders ──────────────────────────────────────────────
function bindRange(id, valId, chipId) {
  const input = document.getElementById(id);
  const val = document.getElementById(valId);
  const chip = document.getElementById(chipId);
  const update = () => {
    const v = parseFloat(input.value).toFixed(2);
    val.textContent = v;
    if (chip) chip.textContent = v;
  };
  input.addEventListener('input', update);
  update();
}

bindRange('exaggeration', 'exaggerationVal', 'exaggerationChip');
bindRange('cfgWeight', 'cfgVal', 'cfgChip');

// ── Script provider switching ─────────────────────────────────
const providerPanels = {
  gemini:        document.getElementById('providerGemini'),
  qwen_dashscope:document.getElementById('providerQwen'),
  ollama:        document.getElementById('providerOllama'),
  lmstudio:      document.getElementById('providerLMStudio'),
};

const scriptProviderSel = document.getElementById('scriptProvider');
if (scriptProviderSel) {
  scriptProviderSel.addEventListener('change', () => {
    const p = scriptProviderSel.value;
    Object.entries(providerPanels).forEach(([id, el]) => {
      if (el) el.style.display = id === p ? 'block' : 'none';
    });
    if (p === 'ollama') checkOllamaStatus();
  });
}

async function checkOllamaStatus() {
  const el = document.getElementById('ollamaStatus');
  const sel = document.getElementById('ollamaModel');
  if (!el) return;
  try {
    const res = await fetch(`${BACKEND_URL}/script-providers`);
    const data = await res.json();
    const ollama = data.providers.find(p => p.id === 'ollama');
    if (ollama && ollama.available) {
      el.innerHTML = `<span style="color:var(--success)">Ollama running — ${ollama.ollama_models.length} model(s)</span>`;
      if (sel && ollama.ollama_models.length) {
        sel.innerHTML = ollama.ollama_models.map(m =>
          `<option value="${m}">${m}</option>`).join('');
      }
    } else {
      el.innerHTML = `<span style="color:var(--warning)">Ollama not running. <a href="https://ollama.com" style="color:var(--accent2)">Install</a> then run: ollama pull qwen2.5:7b</span>`;
    }
  } catch { el.textContent = 'Could not check Ollama status'; }
}

// ── TTS model selector ────────────────────────────────────────
const ttsModelSel = document.getElementById('ttsModel');
const voiceSel = document.getElementById('voiceSelect');
const modelDesc = document.getElementById('modelDesc');
const voiceSelectGroup = document.getElementById('voiceSelectGroup');

let _ttsRegistry = {};

async function loadTTSModels() {
  try {
    const res = await fetch(`${BACKEND_URL}/tts-models`);
    const data = await res.json();
    _ttsRegistry = {};
    data.models.forEach(m => { _ttsRegistry[m.id] = m; });
    updateVoiceList(ttsModelSel.value);
  } catch { /* keep defaults */ }
}

function updateVoiceList(modelId) {
  const m = _ttsRegistry[modelId];
  if (!m) return;

  // Update description
  if (modelDesc) {
    let badge = m.installed ? '' : ' <span style="color:var(--warning)">(not installed)</span>';
    modelDesc.innerHTML = `${m.description} — <b>${m.vram} VRAM</b>${badge}`;
  }

  // Update voice dropdown
  const voices = m.voices || {};
  voiceSel.innerHTML = '';
  Object.entries(voices).forEach(([label, val]) => {
    const opt = document.createElement('option');
    opt.value = val;
    opt.textContent = label;
    voiceSel.appendChild(opt);
  });

  // Hide voice selector for clone-only models (single option)
  const hasChoices = Object.keys(voices).length > 1;
  if (voiceSelectGroup) voiceSelectGroup.style.display = hasChoices ? 'block' : 'none';

  // Hide exaggeration/cfg sliders for cloud/light models that don't support them
  const hideAdv = modelId === 'kokoro' || modelId === 'gemini_tts';
  const advPanel = document.getElementById('advancedVoicePanel');
  if (advPanel) advPanel.style.display = hideAdv ? 'none' : 'block';

  // Show API key reminder for Gemini TTS
  const geminiNote = document.getElementById('geminiTtsNote');
  if (modelId === 'gemini_tts') {
    if (!geminiNote) {
      const note = document.createElement('div');
      note.id = 'geminiTtsNote';
      note.style.cssText = 'font-size:10px;color:var(--text3);margin-top:4px';
      note.textContent = 'Uses your Gemini API key (enter in the sidebar) — no GPU needed.';
      document.getElementById('modelDesc')?.after(note);
    }
  } else {
    geminiNote?.remove();
  }
}

if (ttsModelSel) {
  ttsModelSel.addEventListener('change', () => updateVoiceList(ttsModelSel.value));
}

// ── Speed slider ───────────────────────────────────────────────
const speedSlider = document.getElementById('speedSlider');
const speedChip = document.getElementById('speedChip');
if (speedSlider) {
  speedSlider.addEventListener('input', () => {
    const v = parseFloat(speedSlider.value).toFixed(2);
    if (speedChip) speedChip.textContent = v + 'x';
  });
}

// ── Voice style presets ────────────────────────────────────────
document.querySelectorAll('.preset-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const exag = parseFloat(btn.dataset.exag);
    const cfg = parseFloat(btn.dataset.cfg);
    const exagSlider = document.getElementById('exaggeration');
    const cfgSlider = document.getElementById('cfgWeight');
    if (exagSlider) { exagSlider.value = exag; exagSlider.dispatchEvent(new Event('input')); }
    if (cfgSlider)  { cfgSlider.value  = cfg;  cfgSlider.dispatchEvent(new Event('input'));  }
  });
});

// ── Clear / copy script ─────────────────────────────────────────
document.getElementById('clearScriptBtn').addEventListener('click', () => {
  if (scriptEditor.value && confirm('Clear the script?')) {
    scriptEditor.value = '';
    updateStats();
  }
});

document.getElementById('copyScriptBtn').addEventListener('click', () => {
  if (!scriptEditor.value) return toast('Nothing to copy', 'info');
  navigator.clipboard.writeText(scriptEditor.value);
  toast('Script copied to clipboard', 'success');
});

// ── Voice reference browse ──────────────────────────────────────
document.getElementById('browseVoiceRef').addEventListener('click', async () => {
  // Use native file picker via backend
  try {
    const res = await fetch(`${BACKEND_URL}/pick-file`, { method: 'POST' });
    const data = await res.json();
    if (data.path) document.getElementById('voiceRef').value = data.path;
  } catch {
    toast('Could not open file picker', 'error');
  }
});

document.getElementById('clearVoiceRef').addEventListener('click', () => {
  document.getElementById('voiceRef').value = '';
});

function _getScriptModel() {
  const provider = document.getElementById('scriptProvider')?.value || 'gemini';
  if (provider === 'qwen_dashscope') return document.getElementById('qwenModel')?.value || 'qwen-plus';
  if (provider === 'ollama') return document.getElementById('ollamaModel')?.value || 'qwen2.5:7b';
  if (provider === 'lmstudio') return 'local-model';
  return document.getElementById('geminiModel')?.value || 'gemini-2.0-flash';
}

function _getScriptApiKey() {
  const provider = document.getElementById('scriptProvider')?.value || 'gemini';
  if (provider === 'qwen_dashscope') return document.getElementById('qwenApiKey')?.value?.trim() || '';
  if (provider === 'ollama' || provider === 'lmstudio') return 'ollama'; // no real key needed
  return document.getElementById('apiKey')?.value?.trim() || '';
}

// ── Generate Script ─────────────────────────────────────────────
document.getElementById('generateScriptBtn').addEventListener('click', generateScript);

async function generateScript() {
  const provider = document.getElementById('scriptProvider')?.value || 'gemini';
  const apiKey = _getScriptApiKey();
  const topic = document.getElementById('topic').value.trim();

  // Only require API key for cloud providers
  if (provider === 'gemini' && !apiKey) return toast('Enter your Gemini API key first', 'error');
  if (provider === 'qwen_dashscope' && !apiKey) return toast('Enter your DashScope API key', 'error');
  if (!topic) return toast('Enter a video topic', 'error');

  showProgress('Generating Script with Gemini…', 'Connecting to Gemini 2.0 Flash', [
    { label: 'Connecting to Gemini API', active: true },
    { label: 'Generating script outline' },
    { label: 'Writing full script' }
  ]);

  try {
    setProgressStep(0, 'done');
    setProgressStep(1, 'active');

    const res = await fetch(`${BACKEND_URL}/generate-script`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        topic,
        niche:    document.getElementById('niche').value,
        duration: document.getElementById('duration').value,
        tone:     document.getElementById('tone').value,
        context:  document.getElementById('context').value,
        provider: provider,
        api_key:  apiKey,
        model:    _getScriptModel(),
      })
    });

    setProgressStep(1, 'done');
    setProgressStep(2, 'active');

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || 'Script generation failed');
    }

    const data = await res.json();
    setProgressStep(2, 'done');

    scriptEditor.value = data.script;
    updateStats();
    hideProgress();
    switchTab('script');
    toast('Script generated!', 'success');
    if (provider === 'gemini') localStorage.setItem('gemini_api_key', apiKey);

  } catch (err) {
    hideProgress();
    const msg = err.message || '';
    if (msg.includes('quota') || msg.includes('RESOURCE_EXHAUSTED') || msg.includes('429')) {
      toast('Quota exhausted — try "Flash Lite" model or wait until tomorrow for daily reset', 'error');
    } else {
      toast(msg || 'Script generation failed', 'error');
    }
  }
}

// ── Script source toggle ───────────────────────────────────────
window.setScriptSource = function setScriptSource(src) {
  scriptSource = src;
  document.getElementById('srcEditorBtn').classList.toggle('active', src === 'editor');
  document.getElementById('srcCustomBtn').classList.toggle('active', src === 'custom');
  document.getElementById('customInputArea').style.display = src === 'custom' ? 'block' : 'none';
}

function getVoiceoverText() {
  if (scriptSource === 'custom') {
    return document.getElementById('customScriptInput').value.trim();
  }
  return scriptEditor.value.trim();
}

const customInput = document.getElementById('customScriptInput');
if (customInput) {
  customInput.addEventListener('input', () => {
    const text = customInput.value.trim();
    const words = text ? text.split(/\s+/).length : 0;
    document.getElementById('customWordCount').textContent = words.toLocaleString();
    document.getElementById('customReadTime').textContent = Math.ceil(words / 140);
  });
}

const clearCustomBtn = document.getElementById('clearCustomBtn');
if (clearCustomBtn && customInput) {
  clearCustomBtn.addEventListener('click', () => {
    customInput.value = '';
    customInput.dispatchEvent(new Event('input'));
  });
}

// ── Generate Audio ─────────────────────────────────────────────
document.getElementById('generateAudioBtn').addEventListener('click', generateAudio);

async function generateAudio() {
  const script = getVoiceoverText();
  const label = scriptSource === 'custom' ? 'Paste a script in the text box first' : 'Write or generate a script first';
  if (!script) return toast(label, 'error');

  const ttsModel = document.getElementById('ttsModel')?.value || 'chatterbox';
  if (ttsModel === 'gemini_tts') {
    const key = document.getElementById('apiKey')?.value?.trim() || '';
    if (!key) return toast('Enter your Gemini API key in the sidebar to use Gemini TTS', 'error');
  }

  const chunkSize = document.getElementById('chunkSize').value;
  const exaggeration = parseFloat(document.getElementById('exaggeration').value);
  const cfgWeight = parseFloat(document.getElementById('cfgWeight').value);
  const voiceRef = document.getElementById('voiceRef').value.trim();

  // Estimate chunks for display
  const estChunks = chunkSize === 'full' ? 1
    : chunkSize === 'paragraph' ? (script.split(/\n{2,}/).filter(Boolean).length || 1)
    : script.split(/(?<=[.!?])\s+/).length;

  showAudioProgress('Starting voiceover generation…', estChunks);

  try {
    const res = await fetch(`${BACKEND_URL}/generate-audio`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: script,
        topic:          document.getElementById('topic')?.value?.trim() || '',
        tts_model:      document.getElementById('ttsModel')?.value || 'chatterbox',
        voice:          document.getElementById('voiceSelect')?.value || 'default',
        exaggeration,
        cfg_weight:     cfgWeight,
        voice_ref:      voiceRef || null,
        chunk_size:     chunkSize,
        speed:          parseFloat(document.getElementById('speedSlider')?.value || '1.0'),
        gemini_api_key: document.getElementById('apiKey')?.value?.trim() || '',
      })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || 'Failed to start generation');
    }

    const { job_id } = await res.json();
    const filename = await pollAudioJob(job_id);

    hideProgress();
    currentAudioFilename = filename;
    loadAudio(`${BACKEND_URL}/audio/${filename}`, filename);
    updateVoiceoverIndicator();
    switchTab('audio');
    toast('Voiceover ready!', 'success');

  } catch (err) {
    hideProgress();
    toast(err.message, 'error');
  }
}

function showAudioProgress(msg, estChunks) {
  document.getElementById('progressTitle').textContent = 'Generating Voiceover…';
  document.getElementById('progressSub').textContent = msg;

  const stepsEl = document.getElementById('progressSteps');
  stepsEl.innerHTML = `
    <div style="margin-top:12px">
      <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:6px">
        <span id="chunkLabel" style="color:var(--text2)">Loading model…</span>
        <span id="chunkPct" style="color:var(--text3)">0%</span>
      </div>
      <div style="height:6px;background:var(--bg4);border-radius:4px;overflow:hidden">
        <div id="audioProgressBar" style="height:100%;width:0%;background:var(--accent2);border-radius:4px;transition:width 0.4s ease"></div>
      </div>
      <div style="font-size:11px;color:var(--text3);margin-top:8px">
        ${estChunks > 1 ? `~${estChunks} chunks to synthesize` : 'Synthesizing full script'}
        &nbsp;·&nbsp; RTX 4070 CUDA
      </div>
    </div>`;

  // Override polling to also update the chunk label + pct
  const origPoll = window._audioProgressInterval;
  if (origPoll) clearInterval(origPoll);

  document.getElementById('progressOverlay').classList.remove('hidden');
}

async function pollAudioJob(jobId) {
  return new Promise((resolve, reject) => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/audio-status/${jobId}`);
        const job = await res.json();

        const chunk = job.chunk || 0;
        const total = job.total || 0;
        const msg = job.msg || '';
        const pct = total > 0 ? Math.min(99, Math.round((chunk / total) * 100)) : 0;

        const bar = document.getElementById('audioProgressBar');
        const lbl = document.getElementById('chunkLabel');
        const pctEl = document.getElementById('chunkPct');
        const sub = document.getElementById('progressSub');

        if (bar) bar.style.width = `${pct}%`;
        if (lbl) lbl.textContent = msg || 'Processing…';
        if (pctEl) pctEl.textContent = `${pct}%`;
        if (sub) sub.textContent = msg;

        if (job.status === 'done') {
          if (bar) bar.style.width = '100%';
          if (pctEl) pctEl.textContent = '100%';
          if (lbl) lbl.textContent = 'Complete!';
          clearInterval(interval);
          setTimeout(() => resolve(job.filename), 300);
        } else if (job.status === 'error') {
          clearInterval(interval);
          reject(new Error(job.error || 'Audio generation failed'));
        }
      } catch (err) {
        clearInterval(interval);
        reject(err);
      }
    }, 600);
  });
}

// ── Audio player ───────────────────────────────────────────────
function loadAudio(url, filename) {
  document.getElementById('playerIdle').style.display = 'none';
  document.getElementById('playerReady').style.display = 'block';
  document.getElementById('audioTitle').textContent = filename;

  if (audioElement) {
    audioElement.pause();
    cancelAnimationFrame(animFrame);
  }

  audioElement = new Audio(url);

  audioElement.addEventListener('timeupdate', updateProgress);
  audioElement.addEventListener('ended', () => {
    document.getElementById('playIcon').innerHTML = '<polygon points="5 3 19 12 5 21 5 3"/>';
    cancelAnimationFrame(animFrame);
  });
  audioElement.addEventListener('loadedmetadata', () => {
    updateProgress();
  });

  drawStaticWave();
}

function updateProgress() {
  if (!audioElement) return;
  const pct = audioElement.duration ? (audioElement.currentTime / audioElement.duration) * 100 : 0;
  document.getElementById('progressFill').style.width = `${pct}%`;
  document.getElementById('timeDisplay').textContent =
    `${fmt(audioElement.currentTime)} / ${fmt(audioElement.duration || 0)}`;
}

function fmt(s) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60).toString().padStart(2, '0');
  return `${m}:${sec}`;
}

document.getElementById('playBtn').addEventListener('click', () => {
  if (!audioElement) return;
  if (audioElement.paused) {
    audioElement.play();
    document.getElementById('playIcon').innerHTML = '<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>';
    animateWave();
  } else {
    audioElement.pause();
    document.getElementById('playIcon').innerHTML = '<polygon points="5 3 19 12 5 21 5 3"/>';
    cancelAnimationFrame(animFrame);
  }
});

document.getElementById('progressBar').addEventListener('click', (e) => {
  if (!audioElement || !audioElement.duration) return;
  const rect = e.currentTarget.getBoundingClientRect();
  const pct = (e.clientX - rect.left) / rect.width;
  audioElement.currentTime = pct * audioElement.duration;
});

function drawStaticWave() {
  const canvas = document.getElementById('waveform');
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.offsetWidth * devicePixelRatio;
  canvas.height = canvas.offsetHeight * devicePixelRatio;
  ctx.scale(devicePixelRatio, devicePixelRatio);

  const w = canvas.offsetWidth;
  const h = canvas.offsetHeight;
  ctx.clearRect(0, 0, w, h);

  const bars = 80;
  const gap = 2;
  const barW = (w - (bars - 1) * gap) / bars;

  ctx.fillStyle = '#2e2e2e';
  for (let i = 0; i < bars; i++) {
    const height = 4 + Math.random() * (h - 8);
    const x = i * (barW + gap);
    const y = (h - height) / 2;
    ctx.beginPath();
    ctx.roundRect(x, y, barW, height, 2);
    ctx.fill();
  }
}

function animateWave() {
  const canvas = document.getElementById('waveform');
  const ctx = canvas.getContext('2d');
  const w = canvas.offsetWidth;
  const h = canvas.offsetHeight;

  const bars = 80;
  const gap = 2;
  const barW = (w - (bars - 1) * gap) / bars;
  let t = 0;

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    t += 0.05;

    for (let i = 0; i < bars; i++) {
      const amp = 0.3 + 0.7 * Math.abs(Math.sin(i * 0.15 + t));
      const height = Math.max(4, amp * h * 0.85);
      const x = i * (barW + gap);
      const y = (h - height) / 2;
      const progress = audioElement ? audioElement.currentTime / (audioElement.duration || 1) : 0;

      ctx.fillStyle = (i / bars) < progress ? '#ff4500' : '#2e2e2e';
      ctx.beginPath();
      ctx.roundRect(x, y, barW, height, 2);
      ctx.fill();
    }

    animFrame = requestAnimationFrame(draw);
  }

  draw();
}

// ── Save audio ─────────────────────────────────────────────────
document.getElementById('saveAudioBtn').addEventListener('click', async () => {
  if (!currentAudioFilename) return;
  const saved = await window.electron.saveAudio(currentAudioFilename);
  if (saved) toast(`Saved to ${saved}`, 'success');
});

document.getElementById('openFolderBtn').addEventListener('click', () => {
  window.electron.openOutputFolder();
});

// ── Progress overlay helpers ────────────────────────────────────
let progressStepData = [];

function showProgress(title, sub, steps) {
  progressStepData = steps.map(s => ({ ...s }));
  document.getElementById('progressTitle').textContent = title;
  document.getElementById('progressSub').textContent = sub;

  const stepsEl = document.getElementById('progressSteps');
  stepsEl.innerHTML = progressStepData.map((s, i) =>
    `<div class="progress-step ${s.active ? 'active' : ''}" id="pstep-${i}">
      <div class="step-icon">${s.active ? '…' : i + 1}</div>
      ${s.label}
    </div>`
  ).join('');

  document.getElementById('progressOverlay').classList.remove('hidden');
}

function setProgressStep(index, state) {
  const el = document.getElementById(`pstep-${index}`);
  if (!el) return;
  el.className = `progress-step ${state}`;
  el.querySelector('.step-icon').textContent =
    state === 'done' ? '✓' : state === 'active' ? '…' : index + 1;
}

function hideProgress() {
  document.getElementById('progressOverlay').classList.add('hidden');
}

// ── Toast ──────────────────────────────────────────────────────
function toast(msg, type = 'info') {
  const icon = type === 'success' ? '✓' : type === 'error' ? '✕' : 'ℹ';
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.innerHTML = `<span>${icon}</span><span>${msg}</span>`;
  document.getElementById('toastContainer').appendChild(el);
  setTimeout(() => el.remove(), 4000);
}

// ── Fetch real Gemini models from API ──────────────────────────
async function fetchModels(apiKey) {
  if (!apiKey) return;
  try {
    const res = await fetch(`${BACKEND_URL}/list-models`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ api_key: apiKey })
    });
    if (!res.ok) return;
    const data = await res.json();
    if (!data.models || !data.models.length) return;

    const sel = document.getElementById('geminiModel');
    const current = sel.value;
    sel.innerHTML = '';
    data.models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.id;
      opt.textContent = m.name || m.id;
      if (m.id === current || (!sel.options.length)) opt.selected = true;
      sel.appendChild(opt);
    });
    // Restore saved preference or pick first
    if (data.models.find(m => m.id === current)) sel.value = current;
    console.log('[models] Loaded', data.models.length, 'models from API');
  } catch { /* silent — keep defaults */ }
}

// ── Device info ────────────────────────────────────────────────
async function fetchDeviceInfo() {
  try {
    const res = await fetch(`${BACKEND_URL}/device-info`);
    const d = await res.json();
    const el = document.getElementById('deviceStatus');
    if (!el) return;
    if (d.device === 'cuda') {
      const vram = (d.vram_mb / 1024).toFixed(1);
      el.innerHTML = `<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
        <span style="color:var(--success)">CUDA · ${d.name.replace('NVIDIA ', '')} · ${vram} GB</span>`;
    } else {
      el.innerHTML = `<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg> Chatterbox TTS · CPU`;
    }
  } catch { /* ignore */ }
}

// ── Window controls ────────────────────────────────────────────
document.getElementById('winMinimize').addEventListener('click', () => window.electron.minimizeWindow());
document.getElementById('winMaximize').addEventListener('click', () => window.electron.maximizeWindow());
document.getElementById('winClose').addEventListener('click', () => window.electron.closeWindow());

// ── Tabs via addEventListener (not just onclick) ────────────────
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    const name = tab.id.replace('tab-', '');
    window.switchTab(name);
  });
});

// ════════════════════════════════════════════════════════════════
//  VIDEO EDITOR
// ════════════════════════════════════════════════════════════════

// ── Render the clip sequence ─────────────────────────────────
function renderSequence() {
  const container = document.getElementById('clipSequence');
  const empty     = document.getElementById('clipSeqEmpty');
  if (!container) return;

  // Remove existing clip cards (keep empty placeholder)
  container.querySelectorAll('.clip-card').forEach(el => el.remove());

  if (videoSequence.length === 0) {
    if (empty) empty.style.display = 'flex';
    return;
  }
  if (empty) empty.style.display = 'none';

  videoSequence.forEach((clip, idx) => {
    const card = document.createElement('div');
    card.className = 'clip-card';
    card.dataset.id = clip.id;

    // Thumbnail (clickable to open preview modal)
    const thumbEl = document.createElement('div');
    thumbEl.className = 'clip-thumb';
    thumbEl.title = 'Click to preview';
    thumbEl.style.cursor = 'pointer';
    if (clip.thumb) {
      const img = document.createElement('img');
      img.src = clip.thumb;
      img.onerror = () => { thumbEl.textContent = clip.type === 'slide' ? '📝' : '🎬'; };
      thumbEl.appendChild(img);
    } else {
      thumbEl.textContent = clip.type === 'slide' ? '📝' : '🎬';
    }
    thumbEl.addEventListener('click', (e) => {
      e.stopPropagation();
      _previewClipCard(clip);
    });
    card.appendChild(thumbEl);

    // Info
    const info = document.createElement('div');
    info.className = 'clip-info';
    info.innerHTML = `
      <div class="clip-title">${clip.title || 'Untitled'}</div>
      <div class="clip-meta">
        <span class="clip-tag ${clip.type}">${clip.type}</span>
        &nbsp;
        <input class="clip-dur-input" type="number" value="${(clip.duration||5).toFixed(0)}"
               min="1" max="120" title="Duration (seconds)"
               data-idx="${idx}"/>s
      </div>`;
    card.appendChild(info);

    // Actions: up, down, delete
    const actions = document.createElement('div');
    actions.className = 'clip-actions';
    actions.innerHTML = `
      <button class="clip-btn" data-action="up" data-idx="${idx}" title="Move up">▲</button>
      <button class="clip-btn" data-action="down" data-idx="${idx}" title="Move down">▼</button>
      <button class="clip-btn danger" data-action="del" data-idx="${idx}" title="Remove">✕</button>`;
    card.appendChild(actions);

    container.appendChild(card);

    // Duration change
    card.querySelector('.clip-dur-input').addEventListener('change', e => {
      videoSequence[idx].duration = parseFloat(e.target.value) || 5;
    });

    // Action clicks
    actions.addEventListener('click', e => {
      const btn = e.target.closest('[data-action]');
      if (!btn) return;
      const i = parseInt(btn.dataset.idx);
      if (btn.dataset.action === 'up' && i > 0) {
        [videoSequence[i - 1], videoSequence[i]] = [videoSequence[i], videoSequence[i - 1]];
      } else if (btn.dataset.action === 'down' && i < videoSequence.length - 1) {
        [videoSequence[i], videoSequence[i + 1]] = [videoSequence[i + 1], videoSequence[i]];
      } else if (btn.dataset.action === 'del') {
        videoSequence.splice(i, 1);
      }
      renderSequence();
    });
  });
}

function addClip(clip) {
  if (!clip.id) clip.id = Math.random().toString(36).slice(2);
  videoSequence.push(clip);
  renderSequence();
  updateVoiceoverIndicator();
}

function updateVoiceoverIndicator() {
  const el = document.getElementById('voIndicator');
  const lbl = document.getElementById('voLabel');
  if (!el || !lbl) return;
  if (currentAudioFilename) {
    el.style.display = 'flex';
    lbl.textContent = `Voiceover: ${currentAudioFilename}`;
  } else {
    el.style.display = 'none';
  }
}

// ── Pexels search ─────────────────────────────────────────────
const pexelsSearchBtn = document.getElementById('pexelsSearchBtn');
if (pexelsSearchBtn) {
  pexelsSearchBtn.addEventListener('click', async () => {
    const query = document.getElementById('pexelsQuery')?.value?.trim();
    const key   = document.getElementById('pexelsKey')?.value?.trim();
    if (!query) return toast('Enter a search term', 'error');
    if (!key)   return toast('Enter your Pexels API key (free at pexels.com/api)', 'error');

    pexelsSearchBtn.disabled = true;
    pexelsSearchBtn.textContent = '…';
    const grid = document.getElementById('pexelsResults');
    grid.innerHTML = '<div style="font-size:11px;color:var(--text3);padding:8px">Searching…</div>';

    try {
      const res = await fetch(`${BACKEND_URL}/pexels-search`, {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ query, pexels_key: key, per_page: 6 })
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      renderPexelsResults(data.results || []);
    } catch (e) {
      grid.innerHTML = `<div style="font-size:11px;color:var(--error);padding:8px">${e.message}</div>`;
    } finally {
      pexelsSearchBtn.disabled = false;
      pexelsSearchBtn.textContent = 'Search';
    }
  });
}

function renderPexelsResults(results) {
  const grid = document.getElementById('pexelsResults');
  if (!grid) return;
  if (!results.length) { grid.innerHTML = '<div style="font-size:11px;color:var(--text3)">No results</div>'; return; }

  grid.innerHTML = '';
  results.forEach(item => {
    const el = document.createElement('div');
    el.className = 'pexels-item';
    el.innerHTML = `
      <img src="${item.thumb}" loading="lazy" alt="${item.photographer}"/>
      <div class="pex-overlay">＋</div>
      <div class="pex-dur">${item.duration}s</div>`;

    el.addEventListener('click', async () => {
      if (el.classList.contains('loading') || el.classList.contains('added')) return;
      el.classList.add('loading');
      el.querySelector('.pex-overlay').textContent = '⬇';
      try {
        const r = await fetch(`${BACKEND_URL}/download-clip`, {
          method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify({ url: item.url, id: String(item.id) })
        });
        const d = await r.json();
        if (d.error) throw new Error(d.error);
        addClip({
          type: 'video',
          title: `Pexels #${item.id} — ${item.photographer}`,
          src: d.path,
          thumb: item.thumb,
          duration: item.duration,
        });
        el.classList.remove('loading');
        el.classList.add('added');
        el.querySelector('.pex-overlay').textContent = '✓';
        toast('Clip added to sequence', 'success');
      } catch(e) {
        el.classList.remove('loading');
        toast(`Download failed: ${e.message}`, 'error');
      }
    });
    grid.appendChild(el);
  });
}

// ── Text slide creator ────────────────────────────────────────
const addSlideBtn = document.getElementById('addSlideBtn');
const slideCreator = document.getElementById('slideCreatorSection');
if (addSlideBtn && slideCreator) {
  addSlideBtn.addEventListener('click', () => {
    slideCreator.style.display = slideCreator.style.display === 'none' ? 'block' : 'none';
  });
}

const addSlideConfirmBtn = document.getElementById('addSlideConfirmBtn');
if (addSlideConfirmBtn) {
  addSlideConfirmBtn.addEventListener('click', async () => {
    const text = document.getElementById('slideText')?.value?.trim();
    if (!text) return toast('Enter slide text', 'error');

    addSlideConfirmBtn.disabled = true;
    try {
      const res = await fetch(`${BACKEND_URL}/generate-slide`, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({
          text,
          duration: parseFloat(document.getElementById('slideDuration')?.value || '5'),
          style:    document.getElementById('slideStyle')?.value || 'dark',
          subtitle: document.getElementById('slideSubtitle')?.value?.trim() || '',
        })
      });
      const d = await res.json();
      if (d.error) throw new Error(d.error);

      const fname = d.path.split(/[\\/]/).pop();
      addClip({
        type: 'slide',
        title: text.slice(0, 40),
        src: d.path,
        thumb: `${BACKEND_URL}/slide-preview/${fname}`,
        duration: parseFloat(document.getElementById('slideDuration')?.value || '5'),
        text, style: document.getElementById('slideStyle')?.value || 'dark',
        subtitle: document.getElementById('slideSubtitle')?.value?.trim() || '',
      });
      document.getElementById('slideText').value = '';
      if (slideCreator) slideCreator.style.display = 'none';
      toast('Slide added', 'success');
    } catch(e) { toast(e.message, 'error'); }
    finally { addSlideConfirmBtn.disabled = false; }
  });
}

// ── Add local file ────────────────────────────────────────────
const addLocalClipBtn = document.getElementById('addLocalClipBtn');
if (addLocalClipBtn) {
  addLocalClipBtn.addEventListener('click', async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/pick-file`, { method:'POST' });
      const d = await res.json();
      if (!d.path) return;
      const fname = d.path.split(/[\\/]/).pop();
      const ext = fname.split('.').pop().toLowerCase();
      addClip({
        type: ['mp4','mov','avi','mkv','webm'].includes(ext) ? 'video' : 'slide',
        title: fname,
        src: d.path,
        thumb: null,
        duration: 5,
      });
      toast('Clip added', 'success');
    } catch { toast('Could not open file picker', 'error'); }
  });
}

// ── Export video ──────────────────────────────────────────────
const exportVideoBtn = document.getElementById('exportVideoBtn');
if (exportVideoBtn) {
  exportVideoBtn.addEventListener('click', async () => {
    if (!currentAudioFilename) return toast('Generate a voiceover first (Voiceover tab)', 'error');
    if (videoSequence.length === 0) return toast('Add at least one clip to the sequence', 'error');

    exportVideoBtn.disabled = true;
    const prog = document.getElementById('exportProgress');
    const vidReady = document.getElementById('videoReady');
    if (prog) prog.style.display = 'block';
    if (vidReady) vidReady.style.display = 'none';

    const resVal = document.getElementById('exportRes')?.value || '1920x1080';
    const [rw, rh] = resVal.split('x').map(Number);

    try {
      const res = await fetch(`${BACKEND_URL}/assemble-video`, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({
          topic:          document.getElementById('topic')?.value?.trim() || '',
          audio_filename: currentAudioFilename,
          sequence: videoSequence.map(c => ({
            type: c.type, src: c.src || '', text: c.text || c.title || '',
            style: c.style || 'dark', subtitle: c.subtitle || '',
            duration: c.duration, enabled: true,
          })),
          burn_captions: document.getElementById('burnCaptions')?.checked || false,
          resolution: [rw, rh],
          fps: parseInt(document.getElementById('exportFps')?.value || '30'),
        })
      });
      const d = await res.json();
      if (d.error) throw new Error(d.error);
      exportJobId = d.job_id;
      startExportPoll();
    } catch(e) {
      exportVideoBtn.disabled = false;
      if (prog) prog.style.display = 'none';
      toast(e.message, 'error');
    }
  });
}

function startExportPoll() {
  if (exportPollInterval) clearInterval(exportPollInterval);
  exportPollInterval = setInterval(async () => {
    if (!exportJobId) return;
    try {
      const res = await fetch(`${BACKEND_URL}/video-status/${exportJobId}`);
      const job = await res.json();

      const msgEl = document.getElementById('exportMsg');
      const pctEl = document.getElementById('exportPct');
      const bar   = document.getElementById('exportBar');
      if (msgEl) msgEl.textContent = job.msg || '';
      if (pctEl) pctEl.textContent = `${job.pct || 0}%`;
      if (bar) bar.style.width = `${job.pct || 0}%`;

      if (job.status === 'done') {
        clearInterval(exportPollInterval);
        const btn = document.getElementById('exportVideoBtn');
        if (btn) btn.disabled = false;
        const vid = document.getElementById('exportedVideo');
        const ready = document.getElementById('videoReady');
        if (vid) vid.src = `${BACKEND_URL}/video/${job.filename}`;
        if (ready) ready.style.display = 'block';
        const pct2 = document.getElementById('exportPct');
        if (pct2) pct2.textContent = '100%';
        if (bar) bar.style.width = '100%';
        toast('Video exported! 🎉', 'success');
        // Store for save
        window._lastVideoFilename = job.filename;

      } else if (job.status === 'error') {
        clearInterval(exportPollInterval);
        const btn = document.getElementById('exportVideoBtn');
        if (btn) btn.disabled = false;
        const prog = document.getElementById('exportProgress');
        if (prog) prog.style.display = 'none';
        toast(`Export failed: ${job.error}`, 'error');
      }
    } catch { clearInterval(exportPollInterval); }
  }, 1000);
}

// Save + open folder for video
const saveVideoBtn = document.getElementById('saveVideoBtn');
if (saveVideoBtn) {
  saveVideoBtn.addEventListener('click', async () => {
    const fn = window._lastVideoFilename;
    if (!fn) return;
    const saved = await window.electron.saveVideo(fn);
    if (saved) toast(`Saved to ${saved}`, 'success');
  });
}

const openVideoFolderBtn = document.getElementById('openVideoFolderBtn');
if (openVideoFolderBtn) {
  openVideoFolderBtn.addEventListener('click', () => window.electron.openVideoFolder());
}

// Full-screen button for the assembled/exported video
document.getElementById('fullscreenExportedVideoBtn')?.addEventListener('click', () => {
  const vid = document.getElementById('exportedVideo');
  const fname = (window._lastVideoFilename || '').split(/[\\/]/).pop();
  if (fname) {
    openPreviewModal({
      title: fname,
      videoUrl: `${BACKEND_URL}/video/${fname}`,
    });
  } else if (vid?.src) {
    openPreviewModal({ title: 'Exported Video', videoUrl: vid.src });
  }
});

// ── Thumbnail generator ───────────────────────────────────────
const genThumbBtn = document.getElementById('genThumbBtn');
if (genThumbBtn) {
  genThumbBtn.addEventListener('click', async () => {
    const title = document.getElementById('thumbTitle')?.value?.trim() || 'YouTube Video';
    const topic = document.getElementById('thumbTopic')?.value?.trim() || 'content';
    const openaiKey = document.getElementById('openaiKeyThumb')?.value?.trim() || '';

    genThumbBtn.disabled = true;
    genThumbBtn.textContent = 'Generating…';
    try {
      const videoTopic = document.getElementById('topic')?.value?.trim() || topic;
      const res = await fetch(`${BACKEND_URL}/generate-thumbnail`, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ title, topic: videoTopic, openai_key: openaiKey })
      });
      const d = await res.json();
      if (d.error) throw new Error(d.error);

      const img = document.getElementById('thumbImg');
      const preview = document.getElementById('thumbPreview');
      if (img) img.src = `${BACKEND_URL}/thumbnail/${d.filename}`;
      if (preview) preview.style.display = 'block';
      window._lastThumbFilename = d.filename;
      toast('Thumbnail generated!', 'success');
    } catch(e) { toast(e.message, 'error'); }
    finally { genThumbBtn.disabled = false; genThumbBtn.textContent = 'Generate Thumbnail'; }
  });
}

const saveThumbBtn = document.getElementById('saveThumbBtn');
if (saveThumbBtn) {
  saveThumbBtn.addEventListener('click', async () => {
    const fn = window._lastThumbFilename;
    if (!fn) return;
    const saved = await window.electron.saveThumbnail(fn);
    if (saved) toast(`Saved to ${saved}`, 'success');
  });
}

// Keep voiceover indicator in sync — called directly after loadAudio below

// Expose for tab switch
window.initVideoEditor = function() {
  updateVoiceoverIndicator();
  renderSequence();
};

// ════════════════════════════════════════════════════════════════
//  AI VIDEO GENERATOR
// ════════════════════════════════════════════════════════════════

// Gemini image model IDs (use Gemini API key — no HF token, no Higgsfield credits)
const GEMINI_IMG_MODELS = new Set([
  "gemini_flash_img", "gemini_imagen_3",
]);

// Higgsfield image model IDs (no HF token needed)
const HIGGSFIELD_IMG_MODELS = new Set([
  "nano_banana_2","nano_banana_flash","nano_banana",
  "flux_2","flux_kontext","grok_image",
  "seedream_v5_lite","seedream_v4_5",
  "cinematic_studio_2_5","text2image_soul_v2",
  "imagegen_2_0","z_image","image_auto",
]);

const AI_METHOD_MODELS = {
  ken_burns: {
    // ── Google Gemini (free — uses existing Gemini API key) ────────────────────
    "gemini_flash_img": "Gemini Flash Image (Free — Gemini Key ⭐)",
    "gemini_imagen_3":  "Imagen 3 (Free — Gemini Key)",
    // ── HuggingFace (free) ────────────────────────────────────────────────────
    "black-forest-labs/FLUX.1-schnell":         "FLUX Schnell (HF — Fast, Free)",
    "stabilityai/stable-diffusion-xl-base-1.0": "SDXL (HF — Best Quality, Free)",
    "stabilityai/stable-diffusion-2-1":         "SD 2.1 (HF — Lighter, Free)",
    "Lykon/dreamshaper-xl-1-0":                 "DreamShaper XL (HF — Free)",
    // ── Higgsfield (credit-based) ─────────────────────────────────────────────
    "nano_banana_2":        "Nano Banana Pro (Higgsfield ⭐)",
    "nano_banana_flash":    "Nano Banana 2 (Higgsfield)",
    "nano_banana":          "Nano Banana (Higgsfield)",
    "flux_2":               "FLUX.2 (Higgsfield)",
    "flux_kontext":         "Flux Kontext (Higgsfield)",
    "grok_image":           "Grok Image (Higgsfield)",
    "seedream_v5_lite":     "Seedream V5 Lite (Higgsfield)",
    "seedream_v4_5":        "Seedream 4.5 (Higgsfield)",
    "cinematic_studio_2_5": "Cinematic Studio 2.5 (Higgsfield)",
    "text2image_soul_v2":   "Soul V2 (Higgsfield)",
    "imagegen_2_0":         "GPT Image 2 (Higgsfield)",
    "z_image":              "Z Image (Higgsfield)",
    "image_auto":           "Image Auto / Best (Higgsfield)",
  },
  hf_video: {
    "damo-vilab/text-to-video-ms-1.7b": "Text-to-Video 1.7B (DAMO)",
    "ali-vilab/text-to-video-ms-1.7b": "Text-to-Video 1.7B (Ali)",
  },
  higgsfield: {
    "kling3_0":             "Kling v3.0",
    "kling2_6":             "Kling 2.6 Video",
    "minimax_hailuo":       "Minimax Hailuo",
    "veo3":                 "Google Veo 3",
    "veo3_1":               "Google Veo 3.1",
    "veo3_1_lite":          "Google Veo 3.1 Lite (cheaper)",
    "grok_video":           "Grok Video",
    "wan2_7":               "Wan 2.7",
    "wan2_6":               "Wan 2.6 Video",
    "seedance_2_0":         "Seedance 2.0",
    "seedance1_5":          "Seedance 1.5 Pro",
    "cinematic_studio_3_0": "Cinematic Studio 3.0",
  },
};

let _aiMethod = 'ken_burns';

// ── Method toggle ─────────────────────────────────────────────
document.querySelectorAll('#videoMethodToggle .source-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#videoMethodToggle .source-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    _aiMethod = btn.dataset.method;
    _updateAiMethodUI();
  });
});

function _updateAiMethodUI() {
  // Show/hide descriptions
  ['ken_burns', 'hf_video', 'higgsfield'].forEach(m => {
    const el = document.getElementById(`methodDesc-${m}`);
    if (el) el.style.display = m === _aiMethod ? 'block' : 'none';
  });

  // Populate model dropdown
  const models = AI_METHOD_MODELS[_aiMethod] || {};
  const sel = document.getElementById('aiModel');
  if (sel) {
    sel.innerHTML = Object.entries(models)
      .map(([id, name]) => `<option value="${id}">${name}</option>`).join('');
  }

  // HF token not needed for: Higgsfield video/image methods
  _updateHfTokenVisibility();

  // Check auth when Higgsfield is selected
  if (_aiMethod === 'higgsfield') _checkHiggsfieldAuth();
}

function _updateHfTokenVisibility() {
  const hfGrp = document.getElementById('hfTokenGroup');
  if (!hfGrp) return;
  const model = document.getElementById('aiModel')?.value || '';
  const isGemini     = GEMINI_IMG_MODELS.has(model);
  const isHighsfield = HIGGSFIELD_IMG_MODELS.has(model);
  const needsHf = _aiMethod !== 'higgsfield' && !isHighsfield && !isGemini;
  hfGrp.style.display = needsHf ? 'block' : 'none';

  // Remove stale notes first
  document.getElementById('hfNoteHiggsfield')?.remove();
  document.getElementById('hfNoteGemini')?.remove();

  if (_aiMethod === 'ken_burns') {
    if (isGemini) {
      const note = document.createElement('div');
      note.id = 'hfNoteGemini';
      note.style.cssText = 'font-size:10px;color:var(--text3);margin-bottom:8px';
      note.textContent = 'Uses your Gemini API key — no HF token or credits needed.';
      hfGrp.parentNode.insertBefore(note, hfGrp.nextSibling);
    } else if (isHighsfield) {
      const note = document.createElement('div');
      note.id = 'hfNoteHiggsfield';
      note.style.cssText = 'font-size:10px;color:var(--text3);margin-bottom:8px';
      note.textContent = 'Uses Higgsfield credits — no HF token needed.';
      hfGrp.parentNode.insertBefore(note, hfGrp.nextSibling);
    }
  }
}

async function _checkHiggsfieldAuth() {
  const statusEl = document.getElementById('higgsfieldAuthStatus');
  if (!statusEl) return;
  statusEl.textContent = 'Checking…';
  statusEl.style.color = 'var(--text3)';
  try {
    const r = await fetch(`${BACKEND_URL}/higgsfield-status`);
    const d = await r.json();
    if (!d.installed) {
      statusEl.innerHTML = `<span style="color:var(--warning)">⚠ CLI not found — run: <code style="background:var(--bg4);padding:1px 4px;border-radius:3px;font-size:10px">npm install -g @higgsfield/cli</code></span>`;
    } else if (d.authenticated) {
      statusEl.innerHTML = `<span style="color:var(--success)">✓ Logged in — ready to generate</span>`;
    } else {
      statusEl.innerHTML = `<span style="color:var(--warning)">⚠ Not logged in — run: <code style="background:var(--bg4);padding:1px 4px;border-radius:3px;font-size:10px">higgsfield auth login</code></span>`;
    }
  } catch {
    statusEl.textContent = '';
  }
}

// Pre-fill HF token from backend env or localStorage
async function _loadHfToken() {
  const saved = localStorage.getItem('hf_token');
  const inp = document.getElementById('aiHfToken');
  if (!inp) return;
  if (saved) { inp.value = saved; return; }
  try {
    const r = await fetch(`${BACKEND_URL}/hf-token`);
    const d = await r.json();
    if (d.token) { inp.value = d.token; localStorage.setItem('hf_token', d.token); }
  } catch { /* ignore */ }
}

document.getElementById('aiHfToken')?.addEventListener('change', e => {
  localStorage.setItem('hf_token', e.target.value.trim());
});

// ════════════════════════════════════════════════════════════════
//  FULL-SCREEN PREVIEW MODAL
// ════════════════════════════════════════════════════════════════

let _pendingPreviewClip = null;   // clip object waiting to be added from preview

function openPreviewModal({ title = '', imgUrl = '', videoUrl = '', pendingClip = null }) {
  const modal    = document.getElementById('previewModal');
  const img      = document.getElementById('previewModalImg');
  const vid      = document.getElementById('previewModalVideo');
  const titleEl  = document.getElementById('previewModalTitle');
  const addBtn   = document.getElementById('previewModalAddBtn');

  titleEl.textContent = title;
  _pendingPreviewClip = pendingClip;

  // Reset
  img.style.display = 'none'; img.src = '';
  vid.style.display = 'none'; vid.src = '';
  addBtn.style.display = pendingClip ? 'inline-flex' : 'none';

  if (videoUrl) {
    vid.src = videoUrl;
    vid.style.display = 'block';
  } else if (imgUrl) {
    img.src = imgUrl;
    img.style.display = 'block';
  }

  modal.style.display = 'flex';
}

function closePreviewModal() {
  const modal = document.getElementById('previewModal');
  const vid   = document.getElementById('previewModalVideo');
  vid.pause(); vid.src = '';
  modal.style.display = 'none';
  _pendingPreviewClip = null;
}

document.getElementById('previewModalClose')?.addEventListener('click', closePreviewModal);
document.getElementById('previewModal')?.addEventListener('click', e => {
  if (e.target === document.getElementById('previewModal')) closePreviewModal();
});
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closePreviewModal();
});
document.getElementById('previewModalAddBtn')?.addEventListener('click', () => {
  if (_pendingPreviewClip) { addClip(_pendingPreviewClip); toast('Clip added!', 'success'); }
  closePreviewModal();
});

// ════════════════════════════════════════════════════════════════
//  AI CLIP PREVIEW PANEL
// ════════════════════════════════════════════════════════════════

let _lastGeneratedJob = null;   // job result kept for "Add to Sequence" confirm

function showAiPreview(job, promptText, duration) {
  const panel   = document.getElementById('aiPreviewPanel');
  const imgWrap = document.getElementById('aiPreviewImgWrap');
  const imgEl   = document.getElementById('aiPreviewImg');
  const vidEl   = document.getElementById('aiPreviewVideo');

  if (!panel) return;

  const clipPath = job.clip_path || '';
  const imgPath  = job.img_path  || '';
  const imgFname = imgPath.split(/[\\/]/).pop();
  const vidFname = clipPath.split(/[\\/]/).pop();

  _lastGeneratedJob = { job, promptText, duration };

  // Show image
  if (imgFname) {
    const url = `${BACKEND_URL}/ai-clip-thumb/${imgFname}`;
    imgEl.src = url;
    imgWrap.style.display = 'block';
  } else {
    imgWrap.style.display = 'none';
  }

  // Show video if available and short enough to preview inline
  if (vidFname && clipPath) {
    const vidUrl = `${BACKEND_URL}/ai-clip-file/${vidFname}`;
    vidEl.src = vidUrl;
    vidEl.style.display = 'block';
  } else {
    vidEl.style.display = 'none';
  }

  panel.style.display = 'block';
}

// Image hover overlay
const _previewImgWrap = document.getElementById('aiPreviewImgWrap');
const _previewOverlay = document.getElementById('aiPreviewImgOverlay');
if (_previewImgWrap && _previewOverlay) {
  _previewImgWrap.addEventListener('mouseenter', () => { _previewOverlay.style.opacity = '1'; });
  _previewImgWrap.addEventListener('mouseleave', () => { _previewOverlay.style.opacity = '0'; });
  _previewImgWrap.addEventListener('click', () => {
    const src = document.getElementById('aiPreviewImg')?.src;
    if (src) openPreviewModal({ title: 'Generated Image', imgUrl: src });
  });
}

// Full-screen from preview panel
document.getElementById('fullPreviewBtn')?.addEventListener('click', () => {
  if (!_lastGeneratedJob) return;
  const { job, promptText, duration } = _lastGeneratedJob;
  const imgPath  = (job.img_path  || '').split(/[\\/]/).pop();
  const clipPath = (job.clip_path || '').split(/[\\/]/).pop();
  const clip = _buildClipObj(job, promptText, duration);
  openPreviewModal({
    title: promptText.slice(0, 60),
    imgUrl:  imgPath  ? `${BACKEND_URL}/ai-clip-thumb/${imgPath}`  : '',
    videoUrl: clipPath ? `${BACKEND_URL}/ai-clip-file/${clipPath}` : '',
    pendingClip: clip,
  });
});

// Confirm add from preview panel
document.getElementById('confirmAddClipBtn')?.addEventListener('click', () => {
  if (!_lastGeneratedJob) return;
  const { job, promptText, duration } = _lastGeneratedJob;
  addClip(_buildClipObj(job, promptText, duration));
  toast('Clip added to sequence!', 'success');
  document.getElementById('aiPreviewPanel').style.display = 'none';
  _lastGeneratedJob = null;
});

// Discard from preview panel
document.getElementById('discardAiPreviewBtn')?.addEventListener('click', () => {
  document.getElementById('aiPreviewPanel').style.display = 'none';
  _lastGeneratedJob = null;
});

function _previewClipCard(clip) {
  const fname = (clip.src || '').split(/[\\/]/).pop();
  const isVideo = clip.type === 'video' && fname && fname.endsWith('.mp4');
  openPreviewModal({
    title: clip.title || 'Preview',
    imgUrl:   clip.thumb || '',
    videoUrl: isVideo ? `${BACKEND_URL}/ai-clip-file/${fname}` : '',
  });
}

function _buildClipObj(job, promptText, duration) {
  const clipPath = job.clip_path || '';
  const imgPath  = job.img_path  || '';
  const imgFname = imgPath.split(/[\\/]/).pop();
  return {
    type: 'video', title: promptText.slice(0, 48),
    src: clipPath,
    thumb: imgFname ? `${BACKEND_URL}/ai-clip-thumb/${imgFname}` : null,
    duration: parseFloat(duration) || 5,
  };
}

// ── Generate single AI clip ───────────────────────────────────
document.getElementById('genAiClipBtn')?.addEventListener('click', generateAiClip);

async function generateAiClip() {
  const prompt = document.getElementById('aiPrompt')?.value?.trim();
  if (!prompt) return toast('Enter a prompt for the clip', 'error');

  const model    = document.getElementById('aiModel')?.value || '';
  const hfToken  = document.getElementById('aiHfToken')?.value?.trim() || '';
  const geminiKey = document.getElementById('apiKey')?.value?.trim() || '';
  const needsHf  = (_aiMethod === 'ken_burns' || _aiMethod === 'hf_video')
                   && !HIGGSFIELD_IMG_MODELS.has(model) && !GEMINI_IMG_MODELS.has(model);
  if (needsHf && !hfToken) {
    return toast('Enter your HF token (free at hf.co/settings/tokens)', 'error');
  }
  if (GEMINI_IMG_MODELS.has(model) && !geminiKey) {
    return toast('Enter your Gemini API key above to use Gemini image generation', 'error');
  }

  const btn   = document.getElementById('genAiClipBtn');
  const prog  = document.getElementById('aiClipProgress');
  btn.disabled = true;
  btn.textContent = 'Generating…';
  if (prog) prog.style.display = 'block';

  try {
    const res = await fetch(`${BACKEND_URL}/generate-ai-clip`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        topic:      document.getElementById('topic')?.value?.trim() || '',
        method:     _aiMethod,
        prompt,
        duration:   parseFloat(document.getElementById('aiDuration')?.value || '5'),
        hf_token:   hfToken,
        gemini_key: geminiKey,
        model,
      }),
    });
    const d = await res.json();
    if (d.error) throw new Error(d.error);

    const job = await _pollAiClipJob(d.job_id);
    // Show preview panel — user confirms before adding
    showAiPreview(job, prompt, parseFloat(document.getElementById('aiDuration')?.value || '5'));
    toast('Clip ready — preview below', 'success');
  } catch (e) {
    // Show full error so user knows what to fix
    const msg = e.message || 'Generation failed';
    toast(msg, 'error');
    // Also write it into the progress bar area for visibility
    const msgEl = document.getElementById('aiClipMsg');
    const pctEl = document.getElementById('aiClipPct');
    const bar   = document.getElementById('aiClipBar');
    if (msgEl) { msgEl.textContent = `Error: ${msg.slice(0, 80)}`; msgEl.style.color = 'var(--error)'; }
    if (pctEl) pctEl.textContent = 'Failed';
    if (bar)   bar.style.background = 'var(--error)';
    // Keep progress visible so user can read the error; hide after 6s
    setTimeout(() => {
      const p = document.getElementById('aiClipProgress');
      if (p) p.style.display = 'none';
      if (msgEl) msgEl.style.color = '';
      if (bar)   bar.style.background = '';
    }, 6000);
    return;  // skip the finally-hide of progress
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg> Generate &amp; Add Clip`;
    if (prog) prog.style.display = 'none';
    const bar = document.getElementById('aiClipBar');
    if (bar) bar.style.width = '0%';
  }
}

function _pollAiClipJob(jobId) {
  return new Promise((resolve, reject) => {
    const iv = setInterval(async () => {
      try {
        const r = await fetch(`${BACKEND_URL}/ai-clip-status/${jobId}`);
        const job = await r.json();

        const bar = document.getElementById('aiClipBar');
        const msg = document.getElementById('aiClipMsg');
        const pct = document.getElementById('aiClipPct');
        if (bar) bar.style.width = `${job.pct || 0}%`;
        if (msg) msg.textContent = job.msg || 'Working…';
        if (pct) pct.textContent = `${job.pct || 0}%`;

        if (job.status === 'done') {
          clearInterval(iv);
          resolve(job);
        } else if (job.status === 'error') {
          clearInterval(iv);
          reject(new Error(job.error || 'Generation failed'));
        }
      } catch (e) { clearInterval(iv); reject(e); }
    }, 1000);
  });
}

function _addAiClipToSequence(job, promptText) {
  const clipPath = job.clip_path || '';
  const imgPath  = job.img_path  || '';
  const fname    = clipPath.split(/[\\/]/).pop();
  const imgFname = imgPath.split(/[\\/]/).pop();

  addClip({
    type:     'video',
    title:    promptText.slice(0, 48),
    src:      clipPath,
    thumb:    imgFname ? `${BACKEND_URL}/ai-clip-thumb/${imgFname}` : null,
    duration: parseFloat(document.getElementById('aiDuration')?.value || '5'),
  });
}

// ── Auto-fill from voiceover ──────────────────────────────────
document.getElementById('autoFillBtn')?.addEventListener('click', autoFillFromVoiceover);

async function autoFillFromVoiceover() {
  const hfToken = document.getElementById('aiHfToken')?.value?.trim() || '';
  const _autoModel = document.getElementById('aiModel')?.value || '';
  const _needsHfAuto = (_aiMethod === 'ken_burns' || _aiMethod === 'hf_video')
                       && !HIGGSFIELD_IMG_MODELS.has(_autoModel)
                       && !GEMINI_IMG_MODELS.has(_autoModel);
  if (_needsHfAuto && !hfToken) {
    return toast('Enter your HF token first', 'error');
  }
  if (GEMINI_IMG_MODELS.has(_autoModel) && !document.getElementById('apiKey')?.value?.trim()) {
    return toast('Enter your Gemini API key above to use Gemini image generation', 'error');
  }
  if (!currentAudioFilename) {
    return toast('Generate a voiceover first — it\'s needed for scene timing', 'error');
  }

  const btn      = document.getElementById('autoFillBtn');
  const prog     = document.getElementById('autoFillProgress');
  const msgEl    = document.getElementById('autoFillMsg');
  const pctEl    = document.getElementById('autoFillPct');
  const barEl    = document.getElementById('autoFillBar');
  const detailEl = document.getElementById('autoFillDetail');
  btn.disabled = true;
  if (prog) prog.style.display = 'block';

  const _upd = (msg, pct, detail = '') => {
    if (msgEl) msgEl.textContent = msg;
    if (pctEl) pctEl.textContent = `${pct}%`;
    if (barEl) barEl.style.width = `${pct}%`;
    if (detailEl) detailEl.textContent = detail;
  };

  const _topic = document.getElementById('topic')?.value?.trim() || '';
  try {
    // Step 1: use cached scenes from Get Transcript, or fetch fresh
    let scenes;
    if (_cachedScenes && _cachedScenes.length > 0) {
      scenes = _cachedScenes;
      _upd(`Using ${scenes.length} cached scenes from transcript…`, 10);
    } else {
      _upd('Analysing voiceover…', 5);
      const scenesRes = await fetch(`${BACKEND_URL}/generate-scenes`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          audio_filename: currentAudioFilename,
          script_text:    document.getElementById('scriptEditor')?.value?.trim() || '',
          topic:          _topic || 'video',
          style:          document.getElementById('aiImageStyle')?.value || 'cinematic',
          interval:       parseFloat(document.getElementById('sceneInterval')?.value || '8'),
          api_key:        document.getElementById('apiKey')?.value?.trim() || '',
        }),
      });
      const scenesData = await scenesRes.json();
      if (scenesData.error) throw new Error(scenesData.error);
      scenes = scenesData.scenes;
    }

    _upd(`Found ${scenes.length} scenes — generating clips…`, 10);

    // Step 2: generate a clip per scene
    let done = 0;
    for (const scene of scenes) {
      const pct = 10 + Math.round((done / scenes.length) * 85);
      _upd(`Clip ${done + 1}/${scenes.length}`, pct, scene.text.slice(0, 60));

      try {
        const genRes = await fetch(`${BACKEND_URL}/generate-ai-clip`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            topic:      _topic,
            method:     _aiMethod,
            prompt:     scene.prompt || scene.text.slice(0, 80),
            duration:   scene.duration || 6,
            hf_token:   hfToken,
            gemini_key: document.getElementById('apiKey')?.value?.trim() || '',
            model:      document.getElementById('aiModel')?.value || '',
          }),
        });
        const genData = await genRes.json();
        if (genData.error) throw new Error(genData.error);

        // Override progress display with per-clip bar while polling
        const job = await _pollAiClipJobSilent(genData.job_id, done, scenes.length, _upd);
        _addAiClipToSequenceWithDuration(job, scene.text, scene.duration);
      } catch (e) {
        const errMsg = e.message || 'Unknown error';
        // Show the real error in the progress bar
        _upd(`⚠ Clip ${done + 1} failed: ${errMsg.slice(0, 80)}`, Math.round(10 + (done / scenes.length) * 85));
        // Add a text slide placeholder so timing stays aligned, but mark it
        addClip({
          type: 'slide', title: `⚠ ${scene.text.slice(0, 38)}`,
          text: scene.text.slice(0, 80), style: 'dark',
          duration: scene.duration || 6,
        });
        toast(`Clip ${done + 1} failed: ${errMsg.slice(0, 60)}`, 'error');
        // If the very first clip fails, stop early so user can fix settings
        if (done === 0) {
          throw new Error(`First clip failed — check your settings.\n${errMsg}`);
        }
      }
      done++;
    }

    _upd(`Done — ${done} clips added!`, 100);
    toast(`Auto-filled ${done} clips!`, 'success');
    setTimeout(() => { if (prog) prog.style.display = 'none'; }, 3000);

  } catch (e) {
    toast(e.message, 'error');
    if (prog) prog.style.display = 'none';
  } finally {
    btn.disabled = false;
  }
}

function _pollAiClipJobSilent(jobId, sceneIdx, total, updFn) {
  return new Promise((resolve, reject) => {
    const iv = setInterval(async () => {
      try {
        const r = await fetch(`${BACKEND_URL}/ai-clip-status/${jobId}`);
        const job = await r.json();
        const basePct = 10 + Math.round((sceneIdx / total) * 85);
        const innerPct = Math.round((job.pct || 0) * (85 / total) / 100);
        updFn(`Clip ${sceneIdx + 1}/${total}: ${job.msg || ''}`, basePct + innerPct);
        if (job.status === 'done') { clearInterval(iv); resolve(job); }
        else if (job.status === 'error') { clearInterval(iv); reject(new Error(job.error || 'failed')); }
      } catch (e) { clearInterval(iv); reject(e); }
    }, 1000);
  });
}

function _addAiClipToSequenceWithDuration(job, promptText, duration) {
  const clipPath = job.clip_path || '';
  const imgPath  = job.img_path  || '';
  const imgFname = imgPath.split(/[\\/]/).pop();
  addClip({
    type: 'video', title: promptText.slice(0, 48),
    src: clipPath,
    thumb: imgFname ? `${BACKEND_URL}/ai-clip-thumb/${imgFname}` : null,
    duration: duration || 6,
  });
}

// ════════════════════════════════════════════════════════════════
//  LOAD EXTERNAL AUDIO
// ════════════════════════════════════════════════════════════════

document.getElementById('loadAudioBtn')?.addEventListener('click', loadExternalAudio);

async function loadExternalAudio() {
  const btn = document.getElementById('loadAudioBtn');
  btn.disabled = true;
  btn.textContent = 'Picking file…';

  try {
    // Open native file picker (picks from disk)
    const pickRes = await fetch(`${BACKEND_URL}/pick-file`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title: 'Select audio file',
        filetypes: 'audio',
      }),
    });
    const pick = await pickRes.json();
    if (!pick.path) { btn.disabled = false; btn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg> Load Existing Audio (WAV / MP3 / FLAC)`; return; }

    btn.textContent = 'Loading…';

    // Copy into topic folder and register
    const loadRes = await fetch(`${BACKEND_URL}/load-audio`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        file_path: pick.path,
        topic: document.getElementById('topic')?.value?.trim() || '',
      }),
    });
    const loaded = await loadRes.json();
    if (loaded.error) throw new Error(loaded.error);

    // Set as current audio and update player
    currentAudioFilename = loaded.filename;
    const fname = loaded.filename;
    loadAudio(`${BACKEND_URL}/audio/${fname}`, fname);
    updateVoiceoverIndicator();
    switchTab('audio');
    toast(`Audio loaded: ${fname}`, 'success');

  } catch (e) {
    toast(e.message, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg> Load Existing Audio (WAV / MP3 / FLAC)`;
  }
}

// ════════════════════════════════════════════════════════════════
//  TRANSCRIPT PREVIEW + SCENE PROMPTS
// ════════════════════════════════════════════════════════════════

let _cachedScenes = null;  // scenes from last transcription, used by autoFill

document.getElementById('getTranscriptBtn')?.addEventListener('click', getTranscript);
document.getElementById('closeTranscriptBtn')?.addEventListener('click', () => {
  document.getElementById('transcriptSection').style.display = 'none';
});

// ── Transcript source toggle ──────────────────────────────────
let _transcriptSrc = 'transcribe';

document.querySelectorAll('#transcriptSourceToggle .source-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#transcriptSourceToggle .source-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    _transcriptSrc = btn.dataset.tsrc;

    const pasteArea = document.getElementById('pasteTranscriptArea');
    const btnLabel  = document.getElementById('getTranscriptBtnLabel');
    if (_transcriptSrc === 'paste') {
      if (pasteArea) pasteArea.style.display = 'block';
      if (btnLabel)  btnLabel.textContent = 'Use This Transcript';
    } else {
      if (pasteArea) pasteArea.style.display = 'none';
      if (btnLabel)  btnLabel.textContent = 'Get Transcript';
    }

    // Clear stale cached scenes when switching modes
    _cachedScenes = null;
    const section = document.getElementById('transcriptSection');
    if (section) section.style.display = 'none';
  });
});

// Paste transcript word count
document.getElementById('pasteTranscriptInput')?.addEventListener('input', e => {
  const words = e.target.value.trim().split(/\s+/).filter(Boolean).length;
  const el = document.getElementById('pasteWordCount');
  if (el) el.textContent = `${words} word${words !== 1 ? 's' : ''}`;
});
document.getElementById('clearPasteTranscriptBtn')?.addEventListener('click', () => {
  const el = document.getElementById('pasteTranscriptInput');
  if (el) { el.value = ''; el.dispatchEvent(new Event('input')); }
  _cachedScenes = null;
  const section = document.getElementById('transcriptSection');
  if (section) section.style.display = 'none';
});

// ── Timestamp helpers ─────────────────────────────────────────
function _parseTs(str) {
  const parts = str.trim().split(':').map(Number);
  if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
  if (parts.length === 2) return parts[0] * 60 + parts[1];
  return 0;
}

function _fmtTimestamp(t) {
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const s = Math.floor(t % 60);
  if (h > 0) return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
  return `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
}

function _segmentsToTimestampedText(segments) {
  return segments.map(s =>
    `[${_fmtTimestamp(s.start || 0)} – ${_fmtTimestamp(s.end || 0)}] ${s.text}`
  ).join('\n');
}

// Parse a pasted transcript that may contain [MM:SS – MM:SS] timestamps
function _parsePastedTranscript(text) {
  const tsLine = /^\[(\d{1,2}:\d{2}(?::\d{2})?)\s*[–\-]\s*(\d{1,2}:\d{2}(?::\d{2})?)\]\s*(.*)/;
  const lines   = text.trim().split('\n');
  const segments = [];

  for (const line of lines) {
    const m = line.match(tsLine);
    if (m) {
      const start = _parseTs(m[1]), end = _parseTs(m[2]);
      segments.push({ start, end, text: m[3].trim(), duration: Math.max(0.5, end - start) });
    }
  }
  return segments;   // empty if no timestamps found
}

// Build display text from raw text (no timestamps) — format as-is
function _plainTranscriptDisplay(text) { return text; }

// Clear cached scenes when style/interval changes (they'd use wrong prompts)
['aiImageStyle', 'sceneInterval'].forEach(id => {
  document.getElementById(id)?.addEventListener('change', () => {
    _cachedScenes = null;
    const section = document.getElementById('transcriptSection');
    if (section) section.style.display = 'none';
  });
});

async function getTranscript() {
  const btn      = document.getElementById('getTranscriptBtn');
  const style    = document.getElementById('aiImageStyle')?.value || 'cinematic';
  const interval = parseFloat(document.getElementById('sceneInterval')?.value || '8');
  const apiKey   = document.getElementById('apiKey')?.value?.trim() || '';
  const topic    = document.getElementById('topic')?.value?.trim() || 'video';

  btn.disabled = true;

  const _restoreBtn = () => {
    btn.disabled = false;
    const lbl = document.getElementById('getTranscriptBtnLabel');
    if (lbl) lbl.textContent = _transcriptSrc === 'paste' ? 'Use This Transcript' : 'Get Transcript';
  };

  try {
    let rawSegments = [];     // [{start,end,text,duration}]
    let displayText = '';
    let scenesPayload = {};

    // ── Mode: paste ──────────────────────────────────────────────────────────
    if (_transcriptSrc === 'paste') {
      const pasted = document.getElementById('pasteTranscriptInput')?.value?.trim() || '';
      if (!pasted) { toast('Paste your transcript first', 'error'); _restoreBtn(); return; }

      // Try to detect and parse timestamps
      rawSegments = _parsePastedTranscript(pasted);

      if (rawSegments.length > 0) {
        // Has timestamps — display as-is (already has [MM:SS – MM:SS] format)
        displayText = pasted;
        // Build scenes directly from parsed segments (no audio, no server transcription)
        scenesPayload = {
          script_text: rawSegments.map(s => s.text).join(' '),
          topic, style, interval, api_key: apiKey,
        };
      } else {
        // Plain text — display as-is, send as script_text
        displayText = pasted;
        scenesPayload = { script_text: pasted, topic, style, interval, api_key: apiKey };
      }

    // ── Mode: transcribe ─────────────────────────────────────────────────────
    } else {
      if (!currentAudioFilename) {
        toast('Load or generate a voiceover first (Voiceover tab)', 'error');
        _restoreBtn(); return;
      }
      const lbl = document.getElementById('getTranscriptBtnLabel');
      if (lbl) lbl.textContent = 'Transcribing…';

      const tRes = await fetch(`${BACKEND_URL}/transcribe-audio`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ audio_filename: currentAudioFilename }),
      });
      const tData = await tRes.json();
      if (tData.error) throw new Error(tData.error);

      rawSegments = tData.segments || [];
      // Build timestamped display: [MM:SS – MM:SS] text per segment
      displayText = rawSegments.length
        ? _segmentsToTimestampedText(rawSegments)
        : (tData.transcript || '(no speech detected)');

      scenesPayload = {
        audio_filename: currentAudioFilename,
        script_text: tData.transcript,
        topic, style, interval, api_key: apiKey,
      };
    }

    // ── Show transcript ──────────────────────────────────────────────────────
    const section = document.getElementById('transcriptSection');
    const textEl  = document.getElementById('transcriptText');
    if (section) section.style.display = 'block';
    if (textEl)  textEl.value = displayText;

    // ── Build scene prompts ──────────────────────────────────────────────────
    const lbl2 = document.getElementById('getTranscriptBtnLabel');
    if (lbl2) lbl2.textContent = 'Building scene prompts…';

    const sRes = await fetch(`${BACKEND_URL}/generate-scenes`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(scenesPayload),
    });
    const sData = await sRes.json();
    if (sData.error) throw new Error(sData.error);

    _cachedScenes = sData.scenes;

    // ── Update UI ────────────────────────────────────────────────────────────
    const countEl = document.getElementById('transcriptSegCount');
    if (countEl) countEl.textContent = `${_cachedScenes.length} scene${_cachedScenes.length !== 1 ? 's' : ''}`;

    const previewEl = document.getElementById('scenePromptsPreview');
    const listEl    = document.getElementById('scenePromptsList');
    if (previewEl && listEl) {
      previewEl.style.display = 'block';
      listEl.innerHTML = _cachedScenes.map((s, i) => {
        const ts   = `${_fmtTimestamp(s.start || 0)} – ${_fmtTimestamp(s.end || (s.start || 0) + (s.duration || 6))}`;
        const prompt = (s.prompt || '').slice(0, 130);
        return `<div style="padding:4px 0;border-bottom:1px solid var(--border)">
          <span style="color:var(--accent2);font-weight:600">${i + 1}.</span>
          <span style="color:var(--accent2)">&nbsp;[${ts}]&nbsp;</span>
          <span style="color:var(--text3)">${(s.text || '').slice(0, 55)}${(s.text||'').length > 55 ? '…' : ''}</span>
          <div style="color:var(--text2);margin-top:2px;padding-left:14px;font-size:10px">${prompt}${(s.prompt||'').length > 130 ? '…' : ''}</div>
        </div>`;
      }).join('');
    }

    toast(`Transcript ready — ${_cachedScenes.length} scenes`, 'success');

  } catch (e) {
    toast(e.message, 'error');
  } finally {
    _restoreBtn();
  }
}


// Update HF-token visibility when model changes
document.getElementById('aiModel')?.addEventListener('change', _updateHfTokenVisibility);

// ── Init AI method UI on load ─────────────────────────────────
_updateAiMethodUI();
_loadHfToken();

// ── Init ───────────────────────────────────────────────────────
enableControls(false);
updateStats();
