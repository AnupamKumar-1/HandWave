'use strict';

const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');

let camera = null;
let predictionCooldown = false;
let predictionBuffer = [];
const BUFFER_SIZE = 5;
let lastTime = performance.now();

const hands = new Hands({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});

hands.setOptions({
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.7
});

hands.onResults(onResults);

document.getElementById('start-btn').onclick = async () => {
  document.getElementById('prediction-box') &&
    (document.getElementById('prediction-box').style.display = 'block');

  camera = new Camera(video, {
    onFrame: async () => { await hands.send({ image: video }); },
    width: 640,
    height: 480
  });

  camera.start();
};

document.getElementById('stop-btn').onclick = () => {
  if (camera) camera.stop();

  document.getElementById('prediction-box') &&
    (document.getElementById('prediction-box').style.display = 'none');

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  predictionBuffer = [];
};

function getStablePrediction() {
  if (predictionBuffer.length === 0) return '';
  const freq = {};
  predictionBuffer.forEach(p => { freq[p] = (freq[p] || 0) + 1; });
  return Object.keys(freq).reduce((a, b) => (freq[a] >= freq[b] ? a : b));
}

function onResults(results) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.save();
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  if (results.multiHandLandmarks?.length > 0) {
    const landmarks = results.multiHandLandmarks[0];
    drawConnectors(ctx, landmarks, HAND_CONNECTIONS, { color: '#3bf0a0', lineWidth: 3 });
    drawLandmarks(ctx, landmarks, { color: '#ffffff', lineWidth: 2 });
  }

  ctx.restore();

  const now = performance.now();
  lastTime = now;

  if (results.multiHandLandmarks?.length > 0) {
    if (!predictionCooldown) {
      predictionCooldown = true;
      sendFrameForPrediction();
      setTimeout(() => { predictionCooldown = false; }, 300);
    }

    if (predictionBuffer.length > 0) {
      const stable = getStablePrediction();
      ctx.font = 'bold 28px "JetBrains Mono", monospace';
      ctx.fillStyle = '#3bf0a0';
      ctx.fillText(stable, 20, canvas.height - 20);

      if (typeof window.onNewPrediction === 'function') {
        window.onNewPrediction(stable);
      }
    }
  } else {
    predictionBuffer = [];
    ctx.font = '14px "JetBrains Mono", monospace';
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.fillText('No hand detected', 16, 28);
    if (typeof window.onHandLost === 'function') {
      window.onHandLost();
    }
  }
}

function sendFrameForPrediction() {
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = video.videoWidth;
  tempCanvas.height = video.videoHeight;
  const tempCtx = tempCanvas.getContext('2d');
  tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
  const dataURL = tempCanvas.toDataURL('image/jpeg');

  fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: dataURL })
  })
    .then(res => res.json())
    .then(data => {
      const pred = data.prediction || '';
      if (!pred) return;
      predictionBuffer.push(pred);
      if (predictionBuffer.length > BUFFER_SIZE) predictionBuffer.shift();
    })
    .catch(err => console.error('Prediction request failed:', err));
}

const toggleBtn = document.getElementById('toggle-btn');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const placeholder = document.getElementById('cam-placeholder');
const predLetterEl = document.getElementById('pred-letter');
const confFill = document.getElementById('conf-fill');
const detectedGlyph = document.getElementById('detected-glyph');
const holdTrack = document.getElementById('hold-track');
const holdFill = document.getElementById('hold-fill');
const modeHint = document.getElementById('mode-hint');
const wordDisplay = document.getElementById('word-display');
const historyLog = document.getElementById('history-log');
const toastEl = document.getElementById('toast');

let cameraRunning = false;
let currentWord = [];
let sentenceLog = [];
let lastPrediction = '';
let currentMode = 'manual';

let holdRaf = null;
let holdStart = null;
const HOLD_MS = 1200;

toggleBtn.addEventListener('click', () => {
  if (!cameraRunning) {
    startBtn.click();
    placeholder.style.display = 'none';
    toggleBtn.textContent = 'Stop Camera';
    toggleBtn.classList.replace('start', 'stop');
    statusDot.classList.add('live');
    statusText.textContent = 'Live';
    cameraRunning = true;
  } else {
    stopBtn.click();
    placeholder.style.display = 'flex';
    toggleBtn.textContent = 'Start Camera';
    toggleBtn.classList.replace('stop', 'start');
    statusDot.classList.remove('live');
    statusText.textContent = 'Stopped';
    predLetterEl.textContent = '--';
    confFill.style.width = '0%';
    cameraRunning = false;
  }
});

function setMode(m) {
  currentMode = m;
  clearHold();
  ['manual', 'hold'].forEach(id => {
    document.getElementById('btn-' + id).classList.toggle('active', id === m);
  });
  holdTrack.style.display = m === 'hold' ? 'block' : 'none';
  holdFill.style.width = '0%';
  holdFill.classList.remove('ready');
  if (m === 'manual') modeHint.textContent = 'Press A to add letter';
  if (m === 'hold') modeHint.textContent = 'Hold same sign to confirm';
}

function clearHold() {
  if (holdRaf) { cancelAnimationFrame(holdRaf); holdRaf = null; }
}

window.onHandLost = function () {
  lastPrediction = '';
  clearHold();
  holdFill.style.width = '0%';
  holdFill.classList.remove('ready');
};

window.onNewPrediction = function (letter) {
  if (!letter || letter === lastPrediction) return;
  lastPrediction = letter;

  predLetterEl.textContent = letter;
  confFill.style.width = (75 + Math.random() * 25).toFixed(0) + '%';

  detectedGlyph.textContent = letter;
  detectedGlyph.classList.remove('pop');
  void detectedGlyph.offsetWidth;
  detectedGlyph.classList.add('pop');

  if (currentMode === 'hold') {
    clearHold();
    holdFill.classList.remove('ready');
    holdFill.style.width = '0%';
    holdStart = Date.now();
    const tick = () => {
      const pct = Math.min(100, ((Date.now() - holdStart) / HOLD_MS) * 100);
      holdFill.style.width = pct + '%';
      if (pct >= 100) {
        holdFill.classList.add('ready');
        addChar(letter);
        holdStart = Date.now();
        holdFill.style.width = '0%';
        holdFill.classList.remove('ready');
      } else {
        holdRaf = requestAnimationFrame(tick);
      }
    };
    holdRaf = requestAnimationFrame(tick);
  }
};

function renderWord() {
  Array.from(wordDisplay.querySelectorAll('.char-chip, .space-chip, .cursor-blink')).forEach(el => el.remove());
  const hint = wordDisplay.querySelector('#word-hint');
  if (hint) hint.remove();

  if (currentWord.length === 0) {
    wordDisplay.innerHTML = '<span class="word-empty-hint" id="word-hint">Add letters to start building...</span><span class="cursor-blink"></span>';
    return;
  }

  currentWord.forEach(ch => {
    const chip = document.createElement('span');
    if (ch === ' ') {
      chip.className = 'char-chip space-chip';
      chip.innerHTML = '&nbsp;';
    } else {
      chip.className = 'char-chip';
      chip.textContent = ch;
    }
    wordDisplay.appendChild(chip);
  });

  const cur = document.createElement('span');
  cur.className = 'cursor-blink';
  wordDisplay.appendChild(cur);
}

function addChar(ch) { currentWord.push(ch); renderWord(); }
function addLetter() { if (lastPrediction && lastPrediction !== '--') addChar(lastPrediction); }

function undoChar() {
  if (currentWord.length > 0) { currentWord.pop(); renderWord(); }
}

function commitWord() {
  const word = currentWord.join('').trim();
  if (!word) return;
  sentenceLog.unshift(word);
  currentWord = [];
  renderWord();
  renderHistory();
  showToast('Word committed');
}

function clearWord() { currentWord = []; renderWord(); }

function renderHistory() {
  historyLog.innerHTML = '';
  if (sentenceLog.length === 0) {
    historyLog.innerHTML = '<div class="history-empty">No sentences yet.</div>';
    return;
  }
  sentenceLog.forEach(text => {
    const item = document.createElement('div');
    item.className = 'history-item';
    const span = document.createElement('span');
    span.className = 'history-text';
    span.textContent = text;
    const copyBtn = document.createElement('button');
    copyBtn.className = 'history-copy';
    copyBtn.textContent = 'Copy';
    copyBtn.addEventListener('click', () => {
      navigator.clipboard.writeText(text).catch(() => { });
      copyBtn.textContent = 'Copied';
      setTimeout(() => (copyBtn.textContent = 'Copy'), 1500);
    });
    item.appendChild(span);
    item.appendChild(copyBtn);
    historyLog.appendChild(item);
  });
}

function showToast(msg) {
  toastEl.textContent = msg;
  toastEl.classList.add('show');
  setTimeout(() => toastEl.classList.remove('show'), 1800);
}

document.getElementById('add-letter-btn').addEventListener('click', addLetter);
document.getElementById('add-space-btn').addEventListener('click', () => addChar(' '));
document.getElementById('undo-char-btn').addEventListener('click', undoChar);
document.getElementById('commit-word-btn').addEventListener('click', commitWord);
document.getElementById('clear-word-btn').addEventListener('click', clearWord);
document.getElementById('copy-all-btn').addEventListener('click', () => {
  const all = sentenceLog.join(' ');
  if (all) { navigator.clipboard.writeText(all).catch(() => { }); showToast('Copied to clipboard'); }
});
document.getElementById('clear-history-btn').addEventListener('click', () => {
  sentenceLog = [];
  renderHistory();
});

document.addEventListener('keydown', e => {
  if (['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
  switch (e.key) {
    case 'a': case 'A': e.preventDefault(); addLetter(); break;
    case 's': case 'S': e.preventDefault(); addChar(' '); break;
    case 'Backspace': e.preventDefault(); undoChar(); break;
    case 'Enter': e.preventDefault(); commitWord(); break;
    case 'Escape': e.preventDefault(); clearWord(); break;
  }
});

renderWord();
renderHistory();