const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const predictionDiv = document.getElementById('prediction');

let camera = null;
let predictionCooldown = false;

let predictionBuffer = [];
const BUFFER_SIZE = 5;

let lastTime = performance.now();
let fps = 0;

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
  document.getElementById('prediction-box').style.display = 'block';

  camera = new Camera(video, {
    onFrame: async () => {
      await hands.send({ image: video });
    },
    width: 640,
    height: 480
  });

  camera.start();
};

document.getElementById('stop-btn').onclick = () => {
  if (camera) camera.stop();

  document.getElementById('prediction-box').style.display = 'none';

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  predictionDiv.innerText = 'Camera stopped';
  predictionBuffer = [];
};

function getStablePrediction() {
  if (predictionBuffer.length === 0) return "";

  const freq = {};
  predictionBuffer.forEach(p => {
    freq[p] = (freq[p] || 0) + 1;
  });

  return Object.keys(freq).reduce((a, b) => freq[a] > freq[b] ? a : b);
}

function onResults(results) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.save();

ctx.translate(canvas.width, 0);
ctx.scale(-1, 1);


ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

if (results.multiHandLandmarks?.length > 0) {
  const landmarks = results.multiHandLandmarks[0];

  drawConnectors(ctx, landmarks, HAND_CONNECTIONS, {
    color: '#00FF00',
    lineWidth: 3
  });

  drawLandmarks(ctx, landmarks, {
    color: '#FF0000',
    lineWidth: 2
  });
}

ctx.restore();
  const now = performance.now();
  fps = Math.round(1000 / (now - lastTime));
  lastTime = now;

  if (results.multiHandLandmarks?.length > 0) {

    if (!predictionCooldown) {
      predictionCooldown = true;
      sendFrameForPrediction();
      setTimeout(() => predictionCooldown = false, 300);
    }

    if (predictionBuffer.length > 0) {
      const stable = getStablePrediction();

      // canvas label
      ctx.font = 'bold 28px sans-serif';
      ctx.fillStyle = '#22c55e';
      ctx.fillText(stable, 20, canvas.height - 20);

      predictionDiv.innerHTML = `
        <div style="font-size:12px; color:#94a3b8;">Detected Letter</div>
        <div style="font-size:36px; font-weight:bold; color:#22c55e;">
          ${stable}
        </div>
      `;
    }

  } else {
    predictionDiv.innerText = "Show your hand ✋";
    predictionBuffer = [];

    ctx.font = '18px sans-serif';
    ctx.fillStyle = 'rgba(255,255,255,0.6)';
    ctx.fillText("No hand detected", 20, 30);
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
    if (predictionBuffer.length > BUFFER_SIZE) {
      predictionBuffer.shift();
    }
  })
  .catch(err => console.error('Prediction error:', err));
}