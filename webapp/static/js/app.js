const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const predictionDiv = document.getElementById('prediction');

let camera = null;
let predictionCooldown = false;
let currentPrediction = '';

// --- MediaPipe Hands setup ---
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

// Start camera + processing
document.getElementById('start-btn').onclick = async () => {
  camera = new Camera(video, {
    onFrame: async () => {
      await hands.send({ image: video });
    },
    width: 640,
    height: 480
  });
  camera.start();
};

// Stop camera
document.getElementById('stop-btn').onclick = () => {
  if (camera) camera.stop();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  predictionDiv.innerText = 'Prediction: ...';
  currentPrediction = '';
};

// Handle MediaPipe results
function onResults(results) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    const landmarks = results.multiHandLandmarks[0];

    drawConnectors(ctx, landmarks, HAND_CONNECTIONS, { color: '#00FF00', lineWidth: 3 });
    drawLandmarks(ctx, landmarks, { color: '#FF0000', lineWidth: 2 });

    // Draw prediction text above wrist
    const wrist = landmarks[0]; // landmark index 0 is wrist
    const x = wrist.x * canvas.width;
    const y = wrist.y * canvas.height;
    ctx.font = 'bold 24px sans-serif';
    ctx.fillStyle = 'blue';
    ctx.fillText(currentPrediction, x + 10, y - 10);

    // Send prediction every 300ms
    if (!predictionCooldown) {
      predictionCooldown = true;
      sendFrameForPrediction();
      setTimeout(() => predictionCooldown = false, 300);
    }
  }
}

// Send frame to backend for prediction
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
    currentPrediction = data.prediction || '';
    predictionDiv.innerText = `Prediction: ${currentPrediction}`;
  })
  .catch(err => console.error('Prediction error:', err));
}
