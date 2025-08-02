from flask import Flask, render_template, request, jsonify
import os
import base64
import re
import io
from pathlib import Path
import sys
from PIL import Image  # ensure PIL is imported

# Ensure project root is on sys.path so we can import processing_data
project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

# Import your ASLModel factory and the preprocess function
from webapp.asl_model import load_model
from processing_data import preprocess

app = Flask(
    __name__, static_folder="static", template_folder="templates"
)

# Load a ready-to-use ASLModel instance at startup
model = load_model()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get and decode base64 image from client
        data = request.get_json()
        img_data = re.sub(r"^data:image/.+;base64,", "", data.get("image", ""))
        img_bytes = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Preprocess image → landmark features
        features = preprocess(img)

        # Check if valid landmarks were found
        if not features or sum(features) == 0.0:
            return jsonify({"prediction": "No hand detected"})

        # Run prediction
        label = model.predict(img)  # or model.predict_from_landmarks(features)
        return jsonify({"prediction": label})

    except Exception as e:
        app.logger.error("❌ Prediction Error: %s", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
