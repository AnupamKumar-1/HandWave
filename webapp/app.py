from flask import Flask, render_template, request, jsonify
import os
import base64
import re
import io
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

from webapp.asl_model import load_model, ASLModel
from webapp.processing_data import preprocess

app = Flask(
    __name__, static_folder="static", template_folder="templates"
)

# Load model and label map once at startup
model_obj, label_map = load_model()
model = ASLModel(model_obj, label_map)

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
        label = model.predict_from_landmarks(features)
        return jsonify({"prediction": label})

    except Exception as e:
        app.logger.error("❌ Prediction Error: %s", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

