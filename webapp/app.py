"""Flask entry point for the HandWave ASL recognition app."""

import base64
import io
import os
import re
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image

project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

from webapp.asl_model import load_model  # noqa: E402

app = Flask(__name__, static_folder="static", template_folder="templates")
model = load_model()


@app.route("/")
def home():
    """Serve the main UI."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Decode a base64 webcam frame, run inference, return the predicted letter."""
    try:
        data = request.get_json()
        raw_b64 = re.sub(r"^data:image/.+;base64,", "", data.get("image", ""))
        img_bytes = base64.b64decode(raw_b64)
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        label = model.predict(pil_image)
        return jsonify({"prediction": label})
    except Exception as e:
        app.logger.error("Prediction error: %s", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
