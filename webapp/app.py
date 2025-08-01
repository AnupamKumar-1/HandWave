from flask import Flask, render_template, request, jsonify
import base64
import re
import io
from PIL import Image
import asl_model
from processing_data import preprocess

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load model and label map once at startup
model = asl_model.load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get and decode base64 image from client
        data = request.get_json()
        img_data = re.sub(r'^data:image/.+;base64,', '', data['image'])
        img_bytes = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Preprocess image → landmark features
        features = preprocess(img)

        # Check if valid landmarks were found
        if not features or sum(features) == 0.0:
            return jsonify({'prediction': "No hand detected"})

        # Run prediction
        label = model.predict_from_landmarks(features)
        return jsonify({'prediction': label})

    except Exception as e:
        print("❌ Prediction Error:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)