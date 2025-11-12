from flask import Flask, request, jsonify
from flask_cors import CORS
from model_loader import get_prediction_and_remedy 
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the Flask application
app = Flask(__name__)
CORS(app) 

# --- FIX for the 404 "Not Found" error on the root URL ---
@app.route('/', methods=['GET'])
def home():
    return "Plant Disease API Server is running. Use /api/detect (POST method) to analyze images.", 200

# THIS IS THE CORRECT ENDPOINT ADDRESS: /api/detect (Must match frontend JS)
@app.route('/api/detect', methods=['POST'])
def detect_disease():
    """
    Handles image upload from the front-end and passes it to the PyTorch model.
    """
    if 'file' not in request.files:
        return jsonify({'disease': 'API Error', 'confidence': 0.0, 'remedy': 'Upload payload missing file.'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'disease': 'API Error', 'confidence': 0.0, 'remedy': 'No selected file in upload.'}), 400

    if file:
        try:
            image_bytes = file.read()
            logging.info("Calling model prediction function...")
            
            # --- MODEL PREDICTION CALL ---
            result = get_prediction_and_remedy(image_bytes)
            
            return jsonify(result), 200

        except Exception as e:
            logging.error(f"Prediction Error in app.py: {e}")
            return jsonify({'disease': 'Internal Server Error', 'confidence': 0.0, 'remedy': 'Backend failed to process the request.'}), 500

if __name__ == '__main__':
    logging.info("Flask server starting on port 5000...")
    app.run(host='0.0.0.0', port=5000)