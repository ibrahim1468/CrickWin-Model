from flask import Flask, request, jsonify
import joblib
import os
import gdown
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Google Drive file ID and model path
FILE_ID = "1k1YdOdPzg9eyGUObTSDyFwNxsz9Urp7m"
MODEL_PATH = "live_probability_model_comp.joblib"

def load_model():
    """Load the machine learning model from Google Drive if not already present."""
    try:
        if not os.path.exists(MODEL_PATH):
            logger.info("Downloading model from Google Drive...")
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
        logger.info("Loading model...")
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully!")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

# Load the model at startup
try:
    model = load_model()
except Exception as e:
    logger.error("Failed to initialize model. API will not start.")
    raise

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to make predictions using the loaded model."""
    try:
        # Get JSON data from the request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Assuming the model expects a list of features (modify based on your model's input)
        features = data.get('features')
        if not features:
            return jsonify({"error": "Missing 'features' in input data"}), 400

        # Make prediction
        prediction = model.predict([features]).tolist()  # Convert to list for JSON serialization
        probability = model.predict_proba([features]).tolist() if hasattr(model, 'predict_proba') else None

        # Prepare response
        response = {
            "prediction": prediction,
            "probability": probability
        }
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model_loaded": model is not None}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)