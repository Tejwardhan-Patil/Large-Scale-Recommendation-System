import os
import joblib
import json
import numpy as np
from flask import Flask, request, jsonify
import logging
from logging.handlers import RotatingFileHandler
from sklearn.exceptions import NotFittedError

# Configuration variables
MODEL_PATH = os.getenv('MODEL_PATH', '/model.joblib')
HOST = os.getenv('HOST', '0.0.0.0')
PORT = os.getenv('PORT', 5000)
LOG_FILE = os.getenv('LOG_FILE', 'inference.log')
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
BACKUP_COUNT = 5

# Set up logging
def setup_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(LOG_FILE, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    return logger

logger = setup_logging()

# Load the model
def load_model():
    logger.info("Loading model from path: %s", MODEL_PATH)
    if not os.path.exists(MODEL_PATH):
        logger.error("Model file does not exist at path: %s", MODEL_PATH)
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.exception("Error loading model: %s", str(e))
        raise

model = load_model()

# Initialize the Flask application
app = Flask(__name__)

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    logger.info("Health check requested")
    return jsonify({"status": "healthy"}), 200

# Predict function
def perform_prediction(features):
    try:
        logger.info("Performing prediction for features: %s", features)
        prediction = model.predict(features)
        logger.info("Prediction successful: %s", prediction)
        return prediction
    except NotFittedError as e:
        logger.error("Model is not fitted: %s", str(e))
        raise
    except Exception as e:
        logger.exception("Error during prediction: %s", str(e))
        raise

# Preprocess the input features
def preprocess_input(data):
    logger.info("Preprocessing input data: %s", data)
    try:
        features = np.array(data['features']).reshape(1, -1)
        logger.info("Input data preprocessed: %s", features)
        return features
    except KeyError as e:
        logger.error("Missing 'features' in input data: %s", str(e))
        raise
    except Exception as e:
        logger.exception("Error during preprocessing: %s", str(e))
        raise

# Endpoint for model prediction
@app.route('/predict', methods=['POST'])
def predict():
    logger.info("Prediction request received")
    try:
        data = request.get_json(force=True)
        features = preprocess_input(data)
        prediction = perform_prediction(features)
        response = {
            'prediction': prediction.tolist()
        }
        logger.info("Prediction response: %s", response)
        return jsonify(response), 200
    except Exception as e:
        logger.error("Error in /predict endpoint: %s", str(e))
        response = {
            'error': str(e)
        }
        return jsonify(response), 500

# Error handler for invalid routes
@app.errorhandler(404)
def route_not_found(e):
    logger.warning("404 error: %s", request.url)
    response = {
        'error': 'Endpoint not found'
    }
    return jsonify(response), 404

# Error handler for internal server errors
@app.errorhandler(500)
def internal_error(e):
    logger.error("500 error: %s", str(e))
    response = {
        'error': 'Internal server error'
    }
    return jsonify(response), 500

# Configuration summary endpoint
@app.route('/config', methods=['GET'])
def get_config():
    config_info = {
        'MODEL_PATH': MODEL_PATH,
        'HOST': HOST,
        'PORT': PORT,
        'LOG_FILE': LOG_FILE
    }
    logger.info("Configuration requested: %s", config_info)
    return jsonify(config_info), 200

# Start the Flask app
if __name__ == '__main__':
    logger.info("Starting Flask app on host: %s, port: %s", HOST, PORT)
    try:
        app.run(host=HOST, port=int(PORT))
    except Exception as e:
        logger.exception("Error starting Flask app: %s", str(e))
        raise