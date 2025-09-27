import flask
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
import cv2
import mediapipe as mp
import base64
from io import BytesIO

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for browser requests

# Paths to model and encoder
MODEL_NAME = '/Users/ashishsingh/Desktop/Sign/models/best_model.p'
LABEL_ENCODER_NAME = '/Users/ashishsingh/Desktop/Sign/models/labels.p'

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    min_detection_confidence=0.3
)

# Load model and label encoder
def load_model_and_encoder():
    try:
        # Load model
        with open(MODEL_NAME, 'rb') as f:
            model_data = pickle.load(f)
        
        # Check if model_data is a dictionary and extract the model
        if isinstance(model_data, dict):
            logger.info("Loaded model is a dictionary. Attempting to extract 'model' key.")
            if 'model' in model_data:
                model = model_data['model']
            else:
                raise ValueError("Dictionary loaded from model file does not contain 'model' key.")
        else:
            model = model_data
        
        # Verify model has predict method
        if not hasattr(model, 'predict'):
            raise ValueError(f"Loaded model of type {type(model)} does not have a 'predict' method. Ensure it's a scikit-learn model.")
        
        # Load label encoder
        with open(LABEL_ENCODER_NAME, 'rb') as f:
            le = pickle.load(f)
        
        logger.info("Model and label encoder loaded successfully.")
        return model, le
    except Exception as e:
        logger.error(f"Error loading model or encoder: {e}")
        raise

try:
    model, le = load_model_and_encoder()
except Exception as e:
    logger.error(f"Failed to initialize model: {e}")
    raise

@app.route('/')
def index():
    logger.info("Root endpoint accessed")
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    logger.info("Favicon endpoint accessed")
    return '', 204

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        mode = data.get('mode', 'numbers')
        features = np.array(data['features']).reshape(1, -1)
        logger.info(f"Received mode: {mode}, features shape: {features.shape}, features: {data['features']}")
        
        if mode != 'numbers':
            return jsonify({'error': f"Mode '{mode}' not supported. Only 'numbers' is supported."}), 400
        
        if features.shape[1] != 84:
            return jsonify({'error': f"Expected 84 features, got {features.shape[1]}"}), 400
        
        prediction = model.predict(features)[0]
        prediction_label = le.inverse_transform([prediction])[0]
        confidence = model.predict_proba(features)[0].max()
        logger.info(f"Prediction: {prediction_label}, Confidence: {confidence}")
        return jsonify({'prediction': prediction_label, 'confidence': float(confidence)})
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict_image', methods=['POST'])
def predict_image():
    try:
        data = request.get_json()
        mode = data.get('mode', 'numbers')
        base64_image = data['image']
        
        if mode != 'numbers':
            return jsonify({'error': f"Mode '{mode}' not supported. Only 'numbers' is supported."}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(base64_image)
        image_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Extract features using MediaPipe
        results = hands.process(img_rgb)
        if not results.multi_hand_landmarks:
            return jsonify({'error': 'No hand detected'}), 400
        
        features = [0] * 84  # 2 hands * 21 landmarks * 2 coords
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
            x_ = [landmark.x for landmark in hand_landmarks.landmark]
            y_ = [landmark.y for landmark in hand_landmarks.landmark]
            min_x = min(x_)
            min_y = min(y_)
            hand_data = []
            for landmark in hand_landmarks.landmark:
                hand_data.append(landmark.x - min_x)
                hand_data.append(landmark.y - min_y)
            start_idx = i * 42
            features[start_idx:start_idx + 42] = hand_data
        
        # Predict using model
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]
        prediction_label = le.inverse_transform([prediction])[0]
        confidence = model.predict_proba(features_array)[0].max()
        logger.info(f"Prediction: {prediction_label}, Confidence: {confidence}")
        return jsonify({'prediction': prediction_label, 'confidence': float(confidence)})
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9800)