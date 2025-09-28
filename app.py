import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import cv2
import numpy as np
import pickle
import mediapipe as mp
import av
import os

# Custom CSS for a clean UI
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f4f8;
        font-family: 'Arial', sans-serif;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .stSelectbox > div {
        background-color: white;
        border-radius: 4px;
        padding: 5px;
    }
    h1, h2 {
        color: #333;
    }
    .prediction {
        font-size: 24px;
        font-weight: bold;
        color: #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

# Model directories (relative to repo root)
COMBINED_MODEL_DIR = "character and numbers/models"
CHARS_MODEL_DIR = "characters/models"
NUMS_MODEL_DIR = "Numbers/models"

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,  # Real-time processing
    max_num_hands=2,
    min_detection_confidence=0.3
)

# Cache model loading
@st.cache_resource
def load_model(model_dir):
    try:
        model_path = os.path.join(model_dir, 'best_model.p')
        labels_path = os.path.join(model_dir, 'labels.p')
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            model = data['model']  # Match save_best_model structure
        with open(labels_path, 'rb') as f:
            le = pickle.load(f)
        return model, le
    except Exception as e:
        st.error(f"Error loading model from {model_dir}: {str(e)}")
        return None, None

# Load all models
combined_model, combined_le = load_model(COMBINED_MODEL_DIR)
chars_model, chars_le = load_model(CHARS_MODEL_DIR)
nums_model, nums_le = load_model(NUMS_MODEL_DIR)

# Preprocessing function with MediaPipe hand landmarks
def preprocess_image(image):
    try:
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Process with MediaPipe
        results = hands.process(img_rgb)
        if not results.multi_hand_landmarks:
            return None, None, None
        
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
        
        # Compute bounding box for hand(s)
        h, w = image.shape[:2]
        bboxes = []
        for hand_landmarks in results.multi_hand_landmarks[:2]:
            x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            # Expand box slightly
            padding = 20
            x_min = max(0, x_min - padding)
            x_max = min(w, x_max + padding)
            y_min = max(0, y_min - padding)
            y_max = min(h, y_max + padding)
            bboxes.append((x_min, y_min, x_max, y_max))
        
        # Merge bboxes if two hands are close
        if len(bboxes) == 2:
            x1_1, y1_1, x2_1, y2_1 = bboxes[0]
            x1_2, y1_2, x2_2, y2_2 = bboxes[1]
            # Check if boxes overlap or are close
            if (abs(x1_1 - x1_2) < 50 and abs(y1_1 - y1_2) < 50) or \
               (abs(x2_1 - x2_2) < 50 and abs(y2_1 - y2_2) < 50):
                x_min = min(x1_1, x1_2)
                y_min = min(y1_1, y1_2)
                x_max = max(x2_1, x2_2)
                y_max = max(y2_1, y2_2)
                bboxes = [(x_min, y_min, x_max, y_max)]
        
        return np.array(features).reshape(1, -1), bboxes, None
    except Exception as e:
        st.error(f"Preprocessing error: {str(e)}")
        return None, None, None

# Prediction function
def predict(model, le, image, threshold=0.7):
    if model is None or le is None:
        return None, 0.0
    features, bboxes, _ = preprocess_image(image)
    if features is None:
        return None, 0.0
    try:
        probs = model.predict_proba(features)[0]
        max_prob = np.max(probs)
        if max_prob > threshold:
            pred = np.argmax(probs)
            label = le.inverse_transform([pred])[0]
            return str(label), max_prob
        else:
            return None, 0.0
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, 0.0

# Video Processor for stable predictions on dynamic bounding box
class VideoProcessor:
    def __init__(self, model, le, facing_mode):
        self.model = model
        self.le = le
        self.facing_mode = facing_mode
        self.frame_count = 0
        self.prev_label = None
        self.stable_count = 0
        self.stable_label = None
        self.stable_prob = 0.0

    def recv(self, frame):
        try:
            self.frame_count += 1
            img = frame.to_ndarray(format="bgr24")
            h, w = img.shape[:2]
            # Process every 2nd frame to improve performance
            if self.frame_count % 2 == 0:
                label, prob = predict(self.model, self.le, img)
                bboxes = preprocess_image(img)[1]  # Get bboxes
                # Stabilize predictions
                if label == self.prev_label and label is not None:
                    self.stable_count += 1
                    if self.stable_count >= 5:  # Require 5 consistent frames
                        self.stable_label = label
                        self.stable_prob = prob
                else:
                    self.stable_count = 0
                    self.prev_label = label
                    if label != self.stable_label:
                        self.stable_label = None
                        self.stable_prob = 0.0
            else:
                bboxes = preprocess_image(img)[1]  # Get bboxes for display
            
            # Draw bounding box(es)
            if bboxes:
                for bbox in bboxes:
                    x1, y1, x2, y2 = bbox
                    # Draw white bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1, lineType=cv2.LINE_AA)
                    # Draw prediction on primary box (first hand or merged)
                    if self.stable_label:
                        # Semi-transparent black rectangle for text background
                        overlay = img.copy()
                        rect_y = y2 - 40  # Position at bottom of box
                        cv2.rectangle(overlay, (x1, rect_y), (x2, y2), (0, 0, 0), -1)
                        alpha = 0.6
                        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
                        # Draw stable prediction text (large, bold, green, centered)
                        text = self.stable_label
                        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                        text_x = x1 + (x2 - x1 - text_size[0]) // 2  # Center in box
                        cv2.putText(img, text, (text_x, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            img = frame.to_ndarray(format="bgr24")
            return av.VideoFrame.from_ndarray(img, format="bgr24")

# Main app
st.title("Sign Language Predictor")
st.write("Predict numbers or characters (A-Z) in real-time using your webcam (laptop or mobile). Predictions appear on the bounding box around your hand(s).")

# Model and camera selection
st.header("Live Prediction")
model_type = st.selectbox("Select Model", ["Combined (Chars + Numbers)", "Characters Only", "Numbers Only"])
if model_type == "Combined (Chars + Numbers)":
    selected_model, selected_le = combined_model, combined_le
elif model_type == "Characters Only":
    selected_model, selected_le = chars_model, chars_le
else:
    selected_model, selected_le = nums_model, nums_le

camera_side = st.selectbox("Camera Facing", ["Front", "Back"])
facing_mode = "user" if camera_side == "Front" else "environment"

# RTC config for STUN
rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

def video_factory():
    return VideoProcessor(selected_model, selected_le, facing_mode)

# Start webcam with optimized settings
webrtc_ctx = webrtc_streamer(
    key="live",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_config,
    video_processor_factory=video_factory,
    media_stream_constraints={"video": {"facingMode": facing_mode, "width": 320, "height": 240}, "audio": False},
    async_processing=True,
)

# About section
st.header("About")
st.write("This app uses your trained models for real-time sign language recognition.")
st.write("Predictions appear on the bounding box around your hand(s) when stable for ~5 frames.")
st.write("Assumptions: Models expect 84 MediaPipe hand landmark features from webcam input.")
st.write("Troubleshooting: If predictions don't appear, ensure hands are visible and check browser console for errors.")