import streamlit as st
import pickle
import cv2
import numpy as np
from PIL import Image

# ===============================
# Model Paths
# ===============================
MODEL_PATHS = {
    "Characters (A-Z)": {
        "model": "characters/models/best_model.p",
        "labels": "characters/models/labels.p"
    },
    "Numbers (0-9)": {
        "model": "Numbers/models/best_model.p",
        "labels": "Numbers/models/labels.p"
    },
    "Characters + Numbers": {
        "model": "character and numbers/models/best_model.p",
        "labels": "character and numbers/models/labels.p"
    },
    "Words": {
        "model": "words/models/best_model.p",
        "labels": "words/models/labels.p"
    }
}

# ===============================
# Load Models
# ===============================
@st.cache_resource
def load_model(model_path, label_path):
    with open(model_path, "rb") as f:
        data = pickle.load(f)
        model = data["model"]

    with open(label_path, "rb") as f:
        labels = pickle.load(f)

    return model, labels

# ===============================
# Prediction Function
# ===============================
def predict(model, labels, image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (128, 128))   # match training size
    img = img.flatten().reshape(1, -1) / 255.0

    probs = model.predict_proba(img)[0]
    pred_class = np.argmax(probs)
    confidence = probs[pred_class]

    return labels.inverse_transform([pred_class])[0], confidence

# ===============================
# Custom CSS
# ===============================
st.markdown("""
    <style>
        .prediction-card {
            background: white;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
            text-align: center;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ===============================
# UI
# ===============================
st.title("🤟 SignSpeak - Live Webcam Prediction")

# Select model
choice = st.selectbox(
    "Choose Model",
    ["Characters (A-Z)", "Numbers (0-9)", "Characters + Numbers", "Words"]
)

# Load model + labels
model, labels = load_model(MODEL_PATHS[choice]["model"], MODEL_PATHS[choice]["labels"])

# Webcam live stream
st.write("🎥 Enable webcam to start prediction")
run = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image([])
cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("⚠️ Failed to access webcam.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)

    label, confidence = predict(model, labels, img_pil)

    cv2.putText(frame_rgb, f"{label} ({confidence:.2%})", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    FRAME_WINDOW.image(frame_rgb)

cap.release()
