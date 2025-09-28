import streamlit as st
import pickle
import torch
import cv2
import numpy as np
from PIL import Image
from streamlit_option_menu import option_menu

# ===============================
# Model Management
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

@st.cache_resource
def load_model(model_path, label_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    with open(label_path, "rb") as f:
        labels = pickle.load(f)
    return model, labels

def preprocess_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (128, 128))  # adjust based on your training
    img = img / 255.0
    img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return img_tensor

def predict(model, labels, image):
    img_tensor = preprocess_image(image)
    with torch.no_grad():
        preds = model(img_tensor)
        probs = torch.nn.functional.softmax(preds[0], dim=0)
        conf, pred_class = torch.max(probs, dim=0)
    return labels[pred_class.item()], conf.item()

# ===============================
# Custom CSS
# ===============================
st.markdown("""
    <style>
        .main { background-color: #f9fafb; }
        .stButton>button {
            background: linear-gradient(90deg, #4f46e5, #3b82f6);
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 0.6em 1.2em;
        }
        .prediction-card {
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
            text-align: center;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# ===============================
# Sidebar Navigation
# ===============================
with st.sidebar:
    choice = option_menu(
        "SignSpeak - Prediction",
        ["Characters (A-Z)", "Numbers (0-9)", "Characters + Numbers", "Words"],
        icons=["alphabet", "123", "layers", "chat-square-text"],
        menu_icon="cast",
        default_index=0,
    )
    mode = st.radio("Choose Input Mode", ["Upload Image", "Camera Snapshot", "Live Webcam"])

# ===============================
# Main UI
# ===============================
st.title("🤟 SignSpeak - Sign Language Prediction")

model, labels = load_model(MODEL_PATHS[choice]["model"], MODEL_PATHS[choice]["labels"])

# -------- Upload Image ----------
if mode == "Upload Image":
    upload = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if upload:
        image = Image.open(upload)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        label, confidence = predict(model, labels, image)
        st.markdown(f"""
            <div class="prediction-card">
                <h3>Prediction: <b>{label}</b></h3>
                <p>Confidence: {confidence:.2%}</p>
            </div>
        """, unsafe_allow_html=True)

# -------- Camera Snapshot ----------
elif mode == "Camera Snapshot":
    camera = st.camera_input("Capture from Camera")
    if camera:
        image = Image.open(camera)
        st.image(image, caption="Captured Image", use_column_width=True)
        label, confidence = predict(model, labels, image)
        st.markdown(f"""
            <div class="prediction-card">
                <h3>Prediction: <b>{label}</b></h3>
                <p>Confidence: {confidence:.2%}</p>
            </div>
        """, unsafe_allow_html=True)

# -------- Live Webcam ----------
elif mode == "Live Webcam":
    st.write("🎥 Live Webcam Stream (Press 'Stop' to end)")
    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to access webcam.")
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)

        label, confidence = predict(model, labels, img_pil)

        cv2.putText(frame_rgb, f"{label} ({confidence:.2%})", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        FRAME_WINDOW.image(frame_rgb)

    cap.release()
