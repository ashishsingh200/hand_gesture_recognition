import os
import numpy as np
import joblib
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import warnings
import skfuzzy as fuzz
import json
from log import get_logger
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from av import VideoFrame

# Initialize logger
logger = get_logger('app')

# === Suppress warnings/logs ===
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# === Config ===
DATA_DIR = "/Users/ashishsingh/Desktop/words/data/small_words"
MODEL_DIR = "/Users/ashishsingh/Desktop/words/src/models"

SEQ_LEN = 100
FEATURE_DIM = 258
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Device selected: {DEVICE}")

# === Custom CSS for eye-catching UI ===
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .stSelectbox {
        background-color: #e6f3ff;
        border-radius: 5px;
        padding: 10px;
    }
    .stSelectbox label {
        font-weight: bold;
        color: #1e3a8a;
    }
    h1 {
        color: #1e3a8a;
        font-family: 'Arial', sans-serif;
        text-align: center;
    }
    .stMarkdown {
        font-size: 18px;
        color: #333;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# === Load label encoder ===
try:
    classes = np.load(os.path.join(DATA_DIR, "label_encoder.npy"))
    logger.info(f"Successfully loaded label_encoder.npy with {len(classes)} classes")
except FileNotFoundError:
    logger.error("label_encoder.npy not found. Please run the preprocessing script first.")
    st.error("label_encoder.npy not found. Please run the preprocessing script first.")
    st.stop()

# === Load best model name ===
try:
    with open(os.path.join(MODEL_DIR, "best_model_name.txt")) as f:
        best_name = f.read().strip()
    logger.info(f"Using best model: {best_name}")
except FileNotFoundError:
    logger.error("best_model_name.txt not found. Please run the training script first.")
    st.error("best_model_name.txt not found. Please run the training script first.")
    st.stop()

# === Define NN models ===
class MLPModel(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM*SEQ_LEN, num_classes=len(classes)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, hidden_size=128, num_classes=len(classes)):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(128, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = x.transpose(1, 2)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

class TransformerModel(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, num_classes=len(classes), hidden=256, layers=2, heads=4):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.fc = nn.Linear(hidden, num_classes)
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x.mean(dim=1))

class BidirectionalLSTMModel(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, hidden_size=128, num_layers=2, dropout=0.3, num_classes=len(classes)):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h_forward = h[-2, :, :]
        h_backward = h[-1, :, :]
        h_combined = torch.cat((h_forward, h_backward), dim=1)
        return self.fc(h_combined)

class CNNBiLSTMModel(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, conv_channels=128, lstm_hidden=128, num_layers=2, dropout=0.3, num_classes=len(classes)):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, conv_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(conv_channels, lstm_hidden, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = x.transpose(1, 2)
        _, (h, _) = self.lstm(x)
        h_forward = h[-2, :, :]
        h_backward = h[-1, :, :]
        h_combined = torch.cat((h_forward, h_backward), dim=1)
        return self.fc(h_combined)

class CNNGRUModel(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, conv_channels=128, gru_hidden=128, num_layers=2, dropout=0.3, num_classes=len(classes)):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, conv_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(conv_channels, gru_hidden, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(gru_hidden, num_classes)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = x.transpose(1, 2)
        _, h = self.gru(x)
        return self.fc(h[-1])

# === Load the best model ===
models_dict = {}
model = None
try:
    if best_name == "ensemble":
        ensemble_dict = torch.load(os.path.join(MODEL_DIR, "ensemble_models.pt"), map_location=DEVICE)
        with open(os.path.join(MODEL_DIR, "ensemble_model_names.txt")) as f:
            ensemble_names = [line.strip() for line in f.readlines()]
        logger.info(f"Loading ensemble models: {ensemble_names}")
        for name in ensemble_names:
            params = {}
            params_file = None
            if name == "bi_lstm":
                params_file = os.path.join(MODEL_DIR, "bi_lstm_params.json")
            elif name == "cnn_bilstm":
                params_file = os.path.join(MODEL_DIR, "cnn_bilstm_params.json")
            if params_file and os.path.exists(params_file):
                with open(params_file) as f:
                    params = json.load(f)
                logger.debug(f"Loaded parameters for {name}: {params}")
            if name == "mlp":
                sub_model = MLPModel()
            elif name == "cnn_lstm":
                sub_model = CNNLSTMModel()
            elif name == "transformer":
                sub_model = TransformerModel()
            elif name == "bi_lstm":
                sub_model = BidirectionalLSTMModel(hidden_size=params.get("hidden_size", 128), num_layers=params.get("num_layers", 2), num_classes=len(classes))
            elif name == "cnn_bilstm":
                sub_model = CNNBiLSTMModel(conv_channels=params.get("conv_channels", 128), lstm_hidden=params.get("lstm_hidden", 128),
                                           num_layers=params.get("num_layers", 2), num_classes=len(classes))
            elif name == "cnn_gru":
                sub_model = CNNGRUModel()
            else:
                logger.warning(f"Skipping unknown ensemble model {name}")
                continue
            sub_model.load_state_dict(ensemble_dict[name])
            sub_model.to(DEVICE)
            sub_model.eval()
            models_dict[name] = sub_model
        logger.info(f"Loaded ensemble of {len(models_dict)} models: {list(models_dict.keys())}")
    elif best_name == "mlp":
        model = MLPModel()
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pt"), map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        logger.info("Loaded MLP model")
    elif best_name == "cnn_lstm":
        model = CNNLSTMModel()
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pt"), map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        logger.info("Loaded CNN-LSTM model")
    elif best_name == "transformer":
        model = TransformerModel()
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pt"), map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        logger.info("Loaded Transformer model")
    elif best_name == "bi_lstm":
        params_file = os.path.join(MODEL_DIR, "bi_lstm_params.json")
        if not os.path.exists(params_file):
            logger.error("bi_lstm_params.json not found. Please run the training script again.")
            st.error("bi_lstm_params.json not found. Please run the training script again.")
            st.stop()
        with open(params_file) as f:
            best_params = json.load(f)
        logger.debug(f"Loaded parameters for bi_lstm: {best_params}")
        model = BidirectionalLSTMModel(hidden_size=best_params["hidden_size"], num_layers=best_params["num_layers"], num_classes=len(classes))
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pt"), map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        logger.info("Loaded Bidirectional LSTM model")
    elif best_name == "cnn_bilstm":
        params_file = os.path.join(MODEL_DIR, "cnn_bilstm_params.json")
        if not os.path.exists(params_file):
            logger.error("cnn_bilstm_params.json not found. Please run the training script again.")
            st.error("cnn_bilstm_params.json not found. Please run the training script again.")
            st.stop()
        with open(params_file) as f:
            best_params = json.load(f)
        logger.debug(f"Loaded parameters for cnn_bilstm: {best_params}")
        model = CNNBiLSTMModel(conv_channels=best_params["conv_channels"], lstm_hidden=best_params["lstm_hidden"], 
                               num_layers=best_params["num_layers"], num_classes=len(classes))
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pt"), map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        logger.info("Loaded CNN-BiLSTM model")
    elif best_name == "cnn_gru":
        model = CNNGRUModel()
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pt"), map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        logger.info("Loaded CNN-GRU model")
    elif best_name in ["hmm", "fcm"]:
        model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
        logger.info(f"Loaded {best_name.upper()} model")
    else:
        logger.error(f"Unknown model type '{best_name}'. Check best_model_name.txt.")
        st.error(f"Unknown model type '{best_name}'. Check best_model_name.txt.")
        st.stop()

except (FileNotFoundError, ValueError, KeyError) as e:
    logger.error(f"Failed to load model for inference. Reason: {e}")
    st.error(f"Failed to load model for inference. Reason: {e}")
    st.stop()

# === Mediapipe setup ===
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
logger.info("Mediapipe setup completed")

def extract_landmarks(results):
    pose = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            pose.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        pose.extend([0] * 132)
    lh = []
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            lh.extend([lm.x, lm.y, lm.z])
    else:
        lh.extend([0] * 63)
    rh = []
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            rh.extend([lm.x, lm.y, lm.z])
    else:
        rh.extend([0] * 63)
    landmarks = np.array(pose + lh + rh)
    if np.all(landmarks == 0):
        logger.warning("No landmarks detected in this frame")
    else:
        logger.debug("Landmarks extracted successfully")
    return landmarks

def get_hand_bounding_box(landmarks, img_shape, padding=20):
    """Compute bounding box for hand landmarks with padding."""
    if not landmarks:
        return None
    h, w, _ = img_shape
    x_coords = [lm.x * w for lm in landmarks.landmark]
    y_coords = [lm.y * h for lm in landmarks.landmark]
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)
    return (x_min, y_min, x_max, y_max)

def predict(sequence):
    pred_class, confidence = None, None
    seq_input = np.expand_dims(sequence, axis=0).astype(np.float32)
    if best_name == "ensemble":
        ensemble_probs = np.zeros(len(classes))
        for sub_model in models_dict.values():
            X_tensor = torch.tensor(seq_input, dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                probs = torch.softmax(sub_model(X_tensor), dim=1)[0].cpu().numpy()
            ensemble_probs += probs / len(models_dict)
        pred_idx = np.argmax(ensemble_probs)
        pred_class = classes[pred_idx]
        confidence = ensemble_probs[pred_idx]
    elif best_name in ["mlp", "cnn_lstm", "transformer", "bi_lstm", "cnn_bilstm", "cnn_gru"]:
        X_tensor = torch.tensor(seq_input, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(X_tensor), dim=1)[0].cpu().numpy()
        pred_idx = np.argmax(probs)
        pred_class = classes[pred_idx]
        confidence = probs[pred_idx]
    elif best_name == "hmm":
        scores = {lbl: mdl.score(seq_input[0]) for lbl, mdl in model.items()}
        if scores:
            pred_idx = max(scores, key=scores.get)
            pred_class = classes[pred_idx]
            all_scores = np.array(list(scores.values()))
            confidence = np.exp(scores[pred_idx]) / np.sum(np.exp(all_scores))
        else:
            pred_class = "Undetermined"
            confidence = 0.0
            logger.warning("No scores computed for HMM prediction")
    elif best_name == "fcm":
        cntr, clf = model
        X_flat = seq_input.reshape(seq_input.shape[0], -1).T
        try:
            preds_u = fuzz.cluster.cmeans_predict(X_flat, cntr, m=2, error=0.005, maxiter=1000)
            u_val = preds_u[1]
            pred_idx = clf.predict(u_val.T)[0]
            pred_class = classes[pred_idx]
            confidence = 0.8
        except ValueError as ve:
            logger.error(f"ValueError in FCM prediction: {ve}")
            pred_class = "Undetermined"
            confidence = 0.0
    return pred_class, confidence

class VideoProcessor:
    def __init__(self):
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.sequence = []
        self.pred_class = None
        self.confidence = None
        logger.info("VideoProcessor initialized")

    def recv(self, frame: VideoFrame) -> VideoFrame:
        try:
            img = frame.to_ndarray(format="bgr24")
            logger.debug(f"Received frame with shape: {img.shape}")
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(rgb)
            keypoints = extract_landmarks(results)
            self.sequence.append(keypoints)
            if len(self.sequence) > SEQ_LEN:
                self.sequence.pop(0)
                logger.debug(f"Sequence trimmed to length {SEQ_LEN}")
            
            # Draw hand landmarks
            mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                     landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                                     connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2))
            mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                     landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
                                     connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 200), thickness=2))

            if len(self.sequence) == SEQ_LEN:
                logger.debug("Sequence ready for prediction")
                self.pred_class, self.confidence = predict(self.sequence)
                if self.pred_class is not None and self.confidence is not None:
                    logger.debug(f"Predicted: {self.pred_class} with confidence {self.confidence:.2f}")
                else:
                    logger.warning("Prediction failed to produce a class or confidence")

            if self.pred_class is not None and self.pred_class != "Undetermined":
                # Draw bounding boxes around hands
                lh_box = get_hand_bounding_box(results.left_hand_landmarks, img.shape)
                rh_box = get_hand_bounding_box(results.right_hand_landmarks, img.shape)

                if lh_box:
                    x_min, y_min, x_max, y_max = lh_box
                    # Semi-transparent green box for left hand
                    overlay = img.copy()
                    cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 0), -1)
                    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 200, 0), 2)
                    cv2.putText(img, self.pred_class, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                if rh_box:
                    x_min, y_min, x_max, y_max = rh_box
                    # Semi-transparent blue box for right hand
                    overlay = img.copy()
                    cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 0, 255), -1)
                    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 200), 2)
                    cv2.putText(img, self.pred_class, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            return VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            logger.error(f"Video processing error: {e}", exc_info=True)
            return frame

# Streamlit app
st.title("✋ Sign Language Recognition")
st.markdown("Select your camera and perform sign language gestures to see real-time predictions!", unsafe_allow_html=True)

camera_choice = st.selectbox("Choose Camera", ["Front", "Back"], help="Select front or back camera for gesture recognition")
facing_mode = "user" if camera_choice == "Front" else "environment"

logger.info(f"Selected camera facing mode: {facing_mode}")

webrtc_streamer(
    key="sign_language_inference",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": {
            "facingMode": {"ideal": facing_mode},
            "width": {"ideal": 1280},
            "height": {"ideal": 720}
        },
        "audio": False
    },
    video_processor_factory=VideoProcessor,
)

st.markdown("**Instructions**: Position your hands clearly in the camera view. Predictions will appear on boxes around your hands.", unsafe_allow_html=True)

logger.info("Streamlit app running")