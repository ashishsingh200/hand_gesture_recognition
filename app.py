import streamlit as st
import pickle
import numpy as np
import os
from PIL import Image
import time
import cv2
import mediapipe as mp
import av # Required for streamlit_webrtc frame processing

# --- External Libraries for Live Stream (MUST BE INSTALLED) ---
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings

# --- 1. CONFIGURATION AND UTILITIES ---

# Initialize MediaPipe Hands and Drawing Utilities globally
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Define the relative paths for the models
MODEL_BASE_DIR = "models" 
MODEL_PATHS = {
    "Characters & Numbers": {
        "model": 'character and numbers/models/best_model.p',
        "labels": 'character and numbers/models/labels.p',
    },
    "Characters Only": {
        "model": 'characters/models/best_model.p',
        "labels":'characters/models/labels.p',
    },
    "Numbers Only": {
        "model": 'numbers/models/best_model.p',
        "labels": 'numbers/models/labels.p',
    },
    "Words": {
        "model": 'words/models/best_model',
        "labels": 'words/models/labels.p',
    },
}

# RTC configuration for webcam access
RTC_CONFIGURATION = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False}
)

@st.cache_resource
def load_model_assets(mode):
    """Loads the model and label encoder for the selected mode."""
    st.info(f"Attempting to load assets for: {mode}")
    paths = MODEL_PATHS.get(mode)
    
    if not paths:
        st.error(f"Configuration error: Paths for mode '{mode}' not found.")
        return None, None
    
    model_path = paths["model"]
    labels_path = paths["labels"]

    try:
        # Load Model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            model = model_data.get('model')
            
        # Load Labels
        with open(labels_path, 'rb') as f:
            le = pickle.load(f)
        
        st.success(f"Successfully loaded model for {mode}!")
        return model, le
    except FileNotFoundError as e:
        # Creating mock assets if the real files are not found (robust fallback)
        st.error(f"Model/Label file not found at: {e.filename}. Using mock assets. **Upload your model files!**")
        return create_mock_assets(mode)
    except Exception as e:
        st.error(f"An error occurred loading the model assets for {mode}: {e}")
        return create_mock_assets(mode)


def create_mock_assets(mode):
    """Creates mock model and label encoder for demonstration purposes."""
    class MockModel:
        def predict(self, X):
            return np.array([np.random.randint(0, len(self.classes))])
        def predict_proba(self, X):
            # Mock confidence
            probs = np.random.rand(1, len(self.classes))
            probs /= probs.sum(axis=1, keepdims=True)
            return probs

    class MockLabelEncoder:
        def __init__(self, classes):
            self.classes = classes
            
        def inverse_transform(self, y):
            return np.array([self.classes[i] for i in y])
    
    if "Numbers" in mode:
        classes = [str(i) for i in range(10)]
    elif "Characters" in mode:
        classes = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    elif mode == "Words":
        classes = ["HELLO", "YES", "NO", "THANK YOU"]
    else:
        classes = list('A0B1C2') # Mixed
        
    mock_model = MockModel()
    mock_model.classes = classes
    mock_le = MockLabelEncoder(classes)
    
    return mock_model, mock_le


# --- 2. CORE ML LOGIC (REAL IMPLEMENTATION) ---

def process_and_extract_features(image_rgb, hands_model):
    """
    Processes an image (BGR) to detect hand landmarks and extract 
    the 63-feature vector, and draws the landmarks.
    Returns: features (np.array), annotated_image (np.array), hand_detected (bool)
    """
    # Convert the BGR image to RGB for MediaPipe
    image_rgb.flags.writeable = False
    results = hands_model.process(image_rgb)
    image_rgb.flags.writeable = True

    h, w, c = image_rgb.shape
    features = np.zeros((1, 63), dtype=np.float32)
    hand_detected = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            # 1. Draw landmarks on the image
            mp_drawing.draw_landmarks(
                image_rgb,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            # 2. Extract and Normalize Features
            # Flatten the 21 landmarks (x, y, z) into a 63-element array
            
            # Find center of the hand (e.g., wrist) for normalization
            wrist_x = hand_landmarks.landmark[0].x
            wrist_y = hand_landmarks.landmark[0].y
            
            # Create a list to hold the 63 normalized coordinates
            feature_list = []
            
            for landmark in hand_landmarks.landmark:
                # Normalize all coordinates relative to the wrist (0, 0)
                # and append to the list
                feature_list.append(landmark.x - wrist_x)
                feature_list.append(landmark.y - wrist_y)
                # Note: Z is often used, but sometimes omitted or normalized differently. 
                # Assuming (x, y, z) for 63 features as standard.
                feature_list.append(landmark.z - hand_landmarks.landmark[0].z) 
                
            features = np.array([feature_list], dtype=np.float32)
            hand_detected = True
            break # Only process the first detected hand

    # Convert back to BGR for OpenCV display compatibility
    return features, image_rgb, hand_detected


def predict_sign(model, le, features):
    """Performs the prediction using the loaded model."""
    if model is None or le is None:
        return "Model Error", "0.00%", None

    try:
        # 1. Predict the label index
        prediction_index = model.predict(features)[0]
        
        # 2. Inverse transform the index to get the class name
        predicted_sign = le.inverse_transform([prediction_index])[0]

        # 3. Get probability/confidence
        if hasattr(model, 'predict_proba'):
             probabilities = model.predict_proba(features)[0]
             confidence = probabilities[prediction_index] * 100
        else:
             # Fallback confidence for models like SVC without probability=True
             confidence = 99.99
             
        return predicted_sign, f"{confidence:.2f}%", prediction_index
        
    except Exception as e:
        # st.error(f"Prediction Error: {e}") # Do not show in the video stream
        return "Error", "0.00%", None


# --- 3. VIDEO TRANSFORMER FOR LIVE STREAMING ---

class MLVideoTransformer(VideoTransformerBase):
    """
    Video Transformer for streamlit-webrtc. Processes frames in real-time.
    """
    def __init__(self, model, le):
        self.model = model
        self.le = le
        self.hands_model = hands # Global MediaPipe hands instance
        self.prediction_result = "START"

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert av.VideoFrame to numpy array (BGR format for OpenCV)
        image = frame.to_ndarray(format="bgr24")
        
        # Convert BGR to RGB for MediaPipe processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 1. Process Frame and Extract Features
        features, annotated_image_rgb, hand_detected = process_and_extract_features(image_rgb, self.hands_model)

        # Convert back to BGR for display
        annotated_image_bgr = cv2.cvtColor(annotated_image_rgb, cv2.COLOR_RGB2BGR)

        # 2. Prediction (Only if hand is detected and model is available)
        if hand_detected and self.model and self.le:
            predicted_sign, confidence, _ = predict_sign(self.model, self.le, features)
            self.prediction_result = f"{predicted_sign} ({confidence})"
        elif self.model is None or self.le is None:
            self.prediction_result = "MODEL ERROR"
        else:
            self.prediction_result = "NO HAND"

        # 3. Overlay Prediction Text on the Frame
        cv2.putText(
            annotated_image_bgr, 
            f"Prediction: {self.prediction_result}", 
            (50, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, (0, 255, 0), 2, cv2.LINE_AA
        )

        # Return the processed frame
        return av.VideoFrame.from_ndarray(annotated_image_bgr, format="bgr24")


# --- 4. CUSTOM STYLING (For beauty and responsiveness) ---

def custom_styling():
    """Injects custom CSS for a better UI/UX."""
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');
        
        :root {
            --primary-color: #4CAF50; /* Green */
            --secondary-color: #3F51B5; /* Indigo */
            --background-color: #f0f2f6;
            --card-color: #ffffff;
            --text-color: #1f2937;
        }

        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Inter', sans-serif;
            background-color: var(--background-color);
        }
        
        /* Main Header/Title */
        h1 {
            color: var(--secondary-color);
            text-align: center;
            font-weight: 900;
            padding-bottom: 10px;
            border-bottom: 4px solid var(--primary-color);
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: var(--card-color);
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border-radius: 12px;
        }

        /* Card/Main Content Area Styling */
        .stCard {
            background-color: var(--card-color);
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            margin-bottom: 20px;
        }

        /* Prediction Result Box (For upload section) */
        .prediction-box {
            text-align: center;
            padding: 30px;
            margin-top: 20px;
            border-radius: 12px;
            background: linear-gradient(135deg, var(--primary-color), #8BC34A);
            color: white;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        }
        .prediction-sign {
            font-size: 8rem; /* Large font size */
            font-weight: 900;
            line-height: 1;
            margin: 0;
        }
        .prediction-confidence {
            font-size: 1.5rem;
            font-weight: 700;
        }

        /* Streamlit elements styling (buttons, radio) */
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            border: 2px solid var(--secondary-color);
            color: var(--secondary-color);
            background-color: var(--card-color);
            padding: 10px 20px;
            font-weight: 700;
            transition: all 0.2s ease-in-out;
        }
        .stButton>button:hover {
            background-color: var(--secondary-color);
            color: white;
            box-shadow: 0 4px 8px rgba(63, 81, 181, 0.5);
        }
        
        /* Mobile Responsiveness for Prediction Box */
        @media (max-width: 600px) {
            .prediction-sign {
                font-size: 5rem;
            }
        }
        </style>
    """, unsafe_allow_html=True)


# --- 5. MAIN STREAMLIT APPLICATION ---

def main():
    """The main function that runs the Streamlit application."""
    custom_styling()
    
    st.title("🧏 Real-Time Sign Language Predictor")
    
    st.markdown("""
        <div style="text-align: center; color: #555; margin-bottom: 20px; font-weight: 400;">
            Live recognition for Characters, Numbers, and Words using classic ML models and MediaPipe.
        </div>
    """, unsafe_allow_html=True)


    # --- Sidebar for Mode Selection ---
    st.sidebar.header("⚙️ Select Recognition Mode")
    
    mode_options = list(MODEL_PATHS.keys())
    selected_mode = st.sidebar.radio("Choose the dataset/model to use:", mode_options, index=0)
    
    # Load Model Assets
    model, le = load_model_assets(selected_mode)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Current Model:** `{selected_mode}`")

    # --- Main Content Area - Live Prediction ---
    st.header("1. Live Webcam Stream")
    
    if selected_mode == "Words":
        st.info("The **Words** mode is ready, but requires your trained model files to be placed in `models/words/`.")

    # Container for the live video stream
    ctx = webrtc_streamer(
        key="sign-language-recognition",
        mode=WebRtcMode.SENDRECV,
        client_settings=RTC_CONFIGURATION,
        video_transformer_factory=lambda: MLVideoTransformer(model=model, le=le),
        async_transform=True,
    )
    
    if ctx.state.playing:
        st.success("Camera is active! Frame the hand clearly for recognition.")
    else:
        st.info("Click 'Start' above to activate your webcam/mobile camera.")
    
    st.markdown("---")
    
    # --- Main Content Area - Upload Image ---
    st.header("2. Upload Image (Single-Shot Test)")
    
    uploaded_file = st.file_uploader(
        "Upload an image of a hand sign (PNG/JPG)", 
        type=['png', 'jpg', 'jpeg'], 
        disabled=(model is None or le is None)
    )
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            
            col_up_img, col_up_pred = st.columns(2)
            
            # Display the uploaded image
            with col_up_img:
                st.image(image, caption='Uploaded Sign', use_column_width=True)
                
            # Perform prediction
            with col_up_pred:
                with st.spinner('Analyzing uploaded sign...'):
                    # Convert PIL to BGR NumPy array for processing
                    image_np = np.array(image)
                    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                    # 1. Process Frame and Extract Features (using RGB input for the function)
                    # We pass RGB and the function will handle conversion/processing
                    features, _, hand_detected = process_and_extract_features(image_np, hands)
                    
                    if hand_detected:
                        # 2. Predict the sign
                        predicted_sign, confidence, _ = predict_sign(model, le, features)

                        # 3. Display the result beautifully
                        st.markdown("<h3>Prediction Result:</h3>", unsafe_allow_html=True)
                        st.markdown(f"""
                            <div class="prediction-box">
                                <p class="prediction-sign">{predicted_sign}</p>
                                <p class="prediction-confidence">Confidence: {confidence}</p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("No hand landmarks were detected in the uploaded image.")

        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")

if __name__ == "__main__":
    main()
