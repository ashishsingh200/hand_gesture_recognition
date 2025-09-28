import streamlit as st
import pickle
import numpy as np
from PIL import Image
from io import BytesIO
from streamlit_drawable_canvas import st_canvas  # Requires: pip install streamlit-drawable-canvas

# Custom CSS for a beautiful and user-friendly UI
st.markdown("""
    <style>
    /* Global styles */
    body {
        background-color: #f0f4f8;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #333;
    }
    .stApp {
        max-width: 800px;
        margin: 0 auto;
        padding: 2rem;
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #007bff;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .stSelectbox, .stFileUploader, .stButton {
        margin-bottom: 1rem;
    }
    .stButton > button {
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #0056b3;
    }
    .prediction-result {
        background-color: #e9f7ff;
        border: 1px solid #007bff;
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        font-size: 1.2rem;
        margin-top: 1rem;
    }
    .uploaded-image {
        display: block;
        margin: 1rem auto;
        max-width: 100%;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
    }
    .sidebar .stSelectbox {
        background-color: #f8f9fa;
        border-radius: 6px;
        padding: 0.5rem;
    }
    .canvas-container {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        background-color: #fff;
    }
    </style>
""", unsafe_allow_html=True)

# Paths to models and labels (update if needed; use relative paths for cloud)
CHAR_NUM_MODEL_PATH = 'character and numbers/models/best_model.p'
CHAR_NUM_LABELS_PATH = 'character and numbers/models/labels.p'

CHAR_MODEL_PATH = 'characters/models/best_model.p'
CHAR_LABELS_PATH = 'characters/models/labels.p'

NUM_MODEL_PATH = 'Numbers/models/best_model.p'
NUM_LABELS_PATH = 'Numbers/models/labels.p'

# Placeholder for word model (add your actual paths when ready)
WORD_MODEL_PATH = 'words/models/best_model.p'  # Assuming you'll save it here
WORD_LABELS_PATH = 'words/models/labels.p'    # Assuming you'll save it here

# Function to load model and labels
@st.cache_resource
def load_model_and_labels(model_path, labels_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(labels_path, 'rb') as f:
        labels = pickle.load(f)
    return model, labels

# Load all models
char_num_model, char_num_labels = load_model_and_labels(CHAR_NUM_MODEL_PATH, CHAR_NUM_LABELS_PATH)
char_model, char_labels = load_model_and_labels(CHAR_MODEL_PATH, CHAR_LABELS_PATH)
num_model, num_labels = load_model_and_labels(NUM_MODEL_PATH, NUM_LABELS_PATH)

# Load word model (comment out if not ready, or handle gracefully)
try:
    word_model, word_labels = load_model_and_labels(WORD_MODEL_PATH, WORD_LABELS_PATH)
except FileNotFoundError:
    word_model, word_labels = None, None
    st.warning("Word model not found. Word prediction will use character+number model with numpy-based segmentation as fallback.")

# Preprocess image (assuming 28x28 grayscale, flattened, normalized)
def preprocess_image(image):
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image).convert('L')
    else:
        img = Image.open(image).convert('L')
    img = img.resize((28, 28))  # Resize to model input size (adjust if your model uses different size)
    img_array = np.array(img).flatten() / 255.0  # Flatten and normalize
    return img_array.reshape(1, -1)  # 2D for model predict

# Predict single item
def predict_single(model, labels, image):
    features = preprocess_image(image)
    pred_index = model.predict(features)[0]
    return labels[pred_index]

# Word prediction with segmentation (using numpy for contour-like detection)
def predict_word(image, use_fallback=True):
    # If word model exists, use it; else fallback to segmentation with char_num_model
    if word_model is not None:
        # Assuming word_model predicts on whole image or preprocessed sequence (adjust as per your model)
        features = preprocess_image(image)  # Placeholder; update if word model differs
        pred = word_model.predict(features)[0]
        return word_labels[pred] if isinstance(word_labels, list) else word_labels[pred]
    elif use_fallback:
        # Fallback: Segment image into characters using numpy
        if isinstance(image, BytesIO):
            image.seek(0)
            img = np.array(Image.open(image).convert('L'))
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                # Convert RGB to grayscale
                img = np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
            else:
                img = image
        else:
            img = np.array(Image.open(image).convert('L'))
        
        # Binarize (assume dark text on light background; adjust threshold if needed)
        binary = (img < 128).astype(np.uint8)
        
        # Vertical projection to find character widths
        proj = np.sum(binary, axis=0)
        
        # Find start and end of each character (where projection > 0)
        mask = proj > 0
        diff = np.diff(mask.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        
        if mask[0]:
            starts = np.insert(starts, 0, 0)
        if mask[-1]:
            ends = np.append(ends, len(proj))
        
        word = ''
        for s, e in zip(starts, ends):
            if e - s < 5:  # Ignore small noise (adjust threshold as needed)
                continue
            
            char_img = img[:, s:e]
            
            # Optional: Trim vertically for better accuracy
            char_binary = binary[:, s:e]
            v_proj = np.sum(char_binary, axis=1)
            v_mask = v_proj > 0
            v_diff = np.diff(v_mask.astype(int))
            v_starts = np.where(v_diff == 1)[0] + 1
            v_ends = np.where(v_diff == -1)[0] + 1
            
            if v_mask[0]:
                v_starts = np.insert(v_starts, 0, 0)
            if v_mask[-1]:
                v_ends = np.append(v_ends, len(v_proj))
            
            if len(v_starts) > 0:
                top, bottom = v_starts[0], v_ends[0]  # Take the main block
                char_img = char_img[top:bottom, :]
            
            # Resize to 28x28
            char_pil = Image.fromarray(char_img).resize((28, 28))
            char_array = np.array(char_pil) / 255.0
            features = char_array.flatten().reshape(1, -1)
            pred_index = char_num_model.predict(features)[0]
            word += char_num_labels[pred_index]
        
        return word if word else "No characters detected"
    else:
        return "Word model not available."

# Main app
st.title("Character, Number, and Word Predictor")

# Sidebar for mode selection
mode = st.sidebar.selectbox("Select Prediction Mode", ["Character (A-Z)", "Number (0-9)", "Alphanumeric (A-Z + 0-9)", "Word"])

# Tabs for input methods
tab1, tab2 = st.tabs(["Upload Image", "Live Drawing"])

with tab1:
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True, cls="uploaded-image")
        
        if st.button("Predict"):
            if mode == "Character (A-Z)":
                result = predict_single(char_model, char_labels, uploaded_file)
            elif mode == "Number (0-9)":
                result = predict_single(num_model, num_labels, uploaded_file)
            elif mode == "Alphanumeric (A-Z + 0-9)":
                result = predict_single(char_num_model, char_num_labels, uploaded_file)
            elif mode == "Word":
                result = predict_word(uploaded_file)
            
            st.markdown(f'<div class="prediction-result">Predicted: <strong>{result}</strong></div>', unsafe_allow_html=True)

with tab2:
    st.subheader("Draw Live")
    st.write("Draw below, and the prediction will update in real-time as you draw.")
    
    # Drawing canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=20,
        stroke_color="#000000",
        background_color="#FFFFFF",
        update_streamlit=True,  # Real-time update
        height=200,
        width=600 if mode == "Word" else 200,  # Wider for words
        drawing_mode="freedraw",
        key="canvas",
    )
    
    # Predict if canvas has been drawn on
    if canvas_result.image_data is not None:
        # Convert canvas to image (remove alpha channel if present)
        img_data = canvas_result.image_data
        if img_data.shape[2] == 4:  # RGBA
            img_data = img_data[:, :, :3]  # RGB
        
        # Grayscale for prediction
        gray_img = np.dot(img_data[..., :3], [0.299, 0.587, 0.114])  # Simple RGB to gray
        
        # Threshold to make it binary-like (for better prediction)
        gray_img = np.where(gray_img < 128, 0, 255).astype(np.uint8)
        
        if mode == "Character (A-Z)":
            result = predict_single(char_model, char_labels, gray_img)
        elif mode == "Number (0-9)":
            result = predict_single(num_model, num_labels, gray_img)
        elif mode == "Alphanumeric (A-Z + 0-9)":
            result = predict_single(char_num_model, char_num_labels, gray_img)
        elif mode == "Word":
            result = predict_word(gray_img)
        
        st.markdown(f'<div class="prediction-result">Predicted: <strong>{result}</strong></div>', unsafe_allow_html=True)