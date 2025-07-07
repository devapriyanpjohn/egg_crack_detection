# app.py
import streamlit as st
import numpy as np
import pandas as pd
import cv2
import torch
import timm
import joblib
from PIL import Image
from torchvision import transforms

# Load models and objects
model = joblib.load("xgb_model.pkl")
pca = joblib.load("pca.pkl")
scaler = joblib.load("scaler.pkl")

# Load MobileNetV3 model for embedding extraction
mobilenet = timm.create_model('mobilenetv3_large_100', pretrained=True, features_only=True)
mobilenet.eval()
cfg = mobilenet.default_cfg

# Preprocessing pipeline for CNN
preprocess = transforms.Compose([
    transforms.Resize((cfg['input_size'][1], cfg['input_size'][2])),
    transforms.ToTensor(),
    transforms.Normalize(mean=cfg['mean'], std=cfg['std']),
])

# Handcrafted feature extraction function
def extract_handcrafted_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    edge_pixel_count = np.sum(edges > 0)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    small_contours = [c for c in contours if cv2.contourArea(c) < 5000]
    num_crack_contours = len(small_contours)

    if not contours:
        return (edge_pixel_count, 0, 0, 0, np.std(gray), num_crack_contours)

    largest = max(contours, key=cv2.contourArea)
    max_area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)
    circularity = (4 * np.pi * max_area) / (perimeter ** 2) if perimeter > 0 else 0
    hull = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull)
    solidity = max_area / hull_area if hull_area > 0 else 0
    std_intensity = np.std(gray)

    return (edge_pixel_count, max_area, circularity, solidity, std_intensity, num_crack_contours)

# CNN embedding extractor
def extract_cnn_embedding(pil_img):
    img_tensor = preprocess(pil_img).unsqueeze(0)
    with torch.no_grad():
        features = mobilenet(img_tensor)
    return features[-1].flatten().cpu().numpy()

# Streamlit app
st.title("🥚 Egg Crack Detection")
st.write("Upload an egg image to detect whether it is **Normal** or **Cracked**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Show image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert PIL to OpenCV format
    open_cv_image = np.array(image)[:, :, ::-1].copy()

    # Extract features
    handcrafted = extract_handcrafted_features(open_cv_image)
    embedding = extract_cnn_embedding(image)

    # Apply PCA
    embedding_pca = pca.transform([embedding])

    # Combine features
    final_input = np.hstack([handcrafted, embedding_pca[0]]).reshape(1, -1)
    final_input_scaled = scaler.transform(final_input)

    # Predict
    pred = model.predict(final_input_scaled)[0]
    label = "Normal Egg 🥚" if pred == 0 else "Cracked Egg ⚠️"

    st.subheader("🔎 Prediction")
    st.success(f"Result: **{label}**")

    st.subheader("📊 Extracted Handcrafted Features")
    columns = ["Edge Pixels", "Max Area", "Circularity", "Solidity", "Std Intensity", "Num Crack Contours"]
    df_features = pd.DataFrame([handcrafted], columns=columns)
    st.dataframe(df_features)

'''# app.py
# app.py
import streamlit as st
import numpy as np
import pandas as pd
import cv2
import torch
import timm
import joblib
from PIL import Image
from torchvision import transforms

# === Load XGBoost model and Scaler ===
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

# === Embed PCA parameters from pca.pkl (copy from your own pca) ===
# Replace these with actual printed values
PCA_MEAN = np.array([
    # example values, replace with actual mean
    0.0142, -0.0015, ..., 0.0032
])

PCA_COMPONENTS = np.array([
    # each row = one principal component (shape: [n_components, original_dim])
    [0.0123, -0.0532, ..., 0.0191],
    [ ... ],
    ...
])

def apply_pca(vec):
    """Apply PCA transform manually using mean and components"""
    return (vec - PCA_MEAN) @ PCA_COMPONENTS.T

# === Load MobileNetV3 for feature extraction ===
mobilenet = timm.create_model('mobilenetv3_large_100', pretrained=True, features_only=True)
mobilenet.eval()
cfg = mobilenet.default_cfg

preprocess = transforms.Compose([
    transforms.Resize((cfg['input_size'][1], cfg['input_size'][2])),
    transforms.ToTensor(),
    transforms.Normalize(mean=cfg['mean'], std=cfg['std']),
])

# === Handcrafted feature extraction ===
def extract_handcrafted_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edge_count = int((edges > 0).sum())

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    small = [c for c in contours if cv2.contourArea(c) < 5000]
    num_cracks = len(small)

    if not contours:
        return (edge_count, 0, 0, 0, float(np.std(gray)), num_cracks)

    largest = max(contours, key=cv2.contourArea)
    max_area = float(cv2.contourArea(largest))
    perimeter = float(cv2.arcLength(largest, True))
    circularity = (4 * np.pi * max_area) / (perimeter ** 2) if perimeter > 0 else 0.0
    hull = cv2.convexHull(largest)
    hull_area = float(cv2.contourArea(hull))
    solidity = max_area / hull_area if hull_area > 0 else 0.0
    std_intensity = float(np.std(gray))

    return (edge_count, max_area, circularity, solidity, std_intensity, num_cracks)

# === CNN embedding extraction ===
def extract_cnn_embedding(pil_img):
    tensor = preprocess(pil_img).unsqueeze(0)
    with torch.no_grad():
        feats = mobilenet(tensor)
    return feats[-1].flatten().cpu().numpy()

# === Streamlit UI ===
st.title("🥚 Egg Crack Detection")
st.write("Upload an egg image to detect whether it is **Normal** or **Cracked**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to OpenCV format
    open_cv_image = np.array(image)[:, :, ::-1].copy()

    # Step 1: Feature extraction
    handcrafted = extract_handcrafted_features(open_cv_image)
    embedding = extract_cnn_embedding(image)

    # Step 2: PCA reduction (inline, no file)
    embedding_pca = apply_pca(embedding)

    # Step 3: Combine features & scale
    full_features = np.hstack([handcrafted, embedding_pca]).reshape(1, -1)
    scaled = scaler.transform(full_features)

    # Step 4: Predict
    pred = model.predict(scaled)[0]
    label = "Normal Egg 🥚" if pred == 0 else "Cracked Egg ⚠️"

    st.subheader("🔎 Prediction")
    st.success(f"Result: **{label}**")

    # Step 5: Display handcrafted features
    st.subheader("📊 Extracted Handcrafted Features")
    columns = ["Edge Pixels", "Max Area", "Circularity", "Solidity", "Std Intensity", "Num Crack Contours"]
    df = pd.DataFrame([handcrafted], columns=columns)
    st.dataframe(df)
'''

