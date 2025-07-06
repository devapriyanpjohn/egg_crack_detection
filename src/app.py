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
