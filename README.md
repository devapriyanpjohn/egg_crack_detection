# 🥚 Egg Crack Detection

A Streamlit + OpenCV pipeline that combines handcrafted features, MobileNetV3 CNN embeddings, PCA, and XGBoost to classify eggs as **cracked** or **normal**.

---

## 📂 Project Structure

egg-crack-detection/ # Root folder
├── src/
│ ├── app.py # Streamlit interface: uploads image, runs pipeline, shows output
│ ├── extract_features.py # Extracts handcrafted OpenCV features (edges, contours, textures)
│ ├── extract_embeddings.py # Generates CNN embeddings using MobileNetV3
│ ├── cnn_embeddings.csv # (Local-only) Stores embeddings for training—ignored via .gitignore
│ ├── pca.pkl # (Local-only) PCA model to reduce embedding dimensions—ignored via .gitignore
│ ├── model.pkl # Final trained XGBoost classifier used by app.py
├── requirements.txt # List of Python dependencies
├── .gitignore # Excludes large files and folders (e.g. venv/, Data/, cnn_embeddings.csv, pca.pkl)
└── README.md # This documentation file


