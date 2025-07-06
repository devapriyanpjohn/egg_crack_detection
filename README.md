# 🥚 Egg Crack Detection

A Streamlit + OpenCV pipeline that combines handcrafted features, MobileNetV3 CNN embeddings, PCA, and XGBoost to classify eggs as **cracked** or **normal**.

---

## 📂 Project Structure

```text
egg-crack-detection/        # Root folder
├── src/
│   ├── app.py               # Streamlit interface: upload image, run pipeline, show output
│   ├── extract_features.py  # Extracts handcrafted OpenCV features (edges, contours, textures)
│   ├── extract_embeddings.py# Generates CNN embeddings using MobileNetV3
│   ├── cnn_embeddings.csv   # (Local‑only) Embeddings per image—ignored via .gitignore
│   ├── pca.pkl              # (Local‑only) PCA model for dimensionality reduction—ignored
│   ├── model.pkl            # Trained XGBoost classifier used by app.py
├── requirements.txt         # Python dependencies
├── .gitignore               # Excludes venv/, Data/, embeddings & pickle files
└── README.md                # This documentation file
```
## 🔧 Installation

```bash
git clone https://github.com/your-username/egg-crack-detection.git
cd egg-crack-detection
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate

pip install -r requirements.txt




