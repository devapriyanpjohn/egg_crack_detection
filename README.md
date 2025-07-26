# 🥚 Egg Crack Detection

A Streamlit + OpenCV pipeline that combines handcrafted features, MobileNetV3 CNN embeddings, PCA, and XGBoost to classify eggs as **cracked** or **normal**.

---

## 📂 Project Structure

```text
egg-crack-detection/             # Root folder
├── src/
│   ├── app.py                   # Streamlit interface: upload image, run pipeline, show output
│   ├── extract_features.py      # Extracts handcrafted OpenCV features
│   ├── extract_embeddings.py    # Generates CNN embeddings using MobileNetV3
│   ├── train_model.ipynb        # Jupyter notebook for model training & exploration
│   ├── features.csv             # Saved handcrafted feature vectors
│   ├── cnn_embeddings.csv       # (Local‑only) CNN embeddings—ignored via .gitignore
│   ├── scaler.pkl               # Scaler for handcrafted features
│   ├── pca.pkl                  # PCA model for embedding reduction—ignored
│   ├── xgb_model.pkl            # Trained XGBoost model
│   └── requirements.txt         # Python dependencies
├── Data/                        # Raw image datasets—ignored
├── .gitignore                   # Specifies files/folders to exclude from Git
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
```
## 🚀 Usage

```bash
streamlit run src/app.py
```
## 🛠 How It Works

1. **Handcrafted Features** (`extract_features.py`)  
   Uses OpenCV to compute edge density, contour shapes, and texture descriptors.

2. **CNN Embeddings** (`extract_embeddings.py`)  
   Uses MobileNetV3 to extract high‑level image embeddings, saved locally to `cnn_embeddings.csv`.

3. **PCA Reduction** (`pca.pkl`)  
   Applies Principal Component Analysis to reduce embedding dimensions for faster inference.

4. **XGBoost Classifier** (`model.pkl`)  
   Trained on combined handcrafted features and PCA‑reduced embeddings to classify egg integrity.

5. **Streamlit Frontend** (`app.py`)  
   Orchestrates image upload, feature/embedding extraction, PCA transformation, model prediction, and displays results.

## 🔍 Try the App
[Open Egg Crack Detection App]([https://<your-app-subdomain>.streamlit.app](https://eggcrackdetection-ag5s46tgu2fcahkriej7as.streamlit.app/))







