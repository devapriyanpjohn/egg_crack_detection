# 🥚 Egg Crack Detection

Streamlit + OpenCV pipeline that combines handcrafted features, MobileNetV3 CNN embeddings, PCA, and XGBoost to classify eggs as **cracked** or **normal**.

---

## 📂 Project Structure

src/
├── app.py # Streamlit interface
├── extract_features.py # Extracts handcrafted OpenCV features (edges, contours, textures)
├── extract_embeddings.py # Uses MobileNetV3 to create deep CNN embeddings per image
├── cnn_embeddings.csv # (Local-only) Stores embeddings — ignored in Git
├── pca.pkl # (Local-only) PCA model to reduce embedding dimensions
├── model.pkl # Trained XGBoost classifier
requirements.txt # Required Python packages
README.md # Project documentation
.gitignore # Excludes large files & data like cnn_embeddings.csv, pca.pkl, Data/


---

## 🔧 Installation

```bash
git clone https://github.com/your-username/egg-crack-detection.git
cd egg-crack-detection
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
pip install -r requirements.txt

🚀 Usage
streamlit run src/app.py
Use the web interface to upload an egg image—then see “cracked” or “normal” with heandcrafted features.

🛠 How It Works
Handcrafted Feature Extraction (extract_features.py):

Uses OpenCV to measure edges, contours, and texture details.

CNN Embeddings (extract_embeddings.py):

Leverages MobileNetV3 to extract high-level image features, saved to cnn_embeddings.csv.

PCA (pca.pkl):

Reduces embedding dimensions to improve performance and model speed.

XGBoost Classifier (model.pkl):

Trained on combined handcrafted + reduced embeddings to predict egg integrity.

Streamlit Frontend (app.py):

Accepts image uploads, runs feature & embedding extraction, applies PCA & model, and displays results interactively.


