# 🤖 Gesture Classifier (MediaPipe + ML)

Classifies body gestures using MediaPipe Holistic landmarks and a trained ML model.  
Built for real-time gesture recognition using pose and hand landmarks.

---

## 📦 Project Files

gesture-classifier/ ├── gesture_app.py # Main real-time app ├── ai body language.ipynb # Notebook for training & testing ├── pose_classifier.h5 # Trained Keras model ├── coords.csv # Extracted features (landmarks + labels) ├── requirements.txt # Dependencies ├── .gitignore # Ignored files config


---

## ⚙️ Setup & Run

```bash
# Create and activate virtual environment (optional)
python -m venv venv
venv\Scripts\activate      # For Windows
source venv/bin/activate  # For macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
python gesture_app.py
```

# 🧠 Models & Tools
MediaPipe Holistic – full-body landmark detection

TensorFlow / Keras – for model training & classification

Jupyter Notebook – for training and exporting features to CSV

# 📝 Notes
.mp4, .venv/, and .ipynb_checkpoints/ are excluded via .gitignore

coords.csv is large (~81MB) — consider using Git LFS if pushing to GitHub
