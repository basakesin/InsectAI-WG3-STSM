import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # ensure CPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from flask import Flask, request, jsonify, render_template

# --- Config ---
MODEL_PATH = "model.h5"
CLASS_NAMES_PATH = "class_names.txt"
IMG_SIZE = (224, 224)

# --- Lazy model load ---
model = None
class_labels = None

def get_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")
        _m = load_model(MODEL_PATH)
        # warm-up (optional)
        _ = _m.predict(np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32))
        model = _m
    return model

def get_class_labels():
    global class_labels
    if class_labels is None:
        if not os.path.exists(CLASS_NAMES_PATH):
            raise FileNotFoundError(f"Class names file '{CLASS_NAMES_PATH}' not found.")
        with open(CLASS_NAMES_PATH, "r") as f:
            class_labels = [line.strip() for line in f if line.strip()]
    return class_labels

# --- Flask ---
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads/"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

def preprocess_image(image_path):
    image = load_img(image_path, target_size=IMG_SIZE)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

@app.get("/healthz")
def healthz():
    return "ok", 200

@app.get("/")
def home():
    # If you have templates/index.html, this will render it; otherwise we fall back to plain text.
    try:
        return render_template("index.html")
    except Exception:
        return "Service is up. POST an image to /predict", 200

@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No selected file"}), 400

    # Save and preprocess
    filename = file.filename
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)
    image = preprocess_image(file_path)

    # Inference
    m = get_model()
    labels = get_class_labels()
    preds = m.predict(image)[0]
    idx = int(np.argmax(preds))
    return jsonify({
        "class": labels[idx] if 0 <= idx < len(labels) else str(idx),
        "confidence": float(np.max(preds))
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
