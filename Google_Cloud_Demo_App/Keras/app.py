import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# Load class labels from file
with open("class_names.txt", "r") as f:
    class_labels = [line.strip() for line in f if line.strip()]
num_classes = len(class_labels)

img_size = (224, 224)

# Load the single model
MODEL_PATH = "model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")
model = load_model(MODEL_PATH)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads/"
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


# Function to preprocess image
def preprocess_image(image_path):
    image = load_img(image_path, target_size=img_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize
    return image


# Home Route
@app.route('/')
def home():
    return render_template('index.html')


# Upload and Predict Route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Preprocess the image and make prediction
    image = preprocess_image(file_path)
    prediction = model.predict(image)
    class_index = np.argmax(prediction)
    class_name = class_labels[class_index]
    confidence = float(np.max(prediction))

    return jsonify({'class': class_name, 'confidence': confidence})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))  # Use PORT=8080 from Cloud Run
    app.run(host='0.0.0.0', port=port, debug=True)
