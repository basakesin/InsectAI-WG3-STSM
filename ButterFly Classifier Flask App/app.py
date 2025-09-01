import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# Define available models
AVAILABLE_MODELS = ["VGG16", "MobileNetV2", "InceptionV3"]

# Load class labels
class_labels = ['ADONIS', 'AFRICAN GIANT SWALLOWTAIL', 'AMERICAN SNOOT', 'AN 88', 'APPOLLO',
                'ARCIGERA FLOWER MOTH', 'ATALA', 'ATLAS MOTH', 'BANDED ORANGE HELICONIAN',
                'BANDED PEACOCK', 'BANDED TIGER MOTH', 'BECKERS WHITE', 'BIRD CHERRY ERMINE MOTH',
                'BLACK HAIRSTREAK', 'BLUE MORPHO', 'BLUE SPOTTED CROW', 'BROOKES BIRDWING',
                'BROWN ARGUS', 'BROWN SIPROETA', 'CABBAGE WHITE', 'CAIRNS BIRDWING',
                'CHALK HILL BLUE', 'CHECQUERED SKIPPER', 'CHESTNUT', 'CINNABAR MOTH',
                'CLEARWING MOTH', 'CLEOPATRA', 'CLODIUS PARNASSIAN', 'CLOUDED SULPHUR',
                'COMET MOTH', 'COMMON BANDED AWL', 'COMMON WOOD-NYMPH', 'COPPER TAIL', 'CRECENT',
                'CRIMSON PATCH', 'DANAID EGGFLY', 'EASTERN COMA', 'EASTERN DAPPLE WHITE',
                'EASTERN PINE ELFIN', 'ELBOWED PIERROT', 'EMPEROR GUM MOTH', 'GARDEN TIGER MOTH',
                'GIANT LEOPARD MOTH', 'GLITTERING SAPPHIRE', 'GOLD BANDED', 'GREAT EGGFLY',
                'GREAT JAY', 'GREEN CELLED CATTLEHEART', 'GREEN HAIRSTREAK', 'GREY HAIRSTREAK',
                'HERCULES MOTH', 'HUMMING BIRD HAWK MOTH', 'INDRA SWALLOW', 'IO MOTH',
                'Iphiclus sister', 'JULIA', 'LARGE MARBLE', 'LUNA MOTH', 'MADAGASCAN SUNSET MOTH',
                'MALACHITE', 'MANGROVE SKIPPER', 'MESTRA', 'METALMARK', 'MILBERTS TORTOISESHELL',
                'MONARCH', 'MOURNING CLOAK', 'OLEANDER HAWK MOTH', 'ORANGE OAKLEAF',
                'ORANGE TIP', 'ORCHARD SWALLOW', 'PAINTED LADY', 'PAPER KITE', 'PEACOCK',
                'PINE WHITE', 'PIPEVINE SWALLOW', 'POLYPHEMUS MOTH', 'POPINJAY', 'PURPLE HAIRSTREAK',
                'PURPLISH COPPER', 'QUESTION MARK', 'RED ADMIRAL', 'RED CRACKER', 'RED POSTMAN',
                'RED SPOTTED PURPLE', 'ROSY MAPLE MOTH', 'SCARCE SWALLOW', 'SILVER SPOT SKIPPER',
                'SIXSPOT BURNET MOTH', 'SLEEPY ORANGE', 'SOOTYWING', 'SOUTHERN DOGFACE',
                'STRAITED QUEEN', 'TROPICAL LEAFWING', 'TWO BARRED FLASHER', 'ULYSES', 'VICEROY',
                'WHITE LINED SPHINX MOTH', 'WOOD SATYR', 'YELLOW SWALLOW TAIL', 'ZEBRA LONG WING']

img_size = (224, 224)

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


# Function to load model dynamically
def load_selected_model(model_name):
    if model_name not in AVAILABLE_MODELS:
        return None, f"Model '{model_name}' not found. Available models: {AVAILABLE_MODELS}"

    model_path = f"butterfly_classifier_{model_name}.h5"

    if not os.path.exists(model_path):
        return None, f"Model file '{model_path}' not found."

    model = load_model(model_path)
    return model, None


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

    if 'model' not in request.form:
        return jsonify({'error': 'No model specified'})

    model_name = request.form['model']
    model, error_message = load_selected_model(model_name)

    if model is None:
        return jsonify({'error': error_message})

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
