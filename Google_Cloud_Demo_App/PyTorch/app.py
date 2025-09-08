import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# Load class labels
with open("class_names.txt", "r") as f:
    class_labels = [line.strip() for line in f if line.strip()]
num_classes = len(class_labels)

img_size = (224, 224)

# Load model info
MODEL_PATH = "model.pth"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")

checkpoint = torch.load(MODEL_PATH, map_location="cpu")
backbone = checkpoint.get("backbone", "mobilenet_v2")  # default mobilenet_v2
state_dict = checkpoint["state_dict"]

# Recreate the model dynamically
if backbone == "mobilenet_v2":
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
elif backbone == "resnet50":
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
elif backbone == "efficientnet_b0":
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
elif backbone == "inception_v3":
    model = models.inception_v3(pretrained=False, aux_logits=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
else:
    raise ValueError(f"Unsupported backbone: {backbone}")

model.load_state_dict(state_dict)
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads/"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
    file.save(file_path)

    image = preprocess_image(file_path)
    with torch.no_grad():
        outputs = model(image)[0]
        probs = torch.softmax(outputs, dim=0).numpy()

    class_index = int(np.argmax(probs))
    return jsonify({
        "class": class_labels[class_index],
        "confidence": float(np.max(probs)),
        "backbone": backbone
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
