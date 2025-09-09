import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU only for Cloud Run
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# ---------- Config ----------
MODEL_PATH = "model.pth"
CLASS_NAMES_PATH = "class_names.txt"
DEFAULT_BACKBONE = "mobilenet_v2"  # fallback if not in checkpoint

# ---------- Load class labels ----------
with open(CLASS_NAMES_PATH, "r") as f:
    class_labels = [line.strip() for line in f if line.strip()]
NUM_CLASSES = len(class_labels)

# ---------- Build model ----------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")

checkpoint = torch.load(MODEL_PATH, map_location="cpu")
state_dict = checkpoint.get("state_dict", checkpoint)  # support raw state_dict too
backbone = checkpoint.get("backbone", DEFAULT_BACKBONE)

def build_model(backbone_name: str, num_classes: int):
    if backbone_name == "mobilenet_v2":
        m = models.mobilenet_v2(pretrained=False)
        m.classifier[1] = nn.Linear(m.last_channel, num_classes)
    elif backbone_name == "resnet50":
        m = models.resnet50(pretrained=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif backbone_name == "efficientnet_b0":
        m = models.efficientnet_b0(pretrained=False)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif backbone_name == "inception_v3":
        # AuxLogits were present in your checkpoint, so keep aux_logits=True to match keys
        m = models.inception_v3(pretrained=False, aux_logits=True)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    return m

model = build_model(backbone, NUM_CLASSES)

# ---------- Strip classifier weights from state_dict if class count differs ----------
def remove_classifier_keys(sd: dict, backbone_name: str) -> dict:
    sd = sd.copy()
    # keys to always ignore (final layers)
    head_keys = []
    if backbone_name in ("resnet50", "inception_v3"):
        head_keys += ["fc.weight", "fc.bias"]
    if backbone_name == "inception_v3":
        # aux head as well
        head_keys += ["AuxLogits.fc.weight", "AuxLogits.fc.bias"]
    if backbone_name in ("mobilenet_v2", "efficientnet_b0"):
        head_keys += ["classifier.1.weight", "classifier.1.bias"]

    for k in head_keys:
        if k in sd:
            del sd[k]
    return sd

state_dict = remove_classifier_keys(state_dict, backbone)

# Load remaining weights (features/backbone) and ignore missing classifier weights
missing, unexpected = model.load_state_dict(state_dict, strict=False)
# (optional) print to logs for debugging in Cloud Run
print("Missing keys (expected for new head):", missing)
print("Unexpected keys (safe to ignore):", unexpected)

model.eval()
model.to("cpu")
torch.set_num_threads(1)  # friendlier on tiny instances

# ---------- Preprocessing ----------
# Inception usually expects 299x299; others 224x224
IMG_SIZE = (299, 299) if backbone == "inception_v3" else (224, 224)

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# ---------- Flask app ----------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads/"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

@app.get("/")
def home():
    try:
        return render_template("index.html")
    except Exception:
        return "Service is up. POST an image to /predict", 200

@app.get("/healthz")
def healthz():
    return "ok", 200

@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
    file.save(file_path)
    x = preprocess_image(file_path)

    with torch.no_grad():
        out = model(x)  # may be [1, C] or a tuple
        # Handle Inception / odd returns
        if isinstance(out, (tuple, list)):
            out = out[0]
        # If it's a namedtuple with .logits (rare), use that
        if hasattr(out, "logits"):
            out = out.logits
        # Remove batch dimension if present
        if out.ndim == 2 and out.shape[0] == 1:
            out = out.squeeze(0)

        # Now out is [C]; softmax over class dim
        probs = torch.softmax(out, dim=-1).cpu().numpy()

    idx = int(np.argmax(probs))
    return jsonify({
        "class": class_labels[idx],
        "confidence": float(probs[idx]),  # now a scalar
        "backbone": backbone
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
