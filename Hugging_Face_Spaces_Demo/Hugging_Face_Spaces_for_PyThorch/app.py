# app.py — Hugging Face Spaces (Gradio)
import gradio as gr
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import json

CKPT_PATH = "model.pth"            # File we saved during PyTorch export
CLASS_TXT = "class_names.txt"      # Can also read from file if exists (optional)

# -----------------------
# 1) Load checkpoint
# -----------------------
ckpt = torch.load(CKPT_PATH, map_location="cpu")
backbone = ckpt.get("backbone", "mobilenet_v2")
state_dict = ckpt.get("state_dict", ckpt)

# class_names: priority checkpoint, then class_names.txt
if "class_names" in ckpt and isinstance(ckpt["class_names"], (list, tuple)):
    class_names = list(ckpt["class_names"])
elif os.path.exists(CLASS_TXT):
    with open(CLASS_TXT, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]
else:
    raise RuntimeError("class_names not found: neither in checkpoint nor class_names.txt.")

num_classes = len(class_names)

# -----------------------
# 2) Model builder
# -----------------------
def build_model(backbone_name: str, num_classes: int) -> nn.Module:
    name = backbone_name.lower()
    if name == "resnet50":
        m = models.resnet50(weights=None)  # weights not needed for inference
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
        return m
    elif name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=None)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m
    elif name == "inception_v3":
        # For Inception aux_logits=False => single output
        m = models.inception_v3(weights=None, aux_logits=False)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
        return m
    else:
        raise ValueError(f"Unsupported backbone '{backbone_name}'. "
                         "Use one of: resnet50, mobilenet_v2, efficientnet_b0, inception_v3")

model = build_model(backbone, num_classes)
model.load_state_dict(state_dict, strict=False)  # strict=False: tolerate different meta keys
model.eval()

# -----------------------
# 3) Preprocess settings
# -----------------------
input_size = 299 if backbone.lower() == "inception_v3" else 224
transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def preprocess(img: Image.Image) -> torch.Tensor:
    return transform(img.convert("RGB")).unsqueeze(0)

# -----------------------
# 4) Prediction
# -----------------------
@torch.no_grad()
def predict(image: Image.Image):
    x = preprocess(image)
    logits = model(x)
    # (for inception_v3 aux disabled so no tuple, but keep safe)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    probs = torch.softmax(logits[0], dim=0).cpu().numpy()

    top_idx = probs.argsort()[-3:][::-1]
    return {class_names[i]: float(round(probs[i], 3)) for i in top_idx}

# -----------------------
# 5) Gradio UI
# -----------------------
title = "Image Classifier"
description = (
    f"Backbone: **{backbone}** — Upload an image to get top-3 predictions."
)

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.JSON(label="Top-3 Predictions"),
    title=title,
    description=description
)

if __name__ == "__main__":
    iface.launch()
