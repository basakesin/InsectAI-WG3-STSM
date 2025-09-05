import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Load model (make sure to define your architecture if needed)
# If model was saved with torch.save(model.state_dict(), "model.pth")
# you must recreate the architecture first, then load_state_dict
model = torch.load("model.pth", map_location=torch.device("cpu"))
model.eval()

# Load Class Names from file
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f if line.strip()]

# Preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),                      # Converts to [0,1] tensor
    transforms.Normalize([0.485, 0.456, 0.406], # Imagenet normalization
                         [0.229, 0.224, 0.225])
])

def preprocess(img):
    return transform(img).unsqueeze(0)  # add batch dimension

def predict(image):
    tensor = preprocess(image)
    with torch.no_grad():
        preds = model(tensor)[0]
        probs = torch.softmax(preds, dim=0).numpy()
    
    # Take top 3 predictions
    top_indices = probs.argsort()[-3:][::-1]
    top_results = {class_names[i]: round(float(probs[i]), 3) for i in top_indices}
    return top_results

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Butterfly Image"),
    outputs=gr.JSON(label="Top 3 Predictions"),
    title="Butterfly Classifier",
    description=(
        "Upload a butterfly image to classify its species. "
        "This model was trained on the Kaggle Butterfly Image Classification dataset: "
        "https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species"
    )
)

if __name__ == "__main__":
    interface.launch()
