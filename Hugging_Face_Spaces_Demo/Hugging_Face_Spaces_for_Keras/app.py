import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load single model
model = tf.keras.models.load_model("model.h5")

# Load Class Names from file
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f if line.strip()]

def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

def predict(image):
    preprocessed = preprocess(image)
    preds = model.predict(preprocessed)[0]
    
    # Take top 3 predictions
    top_indices = preds.argsort()[-3:][::-1]
    top_results = {class_names[i]: round(float(preds[i]), 3) for i in top_indices}
    
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
