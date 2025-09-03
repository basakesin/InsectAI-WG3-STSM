import os
import gradio as gr
from transformers import pipeline

MODEL_ID = "basakesin/butterfly-classifier-auto-train" #Change this with your Autotrain project name
HF_TOKEN = os.getenv("HF_TOKEN")  # if the model is private, the token comes from Space secrets

pipe = pipeline("image-classification", model=MODEL_ID, token=HF_TOKEN)

def predict(img, top_k=5):
    preds = pipe(img, top_k=top_k)  # [{'label': '...', 'score': 0.97}, ...]
    # Convert to dictionary for Gradio Label
    return {p["label"]: float(p["score"]) for p in preds}

demo = gr.Interface(
    fn=predict,
    inputs=[gr.Image(type="pil"), gr.Slider(1, 10, value=5, step=1, label="Top-K")],
    outputs=gr.Label(num_top_classes=5),  # by default shows top 5 predictions
    title="Butterfly Classifier",
    description="Upload an image to get top-k predictions."
)

if __name__ == "__main__":
    demo.launch()  # if share=True is set, it provides a public link