# 🦋 Butterfly Classifier (Keras + Gradio on Hugging Face Spaces)

This repository contains a **butterfly image classifier** trained with **Keras** and deployed on **Hugging Face Spaces** using **Gradio**.  
The model was trained on the [Kaggle Butterfly Image Classification dataset](https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species).

Upload a butterfly image and the app will return the **Top-3 predicted species** with probabilities.


## 🚀 Demo

👉 [Try the demo on Hugging Face Spaces](https://huggingface.co/spaces/basakesin/image_classifier_with_keras)  


## 🚀 Deploy on Hugging Face Spaces (No coding needed)

1. **Open your HF profile → New Space**  
   - On Hugging Face, click your avatar (top-right) → **Create new Space**.  
   - Give it a name, add a short description, pick a license (optional).

2. **Select the SDK**  
   - Choose **Gradio** as the Space SDK.  
   - Hardware: **CPU** is enough for this demo.

![Create Spaces](HF_create_Space.png)

3. **Upload the project files**  
   - Go to the Space’s **Files** tab → **Upload files**.  
   - Upload these files from this repo:
     - `app.py`
     - `requirements.txt`
     - `model.h5` (your trained Keras model)
     - `class_names.txt` (one class name per line, in the same order used during training)

4. **Wait for the build**  
   - The Space will automatically build and switch to **Running**.  
   - You’ll see the Gradio interface with an **Upload Image** button.

5. **Test the app**  
   - Drop a butterfly image and check the JSON output for the **Top-3 Predictions**.

![Add Files](Add_files_to_HF.png)

## ✏️ Customize the UI Text

You can change the app title, description, and labels by editing **`app.py`**:

- **Title** (appears at the top of the app)
- **Description** (the explanatory text under the title)
- **Input/Output labels** (e.g., “Upload Butterfly Image”, “Top 3 Predictions”)

> After editing `app.py`, commit/push or re-upload it to your Space. The Space will rebuild automatically.


## 📁 What each file is

- **`model.h5`** — Your trained Keras model.  
- **`class_names.txt`** — Class labels used during training (one per line, in order).  
- **`app.py`** — The Gradio app that loads the model and serves predictions.  
- **`requirements.txt`** — Python dependencies for the Space.


## ❗️Common pitfalls

- **Wrong class order:** Ensure `class_names.txt` matches the exact order of your model’s output classes.  
- **Model mismatch:** If loading fails, re-save your model locally (same TensorFlow/Keras version) and re-upload `model.h5`.  
- **Build errors:** Check the **Logs** tab in your Space; missing packages usually mean updating `requirements.txt`.

## 🙌 Acknowledgements

Dataset: *Butterfly Image Classification (40 species)* on Kaggle.  
Tech: Hugging Face Spaces • Gradio • TensorFlow/Keras.
