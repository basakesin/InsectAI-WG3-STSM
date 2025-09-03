# InsectAI-WG3-STSM – Educational Materials & Tools  

This repository was developed during the **InsectAI Short-Term Scientific Mission (STSM)**, hosted by **Dr. Paul Bodesheim** at the [Computer Vision Group, Friedrich Schiller University Jena](https://inf-cv.uni-jena.de/), Germany, between **18 August and 12 September 2025**.  

It provides **hands-on tutorials** and **demo applications** to train, test, and deploy insect image classifiers without requiring advanced coding knowledge.  

---

## Repository Structure  

- **[`Google Colab Notebooks/`](https://github.com/basakesin/InsectAI-WG3-STSM/tree/main/Google_Colab_Notebooks)**  
  Train and test deep learning models (TensorFlow/Keras or PyTorch) entirely in Google Colab.  
  - [`train_model_with_keras.ipynb`](https://colab.research.google.com/drive/14ZDe3DR6h4fQKy2NaXy1SV6T45sC3c1f?usp=sharing)  
  - [`train_model_with_pytorch.ipynb`](https://colab.research.google.com/drive/1uVfFELx63I2pJr29wD0Yyz5MLsGTOOR9?usp=sharing)  
  - [`test_model_with_gradio.ipynb`](https://colab.research.google.com/drive/1waaanvDYt3pdtK7MAqvpAW9AE_TdR2uC?usp=sharing)  

- **[`Hugging Face Spaces Demo/`](https://github.com/basakesin/InsectAI-WG3-STSM/tree/main/Hugging_Face_Spaces_Demo)**  
  Minimal working examples showing how to upload and interact with models via **Hugging Face Spaces**.  

- **[`Google Cloud Demo App/`](https://github.com/basakesin/InsectAI-WG3-STSM/tree/main/Google_Cloud_Demo_App)**    
  A Flask-based deployment example for serving models on **Google Cloud Run**, including a ready-to-use demo web interface.  

---

## Quick Start  

### 1. Model Training & Testing (Google Colab)  

- Open a notebook from the [Google_Colab_Notebooks](./Google_Colab_Notebooks) folder.  
- Follow the step-by-step instructions to:  
  - Prepare your dataset  
  - Train a model (Keras or PyTorch)  
  - Export trained models (`.keras`, `.tflite`, `.pt`, `.pth`)  
  - Test predictions interactively with **Gradio**  

No local installation required — everything runs in Colab.  

---

### 2. Hugging Face Demo  

The [`Hugging_Face_Spaces_Demo`](./Hugging_Face_Spaces_Demo) folder shows how to host your model on **Hugging Face Spaces**.  

You have two options:  

- **Deploying a Trained Model with Gradio**  
  - Upload your model files and Python script.  
  - Build an interactive demo where users can test your classifier directly in the browser.  

- **Deploying AutoTrain Models**  
  - Use Hugging Face’s **AutoTrain** platform to fine-tune and publish models.  
  - Connect your AutoTrain model directly to a Space with minimal effort.  

👉 This approach is ideal for quickly sharing results with collaborators without requiring any infrastructure setup.  

---

### 3. Google Cloud Deployment with Flask App  

The [`Google_Cloud_Demo_App`](./Google_Cloud_Demo_App) folder provides a **step-by-step guide** to deploy your model as a web application using **Flask** and **Google Cloud Run**.  

Features:  
- Ready-to-use `app.py` (Flask backend) and `index.html` (frontend).  
- Example trained model included for quick testing.  
- Deployment instructions with `Dockerfile` and `requirements.txt`.  
- Scalable and secure — users can access your app via a public URL.  

Example workflow:  
1. Train a model in Colab and export it.  
2. Place the model file inside the `Google_Cloud_Demo_App` folder.  
3. Build and push a Docker image to Google Cloud.  
4. Deploy with **Cloud Run** and share the URL.  

---

## License  

This repository is released under the **MIT License**. You are free to use, modify, and share it for educational and research purposes.  




