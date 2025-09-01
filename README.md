# InsectAI-WG3-STSM – Educational Materials & Tools  

This repository was developed during the **InsectAI Short-Term Scientific Mission (STSM)**, hosted by **Dr. Paul Bodesheim** at the [Computer Vision Group, Friedrich Schiller University Jena](https://inf-cv.uni-jena.de/), Germany, between **18 August and 12 September 2025**.  

It provides **hands-on tutorials** and **demo applications** to train, test, and deploy insect image classifiers without requiring advanced coding knowledge.  

---

## Repository Structure  

- **[`Google Colab Notebooks/`](https://github.com/basakesin/InsectAI-WG3-STSM/tree/main/Google%20Colab%20Notebooks)**  
  Train and test deep learning models (TensorFlow/Keras or PyTorch) entirely in Google Colab.  
  - `train_model_with_keras.ipynb`  
  - `train_model_with_pytorch.ipynb`  
  - `test_model_with_gradio.ipynb`  

- **[`ButterFly Classifier Flask App/`](https://github.com/basakesin/InsectAI-WG3-STSM/tree/main/ButterFly%20Classifier%20Flask%20App)**    
  A minimal **Flask web app** showing how to deploy a trained model to **Google Cloud Run** with a simple web interface.  

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


### 2. Deployment (Flask + Google Cloud Run)  


