# 🦋 Butterfly Classifier – Google Cloud Flask App

This repository provides a **Flask-based web application** for serving trained image classification models on **Google Cloud Run**.  
It includes both **Keras (TensorFlow)** and **PyTorch** versions of the app.  

## 🎯 Aim of the Project

The goal of this code is to demonstrate how to deploy a trained deep learning model as a **user-friendly web app**.  
Users can upload butterfly images (or select example ones) to classify them into species.  

The provided web interface (`index.html`) supports drag-and-drop uploads and displays the predicted class with confidence.  

## 🗂 Project Structure


## 🖥️ How to Run Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/basakesin/InsectAI-WG3-STSM.git
   cd InsectAI-WG3-STSM/Google_Cloud_Demo_App
2. Install dependencies:
   ```bash
    pip install -r requirements.txt

3. Run the app:
   ```bash
   python app.py

4. Open in browser:
   ```cpp
   http://127.0.0.1:8080

## 🌐 Deploying on Google Cloud Run

Before deploying, make sure you have:

1. A **Google Cloud account** → [Sign up here](https://cloud.google.com/)  
2. A **Google Cloud project** (you can create one in the [Google Cloud Console](https://console.cloud.google.com/))  
3. **Billing enabled** on your project (required for Cloud Run, though there is a free tier).  
4. **Google Cloud SDK (gcloud)** installed on your computer → [Install instructions](https://cloud.google.com/sdk/docs/install)  

Step 1: Authenticate Google Cloud:
  ```bash
  gcloud auth login
```

Step 2: Set your project:

  ```bash
  gcloud config set project PROJECT-ID
```

Step 3: Build the Docker image:

  ```bash
  gcloud builds submit --tag gcr.io/PROJECT-ID/butterfly-classifier
```

Step 4: Deploy to Cloud Run
  ```bash
  gcloud run deploy butterfly-classifier \
    --image gcr.io/PROJECT-ID/butterfly-classifier \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated
```
After deployment, Cloud Run will print a service URL. Open it in your browser to use the app.




