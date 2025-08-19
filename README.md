# InsectAI-WG3-STSM - Model Training 

This repository contains educational materials and tutorials developed during the **InsectAI Short-Term Scientific Mission (STSM)**, hosted by **Dr. Paul Bodesheim** at the [Computer Vision Group, Friedrich Schiller University Jena](https://inf-cv.uni-jena.de/), Germany, between **18 August and 12 September 2025**.


## 1. Open the Notebook in Google Colab
   - Click the Colab link for ['train_model.ipynb'](https://colab.research.google.com/drive/15nQznMSMnyAkQyuxM7ZbdzXM_5f_9FMy?usp=sharing).
   - In Colab, go to **File → Save a copy in Drive** to save your own copy.

## 2. Prepare Your Dataset  
   Organize your insect images so each species has its own folder:

```
train/
├── butterfly/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── beetle/
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```
   
- Folder names = class labels  
- Use `.jpg` or `.png` files  
- Aim for ≥20–50 images per class
- Zip your train folder

## 3. Upload Your Dataset to Colab
- Mount Google Drive and point the notebook to your dataset (train.zip) folder.

## 4. Run the Notebook 
- In the Colab menu: **Runtime → Run all**  
- The notebook will automatically:  
  - Install dependencies  
  - Load your dataset
  - Provide class_names.txt for prediction
  - Train a model  
  - Save multiple export formats  
  - Provide clickable download buttons  
## 5.  Download Your Model
- At the end of the notebook, you’ll see **download buttons**:  
  - **Download .keras** (Keras model)  
  - **Download .tflite** (TensorFlow Lite model for mobile/edge)  
  - **Download SavedModel (.zip)**  
  - **Download ALL outputs (.zip)**
## Model Outputs  

After training, the following files are generated inside the output folder:  

- `<BACKBONE>_final.keras` → Native Keras model  
- `<BACKBONE>_savedmodel/` → TensorFlow SavedModel format  
- `<BACKBONE>.tflite` → TensorFlow Lite model  
- Optional `.zip` archives created for easy downloading  

---

## Adjustable Settings  

The notebook exposes simple variables you can change at the top:  

- `DATA_DIR` – your dataset path  
- `OUTPUT_DIR` – where to save models  
- `BACKBONE` – backbone network (e.g., `mobilenet_v2`, `efficientnet_b0`, `resnet50`)  
- `IMG_SIZE` – input image size (e.g., 224)  
- `BATCH_SIZE` – batch size for training  
- `EPOCHS` – number of training epochs  

> If unsure, just keep the defaults — they work out of the box.  

---

## Tips for Good Training  

- Use **balanced classes** (similar number of images per species)  
- Ensure images are **clear and correctly labeled**  
- Train longer (`EPOCHS=20+`) if accuracy keeps improving  
- Enable **GPU runtime** in Colab (Runtime → Change runtime type → Hardware accelerator: GPU)  

---

## Evaluation  

The notebook shows:  
- Training and validation accuracy/loss curves   
- Confusion matrix 

# InsectAI-WG3-STSM – Model Testing  

This repository provides a **testing notebook** (`test_model_with_gradio.ipynb`) that lets you upload a trained model and interactively run predictions via a **Gradio web interface**.  

## 1. Open the Notebook in Google Colab  
- Click the Colab link for ['test_model_with_gradio.ipynb'](https://colab.research.google.com/drive/1waaanvDYt3pdtK7MAqvpAW9AE_TdR2uC?usp=sharing).  
- In Colab, go to **File → Save a copy in Drive** to save your own copy.  

## 2. Run the Notebook  
- In the Colab menu: **Runtime → Run all**  
- The notebook will automatically:  
  - Install dependencies  
  - Launch a **Gradio interface** for image classification  

## 3. Use the Gradio Interface  
Once the interface launches, you can:  
1. Upload your **trained model** (`.keras`, `.h5`, or SavedModel `.zip`)  
2. Upload the **class_names.txt** file (created by `train_model.ipynb`)  
3. Select the backbone (MobileNetV2, EfficientNetB0, ResNet50, or InceptionV3) if needed  
4. Upload an **image of an insect**  
5. Click **Predict** to view the **Top-5 class probabilities**  

## 4. Typical Workflow  
1. Train your model with `train_model.ipynb`  
2. Download `class_names.txt` and your model (`.keras`, `.h5`, or `.zip`)  
3. Open `test_model_with_gradio.ipynb`  
4. Run the notebook and test with your own insect images  

---

## Notes  
- If your model already contains its preprocessing layers (e.g., `Lambda(preprocess_input)`), the notebook automatically detects this — you don’t need to scale inputs manually.  
- Works with backbones: **MobileNetV2, EfficientNetB0, ResNet50, InceptionV3**  
- Predictions may be inaccurate if the **wrong backbone is selected** at load time.  

