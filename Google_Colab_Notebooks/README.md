# InsectAI-WG3-STSM – Model Training & Testing  

This repository contains educational materials and tutorials developed during the **InsectAI Short-Term Scientific Mission (STSM)**, hosted by **Dr. Paul Bodesheim** at the [Computer Vision Group, Friedrich Schiller University Jena](https://inf-cv.uni-jena.de/), Germany, between **18 August and 12 September 2025**.  

It provides unified **Google Colab notebooks** to train and test insect image classifiers using either **TensorFlow/Keras** or **PyTorch**.  

## Model Training  

Two framework-specific notebooks are provided:  

- [**Train with Keras** (`train_model_with_keras.ipynb`)](https://colab.research.google.com/drive/14ZDe3DR6h4fQKy2NaXy1SV6T45sC3c1f?)  
- [**Train with PyTorch** (`train_model_with_pytorch.ipynb`)](https://colab.research.google.com/drive/1KI8h4VPXMwSWwkqq-utlBwO1zOEP2kpJ?usp=sharing)  

### 1. Open the Notebook in Colab  
- Click one of the above links  
- In Colab, go to **File → Save a copy in Drive**  

### 2. Prepare Your Dataset  

You can either use your own dataset or start with the provided demo files.  

- This repository already includes `train.zip` and `val.zip` for quick testing.  
- For a fun demo, you can also try the  
  [**Tom & Jerry Image Classification Dataset**](https://www.kaggle.com/datasets/balabaskar/tom-and-jerry-image-classification/data).  
  - Download the prepared archives: [`train.zip`](/train.zip) and [`val.zip`](/val.zip).  
  - Then run the training notebook to build a **Tom vs. Jerry classifier**.  


Organize your images so each species has its own folder:  

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
- `.jpg` or `.png` images  
- ≥20–50 images per class recommended  

Zip the dataset folder (`train.zip`, optionally `val.zip`).  

### 3. Run the Notebook  
- In Colab: **Runtime → Run all**  
- The notebook will:  
  - Install dependencies  
  - Load and preprocess your dataset  
  - Generate `class_names.txt` (one label per line)  
  - Train a model using the chosen backbone  
  - Export models in multiple formats  

### 4. Download Model Outputs  

**Keras exports:**  
- `<BACKBONE>.keras`  
- `<BACKBONE>_savedmodel/` (SavedModel format)  
- `<BACKBONE>.tflite` (TensorFlow Lite)  

**PyTorch exports:**  
- `<BACKBONE>.pt` (full model: `torch.save(model, ...)`)  
- `<BACKBONE>_state_dict.pth` (weights only: `torch.save(model.state_dict(), ...)`)  


## Adjustable Settings  

At the top of each notebook, you can change:  
- `DATA_DIR` – dataset path  
- `OUTPUT_DIR` – output folder  
- `BACKBONE` – backbone architecture (`mobilenet_v2`, `efficientnet_b0`, `resnet50`, `inception_v3`)  
- `IMG_SIZE` – input size (224 or 299 for InceptionV3)  
- `BATCH_SIZE`, `EPOCHS`  

If unsure, just keep the defaults — they work out of the box.  

## Training Tips  

- Keep classes balanced  
- Ensure images are clean and well-labeled  
- Train longer (`EPOCHS=20+`) if accuracy improves  
- Enable GPU in Colab (**Runtime → Change runtime type → GPU**)  

## Model Testing  

Testing is provided in a **single unified notebook**:  

- [**Test with Gradio** (`test_model_with_gradio.ipynb`)](https://colab.research.google.com/drive/1waaanvDYt3pdtK7MAqvpAW9AE_TdR2uC?usp=sharing)  

### 1. Run the Notebook in Colab  
- **Runtime → Run all**  

### 2. Use the Gradio Interface  
When launched, the interface lets you:  

1. Select **Framework**: Keras or PyTorch  
2. Upload your trained model  
   - Keras: `.keras`, `.h5`, or `.zip` (SavedModel)  
   - PyTorch: `.pt` or `.pth` (full model or state_dict)  
3. Upload `class_names.txt`  
4. (If PyTorch state_dict) select the backbone (ResNet50, MobileNetV2, EfficientNetB0, InceptionV3)  
5. Upload an test image and click **Predict**  

Outputs:  
- **Top-5 probabilities** for multiclass models

## Notes  

- The testing notebook auto-detects if preprocessing is embedded inside the Keras model (`Lambda(preprocess_input)`).  
- PyTorch models always apply ImageNet normalization.  
- Wrong backbone selection (for state_dict) may lead to incorrect predictions.  
- Both training and testing workflows run entirely in Colab — no local installation needed.  

Some code and explanatory texts were refined with the assistance of **ChatGPT (OpenAI, 2025)**.  

