📌 Path 2: Use Hugging Face AutoTrain

[AutoTrain](https://huggingface.co/autotrain) lets you train models via web UI without coding.

### 1. Create a New AutoTrain Project
- Go to [AutoTrain](https://huggingface.co/autotrain)  
- Select **Image Classification** as task  
- Upload your dataset (zip file with `classA/`, `classB/` at top level)  
- Choose a base model (e.g. `resnet50`, `efficientnet-b0`)  
- Start training (⚠️ note: AutoTrain cloud is a paid service, but local training is free)  

### 2. Deploy AutoTrain Model on Spaces

After training, the model is pushed to your HF account.  

To create a demo:

1. Create a new **Space** (Gradio)  
2. Add [`app.py`](https://github.com/basakesin/InsectAI-WG3-STSM/blob/main/Hugging_Face_Spaces_Demo/Hugging%20Face%20Spaces%20for%20Autotrain/app.py) to the Files section.
3. Add [`requirements.txt`](https://github.com/basakesin/InsectAI-WG3-STSM/blob/main/Hugging_Face_Spaces_Demo/Hugging%20Face%20Spaces%20for%20Autotrain/requirements.txt) to the Files section.
