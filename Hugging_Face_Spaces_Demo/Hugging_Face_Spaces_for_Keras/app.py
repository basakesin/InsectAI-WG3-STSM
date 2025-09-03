# Allow unsafe deserialization (for trusted models with Lambda layers)
import tensorflow as tf, keras, numpy as np
import os, io, zipfile, tempfile, json
from PIL import Image
import gradio as gr


try:
    keras.config.enable_unsafe_deserialization()
except Exception:
    pass

# Patch for DepthwiseConv2D with "groups" argument compatibility
from tensorflow.keras.layers import DepthwiseConv2D as _TFDepthwiseConv2D
class DepthwiseConv2DCompat(_TFDepthwiseConv2D):
    def __init__(self, *args, groups=None, **kwargs):
        super().__init__(*args, **kwargs)
CUSTOM_OBJECTS_BASE = {"DepthwiseConv2D": DepthwiseConv2DCompat}

# Candidate preprocess functions
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mob_pre
from tensorflow.keras.applications.resnet50 import preprocess_input as res_pre
from tensorflow.keras.applications.inception_v3 import preprocess_input as inc_pre

PREPROCESS_CANDIDATES = {
    "EfficientNetB0": eff_pre,
    "MobileNetV2":   mob_pre,
    "ResNet50":      res_pre,
    "InceptionV3":   inc_pre,
}

def _infer_input_size(model):
    """Infer input image size and channels from the model."""
    ishape = model.input_shape
    if isinstance(ishape, list):
        ishape = ishape[0]
    _, h, w, c = ishape
    if h is None or w is None:
        h, w = 224, 224
    return (w, h), c

def _prep_image(pil_img, size, channels, preprocess=None, scale_255=True):
    """Prepare image for prediction, avoiding double-preprocessing."""
    img = pil_img.convert("RGB").resize(size)
    x = np.array(img, dtype=np.float32)
    if channels == 1:
        x = np.mean(x, axis=-1, keepdims=True)
    if preprocess is not None:
        x = preprocess(x)  # don't /255 here
    else:
        if scale_255:
            x = x / 255.0
        # else keep raw 0..255
    return np.expand_dims(x, 0)

def _try_load_no_custom(path):
    return tf.keras.models.load_model(path, compile=False, safe_mode=False)


def _read_class_names(txt_path: str):
    """Read class names from a text file (one class per line)."""
    if not txt_path:
        return None
    with open(txt_path, "r", encoding="utf-8") as f:
        names = [ln.strip() for ln in f if ln.strip()]
    return names or None

def _load_model_any(model_path: str, backbone_hint: str | None):
    """Load model and detect whether it contains internal preprocess_input Lambda."""
    def _resolve_zip(path):
        workdir = tempfile.mkdtemp()
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(workdir)
        cand = None
        for root, _, files in os.walk(workdir):
            if "saved_model.pb" in files:
                cand = root; break
        if not cand:
            raise RuntimeError("No SavedModel (saved_model.pb) found inside the zip.")
        return cand

    low = model_path.lower()
    target = _resolve_zip(model_path) if low.endswith(".zip") else model_path

    # 1) Try WITHOUT preprocess_input (means no internal Lambda)
    try:
        m = _try_load_no_custom(target)
        return m, None, None, False  # (model, used_backbone, used_pre_fn, has_internal_pre)
    except Exception:
        pass

    # 2) Fall back WITH candidates (selected backbone first)
    order = []
    if backbone_hint in PREPROCESS_CANDIDATES:
        order.append((backbone_hint, PREPROCESS_CANDIDATES[backbone_hint]))
    for name, fn in PREPROCESS_CANDIDATES.items():
        if name != backbone_hint:
            order.append((name, fn))

    last_err = None
    for name, fn in order:
        try:
            co = dict(CUSTOM_OBJECTS_BASE); co["preprocess_input"] = fn
            m = tf.keras.models.load_model(target, compile=False, safe_mode=False, custom_objects=co)
            return m, name, fn, True   # internal Lambda present
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Model could not be deserialized. Last error: {last_err}")

# === Gradio UI ===
with gr.Blocks(title="Keras Image Classifier") as demo:
    gr.Markdown("## Keras Image Classifier\nUpload your model (.keras/.h5/.zip) and **class_names.txt**, pick the backbone if needed, then select an image to get predictions.")

    with gr.Row():
        model_file = gr.File(
            label="Model File (.keras / .h5 / SavedModel .zip)",
            file_types=[".keras", ".h5", ".zip"],
            type="filepath"
        )
        class_file = gr.File(
            label="Class Names File (class_names.txt)",
            file_types=[".txt"],
            type="filepath"
        )
        backbone_dropdown = gr.Dropdown(
            choices=["EfficientNetB0", "MobileNetV2", "ResNet50", "InceptionV3"],
            label="Backbone (used for deserialization if needed)",
            value="MobileNetV2"
        )

    load_btn = gr.Button("Load Model", variant="primary")
    status = gr.Markdown()

    # States
    model_state = gr.State()
    input_size_state = gr.State()
    channels_state = gr.State()
    class_names_state = gr.State()
    preprocess_fn_state = gr.State()
    backbone_used_state = gr.State()
    has_internal_pre_state = gr.State()  # <— moved here with the others

    with gr.Row():
        image_in = gr.Image(type="pil", label="Input Image")
        predict_btn = gr.Button("Predict", variant="primary")

    label_out = gr.Label(num_top_classes=5, label="Top-5 Predictions")

    def on_load(model_path, class_path, backbone_hint):
        if not model_path:
            return "Please choose a model.", None, None, None, None, None, None, None
        try:
            m, used_backbone, used_pre, has_internal = _load_model_any(model_path, backbone_hint)
            (w, h), c = _infer_input_size(m)
            classes = _read_class_names(class_path)
            msg = f"Model loaded. Input shape: {(h,w,c)}"
            if has_internal:
                msg += " | Preprocessing: inside model (no external scaling)."
            else:
                msg += f" | External preprocessing: {used_backbone or '/255'}"
            if classes:
                msg += f" | {len(classes)} classes loaded."
            else:
                msg += " | No class_names.txt found, using class_0, class_1, ..."
            # If internal preprocess present, do NOT pass a preprocess fn; also do NOT /255
            preprocess_for_runtime = None if has_internal else used_pre  # may be None -> then we will /255
            return msg, m, (w, h), c, classes, preprocess_for_runtime, (used_backbone or "/255"), has_internal
        except Exception as e:
            return f"Failed to load model: {e}", None, None, None, None, None, None, None

    load_btn.click(
        on_load,
        inputs=[model_file, class_file, backbone_dropdown],
        outputs=[status, model_state, input_size_state, channels_state, class_names_state, preprocess_fn_state, backbone_used_state, has_internal_pre_state]
    )

    def on_predict(img, model, input_size, channels, class_names, preprocess_fn, backbone_used, has_internal_pre):
        if model is None:
            return {"error": 1.0}
        if img is None:
            return {"no_image": 1.0}
        try:
            x = _prep_image(
                img, input_size, channels,
                preprocess=(None if has_internal_pre else preprocess_fn),
                scale_255=(False if has_internal_pre else (preprocess_fn is None))
            )
            y = model.predict(x, verbose=0)
            y = y[0] if isinstance(y, (list, tuple)) else y
            y = np.array(y).reshape(-1)

            # Binary case
            if y.shape == () or y.shape == (1,):
                p = float(y if y.shape == () else y[0])
                p = 1 / (1 + np.exp(-p)) if (p < 0 or p > 1) else p
                return {"positive": p, "negative": 1.0 - p}

            # Multiclass case
            if class_names and len(class_names) == y.shape[-1]:
                return {cls: float(p) for cls, p in zip(class_names, y)}
            return {f"class_{i}": float(p) for i, p in enumerate(y)}
        except Exception as e:
            return {f"error: {e}": 1.0}

    predict_btn.click(
        on_predict,
        inputs=[image_in, model_state, input_size_state, channels_state, class_names_state, preprocess_fn_state, backbone_used_state, has_internal_pre_state],
        outputs=[label_out]
    )

demo.queue().launch()