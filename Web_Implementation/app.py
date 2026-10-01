# ██████   ██████  ██████  ████████ ███████ ██ 
#██       ██    ██ ██   ██    ██       ███  ██ 
#██   ███ ██    ██ ██   ██    ██      ███   ██ 
#██    ██ ██    ██ ██   ██    ██     ███    ██ 
# ██████   ██████  ██████     ██    ███████ ██ 
                                                  

import io
import base64

import numpy as np
import torch
import timm
import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms
from flask import Flask, request, render_template, jsonify


MODEL_NAME = "repvit_m1_1"
MODEL_PATH = r"repvit_brain_tumor_best.pt"
CLASS_NAMES = ["normal", "tumor"]
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

app = Flask(__name__)


def gradcam_overlay(acts, grads, orig_img):
    """Build a simple Grad-CAM heatmap overlay from the last feature map."""
    weights = grads.mean(dim=(1, 2))                            # (C,)
    cam = torch.relu((weights[:, None, None] * acts).sum(0))    # (H, W)

    cam = cam.cpu().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()

    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(orig_img.size, Image.BILINEAR)
    cam_arr = np.array(cam_img) / 255.0

    heatmap = (cm.jet(cam_arr)[:, :, :3] * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap).convert("RGB")

    overlay = Image.blend(orig_img.convert("RGB"), heatmap_img, alpha=0.45)

    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    tensor = transform(img).unsqueeze(0).to(device)

    model.zero_grad(set_to_none=True)
    features = model.forward_features(tensor)
    features.retain_grad()
    outputs = model.forward_head(features)
    probs = torch.softmax(outputs, dim=1)[0]

    prob_normal = probs[0].item()
    prob_tumor = probs[1].item()
    pred_idx = int(probs.argmax().item())
    pred_label = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx].item() * 100

    outputs[0, pred_idx].backward()
    gradcam_b64 = gradcam_overlay(
        features.detach()[0],
        features.grad[0],
        img.resize((IMG_SIZE, IMG_SIZE)),
    )

    return jsonify({
        "label": pred_label,
        "confidence": round(confidence, 1),
        "prob_normal": round(prob_normal * 100, 1),
        "prob_tumor": round(prob_tumor * 100, 1),
        "gradcam": gradcam_b64,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
