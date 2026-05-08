import os
import io
import torch
import torch.nn as nn

models = None
transforms = None
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import logging
import argparse

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
model_loaded = False
class_names = ['Healthy', 'Diseased']


def create_model(pretrained: bool = False):
    """Create a VGG16 model. If pretrained=True, use ImageNet weights for feature extractor."""
    global models

    try:
        if models is None:
            from torchvision import models as _models
            models = _models

        try:
            if pretrained:
                weights = models.VGG16_Weights.DEFAULT
            else:
                weights = None
            m = models.vgg16(weights=weights)
        except Exception:
            m = models.vgg16(pretrained=pretrained)

        m.classifier[6] = nn.Linear(4096, 2)
        return m
    except Exception as e:
        logger.warning("Could not import torchvision.models (falling back to small model): %s", e)
        fallback = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(3, 2)
        )
        return fallback


def load_model(model_path: str):
    """Try to load a model checkpoint.

    Accepts either a state_dict (dict) or a full serialized model object saved with torch.save(model).
    If the global `model` is None this will create a default VGG model (no pretrained weights).
    """
    global model, model_loaded
    if model is None:
        logger.info('Model object not initialized. Creating default VGG model (no pretrained weights).')
        model = create_model(pretrained=False)

    if os.path.exists(model_path):
        try:
            loaded = torch.load(model_path, map_location=device)
            if isinstance(loaded, dict):
                model.load_state_dict(loaded)
            else:
                model = loaded

            model = model.to(device)
            model.eval()
            model_loaded = True
            logger.info("✅ VGG16 Model Loaded from %s", model_path)
        except Exception as e:
            model_loaded = False
            logger.exception("Failed to load model '%s': %s", model_path, e)
    else:
        logger.warning("Model file not found at %s.", model_path)
        logger.info("If you want to run inference without a checkpoint use '--pretrained' to initialize ImageNet weights.")

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

def get_transform():
    """Lazily build image transform. Falls back to a simple PIL->tensor converter if torchvision.transforms
    is unavailable."""
    global transforms
    try:
        if transforms is None:
            from torchvision import transforms as _transforms
            transforms = _transforms

        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
    except Exception as e:
        logger.warning("Could not import torchvision.transforms; using fallback transform: %s", e)
        import numpy as _np

        def fallback_transform(img: Image.Image):
            img = img.resize((224, 224)).convert('RGB')
            arr = torch.from_numpy(_np.array(img)).permute(2, 0, 1).float() / 255.0
            # normalize
            mean = torch.tensor(imagenet_mean).view(3, 1, 1)
            std = torch.tensor(imagenet_std).view(3, 1, 1)
            return (arr - mean) / std

        return fallback_transform

@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({"error": "Model not loaded. Provide a valid model file on startup."}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as e:
        logger.exception("Failed to read uploaded image: %s", e)
        return jsonify({"error": "Invalid image uploaded."}), 400

    tf = get_transform()
    img_tensor = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0].tolist()
        pred_class = int(torch.argmax(output, dim=1).item())

    result = class_names[pred_class] if pred_class < len(class_names) else str(pred_class)

    return jsonify({
        "prediction": result,
        "class_id": pred_class,
        "probabilities": probs
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model_loaded": model_loaded}), 200

def parse_args():
    parser = argparse.ArgumentParser(description="VGG16 Flask inference server")
    parser.add_argument('--model-path', type=str, default='vgg16_plant_model.pth', help='Path to model .pth file')
    parser.add_argument('--pretrained', action='store_true', help='Use ImageNet pretrained weights for VGG feature extractor')
    parser.add_argument('--port', type=int, default=5001, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run Flask in debug mode')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if os.path.exists(args.model_path):
        logger.info("Found checkpoint at %s. Loading...", args.model_path)
        model = create_model(pretrained=False)
        load_model(args.model_path)
    else:
        try_pretrained = args.pretrained
        try:
            logger.info("No checkpoint found at %s. Initializing VGG16 pretrained=%s", args.model_path, try_pretrained)
            model = create_model(pretrained=try_pretrained)
            model = model.to(device)
            model.eval()
            model_loaded = True
            logger.info("Initialized VGG16 (pretrained=%s). model_loaded=%s", try_pretrained, model_loaded)
        except Exception as e:
            logger.warning("Failed to initialize pretrained weights: %s", e)
            logger.info("Falling back to randomly initialized VGG16; predictions may be meaningless.")
            model = create_model(pretrained=False)
            model = model.to(device)
            model.eval()
            model_loaded = True

    logger.info('Starting Flask server on 0.0.0.0:%s (debug=%s)', args.port, args.debug)
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)
