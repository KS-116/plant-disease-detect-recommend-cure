Run the Flask inference server

This repo includes a Flask inference server for a VGG16-based classifier in `vgg_flask.py`.

Basic usage (defaults to `vgg16_plant_model.pth` and port `5001`):

```bash
python3 vgg_flask.py
```

Provide a specific checkpoint path:

```bash
python3 vgg_flask.py --model-path path/to/your_vgg_weights.pth
```

Initialize VGG with ImageNet pretrained weights (useful if you don't have a checkpoint):

```bash
python3 vgg_flask.py --pretrained
```

Other options:

```bash
python3 vgg_flask.py --port 8080 --debug
```

Health and prediction examples

- Health check (returns JSON with `model_loaded` boolean):

```bash
curl http://127.0.0.1:5001/health
```

- Predict (when a model is loaded):

```bash
curl -F "image=@/path/to/image.jpg" http://127.0.0.1:5001/predict
```

Notes
Notes
- If no model file is found at startup the server will try to initialize a VGG16 model only when you pass `--pretrained` (this may download ImageNet weights).
- If you do NOT pass `--pretrained` and no checkpoint exists, the server will fall back to a small, fast-to-import model so the API is usable immediately (predictions will be meaningless without a trained checkpoint).
- To use a trained VGG checkpoint (preferred), provide a path with `--model-path path/to/checkpoint.pth`.
- `class_names` in `vgg_flask.py` is set to `['Healthy', 'Diseased']` — ensure this matches your training labels.
