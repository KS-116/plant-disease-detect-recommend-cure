---
title: PlantCare AI
emoji: 🌿
colorFrom: green
colorTo: green
sdk: gradio
sdk_version: 4.44.1
python_version: 3.10.13
app_file: app.py
pinned: false
license: mit
---

# 🌿 PlantCare AI — Plant Disease Detection

AI-powered plant disease detection and treatment recommendations built with EfficientNet-B4,
trained on the PlantVillage + PlantDoc combined dataset.

## Supported Plants
Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper,
Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

## How to Use
1. Upload a clear photo of a plant leaf
2. Click **Analyse Leaf**
3. Get instant disease identification and treatment recommendations

## Tips for Best Results
- Photograph a single leaf in bright natural light
- Fill the frame with the leaf
- Ensure the image is sharp and in focus
- For disease detection, photograph the most affected leaf

## Model
- Architecture: EfficientNet-B4
- Training data: PlantVillage + PlantDoc (merged dataset)
- Classes: 38 plant disease categories
- Input size: 224×224px
- Inference: 5-pass Test Time Augmentation (TTA)

---
*For educational purposes. Always consult a qualified agronomist for critical crop decisions.*
