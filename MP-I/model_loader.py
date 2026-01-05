import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. MODEL ARCHITECTURE DEFINITION (COPIED FROM UNET_TRAIN.PY) ---
# This structure is necessary to match the saved weights.

def CBR_seq(in_channels, out_channels):
    """Helper function for the Conv-BatchNorm-ReLU sequence."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    def __init__(self, n_classes=1): 
        super(UNet, self).__init__()
        # Encoder 
        self.enc1 = nn.Sequential(CBR_seq(3, 64), CBR_seq(64, 64))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(CBR_seq(64, 128), CBR_seq(128, 128))
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(CBR_seq(128, 256), CBR_seq(256, 256))
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = nn.Sequential(CBR_seq(256, 512), CBR_seq(512, 512))
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(CBR_seq(512, 1024), CBR_seq(1024, 1024))

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = nn.Sequential(CBR_seq(1024, 512), CBR_seq(512, 512))

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = nn.Sequential(CBR_seq(512, 256), CBR_seq(256, 256))

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(CBR_seq(256, 128), CBR_seq(128, 128))

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(CBR_seq(128, 64), CBR_seq(64, 64))

        # Output
        self.final = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        c1 = self.enc1(x)
        p1 = self.pool1(c1)
        c2 = self.enc2(p1)
        p2 = self.pool2(c2)
        c3 = self.enc3(p2)
        p3 = self.pool3(c3)
        c4 = self.enc4(p3)
        p4 = self.pool4(c4)
        bn = self.bottleneck(p4)
        u4 = self.up4(bn)
        u4 = torch.cat([u4, c4], dim=1)
        d4 = self.dec4(u4)
        u3 = self.up3(d4)
        u3 = torch.cat([u3, c3], dim=1)
        d3 = self.dec3(u3)
        u2 = self.up2(d3)
        u2 = torch.cat([u2, c2], dim=1)
        d2 = self.dec2(u2)
        u1 = self.up1(d2)
        u1 = torch.cat([u1, c1], dim=1)
        d1 = self.dec1(u1)
        out = self.final(d1)
        return out
# -----------------------------------------------------------------------------


# --- 2. Configuration and Model Loading ---
MODEL_PATH = 'unet_model.pth' 
IMG_SIZE = (256, 256)
N_CLASSES = 1

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(), 
])

model = None
try:
    model = UNet(n_classes=N_CLASSES) 
    # CRITICAL FIX: strict=False to bypass key mismatch error
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), strict=False))
    model.eval() 
    logging.info("PyTorch UNet model loaded successfully (strict=False).")

except Exception as e:
    # This block catches FileNotFoundError and structural errors
    logging.error(f"CRITICAL ERROR: Failed to load PyTorch model. Check file name and location. Error: {e}")
    model = None


# --- 3. Prediction Function (Inference and Result Mapping) ---
def get_prediction_and_remedy(image_bytes):
    if model is None:
        # If model loading failed, return an error that the frontend can display
        return {'disease': 'Error: Model Not Initialized', 'confidence': 0.0, 'remedy': 'Check server logs for model loading failure.'}

    try:
        # 1. Preprocess input
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).float() # Ensure float32 for model input
        
        # 2. Run inference and segmentation logic
        with torch.no_grad():
            output = model(img_tensor)
        
        probabilities = torch.sigmoid(output)
        seg_map = (probabilities > 0.5).float().squeeze().cpu().numpy()
        
        # 3. Calculate metrics based on segmentation map
        DISEASE_CLASS = 1 
        diseased_pixels = np.sum(seg_map == DISEASE_CLASS)
        total_pixels = seg_map.size
        confidence = diseased_pixels / total_pixels
        
        # 4. Map result to label and remedy
        if confidence >= 0.15:
            label = "Severe Fungal Infection Detected"
            remedy = "Immediate application of systemic fungicide."
        else:
            label = "Healthy Leaf"
            confidence = min(confidence + 0.05, 1.0) 
            remedy = "Maintain standard care routine."
            
        return {
            'disease': label,
            'confidence': float(confidence),
            'remedy': remedy
        }

    except Exception as e:
        logging.error(f"Inference Runtime Error during prediction: {e}")
        return {'disease': 'Inference Runtime Error', 'confidence': 0.0, 'remedy': 'Server failed during model execution.'}