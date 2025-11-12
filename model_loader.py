import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import io
import numpy as np
import logging
import random 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. MODEL ARCHITECTURE DEFINITION (COPIED IN FULL) ---

# Helper Class 1: DoubleConv
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

# Helper Function for UNet
def CBR(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

# Main Model Class: UNet
class UNet(nn.Module):
    def __init__(self, n_classes=1):
        super(UNet, self).__init__()

        # Encoder (Using sequential blocks)
        self.enc1 = nn.Sequential(CBR(3, 64), CBR(64, 64))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(CBR(64, 128), CBR(128, 128))
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(CBR(128, 256), CBR(256, 256))
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = nn.Sequential(CBR(256, 512), CBR(512, 512))
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(CBR(512, 1024), CBR(1024, 1024))

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = nn.Sequential(CBR(1024, 512), CBR(512, 512))

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = nn.Sequential(CBR(512, 256), CBR(256, 256))

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(CBR(256, 128), CBR(128, 128))

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(CBR(128, 64), CBR(64, 64))

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


# --- 2. Configuration and Model Loading (BYPASSED) ---
MODEL_PATH = 'unet_model.pth' 
IMG_SIZE = (256, 256)
N_CLASSES = 1

# FIX: Set model to True to bypass loading error and allow API function to run
model = True 
logging.info("PyTorch Model Bypass Activated: Server will return simulated results.")


# --- 3. Prediction Function (MOCK DATA GENERATOR) ---
def get_prediction_and_remedy(image_bytes):
    # Check if we are in bypass mode
    if model is not True: 
        return {'disease': 'Error: Model Not Initialized', 'confidence': 0.0, 'remedy': 'Check server logs for model loading failure.'}

    try:
        # Load the image just to confirm the uploaded file is valid
        Image.open(io.BytesIO(image_bytes))
        
        # --- RETURN SIMULATED RESULT FOR DEMONSTRATION ---
        
        disease_name = "Simulated Early Blight (Demo Success)"
        # Score between 85% and 95%
        confidence_score = 0.85 + random.random() * 0.1 
        
        mock_result = {
            'disease': disease_name,
            'confidence': float(confidence_score),
            'remedy': 'Apply a systemic fungicide immediately and ensure the plant has improved air circulation. (Result from Simulation)'
        }
        
        logging.info(f"API call successful. Returning mock result for: {disease_name}")
        return mock_result

    except Exception as e:
        # If the uploaded file is corrupted or not a valid image
        logging.error(f"Image Load Error during bypass: {e}")
        return {'disease': 'Image Processing Error', 'confidence': 0.0, 'remedy': 'The uploaded file could not be read by the server.'}