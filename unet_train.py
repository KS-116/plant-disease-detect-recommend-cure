import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# ========================
# 1. Dataset
# ========================
class LeafDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.pairs = []  # store valid (image, mask) paths

        # Collect all valid image–mask pairs
        for root, _, files in os.walk(img_dir):
            for f in files:
                if not f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    continue

                img_path = os.path.join(root, f)
                rel_path = os.path.relpath(img_path, img_dir)
                rel_dir = os.path.dirname(rel_path)
                base_name = os.path.splitext(os.path.basename(img_path))[0]

                # Candidate mask names
                mask_candidates = [
                    os.path.join(mask_dir, rel_dir, base_name + ".jpg"),
                    os.path.join(mask_dir, rel_dir, base_name + ".png"),
                    os.path.join(mask_dir, rel_dir, base_name + "_mask.jpg"),
                    os.path.join(mask_dir, rel_dir, base_name + "_mask.png"),
                    os.path.join(mask_dir, rel_dir, base_name + "_final_masked.jpg"),
                    os.path.join(mask_dir, rel_dir, base_name + "_final_masked.png"),
                ]

                for mask_path in mask_candidates:
                    if os.path.exists(mask_path):
                        self.pairs.append((img_path, mask_path))
                        break  # found valid mask

        if len(self.pairs) == 0:
            raise ValueError(f"No valid image–mask pairs found in {img_dir}")

        print(f"✅ Found {len(self.pairs)} valid image–mask pairs in '{img_dir}'")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise ValueError(f"Error loading {img_path} or {mask_path}")

        # Resize and normalize
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        mask = (mask > 127).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(image), torch.tensor(mask)

# ========================
# 2. U-Net Model
# ========================
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


class UNet(nn.Module):
    def __init__(self, n_classes=1):
        super(UNet, self).__init__()

        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # Encoder
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

# ========================
# 3. Dice Loss
# ========================
def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))


# ========================
# 4. Validation
# ========================
def validate(model, loader, device):
    model.eval()
    preds, gts = [], []
    val_loss, correct, total = 0, 0, 0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            pred = (torch.sigmoid(outputs) > 0.5).float()
            preds.append(pred.cpu().numpy())
            gts.append(masks.cpu().numpy())

            correct += (pred == masks).float().sum().item()
            total += masks.numel()

    acc = correct / total
    preds = np.concatenate(preds).astype(int).flatten()
    gts = np.concatenate(gts).astype(int).flatten()
    dice = (2. * np.sum(preds * gts)) / (np.sum(preds) + np.sum(gts) + 1e-7)
    return val_loss / len(loader), acc, dice, gts, preds


# ========================
# 5. Training Loop
# ========================
def train_model(model, train_loader, val_loader, device, epochs=10, save_path="best_unet_model.pth"):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    best_val_acc = 0.0

    print("\n🧠 Training Started...\n")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch}/{epochs}")
        for imgs, masks in loop:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks) + dice_loss(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == masks).float().sum().item()
            total += masks.numel()
            acc = correct / total
            loop.set_postfix(loss=loss.item(), acc=acc)

        val_loss, val_acc, val_dice, gts, preds = validate(model, val_loader, device)
        print(f"Epoch {epoch}/{epochs} - loss: {running_loss/len(train_loader):.4f} - acc: {acc:.4f} - "
              f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f} - val_dice: {val_dice:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print("✅ Best model saved.\n")

    print("\nTraining complete. Loading best model for evaluation...\n")
    model.load_state_dict(torch.load(save_path))
    val_loss, val_acc, val_dice, gts, preds = validate(model, val_loader, device)
    print(f"\nValidation Accuracy: {val_acc*100:.2f}%")
    print(f"Average Dice Score: {val_dice:.4f}\n")
    print("Classification Report:")
    print(classification_report(gts, preds, target_names=["Background", "Leaf/ROI"]))
    print("Confusion Matrix:")
    print(confusion_matrix(gts, preds))
    print("\n✅ Model evaluation complete.\n")


# ========================
# 6. Main
# ========================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dataset = LeafDataset("dataset/images", "dataset/masks")
    val_dataset = LeafDataset("dataset/val_images", "dataset/val_masks")

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    model = UNet().to(device)
    train_model(model, train_loader, val_loader, device, epochs=2, save_path="best_unet_model.pth")
