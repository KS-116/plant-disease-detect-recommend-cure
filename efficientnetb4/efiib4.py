import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm
from collections import Counter
from pathlib import Path

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import label_binarize

from transformers import AutoImageProcessor, AutoModelForImageClassification
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


# ============================================================
# CONFIGURATION
# ============================================================

MERGED_DATASET_PATH = r"D:\Plant disease\Merged_Dataset"
OUTPUT_DIR          = r"D:\Plant disease\final_model"
MODEL_NAME          = "google/efficientnet-b4"

IMG_SIZE            = 224
BATCH_SIZE          = 16        
EPOCHS              = 25
GRAD_CLIP_NORM      = 1.0
EARLY_STOP_PATIENCE = 6
NUM_WORKERS         = 4         

LABEL_SMOOTHING     = 0.05      
MIXUP_ALPHA         = 0.2       
CUTMIX_ALPHA        = 1.0
MIXUP_CUTMIX_PROB   = 0.4       

PLANTDOC_BOOST      = 8.0

CONFIDENCE_THRESHOLD = 0.65

UNFREEZE_EPOCH  = 6
PHASE1_LAYERS   = ["blocks.5", "blocks.6"]
PHASE2_LAYERS   = ["blocks.3", "blocks.4", "blocks.5", "blocks.6"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SEED & DEVICE
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda")
print(f"Using device : {torch.cuda.get_device_name(0)}")
print(f"VRAM         : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ============================================================
# LOAD PROCESSOR
# ============================================================

print(f"\nLoading processor for {MODEL_NAME}...")
processor  = AutoImageProcessor.from_pretrained(MODEL_NAME)
IMAGE_MEAN = processor.image_mean
IMAGE_STD  = processor.image_std
print(f"mean : {IMAGE_MEAN}")
print(f"std  : {IMAGE_STD}")


# ============================================================
# AUGMENTATION
# ============================================================
# Deliberately lighter than previous runs to fix the underfitting
# (60% train / 88% val gap) seen with B3.
# Rule: smaller gap between train and val transform difficulty
# = more honest training signal = better real-world accuracy.

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0), ratio=(0.8, 1.25)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(25),
    transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), shear=8),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    transforms.RandomAutocontrast(p=0.3),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    transforms.RandomGrayscale(p=0.04),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.10),
                             ratio=(0.3, 3.0), value="random"),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
])

tta_transforms = [
    # 1. Plain resize
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ]),
    # 2. Larger + centre crop
    transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ]),
    # 3. Horizontal flip
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ]),
    # 4. Vertical flip
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ]),
    # 5. Larger + random crop
    transforms.Compose([
        transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ]),
]


# ============================================================
# MIXUP + CUTMIX
# ============================================================

def mixup_batch(images, labels, alpha):
    """lam clamped >= 0.5 so labels_a is always the dominant label."""
    lam   = np.random.beta(alpha, alpha)
    lam   = max(lam, 1.0 - lam)
    idx   = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1 - lam) * images[idx]
    return mixed, labels, labels[idx], lam


def cutmix_batch(images, labels, alpha):
    """Paste a patch from one image onto another."""
    lam          = np.random.beta(alpha, alpha)
    lam          = max(lam, 1.0 - lam)
    idx          = torch.randperm(images.size(0), device=images.device)
    B, C, H, W   = images.shape
    cut_ratio    = np.sqrt(1.0 - lam)
    cut_h        = int(H * cut_ratio)
    cut_w        = int(W * cut_ratio)
    cx           = random.randint(0, W)
    cy           = random.randint(0, H)
    x1 = max(cx - cut_w // 2, 0);  x2 = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0);  y2 = min(cy + cut_h // 2, H)
    mixed        = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
    lam          = 1.0 - ((x2 - x1) * (y2 - y1) / (H * W))
    lam          = max(lam, 1.0 - lam)
    return mixed, labels, labels[idx], lam


def mixed_criterion(criterion, pred, a, b, lam):
    return lam * criterion(pred, a) + (1 - lam) * criterion(pred, b)


# ============================================================
# FREEZE HELPER
# ============================================================

def set_trainable_layers(model, block_names: list):
    """
    Freeze entire EfficientNet backbone then unfreeze requested blocks.
    Classifier head always stays trainable.
    Prints warning with actual param names if block_names match nothing.
    """
    for param in model.efficientnet.parameters():
        param.requires_grad = False

    unfrozen = 0
    for name, param in model.efficientnet.named_parameters():
        if any(bn in name for bn in block_names):
            param.requires_grad = True
            unfrozen += 1

    for param in model.classifier.parameters():
        param.requires_grad = True

    if unfrozen == 0:
        print(f"\n  WARNING: block_names={block_names} matched NOTHING.")
        print("  First 10 param names for reference:")
        for i, (n, _) in enumerate(model.efficientnet.named_parameters()):
            print(f"    {n}")
            if i >= 9: break
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        print(f"  Trainable: {trainable:,} / {total:,}  ({unfrozen} tensors unfrozen)")


# ============================================================
# LOAD DATASET
# ============================================================
def main():
    print("\n[1/5] Loading dataset...")
    if not os.path.exists(MERGED_DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at: {MERGED_DATASET_PATH}\n"
            "Check MERGED_DATASET_PATH points at your merged folder."
        )

    full_dataset = datasets.ImageFolder(MERGED_DATASET_PATH)
    class_names  = full_dataset.classes
    NUM_CLASSES  = len(class_names)
    all_paths    = [s[0] for s in full_dataset.samples]
    all_targets  = full_dataset.targets

    id2label = {str(i): c for i, c in enumerate(class_names)}
    label2id = {c: str(i) for i, c in enumerate(class_names)}

    is_plantdoc = [Path(p).name.startswith("plantdoc_") for p in all_paths]
    n_pd        = sum(is_plantdoc)
    n_pv        = len(full_dataset) - n_pd

    print(f"  Classes      : {NUM_CLASSES}")
    print(f"  Total images : {len(full_dataset):,}")
    print(f"  PlantVillage : {n_pv:,}")
    print(f"  PlantDoc     : {n_pd:,}  ({100 * n_pd / len(full_dataset):.1f}%)")

    if n_pd == 0:
        raise RuntimeError(
            "No PlantDoc images found. Filenames must start with 'plantdoc_'.\n"
            "Re-run your merge script and check the output folder."
        )

    with open(os.path.join(OUTPUT_DIR, "labels.json"), "w") as f:
        json.dump(full_dataset.class_to_idx, f, indent=2)


    # ============================================================
    # TRAIN / VALIDATION SPLIT
    # ============================================================

    indices      = list(range(len(full_dataset)))
    stratify_key = [
        f"{all_targets[i]}_{'pd' if is_plantdoc[i] else 'pv'}"
        for i in indices
    ]

    # Fallback to class-only if any combo has just 1 sample
    if any(v == 1 for v in Counter(stratify_key).values()):
        print("\n[INFO] Some class+source combos have 1 sample — class-only stratification.")
        stratify_key = all_targets

    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, stratify=stratify_key, random_state=SEED
    )

    pd_in_train = sum(1 for i in train_idx if is_plantdoc[i])
    pd_in_val   = sum(1 for i in val_idx   if is_plantdoc[i])
    print(f"\n  Train : {len(train_idx):,}  ({pd_in_train} PlantDoc)")
    print(f"  Val   : {len(val_idx):,}    ({pd_in_val} PlantDoc)")

    if pd_in_val == 0:
        raise RuntimeError(
            "No PlantDoc images in val set.\n"
            "Reduce test_size or check your merged folder."
        )


    # ============================================================
    # DATASETS & SOURCE-AWARE SAMPLER
    # ============================================================

    train_dataset = torch.utils.data.Subset(
        datasets.ImageFolder(MERGED_DATASET_PATH, transform=train_transform), train_idx
    )
    val_dataset = torch.utils.data.Subset(
        datasets.ImageFolder(MERGED_DATASET_PATH, transform=val_transform), val_idx
    )

    train_labels   = [all_targets[i] for i in train_idx]
    train_is_pd    = [is_plantdoc[i] for i in train_idx]
    class_counts   = Counter(train_labels)
    sample_weights = [
        (1.0 / class_counts[lbl]) * (PLANTDOC_BOOST if pd else 1.0)
        for lbl, pd in zip(train_labels, train_is_pd)
    ]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size  = BATCH_SIZE,
        sampler     = sampler,
        num_workers = NUM_WORKERS,
        pin_memory  = True,
        persistent_workers = NUM_WORKERS > 0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = NUM_WORKERS,
        pin_memory  = True,
        persistent_workers = NUM_WORKERS > 0
    )


    # ============================================================
    # LOAD MODEL
    # ============================================================

    print(f"\n[2/5] Loading {MODEL_NAME}...")
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels              = NUM_CLASSES,
        id2label                = id2label,
        label2id                = label2id,
        ignore_mismatched_sizes = True,
    )
    model.to(device)
    print("Model loaded.")

    set_trainable_layers(model, PHASE1_LAYERS)


    # ============================================================
    # LOSS / OPTIMIZER / SCHEDULER
    # ============================================================

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-4, weight_decay=0.01,
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=UNFREEZE_EPOCH, T_mult=2, eta_min=1e-6
    )

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "lr":         [],
    }

    best_val_acc      = 0.0
    epochs_no_improve = 0
    best_model_path   = os.path.join(OUTPUT_DIR, "best_model.pth")


    # ============================================================
    # TRAINING LOOP
    # ============================================================

    print("\n[3/5] Training...")

    for epoch in range(EPOCHS):

        current_lr = scheduler.get_last_lr()[0]
        print(f"\nEpoch {epoch + 1}/{EPOCHS}  (lr={current_lr:.2e})")

        # Progressive unfreeze
        if epoch == UNFREEZE_EPOCH:
            print("  → Unfreezing deeper blocks...")
            set_trainable_layers(model, PHASE2_LAYERS)
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-4, weight_decay=0.01,
            )
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=max(4, (EPOCHS - UNFREEZE_EPOCH) // 2),
                T_mult=1, eta_min=1e-6,
            )

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc="  Train", leave=False):
            images, labels = images.to(device), labels.to(device)

           
            if random.random() < MIXUP_CUTMIX_PROB:
                if random.random() < 0.5:
                    images, la, lb, lam = mixup_batch(images, labels, MIXUP_ALPHA)
                else:
                    images, la, lb, lam = cutmix_batch(images, labels, CUTMIX_ALPHA)
                optimizer.zero_grad()
                outputs = model(pixel_values=images).logits
                loss    = mixed_criterion(criterion, outputs, la, lb, lam)
            else:
               
                la = labels
                optimizer.zero_grad()
                outputs = model(pixel_values=images).logits
                loss    = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            bn          = la.size(0)
            train_loss += loss.item() * bn
            
            correct    += (outputs.argmax(1) == la).sum().item()
            total      += bn

        avg_tl = train_loss / total
        t_acc  = correct / total
        history["train_loss"].append(avg_tl)
        history["train_acc"].append(t_acc)
        history["lr"].append(current_lr)
        print(f"  Train  — loss: {avg_tl:.4f}  acc: {t_acc:.4f}")

        # ── Validation ─────────────────────────────────────────────────────
        model.eval()
        val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="  Val  ", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(pixel_values=images).logits
                loss    = criterion(outputs, labels)
                preds   = torch.softmax(outputs, dim=1).argmax(1)
                val_loss += loss.item() * labels.size(0)
                correct  += (preds == labels).sum().item()
                total    += labels.size(0)

        avg_vl = val_loss / total
        v_acc  = correct / total
        history["val_loss"].append(avg_vl)
        history["val_acc"].append(v_acc)

        # Overfitting gap warning
        gap          = t_acc - v_acc
        gap_note     = f"  [OVERFIT gap={gap:.3f}]" if gap > 0.15 else ""
        underfit_note= f"  [UNDERFIT gap={abs(gap):.3f}]" if gap < -0.10 else ""
        print(f"  Val    — loss: {avg_vl:.4f}  acc: {v_acc:.4f}{gap_note}{underfit_note}")

        scheduler.step()

        if v_acc > best_val_acc:
            best_val_acc      = v_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ Best model saved (val_acc={best_val_acc:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement ({epochs_no_improve}/{EARLY_STOP_PATIENCE})")

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping after {epoch + 1} epochs.")
            break


    # ============================================================
    # RELOAD BEST WEIGHTS
    # ============================================================

    print(f"\n[4/5] Reloading best weights (val_acc={best_val_acc:.4f})...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()


    # ============================================================
    # FINAL EVALUATION WITH TTA — SPLIT BY SOURCE
    # ============================================================

    print("Running TTA evaluation split by source...")
    print("(PlantDoc accuracy = what your website will achieve on real photos)")

    raw_val = torch.utils.data.Subset(
        datasets.ImageFolder(MERGED_DATASET_PATH, transform=None), val_idx
    )

    results = {
        "all":          {"preds": [], "labels": [], "probs": []},
        "plantvillage": {"preds": [], "labels": [], "probs": []},
        "plantdoc":     {"preds": [], "labels": [], "probs": []},
    }

    with torch.no_grad():
        for idx, (pil_img, label) in enumerate(tqdm(raw_val, desc="  TTA eval")):
            tta_probs = []
            for tfm in tta_transforms:
                t    = tfm(pil_img).unsqueeze(0).to(device)
                prob = torch.softmax(
                    model(pixel_values=t).logits, dim=1
                ).squeeze(0).cpu().numpy()
                tta_probs.append(prob)

            avg_prob = np.mean(tta_probs, axis=0)
            pred     = int(np.argmax(avg_prob))
            src      = "plantdoc" \
                    if Path(all_paths[val_idx[idx]]).name.startswith("plantdoc_") \
                    else "plantvillage"

            for key in ["all", src]:
                results[key]["preds"].append(pred)
                results[key]["labels"].append(label)
                results[key]["probs"].append(avg_prob)


    # ============================================================
    # METRICS
    # ============================================================

    def report(title, labels, preds, class_names):
        print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")
        if not labels:
            print("  No samples.")
            return {}
        acc  = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, average="macro", zero_division=0)
        rec  = recall_score(labels, preds, average="macro", zero_division=0)
        f1   = f1_score(labels, preds, average="macro", zero_division=0)
        print(f"  Samples  : {len(labels):,}")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1       : {f1:.4f}")
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


    m_all = report("ALL IMAGES (best checkpoint + TTA)",
                results["all"]["labels"], results["all"]["preds"], class_names)
    m_pv  = report("PLANTVILLAGE ONLY — clean lab images",
                results["plantvillage"]["labels"],
                results["plantvillage"]["preds"], class_names)
    m_pd  = report("PLANTDOC ONLY — real phone photos  ← website accuracy",
                results["plantdoc"]["labels"],
                results["plantdoc"]["preds"], class_names)

    print("\nClassification Report (all val images)\n")
    print(classification_report(
        results["all"]["labels"], results["all"]["preds"],
        target_names=class_names, zero_division=0,
    ))

    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump({
            "model"        : MODEL_NAME,
            "best_val_acc" : best_val_acc,
            "all"          : m_all,
            "plantvillage" : m_pv,
            "plantdoc"     : m_pd,
        }, f, indent=2)
    print(f"\nMetrics saved.")


    # ============================================================
    # PLOTS
    # ============================================================

    epochs_ran = len(history["train_loss"])
    x          = list(range(1, epochs_ran + 1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(x, history["train_loss"], label="Train")
    axes[0].plot(x, history["val_loss"],   label="Val")
    axes[1].plot(x, history["train_acc"],  label="Train")
    axes[1].plot(x, history["val_acc"],    label="Val")
    axes[2].plot(x, history["lr"], color="purple")

    for ax in axes:
        if UNFREEZE_EPOCH < epochs_ran:
            ax.axvline(x=UNFREEZE_EPOCH, color="gray",
                    linestyle="--", label="Unfreeze")
        ax.legend()
        ax.set_xlabel("Epoch")

    axes[0].set_title("Loss");     axes[0].set_ylabel("Loss")
    axes[1].set_title("Accuracy"); axes[1].set_ylabel("Accuracy")
    axes[2].set_title("LR");       axes[2].set_ylabel("LR")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
    plt.close(fig)

    # Confusion matrix
    cm = confusion_matrix(results["all"]["labels"], results["all"]["preds"])
    fig, ax = plt.subplots(
        figsize=(max(12, NUM_CLASSES // 2), max(10, NUM_CLASSES // 2))
    )
    sns.heatmap(cm, xticklabels=class_names, yticklabels=class_names,
                cmap="Blues", ax=ax, annot=(NUM_CLASSES <= 20))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # ROC curve
    y_true_bin = label_binarize(results["all"]["labels"], classes=range(NUM_CLASSES))
    y_score    = np.array(results["all"]["probs"])
    fig, ax    = plt.subplots(figsize=(10, 8))
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        ax.plot(fpr, tpr, lw=1,
                label=f"{class_names[i]} (AUC={auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC Curves (one-vs-rest with TTA)")
    ax.legend(
        loc="lower right" if NUM_CLASSES <= 20 else "upper left",
        bbox_to_anchor=(None if NUM_CLASSES <= 20 else (1.02, 1)),
        fontsize=7,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Plots saved.")


    # ============================================================
    # SAVE MODEL
    # ============================================================

    print("\n[5/5] Saving deployable model...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    # Clean up the raw .pth file — only HuggingFace format needed for website
    if os.path.exists(best_model_path):
        os.remove(best_model_path)
        print("Cleaned up temporary .pth file.")

    with open(os.path.join(OUTPUT_DIR, "inference_config.json"), "w") as f:
        json.dump({
            "model_name"          : MODEL_NAME,
            "img_size"            : IMG_SIZE,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "num_tta_passes"      : len(tta_transforms),
            "note"                : (
                f"Average softmax probs across {len(tta_transforms)} TTA variants "
                f"then argmax. If max(prob) < {CONFIDENCE_THRESHOLD} show "
                f"'Uncertain — please retake photo in better lighting'."
            ),
        }, f, indent=2)

    print(f"\nAll files saved to: {OUTPUT_DIR}")

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  Model                   : {MODEL_NAME}")
    print(f"  Best val accuracy       : {best_val_acc:.4f}")
    print(f"  PlantVillage acc (TTA)  : {m_pv.get('accuracy', 0):.4f}  ← clean images")
    print(f"  PlantDoc acc     (TTA)  : {m_pd.get('accuracy', 0):.4f}  ← real phone photos")
    print(f"\n  Target: PlantDoc acc > 0.75")
    if m_pd.get("accuracy", 0) >= 0.75:
        print("  Result : PASSED — model ready for website deployment.")
    else:
        print("  Result : BELOW TARGET — add more real-world images.")
    print("=" * 60)
    print("\nDone!")


# ============================================================
# WEBSITE INFERENCE GUIDE
# ============================================================
# Copy this function into your website backend:
#
# from transformers import AutoImageProcessor, AutoModelForImageClassification
# from torchvision import transforms
# from PIL import Image
# import torch, numpy as np
#
# model     = AutoModelForImageClassification.from_pretrained(OUTPUT_DIR)
# processor = AutoImageProcessor.from_pretrained(OUTPUT_DIR)
# model.eval()
#
# def predict(image_path):
#     pil_img   = Image.open(image_path).convert("RGB")
#     probs_list = []
#     for tfm in tta_transforms:
#         tensor = tfm(pil_img).unsqueeze(0)
#         with torch.no_grad():
#             logits = model(pixel_values=tensor).logits
#         probs_list.append(torch.softmax(logits, dim=1).squeeze().numpy())
#     avg       = np.mean(probs_list, axis=0)
#     confidence= float(np.max(avg))
#     if confidence < 0.65:
#         return "Uncertain — please retake photo in better lighting", confidence
#     pred_idx  = int(np.argmax(avg))
#     label     = model.config.id2label[str(pred_idx)]
#     return label, confidence
if __name__ == "__main__":
    main()