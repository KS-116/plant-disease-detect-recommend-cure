import os
import sys
import json
import time
import shutil
import math
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm

from torch.optim import AdamW

from transformers import AutoFeatureExtractor, AutoModelForImageClassification
try:
    from transformers import get_cosine_schedule_with_warmup
except Exception:
    from transformers import get_scheduler
    def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
        return get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedShuffleSplit

DATA_DIR = r"D:\Plant disease\plantvillage dataset"   
MERGE_SUBFOLDERS = ["color", "grayscale", "segmented"]
MERGED_DIR_NAME = "plantvillage_merged_color_grayscale_segmented"
OUTPUT_DIR = os.path.join(os.path.dirname(DATA_DIR), "vit_final_outputs")  
MODEL_NAME = "google/vit-base-patch16-224-in21k"
IMG_SIZE = 224
BATCH_SIZE = 24

NUM_WORKERS = max(0, min(4, (os.cpu_count() or 2) - 1))
SEED = 42
VAL_SPLIT = 0.2
INITIAL_EPOCHS = 6
FINE_TUNE_EPOCHS = 18
MAX_TOTAL_EPOCHS = 1000

REPORT_PDF = "/mnt/data/RG85.pdf"


os.makedirs(OUTPUT_DIR, exist_ok=True)
BEST_HEAD_PATH = os.path.join(OUTPUT_DIR, "vit_best_head.pth")
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "vit_best.pth")
FINAL_MODEL_DIR = os.path.join(OUTPUT_DIR, "vit_final_model")
CLASS_JSON = os.path.join(OUTPUT_DIR, "class_names.json")
HISTORY_CSV = os.path.join(OUTPUT_DIR, "training_history.csv")
METRICS_CSV = os.path.join(OUTPUT_DIR, "per_class_metrics.csv")
METRICS_JSON = os.path.join(OUTPUT_DIR, "overall_metrics.json")
CONFUSION_CSV = os.path.join(OUTPUT_DIR, "confusion_matrix.csv")
CONFUSION_PNG = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
ROC_PNG = os.path.join(OUTPUT_DIR, "roc_auc.png")
AUC_CSV = os.path.join(OUTPUT_DIR, "per_class_auc.csv")
ARTIFACTS_JSON = os.path.join(OUTPUT_DIR, "artifacts.json")


def ensure_output_outside_dataset(data_dir, out_dir):
    data_dir = os.path.abspath(data_dir)
    out_dir = os.path.abspath(out_dir)
    try:
        common = os.path.commonpath([data_dir, out_dir])
    except Exception:
        common = ""
    if common == data_dir:
        parent = os.path.dirname(data_dir)
        new_out = os.path.join(parent, os.path.basename(out_dir))
        if os.path.exists(new_out):
            i = 1
            while os.path.exists(new_out + f"_{i}"):
                i += 1
            new_out = new_out + f"_{i}"
        if os.path.exists(out_dir):
            print(f"[INFO] Moving existing output folder out of dataset:\n  {out_dir} -> {new_out}")
            shutil.move(out_dir, new_out)
        else:
            print(f"[INFO] Setting output folder outside dataset:\n  {out_dir} -> {new_out}")
        return new_out
    return out_dir

def make_merged_dataset(data_dir, merge_subfolders, merged_dir_name):
    parent = os.path.dirname(data_dir)
    merged_root = os.path.join(parent, merged_dir_name)
    if os.path.exists(merged_root):
        print(f"[INFO] Merged folder already exists: {merged_root}")
        return merged_root
    os.makedirs(merged_root, exist_ok=True)
    created = set()
    for sub in merge_subfolders:
        candidate = os.path.join(data_dir, sub)
        if not os.path.isdir(candidate):
            print(f"[WARN] Subfolder not found, skipping: {candidate}")
            continue
        for class_name in sorted(os.listdir(candidate)):
            class_src = os.path.join(candidate, class_name)
            if not os.path.isdir(class_src):
                continue
            class_dest = os.path.join(merged_root, class_name)
            os.makedirs(class_dest, exist_ok=True)
            for fname in os.listdir(class_src):
                src_file = os.path.join(class_src, fname)
                if not os.path.isfile(src_file):
                    continue
                ext = os.path.splitext(fname)[1].lower()
                if ext not in ('.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'):
                    continue
                dest_file = os.path.join(class_dest, fname)
                if os.path.exists(dest_file):
                    base, extn = os.path.splitext(fname)
                    i = 1
                    while os.path.exists(dest_file):
                        dest_file = os.path.join(class_dest, f"{base}_{i}{extn}")
                        i += 1
                try:
                    os.symlink(src_file, dest_file)
                except Exception:
                    shutil.copy2(src_file, dest_file)
            created.add(class_name)
    if not created:
        raise RuntimeError("No classes found while merging. Check dataset structure.")
    print(f"[INFO] Merged dataset created at: {merged_root} with {len(created)} classes")
    return merged_root


def main():
    if os.environ.get("VIT_TRAINING_RUNNING") == "1":
        print("[WARN] Training already running in this environment (VIT_TRAINING_RUNNING=1). Exiting to avoid duplicate runs.")
        return
    os.environ["VIT_TRAINING_RUNNING"] = "1"


    total_epochs = INITIAL_EPOCHS + FINE_TUNE_EPOCHS
    if total_epochs <= 0:
        raise ValueError("Total epochs <= 0. Check INITIAL_EPOCHS and FINE_TUNE_EPOCHS.")
    if total_epochs > MAX_TOTAL_EPOCHS:
        raise ValueError(f"Requested TOTAL_EPOCHS ({total_epochs}) is > {MAX_TOTAL_EPOCHS}. Possible config error.")

    global OUTPUT_DIR
    OUTPUT_DIR = ensure_output_outside_dataset(DATA_DIR, OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("OUTPUT_DIR:", OUTPUT_DIR)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


    merged_dir = make_merged_dataset(DATA_DIR, MERGE_SUBFOLDERS, MERGED_DIR_NAME)
    TRAIN_DIR = merged_dir
    print("Training from merged dataset at:", TRAIN_DIR)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.1,0.1,0.1,0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

  
    full_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    class_names = full_dataset.classes
    NUM_CLASSES = len(class_names)
    print("Detected classes:", NUM_CLASSES)
    print(class_names[:10], "..." if len(class_names) > 10 else "")

  
    with open(CLASS_JSON, "w") as f:
        json.dump(class_names, f, indent=2)

    targets = np.array([s[1] for s in full_dataset.samples])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=SEED)
    train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))

    train_dataset = Subset(datasets.ImageFolder(TRAIN_DIR, transform=train_transform), train_idx)
    val_dataset = Subset(datasets.ImageFolder(TRAIN_DIR, transform=val_transform), val_idx)

  
    train_labels = [full_dataset.samples[i][1] for i in train_idx]
    class_counts = Counter(train_labels)
    max_c = max(class_counts.values())
    weights_for_loss = []
    for cls in range(NUM_CLASSES):
        cnt = class_counts.get(cls, 0)
        weights_for_loss.append(max_c / (cnt if cnt > 0 else 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_tensor = torch.tensor(weights_for_loss, dtype=torch.float).to(device)
    sample_weights = [weights_for_loss[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

  
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)


    print("Loading ViT model (PyTorch):", MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME, num_labels=NUM_CLASSES)
    model.to(device)
    use_amp = torch.cuda.is_available()

    for name, param in model.base_model.named_parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=0.01)

    scaler = torch.amp.GradScaler() if use_amp else None

    def train_one_epoch(epoch):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        n = 0
        print(f"\n>>> Starting Stage 1 epoch {epoch}/{INITIAL_EPOCHS} <<<")
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}", unit="batch")
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(pixel_values=imgs)
                logits = outputs.logits
                loss = criterion(logits, labels)
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(logits.detach(), dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            n += imgs.size(0)
            pbar.set_postfix({"loss": running_loss / n, "acc": accuracy_score(all_labels, all_preds)})
        avg_loss = running_loss / n
        acc = accuracy_score(all_labels, all_preds)
        print(f">>> Finished Stage 1 epoch {epoch}/{INITIAL_EPOCHS}: train_loss={avg_loss:.4f} train_acc={acc:.4f}")
        return avg_loss, acc

    def evaluate(loader, compute_loss=True):
        model.eval()
        all_logits = []
        all_labels = []
        total_loss = 0.0
        n = 0
        pbar = tqdm(loader, desc="Validating", unit="batch")
        with torch.no_grad():
            for imgs, labels in pbar:
                imgs = imgs.to(device)
                labels = labels.to(device)
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    outputs = model(pixel_values=imgs)
                    logits = outputs.logits
                    if compute_loss:
                        loss = criterion(logits, labels)
                        total_loss += loss.item() * imgs.size(0)
                all_logits.append(logits.cpu().numpy())
                all_labels.extend(labels.cpu().numpy().tolist())
                n += imgs.size(0)
        all_logits = np.vstack(all_logits)
        preds = np.argmax(all_logits, axis=1)
        avg_loss = (total_loss / n) if compute_loss else None
        return np.array(all_labels), preds, all_logits, avg_loss

    history = {"stage":"head", "train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
    best_val_acc = 0.0
    print("\n=== Stage 1: Training head (backbone frozen) ===")
    for epoch in range(1, INITIAL_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(epoch)
        y_true_val, y_pred_val, val_logits, val_loss = evaluate(val_loader, compute_loss=True)
        val_acc = accuracy_score(y_true_val, y_pred_val)
        history["train_loss"].append(train_loss); history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss); history["val_acc"].append(val_acc)
        print(f"Epoch {epoch} HEAD: val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_HEAD_PATH)
            print(f"[INFO] Saved best head checkpoint: {BEST_HEAD_PATH}")

  
    print("\n=== Stage 2: Unfreezing backbone and fine-tuning ===")
    for name, param in model.base_model.named_parameters():
        param.requires_grad = True

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    num_training_steps = len(train_loader) * FINE_TUNE_EPOCHS
    warmup_steps = max(1, int(0.05 * num_training_steps))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

    best_val_acc_overall = best_val_acc
    history["stage2_train_loss"] = []; history["stage2_val_loss"] = []; history["stage2_val_acc"] = []

    for epoch in range(1, FINE_TUNE_EPOCHS + 1):
 
        model.train()
        running_loss = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"FineTune Epoch {epoch}", unit="batch")
        for imgs, labels in pbar:
            imgs = imgs.to(device); labels = labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(pixel_values=imgs)
                logits = outputs.logits
                loss = criterion(logits, labels)
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()
            running_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
            pbar.set_postfix({"loss": running_loss / n})
        train_loss = running_loss / n

        y_true_val, y_pred_val, val_logits, val_loss = evaluate(val_loader, compute_loss=True)
        val_acc = accuracy_score(y_true_val, y_pred_val)
        history["stage2_train_loss"].append(train_loss)
        history["stage2_val_loss"].append(val_loss)
        history["stage2_val_acc"].append(val_acc)
        print(f"Finetune epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val_acc_overall:
            best_val_acc_overall = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"[INFO] Saved best overall checkpoint: {BEST_MODEL_PATH}")

  
    print("\nSaving final model and artifacts...")
  
    torch.save(model.state_dict(), BEST_MODEL_PATH)
  
    model.save_pretrained(FINAL_MODEL_DIR)
    try:
        fe = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        fe.save_pretrained(FINAL_MODEL_DIR)
    except Exception as e:
        print("[WARN] Could not save feature_extractor:", e)

    pd.DataFrame(history).to_csv(HISTORY_CSV, index=False)

    y_true, y_pred, y_logits, _ = evaluate(val_loader, compute_loss=False)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(METRICS_CSV)
    conf_mat = confusion_matrix(y_true, y_pred)
    pd.DataFrame(conf_mat, index=class_names, columns=class_names).to_csv(CONFUSION_CSV)
    overall = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    }
    with open(METRICS_JSON, "w") as f:
        json.dump({"overall": overall}, f, indent=2)

    plt.figure(figsize=(12,10))
    import seaborn as sns
    sns.heatmap(conf_mat, cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar=True)
    plt.ylabel('True'); plt.xlabel('Predicted'); plt.title('Confusion Matrix')
    plt.tight_layout(); plt.savefig(CONFUSION_PNG); plt.close()

   
    y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    y_score = y_logits
    fpr = {}; tpr = {}; roc_auc = {}
    for i in range(NUM_CLASSES):
        try:
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        except Exception:
            fpr[i], tpr[i], roc_auc[i] = np.array([0.0,1.0]), np.array([0.0,1.0]), float('nan')
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NUM_CLASSES)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(NUM_CLASSES):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= NUM_CLASSES
    fpr["macro"], tpr["macro"], roc_auc["macro"] = all_fpr, mean_tpr, auc(all_fpr, mean_tpr)

    auc_rows = []
    for i,name in enumerate(class_names):
        auc_rows.append({"class_index": i, "class_name": name, "auc": float(roc_auc.get(i, np.nan))})
    auc_rows.append({"class_index":"micro","class_name":"micro-average","auc":float(roc_auc.get("micro", np.nan))})
    auc_rows.append({"class_index":"macro","class_name":"macro-average","auc":float(roc_auc.get("macro", np.nan))})
    pd.DataFrame(auc_rows).to_csv(AUC_CSV, index=False)

    plt.figure(figsize=(10,8))
    plt.plot(fpr["micro"], tpr["micro"], label=f'micro (AUC={roc_auc["micro"]:.3f})', color='deeppink', linestyle=':', linewidth=3)
    plt.plot(fpr["macro"], tpr["macro"], label=f'macro (AUC={roc_auc["macro"]:.3f})', color='navy', linestyle=':', linewidth=3)
    num_to_plot = min(6, NUM_CLASSES)
    indices_to_plot = list(range(0, NUM_CLASSES, max(1, NUM_CLASSES // num_to_plot)))[:num_to_plot]
    colors = plt.cm.get_cmap('tab10')
    for idx_i, i in enumerate(indices_to_plot):
        plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC={roc_auc.get(i, np.nan):.3f})', color=colors(idx_i%10))
    plt.plot([0,1],[0,1],'k--', lw=1)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC curves')
    plt.legend(loc='lower right', fontsize='small'); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(ROC_PNG); plt.close()

   
    artifacts = {
        "best_head_checkpoint": BEST_HEAD_PATH,
        "best_checkpoint": BEST_MODEL_PATH,
        "final_hf_dir": FINAL_MODEL_DIR,
        "class_names": CLASS_JSON,
        "training_history": HISTORY_CSV,
        "per_class_metrics_csv": METRICS_CSV,
        "overall_metrics_json": METRICS_JSON,
        "confusion_csv": CONFUSION_CSV,
        "confusion_png": CONFUSION_PNG,
        "roc_png": ROC_PNG,
        "auc_csv": AUC_CSV,
        "merged_dataset_dir": TRAIN_DIR,
        "report_pdf": REPORT_PDF if os.path.exists(REPORT_PDF) else None
    }
    with open(ARTIFACTS_JSON, "w") as f:
        json.dump(artifacts, f, indent=2)

    print("\nAll artifacts saved to:", OUTPUT_DIR)
    print(json.dumps(artifacts, indent=2))


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
