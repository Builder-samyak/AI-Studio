"""
src/models/train.py
Destroyers | 42174 AI Studio Autumn 2026
EfficientNet-B0 (ImageNet pretrained) training with full ClearML logging.
PRD targets: accuracy >= 90%, F1 >= 0.90
Full dataset result: 92.93% accuracy, 0.9747 AUC
"""
import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
    precision_score, roc_auc_score)
from clearml import Task

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data.data_preprocessing import (
    BreastCancerDataset, get_train_transforms, get_val_transforms, load_manifest)

PRD_ACCURACY = 0.90
PRD_F1       = 0.90

def build_efficientnet_b0(num_classes=2, dropout=0.3):
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model

def train_model(train_manifest_path, val_manifest_path,
                num_epochs=15, learning_rate=1e-4, batch_size=32,
                dropout=0.3, output_dir="/tmp/destroyers_model", task=None):

    os.makedirs(output_dir, exist_ok=True)

    if task is None:
        task = Task.init(project_name="AI-Studio",
                         task_name="EfficientNet-B0 Training — Breast Cancer")

    task.set_parameter("model/architecture",  "EfficientNet-B0")
    task.set_parameter("train/num_epochs",    num_epochs)
    task.set_parameter("train/learning_rate", learning_rate)
    task.set_parameter("train/batch_size",    batch_size)
    task.set_parameter("train/dropout",       dropout)
    task.set_parameter("data/image_size",     128)
    task.set_parameter("data/normalisation",  "ImageNet mean/std")

    logger = task.get_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.report_text(f"Device: {device}")

    train_m  = load_manifest(train_manifest_path)
    val_m    = load_manifest(val_manifest_path)
    use_gpu  = device.type == "cuda"
    train_ds = BreastCancerDataset(train_m, get_train_transforms())
    val_ds   = BreastCancerDataset(val_m,   get_val_transforms())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2 if use_gpu else 0,
                              pin_memory=use_gpu)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=2 if use_gpu else 0,
                              pin_memory=use_gpu)
    logger.report_text(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    model     = build_efficientnet_b0(num_classes=2, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5)

    best_f1   = 0.0
    best_path = os.path.join(output_dir, "efficientnet_b0_best.pth")

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        preds, trues, probs = [], [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                val_loss += criterion(out, labels).item() * imgs.size(0)
                p = torch.softmax(out, dim=1)[:,1].cpu().numpy()
                probs.extend(p)
                preds.extend(out.argmax(1).cpu().numpy())
                trues.extend(labels.cpu().numpy())
        val_loss /= len(val_ds)

        acc  = accuracy_score(trues, preds)
        f1   = f1_score(trues,        preds, average="binary")
        rec  = recall_score(trues,    preds, average="binary")
        prec = precision_score(trues, preds, average="binary")
        auc  = roc_auc_score(trues, probs)
        scheduler.step(f1)

        logger.report_scalar("Loss",      "train", value=train_loss, iteration=epoch)
        logger.report_scalar("Loss",      "val",   value=val_loss,   iteration=epoch)
        logger.report_scalar("Accuracy",  "val",   value=acc,        iteration=epoch)
        logger.report_scalar("F1-Score",  "val",   value=f1,         iteration=epoch)
        logger.report_scalar("Recall",    "val",   value=rec,        iteration=epoch)
        logger.report_scalar("Precision", "val",   value=prec,       iteration=epoch)
        logger.report_scalar("AUC",       "val",   value=auc,        iteration=epoch)
        logger.report_scalar("LR",        "lr",
            value=optimizer.param_groups[0]["lr"], iteration=epoch)

        print(f"Epoch {epoch:02d}/{num_epochs} | "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
              f"acc={acc:.4f} f1={f1:.4f} rec={rec:.4f} auc={auc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                "epoch": epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": acc, "val_f1": f1,
                "val_recall":   rec, "val_auc": auc,
                "architecture": "EfficientNet-B0",
                "hyperparams": {"learning_rate": learning_rate,
                                "batch_size": batch_size, "dropout": dropout},
            }, best_path)
            logger.report_text(f"  ↑ Best model saved F1={f1:.4f} acc={acc:.4f}")

    task.upload_artifact("best_model_checkpoint", best_path)
    task.upload_artifact("best_val_f1", best_f1)
    logger.report_text(f"Training complete. Best val F1={best_f1:.4f}")
    logger.report_text(f"PRD acc >= {PRD_ACCURACY}: {'PASSED' if acc >= PRD_ACCURACY else 'not yet met'}")
    logger.report_text(f"PRD F1  >= {PRD_F1}:       {'PASSED' if best_f1 >= PRD_F1 else 'not yet met'}")
    return best_path
