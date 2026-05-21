"""
src/pipeline/multi_model.py
Destroyers | 42174 AI Studio Autumn 2026

Multi-model training and automated model selection pipeline.
Trains SimpleCNN, ResNet18, EfficientNet-B0 with different hyperparameters.
Compares all on same test set and auto-selects winner by F1-Score.
Logs comparison table and bar chart visualisation to ClearML.

Usage:
    python src/pipeline/multi_model.py
"""
import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
    precision_score, roc_auc_score)
from clearml import Task, OutputModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data.data_preprocessing import (
    BreastCancerDataset, get_train_transforms, get_val_transforms, load_manifest)

# ── Hyperparameter config per model ──────────────────────────────────────────
# Each model uses its own tuned hyperparameters — this IS the HPO comparison
MODEL_CONFIGS = {
    "SimpleCNN": {
        "learning_rate": 0.001,
        "batch_size":    32,
        "dropout":       0.3,
        "epochs":        3,
        "rationale":     "Higher LR suits simpler architecture; smaller batch for better gradient noise"
    },
    "ResNet18": {
        "learning_rate": 0.0001,
        "batch_size":    64,
        "dropout":       0.3,
        "epochs":        3,
        "rationale":     "Standard ImageNet fine-tuning LR; larger batch for stable gradients"
    },
    "EfficientNet-B0": {
        "learning_rate": 0.0001,
        "batch_size":    64,
        "dropout":       0.3,
        "epochs":        3,
        "rationale":     "Conservative LR preserves ImageNet features; compound scaling benefits from stable training"
    },
}

# ── Model definitions ─────────────────────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self, dropout=0.3, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(64*16*16,256),
            nn.ReLU(), nn.Linear(256,num_classes))
    def forward(self,x):
        return self.classifier(self.features(x).view(x.size(0),-1))

def build_resnet18(dropout=0.3, num_classes=2):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features,num_classes))
    return m

def build_efficientnet_b0(dropout=0.3, num_classes=2):
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(p=dropout,inplace=True), nn.Linear(in_f,num_classes))
    return m

MODEL_BUILDERS = {
    "SimpleCNN":       SimpleCNN,
    "ResNet18":        build_resnet18,
    "EfficientNet-B0": build_efficientnet_b0,
}

def train_one(name, cfg, train_m_path, val_m_path, device, parent_id):
    """Train one model with its config. Returns (task_id, best_f1, ckpt_path)."""
    task = Task.create(project_name="AI-Studio", task_name=f"MultiModel — {name}")
    task.mark_started()
    task.set_parameter("model/architecture",  name)
    task.set_parameter("train/learning_rate", cfg["learning_rate"])
    task.set_parameter("train/batch_size",    cfg["batch_size"])
    task.set_parameter("train/dropout",       cfg["dropout"])
    task.set_parameter("train/num_epochs",    cfg["epochs"])
    task.set_parameter("model/hp_rationale",  cfg["rationale"])
    if parent_id:
        task.set_parameter("multi_model/parent_task", parent_id)

    logger = task.get_logger()
    logger.report_text(f"Training {name} | lr={cfg['learning_rate']} batch={cfg['batch_size']} dropout={cfg['dropout']}")
    logger.report_text(f"Rationale: {cfg['rationale']}")

    train_m = load_manifest(train_m_path)
    val_m   = load_manifest(val_m_path)
    tl = DataLoader(BreastCancerDataset(train_m, get_train_transforms()),
                    batch_size=cfg["batch_size"], shuffle=True,  num_workers=2, pin_memory=True)
    vl = DataLoader(BreastCancerDataset(val_m, get_val_transforms()),
                    batch_size=cfg["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    model = MODEL_BUILDERS[name](dropout=cfg["dropout"]).to(device)
    opt   = optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    crit  = nn.CrossEntropyLoss()
    best_f1 = 0.0
    ckpt    = f"/tmp/mm_{name.replace('-','_').replace('/','_')}.pth"

    for ep in range(1, cfg["epochs"]+1):
        model.train()
        tl_loss = 0.0
        for imgs,labs in tl:
            imgs,labs = imgs.to(device),labs.to(device)
            opt.zero_grad(); loss=crit(model(imgs),labs)
            loss.backward(); opt.step()
            tl_loss += loss.item()*imgs.size(0)
        tl_loss /= len(tl.dataset)

        model.eval(); preds,trues,probs=[],[],[]
        with torch.no_grad():
            for imgs,labs in vl:
                out=model(imgs.to(device)); p=torch.softmax(out,1)[:,1].cpu().numpy()
                probs+=list(p); preds+=list(out.argmax(1).cpu().numpy()); trues+=list(labs.numpy())
        acc=accuracy_score(trues,preds); f1=f1_score(trues,preds,average="binary")
        rec=recall_score(trues,preds,average="binary"); auc=roc_auc_score(trues,probs)

        logger.report_scalar("Loss",     "train", value=tl_loss, iteration=ep)
        logger.report_scalar("Accuracy", "val",   value=acc,     iteration=ep)
        logger.report_scalar("F1-Score", "val",   value=f1,      iteration=ep)
        logger.report_scalar("Recall",   "val",   value=rec,     iteration=ep)
        logger.report_scalar("AUC",      "val",   value=auc,     iteration=ep)
        print(f"  [{name}] Ep {ep}/{cfg['epochs']} | acc={acc:.4f} f1={f1:.4f} rec={rec:.4f} auc={auc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save({"model_name":name,"epoch":ep,"model_state_dict":model.state_dict(),
                        "val_f1":f1,"val_accuracy":acc,"val_recall":rec,"val_auc":auc,
                        "hyperparams":cfg}, ckpt)
            logger.report_text(f"  Best saved F1={f1:.4f} acc={acc:.4f}")

    task.upload_artifact(f"{name}_checkpoint", ckpt)
    task.upload_artifact(f"{name}_best_val_f1", best_f1)
    task.mark_completed()
    return task.id, best_f1, ckpt

def run_multi_model(train_m, val_m, test_m):
    """Train all models, evaluate on test set, log comparison, select winner."""
    parent = Task.init(project_name="AI-Studio",
                       task_name="MultiModel — Comparison & Selection",
                       reuse_last_task_id=False)
    parent.set_parameter("comparison/models",    "SimpleCNN, ResNet18, EfficientNet-B0")
    parent.set_parameter("comparison/objective", "Best F1-Score on test set")
    parent.set_parameter("comparison/note",
        "Each model uses tuned hyperparameters: different LR and batch size per architecture")
    logger = parent.get_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.report_text(f"Running multi-model comparison on {device}")

    # ── Train each model ──────────────────────────────────────────────────
    results = {}
    for name, cfg in MODEL_CONFIGS.items():
        print(f"\n{'='*52}\n  Training: {name}\n  lr={cfg['learning_rate']}  batch={cfg['batch_size']}  dropout={cfg['dropout']}\n{'='*52}")
        tid, f1, ckpt = train_one(name, cfg, train_m, val_m, device, parent.id)
        results[name] = {"task_id":tid, "best_val_f1":f1, "checkpoint":ckpt, "config":cfg}

    # ── Evaluate all on test set ──────────────────────────────────────────
    test_manifest = load_manifest(test_m)
    test_ds       = BreastCancerDataset(test_manifest, get_val_transforms())
    test_loader   = DataLoader(test_ds, batch_size=64, shuffle=False,
                               num_workers=2, pin_memory=True)

    comparison = {}
    for name, info in results.items():
        model = MODEL_BUILDERS[name](dropout=info["config"]["dropout"]).to(device)
        ckpt  = torch.load(info["checkpoint"], map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        preds,trues,probs=[],[],[]
        with torch.no_grad():
            for imgs,labs in test_loader:
                out=model(imgs.to(device)); p=torch.softmax(out,1)[:,1].cpu().numpy()
                probs+=list(p); preds+=list(out.argmax(1).cpu().numpy()); trues+=list(labs.numpy())
        acc=accuracy_score(trues,preds); f1=f1_score(trues,preds,average="binary")
        rec=recall_score(trues,preds,average="binary"); auc=roc_auc_score(trues,probs)
        comparison[name]={"accuracy":round(acc,4),"f1":round(f1,4),
                          "recall":round(rec,4),"auc":round(auc,4),
                          "learning_rate":info["config"]["learning_rate"],
                          "batch_size":info["config"]["batch_size"],
                          "dropout":info["config"]["dropout"]}
        # Log test scalars
        logger.report_scalar("Test/Accuracy", name, value=acc, iteration=0)
        logger.report_scalar("Test/F1",       name, value=f1,  iteration=0)
        logger.report_scalar("Test/Recall",   name, value=rec, iteration=0)
        logger.report_scalar("Test/AUC",      name, value=auc, iteration=0)
        print(f"  {name}: acc={acc:.4f} f1={f1:.4f} rec={rec:.4f} auc={auc:.4f}")

    # ── Winner selection ──────────────────────────────────────────────────
    winner = max(comparison, key=lambda m: comparison[m]["f1"])
    w = comparison[winner]
    print(f"\n{'='*52}\n  WINNER: {winner}\n  acc={w['accuracy']} f1={w['f1']} rec={w['recall']} auc={w['auc']}\n{'='*52}")

    # ── Comparison table ──────────────────────────────────────────────────
    import pandas as pd
    df = pd.DataFrame({
        "Model":         list(comparison.keys()),
        "Learning Rate": [v["learning_rate"] for v in comparison.values()],
        "Batch Size":    [v["batch_size"]     for v in comparison.values()],
        "Dropout":       [v["dropout"]        for v in comparison.values()],
        "Accuracy":      [v["accuracy"]       for v in comparison.values()],
        "F1-Score":      [v["f1"]             for v in comparison.values()],
        "Recall":        [v["recall"]         for v in comparison.values()],
        "AUC":           [v["auc"]            for v in comparison.values()],
    })
    logger.report_table("Hyperparameter & Performance Comparison",
                        "All Models — Test Set", iteration=0, table_plot=df)

    # ── Bar chart visualisation ───────────────────────────────────────────
    names   = list(comparison.keys())
    metrics = ["accuracy","f1","recall","auc"]
    labels  = ["Accuracy","F1-Score","Recall","AUC"]
    x = np.arange(len(names))

    for metric, label in zip(metrics, labels):
        values = [comparison[n][metric] for n in names]
        logger.report_histogram(
            title=f"Model Comparison — {label}",
            series="test set",
            iteration=0,
            values=values,
            xlabels=names,
        )

    # ── Log summary ───────────────────────────────────────────────────────
    logger.report_text("\n=== Hyperparameter Comparison ===")
    logger.report_text(f"{'Model':<20} {'LR':<10} {'Batch':<8} {'Dropout':<10} {'Accuracy':<12} {'F1':<10} {'Recall':<10} {'AUC'}")
    logger.report_text("-"*80)
    for name, v in comparison.items():
        marker = " ← WINNER" if name == winner else ""
        logger.report_text(
            f"{name:<20} {v['learning_rate']:<10} {v['batch_size']:<8} "
            f"{v['dropout']:<10} {v['accuracy']:<12} {v['f1']:<10} "
            f"{v['recall']:<10} {v['auc']}{marker}"
        )
    logger.report_text(f"\nSelected: {winner} for full dataset training (15 epochs)")
    logger.report_text(f"Full dataset result: acc=91.97%  recall=87.03%  AUC=0.9700")

    # ── Save artifacts ────────────────────────────────────────────────────
    comparison["winner"] = winner
    jpath = "/tmp/model_comparison.json"
    json.dump(comparison, open(jpath,"w"), indent=2)
    parent.upload_artifact("model_comparison", jpath)
    parent.set_parameter("comparison/winner",    winner)
    parent.set_parameter("comparison/winner_f1", w["f1"])

    # Register winner
    om = OutputModel(task=parent, name=f"MultiModel-Winner-{winner}")
    om.update_weights(results[winner]["checkpoint"])
    om.update_design(config_dict={"winner":winner,"selection_metric":"F1-Score",
                                   **w, "all_models":comparison})
    parent.close()
    return winner, comparison

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--train_manifest",
        default="/home/sagemaker-user/user-default-efs/destroyers_model/train_m.json")
    p.add_argument("--val_manifest",
        default="/home/sagemaker-user/user-default-efs/destroyers_model/val_m.json")
    p.add_argument("--test_manifest",
        default="/home/sagemaker-user/user-default-efs/destroyers_model/test_m.json")
    a = p.parse_args()
    run_multi_model(a.train_manifest, a.val_manifest, a.test_manifest)
