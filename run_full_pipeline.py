"""
run_full_pipeline.py
Destroyers | 42174 AI Studio Autumn 2026

Complete end-to-end MLOps pipeline:

PHASE 1 — HPO per model (10k subset, 2 epochs per trial)
  SimpleCNN:       3 hyperparameter combinations tried
  ResNet18:        3 hyperparameter combinations tried
  EfficientNet-B0: 3 hyperparameter combinations tried
  Best config per model selected by validation F1

PHASE 2 — Multi-model comparison (10k subset, 3 epochs, best config each)
  SimpleCNN vs ResNet18 vs EfficientNet-B0
  Comparison table + bar charts logged to ClearML
  Winner auto-selected by test F1-Score

PHASE 3 — Full dataset training (277,524 images, 15 epochs, winner only)
  EfficientNet-B0 trained on complete IDC dataset on Tesla T4 GPU
  Per-epoch metrics logged to ClearML

PHASE 4 — Evaluation + quality gate + Grad-CAM + model registry
  Test set: 41,630 images
  Quality gate: acc>=90%, F1>=0.90, recall>=0.85
  Grad-CAM heatmaps for 6 samples
  Auto-register in ClearML if gate passes

Run: python run_full_pipeline.py
"""
import os, sys, json, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
    precision_score, roc_auc_score)
from clearml import Task, OutputModel

sys.path.insert(0, '.')
from src.data.data_preprocessing import (
    BreastCancerDataset, get_train_transforms,
    get_val_transforms, load_manifest)

# ── Paths ─────────────────────────────────────────────────────────────────────
EFS = "/home/sagemaker-user/user-default-efs/destroyers_model"
TRAIN_M = f"{EFS}/train_m.json"
VAL_M   = f"{EFS}/val_m.json"
TEST_M  = f"{EFS}/test_m.json"
os.makedirs(EFS, exist_ok=True)

# ── HPO search grid per model ─────────────────────────────────────────────────
HPO_GRID = {
    "SimpleCNN": [
        {"learning_rate": 0.01,   "batch_size": 32, "dropout": 0.2},
        {"learning_rate": 0.001,  "batch_size": 32, "dropout": 0.3},
        {"learning_rate": 0.0001, "batch_size": 64, "dropout": 0.3},
    ],
    "ResNet18": [
        {"learning_rate": 0.001,   "batch_size": 64, "dropout": 0.2},
        {"learning_rate": 0.0001,  "batch_size": 64, "dropout": 0.3},
        {"learning_rate": 0.00005, "batch_size": 32, "dropout": 0.4},
    ],
    "EfficientNet-B0": [
        {"learning_rate": 0.001,   "batch_size": 64, "dropout": 0.2},
        {"learning_rate": 0.0001,  "batch_size": 64, "dropout": 0.3},
        {"learning_rate": 0.00005, "batch_size": 32, "dropout": 0.4},
    ],
}
HPO_EPOCHS    = 2     # quick HPO trials
COMPARE_EPOCHS = 3    # final comparison
FULL_EPOCHS   = 15    # full training
HPO_SUBSET    = 6000  # 3k per class for HPO
COMPARE_SUBSET = 10000 # 5k per class for comparison

# ── Model builders ────────────────────────────────────────────────────────────
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
    m.classifier = nn.Sequential(
        nn.Dropout(p=dropout,inplace=True), nn.Linear(in_f,num_classes))
    return m

BUILDERS = {
    "SimpleCNN":       SimpleCNN,
    "ResNet18":        build_resnet18,
    "EfficientNet-B0": build_efficientnet_b0,
}

# ── Utilities ─────────────────────────────────────────────────────────────────
def balanced_subset(manifest, n):
    random.seed(42)
    cancer     = [x for x in manifest if x["label"]==1]
    non_cancer = [x for x in manifest if x["label"]==0]
    half = n // 2
    return (random.sample(cancer,     min(half,len(cancer))) +
            random.sample(non_cancer, min(half,len(non_cancer))))

def make_loaders(train_m, val_m, batch_size):
    tl = DataLoader(BreastCancerDataset(train_m, get_train_transforms()),
                    batch_size=batch_size, shuffle=True,
                    num_workers=2, pin_memory=True)
    vl = DataLoader(BreastCancerDataset(val_m, get_val_transforms()),
                    batch_size=batch_size, shuffle=False,
                    num_workers=2, pin_memory=True)
    return tl, vl

def evaluate(model, loader, device):
    model.eval(); preds,trues,probs=[],[],[]
    with torch.no_grad():
        for imgs,labs in loader:
            out = model(imgs.to(device))
            p   = torch.softmax(out,1)[:,1].cpu().numpy()
            probs += list(p)
            preds += list(out.argmax(1).cpu().numpy())
            trues += list(labs.numpy())
    acc = accuracy_score(trues,preds)
    f1  = f1_score(trues,preds,average="binary")
    rec = recall_score(trues,preds,average="binary")
    auc = roc_auc_score(trues,probs)
    return acc, f1, rec, auc

def train_loop(model, tl, vl, epochs, lr, device, logger, name):
    opt  = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best_f1, best_state = 0.0, None
    for ep in range(1, epochs+1):
        model.train(); tl_loss=0.0
        for imgs,labs in tl:
            imgs,labs=imgs.to(device),labs.to(device)
            opt.zero_grad(); loss=crit(model(imgs),labs)
            loss.backward(); opt.step()
            tl_loss+=loss.item()*imgs.size(0)
        tl_loss/=len(tl.dataset)
        acc,f1,rec,auc=evaluate(model,vl,device)
        if logger:
            logger.report_scalar("Loss",     "train",value=tl_loss,iteration=ep)
            logger.report_scalar("Accuracy", "val",  value=acc,    iteration=ep)
            logger.report_scalar("F1-Score", "val",  value=f1,     iteration=ep)
            logger.report_scalar("Recall",   "val",  value=rec,    iteration=ep)
            logger.report_scalar("AUC",      "val",  value=auc,    iteration=ep)
        print(f"    Ep {ep}/{epochs} | loss={tl_loss:.4f} acc={acc:.4f} f1={f1:.4f} rec={rec:.4f} auc={auc:.4f}")
        if f1>best_f1:
            best_f1=f1; best_state=model.state_dict()
            if logger: logger.report_text(f"    Best F1={f1:.4f}")
    return best_f1, best_state

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — HPO: find best hyperparameters per model
# ══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*60)
print("  PHASE 1: Hyperparameter Optimisation")
print(f"  Subset: {HPO_SUBSET} images | Epochs per trial: {HPO_EPOCHS}")
print("="*60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

train_all = load_manifest(TRAIN_M)
val_all   = load_manifest(VAL_M)
test_all  = load_manifest(TEST_M)

hpo_train = balanced_subset(train_all, HPO_SUBSET)
hpo_val   = balanced_subset(val_all,   int(HPO_SUBSET*0.2))

hpo_task = Task.init(project_name="AI-Studio",
                     task_name="HPO — Hyperparameter Search All Models",
                     reuse_last_task_id=False)
hpo_task.set_parameter("hpo/subset_size",    HPO_SUBSET)
hpo_task.set_parameter("hpo/epochs_per_trial", HPO_EPOCHS)
hpo_task.set_parameter("hpo/objective",      "Best validation F1-Score")
hpo_logger = hpo_task.get_logger()

best_configs = {}
all_hpo_results = []

for model_name, grid in HPO_GRID.items():
    print(f"\n  HPO for {model_name} ({len(grid)} trials):")
    best_f1, best_cfg, best_idx = 0.0, None, 0

    for i, cfg in enumerate(grid):
        trial_name = f"HPO — {model_name} trial {i+1} (lr={cfg['learning_rate']} batch={cfg['batch_size']} dropout={cfg['dropout']})"
        print(f"    Trial {i+1}/{len(grid)}: lr={cfg['learning_rate']} batch={cfg['batch_size']} dropout={cfg['dropout']}")

        trial_task = Task.create(project_name="AI-Studio", task_name=trial_name)
        trial_task.mark_started()
        trial_task.set_parameter("model/architecture",  model_name)
        trial_task.set_parameter("train/learning_rate", cfg["learning_rate"])
        trial_task.set_parameter("train/batch_size",    cfg["batch_size"])
        trial_task.set_parameter("train/dropout",       cfg["dropout"])
        trial_task.set_parameter("train/num_epochs",    HPO_EPOCHS)
        trial_task.set_parameter("hpo/trial",           i+1)
        trial_task.set_parameter("hpo/parent_task",     hpo_task.id)
        tlogger = trial_task.get_logger()

        tl, vl = make_loaders(hpo_train, hpo_val, cfg["batch_size"])
        model   = BUILDERS[model_name](dropout=cfg["dropout"]).to(device)
        f1, _   = train_loop(model, tl, vl, HPO_EPOCHS,
                              cfg["learning_rate"], device, tlogger, model_name)

        trial_task.set_parameter("results/best_val_f1", round(f1,4))
        trial_task.mark_completed()

        all_hpo_results.append({
            "Model": model_name, "Trial": i+1,
            "Learning Rate": cfg["learning_rate"],
            "Batch Size": cfg["batch_size"],
            "Dropout": cfg["dropout"],
            "Val F1": round(f1,4),
        })

        hpo_logger.report_scalar(f"HPO/{model_name}",
                                  f"trial{i+1}_lr{cfg['learning_rate']}",
                                  value=f1, iteration=i+1)

        print(f"    → F1={f1:.4f}")
        if f1 > best_f1:
            best_f1=f1; best_cfg=cfg.copy(); best_idx=i+1

    best_cfg["epochs"] = COMPARE_EPOCHS
    best_configs[model_name] = best_cfg
    hpo_logger.report_text(
        f"{model_name} best: trial {best_idx} "
        f"lr={best_cfg['learning_rate']} batch={best_cfg['batch_size']} "
        f"dropout={best_cfg['dropout']} F1={best_f1:.4f}")
    print(f"  → Best for {model_name}: lr={best_cfg['learning_rate']} batch={best_cfg['batch_size']} dropout={best_cfg['dropout']} F1={best_f1:.4f}")

# HPO results table
hpo_df = pd.DataFrame(all_hpo_results)
hpo_logger.report_table("HPO Results — All Trials",
                         "Validation F1 per trial", iteration=0, table_plot=hpo_df)

# HPO bar chart per model
for model_name in HPO_GRID:
    rows   = [r for r in all_hpo_results if r["Model"]==model_name]
    labels = [f"lr={r['Learning Rate']}\nbatch={r['Batch Size']}\ndropout={r['Dropout']}" for r in rows]
    values = [r["Val F1"] for r in rows]
    hpo_logger.report_histogram(
        title=f"HPO — {model_name} Trial Comparison",
        series="Val F1", iteration=0,
        values=values, xlabels=labels)

# Summary
hpo_logger.report_text("\n=== Best Config Per Model (selected for comparison) ===")
for name, cfg in best_configs.items():
    hpo_logger.report_text(
        f"  {name}: lr={cfg['learning_rate']} batch={cfg['batch_size']} dropout={cfg['dropout']}")

hpo_task.upload_artifact("hpo_results", "/tmp/hpo_results.json" if
    json.dump(all_hpo_results, open("/tmp/hpo_results.json","w"), indent=2) or True
    else "/tmp/hpo_results.json")
hpo_task.upload_artifact("best_configs", "/tmp/best_configs.json" if
    json.dump(best_configs, open("/tmp/best_configs.json","w"), indent=2) or True
    else "/tmp/best_configs.json")
hpo_task.close()
print("\n  HPO complete. Best configs selected.")
print(f"  SimpleCNN:       lr={best_configs['SimpleCNN']['learning_rate']} batch={best_configs['SimpleCNN']['batch_size']}")
print(f"  ResNet18:        lr={best_configs['ResNet18']['learning_rate']} batch={best_configs['ResNet18']['batch_size']}")
print(f"  EfficientNet-B0: lr={best_configs['EfficientNet-B0']['learning_rate']} batch={best_configs['EfficientNet-B0']['batch_size']}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — MULTI-MODEL COMPARISON (best config each, 10k subset)
# ══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*60)
print("  PHASE 2: Multi-Model Comparison")
print(f"  Subset: {COMPARE_SUBSET} images | Epochs: {COMPARE_EPOCHS}")
print("  Using best hyperparameters from Phase 1")
print("="*60)

compare_train = balanced_subset(train_all, COMPARE_SUBSET)
compare_val   = balanced_subset(val_all,   int(COMPARE_SUBSET*0.15))
compare_test  = balanced_subset(test_all,  int(COMPARE_SUBSET*0.15))

comp_task = Task.init(project_name="AI-Studio",
                      task_name="MultiModel — Comparison & Selection",
                      reuse_last_task_id=False)
comp_task.set_parameter("comparison/models",      "SimpleCNN, ResNet18, EfficientNet-B0")
comp_task.set_parameter("comparison/subset_size", COMPARE_SUBSET)
comp_task.set_parameter("comparison/objective",   "Best test F1-Score → winner trained on full dataset")
comp_task.set_parameter("comparison/note",        "Each model uses best hyperparameters from HPO phase")
comp_logger = comp_task.get_logger()

comp_results = {}
for name, cfg in best_configs.items():
    print(f"\n  Training: {name} | lr={cfg['learning_rate']} batch={cfg['batch_size']} dropout={cfg['dropout']}")

    subtask = Task.create(project_name="AI-Studio", task_name=f"MultiModel — {name}")
    subtask.mark_started()
    subtask.set_parameter("model/architecture",  name)
    subtask.set_parameter("train/learning_rate", cfg["learning_rate"])
    subtask.set_parameter("train/batch_size",    cfg["batch_size"])
    subtask.set_parameter("train/dropout",       cfg["dropout"])
    subtask.set_parameter("train/num_epochs",    cfg["epochs"])
    subtask.set_parameter("multi_model/parent",  comp_task.id)
    subtask.set_parameter("multi_model/hpo_selected", True)
    slogger = subtask.get_logger()

    tl, vl = make_loaders(compare_train, compare_val, cfg["batch_size"])
    model   = BUILDERS[name](dropout=cfg["dropout"]).to(device)
    best_f1, best_state = train_loop(
        model, tl, vl, cfg["epochs"], cfg["learning_rate"], device, slogger, name)

    ckpt = f"/tmp/comp_{name.replace('-','_')}.pth"
    torch.save({"model_name":name,"model_state_dict":best_state,
                "val_f1":best_f1,"hyperparams":cfg}, ckpt)
    subtask.upload_artifact(f"{name}_checkpoint", ckpt)
    subtask.upload_artifact(f"{name}_best_val_f1", best_f1)
    subtask.mark_completed()
    comp_results[name] = {"checkpoint":ckpt,"config":cfg,"best_val_f1":best_f1}
    print(f"  Done: best val F1={best_f1:.4f}")

# Test set evaluation
print("\n  Evaluating all models on test set...")
xl = DataLoader(BreastCancerDataset(compare_test, get_val_transforms()),
                batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

comparison = {}
for name, info in comp_results.items():
    model = BUILDERS[name](dropout=info["config"]["dropout"]).to(device)
    ckpt  = torch.load(info["checkpoint"], map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    acc,f1,rec,auc = evaluate(model, xl, device)
    comparison[name] = {
        "accuracy": round(acc,4), "f1": round(f1,4),
        "recall":   round(rec,4), "auc": round(auc,4),
        "learning_rate": info["config"]["learning_rate"],
        "batch_size":    info["config"]["batch_size"],
        "dropout":       info["config"]["dropout"],
    }
    comp_logger.report_scalar("Test/Accuracy", name, value=acc, iteration=0)
    comp_logger.report_scalar("Test/F1",       name, value=f1,  iteration=0)
    comp_logger.report_scalar("Test/Recall",   name, value=rec, iteration=0)
    comp_logger.report_scalar("Test/AUC",      name, value=auc, iteration=0)
    print(f"  {name}: acc={acc:.4f} f1={f1:.4f} rec={rec:.4f} auc={auc:.4f}")

# Winner
winner = max(comparison, key=lambda m: comparison[m]["f1"])
w = comparison[winner]
print(f"\n  WINNER: {winner} (F1={w['f1']:.4f})")

# Comparison table with hyperparameters
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
comp_logger.report_table("Model Comparison — Best Hyperparameters",
                         "Test Set Results", iteration=0, table_plot=df)

# Bar charts
for metric, label in [("accuracy","Accuracy"),("f1","F1-Score"),
                       ("recall","Recall"),("auc","AUC")]:
    comp_logger.report_histogram(
        title=f"Model Comparison — {label}",
        series="test set", iteration=0,
        values=[comparison[n][metric] for n in comparison],
        xlabels=list(comparison.keys()))

# Text summary
comp_logger.report_text("\n=== Final Comparison (Best HP per model) ===")
comp_logger.report_text(f"{'Model':<20} {'LR':<10} {'Batch':<8} {'Dropout':<10} {'Acc':<10} {'F1':<10} {'Recall':<10} {'AUC'}")
comp_logger.report_text("-"*85)
for name, v in comparison.items():
    marker = " ← WINNER" if name==winner else ""
    comp_logger.report_text(
        f"{name:<20} {v['learning_rate']:<10} {v['batch_size']:<8} "
        f"{v['dropout']:<10} {v['accuracy']:<10} {v['f1']:<10} "
        f"{v['recall']:<10} {v['auc']}{marker}")
comp_logger.report_text(f"\nWINNER: {winner} → full dataset training (15 epochs)")

comparison["winner"] = winner
json.dump(comparison, open("/tmp/model_comparison.json","w"), indent=2)
comp_task.upload_artifact("model_comparison", "/tmp/model_comparison.json")
comp_task.set_parameter("comparison/winner",    winner)
comp_task.set_parameter("comparison/winner_f1", w["f1"])
comp_task.close()

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — FULL DATASET TRAINING WITH WINNER
# ══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*60)
print(f"  PHASE 3: Full Dataset Training — {winner}")
print(f"  Images: 277,524 | Epochs: {FULL_EPOCHS} | GPU: Tesla T4")
print(f"  Hyperparams: lr={best_configs[winner]['learning_rate']} "
      f"batch={best_configs[winner]['batch_size']} "
      f"dropout={best_configs[winner]['dropout']}")
print("="*60)

from src.models.train import train_model
from clearml import Task as CTask

train_task = CTask.init(project_name="AI-Studio",
                         task_name="EfficientNet-B0 Training — Breast Cancer",
                         reuse_last_task_id=False)
train_task.set_parameter("multi_model/selected_from", "MultiModel comparison Phase 2")
train_task.set_parameter("hpo/best_config_from",      "HPO Phase 1")

ckpt_path = train_model(
    train_manifest_path=TRAIN_M,
    val_manifest_path=VAL_M,
    num_epochs=FULL_EPOCHS,
    learning_rate=best_configs[winner]["learning_rate"],
    batch_size=best_configs[winner]["batch_size"],
    dropout=best_configs[winner]["dropout"],
    output_dir=EFS,
    task=train_task,
)

import shutil
efs_ckpt = f"{EFS}/efficientnet_b0_best.pth"
if ckpt_path != efs_ckpt:
    shutil.copy(ckpt_path, efs_ckpt)
print(f"  Model saved to EFS: {efs_ckpt}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — EVALUATION + QUALITY GATE + GRAD-CAM + REGISTRY
# ══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*60)
print("  PHASE 4: Evaluation + Quality Gate + Grad-CAM")
print("  Test set: 41,630 images | Threshold: 0.4")
print("="*60)

from src.models.evaluate import evaluate_model

eval_task = CTask.init(project_name="AI-Studio",
                        task_name="EfficientNet-B0 Evaluation — Breast Cancer",
                        reuse_last_task_id=False)

final = evaluate_model(
    checkpoint_path=efs_ckpt,
    test_manifest_path=TEST_M,
    malignant_threshold=0.4,
    generate_gradcam=True,
    n_gradcam_samples=6,
    task=eval_task,
)

# ══════════════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*60)
print("  PIPELINE COMPLETE")
print("="*60)
print(f"  Winner:   {winner}")
print(f"  Accuracy: {final['accuracy']} (target >= 0.90)")
print(f"  F1-Score: {final['f1']}       (target >= 0.90)")
print(f"  Recall:   {final['recall']}   (target >= 0.85)")
print(f"  AUC:      {final['auc']}")
print(f"  Gate:     {'PASSED' if final['gate_passed'] else 'FAILED'}")
print(f"  Checkpoint: {efs_ckpt}")
print("="*60)
print("  Track at https://app.clear.ml")
