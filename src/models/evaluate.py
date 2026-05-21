"""
src/models/evaluate.py
Destroyers | 42174 AI Studio Autumn 2026
Test-set evaluation, quality gate, Grad-CAM, ClearML model registry.
"""
import os, sys, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
    precision_score, roc_auc_score, confusion_matrix, classification_report)
from clearml import Task, OutputModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data.data_preprocessing import (
    BreastCancerDataset, get_val_transforms, load_manifest)
from src.models.train import build_efficientnet_b0

PRD_ACCURACY  = 0.90
PRD_F1        = 0.90
PRD_RECALL    = 0.85
CLASS_NAMES   = {0: "Non-Cancer", 1: "Cancer"}
MALIGNANT_IDX = 1

def evaluate_model(checkpoint_path, test_manifest_path,
                   malignant_threshold=0.5, generate_gradcam=True,
                   n_gradcam_samples=6, task=None):

    if task is None:
        task = Task.init(project_name="AI-Studio",
                         task_name="EfficientNet-B0 Evaluation — Breast Cancer")

    task.set_parameter("eval/checkpoint",     checkpoint_path)
    task.set_parameter("eval/threshold",      malignant_threshold)
    task.set_parameter("prd/accuracy_target", PRD_ACCURACY)
    task.set_parameter("prd/f1_target",       PRD_F1)
    task.set_parameter("prd/recall_target",   PRD_RECALL)

    logger = task.get_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_gpu = device.type == "cuda"

    test_m  = load_manifest(test_manifest_path)
    test_ds = BreastCancerDataset(test_m, get_val_transforms())
    loader  = DataLoader(test_ds, batch_size=32, shuffle=False,
                         num_workers=2 if use_gpu else 0,
                         pin_memory=use_gpu)
    logger.report_text(f"Test set: {len(test_ds)} images | Device: {device}")

    model = build_efficientnet_b0(num_classes=2)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    preds, trues, probs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            out = model(imgs.to(device))
            p   = torch.softmax(out, dim=1)[:,MALIGNANT_IDX].cpu().numpy()
            probs.extend(p)
            preds.extend((p >= malignant_threshold).astype(int))
            trues.extend(labels.numpy())

    acc  = accuracy_score(trues, preds)
    f1   = f1_score(trues,        preds, average="binary")
    rec  = recall_score(trues,    preds, average="binary")
    prec = precision_score(trues, preds, average="binary")
    auc  = roc_auc_score(trues, probs)
    cm   = confusion_matrix(trues, preds)

    print(f"\n{'='*52}\n  Test Results — EfficientNet-B0\n{'='*52}")
    print(f"  Accuracy:  {acc:.4f}  (target >= {PRD_ACCURACY})")
    print(f"  F1-Score:  {f1:.4f}  (target >= {PRD_F1})")
    print(f"  Recall:    {rec:.4f}  (target >= {PRD_RECALL})")
    print(f"  Precision: {prec:.4f}")
    print(f"  AUC:       {auc:.4f}")
    print(f"\n{classification_report(trues, preds, target_names=list(CLASS_NAMES.values()))}")

    for name, val in [("Accuracy",acc),("F1-Score",f1),
                      ("Recall",rec),("Precision",prec),("AUC",auc)]:
        logger.report_scalar(f"Test/{name}", "value", value=val, iteration=0)

    logger.report_confusion_matrix(
        "Confusion Matrix", "test", matrix=cm.tolist(), iteration=0,
        xlabels=list(CLASS_NAMES.values()), ylabels=list(CLASS_NAMES.values()))

    acc_pass = acc >= PRD_ACCURACY
    f1_pass  = f1  >= PRD_F1
    rec_pass = rec >= PRD_RECALL
    gate     = acc_pass and f1_pass and rec_pass

    print(f"\n{'='*52}\n  Quality Gate\n{'='*52}")
    print(f"  Accuracy  >= {PRD_ACCURACY}: {'PASS' if acc_pass else 'FAIL'} ({acc:.4f})")
    print(f"  F1-Score  >= {PRD_F1}:       {'PASS' if f1_pass  else 'FAIL'} ({f1:.4f})")
    print(f"  Recall    >= {PRD_RECALL}:   {'PASS' if rec_pass else 'FAIL'} ({rec:.4f})")
    print(f"  Overall:  {'PASSED' if gate else 'FAILED'}")

    logger.report_scalar("QualityGate/Passed",       "value", value=float(gate),     iteration=0)
    logger.report_scalar("QualityGate/AccPassed",    "value", value=float(acc_pass), iteration=0)
    logger.report_scalar("QualityGate/F1Passed",     "value", value=float(f1_pass),  iteration=0)
    logger.report_scalar("QualityGate/RecallPassed", "value", value=float(rec_pass), iteration=0)

    if gate:
        om = OutputModel(task=task, name="EfficientNet-B0-BreastCancer-v0.2",
                         label_enumeration=CLASS_NAMES)
        om.update_weights(checkpoint_path)
        om.update_design(config_dict={
            "architecture": "EfficientNet-B0", "image_size": 128,
            "threshold": malignant_threshold,
            "test_accuracy": round(acc,4), "test_f1": round(f1,4),
            "test_recall": round(rec,4),   "test_auc": round(auc,4),
            "prd_targets_met": True, "classes": CLASS_NAMES,
        })
        logger.report_text("Model registered: EfficientNet-B0-BreastCancer-v0.2")

    if generate_gradcam:
        _run_gradcam(model, test_ds, device, task, logger, n_gradcam_samples)

    results = {"accuracy": round(acc,4), "f1": round(f1,4),
               "recall": round(rec,4),   "auc": round(auc,4),
               "gate_passed": gate}
    task.upload_artifact("eval_results", artifact_object=results)
    return results

def _run_gradcam(model, dataset, device, task, logger, n=6):
    try:
        import cv2
        model.eval()
        grads, acts = [], []

        def hook_fn(module, input, output):
            acts.append(output.detach())
            output.register_hook(lambda g: grads.append(g.detach()))

        hook = model.features[-1].register_forward_hook(hook_fn)
        generated = 0
        for idx in range(min(n*4, len(dataset))):
            if generated >= n: break
            img_t, true_lbl = dataset[idx]
            img_t = img_t.unsqueeze(0).to(device)
            grads.clear(); acts.clear()
            out  = model(img_t)
            pred = out.argmax(1).item()
            mal  = torch.softmax(out,dim=1)[0,1].item()
            model.zero_grad()
            out[0,pred].backward()
            if not grads or not acts: continue
            g = grads[0].squeeze().detach().cpu().numpy()
            a = acts[0].squeeze().detach().cpu().numpy()
            w = g.mean(axis=(1,2))
            cam = np.zeros(a.shape[1:], dtype=np.float32)
            for wi,ai in zip(w,a): cam += wi*ai
            cam = np.maximum(cam,0)
            cam = cv2.resize(cam,(128,128))
            cam = (cam-cam.min())/(cam.max()-cam.min()+1e-8)
            heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
            mean = np.array([0.485,0.456,0.406])
            std  = np.array([0.229,0.224,0.225])
            img_np = img_t.squeeze().cpu().numpy().transpose(1,2,0)
            img_np = np.clip(img_np*std+mean,0,1)
            img_bgr = cv2.cvtColor((img_np*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            overlay = cv2.addWeighted(img_bgr,0.55,heatmap,0.45,0)
            out_path = f"/tmp/gradcam_{generated}_true{true_lbl}_pred{pred}.jpg"
            cv2.imwrite(out_path, overlay)
            logger.report_image("Grad-CAM",
                f"img{generated} true={true_lbl} pred={pred} p(mal)={mal:.2f}",
                iteration=generated, local_path=out_path)
            task.upload_artifact(f"gradcam_{generated}", out_path)
            generated += 1
        hook.remove()
        logger.report_text(f"Grad-CAM: {generated} samples generated")
    except Exception as e:
        logger.report_text(f"Grad-CAM skipped: {e}")
