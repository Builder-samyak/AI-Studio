# AI-Based Breast Cancer Detection System

**Team Destroyers** | 42174 Artificial Intelligence Studio — Autumn 2026 | University of Technology Sydney

| Role | Name |
|------|------|
| Project Manager / Product Owner | Harshitha Kolgatta Swamy |
| Data Scientist / Solution Designer | Aagusthya Shanker |
| Data Engineer / Tech Lead | Samyak Borkar |

---

## Project overview

An AI-assisted system for breast cancer detection using histopathology images. Classifies images as **Cancer** or **Non-Cancer** using a fine-tuned EfficientNet-B0 model. Provides Grad-CAM visual explanations and NLP-based plain-English explanations via a Streamlit web interface.

### Model results — full dataset (277,524 images)

| Model | Dataset | Accuracy | Recall | AUC |
|-------|---------|----------|--------|-----|
| SimpleCNN | 5k subset | 81.8% | 83.62% | 0.8939 |
| ResNet18 | 5k subset | 86.3% | 88.82% | 0.9330 |
| **EfficientNet-B0** | **Full 277k** | **91.97%** ✓ | **87.03%** ✓ | **0.9699** ✓ |

PRD targets: accuracy ≥ 90% ✓ · recall ≥ 85% ✓ · AUC ≥ 0.90 ✓

---

## Repository structure
AI-Studio/
├── src/
│   ├── data/
│   │   ├── data_loader.py          # ClearML dataset fetch
│   │   └── data_preprocessing.py  # Transforms, splits, BreastCancerDataset
│   ├── models/
│   │   ├── train.py                # EfficientNet-B0 training + ClearML logging
│   │   └── evaluate.py             # Test evaluation, quality gate, Grad-CAM, registry
│   ├── pipeline/
│   │   └── clearml_pipeline.py     # 4-stage ClearML pipeline controller
│   └── nlp/
│       └── explainer.py            # NLP explanation module
├── app/
│   └── streamlit_app.py            # Streamlit web UI
├── .github/
│   └── workflows/
│       └── pipeline.yml            # GitHub Actions CI/CD trigger
├── create_dataset.py               # ClearML dataset registration
├── data_loader.py                  # ClearML dataset loader
├── main.py                         # Pipeline entry point
└── requirements.txt
---

## MLOps pipeline (ClearML — Level 1/2)

4-stage automated pipeline:

| Stage | ClearML Task | Output |
|-------|-------------|--------|
| Step 1 | step_data_loading | train/val/test manifests (70/15/15 split) |
| Step 2 | step_preprocessing | Validated splits, transform config logged |
| Step 3 | step_training | Best EfficientNet-B0 checkpoint, per-epoch metrics |
| Step 4 | step_evaluation | Test metrics, quality gate, Grad-CAM, model registry |

CI/CD: GitHub Actions triggers the ClearML pipeline on every push to `main`.

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure ClearML
```bash
clearml-init
# Enter credentials from app.clear.ml → Settings → Workspace → API Credentials
```

### 3. Set dataset path
```bash
export DATA_PATH=/home/sagemaker-user/user-default-efs/data/IDC_regular_ps50_idx5
```

### 4. Quick test (200 images, 2 epochs)
```bash
SUBSET_SIZE=200 NUM_EPOCHS=2 PIPELINE_MODE=local python main.py
```

### 5. Full training run
```bash
python main.py
# Tracks at https://app.clear.ml
```

### 6. Run Streamlit UI
```bash
MODEL_PATH=/path/to/efficientnet_b0_best.pth streamlit run app/streamlit_app.py
```

---

## GitHub Actions CI/CD

Every push to `main` triggers the ClearML pipeline automatically.

Required GitHub secrets (Settings → Secrets → Actions):
- `CLEARML_API_ACCESS_KEY`
- `CLEARML_API_SECRET_KEY`

---

## Links

- [Jira Sprint Board](https://student-team-v90hneil.atlassian.net/jira/software/projects/D1/list)
- [Confluence Documentation](https://student-team-ucce3i5a.atlassian.net/wiki/spaces/Destroyers)
- [ClearML Dashboard](https://app.clear.ml)

---

## Disclaimer

This system is a research prototype and decision-support tool only. It is not a clinical diagnostic device and must not replace professional medical diagnosis.
