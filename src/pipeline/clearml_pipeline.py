from clearml import Task
from clearml.automation.controller import PipelineDecorator

import os

# Init ClearML Task
Task.init(project_name="AI-Studio", task_name="Breast Cancer Pipeline FINAL")

# -----------------------------
# STEP 1: DATA LOADING (100 ONLY)
# -----------------------------
@PipelineDecorator.component(return_values=["data_loaded"])
def data_loading_step():
    print("STEP 1: Loading ONLY 100 images")

    data_path = os.environ.get("DATA_PATH")

    all_files = []
    for root, _, files in os.walk(data_path):
        for f in files:
            if f.endswith(".png"):
                all_files.append(os.path.join(root, f))

    sample_files = all_files[:100]

    print(f"Loaded {len(sample_files)} images")
    return sample_files


# -----------------------------
# STEP 2: PREPROCESSING (LIGHT)
# -----------------------------
@PipelineDecorator.component(return_values=["data_preprocessed"])
def data_preprocessing_step(data_loaded):
    print(f"STEP 2: Preprocessing {len(data_loaded)} images")

    processed = [f for f in data_loaded]

    print("Preprocessing done")
    return processed


# -----------------------------
# STEP 3: TRAINING (FAST DUMMY)
# -----------------------------
@PipelineDecorator.component(return_values=["model"])
def train_model_step(data_preprocessed):
    print("STEP 3: Training on SMALL SAMPLE")

    import torch
    import torch.nn as nn

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(50 * 50 * 3, 2)
    )

    for i in range(2):
        print(f"Epoch {i+1}/2 done")

    print("Training finished (sample run)")
    return model


# -----------------------------
# PIPELINE
# -----------------------------
@PipelineDecorator.pipeline(
    name="Breast Cancer Pipeline",
    project="AI-Studio",
    version="1.0.0"
)
def pipeline():
    data_loaded = data_loading_step()
    data_preprocessed = data_preprocessing_step(data_loaded)
    train_model_step(data_preprocessed)


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    PipelineDecorator.run_locally()
    pipeline()