"""
src/pipeline/clearml_pipeline.py
Destroyers | 42174 AI Studio Autumn 2026
4-stage ClearML pipeline (Level 1 MLOps):
  Step 1: Data Loading
  Step 2: Preprocessing validation
  Step 3: EfficientNet-B0 Training
  Step 4: Evaluation + quality gate + model registry
"""
import os, sys, json, glob, random
from clearml.automation.controller import PipelineDecorator
from clearml import Task

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

PROJECT_NAME    = os.getenv("CLEARML_PROJECT",   "AI-Studio")
PIPELINE_NAME   = os.getenv("CLEARML_PIPELINE",  "Breast Cancer Pipeline")
PIPELINE_VER    = "2.0.0"
EXECUTION_QUEUE = os.getenv("CLEARML_QUEUE",     "default")
DATA_PATH       = os.getenv("DATA_PATH", "/home/sagemaker-user/user-default-efs/data/IDC_regular_ps50_idx5")
SUBSET_SIZE     = int(os.getenv("SUBSET_SIZE",   "0"))
NUM_EPOCHS      = int(os.getenv("NUM_EPOCHS",    "15"))
LEARNING_RATE   = float(os.getenv("LEARNING_RATE","1e-4"))
BATCH_SIZE      = int(os.getenv("BATCH_SIZE",    "64"))
DROPOUT         = float(os.getenv("DROPOUT",     "0.3"))
REPO_ROOT       = os.getenv("REPO_ROOT", "/home/sagemaker-user/AI-Studio-GPU")

@PipelineDecorator.component(
    return_values=["train_manifest_path","val_manifest_path","test_manifest_path"],
    execution_queue=EXECUTION_QUEUE,
    packages=["Pillow","clearml"],
)
def step_data_loading(data_path, subset_size):
    import os, json, glob, random
    from clearml import Task
    logger = Task.current_task().get_logger()
    random.seed(42)
    cancer, non_cancer = [], []
    for patient in sorted(os.listdir(data_path)):
        pp = os.path.join(data_path, patient)
        if not os.path.isdir(pp): continue
        cancer     += glob.glob(os.path.join(pp,"1","*.png"))
        non_cancer += glob.glob(os.path.join(pp,"0","*.png"))
    logger.report_text(f"Cancer: {len(cancer)} | Non-cancer: {len(non_cancer)} | Total: {len(cancer)+len(non_cancer)}")
    Task.current_task().get_logger().report_scalar("Dataset/Cancer",    "count", value=len(cancer),                    iteration=0)
    Task.current_task().get_logger().report_scalar("Dataset/NonCancer", "count", value=len(non_cancer),                iteration=0)
    Task.current_task().get_logger().report_scalar("Dataset/Total",     "count", value=len(cancer)+len(non_cancer),    iteration=0)
    if subset_size > 0:
        half = subset_size // 2
        cancer     = random.sample(cancer,     min(half,len(cancer)))
        non_cancer = random.sample(non_cancer, min(half,len(non_cancer)))
        logger.report_text(f"Balanced subset: {len(cancer)+len(non_cancer)}")
    manifest = ([{"path":p,"label":1} for p in cancer] +
                [{"path":p,"label":0} for p in non_cancer])
    random.shuffle(manifest)
    n = len(manifest)
    n_train = int(n*0.70); n_val = int(n*0.15)
    train = manifest[:n_train]
    val   = manifest[n_train:n_train+n_val]
    test  = manifest[n_train+n_val:]
    logger.report_text(f"Split → train:{len(train)} val:{len(val)} test:{len(test)}")
    Task.current_task().get_logger().report_scalar("Split/Train","count",value=len(train),iteration=0)
    Task.current_task().get_logger().report_scalar("Split/Val",  "count",value=len(val),  iteration=0)
    Task.current_task().get_logger().report_scalar("Split/Test", "count",value=len(test), iteration=0)
    train_path,val_path,test_path = "/tmp/train_m.json","/tmp/val_m.json","/tmp/test_m.json"
    for path,data in [(train_path,train),(val_path,val),(test_path,test)]:
        with open(path,"w") as f: json.dump(data,f)
    Task.current_task().upload_artifact("train_manifest", train_path)
    Task.current_task().upload_artifact("val_manifest",   val_path)
    Task.current_task().upload_artifact("test_manifest",  test_path)
    return train_path, val_path, test_path

@PipelineDecorator.component(
    return_values=["train_manifest_path","val_manifest_path","test_manifest_path"],
    execution_queue=EXECUTION_QUEUE,
    packages=["clearml"],
)
def step_preprocessing(train_manifest_path, val_manifest_path, test_manifest_path):
    import json
    from clearml import Task
    logger = Task.current_task().get_logger()
    def validate(path, name):
        with open(path) as f: m = json.load(f)
        cancer = sum(1 for x in m if x["label"]==1)
        logger.report_text(f"{name}: {len(m)} images | cancer={cancer} non-cancer={len(m)-cancer}")
        Task.current_task().get_logger().report_scalar(f"Validation/{name}","count",value=len(m),iteration=0)
    validate(train_manifest_path,"Train")
    validate(val_manifest_path,  "Val")
    validate(test_manifest_path, "Test")
    logger.report_text("Transforms: Resize(128,128) | RandomFlip | ColorJitter | ToTensor | Normalize(ImageNet)")
    return train_manifest_path, val_manifest_path, test_manifest_path

@PipelineDecorator.component(
    return_values=["checkpoint_path"],
    execution_queue=EXECUTION_QUEUE,
    packages=["torch","torchvision","scikit-learn","Pillow","clearml"],
)
def step_training(train_manifest_path, val_manifest_path,
                  num_epochs, learning_rate, batch_size, dropout, repo_root):
    import sys
    sys.path.insert(0, repo_root)
    from src.models.train import train_model
    from clearml import Task
    return train_model(
        train_manifest_path=train_manifest_path,
        val_manifest_path=val_manifest_path,
        num_epochs=num_epochs, learning_rate=learning_rate,
        batch_size=batch_size, dropout=dropout,
        output_dir="/tmp/destroyers_model",
        task=Task.current_task(),
    )

@PipelineDecorator.component(
    return_values=["eval_results"],
    execution_queue=EXECUTION_QUEUE,
    packages=["torch","torchvision","scikit-learn","opencv-python","Pillow","clearml"],
)
def step_evaluation(checkpoint_path, test_manifest_path, repo_root):
    import sys
    sys.path.insert(0, repo_root)
    from src.models.evaluate import evaluate_model
    from clearml import Task
    return evaluate_model(
        checkpoint_path=checkpoint_path,
        test_manifest_path=test_manifest_path,
        generate_gradcam=True, n_gradcam_samples=6,
        task=Task.current_task(),
    )

@PipelineDecorator.pipeline(
    name=PIPELINE_NAME,
    project=PROJECT_NAME,
    version=PIPELINE_VER,
)
def pipeline(data_path, subset_size, num_epochs, learning_rate,
             batch_size, dropout, repo_root):
    train_m, val_m, test_m = step_data_loading(
        data_path=data_path, subset_size=subset_size)
    train_m, val_m, test_m = step_preprocessing(
        train_manifest_path=train_m, val_manifest_path=val_m, test_manifest_path=test_m)
    ckpt = step_training(
        train_manifest_path=train_m, val_manifest_path=val_m,
        num_epochs=num_epochs, learning_rate=learning_rate,
        batch_size=batch_size, dropout=dropout, repo_root=repo_root)
    results = step_evaluation(
        checkpoint_path=ckpt, test_manifest_path=test_m, repo_root=repo_root)
    return results

if __name__ == "__main__":
    run_mode = os.getenv("PIPELINE_MODE","remote")
    if run_mode == "local":
        PipelineDecorator.run_locally()
    pipeline(
        data_path=DATA_PATH, subset_size=SUBSET_SIZE,
        num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE, dropout=DROPOUT, repo_root=REPO_ROOT,
    )
