"""
main.py — Destroyers | 42174 AI Studio Autumn 2026
Entry point. Configure via environment variables.
Quick test:  SUBSET_SIZE=200 NUM_EPOCHS=2 PIPELINE_MODE=local python main.py
Full run:    python main.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from src.pipeline.clearml_pipeline import (
    pipeline, PipelineDecorator,
    DATA_PATH, SUBSET_SIZE, NUM_EPOCHS,
    LEARNING_RATE, BATCH_SIZE, DROPOUT,
    PIPELINE_NAME, PROJECT_NAME, REPO_ROOT,
)

if __name__ == "__main__":
    print("=" * 52)
    print("  Breast Cancer Detection — Team Destroyers")
    print("  42174 AI Studio, Autumn 2026")
    print("=" * 52)
    print(f"  Project:  {PROJECT_NAME}")
    print(f"  Pipeline: {PIPELINE_NAME}")
    print(f"  Data:     {DATA_PATH}")
    print(f"  Subset:   {'full dataset' if SUBSET_SIZE==0 else SUBSET_SIZE}")
    print(f"  Epochs:   {NUM_EPOCHS}")
    print(f"  Batch:    {BATCH_SIZE}")
    print(f"  LR:       {LEARNING_RATE}")
    print(f"  Mode:     {os.getenv('PIPELINE_MODE','remote')}")
    print("=" * 52)
    if os.getenv("PIPELINE_MODE","remote") == "local":
        PipelineDecorator.run_locally()
    pipeline(
        data_path=DATA_PATH, subset_size=SUBSET_SIZE,
        num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE, dropout=DROPOUT, repo_root=REPO_ROOT,
    )
    print("Pipeline triggered. Track at https://app.clear.ml")
