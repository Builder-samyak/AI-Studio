"""
src/pipeline/hpo.py
Destroyers | 42174 AI Studio Autumn 2026

Hyperparameter Optimisation using ClearML HyperParameterOptimizer.
Searches over: learning_rate, batch_size, dropout.
Objective: maximise validation F1-Score.

Usage:
    python src/pipeline/hpo.py --base_task_id <task_id> --n_trials 6
"""
import argparse
import os
import sys

from clearml import Task
from clearml.automation import (
    HyperParameterOptimizer,
    DiscreteParameterRange,
)

def run_hpo(
    base_task_id: str,
    project_name: str = "AI-Studio",
    n_trials: int = 6,
    execution_queue: str = "default",
):
    task = Task.init(
        project_name=project_name,
        task_name="HPO — EfficientNet-B0 Hyperparameter Search",
        task_type=Task.TaskTypes.optimizer,
        reuse_last_task_id=False,
    )
    task.set_parameter("hpo/n_trials",     n_trials)
    task.set_parameter("hpo/base_task_id", base_task_id)
    task.set_parameter("hpo/objective",    "F1-Score/val (maximise)")
    task.set_parameter("hpo/search_space", "learning_rate: [0.001,0.0001,0.00005] | batch_size: [32,64] | dropout: [0.2,0.3,0.4]")

    logger = task.get_logger()
    logger.report_text(f"Starting HPO with {n_trials} trials on base task: {base_task_id}")

    try:
        from clearml.automation.optuna import OptunaObjective
        optimizer_class = OptunaObjective
        logger.report_text("Using Optuna optimizer")
    except ImportError:
        from clearml.automation import RandomSearch
        optimizer_class = RandomSearch
        logger.report_text("Optuna not found — using RandomSearch")

    optimizer = HyperParameterOptimizer(
        base_task_id=base_task_id,
        hyper_parameters=[
            DiscreteParameterRange("train/learning_rate", values=[0.001, 0.0001, 0.00005]),
            DiscreteParameterRange("train/batch_size",    values=[32, 64]),
            DiscreteParameterRange("train/dropout",       values=[0.2, 0.3, 0.4]),
        ],
        objective_metric_title="F1-Score",
        objective_metric_series="val",
        objective_metric_sign="max",
        max_number_of_concurrent_tasks=1,
        execution_queue=execution_queue,
        total_max_jobs=n_trials,
        min_iteration_per_job=1,
        max_iteration_per_job=15,
        optimizer_class=optimizer_class,
    )

    optimizer.set_report_period(1)
    optimizer.start_locally(job_complete_timeout=7200)
    optimizer.wait()

    top = optimizer.get_top_experiments(top_k=3)
    logger.report_text("\n=== HPO Top Results ===")
    for i, exp in enumerate(top):
        params  = exp.get_parameters_as_dict()
        scalars = exp.get_last_scalar_metrics()
        f1  = scalars.get("F1-Score",{}).get("val",{}).get("last","?")
        acc = scalars.get("Accuracy", {}).get("val",{}).get("last","?")
        lr  = params.get("train",{}).get("learning_rate","?")
        bs  = params.get("train",{}).get("batch_size","?")
        do  = params.get("train",{}).get("dropout","?")
        msg = f"  #{i+1}: lr={lr}  batch={bs}  dropout={do}  →  F1={f1}  acc={acc}"
        logger.report_text(msg)
        print(msg)

    if top:
        best = top[0].get_parameters_as_dict().get("train",{})
        logger.report_text(
            f"\nBest: lr={best.get('learning_rate')}  "
            f"batch={best.get('batch_size')}  "
            f"dropout={best.get('dropout')}"
        )

    optimizer.stop()
    task.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EfficientNet-B0 HPO")
    parser.add_argument("--base_task_id", required=True,
        help="ClearML task ID of a completed EfficientNet-B0 training run")
    parser.add_argument("--n_trials",  type=int, default=6)
    parser.add_argument("--queue",     default="default")
    parser.add_argument("--project",   default="AI-Studio")
    args = parser.parse_args()
    run_hpo(args.base_task_id, args.project, args.n_trials, args.queue)
