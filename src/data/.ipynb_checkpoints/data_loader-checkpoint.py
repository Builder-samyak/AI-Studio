import os
from pathlib import Path


def get_dataset_path():
    data_path = os.getenv("DATA_PATH")

    if data_path is None:
        raise ValueError("DATA_PATH environment variable not set")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    return Path(data_path)


def list_patient_folders():
    dataset_path = get_dataset_path()

    folders = [f for f in dataset_path.iterdir() if f.is_dir()]

    if len(folders) == 0:
        raise ValueError("No patient folders found")

    return folders