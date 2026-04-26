from clearml import Task, Dataset

task = Task.init(
    project_name="AI-Studio",
    task_name="Create IDC Dataset"
)

DATA_PATH = "/home/sagemaker-user/user-default-efs/data/IDC_regular_ps50_idx5"

dataset = Dataset.create(
    dataset_name="IDC_Breast_Cancer",
    dataset_project="AI-Studio"
)

dataset.add_files(DATA_PATH)

dataset.upload()

dataset.finalize()

print("Dataset registered successfully")