from clearml import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_data(batch_size=32):

    dataset = Dataset.get(
        dataset_name="IDC_Breast_Cancer",
        dataset_project="AI-Studio"
    )

    data_path = dataset.get_local_copy()
    data_path = data_path + "/IDC_regular_ps50_idx5"

    transform = transforms.Compose([
        transforms.Resize((50, 50)),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(
        root=data_path,
        transform=transform
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )

    return train_loader, train_data.classes