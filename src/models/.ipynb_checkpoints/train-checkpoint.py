import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim

from src.data.data_preprocessing import get_all_images


class BreastCancerDataset(Dataset):
    def __init__(self, dataset_path, subset_size=None):
        all_images = get_all_images(dataset_path)

        if subset_size is not None:
            self.image_paths = all_images[:subset_size]
        else:
            self.image_paths = all_images

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        image = Image.open(path).convert("RGB")
        image = image.resize((50, 50))
        image = torch.tensor(list(image.getdata()), dtype=torch.float32)
        image = image.view(50, 50, 3).permute(2, 0, 1) / 255.0

        label = 1 if "/1/" in path else 0

        return image, label


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(32 * 11 * 11, 2)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def train_model(dataset_path):

    subset_size = int(os.getenv("SUBSET_SIZE", "3000"))
    epochs = int(os.getenv("EPOCHS", "3"))

    dataset = BreastCancerDataset(dataset_path, subset_size=subset_size)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print(f"Training on {len(dataset)} images for {epochs} epochs")

    for epoch in range(epochs):
        total_loss = 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    return model