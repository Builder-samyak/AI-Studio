"""
src/data/data_preprocessing.py
Destroyers | 42174 AI Studio Autumn 2026
Transforms, BreastCancerDataset, 70/15/15 split.
Dataset structure: IDC_regular_ps50_idx5/{patient_id}/0/*.png and /1/*.png
"""
import os, glob, random, json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

IMAGE_SIZE    = 128
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
RANDOM_SEED   = 42

def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

class BreastCancerDataset(Dataset):
    def __init__(self, manifest, transform=None):
        self.manifest  = manifest
        self.transform = transform
    def __len__(self):
        return len(self.manifest)
    def __getitem__(self, idx):
        item  = self.manifest[idx]
        image = Image.open(item["path"]).convert("RGB")
        label = int(item["label"])
        if self.transform:
            image = self.transform(image)
        return image, label

def build_manifest(data_root, subset_size=None):
    random.seed(RANDOM_SEED)
    cancer, non_cancer = [], []
    for patient in sorted(os.listdir(data_root)):
        pp = os.path.join(data_root, patient)
        if not os.path.isdir(pp): continue
        cancer     += glob.glob(os.path.join(pp, "1", "*.png"))
        non_cancer += glob.glob(os.path.join(pp, "0", "*.png"))
    print(f"  Cancer: {len(cancer)}  Non-cancer: {len(non_cancer)}  Total: {len(cancer)+len(non_cancer)}")
    if subset_size:
        half = subset_size // 2
        cancer     = random.sample(cancer,     min(half, len(cancer)))
        non_cancer = random.sample(non_cancer, min(half, len(non_cancer)))
        print(f"  Using balanced subset: {len(cancer)+len(non_cancer)}")
    manifest = (
        [{"path": p, "label": 1} for p in cancer] +
        [{"path": p, "label": 0} for p in non_cancer]
    )
    random.shuffle(manifest)
    return manifest

def split_manifest(manifest):
    n       = len(manifest)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    train   = manifest[:n_train]
    val     = manifest[n_train:n_train+n_val]
    test    = manifest[n_train+n_val:]
    print(f"  Split → train:{len(train)} val:{len(val)} test:{len(test)}")
    return train, val, test

def save_manifest(manifest, path):
    with open(path, "w") as f: json.dump(manifest, f)

def load_manifest(path):
    with open(path) as f: return json.load(f)
