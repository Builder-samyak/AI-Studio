import os
from PIL import Image

def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    return img

def get_all_images(dataset_path):
    image_paths = []

    for patient in os.listdir(dataset_path):
        patient_path = os.path.join(dataset_path, patient)

        if not os.path.isdir(patient_path):
            continue

        for root, _, files in os.walk(patient_path):
            for file in files:
                if file.endswith(".png"):
                    image_paths.append(os.path.join(root, file))

    return image_paths
