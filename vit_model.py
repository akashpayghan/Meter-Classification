import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast
from PIL import Image
import os
import shutil

# Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['AugTheft', 'Nil']

# Custom Dataset for test images (no labels required)
class TestImageDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path

# Create Vision Transformer model
def create_vit_model(num_classes=NUM_CLASSES):
    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model.to(DEVICE)

# Test and save classified images
def test_and_save(model, test_loader, output_dir):
    model.eval()
    os.makedirs(os.path.join(output_dir, 'AugTheft'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'Nil'), exist_ok=True)

    aug_theft_count = 0
    nil_count = 0

    with torch.no_grad():
        for batch_idx, (images, img_paths) in enumerate(test_loader):
            images = images.to(DEVICE)
            with autocast(DEVICE.type):
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

            for img_path, pred in zip(img_paths, predicted):
                class_name = CLASS_NAMES[pred.item()]
                dest_path = os.path.join(output_dir, class_name, os.path.basename(img_path))
                shutil.copy(img_path, dest_path)
                if class_name == 'AugTheft':
                    aug_theft_count += 1
                else:
                    nil_count += 1

    return aug_theft_count, nil_count

# Function to create a zip file of the output directory
def create_zip(output_dir, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, output_dir)
                zipf.write(file_path, arcname)