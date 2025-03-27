# data_preparation.py
import numpy as np
from astroNN.datasets import load_galaxy10sdss
from astroNN.datasets.galaxy10sdss import galaxy10cls_lookup
from sklearn.model_selection import train_test_split
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Parameters
img_size = (224, 224)
batch_size = 16
data_path = "C:/Users/Ayan/.astroNN/datasets/Galaxy10.h5"

# Custom dataset class for PyTorch
class GalaxyDataset(Dataset):
    def __init__(self, images, labels, indices, target_size, transform=None):
        self.images = images
        self.labels = labels
        self.indices = indices
        self.target_size = target_size
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_idx = self.indices[idx]
        img = cv2.resize(self.images[img_idx], self.target_size, interpolation=cv2.INTER_AREA)
        img = (img / 255.0).astype(np.float32)
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[img_idx], dtype=torch.long)
        return img, label

# Function to load data and create data loaders
def load_galaxy_data():
    print("Loading Galaxy10 SDSS dataset...")
    images, labels = load_galaxy10sdss()
    print(f"Loaded {len(images)} images with shape {images.shape}")

    # Split into train+val and test
    train_val_idx, test_idx = train_test_split(
        np.arange(len(images)), test_size=0.1, random_state=42
    )
    # Split train+val into train and val
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.2, random_state=42
    )

    # Define transforms with increased augmentation
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = GalaxyDataset(images, labels, train_idx, img_size, transform=train_transform)
    val_dataset = GalaxyDataset(images, labels, val_idx, img_size, transform=val_test_transform)
    test_dataset = GalaxyDataset(images, labels, test_idx, img_size, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    num_classes = len(np.unique(labels))
    class_names = [galaxy10cls_lookup(i) for i in range(num_classes)]
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    print(f"Training samples: {len(train_idx)}")
    print(f"Validation samples: {len(val_idx)}")
    print(f"Test samples: {len(test_idx)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader, num_classes

if __name__ == "__main__":
    # This block only runs if data_preparation.py is executed directly
    train_loader, val_loader, test_loader, num_classes = load_galaxy_data()
    print("Data preparation completed successfully!")
    batch_images, batch_labels = next(iter(train_loader))
    print(f"Batch shape: {batch_images.shape}")
    print(f"Labels shape: {batch_labels.shape}")