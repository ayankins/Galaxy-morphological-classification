# extract_test_images.py
import numpy as np
from astroNN.datasets import load_galaxy10sdss
from astroNN.datasets.galaxy10sdss import galaxy10cls_lookup
from sklearn.model_selection import train_test_split
import cv2
import os

# Create a directory to save test images
output_dir = "test_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the dataset
images, labels = load_galaxy10sdss()
print(f"Loaded {len(images)} images with shape {images.shape}")

# Split into train+val and test
train_val_idx, test_idx = train_test_split(
    np.arange(len(images)), test_size=0.1, random_state=42
)

# Select one image from each class (up to 5 classes)
selected_indices = []
selected_labels = []
num_classes = 10  # Galaxy10 has 10 classes
test_labels = labels[test_idx]

for class_id in range(num_classes):
    # Find indices in the test set for this class
    class_indices = test_idx[np.where(test_labels == class_id)[0]]
    if len(class_indices) > 0:
        # Select the first image from this class
        selected_indices.append(class_indices[0])
        selected_labels.append(class_id)
    if len(selected_indices) >= 5:  # Stop after selecting 5 images
        break

# Save the selected images
for i, (idx, label) in enumerate(zip(selected_indices, selected_labels)):
    img = images[idx]
    class_name = galaxy10cls_lookup(label).replace(", ", "_").replace(" ", "_")
    print(f"Index: {idx}, Label: {label}, Class Name: {class_name}")
    filename = f"{output_dir}/test_image_{i}_class_{class_name}.png"
    success = cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if success:
        print(f"Saved {filename}")
    else:
        print(f"Failed to save {filename}")

print("Test images saved successfully!")