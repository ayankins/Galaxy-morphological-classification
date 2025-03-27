# confusion_matrix.py
import torch
import torch.nn as nn
import torchvision.models as models
from data_preparation import load_galaxy_data
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from astroNN.datasets.galaxy10sdss import galaxy10cls_lookup

# Step 1: Set parameters and device
num_classes = 10  # Galaxy10 has 10 classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 2: Load class names for labeling the confusion matrix
class_names = [galaxy10cls_lookup(i) for i in range(num_classes)]
print("Class names:", class_names)

# Step 3: Load the saved model
model = models.resnet18(weights=None)  # Use weights=None instead of pretrained=False
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes)
)
model_path = "final_galaxy_classifier_resnet18.pth"
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()
print(f"Loaded model from {model_path}")

# Step 4: Load the test data
print("Loading test data...")
_, _, test_loader, _ = load_galaxy_data()
print("Test data loaded successfully!")

if __name__ == '__main__':
    # Step 5: Get predictions and true labels from the test set
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            print(f"Processed batch {batch_idx+1}/{len(test_loader)}")

    # Step 6: Compute the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix (raw):")
    print(cm)

    # Step 7: Plot the confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix for Galaxy10 Classification')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    # Save the plot as an image
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Step 8: Compute per-class accuracy (optional, for deeper analysis)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    print("\nPer-class Accuracy:")
    for i, (class_name, acc) in enumerate(zip(class_names, per_class_accuracy)):
        print(f"{class_name}: {acc:.4f}")