# save_final_model.py
import torch
import torchvision.models as models
import torch.nn as nn

# Parameters
num_classes = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model architecture
model = models.resnet18(weights=None)  # Use weights=None instead of pretrained=False
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes)
)
model = model.to(device)

# Load the saved weights
model.load_state_dict(torch.load("best_galaxy_classifier_resnet18.pth"))
print("Loaded model from 'best_galaxy_classifier_resnet18.pth'")

# Save with a new name
torch.save(model.state_dict(), "final_galaxy_classifier_resnet18.pth")
print("Final model saved as 'final_galaxy_classifier_resnet18.pth'")