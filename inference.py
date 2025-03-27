# inference.py
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from astroNN.datasets.galaxy10sdss import galaxy10cls_lookup
import matplotlib.pyplot as plt

# Step 1: Set parameters and device
num_classes = 10  # Galaxy10 has 10 classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 2: Load class names
class_names = [galaxy10cls_lookup(i) for i in range(num_classes)]
print("Class names:", class_names)

# Step 3: Load the saved model
model = models.resnet18(weights=None)
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

# Step 4: Define image preprocessing
img_size = (224, 224)
preprocess = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Step 5: Function to load and preprocess an image
def load_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Step 6: Function to predict the class of an image
def predict_image(image_path):
    image = load_image(image_path)
    if image is None:
        return None, None
    
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_class = class_names[predicted.item()]
        confidence_score = confidence.item()
    
    return predicted_class, confidence_score

# Step 7: Main execution
if __name__ == "__main__":
    # List of image paths to classify
    image_paths = [
        "test_images/test_image_0_class_Disk_Face-on_No_Spiral.png",
        "test_images/test_image_1_class_Smooth_Completely_round.png",
        "test_images/test_image_2_class_Smooth_in-between_round.png",
        "test_images/test_image_3_class_Smooth_Cigar_shaped.png",
        "test_images/test_image_4_class_Disk_Edge-on_Rounded_Bulge.png",
    ]
    
    for i, image_path in enumerate(image_paths):
        print(f"\nClassifying image: {image_path}")
        predicted_class, confidence = predict_image(image_path)
        if predicted_class is not None:
            print(f"Predicted Class: {predicted_class}")
            print(f"Confidence: {confidence:.4f}")
            # Display and save the image
            img = Image.open(image_path)
            plt.figure(figsize=(4, 4))
            plt.imshow(img)
            plt.title(f"Predicted: {predicted_class}\nConfidence: {confidence:.4f}")
            plt.axis("off")
            # Save the plot
            plt.savefig(f"prediction_{i}.png", dpi=300, bbox_inches="tight")
            plt.close()  # Close the figure to free memory
        else:
            print("Failed to predict the class. Please check the image path.")