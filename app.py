import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from astroNN.datasets.galaxy10sdss import galaxy10cls_lookup
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import time

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize model only once when app starts
def init_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model setup
    num_classes = 10
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    model.load_state_dict(torch.load("final_galaxy_classifier_resnet18.pth", map_location=device))
    model = model.to(device)
    model.eval()
    
    # Preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return model, preprocess, device, [galaxy10cls_lookup(i) for i in range(num_classes)]

model, preprocess, device, class_names = init_model()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            return class_names[predicted.item()], round(confidence.item() * 100, 2)
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file selected")
            
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No file selected")
            
        if not allowed_file(file.filename):
            return render_template('index.html', error="Invalid file type. Please upload JPEG or PNG")
            
        try:
            filename = secure_filename(f"upload_{int(time.time())}.jpg")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            predicted_class, confidence = predict_image(filepath)
            if not predicted_class:
                return render_template('index.html', error="Failed to process image")
                
            return render_template('index.html', 
                                 image_path=filename,
                                 predicted_class=predicted_class,
                                 confidence=confidence)
                                 
        except Exception as e:
            print(f"Upload error: {e}")
            return render_template('index.html', error="Error processing your request")
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)