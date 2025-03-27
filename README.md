# Galaxy Morphological Classification CNN Model

This project uses a Convolutional Neural Network (CNN) to classify galaxy morphologies using the Galaxy10 SDSS dataset. The model is built with PyTorch and deployed as a Flask web app, allowing users to upload galaxy images and receive classification predictions.

## Project Structure
- `app.py`: Flask app for serving the web interface and making predictions.
- `templates/index.html`: HTML template for the web interface.
- `static/uploads/`: Directory for storing uploaded images (excluded from Git).
- `test_images/`: Sample galaxy images for testing.
- `final_galaxy_classifier_resnet18.pth`: Trained model weights (excluded from Git due to size).
- `requirements.txt`: List of Python dependencies.
- `extract_test_images.py`: Script to extract test images from the Galaxy10 SDSS dataset.

## Prerequisites
- Python 3.8 or higher
- PyTorch and torchvision (for model inference)
- Flask (for the web app)
- astroNN (for Galaxy10 SDSS dataset utilities)

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/galaxy-morphological-classification.git
   cd galaxy-morphological-classification