# Galaxy Morphological Classification CNN Model

This project uses a Convolutional Neural Network (CNN) based on **ResNet18** to classify galaxy morphologies using the Galaxy10 SDSS dataset. The model is built with PyTorch, fine-tuned on the Galaxy10 SDSS dataset, and deployed as a Flask web app. Users can upload galaxy images through the web interface and receive classification predictions, including the galaxy type and confidence score, presented with an astronomical-themed UI (Orbitron font, space background, starry overlay, and cyan glowing effects).

## Project Structure
- `app.py`: Flask app for serving the web interface and making predictions.
- `templates/index.html`: HTML template for the web interface.
- `static/`: Directory for static files (CSS, JavaScript, etc.) used in the web app.
- `static/uploads/`: Directory for storing uploaded images (excluded from Git).
- `test_images/`: Sample galaxy images for testing (e.g., `test_image_0_class_Disk_Face-on_No_Spiral.png`).
- `final_galaxy_classifier_resnet18.pth`: Trained ResNet18 model weights (excluded from Git due to size).
- `requirements.txt`: List of Python dependencies.
- `extract_test_images.py`: Script to extract test images from the Galaxy10 SDSS dataset.
- `data_preparation.py`: Script for preparing the Galaxy10 SDSS dataset for training.
- `train_model.py`: Script for training the ResNet18 model on the Galaxy10 SDSS dataset.
- `save_final_model.py`: Script to save the trained model weights.
- `inference.py`: Script for running inference on new images using the trained model.
- `confusion_matrix.py`: Script to generate a confusion matrix for model evaluation.
- `confusion_matrix.png`: Generated confusion matrix visualization.
- `prediction_0.png` to `prediction_4.png`: Sample prediction visualizations.
- `Plots/`: Directory containing additional plots and visualizations.
- `confusion matrix/`: Directory containing confusion matrix-related files.

## Prerequisites
- Python 3.8 or higher
- PyTorch and torchvision (for model inference and training)
- Flask (for the web app)
- astroNN (for Galaxy10 SDSS dataset utilities)
- Other dependencies listed in `requirements.txt`

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ayankins/Galaxy-morphological-classification.git
   cd Galaxy-morphological-classification

2.  **Set Up a Virtual Environment** (optional but recommended):bashCollapseUnwrapCopypython -m venv venvsource venv/bin/activate _\# On Windows: venv\\Scripts\\activate_
    
2.  **Install Dependencies**:bashCollapseUnwrapCopypip install -r requirements.txt
    
3.  **Download the Model Weights**:
    
    *   Download the trained ResNet18 model weights (final\_galaxy\_classifier\_resnet18.pth) from \[link-to-your-model\] and place it in the project directory.
        
    *   **Note**: Replace \[link-to-your-model\] with the actual link after hosting the file (see Step 5 below).
        
4.  **Host the Model Weights** (if you’re the project owner):
    
    *   Upload final\_galaxy\_classifier\_resnet18.pth to a file hosting service (e.g., Google Drive, Dropbox).
        
    *   Set the sharing settings to “Anyone with the link can view”.
        
    *   Copy the shareable link and update the README.md with the actual link in the “Download the Model Weights” step above.
        

Usage
-----

1.  **Run the Flask App**:bashCollapseUnwrapCopypython app.py
    
    *   The app will start on http://localhost:5000.
        
2.  **Access the Web Interface**:
    
    *   Open your browser and go to http://localhost:5000.
        
    *   The web interface features an astronomical theme with the Orbitron font, a space background, a starry overlay, and cyan glowing effects.
        
3.  **Upload a Galaxy Image**:
    
    *   Use the upload button to select a galaxy image (e.g., from the test\_images/ directory, such as test\_image\_0\_class\_Disk\_Face-on\_No\_Spiral.png).
        
    *   Click “Submit” to classify the galaxy.
        
4.  **View the Results**:
    
    *   The “Classification Result” section will display:
        
        *   The uploaded galaxy image as a 150x150 thumbnail with a cyan border and glow.
            
        *   **Type**: The predicted galaxy type (e.g., “Disk, Face-on, No Spiral”) in bold with a cyan glow.
            
        *   **Confidence**: The prediction confidence (e.g., 92.34%) in bold with a cyan glow.
            
        *   A confidence bar filled to the corresponding percentage with a cyan-to-blue gradient.
            

Model Details
-------------

*   **Architecture**: ResNet18 (pre-trained on ImageNet, fine-tuned on the Galaxy10 SDSS dataset).
    
*   **Dataset**: Galaxy10 SDSS dataset, containing 10 classes of galaxy morphologies.
    
*   **Classes**:
    
    *   Disk, Face-on, No Spiral
        
    *   Disk, Face-on, Tight Spiral
        
    *   Disk, Face-on, Medium Spiral
        
    *   Disk, Face-on, Loose Spiral
        
    *   Disk, Edge-on, No Bulge
        
    *   Disk, Edge-on, Bulge
        
    *   Smooth, Completely Round
        
    *   Smooth, In-between Round
        
    *   Smooth, Cigar-shaped
        
    *   Unclassifiable
        
*   **Training**: The model was trained using PyTorch, with scripts provided in train\_model.py and save\_final\_model.py.
    
*   **Evaluation**: A confusion matrix is provided in confusion\_matrix.png, generated using confusion\_matrix.py.
    

Testing with Sample Images
--------------------------

*   The test\_images/ directory contains sample galaxy images for testing.
    
*   Example: Upload test\_images/test\_image\_0\_class\_Disk\_Face-on\_No\_Spiral.png to the web app.
    
*   Expected Output:
    
    *   **Type**: Disk, Face-on, No Spiral
        
    *   **Confidence**: ~92% (actual confidence may vary depending on the model’s performance).
        

Visualizations
--------------

*   **Confusion Matrix**: See confusion\_matrix.png for the model’s performance across all classes.
    
*   **Prediction Visualizations**: See prediction\_0.png to prediction\_4.png for sample predictions.
    
*   **Additional Plots**: The Plots/ directory contains additional visualizations generated during training and evaluation.
    

Deployment (Optional)
---------------------

To deploy the app online (e.g., on Heroku or Render), follow these steps:

1.  Ensure all dependencies are listed in requirements.txt.
    
2.  Update app.py to use a production server (e.g., Gunicorn):bashCollapseUnwrapCopypip install gunicorn
    
3.  Create a Procfile for Heroku:textCollapseUnwrapCopyweb: gunicorn app:app
    
4.  Deploy to Heroku:bashCollapseUnwrapCopyheroku creategit push heroku main
    
5.  Ensure the model weights (final\_galaxy\_classifier\_resnet18.pth) are accessible to the deployed app.
    

Contributing
------------

Feel free to fork this repository, make improvements, and submit pull requests. For major changes, please open an issue to discuss your ideas.

Acknowledgments
---------------

*   The Galaxy10 SDSS dataset is provided by the astroNN library.
    
*   ResNet18 architecture is based on the implementation in PyTorch’s torchvision library.
    
*   The web interface design is inspired by astronomical themes, using the Orbitron font and space aesthetics.   
