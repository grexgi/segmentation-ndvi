import os
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image
from matplotlib.image import imread
import io

# Define the Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Path to your YOLO model
data_best_model = "best.pt"

def segmentation(data_best_model, input_path):
    try:
        # Load the YOLO model
        model = YOLO(data_best_model)

        # Perform segmentation
        results = model.predict(input_path, save=False)

        image = imread(input_path)

        original_image = Image.open(input_path)
        original_width, original_height = original_image.size

        # Attempt to get the mask and resize it to the original image size
        mask_tensor = results[0].masks.data.cpu().numpy()

        # Ensure the mask is in uint8 format and has the correct shape (2D)
        mask = (mask_tensor[0] * 255).astype(np.uint8)

        # Resize the mask to match the original image dimensions
        resized_mask = np.array(Image.fromarray(mask).resize((original_width, original_height), Image.NEAREST))

        # Apply the mask to the image
        cropped = np.copy(image)
        cropped[resized_mask == 0] = 0
        
    except (AttributeError, IndexError) as e:
        print(f"Error during segmentation: {e}. Returning original image.")
        # If an error occurs, return the original image
        cropped = imread(input_path)

    return cropped

def calculate_ndvi(cropped_image):
    image = Image.fromarray(cropped_image.astype(np.uint8))
    image_np = np.array(image)

    # Extract the red, green, and NIR channels
    red_channel = np.copy(image_np[:, :, 0])
    green_channel = np.copy(image_np[:, :, 1])
    nir_channel = np.copy(image_np[:, :, 2])

    red_channel = red_channel.astype(np.float32)
    green_channel = green_channel.astype(np.float32)
    nir_channel = nir_channel.astype(np.float32)

    red_channel[red_channel < 170] = 0
    green_channel[green_channel < 170] = 0
    nir_channel[nir_channel < 170] = 0

    # Calculate NDVI with error handling for division by zero
    ndvi = (nir_channel - red_channel) / (nir_channel + red_channel)
    ndvi[np.isnan(ndvi)] = 0

    n_ndvi = np.count_nonzero(ndvi)
    x_ndvi = np.sum(ndvi)
    avg_ndvi = x_ndvi / n_ndvi if n_ndvi != 0 else 0

    if avg_ndvi <= 0.10:
        label = "Dead Plants"
    elif avg_ndvi <= 0.30:
        label = "Unhealthy Plants"
    elif avg_ndvi <= 0.60:
        label = "Moderately Healthy Plants"
    else:
        label = "Very Healthy Plants"

    return label, ndvi, avg_ndvi

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict_vegetation_class', methods=['POST'])
def predict_vegetation_class():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        try:
            # Save the uploaded file temporarily
            input_path = os.path.join('uploads', filename)
            file.save(input_path)
            
            # Perform segmentation
            cropped_image = segmentation(data_best_model, input_path)
            
            # Calculate NDVI and classify
            class_label, ndvi, avg_ndvi = calculate_ndvi(cropped_image)

        except Exception as e:
            return jsonify({"error": f"Error processing image: {e}"}), 500

        return jsonify({"class_label": class_label}), 200
    else:
        return jsonify({"error": "File type not allowed"}), 400

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
