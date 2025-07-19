from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import logging
import base64
from utils1 import descriptions  # Import descriptions from utils.py

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit

# Setup logging
logging.basicConfig(level=logging.INFO)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load model
try:
    model_path = os.getenv('MODEL_PATH', 'model/plant_disease_mobileNetv2.h5')
    model = load_model(model_path)
except Exception as e:
    app.logger.error(f"Failed to load model: {str(e)}")
    raise

# Ensure uploads directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Class names
class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

# Validate descriptions
if isinstance(descriptions, list) and len(descriptions) != len(class_names):
    app.logger.error("Descriptions list length does not match class_names.")
    raise ValueError("Descriptions list length does not match class_names.")
elif isinstance(descriptions, dict) and not all(c in descriptions for c in class_names):
    app.logger.error("Not all class_names have corresponding descriptions.")
    raise ValueError("Not all class_names have corresponding descriptions.")

# Serve uploaded image files (optional, kept for compatibility)
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Main route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('a.html', error="No file uploaded.")
        
        file = request.files['image']
        if file.filename == '':
            return render_template('a.html', error="No file selected.")
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            app.logger.info(f"Uploaded file: {filename}")

            try:
                # Load and preprocess image
                img = image.load_img(filepath, target_size=(224, 224))
                img_array = image.img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Convert image to base64 for display
                with open(filepath, "rb") as image_file:
                    image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
                image_data = f"data:image/{filename.rsplit('.', 1)[1].lower()};base64,{image_base64}"

                # Predict
                prediction = model.predict(img_array)
                predicted_index = np.argmax(prediction)
                predicted_class = class_names[predicted_index]
                confidence = round(100 * np.max(prediction), 2)

                # Get description
                if isinstance(descriptions, dict):
                    description = descriptions.get(predicted_class, "No treatment information available.")
                else:  # Assume list
                    description = descriptions[predicted_index] if predicted_index < len(descriptions) else "No treatment information available."

                # Clean up
                os.remove(filepath)

                return render_template(
                    'a.html',
                    prediction=predicted_class,
                    confidence=confidence,
                    image_data=image_data,  # Pass base64 image data
                    description=description
                )
            except Exception as e:
                os.remove(filepath)  # Clean up even on error
                app.logger.error(f"Error processing image: {str(e)}")
                return render_template('a.html', error=f"Error processing image: {str(e)}")
        else:
            return render_template('a.html', error="Invalid file type. Use JPG, PNG, or WebP.")
    
    return render_template('a.html', prediction=None)

application = app  # For WSGI compatibility

if __name__ == '__main__':
    app.run(debug=False)