from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Mendapatkan path folder model
model_folder = os.path.join(os.getcwd(), "model")
model_path = os.path.join(model_folder, "model.h5")

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Define function to preprocess input image
def preprocess_image(image):
    # Resize image to match model's expected sizing
    image = image.resize((224, 224))
    # Convert PIL image to numpy array
    image_array = np.array(image)
    # Normalize pixel values to [0, 1]
    image_array = image_array / 255.0
    # Expand dimensions to match model's expected input shape
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get the image file from the POST request
        file = request.files["file"]
        # Read the image file
        image = Image.open(file)
        # Preprocess the image
        processed_image = preprocess_image(image)
        # Perform prediction
        predictions = model.predict(processed_image)
        # Decode predictions (if necessary)
        # (e.g., convert from one-hot encoded labels to class labels)
        # Format predictions as needed
        # (e.g., convert to JSON)
        return jsonify(predictions)

if __name__ == "__main__":
    app.run(debug=True)
