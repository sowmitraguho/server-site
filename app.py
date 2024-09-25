from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import numpy as np
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def index():
    return render_template('index.html')

# Replace this with your model loading code
# Example: import your model here (e.g., using joblib, pickle, etc.)
# from your_model_library import load_model
# model = load_model('your_model_file')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get the incoming JSON data
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400
    
    image_data = data['image']  # Extract the base64 image

    # Decode the base64 image
    try:
        image_bytes = base64.b64decode(image_data)
    except Exception as e:
        return jsonify({'error': 'Invalid base64 data'}), 400

    # Convert bytes to a PIL image
    image = Image.open(io.BytesIO(image_bytes))

    # Preprocess the image for your model
    processed_image = preprocess_image(image)

    # Perform inference using your alternative model
    prediction = make_prediction(processed_image)

    # Get the predicted class and confidence from your model's output
    predicted_class, confidence = decode_prediction(prediction)

    # Mapping for tumor types
    if predicted_class == 0:
        result = 'Tumor Type: Glioma'
    elif predicted_class == 1:
        result = 'Tumor Type: Meningioma'
    elif predicted_class == 2:
        result = 'No Tumor Detected'
    else:
        result = 'Tumor Type: Pituitary'

    return jsonify({'result': result, 'confidence': float(confidence)})

def preprocess_image(image):
    image = image.resize((224, 224))  # Example size, modify as needed
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def make_prediction(processed_image):
    # Implement your prediction logic here
    # Example: return model.predict(processed_image)
    # For demonstration, let's assume it returns a random prediction
    return np.random.rand(1, 4)  # Assuming four classes for tumor types

def decode_prediction(prediction):
    # Extract the predicted class and confidence
    predicted_class = prediction.argmax(axis=1)[0]
    confidence = np.max(prediction)  # Get the confidence of the prediction
    return predicted_class, confidence

if __name__ == '__main__':
    app.run(debug=True)
