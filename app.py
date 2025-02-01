from flask import Flask, request, jsonify,render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from PIL import Image
import io

app = Flask(__name__)

# Load models
vgg19_model = load_model('feature_extractor_vgg19.keras')  # Ensure this path is correct
svm_model = joblib.load('svm_model.pkl')
rfecv_selector = joblib.load('rfecv_selector.pkl')

def preprocess_image(image):
    img = Image.open(io.BytesIO(image))
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.vgg19.preprocess_input(img_array)
    return img_array

def predict_class(file):
    # Preprocess the image
    img_array = preprocess_image(file.read())
    
    # Extract features using VGG19
    features = vgg19_model.predict(img_array)
    
    # Select features using RFECV
    selected_features = rfecv_selector.transform(features)
    
    # Predict class using SVM
    prediction = svm_model.predict(selected_features)
    
    # For demonstration, let's assume the confidence is a fixed value
    confidence = 0.95  # Replace with actual confidence calculation if available
    
    return prediction[0], confidence

@app.route('/')
def home():
    return render_template('index.html')  # Ensure 'index.html' is in the 'templates' directory

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    class_name, confidence = predict_class(file)
    return jsonify({'class': class_name, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)