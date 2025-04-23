import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained model
model = load_model("your_model.h5")  # update this path if needed

# Define image preprocessing
def preprocess_image(img):
    img = img.resize((224, 224))  # adjust to match your model input
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Streamlit UI
st.title("Brain Tumor Prediction")
uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI.', use_column_width=True)
    st.write("Predicting...")
    
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    
    class_names = ["No Tumor", "Tumor"]  # update if more classes
    st.write(f"Prediction: **{class_names[np.argmax(prediction)]}**")
