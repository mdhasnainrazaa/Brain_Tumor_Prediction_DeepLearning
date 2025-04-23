import streamlit as st
import numpy as np
import pickle
from PIL import Image
import cv2

# Load the model
with open('svm_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Preprocessing function
def preprocess_image(image):
    image = image.resize((64, 64))  # adjust to your model's input
    img_array = np.array(image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)  # if needed
    img_array = img_array.reshape(1, -1)  # flatten if model expects 1D
    return img_array

# Streamlit UI
st.title("Brain Tumor Prediction")
uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Image.', use_column_width=True)

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    
    class_names = ['No Tumor', 'Tumor']  # update if more classes
    st.success(f'Prediction: **{class_names[int(prediction[0])]}**')
