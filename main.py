import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load the pre-trained model (MobileNetV2 in this case)
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Function to preprocess the uploaded image
def preprocess_image(image):
    size = (224, 224)  # MobileNetV2 expects 224x224 images
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image = np.asarray(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image

# Function to decode predictions
def decode_predictions(predictions):
    return tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]

# Streamlit app
st.title("Image Classifier")
st.write("Upload an image to classify it")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    decoded_predictions = decode_predictions(predictions)
    
    st.write("Top Predictions:")
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        st.write(f"{i+1}. {label}: {score*100:.2f}%")

