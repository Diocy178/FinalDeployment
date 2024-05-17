import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import shutil

# Function to load the Keras model
def load_keras_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to predict the class of an image
def predict_image(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions)
        class_label = class_labels[class_index]
        confidence = predictions[0][class_index]
        return class_label, confidence
    except Exception as e:
        st.error(f"Error predicting image: {e}")
        return None, None

# Streamlit app
st.title("Weather Image Classification")

# Load the model
model_path = 'finals_model.h5'
model = load_keras_model(model_path)

if model is not None:
    # Define the class labels
    class_labels = ['Rain', 'Sunrise', 'Cloudy', 'Shine']

    # Upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        # Save the uploaded image
        img_path = os.path.join("uploaded_image.jpg")
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display the uploaded image
        st.image(img_path, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        
        # Predict the class of the image
        label, confidence = predict_image(img_path, model)
        
        if label is not None and confidence is not None:
            # Display the prediction
            st.write(f"Prediction: {label}")
            st.write(f"Confidence: {confidence:.2f}")

            # Move the image to the corresponding folder
            destination_folder = os.path.join("data", label)
            os.makedirs(destination_folder, exist_ok=True)
            shutil.move(img_path, os.path.join(destination_folder, os.path.basename(img_path)))
            st.write(f"Image moved to {label} folder.")

            # Get the current working directory
            current_directory = os.getcwd()

            # Display the current working directory
            st.write(f"Current Directory: {current_directory}")

            # Display the directory where the images are moved
            st.write(f"Images moved to: {destination_folder}")
