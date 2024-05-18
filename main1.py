import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Load the trained model
model = load_model('finals_model.h5')

# Define the class labels
class_labels = ['Rain', 'Shine', 'Cloudy', 'Sunrise']

# Function to predict the class of an image
def predict_image(img_path, model):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    predictions = model.predict(img_array, verbose=0)  # Added verbose=0 for cleaner output
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_class, confidence

# Apply weather-themed gradient background color and text color using CSS
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(to bottom right, #00c6ff, #0072ff);
        color: black;
    }
    h1, .stMarkdown, .stFileUploader {
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app
st.sidebar.title("Emtech Final Project")
with st.sidebar:
    st.subheader("Developers:")
    st.write("James Solomon Diocena")
    st.write("Mark John Villanueva")

st.title("Weather Image Classification")
st.write("Upload an image to classify the weather condition.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Save the uploaded file
    with open("uploaded_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classify As")

    # Add a spinner while the model is making a prediction
    with st.spinner('Model is working...'):
        label, confidence = predict_image("uploaded_image.jpg", model)
    
    st.write(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.2f}")

# To run the Streamlit app, use the following command:
# streamlit run app.py
