import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np

# Define the class names
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# Load your trained model
model = tf.keras.models.load_model('model.h5')

st.title('CNN Model Prediction')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image to fit your model input requirements
    img = image.resize((160, 160))  # Replace with the input size used during training
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize the image data
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match the model's input shape

    # Make a prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_names[predicted_class_index]

    st.write(f"Predicted Class: {predicted_class_name}")
 