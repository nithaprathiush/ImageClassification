import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model(
    "intel_scene_classifier.h5"
)

class_names = [
    'buildings',
    'forest',
    'glacier',
    'mountain',
    'sea',
    'street'
]

st.title("Intel Scene Classification")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg","png","jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(image)

    image = image.resize((150,150))

    image = np.array(image)

    image = np.expand_dims(image, axis=0)

    pred = model.predict(image)

    class_idx = np.argmax(pred)

    st.success(
        f"Predicted Class: {class_names[class_idx]}"
    )