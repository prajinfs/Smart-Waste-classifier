import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# Page settings
st.set_page_config(
    page_title="Smart Waste Classifier",
    page_icon="♻️",
    layout="centered"
)

st.title("♻️ Smart Waste Classifier")
st.write("Upload an image to identify its waste category.")

MODEL_PATH = Path("waste_classifier.keras")
CLASS_PATH = Path("class_names.txt")

# Check required files
if not MODEL_PATH.exists() or not CLASS_PATH.exists():
    st.error(
        "Model files are missing. Please run "
        "'python train_model.py' first."
    )
    st.stop()

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Load category names
with open(CLASS_PATH, "r", encoding="utf-8") as file:
    class_names = file.read().splitlines()

# Upload an image
uploaded_file = st.file_uploader(
    "Upload a waste image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded image",
        width=300
    )

    if st.button("Classify Waste"):
        # Resize image to match the model input
        resized_image = image.resize((160, 160))

        # Keep pixels in the 0-255 range.
        # MobileNetV2 preprocessing is already inside the model.
        image_array = np.array(
            resized_image,
            dtype=np.float32
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # Predict the waste category
        predictions = model.predict(
            image_array,
            verbose=0
        )[0]

        predicted_index = int(np.argmax(predictions))
        predicted_class = class_names[predicted_index]
        confidence = float(predictions[predicted_index]) * 100

        st.success(
            f"Predicted category: {predicted_class.title()}"
        )

        st.write(f"Confidence: {confidence:.2f}%")

        # Show all category probabilities
        st.subheader("Category probabilities")

        for i, name in enumerate(class_names):
            st.write(f"{name.title()}: {predictions[i] * 100:.2f}%")
            st.progress(float(predictions[i]))