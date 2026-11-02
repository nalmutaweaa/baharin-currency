
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("Bahraini Currency Detection")
st.write("Upload or capture an image to detect Bahraini currency.")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

option = st.radio("Choose input:", ["Upload Image", "Camera"])

if option == "Upload Image":
    image_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )
else:
    image_file = st.camera_input("Take a picture")

if image_file is not None:
    image = Image.open(image_file).convert("RGB")

    st.image(image, caption="Original Image")

    if st.button("Detect Currency"):
        results = model.predict(np.array(image))

        annotated_image = results[0].plot()

        st.image(
            annotated_image,
            caption="Detection Results",
            channels="BGR"
        )

        st.success("Detection completed!")
