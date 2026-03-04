import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Load model
model = YOLO("models/best.pt")

st.title("♻️ Waste Segregation - Image Detector")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_array = np.array(image)

    results = model(img_array, conf=0.7, iou=0.5)

    annotated = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    st.image(annotated_rgb, caption="Detected Output", use_container_width=True)