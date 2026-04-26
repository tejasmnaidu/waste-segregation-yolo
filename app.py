import os
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import torch

st.set_page_config(page_title="Waste Segregation Detector", page_icon="Waste", layout="wide")

st.title("Waste Segregation - Image + Webcam Detector")
st.write("Use the webcam capture or upload an image to detect waste objects with the selected model.")

model_files = [path for path in ["models/best.pt", "models/yolov8_waste.pt", "yolov8n.pt"] if os.path.exists(path)]
if not model_files:
    st.error("No model weights found. Place a YOLO model file in the models/ folder.")
    st.stop()

selected_model = st.sidebar.selectbox("Select detection model", model_files, index=0)
confidence = st.sidebar.slider("Confidence threshold", min_value=0.1, max_value=0.8, value=0.25, step=0.05)
iou_thresh = st.sidebar.slider("NMS IOU threshold", min_value=0.1, max_value=0.7, value=0.45, step=0.05)
augment_inference = st.sidebar.checkbox("Use augmented inference for better accuracy", value=True)
max_det = st.sidebar.slider("Maximum detections", min_value=20, max_value=200, value=100, step=10)
use_broad_labels = st.sidebar.checkbox("Map detailed classes into broad categories (Plastic, Metal, Paper, Organic)", value=False)

device = 0 if torch.cuda.is_available() else "cpu"

@st.cache_resource(show_spinner=False)
def load_model(path):
    return YOLO(path)

model = load_model(selected_model)

from collections import Counter

friendly_label_map = {
    "Aerosols": "Aerosol can",
    "Aluminum can": "Metal can",
    "Aluminum caps": "Metal cap",
    "Cardboard": "Cardboard",
    "Cellulose": "Paper",
    "Ceramic": "Ceramic",
    "Combined plastic": "Mixed plastic",
    "Container for household chemicals": "Chemical bottle",
    "Disposable tableware": "Disposable plate/cup",
    "Electronics": "Electronics",
    "Foil": "Aluminum foil",
    "Furniture": "Furniture",
    "Glass bottle": "Glass bottle",
    "Iron utensils": "Metal utensil",
    "Liquid": "Liquid container",
    "Metal shavings": "Metal scrap",
    "Milk bottle": "Plastic bottle",
    "Organic": "Organic waste",
    "Paper": "Paper",
    "Paper bag": "Paper bag",
    "Paper cups": "Paper cup",
    "Paper shavings": "Paper scrap",
    "Papier mache": "Paper craft",
    "Plastic bag": "Plastic bag",
    "Plastic bottle": "Plastic bottle",
    "Plastic can": "Plastic can",
    "Plastic canister": "Plastic container",
    "Plastic caps": "Plastic cap",
    "Plastic cup": "Plastic cup",
    "Plastic shaker": "Plastic shaker",
    "Plastic shavings": "Plastic scrap",
    "Plastic toys": "Plastic toy",
    "Postal packaging": "Postal packaging",
    "Printing industry": "Printed material",
    "Scrap metal": "Scrap metal",
    "Stretch film": "Plastic film",
    "Tetra pack": "Tetra Pak",
    "Textile": "Textile",
    "Tin": "Tin can",
    "Unknown plastic": "Unknown plastic",
    "Wood": "Wood",
    "Zip plastic bag": "Plastic zip bag",
}

broad_category_map = {
    "Aerosols": "Other",
    "Aluminum can": "Metal",
    "Aluminum caps": "Metal",
    "Cardboard": "Paper",
    "Cellulose": "Paper",
    "Ceramic": "Other",
    "Combined plastic": "Plastic",
    "Container for household chemicals": "Other",
    "Disposable tableware": "Other",
    "Electronics": "Other",
    "Foil": "Metal",
    "Furniture": "Other",
    "Glass bottle": "Glass",
    "Iron utensils": "Metal",
    "Liquid": "Other",
    "Metal shavings": "Metal",
    "Milk bottle": "Plastic",
    "Organic": "Organic",
    "Paper": "Paper",
    "Paper bag": "Paper",
    "Paper cups": "Paper",
    "Paper shavings": "Paper",
    "Papier mache": "Paper",
    "Plastic bag": "Plastic",
    "Plastic bottle": "Plastic",
    "Plastic can": "Plastic",
    "Plastic canister": "Plastic",
    "Plastic caps": "Plastic",
    "Plastic cup": "Plastic",
    "Plastic shaker": "Plastic",
    "Plastic shavings": "Plastic",
    "Plastic toys": "Plastic",
    "Postal packaging": "Other",
    "Printing industry": "Other",
    "Scrap metal": "Metal",
    "Stretch film": "Plastic",
    "Tetra pack": "Other",
    "Textile": "Other",
    "Tin": "Metal",
    "Unknown plastic": "Plastic",
    "Wood": "Other",
    "Zip plastic bag": "Plastic",
}

def summarize_labels(labels):
    if not labels:
        return "None"
    counts = Counter(labels)
    return ", ".join(f"{label} x{count}" for label, count in counts.items())


def detect_image(image: Image.Image):
    img_array = np.array(image.convert("RGB"))
    results = model(
        img_array,
        conf=confidence,
        iou=iou_thresh,
        device=device,
        augment=augment_inference,
        max_det=max_det,
    )

    if len(results) == 0 or results[0].boxes is None:
        return image, [], []

    detected_labels = []
    for r in results:
        for idx, cls in enumerate(r.boxes.cls):
            class_name = model.names[int(cls)]
            if use_broad_labels:
                display_name = broad_category_map.get(class_name, class_name)
            else:
                display_name = friendly_label_map.get(class_name, class_name)
            r.names[int(cls)] = display_name
            detected_labels.append(display_name)

    annotated = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return annotated_rgb, results[0].boxes, detected_labels

col1, col2 = st.columns([1, 1])
with col1:
    st.header("Webcam Capture")
    camera_file = st.camera_input("Use your webcam to capture a waste image")
    if camera_file is not None:
        image = Image.open(camera_file)
        st.image(image, caption="Webcam capture", use_container_width=True)
        annotated, boxes, detected_labels = detect_image(image)
        st.image(annotated, caption="Detected waste objects", use_container_width=True)
        if boxes is not None and len(boxes) > 0:
            st.success(f"Detected {len(boxes)} object(s) in webcam image")
            st.markdown(f"**Detected types:** {summarize_labels(detected_labels)}")
        else:
            st.warning("No waste objects detected. Try a different angle, better lighting, or update the model.")

with col2:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"] )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        annotated, boxes, detected_labels = detect_image(image)
        st.image(annotated, caption="Detected Output", use_container_width=True)
        if boxes is not None and len(boxes) > 0:
            st.success(f"Detected {len(boxes)} object(s) in uploaded image")
            st.markdown(f"**Detected types:** {summarize_labels(detected_labels)}")
        else:
            st.warning("No waste objects detected. Try a different image or update the model.")

st.markdown("---")
st.write("**Model configuration:**")
st.write(f"- Selected weights: `{selected_model}`")
st.write(f"- Device: `{device}`")
st.write(f"- Confidence threshold: `{confidence}`")
st.write(f"- IOU threshold: `{iou_thresh}`")
