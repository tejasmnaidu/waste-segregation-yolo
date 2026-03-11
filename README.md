# AI Waste Segregation System using YOLOv8

## Problem Statement

Waste segregation remains a critical challenge in modern waste management systems. Improper segregation of waste leads to contamination of recyclable materials, inefficient recycling processes, and increased environmental pollution. In many places, waste sorting is still performed manually, which is time-consuming, labor-intensive, and prone to human error.

Advancements in artificial intelligence and computer vision provide an opportunity to automate waste classification. Automated waste segregation systems can significantly improve recycling efficiency and support sustainable environmental practices.

This project presents an AI-based waste segregation system that uses the YOLOv8 object detection model to identify and classify waste materials from images.

---

## Project Overview

The AI Waste Segregation System is built using a deep learning object detection model capable of identifying waste objects within images. The system analyzes an input image and detects waste items such as plastic bottles by drawing bounding boxes around them and predicting their category along with a confidence score.

The trained model is integrated into a simple web interface built with Streamlit. Users can upload images through the interface and receive immediate predictions showing detected waste objects.

The goal of this project is to demonstrate how computer vision models can be used to assist automated waste sorting systems that may be deployed in recycling plants, smart waste bins, or smart city infrastructure.

---

## Key Features

- Object detection using YOLOv8 architecture
- Detection and classification of waste objects from images
- Visualization of bounding boxes with confidence scores
- GPU accelerated training using PyTorch
- Streamlit-based web interface for inference
- Modular project structure for training and deployment
- Easily extensible to additional waste categories

---

## System Architecture

User Image Input  
→ Streamlit Web Interface  
→ YOLOv8 Detection Model  
→ Waste Object Detection  
→ Bounding Box Visualization  
→ Prediction Output with Confidence Score

---

## Technology Stack

**Language**

Python

**Libraries and Frameworks**

- YOLOv8 (Ultralytics)
- PyTorch
- OpenCV
- Streamlit

**Hardware**

GPU acceleration for model training

---

## Project Structure

```
waste-segregation-yolo

datasets/              # Training dataset
   train/
   valid/
   test/

inference/             # Detection scripts
   image_detect.py
   video_detect.py
   webcam_detect.py

training/              # Training pipeline
   train.py

models/                # Trained model weights
   yolov8_waste.pt

runs/                  # YOLO training outputs

app.py                 # Streamlit web application
data.yaml              # Dataset configuration
requirements.txt       # Project dependencies
README.md
```

---

## Model Training

The model is trained using the YOLOv8 architecture optimized for real-time object detection.

**Training Configuration**

- Model: YOLOv8n
- Image Size: 640
- Epochs: 50
- Batch Size: 16
- Framework: PyTorch
- Hardware: GPU acceleration enabled

**Training Command**

```
python training/train.py
```

After training completes, the best model weights are stored in:

```
runs/detect/.../weights/best.pt
```

---

## Running Inference

**Image Detection**

```
python inference/image_detect.py
```

**Video Detection**

```
python inference/video_detect.py
```

**Webcam Detection**

```
python inference/webcam_detect.py
```

---

## Running the Web Application

To launch the Streamlit application:

```
streamlit run app.py
```

Once the application starts, open the provided local URL in your browser. Users can upload an image and the system will display detected waste objects with bounding boxes and confidence scores.

---

## Example Output

The system detects waste objects and labels them with bounding boxes.

Example prediction:

Plastic bottle detected with confidence score 0.88

---

## Applications

- Smart waste bins  
- Automated recycling plants  
- Waste classification systems  
- Environmental monitoring systems  
- Smart city infrastructure  

---

## Future Improvements

- Expand the dataset with additional waste categories such as metal, paper, glass, and organic waste
- Train larger YOLOv8 models for improved detection accuracy
- Implement real-time detection using live camera streams
- Deploy the application on cloud platforms for public access
- Integrate the system with IoT-based smart waste bins

---

## Author

Tejas M Naidu

This project demonstrates the application of deep learning and computer vision techniques to address real-world environmental challenges through automated waste detection and classification.

