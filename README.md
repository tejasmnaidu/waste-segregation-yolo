# AI Waste Segregation System using YOLOv8

## Overview

This project implements an AI-powered waste segregation system using the YOLOv8 object detection model. The system can detect waste objects from images and classify them into categories such as plastic bottles. The goal of this project is to demonstrate how computer vision can assist in automated waste management and recycling processes.

The model is trained on a labeled dataset and deployed through a simple web interface built with Streamlit. Users can upload an image and the system will detect waste objects, draw bounding boxes around them, and display the predicted class with confidence scores.

---

## Features

* Object detection using YOLOv8
* Detection of waste objects from images
* Bounding box visualization with confidence score
* Simple deployment using Streamlit
* GPU accelerated training using PyTorch
* Modular project structure for training and inference

---

## Tech Stack

* Python
* YOLOv8 (Ultralytics)
* PyTorch
* OpenCV
* Streamlit

---

## Project Structure

```
waste-segregation-yolo/
│
├── datasets/              # Training dataset
│   ├── train/
│   ├── valid/
│   └── test/
│
├── inference/             # Scripts for detection
│   ├── image_detect.py
│   ├── video_detect.py
│   └── webcam_detect.py
│
├── runs/                  # YOLO training outputs
│
├── training/
│   └── train.py
│
├── models/
│   └── yolov8_waste.pt
│
├── app.py                 # Streamlit deployment
├── data.yaml              # Dataset configuration
├── requirements.txt       # Dependencies
└── README.md
```

---

## Model Training

The model was trained using YOLOv8 with the following configuration:

* Model: YOLOv8n
* Image size: 640
* Epochs: 50
* Batch size: 16
* GPU acceleration enabled

Training script:

```
python training/train.py
```

After training, the best model weights are saved in:

```
runs/detect/.../weights/best.pt
```

---

## Running the Application

Activate the virtual environment and start the Streamlit app.

```
streamlit run app.py
```

The application will launch in the browser.

Upload an image to see the detected waste objects with bounding boxes and confidence scores.

---

## Example Output

The system detects waste objects and labels them with a bounding box and confidence score.

Example:

Plastic bottle detected with confidence score 0.88.

---

## Applications

* Smart waste bins
* Automated recycling plants
* Waste classification systems
* Environmental monitoring systems

---

## Future Improvements

* Add more waste categories
* Improve dataset size for better accuracy
* Implement real-time webcam detection
* Deploy as a cloud-based service

---

