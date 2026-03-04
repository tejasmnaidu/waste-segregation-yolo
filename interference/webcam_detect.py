from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("models/yolov8_waste.pt")

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Run YOLO on the frame
    results = model(frame)

    # Draw boxes
    annotated_frame = results[0].plot()

    cv2.imshow("Waste Segregation - Webcam", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()