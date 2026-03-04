from ultralytics import YOLO
import cv2

model = YOLO("models/yolov8_waste.pt")

img_path = "test.jpg"  # image in project root
results = model(img_path, imgsz=640, conf=0.6)

annotated = results[0].plot()
cv2.imshow("Waste Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()