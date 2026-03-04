from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")

    model.train(
        data="datasets/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        name="waste_big_v1"
    )