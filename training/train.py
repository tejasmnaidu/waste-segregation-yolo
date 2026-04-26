import torch
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8m.pt")
    device = 0 if torch.cuda.is_available() else "cpu"

    model.train(
        data="datasets/data.yaml",
        epochs=150,
        imgsz=640,
        batch=16,
        device=device,
        name="waste_big_v2",
        project="runs/train",
        exist_ok=True,
        optimizer="Adam",
        lr0=0.01,
        augment=True,
        cache=True,
        rect=True,
        patience=25,
        save_period=5,
        workers=8,
    )
