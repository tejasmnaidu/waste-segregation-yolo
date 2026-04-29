import os
import torch
from pathlib import Path
from ultralytics import YOLO
from prepare_broad_dataset import prepare_broad_dataset


def get_device():
    return 0 if torch.cuda.is_available() else "cpu"


def get_gpu_memory():
    if not torch.cuda.is_available():
        return 0
    try:
        return torch.cuda.get_device_properties(0).total_memory
    except Exception:
        return 0


def select_backbone(memory_bytes: int):
    if memory_bytes >= 24 * 1024**3:
        return "yolov8x.pt"
    if memory_bytes >= 12 * 1024**3:
        return "yolov8l.pt"
    return "yolov8m.pt"


def select_imgsz(memory_bytes: int):
    if memory_bytes >= 24 * 1024**3:
        return 1280
    if memory_bytes >= 16 * 1024**3:
        return 1024
    return 896


def select_batch(memory_bytes: int, imgsz: int):
    if memory_bytes >= 24 * 1024**3:
        return 12
    if memory_bytes >= 16 * 1024**3:
        return 8
    if memory_bytes >= 8 * 1024**3:
        return 4
    return 2


def get_data_config(root: Path):
    broad_config = root / "datasets" / "broad" / "data.yaml"
    if broad_config.exists():
        return str(broad_config)
    prepare_broad_dataset(source_yaml=root / "datasets" / "data.yaml", output_root=root / "datasets" / "broad")
    return str(broad_config)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    device = get_device()
    gpu_memory = get_gpu_memory()
    imgsz = select_imgsz(gpu_memory)
    batch = select_batch(gpu_memory, imgsz)
    backbone = select_backbone(gpu_memory)
    num_workers = min(8, os.cpu_count() or 4)

    data_config = get_data_config(root)
    print(f"Training on dataset: {data_config}")
    print(f"Backbone={backbone}, device={device}, imgsz={imgsz}, batch={batch}, workers={num_workers}")

    model = YOLO(backbone)

    model.train(
        data=data_config,
        epochs=25,
        imgsz=imgsz,
        batch=batch,
        device=device,
        name="waste_extreme_v4",
        project="runs/train",
        exist_ok=True,
        optimizer="Adam",
        lr0=0.002,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=10,
        augment=True,
        cache="ram",
        rect=True,
        patience=100,
        save_period=1,
        workers=num_workers,
        resume=True,
        verbose=True,
    )
