import argparse
import os
from collections import Counter
from ultralytics import YOLO
import cv2

MODEL_CANDIDATES = ["models/best.pt", "models/yolov8_waste.pt", "yolov8n.pt"]
DEFAULT_MODEL = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), "yolov8n.pt")

friendly_label_map = {
    "Aerosols": "Aerosol can",
    "Aluminum can": "Metal can",
    "Aluminum caps": "Metal cap",
    "Cardboard": "Cardboard",
    "Cellulose": "Paper",
    "Ceramic": "Ceramic",
    "Combined plastic": "Mixed plastic",
    "Container for household chemicals": "Chemical container",
    "Disposable tableware": "Disposable tableware",
    "Electronics": "Electronics",
    "Foil": "Aluminum foil",
    "Furniture": "Furniture",
    "Glass bottle": "Glass bottle",
    "Iron utensils": "Metal utensil",
    "Liquid": "Liquid container",
    "Metal shavings": "Metal scrap",
    "Milk bottle": "Plastic milk bottle",
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

def resolve_model_path(path: str) -> str:
    if os.path.exists(path):
        return path
    return DEFAULT_MODEL

def summarize_labels(labels):
    counts = Counter(labels)
    return ", ".join(f"{label} x{count}" for label, count in counts.items())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run live waste detection from webcam")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to YOLO model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IOU threshold")
    parser.add_argument("--source", default=0, help="Webcam device index or video path")
    parser.add_argument("--broad", action="store_true", help="Show broad categories instead of detailed class names")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model)
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    capture_source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(capture_source)
    if not cap.isOpened():
        print("Cannot open webcam or video source")
        exit(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        results = model(frame, conf=args.conf, iou=args.iou)
        annotated_frame = results[0].plot()

        detected = []
        if len(results) > 0 and results[0].boxes is not None:
            for cls in results[0].boxes.cls:
                class_name = model.names[int(cls)]
                if args.broad:
                    detected.append(broad_category_map.get(class_name, class_name))
                else:
                    detected.append(friendly_label_map.get(class_name, class_name))

        if detected:
            label_summary = summarize_labels(detected)
            cv2.putText(
                annotated_frame,
                f"Detected: {label_summary}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated_frame,
                f"Count: {len(detected)}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Waste Segregation - Webcam", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
