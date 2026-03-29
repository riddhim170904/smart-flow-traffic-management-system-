# scripts/yolov8_model.py
from ultralytics import YOLO
from pathlib import Path

DEFAULT_WEIGHTS = Path("models/yolov8_best.pt")  # produced by train_model.py
FALLBACK_WEIGHTS = Path("yolov8n.pt")            # shipped Nano weights

def load_model(weights: Path | None = None) -> YOLO:
    """
    Returns a YOLOv8 model, preferring your trained weights.
    """
    weights = weights or DEFAULT_WEIGHTS if DEFAULT_WEIGHTS.exists() else FALLBACK_WEIGHTS
    print(f"🔹 Loading weights: {weights}")
    return YOLO(str(weights))

# quick sanity-check 👉 python scripts/yolov8_model.py
if __name__ == "__main__":
    model = load_model()
    print(model.model)
