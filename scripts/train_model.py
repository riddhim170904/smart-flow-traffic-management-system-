 # scripts/train_model.py
from ultralytics import YOLO
from pathlib import Path

# --------------------------------------------------
# config – tweak these as you like
# --------------------------------------------------
DATA_YAML = Path("dataset/data.yaml")      # relative to repo root
EPOCHS     = 1
IMG_SIZE   = 640
BATCH      = 16
MODEL_ARCH = "yolov8n.yaml"                # nano backbone -> fast
OUTPUT_DIR = Path("models")                # where to stash weights
# --------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    model = YOLO(MODEL_ARCH)

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        project=str(OUTPUT_DIR),
        name="yolov8n_traffic",
        exist_ok=True,
    )

    # Copy the best.pt file up one level for easy access
    best = OUTPUT_DIR / "yolov8n_traffic" / "weights" / "best.pt"
    if best.exists():
        best.rename(OUTPUT_DIR / "yolov8_best.pt")
        print(f"✔ Model saved to {OUTPUT_DIR/'yolov8_best.pt'}")
    else:
        print("❌ Best-weights file not found – check training log.")

if __name__ == "__main__":
    main()
