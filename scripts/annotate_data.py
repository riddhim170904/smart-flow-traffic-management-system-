# scripts/annotate_data.py
import argparse
from pathlib import Path
import cv2
from tqdm import tqdm

from yolov8_model import load_model  # same folder import

VEHICLE_IDS = {1, 2, 3, 5, 7}          # bicycle, car, motorcycle, bus, truck

def annotate_folder(img_dir: Path, out_dir: Path):
    model = load_model()
    out_dir.mkdir(parents=True, exist_ok=True)

    img_paths = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    for img_path in tqdm(img_paths, desc="Annotating"):
        img = cv2.imread(str(img_path))
        res = model(img, verbose=False)[0]

        for b in res.boxes:
            cls = int(b.cls[0])
            if cls not in VEHICLE_IDS:
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, model.names[cls], (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imwrite(str(out_dir / img_path.name), img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="dataset/test/images", help="Input image folder")
    parser.add_argument("--dst", default="annotated",         help="Output image folder")
    args = parser.parse_args()

    annotate_folder(Path(args.src), Path(args.dst))
    print(f"✅ Done! Annotations saved to {args.dst}")
