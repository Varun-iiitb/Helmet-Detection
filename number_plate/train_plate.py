"""
train_plate.py
Train YOLO11m on the number-plate dataset.
Run from ~/number_plate/ so the relative paths in data.yaml resolve correctly.
"""

import shutil
from pathlib import Path
from ultralytics import YOLO

DATA_YAML   = str(Path(__file__).parent / "data.yaml")
RUN_NAME    = "yolo11_plate_v2"
BEST_WEIGHTS = f"plate_detector/{RUN_NAME}/weights/best.pt"


def train():
    model = YOLO("yolo11m.pt")

    model.train(
        data=DATA_YAML,

        epochs=120,
        imgsz=1280,
        # Why batch=16: server has 16 GB GPU; original batch=4 was for Kaggle
        # free tier (16 GB shared). Batch 16 is 4x faster with same memory.
        batch=16,

        device=0,
        workers=4,
        cache=True,
        amp=True,

        project="plate_detector",
        name=RUN_NAME,
        exist_ok=True,

        pretrained=True,
        patience=30,

        multi_scale=True,
        close_mosaic=15,

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.15,
        scale=0.5,
        shear=2.0,
        fliplr=0.5,
        mosaic=1.0,
        copy_paste=0.1,

        save=True,
    )


def save_to_models():
    dst = Path(__file__).parent.parent / "models" / "plate_model.pt"
    dst.parent.mkdir(exist_ok=True)
    shutil.copy(BEST_WEIGHTS, dst)
    print(f"\nBest weights saved to {dst}")


if __name__ == "__main__":
    train()
    save_to_models()
