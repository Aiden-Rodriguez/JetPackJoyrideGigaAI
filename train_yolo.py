"""
train_yolo.py — Fine-tune YOLOv8 on your Jetpack game dataset

Prerequisites:
    pip install ultralytics

Run after collect_data.py has built the dataset/:
    python train_yolo.py

Output:
    runs/detect/jetpack/weights/best.pt   ← use this in vision.py
"""

import os
from ultralytics import YOLO

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
DATA_YAML   = "dataset/data.yaml"
BASE_MODEL  = "yolov8s.pt"
PROJECT     = "runs/detect"
RUN_NAME    = "jetpack"
EPOCHS      = 80
IMG_SIZE    = 960
BATCH_SIZE  = 8


def main():
    if not os.path.exists(DATA_YAML):
        print(f"ERROR: '{DATA_YAML}' not found.")
        print("Run collect_data.py first to generate the dataset.")
        return

    train_dir = "dataset/images/train"
    val_dir   = "dataset/images/val"
    n_train   = len(os.listdir(train_dir)) if os.path.exists(train_dir) else 0
    n_val     = len(os.listdir(val_dir))   if os.path.exists(val_dir)   else 0
    print(f"Dataset: {n_train} train / {n_val} val images")

    if n_train < 50:
        print("WARNING: Very few training images. Collect at least 200 for reliable results.")

    print(f"\nLoading base model: {BASE_MODEL}")
    model = YOLO(BASE_MODEL)

    print(f"Training for {EPOCHS} epochs at imgsz={IMG_SIZE}...\n")
    results = model.train(
        data      = DATA_YAML,
        epochs    = EPOCHS,
        imgsz     = IMG_SIZE,
        batch     = BATCH_SIZE,
        project   = PROJECT,
        name      = RUN_NAME,
        exist_ok  = True,
        patience  = 20,

        # ── Augmentation — helps player detection since it barely moves horizontally ──
        fliplr    = 0.0,            # NO horizontal flip — missiles/zappers come from the right
        flipud    = 0.0,            # NO vertical flip — gravity direction matters
        mosaic    = 0.5,            # mosaic augmentation — exposes model to varied layouts
        translate = 0.1,            # small random crop/translate
        scale     = 0.4,            # scale jitter — helps with size variation
        hsv_h     = 0.01,           # slight hue shift
        hsv_s     = 0.3,            # saturation shift
        hsv_v     = 0.3,            # brightness shift — helps if game lighting changes

        # ── Small object detection improvements ──
        overlap_mask = True,
        cache     = True,
        verbose   = True,
    )

    best_weights = f"{PROJECT}/{RUN_NAME}/weights/best.pt"
    print(f"\n✓ Training complete!")
    print(f"  Best weights: {best_weights}")
    print(f"  Use in vision.py with:  python vision.py -yolo")

    print("\nRunning validation...")
    metrics = model.val()
    print(f"  mAP50:        {metrics.box.map50:.3f}")
    print(f"  mAP50-95:     {metrics.box.map:.3f}")

    # Per-class breakdown
    print("\nPer-class AP50:")
    class_names = ["player", "zapper", "missile", "warning"]
    if hasattr(metrics.box, 'ap50') and metrics.box.ap50 is not None:
        for name, ap in zip(class_names, metrics.box.ap50):
            bar = "█" * int(ap * 20)
            print(f"  {name:<10} {ap:.3f}  {bar}")


if __name__ == "__main__":
    main()