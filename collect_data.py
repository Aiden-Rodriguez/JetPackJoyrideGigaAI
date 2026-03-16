# collect_data.py -> Generates training data for YOLO from the running game.
import os
import json
import time
import random
import shutil
import numpy as np
import cv2

SHARED_FRAME_PATH = "shared_frame.npy"
SHARED_STATE_PATH = "shared_state.json"   # ground-truth positions from game.py
DATASET_DIR       = "dataset"

TARGET_FRAMES = 500
VAL_SPLIT     = 0.15
MIN_FRAME_GAP = 0.1
MIN_OBJECTS   = 1 

WIDTH, HEIGHT = 960, 540

CLASS_NAMES = ["player", "zapper", "missile", "warning"]
CLASS_ID    = {name: i for i, name in enumerate(CLASS_NAMES)}

# Frames where player overlaps an obstacle above this IoU are skipped
MAX_PLAYER_OBSTACLE_IOU = 0.05

# Skip frames where player center is within this many pixels of any obstacle bbox
PROXIMITY_MARGIN = 20


# Overlap checks
def iou(a: dict, b: dict) -> float:
    """Intersection-over-Union between two x/y/w/h dicts."""
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = ax1 + a["w"], ay1 + a["h"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = bx1 + b["w"], by1 + b["h"]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0

    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def player_too_close(player: dict, obstacles: list) -> bool:
    """Returns True if the PLAYER BBOX intersects any obstacle bbox"""
    px1 = player["x"] - PROXIMITY_MARGIN
    py1 = player["y"] - PROXIMITY_MARGIN
    px2 = player["x"] + player["w"] + PROXIMITY_MARGIN
    py2 = player["y"] + player["h"] + PROXIMITY_MARGIN

    for obs in obstacles:
        ox1 = obs["x"] - PROXIMITY_MARGIN
        oy1 = obs["y"] - PROXIMITY_MARGIN
        ox2 = obs["x"] + obs["w"] + PROXIMITY_MARGIN
        oy2 = obs["y"] + obs["h"] + PROXIMITY_MARGIN
        if px1 < ox2 and px2 > ox1 and py1 < oy2 and py2 > oy1:
            return True
    return False


def frame_is_clean(objects: list) -> bool:
    """
    Returns True only if this frame is safe to use as training data:
      - Player has no significant IoU overlap with any obstacle
      - Player center is not dangerously close to any obstacle
    Filters collision frames, near-death frames, and any post-death frames
    that slipped through.
    """
    player_objs   = [o for o in objects if o["label"] == "player"]
    obstacle_objs = [o for o in objects if o["label"] != "player"]

    if not player_objs:
        return True  # no player

    player = player_objs[0]

    for obs in obstacle_objs:
        if iou(player, obs) > MAX_PLAYER_OBSTACLE_IOU:
            return False  # direct overlap

    if player_too_close(player, obstacle_objs):
        return False  # too close

    return True


# YOLO label format: class_id cx cy w h  (normalised 0–1)
def to_yolo_line(obj: dict) -> str | None:
    """Convert one game object dict to a YOLO label line."""
    label = obj["label"]
    if label not in CLASS_ID:
        return None

    x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]

    # Clamp
    x = max(0, x)
    y = max(0, y)
    w = min(w, WIDTH  - x)
    h = min(h, HEIGHT - y)

    if w <= 0 or h <= 0:
        return None

    cx = (x + w / 2) / WIDTH
    cy = (y + h / 2) / HEIGHT
    nw = w / WIDTH
    nh = h / HEIGHT

    return f"{CLASS_ID[label]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"

def load_frame(path: str) -> np.ndarray | None:
    try:
        rgb = np.load(path)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def load_state(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def get_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except FileNotFoundError:
        return 0.0


# Dataset setup
def setup_dirs():
    for split in ("train", "val"):
        os.makedirs(f"{DATASET_DIR}/images/{split}", exist_ok=True)
        os.makedirs(f"{DATASET_DIR}/labels/{split}",  exist_ok=True)


def write_yaml():
    abs_path = os.path.abspath(DATASET_DIR)
    content = f"""path: {abs_path}
train: images/train
val:   images/val

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    with open(f"{DATASET_DIR}/data.yaml", "w") as f:
        f.write(content)
    print(f"Wrote {DATASET_DIR}/data.yaml")


def main():
    setup_dirs()
    write_yaml()

    print(f"Collecting {TARGET_FRAMES} labeled frames...")
    print(f"  Reading frames from : {SHARED_FRAME_PATH}")
    print(f"  Reading state from  : {SHARED_STATE_PATH}  ← exact game positions")
    print("Play the game normally. Try to get a mix of zappers, missiles, and warnings.")
    print("Press Ctrl+C to stop early.\n")

    collected    = 0
    last_mtime   = 0.0
    last_capture = 0.0
    saved_stems  = []
    class_counts = {}

    try:
        while collected < TARGET_FRAMES:

            # Wait until game.py writes a new state file
            mtime = get_mtime(SHARED_STATE_PATH)
            now   = time.time()

            if mtime == last_mtime or (now - last_capture) < MIN_FRAME_GAP:
                time.sleep(0.01)
                continue

            if mtime == 0.0:
                print("  Waiting for game.py to start...")
                time.sleep(0.5)
                continue

            last_mtime   = mtime
            last_capture = now

            # Load both files
            frame = load_frame(SHARED_FRAME_PATH)
            state = load_state(SHARED_STATE_PATH)

            if frame is None or state is None:
                continue

            objects = state.get("objects", [])

            if not frame_is_clean(objects):
                continue

            lines = [to_yolo_line(obj) for obj in objects]
            lines = [l for l in lines if l is not None]

            if len(lines) < MIN_OBJECTS:
                continue

            # Save image + label
            stem       = f"frame_{collected:05d}"
            img_path   = f"{DATASET_DIR}/images/train/{stem}.jpg"
            label_path = f"{DATASET_DIR}/labels/train/{stem}.txt"

            cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(label_path, "w") as f:
                f.write("\n".join(lines))

            saved_stems.append(stem)
            collected += 1
            for obj in objects:
                class_counts[obj["label"]] = class_counts.get(obj["label"], 0) + 1

            if collected % 50 == 0 or collected == 1:
                counts_str = "  ".join(f"{k}={v}" for k, v in sorted(class_counts.items()))
                print(f"  [{collected:>4}/{TARGET_FRAMES}]  {counts_str}")

    except KeyboardInterrupt:
        print("\nStopped early.")

    if not saved_stems:
        print("No frames collected — make sure game.py is running.")
        return

    n_val      = max(1, int(len(saved_stems) * VAL_SPLIT))
    val_stems  = set(random.sample(saved_stems, n_val))

    for stem in val_stems:
        for kind in ("images", "labels"):
            ext = "jpg" if kind == "images" else "txt"
            src = f"{DATASET_DIR}/{kind}/train/{stem}.{ext}"
            dst = f"{DATASET_DIR}/{kind}/val/{stem}.{ext}"
            if os.path.exists(src):
                shutil.move(src, dst)

    n_train = len(saved_stems) - n_val
    print(f"\nDone!  {n_train} train  /  {n_val} val  frames in '{DATASET_DIR}/'")
    print(f"\nClass label counts:")
    for cls in ["player", "zapper", "missile", "warning"]:
        count = class_counts.get(cls, 0)
        bar   = "█" * min(40, count // 5)
        warn  = "  ← LOW, collect more!" if count < 80 else ""
        print(f"  {cls:<10} {count:>4}  {bar}{warn}")
    print("\nNext step:  python train_yolo.py")


if __name__ == "__main__":
    main()