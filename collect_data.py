"""
collect_data.py — Auto-generate YOLO training data from game.py's internal state

Instead of inferring object positions from pixel colors, this reads the exact
bounding boxes that game.py knows about — perfect ground-truth labels every time.

Output structure:
    dataset/
        images/train/   ← screenshots (.jpg)
        images/val/     ← validation screenshots
        labels/train/   ← YOLO label files (.txt)
        labels/val/
        data.yaml       ← YOLO config file

Usage:
    python game.py          # terminal 1 — play normally
    python collect_data.py  # terminal 2 — collects TARGET_FRAMES then stops
"""

import os
import json
import time
import random
import shutil
import numpy as np
import cv2

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
SHARED_FRAME_PATH = "shared_frame.npy"
SHARED_STATE_PATH = "shared_state.json"   # ground-truth positions from game.py
DATASET_DIR       = "dataset"

TARGET_FRAMES = 500    # stop after this many labeled frames
VAL_SPLIT     = 0.15   # fraction held out for validation
MIN_FRAME_GAP = 0.1    # seconds between captures (avoids near-duplicate frames)
MIN_OBJECTS   = 1      # skip completely empty frames

WIDTH, HEIGHT = 960, 540

CLASS_NAMES = ["player", "zapper", "missile", "warning"]
CLASS_ID    = {name: i for i, name in enumerate(CLASS_NAMES)}

# Frames where player overlaps an obstacle above this IoU are skipped —
# these are collision/near-death frames that teach YOLO the wrong thing
MAX_PLAYER_OBSTACLE_IOU = 0.05

# Skip frames where player center is within this many pixels of any obstacle bbox
# Catches near-miss frames that still create confusing partial overlaps
PROXIMITY_MARGIN = 20


# ──────────────────────────────────────────────
# Overlap / proximity checks
# ──────────────────────────────────────────────
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
    """
    Returns True if the player center is within PROXIMITY_MARGIN pixels
    of any obstacle bounding box. Filters out near-death frames where the
    player visually overlaps an obstacle even without a full IoU hit.
    """
    pcx = player["x"] + player["w"] / 2
    pcy = player["y"] + player["h"] / 2

    for obs in obstacles:
        ox1 = obs["x"] - PROXIMITY_MARGIN
        oy1 = obs["y"] - PROXIMITY_MARGIN
        ox2 = obs["x"] + obs["w"] + PROXIMITY_MARGIN
        oy2 = obs["y"] + obs["h"] + PROXIMITY_MARGIN
        if ox1 <= pcx <= ox2 and oy1 <= pcy <= oy2:
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
            return False  # direct overlap — skip

    if player_too_close(player, obstacle_objs):
        return False  # dangerously close — skip

    return True


# ──────────────────────────────────────────────
# YOLO label format: class_id cx cy w h  (normalised 0–1)
# ──────────────────────────────────────────────
def to_yolo_line(obj: dict) -> str | None:
    """Convert one game object dict to a YOLO label line."""
    label = obj["label"]
    if label not in CLASS_ID:
        return None

    x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]

    # Clamp to image bounds (objects may be partially off-screen)
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


# ──────────────────────────────────────────────
# File loaders
# ──────────────────────────────────────────────
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


# ──────────────────────────────────────────────
# Dataset setup
# ──────────────────────────────────────────────
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


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
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

            # Skip frames where player is touching or near an obstacle —
            # these are the exact frames that may teach YOLO "player looks like zapper"
            if not frame_is_clean(objects):
                continue

            # Build YOLO label lines directly from game state
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

            if collected % 50 == 0 or collected == 1:
                label_summary = ", ".join(obj["label"] for obj in objects)
                print(f"  [{collected:>4}/{TARGET_FRAMES}]  "
                      f"{len(objects)} objects  ({label_summary})")

    except KeyboardInterrupt:
        print("\nStopped early.")

    if not saved_stems:
        print("No frames collected — make sure game.py is running.")
        return

    # ── Move validation split ─────────────────────────────────
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
    print(f"  (Overlap/proximity filter kept your labels clean)")
    print("Next step:  python train_yolo.py")


if __name__ == "__main__":
    main()