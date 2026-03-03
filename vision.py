"""
vision.py — AI Vision Pipeline for Jetpack Rectangle

Usage:
    python vision.py -color     # HSV color detector (no training needed)
    python vision.py -yolo      # trained YOLO detector
"""

import os
import json
import time
import numpy as np
import cv2
import argparse

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
YOLO_WEIGHTS      = "runs/detect/runs/detect/jetpack/weights/best.pt"
YOLO_CONF_THRESH  = 0.25   # lower threshold — we post-process labels anyway

SHARED_FRAME_PATH = "shared_frame.npy"
SHARED_STATE_PATH = "shared_state.json"
POLL_INTERVAL     = 0.005

WIDTH, HEIGHT = 960, 540
PLAY_TOP      = 24
PLAY_BOTTOM   = HEIGHT - 24

# Player is always fixed on the left — anything detected here gets reclassified
PLAYER_X_MAX  = 260   # anything with cx < this AND correct size = player
PLAYER_W_MIN, PLAYER_W_MAX = 20, 80
PLAYER_H_MIN, PLAYER_H_MAX = 25, 90

CLASS_NAMES = ["player", "zapper", "missile", "warning"]


# ──────────────────────────────────────────────
# Detector 1: Color / HSV
# ──────────────────────────────────────────────
COLOR_DEFS = {
    "player":  {"hsv_lower": np.array([20, 150, 180]), "hsv_upper": np.array([35, 255, 255])},
    "zapper":  {"hsv_lower": np.array([100, 100, 150]),"hsv_upper": np.array([130, 255, 255])},
    "missile": {"hsv_lower": np.array([0, 150, 180]),  "hsv_upper": np.array([8, 255, 255])},
    "warning": {"hsv_lower": np.array([170, 150, 180]),"hsv_upper": np.array([180, 255, 255])},
}
MIN_AREA = {"player": 400, "zapper": 300, "missile": 200, "warning": 150}


class ColorObjectDetector:
    def detect(self, frame_bgr: np.ndarray) -> list:
        hsv    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dets   = []
        for label, cfg in COLOR_DEFS.items():
            mask = cv2.inRange(hsv, cfg["hsv_lower"], cfg["hsv_upper"])
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < MIN_AREA.get(label, 200):
                    continue
                bx, by, bw, bh = cv2.boundingRect(cnt)
                dets.append({
                    "label": label, "x": bx, "y": by, "w": bw, "h": bh,
                    "cx": bx + bw // 2, "cy": by + bh // 2,
                    "confidence": min(1.0, area / 2000.0),
                    "source": "color",
                })
        return dets


# ──────────────────────────────────────────────
# Detector 2: YOLOv8
# ──────────────────────────────────────────────
class YOLODetector:
    def __init__(self, weights_path: str, conf: float = YOLO_CONF_THRESH):
        from ultralytics import YOLO
        print(f"Loading YOLO weights from '{weights_path}'...")
        self.model = YOLO(weights_path)
        self.conf  = conf
        print("YOLO model loaded.")

    def detect(self, frame_bgr: np.ndarray) -> list:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results   = self.model(frame_rgb, conf=self.conf, imgsz=960, verbose=False)
        dets      = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf     = float(box.conf[0])
                class_id = int(box.cls[0])
                label    = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id)
                w, h     = x2 - x1, y2 - y1
                dets.append({
                    "label": label,
                    "x": x1, "y": y1, "w": w, "h": h,
                    "cx": x1 + w // 2, "cy": y1 + h // 2,
                    "confidence": conf,
                    "source": "yolo",
                })
        return dets


# ──────────────────────────────────────────────
# Post-processing: fix YOLO label errors using
# spatial and size constraints we know are true
# ──────────────────────────────────────────────
def postprocess(detections: list, state: dict | None) -> list:
    """
    YOLO finds objects accurately but sometimes mislabels them.
    We know hard facts about this game:
      - Player is ALWAYS on the left (cx < PLAYER_X_MAX), small, roughly square
      - Obstacles come from the RIGHT (cx > PLAYER_X_MAX)
      - Zappers are elongated (high aspect ratio or large)
      - Missiles are small rectangles
      - Warnings are wide, thin horizontal bands on the right

    Strategy:
      1. Anything detected on the left in the player size range → force label = player
      2. Anything labeled "player" but found on the right → reclassify by shape
      3. Inject GT player from game state if YOLO missed it entirely
    """
    corrected = []

    for det in detections:
        d = dict(det)  # copy
        cx, cy = d["cx"], d["cy"]
        w, h   = d["w"], d["h"]

        # Rule 1: anything in the player zone with player-like dimensions = player
        # The player is a ~38x48 rectangle, always at x≈121-159
        if (cx < PLAYER_X_MAX and
            PLAYER_W_MIN <= w <= PLAYER_W_MAX and
            PLAYER_H_MIN <= h <= PLAYER_H_MAX):
            if d["label"] != "player":
                d["corrected_from"] = d["label"]
                d["label"] = "player"

        # Rule 2: anything labeled "player" but found on the RIGHT = mislabeled
        # Reclassify by aspect ratio: wide=zapper, small=missile, tall=zapper
        elif d["label"] == "player" and cx >= PLAYER_X_MAX:
            ar = w / h if h > 0 else 1.0
            d["corrected_from"] = "player"
            if w > 100 or h > 100:
                d["label"] = "zapper"
            elif ar > 1.8:
                d["label"] = "missile"
            else:
                d["label"] = "zapper"

        corrected.append(d)

    # Rule 3: if no player detected on left side, inject from game state
    has_player = any(d["label"] == "player" and d["cx"] < PLAYER_X_MAX
                     for d in corrected)
    if not has_player and state:
        for obj in state.get("objects", []):
            if obj["label"] == "player":
                corrected.append({
                    "label":   "player",
                    "x": obj["x"], "y": obj["y"],
                    "w": obj["w"], "h": obj["h"],
                    "cx": obj["x"] + obj["w"] // 2,
                    "cy": obj["y"] + obj["h"] // 2,
                    "confidence": -1.0,  # -1 = ground truth fallback
                    "source": "gt",
                })

    return corrected


# ──────────────────────────────────────────────
# Visualizer
# ──────────────────────────────────────────────
VISION_BG  = (10, 10, 15)
GRID_COLOR = (30, 30, 40)

# Colors chosen to be visually distinct from each other
# and match the game's actual object colors loosely:
# Player = yellow (game color), Zapper = blue (game color),
# Missile = red (game color), Warning = yellow/gold (changed from red)
VIS_STYLE = {
    "player":  {"color": (0,   220, 255), "shape": "rect",  "thickness": 2},  # cyan
    "zapper":  {"color": (255, 140,  40), "shape": "line",  "thickness": 3},  # orange-blue
    "missile": {"color": (50,   60, 255), "shape": "rect",  "thickness": 2},  # red
    "warning": {"color": (0,   210, 255), "shape": "hline", "thickness": 2},  # yellow (was dark red)
}

# Game-accurate colors for reference strip
GAME_COLORS = {
    "player":  (0,   210, 245),  # yellow in BGR
    "zapper":  (255, 160,  70),  # blue in BGR
    "missile": (80,   80, 255),  # red in BGR
    "warning": (0,   200, 255),  # yellow-gold in BGR
}


class VisionVisualizer:
    def __init__(self, width: int, height: int):
        self.w = width
        self.h = height

    def render(self, detections: list, meta: dict | None = None) -> np.ndarray:
        canvas = np.full((self.h, self.w, 3), VISION_BG, dtype=np.uint8)

        # Grid
        for x in range(0, self.w, 80):
            cv2.line(canvas, (x, 0), (x, self.h), GRID_COLOR, 1)
        for y in range(0, self.h, 60):
            cv2.line(canvas, (0, y), (self.w, y), GRID_COLOR, 1)

        # Play area bounds
        cv2.line(canvas, (0, PLAY_TOP),    (self.w, PLAY_TOP),    (60, 60, 80), 1)
        cv2.line(canvas, (0, PLAY_BOTTOM), (self.w, PLAY_BOTTOM), (60, 60, 80), 1)

        # Player zone marker — vertical line showing where player always lives
        cv2.line(canvas, (PLAYER_X_MAX, PLAY_TOP), (PLAYER_X_MAX, PLAY_BOTTOM),
                 (40, 40, 55), 1)

        for det in detections:
            label  = det["label"]
            style  = VIS_STYLE.get(label, {"color": (200, 200, 200), "shape": "rect", "thickness": 1})
            color  = style["color"]
            x, y, w, h = det["x"], det["y"], det["w"], det["h"]
            cx, cy = det["cx"], det["cy"]
            conf   = det.get("confidence", 1.0)
            source = det.get("source", "yolo")
            was_corrected = "corrected_from" in det

            if source == "gt" or conf < 0:
                # Ground truth fallback — dashed box
                self._draw_dashed_rect(canvas, x, y, w, h, color)
                label_text = f"{label.upper()} [GT]"
            elif was_corrected:
                # YOLO found it but label was wrong — solid box with correction note
                cv2.rectangle(canvas, (x, y), (x + w, y + h), color, style["thickness"])
                cv2.circle(canvas, (cx, cy), 3, color, -1)
                label_text = f"{label.upper()} {conf:.0%} ←{det['corrected_from'].upper()}"
            else:
                # Normal detection
                if style["shape"] == "rect":
                    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, style["thickness"])
                    cv2.circle(canvas, (cx, cy), 3, color, -1)
                elif style["shape"] == "line":
                    cv2.line(canvas, (x, cy), (x + w, cy), color, style["thickness"] + 1)
                    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 1)
                elif style["shape"] == "hline":
                    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, style["thickness"])
                label_text = f"{label.upper()} {conf:.0%}"

            # Confidence bar (skip for GT)
            if conf >= 0:
                bar_w = int(w * conf)
                cv2.rectangle(canvas, (x, max(y - 6, 0)),
                              (x + bar_w, max(y - 2, 0)), color, -1)

            cv2.putText(canvas, label_text,
                        (x, max(y - 8, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Legend strip at bottom — shows what each color means
        self._draw_legend(canvas)

        # Stats overlay
        if meta:
            mode_label   = "CNN (YOLO)" if meta.get("mode") == "yolo" else "Color HSV"
            player_src   = meta.get("player_source", "cnn")
            src_color    = (80, 80, 255) if player_src == "gt" else (180, 220, 180)
            src_label    = "GT fallback" if player_src == "gt" else \
                           "corrected"   if player_src == "corrected" else "CNN"
            lines = [
                (f"Detector:   {mode_label}",           (180, 220, 180)),
                (f"Player:     {src_label}",             src_color),
                (f"Detections: {meta.get('n', 0)}",      (180, 220, 180)),
                (f"FPS (vis):  {meta.get('fps', 0):.1f}",(180, 220, 180)),
            ]
            for i, (line, col) in enumerate(lines):
                cv2.putText(canvas, line, (10, 20 + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

        return canvas

    def _draw_dashed_rect(self, canvas, x, y, w, h, color, dash=8):
        """Draw a dashed rectangle for ground-truth fallback detections."""
        for dx in range(0, w, dash * 2):
            x1, x2 = x + dx, min(x + dx + dash, x + w)
            cv2.line(canvas, (x1, y),     (x2, y),     color, 1)
            cv2.line(canvas, (x1, y + h), (x2, y + h), color, 1)
        for dy in range(0, h, dash * 2):
            y1, y2 = y + dy, min(y + dy + dash, y + h)
            cv2.line(canvas, (x,     y1), (x,     y2), color, 1)
            cv2.line(canvas, (x + w, y1), (x + w, y2), color, 1)

    def _draw_legend(self, canvas):
        """Colored legend strip at the bottom of the vision window."""
        legend_y = self.h - 22
        items = [
            ("PLAYER",  VIS_STYLE["player"]["color"]),
            ("ZAPPER",  VIS_STYLE["zapper"]["color"]),
            ("MISSILE", VIS_STYLE["missile"]["color"]),
            ("WARNING", VIS_STYLE["warning"]["color"]),
        ]
        x = 10
        for name, color in items:
            cv2.rectangle(canvas, (x, legend_y), (x + 12, legend_y + 12), color, -1)
            cv2.putText(canvas, name, (x + 16, legend_y + 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
            x += len(name) * 8 + 30


# ──────────────────────────────────────────────
# File readers
# ──────────────────────────────────────────────
def load_frame(path: str):
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


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="AI Vision Pipeline")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-color", action="store_true", help="Use HSV color detector")
    group.add_argument("-yolo",  action="store_true", help="Use trained YOLO detector")
    args   = parser.parse_args()

    detector_mode = "yolo" if args.yolo else "color"

    if detector_mode == "yolo":
        if not os.path.exists(YOLO_WEIGHTS):
            print(f"ERROR: YOLO weights not found at '{YOLO_WEIGHTS}'")
            print("Run train_yolo.py first, or use -color flag")
            return
        detector = YOLODetector(YOLO_WEIGHTS)
    else:
        detector = ColorObjectDetector()
        print("Using color-based detector.")

    visualizer = VisionVisualizer(WIDTH, HEIGHT)

    cv2.namedWindow("AI Vision", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AI Vision", WIDTH, HEIGHT)

    print(f"Waiting for game.py frames at '{SHARED_FRAME_PATH}' ...")
    print("Press Q in the AI Vision window to quit.\n")

    last_mtime  = 0.0
    fps_timer   = time.perf_counter()
    fps_count   = 0
    current_fps = 0.0

    placeholder = np.full((HEIGHT, WIDTH, 3), VISION_BG, dtype=np.uint8)
    cv2.putText(placeholder, "Waiting for game.py...",
                (WIDTH // 2 - 160, HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 180, 100), 2, cv2.LINE_AA)
    cv2.imshow("AI Vision", placeholder)

    while True:
        try:
            mtime = os.path.getmtime(SHARED_FRAME_PATH)
        except FileNotFoundError:
            mtime = 0.0

        if mtime != last_mtime:
            last_mtime = mtime
            frame_bgr  = load_frame(SHARED_FRAME_PATH)
            state      = load_state(SHARED_STATE_PATH)

            if frame_bgr is not None:
                raw_dets   = detector.detect(frame_bgr)
                detections = postprocess(raw_dets, state)

                # Determine player source for HUD
                player_dets = [d for d in detections if d["label"] == "player"]
                if not player_dets:
                    player_source = "missing"
                elif player_dets[0].get("source") == "gt":
                    player_source = "gt"
                elif player_dets[0].get("corrected_from"):
                    player_source = "corrected"
                else:
                    player_source = "cnn"

                fps_count += 1
                now = time.perf_counter()
                if now - fps_timer >= 1.0:
                    current_fps = fps_count / (now - fps_timer)
                    fps_count   = 0
                    fps_timer   = now

                meta = {
                    "n":             len(detections),
                    "fps":           current_fps,
                    "mode":          detector_mode,
                    "player_source": player_source,
                }
                canvas = visualizer.render(detections, meta)
                cv2.imshow("AI Vision", canvas)
        else:
            time.sleep(POLL_INTERVAL)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()