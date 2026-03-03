"""
vision.py — AI Vision Pipeline for Jetpack Rectangle

Usage:
    python vision.py -color     # HSV color detector (no training needed)
    python vision.py -yolo      # trained YOLO detector

No ground-truth data is used at inference time. Detection is purely
from pixel data. The postprocess() function only applies spatial and
size rules that are structural facts about the game layout.
"""

import os
import time
import numpy as np
import cv2
import argparse

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
YOLO_WEIGHTS      = "runs/detect/runs/detect/jetpack/weights/best.pt"
YOLO_CONF_THRESH  = 0.45   # reject weak guesses

SHARED_FRAME_PATH = "shared_frame.npy"
POLL_INTERVAL     = 0.005

WIDTH, HEIGHT = 960, 540
PLAY_TOP      = 24
PLAY_BOTTOM   = HEIGHT - 24


PLAYER_ZONE_CX_MAX = 260   # player cx is always ~140; obstacles spawn at x=1000
PLAYER_W_MIN, PLAYER_W_MAX = 18, 85
PLAYER_H_MIN, PLAYER_H_MAX = 22, 95

CLASS_NAMES = ["player", "zapper", "missile", "warning"]


COLOR_DEFS = {
    "player":  {"hsv_lower": np.array([55,  120, 120]), "hsv_upper": np.array([80,  255, 255])},
    "zapper":  {"hsv_lower": np.array([100, 100, 150]), "hsv_upper": np.array([130, 255, 255])},
    "missile": {"hsv_lower": np.array([0,   150, 180]), "hsv_upper": np.array([8,   255, 255])},
    "warning": {"hsv_lower": np.array([20,  180, 180]), "hsv_upper": np.array([32,  255, 255])},
}
MIN_AREA = {"player": 300, "zapper": 300, "missile": 200, "warning": 150}


# Detector 1: HSV color thresholding + morphology
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


# Detector 2: YOLOv8
class YOLODetector:
    def __init__(self, weights_path: str, conf: float = YOLO_CONF_THRESH):
        from ultralytics import YOLO
        print(f"Loading YOLO weights from '{weights_path}'...")
        self.model = YOLO(weights_path)
        self.conf  = conf
        print("YOLO model loaded.")

    def detect(self, frame_bgr: np.ndarray) -> list:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results   = self.model(frame_rgb, conf=self.conf, imgsz=960,
                               iou=0.3, verbose=False)
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


# Post-processing: correct mislabels using only
# structural facts about the game layout.
def postprocess(detections: list) -> list:
    corrected = []

    for det in detections:
        d  = dict(det)
        cx = d["cx"]
        w, h = d["w"], d["h"]

        # Rule 1: anything in the player zone with player-like dimensions = player.
        # The player never moves horizontally so cx is always ~140.
        # Fast vertical movement can inflate bbox height slightly, hence generous H bounds.
        in_zone     = cx < PLAYER_ZONE_CX_MAX
        right_size  = PLAYER_W_MIN <= w <= PLAYER_W_MAX and PLAYER_H_MIN <= h <= PLAYER_H_MAX

        if in_zone and right_size and d["label"] != "player":
            d["corrected_from"] = d["label"]
            d["label"] = "player"

        # Rule 2: nothing labeled "player" can exist on the right side of the screen.
        # Reclassify by aspect ratio and size.
        elif d["label"] == "player" and cx >= PLAYER_ZONE_CX_MAX:
            ar = w / h if h > 0 else 1.0
            d["corrected_from"] = "player"
            d["label"] = "zapper" if (w > 80 or h > 80 or ar < 1.8) else "missile"

        corrected.append(d)

    return corrected

VIS_STYLE = {
    "player":  {"color": (50,  205,  50), "shape": "rect",  "thickness": 2},
    "zapper":  {"color": (255, 160,  70), "shape": "line",  "thickness": 3},
    "missile": {"color": (80,   80, 255), "shape": "rect",  "thickness": 2},
    "warning": {"color": (0,   200, 255), "shape": "hline", "thickness": 2},
}

VISION_BG  = (10, 10, 15)
GRID_COLOR = (30, 30, 40)


class VisionVisualizer:
    def __init__(self, width: int, height: int):
        self.w = width
        self.h = height

    def render(self, detections: list, meta: dict | None = None) -> np.ndarray:
        canvas = np.full((self.h, self.w, 3), VISION_BG, dtype=np.uint8)

        for x in range(0, self.w, 80):
            cv2.line(canvas, (x, 0), (x, self.h), GRID_COLOR, 1)
        for y in range(0, self.h, 60):
            cv2.line(canvas, (0, y), (self.w, y), GRID_COLOR, 1)

        cv2.line(canvas, (0, PLAY_TOP),    (self.w, PLAY_TOP),    (60, 60, 80), 1)
        cv2.line(canvas, (0, PLAY_BOTTOM), (self.w, PLAY_BOTTOM), (60, 60, 80), 1)
        cv2.line(canvas, (PLAYER_ZONE_CX_MAX, PLAY_TOP),
                         (PLAYER_ZONE_CX_MAX, PLAY_BOTTOM), (40, 55, 40), 1)

        for det in detections:
            label = det["label"]
            style = VIS_STYLE.get(label, {"color": (200, 200, 200), "shape": "rect", "thickness": 1})
            color = style["color"]
            x, y, w, h = det["x"], det["y"], det["w"], det["h"]
            cx, cy     = det["cx"], det["cy"]
            conf       = det.get("confidence", 1.0)

            if "corrected_from" in det:
                cv2.rectangle(canvas, (x, y), (x + w, y + h), color, style["thickness"])
                cv2.circle(canvas, (cx, cy), 3, color, -1)
                label_text = f"{label.upper()} {conf:.0%} \u2190{det['corrected_from'][:3].upper()}"
            else:
                shape = style["shape"]
                if shape == "rect":
                    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, style["thickness"])
                    cv2.circle(canvas, (cx, cy), 3, color, -1)
                elif shape == "line":
                    cv2.line(canvas, (x, cy), (x + w, cy), color, style["thickness"] + 1)
                    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 1)
                elif shape == "hline":
                    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, style["thickness"])
                label_text = f"{label.upper()} {conf:.0%}"

            bar_w = max(1, int(w * conf))
            cv2.rectangle(canvas, (x, max(y - 6, 0)), (x + bar_w, max(y - 2, 0)), color, -1)
            cv2.putText(canvas, label_text, (x, max(y - 8, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        self._draw_legend(canvas)

        if meta:
            mode_label = "CNN (YOLO)" if meta.get("mode") == "yolo" else "Color HSV"
            player_src = meta.get("player_source", "cnn")
            src_color  = (80,  80, 255) if player_src == "missing"   else \
                         (80, 200, 255) if player_src == "corrected" else (180, 220, 180)
            src_label  = {"corrected": "corrected",
                          "cnn":       "CNN OK",
                          "missing":   "MISSING"}.get(player_src, player_src)
            lines = [
                (f"Detector:   {mode_label}",             (180, 220, 180)),
                (f"Player:     {src_label}",               src_color),
                (f"Detections: {meta.get('n', 0)}",        (180, 220, 180)),
                (f"FPS (vis):  {meta.get('fps', 0):.1f}",  (180, 220, 180)),
            ]
            for i, (text, col) in enumerate(lines):
                cv2.putText(canvas, text, (10, 20 + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

        return canvas

    def _draw_legend(self, canvas):
        ly, x = self.h - 22, 10
        for name, style in VIS_STYLE.items():
            color = style["color"]
            cv2.rectangle(canvas, (x, ly), (x + 12, ly + 12), color, -1)
            cv2.putText(canvas, name.upper(), (x + 16, ly + 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
            x += len(name) * 8 + 30


# Frame reader
def load_frame(path: str):
    try:
        rgb = np.load(path)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


# Main
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
    print(f"Waiting for frames at '{SHARED_FRAME_PATH}' ...")
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

            if frame_bgr is not None:
                raw_dets   = detector.detect(frame_bgr)
                detections = postprocess(raw_dets)

                player_dets   = [d for d in detections if d["label"] == "player"]
                player_source = "missing"   if not player_dets else \
                                "corrected" if player_dets[0].get("corrected_from") else "cnn"

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
                cv2.imshow("AI Vision", visualizer.render(detections, meta))
        else:
            time.sleep(POLL_INTERVAL)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()