"""
Collect mouth crops (neutral/open/smile/yawn) from webcam using MediaPipe FaceMesh.

Usage:
    python wraith/collect_yawn_data.py --out data/mouth --img_w 64 --img_h 64

Keys:
  n - save neutral
  o - save open (small open mouth)
  s - save smile
  y - save yawn (wide open)
  q - quit

Saved to:
    data/mouth/neutral
    data/mouth/open
    data/mouth/smile
    data/mouth/yawn
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

# Use a few mouth landmarks to compute a mouth box
MOUTH_LANDMARKS = [13, 14, 61, 291, 78, 308]


def mouth_bbox_from_landmarks(lm, img_w, img_h, pad=1.4):
    xs = [lm[i].x for i in MOUTH_LANDMARKS]
    ys = [lm[i].y for i in MOUTH_LANDMARKS]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    w = (maxx - minx) * pad
    h = (maxy - miny) * pad
    x = int(max(0, min(img_w - 1, (cx - w / 2) * img_w)))
    y = int(max(0, min(img_h - 1, (cy - h / 2) * img_h)))
    ww = int(max(4, min(img_w, w * img_w)))
    hh = int(max(4, min(img_h, h * img_h)))
    return x, y, ww, hh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/mouth")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--img_w", type=int, default=64)
    ap.add_argument("--img_h", type=int, default=64)
    args = ap.parse_args()

    out_root = Path(args.out)
    classes = ["neutral", "open", "smile", "yawn"]
    for c in classes:
        (out_root / c).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("Failed to open webcam")
        return

    mp_face_mesh = mp.solutions.face_mesh  # type: ignore[attr-defined]
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        print("Press n/o/s/y to save neutral/open/smile/yawn. q to quit.")
    crop_resized = None
    while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # MediaPipe sometimes requires contiguous, non-writeable frames
            frame_rgb_c = np.ascontiguousarray(frame_rgb)
            res = face_mesh.process(frame_rgb_c)

            h_img, w_img, _ = frame.shape
            label_text = "[n] neutral  [o] open  [s] smile  [y] yawn  [q] quit"

            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                x, y, ww, hh = mouth_bbox_from_landmarks(lm, w_img, h_img)
                crop = frame[y : y + hh, x : x + ww]
                if crop.size != 0:
                    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    crop_resized = cv2.resize(crop_gray, (args.img_w, args.img_h))
                    disp = cv2.cvtColor(crop_resized, cv2.COLOR_GRAY2BGR)
                    frame[10 : 10 + args.img_h, 10 : 10 + args.img_w] = disp
                    cv2.rectangle(frame, (x, y), (x + ww, y + hh), (0, 255, 0), 2)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key in (ord("n"), ord("o"), ord("s"), ord("y")):
                    if crop_resized is not None:
                        label = {ord("n"): "neutral", ord("o"): "open", ord("s"): "smile", ord("y"): "yawn"}[key]
                        ts = int(time.time() * 1000)
                        fname = out_root / label / f"mouth_{ts}.png"
                        cv2.imwrite(str(fname), crop_resized)
                        print(f"Saved {fname}")
            else:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            cv2.putText(
                frame,
                label_text,
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Collect Yawn Data", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
