"""
Collect eye crops (open/closed) from your webcam using MediaPipe FaceMesh.

Usage:
  python wraith/collect_eye_data.py --out data/eyes --img_w 48 --img_h 24

Keys:
  o   Save current eye crops as OPEN  (both eyes; right eye is flipped)
  c   Save current eye crops as CLOSED (both eyes; right eye is flipped)
  q   Quit

Notes:
  - Ensure good lighting; look at camera. Blink and change head pose to vary samples.
  - Saved files:
      data/eyes/open/*.png   (grayscale 24x48)
      data/eyes/closed/*.png (grayscale 24x48)
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp


LEFT_EYE = {
    "upper": [386, 385], "lower": [374, 380], "left": 263, "right": 362
}
RIGHT_EYE = {
    "upper": [159, 158], "lower": [145, 153], "left": 133, "right": 33
}
EYE_PAD = 1.6


def landmarks_eye_box(lm, eye):
    u = (
        (lm[eye["upper"][0]].x + lm[eye["upper"][1]].x) / 2.0,
        (lm[eye["upper"][0]].y + lm[eye["upper"][1]].y) / 2.0,
    )
    l = (
        (lm[eye["lower"][0]].x + lm[eye["lower"][1]].x) / 2.0,
        (lm[eye["lower"][0]].y + lm[eye["lower"][1]].y) / 2.0,
    )
    left = lm[eye["left"]]
    right = lm[eye["right"]]
    cx = (left.x + right.x) / 2.0
    cy = (u[1] + l[1]) / 2.0
    w = abs(right.x - left.x)
    h = abs(l[1] - u[1]) * 2.2
    return cx, cy, w * EYE_PAD, h * EYE_PAD


def crop_eye(frame_bgr, cx, cy, w, h, out_w, out_h):
    h_img, w_img, _ = frame_bgr.shape
    x = int(max(0, min(w_img - 1, (cx - w / 2) * w_img)))
    y = int(max(0, min(h_img - 1, (cy - h / 2) * h_img)))
    ww = int(max(4, min(w_img, w * w_img)))
    hh = int(max(4, min(h_img, h * h_img)))
    crop = frame_bgr[y : y + hh, x : x + ww]
    if crop.size == 0:
        return None
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return crop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/eyes")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--img_w", type=int, default=48)
    ap.add_argument("--img_h", type=int, default=24)
    args = ap.parse_args()

    out_root = Path(args.out)
    out_open = out_root / "open"
    out_closed = out_root / "closed"
    out_open.mkdir(parents=True, exist_ok=True)
    out_closed.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("Failed to open webcam")
        return

    mp_face_mesh = mp.solutions.face_mesh  # type: ignore[attr-defined]
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        print("Press 'o' to save OPEN, 'c' to save CLOSED, 'q' to quit.")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(frame_rgb)

            h_img, w_img, _ = frame.shape
            label_text = "[o] open  [c] closed  [q] quit"
            color = (0, 255, 0)

            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                # draw simple eye lines
                for eye_def, col in ((LEFT_EYE, (0, 255, 0)), (RIGHT_EYE, (0, 255, 0))):
                    lx = int(lm[eye_def["left"]].x * w_img)
                    ly = int(lm[eye_def["left"]].y * h_img)
                    rx = int(lm[eye_def["right"]].x * w_img)
                    ry = int(lm[eye_def["right"]].y * h_img)
                    cv2.line(frame, (lx, ly), (rx, ry), col, 2)

                # prepare crops
                cxL, cyL, wL, hL = landmarks_eye_box(lm, LEFT_EYE)
                cxR, cyR, wR, hR = landmarks_eye_box(lm, RIGHT_EYE)
                left_eye = crop_eye(frame, cxL, cyL, wL, hL, args.img_w, args.img_h)
                right_eye = crop_eye(frame, cxR, cyR, wR, hR, args.img_w, args.img_h)
                if right_eye is not None:
                    right_eye = cv2.flip(right_eye, 1)  # flip horizontally

                # show small previews
                if left_eye is not None:
                    le_disp = cv2.cvtColor(left_eye, cv2.COLOR_GRAY2BGR)
                    frame[10 : 10 + args.img_h, 10 : 10 + args.img_w] = cv2.resize(
                        le_disp, (args.img_w, args.img_h)
                    )
                if right_eye is not None:
                    re_disp = cv2.cvtColor(right_eye, cv2.COLOR_GRAY2BGR)
                    frame[10 : 10 + args.img_h, 20 + args.img_w : 20 + 2 * args.img_w] = cv2.resize(
                        re_disp, (args.img_w, args.img_h)
                    )

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key in (ord("o"), ord("c")):
                    ts = int(time.time() * 1000)
                    tgt = out_open if key == ord("o") else out_closed
                    if left_eye is not None:
                        cv2.imwrite(str(tgt / f"eyeL_{ts}.png"), left_eye)
                    if right_eye is not None:
                        cv2.imwrite(str(tgt / f"eyeR_{ts}.png"), right_eye)
                    print(f"Saved to {tgt}")
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
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Collect Eye Data", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
