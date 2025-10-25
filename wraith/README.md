# Wraith Wheels — Drowsy Detector (Halloween Edition)

Browser-based drowsiness detector that:
- Uses your webcam on-device (no uploads)
- Detects when your eyes are closed for a configurable duration
- Slaps a jack-o'-lantern over your face and shouts “Boo!” 👻

Powered by MediaPipe Face Mesh, running entirely in the browser.

## Features
- Start/Stop camera controls
- Live EAR (eye aspect ratio) readout for left/right eyes
- Optional tiny CNN eye classifier (TensorFlow.js) for robust open/closed detection
- Adjustable threshold and closed-duration
- Pumpkin face overlay with rotation and scaling tracking your face
- “Boo!” via Web Speech API or WebAudio fallback

## Run locally
Most browsers block camera access on file:// pages, so use a local server.

### Quick start (Python, any OS)
```bash
# from the project folder
python3 -m http.server 5173
```
Then open:
```
http://localhost:5173/
```
Click “Start Camera” and allow permissions.

### Node (optional)
If you prefer Node:
```bash
npx http-server -p 5173
```

## Controls
- Eye threshold: Lower value means stricter “open” requirement. Values ~0.20–0.28 usually work. Default 0.24.
- Closed duration: How long the average EAR must stay below threshold to trigger drowsiness. Default 1.5s.
- CNN closed prob: If you enable the CNN classifier, this is the probability threshold above which an eye is considered closed. Default 0.60.

## CNN eye-classifier (optional but recommended)
This app can use a tiny CNN to classify eye crops as open/closed for better robustness to lighting, glasses, and eye shape.

Enable it via the "Use CNN eye classifier (TF.js)" checkbox. You must provide a TF.js model at:

```
wraith/model/eye_state_model/model.json
```

### Train your own model
We provide a minimal training script using TensorFlow/Keras to train on grayscale eye crops.

1) Prepare a dataset in this layout:
```
data/eyes/
	open/   # PNG/JPG images of open eyes
	closed/ # PNG/JPG images of closed eyes
```
Each image should contain a single eye, roughly centered. The script will resize to 24x48 (HxW) and normalize.

Option A: Collect your own quickly via webcam (recommended to start):
```bash
source .venv/bin/activate
python wraith/collect_eye_data.py --out data/eyes --img_w 48 --img_h 24
# Press 'o' (open) and 'c' (closed) to save examples while looking at camera
```

2) Create a venv and install deps:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3) Train and export to TF.js:
```bash
python wraith/train_eye_cnn.py --epochs 8 --img_w 48 --img_h 24
```
This writes the TF.js model to:
```
wraith/model/eye_state_model/
```

4) Start the web app and enable the checkbox. The model will load from that folder.

Notes:
- You can also try public datasets like CEW (Closed Eyes in the Wild), subject to their licenses. Prepare the images into open/ and closed/ folders.
- The app flips the right eye crop horizontally so the CNN sees consistent left-to-right orientation.
- We apply a short temporal smoothing window to the CNN probabilities.

## Notes and tips
- Good, even lighting helps.
- Keep your face within the frame for best tracking.
- This is for fun/education, not a certified safety system.

## Credits
- MediaPipe Face Mesh by Google
- SVG pumpkin overlay is embedded and generated at runtime
 - Optional model training uses TensorFlow; in-browser inference via TensorFlow.js
