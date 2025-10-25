// Wraith Wheels — client-side drowsiness detector with jack-o'-lantern overlay
// Uses MediaPipe FaceMesh via CDN (no build step)

const els = {
  video: document.getElementById('video'),
  canvas: document.getElementById('output'),
  startBtn: document.getElementById('startBtn'),
  stopBtn: document.getElementById('stopBtn'),
  state: document.getElementById('state'),
  earL: document.getElementById('earL'),
  earR: document.getElementById('earR'),
  cnnL: document.getElementById('cnnL'),
  cnnR: document.getElementById('cnnR'),
  closedTime: document.getElementById('closedTime'),
  msg: document.getElementById('msg'),
  targetFace: document.getElementById('targetFace'),
  threshold: document.getElementById('threshold'),
  thresholdVal: document.getElementById('thresholdVal'),
  duration: document.getElementById('duration'),
  durationVal: document.getElementById('durationVal'),
  hideDelay: document.getElementById('hideDelay'),
  hideDelayVal: document.getElementById('hideDelayVal'),
  tiltThresh: document.getElementById('tiltThresh'),
  tiltThreshVal: document.getElementById('tiltThreshVal'),
  tiltCooldown: document.getElementById('tiltCooldown'),
  tiltCooldownVal: document.getElementById('tiltCooldownVal'),
  yawnThresh: document.getElementById('yawnThresh'),
  yawnThreshVal: document.getElementById('yawnThreshVal'),
  yawnCooldown: document.getElementById('yawnCooldown'),
  yawnCooldownVal: document.getElementById('yawnCooldownVal'),
  yawnMorMin: document.getElementById('yawnMorMin'),
  yawnMorMinVal: document.getElementById('yawnMorMinVal'),
  yawnMinDur: document.getElementById('yawnMinDur'),
  yawnMinDurVal: document.getElementById('yawnMinDurVal'),
  captureMode: document.getElementById('captureMode'),
  useCnn: document.getElementById('useCnn'),
  cnnThresh: document.getElementById('cnnThresh'),
  cnnThreshVal: document.getElementById('cnnThreshVal'),
  cnnPumpkinDelay: document.getElementById('cnnPumpkinDelay'),
  cnnPumpkinDelayVal: document.getElementById('cnnPumpkinDelayVal'),
  useMouthCnn: document.getElementById('useMouthCnn'),
  mouthPred: document.getElementById('mouthPred'),
  // Gaze controls
  useGaze: document.getElementById('useGaze'),
  gazeThreshold: document.getElementById('gazeThreshold'),
  gazeThresholdVal: document.getElementById('gazeThresholdVal'),
  gazeHold: document.getElementById('gazeHold'),
  gazeHoldVal: document.getElementById('gazeHoldVal'),
  gazeX: document.getElementById('gazeX'),
  gazeY: document.getElementById('gazeY'),
  gazeDrift: document.getElementById('gazeDrift'),
  gazeHoldProg: document.getElementById('gazeHoldProg'),
  gazeHoldTarget: document.getElementById('gazeHoldTarget'),
  // Gaze calibration controls
  gazeCalibEnable: document.getElementById('gazeCalibEnable'),
  gazeCalibrateBtn: document.getElementById('gazeCalibrateBtn'),
  gazeSigmaK: document.getElementById('gazeSigmaK'),
  gazeSigmaKVal: document.getElementById('gazeSigmaKVal'),
  gazeSmoothWin: document.getElementById('gazeSmoothWin'),
  gazeSmoothWinVal: document.getElementById('gazeSmoothWinVal'),
  // Sleep profile controls
  autoAdapt: document.getElementById('autoAdapt'),
  idealSleep: document.getElementById('idealSleep'),
  idealSleepVal: document.getElementById('idealSleepVal'),
  lastSleepHours: document.getElementById('lastSleepHours'),
  sleepImport: document.getElementById('sleepImport'),
  connectGoogleFit: document.getElementById('connectGoogleFit'),
  sleepRisk: document.getElementById('sleepRisk'),
  resetAdapt: document.getElementById('resetAdapt'),
  // Predictive alert controls
  usePredict: document.getElementById('usePredict'),
  predictThresh: document.getElementById('predictThresh'),
  predictThreshVal: document.getElementById('predictThreshVal'),
  predictWindow: document.getElementById('predictWindow'),
  predictWindowVal: document.getElementById('predictWindowVal'),
  predictProb: document.getElementById('predictProb'),
};

const ctx = els.canvas.getContext('2d');
let camera = null; // MediaPipe Camera instance
let fm = null; // FaceMesh
let running = false;
let lastTimestamp = 0;
let closedAccum = 0; // seconds for target face
let drowsy = false;
let targetIndexUser = null; // if user manually selects a face index
let lastFaceBoxes = []; // store boxes for hit testing
let lastEyesClosedAtMs = 0; // timestamp when eyes were last detected closed (target)
let pumpkinShowing = false; // overlay visibility state tied to target face
// Smooth pumpkin fade state
let pumpkinAlpha = 0.0; // current alpha 0..1
let pumpkinAlphaTarget = 0.0; // target alpha 0..1
let prevPumpkinVisible = false; // for rising-edge detection
let drowsyCount = 0; // persisted counter
// Siren audio state
let sirenAudio = null;
let sirenTimeout = null;
const SIREN_MAX_MS = 30 * 1000; // 30 seconds max

// Head tilt and skeleton hand settings/state
let TILT_THRESHOLD_RAD = 0.21; // default ~12°
let TILT_COOLDOWN_MS = 2500;  // default cooldown
let handAnim = null; // { side: 'left'|'right', startMs: number }
let lastHandTriggerMs = 0;
// Require head tilt to be held this many ms while eyes closed before ghost pushes
let TILT_HOLD_MS = 1500; // milliseconds (1.5s)
let tiltHoldStartMs = 0; // when tilt+eyes-closed started
let tiltHoldSide = null;

// Gaze detection settings/state
let GAZE_SMOOTH_WIN = 5;
let gazeHist = [];
let gazeHoldStartMs = 0;
let gazeActive = false; // drift active
let GAZE_HOLD_MS = 1200; // default 1.2s
let gazePrevActive = false; // to detect rising edge
let GAZE_SIGMA_K = 2.0; // z-score threshold multiplier when calibrated
let gazeCalib = { enabled: false, gx0: 0, gy0: 0, sx: 0.12, sy: 0.12, ts: 0 };
let gazeCalibSampling = null; // { startMs, samples: Array<{gx,gy}> }

// Sleep profile state
let autoAdapt = false;
let sleepProfile = {
  idealHours: 7.5,
  lastSleepHours: null, // unknown until set/imported
};
let adaptBaseline = null; // capture current thresholds when enabling autoAdapt
let lastAdaptApplyMs = 0;

// Predictive model state
let PREDICT_WIN_SEC = 120; // default 2 minutes
let predictEnabled = false;
let predictThresh = 0.65;
let predSecBucket = null; // { secStartMs, closedFrames, totalFrames, blinks, yawns, tilts, gazeDriftFrames }
let predHistory = []; // array of buckets within window
let prevClosedForBlink = false;
let blinkStartMs = 0;
let tiltEventsCounted = 0; // incremented on tilt hand trigger

// Skeleton head pop animation (for gaze drift)
let skullAnim = null; // { inStartMs, outStartMs: number|null }
let lastSkullTriggerMs = 0;
let SKULL_COOLDOWN_MS = 5000; // 5s default

const skullImg = new Image();
skullImg.src = 'data:image/svg+xml;utf8,' + encodeURIComponent(`
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 260">
  <defs>
    <radialGradient id="skullGrad" cx="50%" cy="35%" r="65%">
      <stop offset="0%" stop-color="#ffffff"/>
      <stop offset="60%" stop-color="#e9edf3"/>
      <stop offset="100%" stop-color="#cfd6df"/>
    </radialGradient>
    <linearGradient id="jawGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#f8f8f8"/>
      <stop offset="100%" stop-color="#e4e7ec"/>
    </linearGradient>
    <filter id="skullGlow" x="-40%" y="-40%" width="180%" height="180%">
      <feDropShadow dx="0" dy="0" stdDeviation="6" flood-color="#95b4ff" flood-opacity="0.55"/>
    </filter>
  </defs>
  <g filter="url(#skullGlow)">
    <!-- cranium -->
    <path d="M120 18 C 65 18, 28 55, 28 110 C 28 150, 52 166, 60 176 C 68 186, 75 198, 78 206 L 162 206 C 165 198, 172 186, 180 176 C 188 166, 212 150, 212 110 C 212 55, 175 18, 120 18 Z"
          fill="url(#skullGrad)" stroke="#5a606b" stroke-width="4"/>
    <!-- eye sockets -->
    <ellipse cx="88" cy="100" rx="20" ry="16" fill="#0e1116"/>
    <ellipse cx="152" cy="100" rx="20" ry="16" fill="#0e1116"/>
    <!-- subtle eye gloss -->
    <ellipse cx="82" cy="94" rx="6" ry="5" fill="#9fb9ff" opacity="0.35"/>
    <ellipse cx="146" cy="94" rx="6" ry="5" fill="#9fb9ff" opacity="0.35"/>
    <!-- nose cavity -->
    <path d="M120 114 L108 132 Q120 126 132 132 Z" fill="#1a1e26"/>
    <!-- zygomatic hint -->
    <path d="M54 130 Q74 120 82 124" stroke="#a9b2bf" stroke-width="3" opacity="0.6"/>
    <path d="M186 130 Q166 120 158 124" stroke="#a9b2bf" stroke-width="3" opacity="0.6"/>
    <!-- jaw / teeth row -->
    <path d="M70 206 Q120 220 170 206" fill="url(#jawGrad)" stroke="#69707d" stroke-width="3"/>
    <path d="M80 200 L160 200" stroke="#565c66" stroke-width="3" opacity="0.6"/>
    <g stroke="#444b56" stroke-width="3" opacity="0.75">
      <path d="M90 196 L90 208"/>
      <path d="M102 196 L102 208"/>
      <path d="M114 196 L114 208"/>
      <path d="M126 196 L126 208"/>
      <path d="M138 196 L138 208"/>
      <path d="M150 196 L150 208"/>
    </g>
  </g>
</svg>
`);

function updateAndDrawSkull(now, bbox){
  if(!skullAnim || !bbox || !skullImg.complete) return;
  const inDur = 320, outDur = 360;
  const inElapsed = now - skullAnim.inStartMs;
  const outElapsed = skullAnim.outStartMs != null ? (now - skullAnim.outStartMs) : 0;

  let phase, phaseT, t;
  if(inElapsed < inDur){
    phase = 'in';
    phaseT = inElapsed / inDur;
    t = phaseT;
  } else if(skullAnim.outStartMs == null){
    phase = 'hold';
    phaseT = 0;
    t = 1;
  } else if(outElapsed < outDur){
    phase = 'out';
    phaseT = outElapsed / outDur;
    t = 1 - phaseT;
  } else {
    skullAnim = null; // finished
    return;
  }
  t = clamp(t, 0, 1);

  const cw = els.canvas.width, ch = els.canvas.height;
  const faceTop = bbox.y;
  const faceCenterX = bbox.x + bbox.w * 0.5;
  const skullH = Math.max(80, bbox.h * 0.70);
  const aspect = 240/260; // from viewBox
  const skullW = skullH * aspect;
  const x = faceCenterX;
  const yStart = faceTop - skullH - Math.max(18, bbox.h * 0.10);
  const yFinal = faceTop - skullH * 0.58; // peek from behind the head
  // Ease
  const te = easeOutCubic(t);
  let yNow = yStart + (yFinal - yStart) * te;
  // Subtle bob during hold
  if(phase === 'hold'){
    const bob = Math.sin((now - skullAnim.inStartMs) / 140) * (bbox.h * 0.01);
    yNow += bob;
  } else if(phase === 'out'){
    yNow = yFinal + (yStart - yFinal) * phaseT;
  }
  ctx.save();
  ctx.globalAlpha = 0.97;
  // center pivot
  ctx.translate(x, yNow);
  ctx.translate(0, 0);
  // gentle scale and tilt
  const sIn = 0.9 + 0.12 * te; // 0.9 -> 1.02
  let rot = 0;
  if(phase === 'hold'){
    rot = Math.sin((now - skullAnim.inStartMs) / 320) * (3 * Math.PI/180);
  } else if(phase === 'in'){
    rot = (1 - te) * (5 * Math.PI/180);
  } else if(phase === 'out'){
    rot = (phaseT) * (4 * Math.PI/180);
  }
  ctx.translate(0, 0);
  ctx.rotate(rot);
  ctx.scale(sIn, sIn);
  // draw centered
  ctx.drawImage(skullImg, -skullW/2, -skullH/2, skullW, skullH);
  ctx.restore();
}

// Ghost asset (base facing right). We'll mirror for RIGHT so it faces inward.
const handImg = new Image();
handImg.src = 'data:image/svg+xml;utf8,' + encodeURIComponent(`
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 200">
  <defs>
    <radialGradient id="ghostBody" cx="50%" cy="35%" r="65%">
      <stop offset="0%" stop-color="#ffffff"/>
      <stop offset="70%" stop-color="#e6ecff"/>
      <stop offset="100%" stop-color="#c7d3ff"/>
    </radialGradient>
    <filter id="glow" x="-40%" y="-40%" width="180%" height="180%">
      <feDropShadow dx="0" dy="0" stdDeviation="6" flood-color="#a4b8ff" flood-opacity="0.6"/>
    </filter>
  </defs>
  <g filter="url(#glow)">
    <!-- ghost body -->
    <path d="M60 150 C 40 120, 40 70, 85 45 C 130 20, 190 40, 195 95 C 200 150, 188 165, 170 150 C 158 160, 142 160, 130 148 C 118 160, 98 160, 85 150 C 76 158, 66 158, 60 150 Z" fill="url(#ghostBody)" stroke="#8fa0d8" stroke-width="3"/>
    <!-- eyes and mouth -->
    <ellipse cx="120" cy="90" rx="10" ry="14" fill="#1f2a44"/>
    <ellipse cx="154" cy="92" rx="10" ry="14" fill="#1f2a44"/>
    <path d="M118 116 C 128 128, 148 128, 158 116" fill="none" stroke="#1f2a44" stroke-width="5" stroke-linecap="round"/>
  </g>
</svg>
`);

function easeOutCubic(t){ return 1 - Math.pow(1 - t, 3); }

function drawSkeletonHand(side, t, bbox, phase, phaseT){
  if(!handImg.complete) return;
  const cw = els.canvas.width, ch = els.canvas.height;
  // Scale hand relative to face height
  const handH = Math.max(80, bbox.h * 0.9);
  const aspect = 240/200; // from viewBox
  const handW = handH * aspect;

  // Base Y position: slightly lower for left side to simulate pushing under the jaw
  const yBase = (side === 'left')
    ? (bbox.y + bbox.h*0.62 - handH*0.5)
    : (bbox.y + bbox.h*0.5 - handH*0.5);
  let xStart, xFinal;
  if(side === 'left'){
    xStart = -handW - 10; // from left edge
    // Stop farther from face so it doesn't overlap too much
    xFinal = Math.max(10, bbox.x - handW*0.6);
  } else {
    xStart = cw + 10; // from right edge
    xFinal = Math.min(cw - handW - 10, bbox.x + bbox.w - handW*0.2);
  }
  const te = easeOutCubic(t);
  const xNow = xStart + (xFinal - xStart) * te;

  // Upward push motion during hold phase (and ease out on retract)
  let yNow = yBase;
  if(side === 'left'){
    let push = 0;
    if(phase === 'hold'){
      push = -bbox.h * 0.08 * easeOutCubic(Math.min(1, Math.max(0, phaseT)));
    } else if(phase === 'out'){
      // fade the push out
      push = -bbox.h * 0.08 * easeOutCubic(Math.max(0, 1 - phaseT));
    }
    yNow += push;
  }

  ctx.save();
  ctx.globalAlpha = 0.95;
  // Face the ghost inward: entering from LEFT uses base (right-facing);
  // entering from RIGHT mirrors the image (left-facing)
  if(side === 'right'){
    ctx.translate(xNow + handW, yNow);
    ctx.scale(-1, 1);
    // slight inward rotation
    ctx.translate(handW*0.03, handH*0.02);
    ctx.rotate(-4 * Math.PI/180);
    ctx.drawImage(handImg, 0, 0, handW, handH);
  } else {
    ctx.translate(xNow, yNow);
    ctx.rotate(2 * Math.PI/180);
    ctx.drawImage(handImg, 0, 0, handW, handH);
  }
  ctx.restore();
}

function updateAndDrawHand(now, bbox){
  if(!handAnim || !bbox) return;
  const inDur = 450, holdDur = 600, outDur = 450;
  const elapsed = now - handAnim.startMs;
  const total = inDur + holdDur + outDur;
  if(elapsed >= total){
    handAnim = null;
    return;
  }
  let t, phase, phaseT;
  if(elapsed < inDur){
    t = elapsed / inDur;
    phase = 'in';
    phaseT = t;
  } else if(elapsed < inDur + holdDur){
    t = 1;
    phase = 'hold';
    phaseT = (elapsed - inDur) / holdDur;
  } else {
    const outT = (elapsed - inDur - holdDur) / outDur;
    t = 1 - outT; // retract
    phase = 'out';
    phaseT = outT;
  }
  t = Math.max(0, Math.min(1, t));
  drawSkeletonHand(handAnim.side, t, bbox, phase, phaseT);
}

// Yawn detection: use mouth opening ratio (vertical / face height or mouth width)
let lastYawnMs = 0;
let yawnHoldStartMs = 0; // require persistence above thresholds to confirm yawn
let eyePopAnim = null; // { startMs, x, y }
const eyeImg = new Image();
eyeImg.src = 'data:image/svg+xml;utf8,' + encodeURIComponent(`
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 160 160">
  <defs>
    <radialGradient id="iris" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#33aaff"/>
      <stop offset="60%" stop-color="#0066aa"/>
      <stop offset="100%" stop-color="#003366"/>
    </radialGradient>
  </defs>
  <circle cx="80" cy="80" r="70" fill="#f5f5f5" stroke="#111" stroke-width="6"/>
  <circle cx="80" cy="80" r="36" fill="url(#iris)" stroke="#0a2236" stroke-width="4"/>
  <circle cx="80" cy="80" r="18" fill="#000"/>
  <circle cx="62" cy="62" r="10" fill="#fff"/>
</svg>
`);

function drawEyePop(now, x, y, faceBox, mor){
  if(!eyeImg.complete) return;
  const inDur = 300, holdDur = 500, outDur = 400;
  const elapsed = now - eyePopAnim.startMs;
  const total = inDur + holdDur + outDur;
  if(elapsed >= total){ eyePopAnim = null; return; }

  let t;
  if(elapsed < inDur) t = elapsed / inDur;
  else if(elapsed < inDur+holdDur) t = 1;
  else t = 1 - (elapsed - inDur - holdDur) / outDur;
  t = Math.max(0, Math.min(1, t));

  // Scale eye relative to face and mouth openness
  const base = Math.max(40, faceBox.h * 0.22);
  const opennessScale = 0.8 + Math.min(1.2, Math.max(0, mor)) * 0.6; // 0.8..1.52 range
  const animScale = 0.7 + 0.3 * t; // subtle grow-in
  const size = base * opennessScale * animScale;
  const yOffset = -faceBox.h * 0.10 * t; // pop upward a bit

  // Wobble around the mouth center while following it
  const time = now / 1000;
  const wobbleAx = faceBox.w * 0.015; // horizontal amplitude
  const wobbleAy = faceBox.h * 0.02;  // vertical amplitude
  const wx = Math.sin(time * 7.5) * wobbleAx;
  const wy = Math.cos(time * 6.0) * wobbleAy;
  ctx.save();
  ctx.globalAlpha = 0.98;
  ctx.translate(x - size/2 + wx, y - size/2 + yOffset + wy);
  ctx.drawImage(eyeImg, 0, 0, size, size);
  ctx.restore();
}

function mouthOpenRatio(lm){
  // Use top lip (13), bottom lip (14) for vertical; left (61) and right (291) for width
  const top = lm[13], bottom = lm[14], left = lm[61], right = lm[291];
  if(!(top&&bottom&&left&&right)) return 0;
  const vert = dist2D(top, bottom);
  const width = dist2D(left, right);
  if(width <= 1e-6) return 0;
  return vert / width; // larger when mouth opens wide
}

function mouthCenterCanvas(lm){
  const top = lm[13], bottom = lm[14];
  if(!(top && bottom)) return null;
  const cx = (top.x + bottom.x)/2 * els.canvas.width;
  const cy = (top.y + bottom.y)/2 * els.canvas.height;
  return {cx, cy};
}

// Preload jack-o'-lantern image (simple emoji or inline SVG). We'll draw a stylized pumpkin.
const pumpkin = new Image();
pumpkin.src = 'data:image/svg+xml;utf8,' + encodeURIComponent(`
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 260">
  <defs>
    <radialGradient id="g" cx="50%" cy="40%" r="60%">
      <stop offset="0%" stop-color="#ffcc33"/>
      <stop offset="70%" stop-color="#ff8800"/>
      <stop offset="100%" stop-color="#cc5500"/>
    </radialGradient>
  </defs>
  <ellipse cx="150" cy="130" rx="130" ry="100" fill="url(#g)" stroke="#552200" stroke-width="6"/>
  <rect x="135" y="10" width="30" height="40" rx="6" fill="#6b8e23" stroke="#3a5f0b" stroke-width="6"/>
  <path d="M70 140 L110 110 L150 140 Z" fill="#1a1000"/>
  <path d="M230 140 L190 110 L150 140 Z" fill="#1a1000"/>
  <path d="M100 180 C130 160 170 160 200 180 C175 205 125 205 100 180 Z" fill="#1a1000"/>
</svg>
`);

function lerp(a,b,t){ return a+(b-a)*t; }
function clamp(v, lo, hi){ return Math.max(lo, Math.min(hi, v)); }
function clamp01(x){ return Math.max(0, Math.min(1, x)); }

// ===================== CNN eye-state classifier (TF.js) =====================
// A tiny CNN that takes a grayscale eye crop (e.g., 24x48 HxW) and outputs P(closed)
// We'll support both eyes by flipping the right eye horizontally so the network sees consistent orientation.
let eyeModel = null; // tf.LayersModel
const EYE_H = 24;
const EYE_W = 48; // width > height to cover typical eye shape
const EYE_PAD = 1.6; // scale the eye bbox a bit
const CNN_SMOOTH_WIN = 5; // temporal smoothing window size
const probHist = { L: [], R: [] };
let PUMPKIN_CNN_SHOW_DELAY_SEC = 2.0; // wait this long with eyes closed (CNN) before showing pumpkin

async function loadEyeModel(){
  if(eyeModel) return eyeModel;
  try{
    // Expect model files under ./model/eye_state_model/model.json
    eyeModel = await tf.loadLayersModel('model/eye_state_model/model.json');
    console.log('Eye CNN loaded');
  }catch(err){
    console.warn('Eye CNN not available:', err);
    eyeModel = null;
  }
  return eyeModel;
}

// ------------------ Mouth classifier ------------------
let mouthModel = null;
const MOUTH_W = 64, MOUTH_H = 64;
const MOUTH_SMOOTH_WIN = 6;
let mouthHist = [];
// history buffers for mouth model smoothing (vector and scalar outputs)
let mouthHistVec = [];
let mouthHistScalar = [];

async function loadMouthModel(){
  if(mouthModel) return mouthModel;
    // Prefer multi-class mouth classifier first
    try{
      mouthModel = await tf.loadLayersModel('model/mouth_classifier_model/model.json');
      mouthModel._yawnBinary = false;
      console.log('Loaded multi-class mouth model');
      return mouthModel;
    }catch(_){ /* ignore */ }
    // Fallback to binary yawn model if present
    try{
      mouthModel = await tf.loadLayersModel('model/yawn_model/model.json');
      mouthModel._yawnBinary = true;
      console.log('Loaded binary yawn model');
    }catch(e){
      console.warn('Mouth CNN not available:', e);
      mouthModel = null;
    }
    return mouthModel;
}

function cropMouthFromCanvas(box, canvasEl){
  const cw = canvasEl.width, ch = canvasEl.height;
  const x = clamp((box.cx - box.w/2) * cw, 0, cw-1);
  const y = clamp((box.cy - box.h/2) * ch, 0, ch-1);
  const w = clamp(box.w * cw, 4, cw);
  const h = clamp(box.h * ch, 4, ch);
  const off = cropMouthFromCanvas._off || (cropMouthFromCanvas._off = document.createElement('canvas'));
  off.width = MOUTH_W; off.height = MOUTH_H;
  const octx = off.getContext('2d');
  octx.drawImage(canvasEl, x, y, w, h, 0, 0, MOUTH_W, MOUTH_H);
  const img = octx.getImageData(0,0,MOUTH_W,MOUTH_H);
  // convert to float32 [H,W,3]
  const buf = new Float32Array(MOUTH_W * MOUTH_H * 3);
  for(let i=0, j=0;i<img.data.length;i+=4,j+=3){
    buf[j]   = img.data[i] / 255.0;
    buf[j+1] = img.data[i+1] / 255.0;
    buf[j+2] = img.data[i+2] / 255.0;
  }
  return tf.tensor(buf, [MOUTH_H, MOUTH_W, 3]);
}

function smoothMouthPred(vec){
  mouthHistVec.push(vec);
  if(mouthHistVec.length > MOUTH_SMOOTH_WIN) mouthHistVec.shift();
  const sum = mouthHistVec.reduce((a,b)=>a.map((v,i)=>v+b[i]), new Array(vec.length).fill(0));
  return sum.map(v=>v/mouthHistVec.length);
}

function smoothMouthScalar(v){
  mouthHistScalar.push(v);
  if(mouthHistScalar.length > MOUTH_SMOOTH_WIN) mouthHistScalar.shift();
  const s = mouthHistScalar.reduce((a,b)=>a+b, 0) / mouthHistScalar.length;
  return s;
}

function landmarksEyeBox(lm, eye){
  // Build a small box around eye using its corner landmarks
  const left = lm[eye.left];
  const right = lm[eye.right];
  const upper = {
    x:(lm[eye.upper[0]].x + lm[eye.upper[1]].x)/2,
    y:(lm[eye.upper[0]].y + lm[eye.upper[1]].y)/2,
  };
  const lower = {
    x:(lm[eye.lower[0]].x + lm[eye.lower[1]].x)/2,
    y:(lm[eye.lower[0]].y + lm[eye.lower[1]].y)/2,
  };
  const cx = (left.x + right.x)/2;
  const cy = (upper.y + lower.y)/2;
  const w = Math.abs(right.x - left.x);
  const h = Math.abs(lower.y - upper.y) * 2.2; // slightly taller than lids
  const padW = w * EYE_PAD;
  const padH = h * EYE_PAD;
  return { cx, cy, w: padW, h: padH };
}

function cropEyeToTensor(sourceImage, box){
  // box in normalized coords (0..1)
  const cw = els.canvas.width, ch = els.canvas.height;
  const x = clamp((box.cx - box.w/2) * cw, 0, cw-1);
  const y = clamp((box.cy - box.h/2) * ch, 0, ch-1);
  const w = clamp(box.w * cw, 4, cw);
  const h = clamp(box.h * ch, 4, ch);

  // Use an offscreen canvas to draw the crop
  const off = cropEyeToTensor._off || (cropEyeToTensor._off = document.createElement('canvas'));
  off.width = EYE_W; off.height = EYE_H;
  const octx = off.getContext('2d');
  // Draw from the main canvas (which already has the frame rendered) to ensure coords align
  octx.drawImage(sourceImage, x, y, w, h, 0, 0, EYE_W, EYE_H);
  // Grayscale and normalize to [0,1]
  const imgData = octx.getImageData(0,0,EYE_W,EYE_H);
  const data = imgData.data;
  const gray = new Float32Array(EYE_W * EYE_H);
  for(let i=0, j=0; i<data.length; i+=4, j++){
    const r=data[i], g=data[i+1], b=data[i+2];
    gray[j] = (0.299*r + 0.587*g + 0.114*b) / 255;
  }
  const t = tf.tensor(gray, [EYE_H, EYE_W, 1]);
  return t;
}

function flipTensorLeftRight(x){
  // x: [H,W,1]
  return tf.tidy(()=>x.reverse(1));
}

function smoothProb(side, p){
  const h = probHist[side];
  h.push(p);
  if(h.length > CNN_SMOOTH_WIN) h.shift();
  const avg = h.reduce((a,b)=>a+b, 0) / h.length;
  return avg;
}

// Eye Aspect Ratio like metric using MediaPipe FaceMesh indices
// We'll use eye landmark sets approximated from MediaPipe indexes
const LEFT_EYE = { // key pairs approximating vertical distances and horizontal width
  upper:[386,385], lower:[374,380], // near top/bottom eyelid center-ish
  left: 263, right: 362
};
const RIGHT_EYE = {
  upper:[159,158], lower:[145,153],
  left: 133, right: 33
};

function dist2D(a,b){ const dx=a.x-b.x, dy=a.y-b.y; return Math.hypot(dx,dy); }

function earForEye(landmarks, eye){
  const u = {
    x:(landmarks[eye.upper[0]].x + landmarks[eye.upper[1]].x)/2,
    y:(landmarks[eye.upper[0]].y + landmarks[eye.upper[1]].y)/2,
  };
  const l = {
    x:(landmarks[eye.lower[0]].x + landmarks[eye.lower[1]].x)/2,
    y:(landmarks[eye.lower[0]].y + landmarks[eye.lower[1]].y)/2,
  };
  const left = landmarks[eye.left];
  const right = landmarks[eye.right];
  const vert = dist2D(u,l);
  const horiz = dist2D(left,right);
  if (horiz <= 1e-6) return 0;
  return vert / horiz; // smaller when eye closed
}

function drawEyeMarkers(landmarks, eye, color){
  const cw = els.canvas.width, ch = els.canvas.height;
  const u = {
    x:(landmarks[eye.upper[0]].x + landmarks[eye.upper[1]].x)/2,
    y:(landmarks[eye.upper[0]].y + landmarks[eye.upper[1]].y)/2,
  };
  const l = {
    x:(landmarks[eye.lower[0]].x + landmarks[eye.lower[1]].x)/2,
    y:(landmarks[eye.lower[0]].y + landmarks[eye.lower[1]].y)/2,
  };
  const left = landmarks[eye.left];
  const right = landmarks[eye.right];
  const pts = [u,l,left,right];
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2;
  // corners line
  ctx.beginPath();
  ctx.moveTo(left.x*cw, left.y*ch);
  ctx.lineTo(right.x*cw, right.y*ch);
  ctx.stroke();
  // vertical line
  ctx.beginPath();
  ctx.moveTo(u.x*cw, u.y*ch);
  ctx.lineTo(l.x*cw, l.y*ch);
  ctx.stroke();
  // dots
  for(const p of pts){
    ctx.beginPath();
    ctx.arc(p.x*cw, p.y*ch, 3, 0, Math.PI*2);
    ctx.fill();
  }
  ctx.restore();
}

// ===================== Gaze estimation =====================
// With refineLandmarks=true, FaceMesh returns iris landmarks.
// Common convention:
//   Left iris: 468..472 (5 points), Right iris: 473..477 (5 points)
const LEFT_IRIS_IDX = [468, 469, 470, 471, 472];
const RIGHT_IRIS_IDX = [473, 474, 475, 476, 477];

function averagePoint(points){
  if(!points || !points.length) return null;
  let sx=0, sy=0; for(const p of points){ sx+=p.x; sy+=p.y; }
  return { x: sx/points.length, y: sy/points.length };
}

function irisCenter(lm, irisIdx){
  const pts = [];
  for(const i of irisIdx){ if(lm[i]) pts.push(lm[i]); }
  if(pts.length < 3) return null;
  return averagePoint(pts);
}

// Compute normalized gaze vector for one eye in eye-local coordinates.
// Returns { gx, gy, iris: {x,y} } where gx,gy in [-1,1] (0,0 is centered).
function gazeForEye(lm, eye, irisIdx){
  const left = lm[eye.left];
  const right = lm[eye.right];
  const upper = {
    x:(lm[eye.upper[0]].x + lm[eye.upper[1]].x)/2,
    y:(lm[eye.upper[0]].y + lm[eye.upper[1]].y)/2,
  };
  const lower = {
    x:(lm[eye.lower[0]].x + lm[eye.lower[1]].x)/2,
    y:(lm[eye.lower[0]].y + lm[eye.lower[1]].y)/2,
  };
  if(!(left && right && upper && lower)) return null;
  const iris = irisCenter(lm, irisIdx);
  if(!iris) return null;

  // Build eye-local axes
  const vx = { x: right.x - left.x, y: right.y - left.y };
  const wx = Math.hypot(vx.x, vx.y);
  if(wx <= 1e-6) return null;
  const ux = { x: vx.x / wx, y: vx.y / wx }; // horizontal unit
  const midUL = { x: (upper.x + lower.x)/2, y: (upper.y + lower.y)/2 };
  const vy = { x: lower.x - upper.x, y: lower.y - upper.y };
  const hy = Math.hypot(vy.x, vy.y);
  const uy = hy>1e-6 ? { x: vy.x / hy, y: vy.y / hy } : { x: -ux.y, y: ux.x }; // vertical unit
  const halfH = Math.max(1e-6, hy * 0.5);

  // Project iris to axes
  const relIrisFromLeft = { x: iris.x - left.x, y: iris.y - left.y };
  const relIrisFromMid = { x: iris.x - midUL.x, y: iris.y - midUL.y };
  const projX = relIrisFromLeft.x * ux.x + relIrisFromLeft.y * ux.y; // 0..wx
  const projY = relIrisFromMid.x * uy.x + relIrisFromMid.y * uy.y;  // -halfH..+halfH (approx)

  let gx = (projX / wx) * 2 - 1; // normalize: 0..1 -> -1..1
  let gy = clamp(projY / halfH, -1, 1); // -1..1
  // Clamp just in case
  gx = clamp(gx, -1, 1);
  gy = clamp(gy, -1, 1);
  return { gx, gy, iris };
}

function boo(){
  // Play Siren.mp3 (project root). If not available, fallback to speech/oscillator.
  try{
    if(sirenAudio && !sirenAudio.paused){
      // already playing
      return;
    }
    // load persistent audio element if possible
    sirenAudio = new Audio('Siren.mp3');
    sirenAudio.loop = true;
    sirenAudio.preload = 'auto';
    // try to play (may be blocked without user gesture in some browsers)
    const p = sirenAudio.play();
    if(p && p.catch){ p.catch(()=>{
      // fallback to speech if playback blocked
      try{ const utter = new SpeechSynthesisUtterance('Boo!'); speechSynthesis.speak(utter);}catch(_){/*ignore*/}
    }); }
    // Ensure we don't play indefinitely — stop after SIREN_MAX_MS
    if(sirenTimeout) clearTimeout(sirenTimeout);
    sirenTimeout = setTimeout(()=>{
      try{ if(sirenAudio){ sirenAudio.pause(); sirenAudio.currentTime = 0; } }catch(_){ }
      sirenAudio = null;
      sirenTimeout = null;
    }, SIREN_MAX_MS);
    return;
  }catch(e){
    // fallback to speech/short oscillator
    try{
      const utter = new SpeechSynthesisUtterance('Boo!');
      utter.pitch = 0.8; utter.rate = 0.9; utter.volume = 1;
      speechSynthesis.speak(utter);
    }catch(e2){
      try{
        const ac = new (window.AudioContext || window.webkitAudioContext)();
        const o = ac.createOscillator();
        const g = ac.createGain();
        o.type = 'square';
        o.frequency.setValueAtTime(220, ac.currentTime);
        o.frequency.linearRampToValueAtTime(110, ac.currentTime+0.25);
        g.gain.setValueAtTime(0.0001, ac.currentTime);
        g.gain.exponentialRampToValueAtTime(0.5, ac.currentTime+0.02);
        g.gain.exponentialRampToValueAtTime(0.0001, ac.currentTime+0.5);
        o.connect(g).connect(ac.destination);
        o.start();
        o.stop(ac.currentTime+0.5);
      }catch(_){ }
    }
  }
}


function setState(text){ els.state.textContent = text; }

function drawPumpkinOverFace(landmarks){
  // Determine face bounding box and orientation using a few key points
  const leftCheek = landmarks[234];
  const rightCheek = landmarks[454];
  const chin = landmarks[152];
  const forehead = landmarks[10];

  if(!leftCheek || !rightCheek || !chin || !forehead) return;

  const center = {
    x: (leftCheek.x + rightCheek.x)/2,
    y: (chin.y + forehead.y)/2,
  };
  const width = dist2D(leftCheek, rightCheek);
  const height = dist2D(forehead, chin);
  const angle = Math.atan2(rightCheek.y - leftCheek.y, rightCheek.x - leftCheek.x);

  const cw = els.canvas.width;
  const ch = els.canvas.height;

  // Default alpha applied by caller via ctx.globalAlpha; if needed, caller sets it.
  ctx.save();
  ctx.translate(center.x * cw, center.y * ch);
  ctx.rotate(angle);
  const scale = 1.35; // exaggerate a bit to cover face
  const w = width * cw * scale;
  const h = height * ch * scale;
  ctx.drawImage(pumpkin, -w/2, -h*0.55, w, h*1.1);
  ctx.restore();
}

function drawFaceBox(landmarks, index, highlight=false, state='neutral'){
  // Compute tight bounding box
  let minX=Infinity, minY=Infinity, maxX=-Infinity, maxY=-Infinity;
  for(const p of landmarks){
    if(p.x<minX) minX=p.x; if(p.y<minY) minY=p.y;
    if(p.x>maxX) maxX=p.x; if(p.y>maxY) maxY=p.y;
  }
  const cw = els.canvas.width, ch = els.canvas.height;
  const x = minX*cw, y=minY*ch, w=(maxX-minX)*cw, h=(maxY-minY)*ch;

  ctx.save();
  ctx.lineWidth = highlight? 3 : 1.5;
  let stroke = 'rgba(255,255,255,0.7)';
  let fill = 'rgba(255,255,255,0.08)';
  if(state==='closed'){
    stroke = 'rgba(255,82,82,0.95)'; // red
    fill = 'rgba(255,82,82,0.15)';
  } else if(state==='open'){
    stroke = 'rgba(0,230,118,0.95)'; // green
    fill = 'rgba(0,230,118,0.12)';
  }
  if(highlight){
    // emphasize highlight by blending with gold
    stroke = 'rgba(255,204,0,0.95)';
  }
  ctx.strokeStyle = stroke;
  ctx.fillStyle = fill;
  ctx.beginPath();
  ctx.roundRect(x,y,w,h, 8);
  ctx.fill();
  ctx.stroke();
  // Label
  const label = `face ${index+1}`;
  ctx.font = '14px system-ui, sans-serif';
  ctx.fillStyle = '#111';
  ctx.strokeStyle = stroke;
  const pad=6;
  const tw = ctx.measureText(label).width;
  ctx.fillStyle = stroke;
  ctx.fillRect(x, Math.max(0,y-22), tw+pad*2, 20);
  ctx.fillStyle = '#111';
  ctx.fillText(label, x+pad, Math.max(14,y-6));
  ctx.restore();

  return {x,y,w,h};
}

async function initFaceMesh(){
  return new Promise((resolve)=>{
    // MediaPipe FaceMesh is exposed under the FaceMesh namespace; use FaceMesh.FaceMesh()
    fm = new FaceMesh.FaceMesh({ locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}` });
    fm.setOptions({
      maxNumFaces: 3,
      refineLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    fm.onResults(onResults);
    resolve();
  });
}

async function startCamera(){
  if(running) return;

  // Match canvas to displayed size
  const rect = els.video.getBoundingClientRect();
  let w = Math.floor(rect.width);
  let h = Math.floor(rect.height);
  // Fallback if layout hasn't sized the video yet
  if(!w || w < 2 || !h || h < 2){
    w = 640; h = 480;
  }
  els.canvas.width = w;
  els.canvas.height = h;

  await initFaceMesh();
  initMovementDetection(); // Initialize movement detection on camera start

  // Preload CNN if opted in
  if(els.useCnn?.checked){
    els.msg.textContent = 'Loading eye CNN…';
    await loadEyeModel();
    els.msg.textContent = '';
  }

  // Use MediaPipe's Camera helper to send frames to FaceMesh
  camera = new Camera(els.video, {
    onFrame: async () => {
      lastTimestamp = performance.now();
      await fm.send({image: els.video});
    },
    width: w,
    height: h,
  });
  await camera.start();
  running = true;
  els.startBtn.disabled = true;
  els.stopBtn.disabled = false;
  setState('running');
  // Initialize drowsy counter from storage and UI
  try{
    const loaded = parseInt(localStorage.getItem('drowsyCount') || '0', 10);
    if(!isNaN(loaded)) drowsyCount = loaded;
    const el = document.getElementById('drowsyCount');
    if(el) el.textContent = String(drowsyCount);
  }catch(_){ /* ignore */ }
}

function stopCamera(){
  if(!running) return;
  try{ camera.stop(); }catch(_){ }
  running = false;
  setState('stopped');
  els.startBtn.disabled = false;
  els.stopBtn.disabled = true;
  // Stop siren audio if playing when camera is closed
  try{
    if(sirenTimeout){ clearTimeout(sirenTimeout); sirenTimeout = null; }
    if(sirenAudio){ sirenAudio.pause(); try{ sirenAudio.currentTime = 0; }catch(_){ } sirenAudio = null; }
  }catch(_){ }
}

function onResults(results){
  // Check for alerts (even without face detection)
  checkAndTriggerAlert();
  
  // Draw video background
  ctx.save();
  ctx.clearRect(0,0,els.canvas.width, els.canvas.height);
  ctx.drawImage(results.image, 0,0, els.canvas.width, els.canvas.height);

  const faces = results.multiFaceLandmarks || [];
  if(faces.length){
    // Determine target face = closest to canvas center
    const cw = els.canvas.width, ch = els.canvas.height;
    const centerX = 0.5, centerY = 0.5; // normalized
    let targetIdx = 0, bestScore = Infinity;
    const centers = faces.map(lm => {
      let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;
      for(const p of lm){ if(p.x<minX)minX=p.x; if(p.y<minY)minY=p.y; if(p.x>maxX)maxX=p.x; if(p.y>maxY)maxY=p.y; }
      const cx = (minX+maxX)/2, cy=(minY+maxY)/2;
      return {cx,cy};
    });
    if(targetIndexUser != null && targetIndexUser < faces.length){
      targetIdx = targetIndexUser;
    } else {
      centers.forEach((c, i)=>{
        const dx=c.cx-centerX, dy=c.cy-centerY;
        const d2 = dx*dx+dy*dy;
        if(d2<bestScore){ bestScore=d2; targetIdx=i; }
      });
    }

    // For each face: compute EAR and CNN closed-prob if enabled. Draw box and markers accordingly.
    const thresh = parseFloat(els.threshold.value);
    const useCnn = !!els.useCnn?.checked;
    const cnnThresh = parseFloat(els.cnnThresh?.value || '0.6');
    const haveModel = !!eyeModel;
    lastFaceBoxes = faces.map((lm, i)=>{
      const leftEAR = earForEye(lm, LEFT_EYE);
      const rightEAR = earForEye(lm, RIGHT_EYE);
      const earAvg = (leftEAR + rightEAR)/2;
      let isClosed = earAvg < thresh;
      let state = isClosed ? 'closed' : 'open';
      let color = isClosed ? 'rgba(255,82,82,0.95)' : 'rgba(0,230,118,0.95)';

      // If CNN is enabled and model is loaded, compute per-eye closed probabilities.
      if(useCnn && haveModel){
        try{
          // Compute eye boxes and crop from the source image (results.image is same as video frame drawn)
          const boxL = landmarksEyeBox(lm, LEFT_EYE);
          const boxR = landmarksEyeBox(lm, RIGHT_EYE);
          const tL = cropEyeToTensor(els.canvas, boxL);
          const tRraw = cropEyeToTensor(els.canvas, boxR);
          const tR = flipTensorLeftRight(tRraw); // flip right eye for consistency
          const batch = tf.stack([tL, tR], 0); // [2, H, W, 1]
          const preds = eyeModel.predict(batch);
          const probs = Array.from(preds.dataSync()); // assuming shape [2,1] or [2]
          tf.dispose([tL, tRraw, tR, batch, preds]);
          const pL = smoothProb('L', probs[0]);
          const pR = smoothProb('R', probs[1] ?? probs[0]);

          // Update UI if this is target face; the next block sets per-target UI anyway, but we display for each just last eval
          if(i===targetIdx){
            if(els.cnnL) els.cnnL.textContent = pL.toFixed(2);
            if(els.cnnR) els.cnnR.textContent = pR.toFixed(2);
          }

          const closedByCnn = ((pL + pR)/2) >= cnnThresh;
          isClosed = closedByCnn; // override EAR with CNN when enabled
          state = isClosed ? 'closed' : 'open';
          color = isClosed ? 'rgba(255,82,82,0.95)' : 'rgba(0,230,118,0.95)';
        }catch(err){
          // If anything fails, fall back to EAR
          console.warn('CNN eye inference failed:', err);
        }
      }
      drawEyeMarkers(lm, LEFT_EYE, color);
      drawEyeMarkers(lm, RIGHT_EYE, color);
      return drawFaceBox(lm, i, i===targetIdx, state);
    });

  // Compute drowsiness for target
    const lm = faces[targetIdx];
    const left = earForEye(lm, LEFT_EYE);
    const right = earForEye(lm, RIGHT_EYE);
    const ear = (left+right)/2;
    els.earL.textContent = left.toFixed(3);
    els.earR.textContent = right.toFixed(3);
    els.targetFace.textContent = `${targetIdx+1} / ${faces.length}`;

    const now = performance.now();
    const dt = lastTimestamp ? (now - lastTimestamp)/1000 : 0; // seconds
    // Decide closed for accumulation based on CNN if enabled and available, else EAR
    let targetClosed = false;
    if(els.useCnn?.checked && eyeModel){
      // Recompute quick CNN probs for the target only to decide accumulation (ensure values exist even if above loop didn't run or target changed)
      try{
        const boxL = landmarksEyeBox(lm, LEFT_EYE);
        const boxR = landmarksEyeBox(lm, RIGHT_EYE);
  const tL = cropEyeToTensor(els.canvas, boxL);
  const tRraw = cropEyeToTensor(els.canvas, boxR);
        const tR = flipTensorLeftRight(tRraw);
        const batch = tf.stack([tL, tR], 0);
        const preds = eyeModel.predict(batch);
        const probs = Array.from(preds.dataSync());
        tf.dispose([tL, tRraw, tR, batch, preds]);
        const pL = smoothProb('L', probs[0]);
        const pR = smoothProb('R', probs[1] ?? probs[0]);
        if(els.cnnL) els.cnnL.textContent = pL.toFixed(2);
        if(els.cnnR) els.cnnR.textContent = pR.toFixed(2);
        const cnnThresh = parseFloat(els.cnnThresh.value);
        targetClosed = ((pL + pR)/2) >= cnnThresh;
      }catch(err){
        console.warn('CNN target inference failed:', err);
        targetClosed = ear < thresh;
      }
    } else {
      targetClosed = ear < thresh;
    }

    if(targetClosed){
      closedAccum += dt;
      lastEyesClosedAtMs = now; // track last closed time
    } else {
      closedAccum = Math.max(0, closedAccum - dt*0.5);
    }
  const closedForDisplaySec = parseFloat(closedAccum.toFixed(2));
  els.closedTime.textContent = `${closedForDisplaySec.toFixed(2)}s`;

    const need = parseFloat(els.duration.value);
    const wasDrowsy = drowsy;
    drowsy = closedAccum >= need;
    if(drowsy && !wasDrowsy){
      els.msg.textContent = 'Drowsiness detected!';
      boo();
    }
    if(!drowsy && wasDrowsy){
      els.msg.textContent = '';
    }

  // Draw pumpkin overlay.
    // When using CNN: do NOT show on quick blinks. Show only after eyes have been
    // closed for at least PUMPKIN_CNN_SHOW_DELAY_SEC. After reopening, keep showing
    // only if it was already visible and we're within hideDelay.
    const hideDelaySec = parseFloat(els.hideDelay.value);
    const sinceClosedSec = (now - lastEyesClosedAtMs)/1000;
    let showPumpkin = false;
    if(els.useCnn?.checked){
      if(targetClosed){
        // Use the same rounded value shown in UI to avoid perceived early display
        if(closedForDisplaySec >= PUMPKIN_CNN_SHOW_DELAY_SEC){
          showPumpkin = true;
          pumpkinShowing = true; // mark as visible for subsequent hideDelay
        }
      } else {
        // Eyes are open: only keep showing if it was previously visible and within hide delay
        if(pumpkinShowing && sinceClosedSec <= hideDelaySec){
          showPumpkin = true;
        } else {
          pumpkinShowing = false;
        }
      }
    } else {
      if (ear < thresh || sinceClosedSec <= hideDelaySec) {
        showPumpkin = true;
      }
    }
    // Set pumpkin visibility target; actual drawing done after other overlays
    pumpkinAlphaTarget = showPumpkin ? 1.0 : 0.0;

    // Gaze detection (experimental)
    if(els.useGaze?.checked){
      // Compute per-eye gaze and average
      const gL = gazeForEye(lm, LEFT_EYE, LEFT_IRIS_IDX);
      const gR = gazeForEye(lm, RIGHT_EYE, RIGHT_IRIS_IDX);
      if(gL && gR){
        // Average gaze; flip right eye horizontal so both are in the same screen orientation
        // Note: We already normalized by eye corners, so directions align; no flip needed here.
        let gx = (gL.gx + gR.gx) * 0.5;
        let gy = (gL.gy + gR.gy) * 0.5;
        // Smooth
        gazeHist.push({gx, gy});
        if(gazeHist.length > GAZE_SMOOTH_WIN) gazeHist.shift();
        const avg = gazeHist.reduce((a,b)=>({gx:a.gx+b.gx, gy:a.gy+b.gy}), {gx:0, gy:0});
        gx = avg.gx / gazeHist.length;
        gy = avg.gy / gazeHist.length;
        if(els.gazeX) els.gazeX.textContent = gx.toFixed(2);
        if(els.gazeY) els.gazeY.textContent = gy.toFixed(2);

        // Handle calibration sampling window (1s)
        if(gazeCalibSampling){
          const ms = performance.now();
          gazeCalibSampling.samples.push({gx, gy});
          if(ms - gazeCalibSampling.startMs >= 1000){
            const arr = gazeCalibSampling.samples;
            if(arr.length >= 8){
              // compute mean and stddev
              let sumX=0,sumY=0; for(const s of arr){ sumX+=s.gx; sumY+=s.gy; }
              const mx = sumX/arr.length, my = sumY/arr.length;
              let vX=0,vY=0; for(const s of arr){ vX+=(s.gx-mx)*(s.gx-mx); vY+=(s.gy-my)*(s.gy-my);} 
              const sx = Math.max(0.05, Math.sqrt(vX/arr.length));
              const sy = Math.max(0.05, Math.sqrt(vY/arr.length));
              gazeCalib = { enabled: !!els.gazeCalibEnable?.checked, gx0: mx, gy0: my, sx, sy, ts: ms };
              if(els.msg) els.msg.textContent = 'Gaze calibrated';
            } else {
              if(els.msg) els.msg.textContent = 'Calibration failed: not enough samples';
            }
            gazeCalibSampling = null;
          }
        }

        // Head roll gating: reuse cheek angle (computed below too)
        const leftCheekG = lm[234];
        const rightCheekG = lm[454];
        let roll = 0;
        if(leftCheekG && rightCheekG){
          roll = Math.atan2(rightCheekG.y - leftCheekG.y, rightCheekG.x - leftCheekG.x);
        }
        const rollAbs = Math.abs(roll);
        const rollGate = rollAbs < (10 * Math.PI/180); // only when head not notably rolled
        const thr = parseFloat(els.gazeThreshold?.value || '0.45');
        let driftNow;
        const useCal = (!!els.gazeCalibEnable?.checked) && !!gazeCalib && gazeCalib.sx>0 && gazeCalib.sy>0;
        if(useCal){
          const zX = Math.abs(gx - gazeCalib.gx0) / Math.max(0.05, gazeCalib.sx);
          const zY = Math.abs(gy - gazeCalib.gy0) / Math.max(0.05, gazeCalib.sy);
          const k = parseFloat(els.gazeSigmaK?.value || String(GAZE_SIGMA_K));
          driftNow = (zX >= k || zY >= k) && !targetClosed && rollGate;
          // Online auto-calibration: when not drifting, update baseline and noise with EMA
          if(!driftNow && !targetClosed && rollGate){
            const alpha = 0.02; // center learning rate
            const beta = 0.02;  // variance learning rate
            // Update mean
            gazeCalib.gx0 = (1 - alpha) * gazeCalib.gx0 + alpha * gx;
            gazeCalib.gy0 = (1 - alpha) * gazeCalib.gy0 + alpha * gy;
            // Update variance proxies
            const dx = gx - gazeCalib.gx0;
            const dy = gy - gazeCalib.gy0;
            const s2x = (gazeCalib.sx * gazeCalib.sx) * (1 - beta) + beta * (dx * dx);
            const s2y = (gazeCalib.sy * gazeCalib.sy) * (1 - beta) + beta * (dy * dy);
            gazeCalib.sx = Math.max(0.03, Math.sqrt(s2x));
            gazeCalib.sy = Math.max(0.03, Math.sqrt(s2y));
          }
        } else {
          driftNow = (Math.abs(gx) >= thr || Math.abs(gy) >= thr) && !targetClosed && rollGate;
        }
        const nowMs = now;
        if(driftNow){
          if(!gazeHoldStartMs) gazeHoldStartMs = nowMs;
          const held = nowMs - gazeHoldStartMs;
          if(held >= GAZE_HOLD_MS){
            gazeActive = true;
          }
          // Update hold progress UI
          if(els.gazeHoldProg){
            const progSec = held / 1000; // allow progress to exceed target
            els.gazeHoldProg.textContent = progSec.toFixed(1);
          }
        } else {
          gazeHoldStartMs = 0;
          gazeActive = false;
          if(els.gazeHoldProg) els.gazeHoldProg.textContent = '0.0';
        }
        if(els.gazeDrift) els.gazeDrift.textContent = gazeActive ? 'on' : 'off';
        if(els.gazeHoldTarget) els.gazeHoldTarget.textContent = (GAZE_HOLD_MS/1000).toFixed(1);

        // Rising edge: trigger skeleton head pop once per cooldown
        if(gazeActive && !gazePrevActive && !skullAnim && (nowMs - lastSkullTriggerMs) > SKULL_COOLDOWN_MS){
          skullAnim = { inStartMs: nowMs, outStartMs: null };
          lastSkullTriggerMs = nowMs;
        }

        gazePrevActive = gazeActive;

        // Draw small iris dots for visualization
        const cw = els.canvas.width, ch = els.canvas.height;
        ctx.save();
        ctx.fillStyle = gazeActive ? 'rgba(255,204,0,0.95)' : 'rgba(255,255,255,0.85)';
        for(const g of [gL, gR]){
          if(!g || !g.iris) continue;
          ctx.beginPath();
          ctx.arc(g.iris.x * cw, g.iris.y * ch, gazeActive ? 3.3 : 2.6, 0, Math.PI*2);
          ctx.fill();
        }
        ctx.restore();
  // Keep showing the live progress even after activation
      } else {
        // No gaze available this frame
        if(els.gazeX) els.gazeX.textContent = '-';
        if(els.gazeY) els.gazeY.textContent = '-';
        if(els.gazeDrift) els.gazeDrift.textContent = 'off';
        if(els.gazeHoldProg) els.gazeHoldProg.textContent = '0.0';
        if(els.gazeHoldTarget) els.gazeHoldTarget.textContent = (GAZE_HOLD_MS/1000).toFixed(1);
        gazeHoldStartMs = 0; gazeActive = false; gazePrevActive = false;
      }
    } else {
      // Reset UI when disabled
      if(els.gazeX) els.gazeX.textContent = '-';
      if(els.gazeY) els.gazeY.textContent = '-';
      if(els.gazeDrift) els.gazeDrift.textContent = 'off';
      if(els.gazeHoldProg) els.gazeHoldProg.textContent = '0.0';
      if(els.gazeHoldTarget) els.gazeHoldTarget.textContent = (GAZE_HOLD_MS/1000).toFixed(1);
      gazeHoldStartMs = 0; gazeActive = false; gazePrevActive = false;
      gazeHist.length = 0;
    }

    // Head tilt detection and skeleton hand trigger for target
    const leftCheek = lm[234];
    const rightCheek = lm[454];
    if(leftCheek && rightCheek){
      const angle = Math.atan2(rightCheek.y - leftCheek.y, rightCheek.x - leftCheek.x);
      let side = null;
      if(angle > TILT_THRESHOLD_RAD) side = 'right';
      else if(angle < -TILT_THRESHOLD_RAD) side = 'left';
      const nowMs = now;
      // Require tilt + eyes closed to hold for TILT_HOLD_MS before triggering
      const eyesClosed = targetClosed; // targetClosed computed earlier based on CNN/EAR
      if(side && eyesClosed){
        // same side continuing
        if(tiltHoldSide === side){
          // already holding
          if(!tiltHoldStartMs) tiltHoldStartMs = nowMs;
        } else {
          // new side - reset timer
          tiltHoldSide = side;
          tiltHoldStartMs = nowMs;
        }
        const heldMs = nowMs - tiltHoldStartMs;
        if(heldMs >= TILT_HOLD_MS && !handAnim && (nowMs - lastHandTriggerMs) > TILT_COOLDOWN_MS){
          handAnim = { side, startMs: nowMs };
          lastHandTriggerMs = nowMs;
          // Count a tilt event for predictive features
          tiltEventsCounted++;
          // reset holding state
          tiltHoldStartMs = 0;
          tiltHoldSide = null;
        }
      } else {
        // not tilted or eyes open -> reset hold timer
        tiltHoldStartMs = 0;
        tiltHoldSide = null;
      }
    }

  // Draw skeleton hand animation on top
  const bbox = lastFaceBoxes && lastFaceBoxes[targetIdx];
  updateAndDrawHand(now, bbox);
  updateAndDrawSkull(now, bbox);

    // Smoothly blend pumpkin alpha toward target to create gradual fade/decay
    try{
      // simple per-frame lerp (frame-rate independent enough for UX)
      const blend = 0.12; // larger -> faster fade
      pumpkinAlpha += (pumpkinAlphaTarget - pumpkinAlpha) * blend;
      pumpkinAlpha = clamp01(pumpkinAlpha);
      // Rising edge detection: increment counter when pumpkin becomes visible
      const visibleNow = pumpkinAlpha >= 0.18;
      if(visibleNow && !prevPumpkinVisible){
        drowsyCount = (drowsyCount || 0) + 1;
        try{ localStorage.setItem('drowsyCount', String(drowsyCount)); }catch(_){ }
        const el = document.getElementById('drowsyCount'); if(el) el.textContent = String(drowsyCount);
      }
      prevPumpkinVisible = visibleNow;
      // Draw pumpkin with current alpha if visible enough
      if(pumpkinAlpha > 0.01){
        ctx.save();
        ctx.globalAlpha = pumpkinAlpha * 0.92; // respect existing art alpha
        drawPumpkinOverFace(lm);
        ctx.restore();
      }
    }catch(_){ /* drawing guard */ }

    // Mouth classification (optional) -> trigger eyePop on yawn
    const yawnCooldownMs = parseFloat(els.yawnCooldown.value) * 1000;
    const yawnMinDurMs = (parseFloat(els.yawnMinDur?.value || '0.5')) * 1000;
    const morGate = parseFloat(els.yawnMorMin?.value || '0.35');
    let mouthClass = null;
    if(els.useMouthCnn?.checked && mouthModel){
      try{
        // approximate mouth box from landmarks
        const top = lm[13], bottom = lm[14];
        const left = lm[61], right = lm[291];
        const cx = (left.x + right.x)/2; const cy = (top.y + bottom.y)/2;
        const w = Math.abs(right.x - left.x) * 1.6; const h = Math.abs(bottom.y - top.y) * 2.2;
        const box = { cx, cy, w, h };
        const t = cropMouthFromCanvas(box, els.canvas);
        const batch = tf.tidy(()=>tf.expandDims(t, 0));
        const preds = mouthModel.predict(batch);
        const probs = Array.from(preds.dataSync());
        tf.dispose([t, batch, preds]);

        // Binary yawn model handling (preferred)
        if(mouthModel._yawnBinary || probs.length <= 2){
          // If softmax of 2, assume index 1 corresponds to 'yawn'
          const rawYawnProb = (probs.length === 1) ? probs[0] : (probs[1] ?? 0);
          const yawnProb = smoothMouthScalar(rawYawnProb);
          const thr = parseFloat(els.yawnThresh.value);
          const mor = mouthOpenRatio(lm);
          const gatingOk = (yawnProb >= thr) && (mor >= morGate);
          mouthClass = gatingOk ? 'yawn' : 'no-yawn';
          if(els.mouthPred) els.mouthPred.textContent = `${mouthClass} (p=${yawnProb.toFixed(2)}, mor=${mor.toFixed(2)})`;
          if(gatingOk){
            if(yawnHoldStartMs === 0) yawnHoldStartMs = now;
            const heldMs = now - yawnHoldStartMs;
            if(heldMs >= yawnMinDurMs && !eyePopAnim && now - lastYawnMs > yawnCooldownMs){
              const mc = mouthCenterCanvas(lm);
              eyePopAnim = { startMs: now, x: mc?.cx, y: mc?.cy };
              lastYawnMs = now;
              yawnHoldStartMs = 0;
            }
          } else {
            yawnHoldStartMs = 0;
          }
        } else {
          // Multi-class fallback: ['neutral','open','smile','yawn']
          const smooth = smoothMouthPred(probs);
          const classes = ['neutral','open','smile','yawn'];
          const yawnProb = smooth[3] ?? 0;
          const thr = parseFloat(els.yawnThresh.value);
          const mor = mouthOpenRatio(lm);
          const gatingOk = (yawnProb >= thr) && (mor >= morGate);
          let idx = 0; for(let i=1;i<smooth.length;i++) if(smooth[i] > smooth[idx]) idx = i;
          mouthClass = gatingOk ? 'yawn' : (classes[idx] ?? 'neutral');
          if(els.mouthPred) els.mouthPred.textContent = `${mouthClass} (p=${yawnProb.toFixed(2)}, mor=${mor.toFixed(2)})`;
          if(gatingOk){
            if(yawnHoldStartMs === 0) yawnHoldStartMs = now;
            const heldMs = now - yawnHoldStartMs;
            if(heldMs >= yawnMinDurMs && !eyePopAnim && now - lastYawnMs > yawnCooldownMs){
              const mc = mouthCenterCanvas(lm);
              eyePopAnim = { startMs: now, x: mc?.cx, y: mc?.cy };
              lastYawnMs = now;
              yawnHoldStartMs = 0;
            }
          } else {
            yawnHoldStartMs = 0;
          }
        }
      }catch(err){
        console.warn('Mouth inference error:', err);
      }

      // Draw eye pop animation (if active) during classifier mode as well
      if(eyePopAnim){
        const mcNow = mouthCenterCanvas(lm);
        const px = mcNow?.cx ?? eyePopAnim.x ?? (bbox ? (bbox.x + bbox.w*0.5) : els.canvas.width*0.5);
        const py = mcNow?.cy ?? eyePopAnim.y ?? (bbox ? (bbox.y + bbox.h*0.6) : els.canvas.height*0.6);
        // use a proxy amplitude: if yawnProb was computed, scale by thresholded value; otherwise a fixed medium
        const amp = 0.8; // simple constant; visual only
        drawEyePop(now, px, py, bbox, amp);
      }
    } else {
      // fallback to simple mouth-open ratio
      const mor = mouthOpenRatio(lm);
      const yawnThresh = parseFloat(els.yawnThresh.value);
      if(mor > yawnThresh){
        if(!eyePopAnim && now - lastYawnMs > yawnCooldownMs){
          const mc = mouthCenterCanvas(lm);
          eyePopAnim = { startMs: now, x: mc?.cx, y: mc?.cy };
          lastYawnMs = now;
        }
      }
      if(eyePopAnim){
        const mcNow = mouthCenterCanvas(lm);
        const px = mcNow?.cx ?? eyePopAnim.x ?? (bbox ? (bbox.x + bbox.w*0.5) : els.canvas.width*0.5);
        const py = mcNow?.cy ?? eyePopAnim.y ?? (bbox ? (bbox.y + bbox.h*0.6) : els.canvas.height*0.6);
        drawEyePop(now, px, py, bbox, mor);
      }
    }
  } else {
    els.earL.textContent = '-';
    els.earR.textContent = '-';
    if(els.cnnL) els.cnnL.textContent = '-';
    if(els.cnnR) els.cnnR.textContent = '-';
    els.targetFace.textContent = '-';
    closedAccum = Math.max(0, closedAccum - 0.05);
    els.closedTime.textContent = `${closedAccum.toFixed(2)}s`;
    pumpkinShowing = false; // reset overlay when no faces
    pumpkinAlphaTarget = 0.0;
    prevPumpkinVisible = false;
    lastEyesClosedAtMs = 0;
  }

  // If skeleton head is holding, hide it on the next blink (eyes closed)
  if(skullAnim && skullAnim.outStartMs == null){
    // Use targetClosed from above (if defined in scope). If not available, infer from pumpkinShowing? Here we rely on targetClosed being in scope.
    try{
      if(typeof targetClosed !== 'undefined' && targetClosed){
        skullAnim.outStartMs = performance.now();
      }
    }catch(_){/* ignore */}
  }

  ctx.restore();

  // Apply sleep-profile-based adaptations on each frame (rate-limited inside)
  try{ applyAdaptations(performance.now()); }catch(_){ /* ignore */ }

  // Update predictive buffers and probability once per frame
  try{
    if(predictEnabled){
      const nowMs = performance.now();
      // Ensure current bucket exists
      if(!predSecBucket || (nowMs - predSecBucket.secStartMs) >= 1000){
        // Push previous bucket to history
        if(predSecBucket){ predHistory.push(predSecBucket); }
        // Trim to window
        const cutoff = nowMs - PREDICT_WIN_SEC*1000;
        predHistory = predHistory.filter(b => b.secStartMs >= cutoff);
        predSecBucket = { secStartMs: nowMs, closedFrames: 0, totalFrames: 0, blinks: 0, yawns: 0, tilts: 0, gazeDriftFrames: 0 };
      }
      // Increment current bucket counts
      predSecBucket.totalFrames++;
      if(typeof targetClosed !== 'undefined' && targetClosed) predSecBucket.closedFrames++;
      if(typeof gazeActive !== 'undefined' && gazeActive) predSecBucket.gazeDriftFrames++;
      // Blink detection via closed state transition
      if(typeof targetClosed !== 'undefined'){
        if(targetClosed && !prevClosedForBlink){ blinkStartMs = nowMs; }
        if(!targetClosed && prevClosedForBlink){
          // A blink if closed period < 800ms (quick closure)
          const dur = nowMs - blinkStartMs;
          if(dur > 60 && dur < 800){ predSecBucket.blinks++; }
        }
        prevClosedForBlink = targetClosed;
      }
      // Yawn detection proxy: use eyePopAnim trigger timing
      if(eyePopAnim && (nowMs - eyePopAnim.startMs) < 50){ predSecBucket.yawns++; }
      // Tilt events: counted when handAnim starts; we bump a global counter; reflect here
      if(tiltEventsCounted > 0){ predSecBucket.tilts += tiltEventsCounted; tiltEventsCounted = 0; }

      // Compute probability from history once per ~second
      if((nowMs - predSecBucket.secStartMs) > 800){
        const buckets = predHistory.concat([predSecBucket]);
        // Aggregate features
        const totalFrames = buckets.reduce((a,b)=>a+b.totalFrames, 0) || 1;
        const closedDuty = buckets.reduce((a,b)=>a+b.closedFrames, 0) / totalFrames; // 0..1
        const gazeDuty = buckets.reduce((a,b)=>a+b.gazeDriftFrames, 0) / totalFrames; // 0..1
        const blinksPerMin = (buckets.reduce((a,b)=>a+b.blinks, 0)) * (60 / (buckets.length||1));
        const yawnsPerMin = (buckets.reduce((a,b)=>a+b.yawns, 0)) * (60 / (buckets.length||1));
        const tiltsPerMin = (buckets.reduce((a,b)=>a+b.tilts, 0)) * (60 / (buckets.length||1));
        // Sleep and circadian features
        const { risk: sleepRisk, debtRisk, circ } = computeSleepRisk(new Date());
        const lastSleepHrs = (sleepProfile.lastSleepHours != null)? sleepProfile.lastSleepHours : sleepProfile.idealHours;
        const sleepDebtHrs = Math.max(0, (sleepProfile.idealHours||7.5) - lastSleepHrs);
        // Time of day features
        const d = new Date();
        const hour = d.getHours() + d.getMinutes()/60;
        const sinH = Math.sin(2*Math.PI*hour/24);
        const cosH = Math.cos(2*Math.PI*hour/24);
        // Simple linear model -> sigmoid
        // Weights are heuristic for hackathon purposes
        const w = {
          bias: -1.4,
          closedDuty: 2.2,
          gazeDuty: 1.0,
          blinksPerMin: 0.02,   // frequent blinking can precede drowsiness
          yawnsPerMin: 0.12,
          tiltsPerMin: 0.04,
          sleepDebtHrs: 0.18,
          sleepRisk: 0.9,
          circ: 0.6,
          sinH: 0.2,
          cosH: -0.05,
        };
        const z = w.bias
          + w.closedDuty*closedDuty
          + w.gazeDuty*gazeDuty
          + w.blinksPerMin*blinksPerMin
          + w.yawnsPerMin*yawnsPerMin
          + w.tiltsPerMin*tiltsPerMin
          + w.sleepDebtHrs*sleepDebtHrs
          + w.sleepRisk*sleepRisk
          + w.circ*circ
          + w.sinH*sinH + w.cosH*cosH;
        const prob = 1/(1+Math.exp(-z));
        if(els.predictProb) els.predictProb.textContent = prob.toFixed(2);
        if(prob >= predictThresh){
          // Non-intrusive message; avoid spamming boo
          if(els.msg && els.msg.textContent === ''){
            els.msg.textContent = 'Predictive: drowsiness risk elevated';
          }
        }
      }
    }
  }catch(_){ /* ignore predictive errors */ }
}

// ===================== Movement Detection & Microgame Safety System =====================
let isMoving = false;
let motionSamples = [];
const MOTION_SAMPLE_SIZE = 10;
const MOTION_THRESHOLD = 0.5; // m/s² threshold for "moving"

// Drowsiness escalation state
let drowsinessScore = 0;
const DROWSINESS_ALERT_THRESHOLD = 3; // trigger alert
let alertActive = false;
let alertStartTime = 0;
let alertAudio = null;

// Microgame state
let microgameActive = false;
let microgameStartTime = 0;
let microgameTargets = [];
let microgameScore = 0;
const MICROGAME_DURATION = 10000; // 10 seconds
const MICROGAME_PASS_THRESHOLD = 8; // need 8/10 correct

function initMovementDetection() {
  if (window.DeviceMotionEvent) {
    window.addEventListener('devicemotion', (event) => {
      if (event.acceleration) {
        const acc = event.acceleration;
        const magnitude = Math.sqrt(
          (acc.x || 0) ** 2 + 
          (acc.y || 0) ** 2 + 
          (acc.z || 0) ** 2
        );
        
        motionSamples.push(magnitude);
        if (motionSamples.length > MOTION_SAMPLE_SIZE) {
          motionSamples.shift();
        }
        
        // Average acceleration over samples
        const avgMotion = motionSamples.reduce((a, b) => a + b, 0) / motionSamples.length;
        isMoving = avgMotion > MOTION_THRESHOLD;
        
        // Update UI
        const moveStatus = document.getElementById('moveStatus');
        if (moveStatus) {
          moveStatus.textContent = isMoving ? 'Moving' : 'Stationary';
          moveStatus.style.color = isMoving ? '#ff6b6b' : '#51cf66';
        }
      }
    });
  } else {
    console.warn('DeviceMotion API not available - using manual toggle');
  }
}

function updateDrowsinessScore() {
  drowsinessScore = 0;
  
  // Score based on current state
  if (closedAccum > 2.0) drowsinessScore += 2;
  else if (closedAccum > 1.0) drowsinessScore += 1;
  
  if (gazeActive) drowsinessScore += 1;
  if (yawnCooldownUntil > Date.now() - 5000) drowsinessScore += 1;
  if (lastHandTriggerMs > Date.now() - 5000) drowsinessScore += 1;
  
  // Predictive risk boost
  const predictProb = parseFloat(els.predictProb?.textContent || '0');
  if (predictProb > 0.7) drowsinessScore += 2;
  else if (predictProb > 0.5) drowsinessScore += 1;
  
  return drowsinessScore;
}

function checkAndTriggerAlert() {
  const score = updateDrowsinessScore();
  
  // Update UI
  const drowsyScoreEl = document.getElementById('drowsyScore');
  if (drowsyScoreEl) drowsyScoreEl.textContent = score;
  
  // Check manual override
  const manualStop = document.getElementById('manualStopToggle')?.checked;
  const effectivelyMoving = manualStop ? false : isMoving;
  
  // Trigger alert if moving and drowsy
  if (effectivelyMoving && score >= DROWSINESS_ALERT_THRESHOLD && !alertActive && !microgameActive) {
    alertActive = true;
    alertStartTime = Date.now();
    startPersistentAlert();
  }
  
  // Clear alert if stopped
  if (!effectivelyMoving && alertActive) {
    stopPersistentAlert();
    // Start microgame
    if (!microgameActive) {
      startMicrogame();
    }
  }
}

function startPersistentAlert() {
  // Oscillating beep using AudioContext
  if (!window.alertOscillator) {
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = audioCtx.createOscillator();
    const gainNode = audioCtx.createGain();
    
    oscillator.type = 'sine';
    oscillator.frequency.setValueAtTime(800, audioCtx.currentTime);
    oscillator.connect(gainNode);
    gainNode.connect(audioCtx.destination);
    
    // Pulsing pattern
    gainNode.gain.setValueAtTime(0, audioCtx.currentTime);
    const now = audioCtx.currentTime;
    for (let i = 0; i < 100; i++) {
      gainNode.gain.setValueAtTime(0.3, now + i * 1.0);
      gainNode.gain.setValueAtTime(0, now + i * 1.0 + 0.3);
    }
    
    oscillator.start();
    window.alertOscillator = oscillator;
    window.alertAudioCtx = audioCtx;
  }
  
  // Show alert overlay
  showAlertOverlay();
}

function stopPersistentAlert() {
  alertActive = false;
  
  if (window.alertOscillator) {
    window.alertOscillator.stop();
    window.alertOscillator = null;
  }
  if (window.alertAudioCtx) {
    window.alertAudioCtx.close();
    window.alertAudioCtx = null;
  }
  
  hideAlertOverlay();
}

function showAlertOverlay() {
  let overlay = document.getElementById('alertOverlay');
  if (!overlay) {
    overlay = document.createElement('div');
    overlay.id = 'alertOverlay';
    overlay.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(255, 0, 0, 0.9);
      color: white;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      z-index: 10000;
      font-size: 48px;
      font-weight: bold;
      text-align: center;
      animation: pulse 1s infinite;
    `;
    document.body.appendChild(overlay);
    
    const style = document.createElement('style');
    style.textContent = `
      @keyframes pulse {
        0%, 100% { opacity: 0.9; }
        50% { opacity: 1.0; }
      }
    `;
    document.head.appendChild(style);
  }
  
  overlay.innerHTML = `
    <div>⚠️ DROWSINESS DETECTED ⚠️</div>
    <div style="font-size: 32px; margin-top: 20px;">
      PULL OVER SAFELY NOW
    </div>
    <div style="font-size: 24px; margin-top: 20px; color: #ffeb3b;">
      Stop moving to complete safety check
    </div>
  `;
  overlay.style.display = 'flex';
}

function hideAlertOverlay() {
  const overlay = document.getElementById('alertOverlay');
  if (overlay) {
    overlay.style.display = 'none';
  }
}

function startMicrogame() {
  microgameActive = true;
  microgameStartTime = Date.now();
  microgameScore = 0;
  microgameTargets = [];
  
  showMicrogameOverlay();
  generateMicrogameTargets();
}

function showMicrogameOverlay() {
  let overlay = document.getElementById('microgameOverlay');
  if (!overlay) {
    overlay = document.createElement('div');
    overlay.id = 'microgameOverlay';
    overlay.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      z-index: 10001;
      font-family: Arial, sans-serif;
    `;
    document.body.appendChild(overlay);
  }
  
  overlay.innerHTML = `
    <div style="padding: 20px; text-align: center;">
      <h1 style="font-size: 36px; margin-bottom: 10px;">🎯 Alertness Check</h1>
      <div style="font-size: 24px; margin-bottom: 20px;">
        Tap the highlighted targets as they appear
      </div>
      <div style="font-size: 48px; font-weight: bold; margin-bottom: 20px;">
        <span id="microgameTimer">10</span>s
      </div>
      <div style="font-size: 32px; margin-bottom: 20px;">
        Score: <span id="microgameScore">0</span> / 10
      </div>
      <div id="microgameTargets" style="position: relative; width: 90%; max-width: 600px; height: 400px; margin: 0 auto; background: rgba(255,255,255,0.1); border-radius: 20px;"></div>
    </div>
  `;
  overlay.style.display = 'block';
}

function generateMicrogameTargets() {
  const container = document.getElementById('microgameTargets');
  if (!container) return;
  
  // Generate 10 targets that appear sequentially
  for (let i = 0; i < 10; i++) {
    setTimeout(() => {
      if (!microgameActive) return;
      
      const target = document.createElement('div');
      const x = Math.random() * 80 + 10; // 10-90%
      const y = Math.random() * 80 + 10;
      
      target.style.cssText = `
        position: absolute;
        left: ${x}%;
        top: ${y}%;
        width: 80px;
        height: 80px;
        background: #51cf66;
        border-radius: 50%;
        cursor: pointer;
        transform: translate(-50%, -50%) scale(0);
        animation: targetAppear 0.3s forwards;
        box-shadow: 0 0 20px rgba(81, 207, 102, 0.8);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 36px;
      `;
      
      target.textContent = '✓';
      target.dataset.targetId = i;
      
      target.addEventListener('click', () => {
        if (microgameActive && target.style.display !== 'none') {
          microgameScore++;
          document.getElementById('microgameScore').textContent = microgameScore;
          target.style.background = '#ffd43b';
          target.textContent = '⭐';
          setTimeout(() => target.remove(), 200);
        }
      });
      
      container.appendChild(target);
      microgameTargets.push(target);
      
      // Auto-remove after 800ms
      setTimeout(() => {
        if (target.parentNode) {
          target.style.opacity = '0';
          setTimeout(() => target.remove(), 300);
        }
      }, 800);
      
    }, i * 1000);
  }
  
  // Add CSS for animation
  if (!document.getElementById('microgameStyle')) {
    const style = document.createElement('style');
    style.id = 'microgameStyle';
    style.textContent = `
      @keyframes targetAppear {
        to { transform: translate(-50%, -50%) scale(1); }
      }
    `;
    document.head.appendChild(style);
  }
  
  // Update timer
  const timerInterval = setInterval(() => {
    if (!microgameActive) {
      clearInterval(timerInterval);
      return;
    }
    
    const elapsed = Date.now() - microgameStartTime;
    const remaining = Math.max(0, Math.ceil((MICROGAME_DURATION - elapsed) / 1000));
    const timerEl = document.getElementById('microgameTimer');
    if (timerEl) timerEl.textContent = remaining;
    
    if (elapsed >= MICROGAME_DURATION) {
      clearInterval(timerInterval);
      endMicrogame();
    }
  }, 100);
}

function endMicrogame() {
  microgameActive = false;
  
  const passed = microgameScore >= MICROGAME_PASS_THRESHOLD;
  
  const overlay = document.getElementById('microgameOverlay');
  if (overlay) {
    overlay.innerHTML = `
      <div style="padding: 40px; text-align: center;">
        <h1 style="font-size: 48px; margin-bottom: 20px;">
          ${passed ? '✅ Test Passed' : '❌ Test Failed'}
        </h1>
        <div style="font-size: 32px; margin-bottom: 30px;">
          Score: ${microgameScore} / 10
        </div>
        <div style="font-size: 24px; line-height: 1.6; max-width: 600px; margin: 0 auto;">
          ${passed ? 
            `<p>You're alert enough to continue, but please:</p>
             <ul style="text-align: left; display: inline-block;">
               <li>Take a 15-minute break</li>
               <li>Drink water or coffee</li>
               <li>Do some stretches</li>
               <li>Get fresh air</li>
             </ul>` :
            `<p style="color: #ff6b6b; font-weight: bold;">You are not fit to drive.</p>
             <p>Your taxi company has been notified.</p>
             <p>Please arrange for another driver or rest before continuing.</p>`
          }
        </div>
        <button id="microgameDismiss" style="
          margin-top: 30px;
          padding: 15px 40px;
          font-size: 24px;
          background: white;
          color: #667eea;
          border: none;
          border-radius: 10px;
          cursor: pointer;
          font-weight: bold;
        ">Continue</button>
      </div>
    `;
    
    document.getElementById('microgameDismiss').addEventListener('click', () => {
      overlay.style.display = 'none';
    });
  }
  
  // Handle failure - notify company
  if (!passed) {
    notifyTaxiCompany();
  }
  
  // Reset drowsiness score
  drowsinessScore = 0;
}

function notifyTaxiCompany() {
  // Stub for company notification API
  console.warn('🚨 DRIVER FAILED ALERTNESS CHECK - NOTIFYING COMPANY');
  
  const payload = {
    timestamp: new Date().toISOString(),
    driverId: 'DRIVER_ID', // Replace with actual ID
    vehicleId: 'VEHICLE_ID', // Replace with actual ID
    score: microgameScore,
    threshold: MICROGAME_PASS_THRESHOLD,
    drowsinessScore: drowsinessScore,
    location: 'GPS_COORDINATES' // Replace with actual location
  };
  
  // Example API call (replace with actual endpoint)
  /*
  fetch('https://api.taxicompany.com/driver-alerts', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  }).then(res => {
    console.log('Company notified:', res);
  }).catch(err => {
    console.error('Failed to notify company:', err);
  });
  */
  
  // For now, just log
  console.log('Notification payload:', JSON.stringify(payload, null, 2));
}

// ===================== Google Fit integration (optional) =====================
async function initGoogleFitIntegration(){
  // Try to load config
  let cfg = null;
  try{
    const res = await fetch('fit.config.json', { cache: 'no-store' });
    if(res.ok){ cfg = await res.json(); }
  }catch(_){ /* ignore */ }

  // Parse OAuth fragment if present
  if(location.hash && location.hash.includes('access_token=')){
    const params = new URLSearchParams(location.hash.substring(1));
    const token = params.get('access_token');
    if(token){
      localStorage.setItem('fitAccessToken', token);
      // Clean the fragment from URL without reloading
      history.replaceState(null, document.title, location.pathname + location.search);
    }
  }

  if(cfg && els.connectGoogleFit){
    els.connectGoogleFit.disabled = false;
    els.connectGoogleFit.title = 'Connect your Google Fit account to fetch sleep';
    els.connectGoogleFit.addEventListener('click', ()=>startFitOAuth(cfg));
  }

  const tokenStored = localStorage.getItem('fitAccessToken');
  if(cfg && tokenStored){
    try{
      await syncFitSleep(cfg, tokenStored);
    }catch(e){ console.warn('Google Fit sync failed:', e); }
  }
}

function startFitOAuth(cfg){
  if(!cfg?.clientId || !cfg?.redirectUri){
    alert('Google Fit clientId/redirectUri missing; edit fit.config.json');
    return;
  }
  const scopes = (cfg.scopes && Array.isArray(cfg.scopes) && cfg.scopes.length)
    ? cfg.scopes.join(' ')
    : 'https://www.googleapis.com/auth/fitness.sleep.read';
  const authUrl = new URL('https://accounts.google.com/o/oauth2/v2/auth');
  authUrl.searchParams.set('client_id', cfg.clientId);
  authUrl.searchParams.set('redirect_uri', cfg.redirectUri);
  authUrl.searchParams.set('response_type', 'token');
  authUrl.searchParams.set('scope', scopes);
  authUrl.searchParams.set('include_granted_scopes', 'true');
  authUrl.searchParams.set('prompt', 'consent');
  window.location.href = authUrl.toString();
}

async function syncFitSleep(cfg, accessToken){
  // Fetch sessions for the last 30 hours and aggregate sleep duration
  const end = new Date();
  const start = new Date(end.getTime() - 30*60*60*1000);
  const url = new URL('https://www.googleapis.com/fitness/v1/users/me/sessions');
  url.searchParams.set('startTime', start.toISOString());
  url.searchParams.set('endTime', end.toISOString());
  const resp = await fetch(url.toString(), { headers: { Authorization: 'Bearer ' + accessToken }});
  if(!resp.ok){ throw new Error('Fit sessions ' + resp.status); }
  const data = await resp.json();
  const sessions = Array.isArray(data?.session) ? data.session : [];
  // activityType 72 corresponds to sleep
  const sleeps = sessions.filter(s => String(s.activityType) === '72');
  let totalMs = 0;
  for(const s of sleeps){
    const sMs = new Date(s.endTimeMillis*1).getTime() - new Date(s.startTimeMillis*1).getTime();
    if(isFinite(sMs) && sMs>0) totalMs += sMs;
  }
  if(totalMs > 0){
    const hrs = totalMs / 3600000;
    sleepProfile.lastSleepHours = +(hrs.toFixed(2));
    if(els.lastSleepHours) els.lastSleepHours.value = String(sleepProfile.lastSleepHours);
    if(els.msg) els.msg.textContent = 'Google Fit sleep synced';
  }
}

// UI wiring
els.threshold.addEventListener('input', ()=>{
  els.thresholdVal.textContent = parseFloat(els.threshold.value).toFixed(2);
});
els.cnnThresh?.addEventListener('input', ()=>{
  els.cnnThreshVal.textContent = parseFloat(els.cnnThresh.value).toFixed(2);
});
els.cnnPumpkinDelay?.addEventListener('input', ()=>{
  const v = parseFloat(els.cnnPumpkinDelay.value);
  PUMPKIN_CNN_SHOW_DELAY_SEC = v;
  if(els.cnnPumpkinDelayVal) els.cnnPumpkinDelayVal.textContent = v.toFixed(1);
});
els.duration.addEventListener('input', ()=>{
  els.durationVal.textContent = parseFloat(els.duration.value).toFixed(1);
});
els.hideDelay.addEventListener('input', ()=>{
  els.hideDelayVal.textContent = parseFloat(els.hideDelay.value).toFixed(2);
});
els.tiltThresh.addEventListener('input', ()=>{
  const deg = parseFloat(els.tiltThresh.value);
  els.tiltThreshVal.textContent = deg.toFixed(0);
  TILT_THRESHOLD_RAD = deg * Math.PI / 180;
});
els.tiltCooldown.addEventListener('input', ()=>{
  const sec = parseFloat(els.tiltCooldown.value);
  els.tiltCooldownVal.textContent = sec.toFixed(1);
  TILT_COOLDOWN_MS = sec * 1000;
});
els.yawnThresh.addEventListener('input', ()=>{
  const v = parseFloat(els.yawnThresh.value);
  els.yawnThreshVal.textContent = v.toFixed(2);
});
els.yawnCooldown.addEventListener('input', ()=>{
  const v = parseFloat(els.yawnCooldown.value);
  els.yawnCooldownVal.textContent = v.toFixed(1);
});
els.yawnMorMin?.addEventListener('input', ()=>{
  const v = parseFloat(els.yawnMorMin.value);
  els.yawnMorMinVal.textContent = v.toFixed(2);
});
els.yawnMinDur?.addEventListener('input', ()=>{
  const v = parseFloat(els.yawnMinDur.value);
  els.yawnMinDurVal.textContent = v.toFixed(1);
});
// Gaze UI listeners
els.gazeThreshold?.addEventListener('input', ()=>{
  const v = parseFloat(els.gazeThreshold.value);
  if(els.gazeThresholdVal) els.gazeThresholdVal.textContent = v.toFixed(2);
});
els.gazeHold?.addEventListener('input', ()=>{
  const v = parseFloat(els.gazeHold.value);
  if(els.gazeHoldVal) els.gazeHoldVal.textContent = v.toFixed(1);
  GAZE_HOLD_MS = v * 1000;
  if(els.gazeHoldTarget) els.gazeHoldTarget.textContent = v.toFixed(1);
});
els.gazeSigmaK?.addEventListener('input', ()=>{
  const v = parseFloat(els.gazeSigmaK.value);
  if(els.gazeSigmaKVal) els.gazeSigmaKVal.textContent = v.toFixed(1);
  GAZE_SIGMA_K = v;
});
els.gazeSmoothWin?.addEventListener('input', ()=>{
  const v = parseInt(els.gazeSmoothWin.value, 10);
  if(els.gazeSmoothWinVal) els.gazeSmoothWinVal.textContent = String(v);
  GAZE_SMOOTH_WIN = Math.max(1, v|0);
});
els.gazeCalibEnable?.addEventListener('change', ()=>{
  gazeCalib.enabled = !!els.gazeCalibEnable.checked;
  // Reset progress to avoid stale state
  gazeHist.length = 0;
  gazeHoldStartMs = 0;
  gazeActive = false;
  gazePrevActive = false;
  if(els.gazeHoldProg) els.gazeHoldProg.textContent = '0.0';
});
els.gazeCalibrateBtn?.addEventListener('click', ()=>{
  // Start a 1s sampling window for calibration
  gazeCalibSampling = { startMs: performance.now(), samples: [] };
  if(els.msg) els.msg.textContent = 'Calibrating gaze center… look straight for 1s';
});
els.useGaze?.addEventListener('change', ()=>{
  // Reset gaze state on toggle
  gazeHist.length = 0;
  gazeHoldStartMs = 0;
  gazeActive = false;
  if(els.gazeDrift) els.gazeDrift.textContent = 'off';
  if(els.gazeHoldProg) els.gazeHoldProg.textContent = '0.0';
  if(els.gazeHoldTarget && els.gazeHold?.value){
    els.gazeHoldTarget.textContent = parseFloat(els.gazeHold.value).toFixed(1);
  }
});
els.startBtn.addEventListener('click', startCamera);
els.stopBtn.addEventListener('click', stopCamera);

// Load model when toggled or on start if checkbox is pre-checked
els.useCnn?.addEventListener('change', async ()=>{
  if(els.useCnn.checked && !eyeModel){
    els.msg.textContent = 'Loading eye CNN…';
    await loadEyeModel();
    els.msg.textContent = '';
  }
  // Reset accumulation and overlay state when switching mode
  closedAccum = 0;
  drowsy = false;
  lastEyesClosedAtMs = 0;
  pumpkinShowing = false;
  els.closedTime.textContent = `${closedAccum.toFixed(2)}s`;
});

els.useMouthCnn?.addEventListener('change', async ()=>{
  if(els.useMouthCnn.checked && !mouthModel){
    els.msg.textContent = 'Loading mouth model…';
    await loadMouthModel();
    els.msg.textContent = '';
  }
  // reset mouth prediction history
  mouthHist = [];
  mouthHistVec = [];
  mouthHistScalar = [];
  if(els.mouthPred) els.mouthPred.textContent = '-';
});

// Predictive UI listeners
els.predictThresh?.addEventListener('input', ()=>{
  const v = parseFloat(els.predictThresh.value);
  predictThresh = v;
  if(els.predictThreshVal) els.predictThreshVal.textContent = v.toFixed(2);
});
els.predictWindow?.addEventListener('input', ()=>{
  const v = parseInt(els.predictWindow.value, 10);
  PREDICT_WIN_SEC = Math.max(30, v|0);
  if(els.predictWindowVal) els.predictWindowVal.textContent = String(PREDICT_WIN_SEC);
  // trim history to new window
  const cutoff = performance.now() - PREDICT_WIN_SEC*1000;
  predHistory = predHistory.filter(b => b.secStartMs >= cutoff);
});
els.usePredict?.addEventListener('change', ()=>{
  predictEnabled = !!els.usePredict.checked;
  // reset buffers when toggled
  predSecBucket = null;
  predHistory = [];
  prevClosedForBlink = false;
  blinkStartMs = 0;
  if(els.predictProb) els.predictProb.textContent = '-';
});

// Handle page visibility to stop cam
window.addEventListener('visibilitychange', ()=>{ if(document.hidden) stopCamera(); });

// Helpful instruction for permissions
window.addEventListener('load', ()=>{
  els.msg.textContent = 'Click Start Camera and grant camera access.';
  // Initialize sleep profile UI values
  if(els.idealSleepVal && els.idealSleep) els.idealSleepVal.textContent = parseFloat(els.idealSleep.value || '7.5').toFixed(1);
  if(els.sleepRisk) els.sleepRisk.textContent = '-';
  // Predictive defaults
  if(els.predictThreshVal && els.predictThresh) els.predictThreshVal.textContent = parseFloat(els.predictThresh.value||'0.65').toFixed(2);
  if(els.predictWindowVal && els.predictWindow) els.predictWindowVal.textContent = String(parseInt(els.predictWindow.value||'120',10));
  // Attempt to load Google Fit config and detect OAuth token
  initGoogleFitIntegration();
});

// Select target face by clicking on a box
els.canvas.addEventListener('click', (e)=>{
  if(!lastFaceBoxes.length) return;
  const rect = els.canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  for(let i=0;i<lastFaceBoxes.length;i++){
    const b = lastFaceBoxes[i];
    if(!b) continue;
    if(x>=b.x && x<=b.x+b.w && y>=b.y && y<=b.y+b.h){
      targetIndexUser = i;
      closedAccum = 0; // reset accumulation when switching
      drowsy = false;
      els.msg.textContent = '';
      lastEyesClosedAtMs = 0;
      pumpkinShowing = false;
      break;
    }
  }
});

// Keyboard navigation to cycle faces
window.addEventListener('keydown', async (e)=>{
  const facesCount = lastFaceBoxes.length;
  if(!facesCount) return;
  // Dataset capture mode: n/o/s/y to download current mouth crop
  if(els.captureMode?.checked && !e.repeat){
    const key = e.key.toLowerCase();
    const labelMap = { n: 'neutral', o: 'open', s: 'smile', y: 'yawn' };
    if(labelMap[key]){
      e.preventDefault();
      const targetIdx = targetIndexUser ?? 0;
      const bbox = lastFaceBoxes[targetIdx];
      if(bbox){
        try{
          // approximate mouth box using face bbox proportion
          const cx = (bbox.x + bbox.w*0.5) / els.canvas.width;
          const cy = (bbox.y + bbox.h*0.62) / els.canvas.height;
          const box = { cx, cy, w: (bbox.w/els.canvas.width)*0.5, h: (bbox.h/els.canvas.height)*0.32 };
          const t = cropMouthFromCanvas(box, els.canvas);
          const off = document.createElement('canvas');
          off.width = MOUTH_W; off.height = MOUTH_H;
          await tf.browser.toPixels(t, off);
          t.dispose();
          off.toBlob((blob)=>{
            if(!blob) return;
            const a = document.createElement('a');
            const ts = Date.now();
            const url = URL.createObjectURL(blob);
            a.href = url;
            a.download = `${labelMap[key]}_${ts}.png`;
            document.body.appendChild(a);
            a.click();
            a.remove();
            setTimeout(()=>URL.revokeObjectURL(url), 1500);
          }, 'image/png');
        }catch(err){ console.warn('Capture failed:', err); }
      }
    }
  }
  if(e.key === 'ArrowRight'){
    if(targetIndexUser == null) targetIndexUser = 0; else targetIndexUser = (targetIndexUser+1) % facesCount;
    closedAccum = 0; drowsy=false; els.msg.textContent=''; lastEyesClosedAtMs = 0; pumpkinShowing = false;
  } else if(e.key === 'ArrowLeft'){
    if(targetIndexUser == null) targetIndexUser = 0; else targetIndexUser = (targetIndexUser-1+facesCount) % facesCount;
    closedAccum = 0; drowsy=false; els.msg.textContent=''; lastEyesClosedAtMs = 0; pumpkinShowing = false;
  }
});

// ===================== Sleep profile & auto-adapt =====================
// moved clamp01 above

function circadianRiskByLocalTime(date){
  // Simple curve: peak drowsiness around 3:00 (hour=3), secondary dip around 15:00.
  const h = date.getHours() + date.getMinutes()/60;
  // Combine two cosine lobes; normalize to ~0..1
  const peak1 = 0.5 * (1 + Math.cos(((h - 3) / 6) * Math.PI)); // ~peaks at 3am
  const peak2 = 0.35 * (1 + Math.cos(((h - 15) / 4) * Math.PI)); // smaller at 3pm
  const r = 0.6*peak1 + 0.4*peak2; // 0..1-ish
  return clamp01(r);
}

function computeSleepRisk(nowDate){
  const ideal = sleepProfile.idealHours || 7.5;
  const last = (sleepProfile.lastSleepHours != null) ? sleepProfile.lastSleepHours : ideal; // assume okay if unknown
  const debtHrs = Math.max(0, ideal - last);
  const debtRisk = clamp01(debtHrs / 2.5); // 0 debt ->0, 2.5h debt -> ~1
  const circ = circadianRiskByLocalTime(nowDate);
  const risk = clamp01(0.65*debtRisk + 0.35*circ);
  return { risk, debtRisk, circ };
}

function captureAdaptBaseline(){
  adaptBaseline = {
    cnnThresh: els.cnnThresh ? parseFloat(els.cnnThresh.value) : null,
    duration: parseFloat(els.duration.value),
    yawnMorMin: els.yawnMorMin ? parseFloat(els.yawnMorMin.value) : null,
    yawnMinDur: els.yawnMinDur ? parseFloat(els.yawnMinDur.value) : null,
    gazeHold: els.gazeHold ? parseFloat(els.gazeHold.value) : null,
  };
}

function applyAdaptations(now){
  if(!autoAdapt || !adaptBaseline) return;
  // Rate limit to ~2 Hz
  if(now - lastAdaptApplyMs < 500) return;
  lastAdaptApplyMs = now;
  const {risk} = computeSleepRisk(new Date());
  if(els.sleepRisk) els.sleepRisk.textContent = risk.toFixed(2);

  // Apply deltas (risk in 0..1)
  const clampNum = (v, min, max)=>Math.max(min, Math.min(max, v));
  // CNN threshold: lower up to 0.1 at high risk
  if(els.cnnThresh && adaptBaseline.cnnThresh != null){
    const v = clampNum(adaptBaseline.cnnThresh - 0.10*risk, 0.30, 0.90);
    els.cnnThresh.value = v.toFixed(2);
    if(els.cnnThreshVal) els.cnnThreshVal.textContent = v.toFixed(2);
  }
  // Eyes-closed duration: reduce up to 0.6s
  if(els.duration && adaptBaseline.duration != null){
    const v = clampNum(adaptBaseline.duration - 0.6*risk, 0.5, 3.0);
    els.duration.value = v.toFixed(1);
    if(els.durationVal) els.durationVal.textContent = v.toFixed(1);
  }
  // Yawn MOR min: reduce up to 0.08
  if(els.yawnMorMin && adaptBaseline.yawnMorMin != null){
    const v = clampNum(adaptBaseline.yawnMorMin - 0.08*risk, 0.20, 0.70);
    els.yawnMorMin.value = v.toFixed(2);
    if(els.yawnMorMinVal) els.yawnMorMinVal.textContent = v.toFixed(2);
  }
  // Yawn min duration: reduce up to 0.3s
  if(els.yawnMinDur && adaptBaseline.yawnMinDur != null){
    const v = clampNum(adaptBaseline.yawnMinDur - 0.3*risk, 0.1, 2.0);
    els.yawnMinDur.value = v.toFixed(1);
    if(els.yawnMinDurVal) els.yawnMinDurVal.textContent = v.toFixed(1);
  }
  // Gaze hold: reduce up to 0.4s and update internal ms
  if(els.gazeHold && adaptBaseline.gazeHold != null){
    const v = clampNum(adaptBaseline.gazeHold - 0.4*risk, 0.5, 3.0);
    els.gazeHold.value = v.toFixed(1);
    if(els.gazeHoldVal) els.gazeHoldVal.textContent = v.toFixed(1);
    GAZE_HOLD_MS = v * 1000;
    if(els.gazeHoldTarget) els.gazeHoldTarget.textContent = v.toFixed(1);
  }
}

// Listeners for sleep profile UI
els.autoAdapt?.addEventListener('change', ()=>{
  autoAdapt = !!els.autoAdapt.checked;
  if(autoAdapt){
    sleepProfile.idealHours = parseFloat(els.idealSleep?.value || '7.5');
    // If user provided last sleep hours, use it; else leave null
    const last = parseFloat(els.lastSleepHours?.value || 'NaN');
    sleepProfile.lastSleepHours = isNaN(last)? null : last;
    captureAdaptBaseline();
  } else {
    // Reset risk display and do not override user's sliders anymore
    if(els.sleepRisk) els.sleepRisk.textContent = '-';
  }
});
els.idealSleep?.addEventListener('input', ()=>{
  const v = parseFloat(els.idealSleep.value);
  sleepProfile.idealHours = v;
  if(els.idealSleepVal) els.idealSleepVal.textContent = v.toFixed(1);
});
els.lastSleepHours?.addEventListener('input', ()=>{
  const v = parseFloat(els.lastSleepHours.value);
  sleepProfile.lastSleepHours = isNaN(v)? null : v;
});
els.sleepImport?.addEventListener('change', (e)=>{
  const file = e.target.files?.[0];
  if(!file) return;
  const reader = new FileReader();
  reader.onload = ()=>{
    try{
      const data = JSON.parse(String(reader.result||'{}'));
      if(typeof data.lastSleepHours === 'number'){
        sleepProfile.lastSleepHours = data.lastSleepHours;
        if(els.lastSleepHours) els.lastSleepHours.value = String(data.lastSleepHours);
      }
      if(typeof data.idealHours === 'number'){
        sleepProfile.idealHours = data.idealHours;
        if(els.idealSleep){ els.idealSleep.value = String(data.idealHours); }
        if(els.idealSleepVal) els.idealSleepVal.textContent = data.idealHours.toFixed(1);
      }
      if(els.msg) els.msg.textContent = 'Sleep profile imported';
    }catch(err){ if(els.msg) els.msg.textContent = 'Import failed: invalid JSON'; }
  };
  reader.readAsText(file);
});
els.resetAdapt?.addEventListener('click', ()=>{
  if(!adaptBaseline) return;
  if(els.cnnThresh && adaptBaseline.cnnThresh!=null){ els.cnnThresh.value = adaptBaseline.cnnThresh.toFixed(2); if(els.cnnThreshVal) els.cnnThreshVal.textContent = adaptBaseline.cnnThresh.toFixed(2); }
  if(els.duration){ els.duration.value = adaptBaseline.duration.toFixed(1); if(els.durationVal) els.durationVal.textContent = adaptBaseline.duration.toFixed(1); }
  if(els.yawnMorMin && adaptBaseline.yawnMorMin!=null){ els.yawnMorMin.value = adaptBaseline.yawnMorMin.toFixed(2); if(els.yawnMorMinVal) els.yawnMorMinVal.textContent = adaptBaseline.yawnMorMin.toFixed(2); }
  if(els.yawnMinDur && adaptBaseline.yawnMinDur!=null){ els.yawnMinDur.value = adaptBaseline.yawnMinDur.toFixed(1); if(els.yawnMinDurVal) els.yawnMinDurVal.textContent = adaptBaseline.yawnMinDur.toFixed(1); }
  if(els.gazeHold && adaptBaseline.gazeHold!=null){ els.gazeHold.value = adaptBaseline.gazeHold.toFixed(1); if(els.gazeHoldVal) els.gazeHoldVal.textContent = adaptBaseline.gazeHold.toFixed(1); GAZE_HOLD_MS = adaptBaseline.gazeHold*1000; if(els.gazeHoldTarget) els.gazeHoldTarget.textContent = adaptBaseline.gazeHold.toFixed(1); }
});
