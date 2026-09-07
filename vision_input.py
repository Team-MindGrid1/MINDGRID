"""
mindgrid/vision_input.py
────────────────────────
Camera-based gaze and hand-gesture input using MediaPipe Tasks.

Provides:
  • WebcamGazeSource    – implements GazeSource ABC via face-mesh iris tracking
  • HandGestureSource   – detects simple hand gestures and maps them to Urdu

Both sources share one camera capture thread so the live display loop can
show the same feed without contending for frames.
"""

from __future__ import annotations

import os
import time
import math
import threading
import urllib.request
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker

from gaze_sensor import GazeSource, GazeSignal


# ── Model management ─────────────────────────────────────────────────────────

_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_MODEL_URLS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "gesture_recognizer.task": "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
}


def _ensure_model(filename: str) -> str:
    """Download a MediaPipe task model if it is not present locally."""
    path = os.path.join(_MODEL_DIR, filename)
    if os.path.isfile(path):
        return path

    os.makedirs(_MODEL_DIR, exist_ok=True)
    url = _MODEL_URLS[filename]
    print(f"[VisionInput] Downloading {filename} ...")
    urllib.request.urlretrieve(url, path)
    print(f"[VisionInput] Saved {filename} ({os.path.getsize(path)} bytes)")
    return path


# ── Webcam Gaze Source ───────────────────────────────────────────────────────

class WebcamGazeSource(GazeSource):
    """
    Gaze source that uses the webcam + MediaPipe Face Landmarker.

    Iris position relative to the eye centre is used to estimate gaze
    direction, which is mapped to a 4-channel synthetic EOG signal.
    """

    _LEFT_IRIS = [468, 469, 470, 471, 472]
    _RIGHT_IRIS = [473, 474, 475, 476, 477]
    _LEFT_EYE = [33, 133, 159, 145]    # inner, outer, top, bottom
    _RIGHT_EYE = [362, 263, 386, 374]  # inner, outer, top, bottom
    # Eye landmarks for blink detection (EAR - Eye Aspect Ratio)
    # Standard 6-point EAR: (v1+v2) / (2*h1)
    _LEFT_EYE_VERT = [159, 145, 144, 153]   # top1, bot1, top2, bot2
    _RIGHT_EYE_VERT = [386, 374, 373, 380]  # top1, bot1, top2, bot2
    _LEFT_EYE_HORIZ = [33, 133]             # inner, outer
    _RIGHT_EYE_HORIZ = [362, 263]
    _NOSE_TIP = 1
    _CHIN = 152
    _LEFT_EAR = [234, 127]    # left face edge
    _RIGHT_EAR = [454, 356]   # right face edge
    _EAR_THRESHOLD = 0.25     # below this = eyes closed (higher = more sensitive)

    def __init__(self, camera_id: int = 0, rate: float = 30.0, debug: bool = False) -> None:
        self._camera_id = camera_id
        self._rate = rate
        self._debug = debug
        self._x_threshold = 0.008
        self._y_threshold = 0.005  # Lowered for easier UP/DOWN detection
        self._cap: Optional[cv2.VideoCapture] = None
        self._connected = False
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

        base_options = BaseOptions(model_asset_path=_ensure_model("face_landmarker.task"))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            running_mode=vision.RunningMode.VIDEO,
        )
        self._face_landmarker = vision.FaceLandmarker.create_from_options(options)
        self._frame_idx = 0

        self._current_direction = "center"
        self._prev_direction = "center"
        self._no_face_frames = 0
        self._offset_history: deque[Tuple[float, float]] = deque(maxlen=5)

        # Blink detection
        self._eyes_closed_frames = 0
        self._BLINK_FRAMES = 3       # ~3 closed frames = blink detected (faster)
        self._blink_detected = False  # True when a blink is newly detected

        # Auto-calibration: collect baseline offsets for first N frames
        self._calibration_frames = 30  # ~1 second at 30fps
        self._calibration_samples: list[Tuple[float, float]] = []
        self._baseline_x: float = 0.0
        self._baseline_y: float = 0.0
        self._calibrated = False

    # ── Public API ───────────────────────────────────────────────────────

    @property
    def sample_rate(self) -> float:
        return self._rate

    def connect(self) -> None:
        backend = cv2.CAP_DSHOW if __import__("sys").platform == "win32" else 0
        self._cap = cv2.VideoCapture(self._camera_id, backend)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self._camera_id}")

        self._connected = True
        self._stop_event.clear()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        print(f"[WebcamGaze] Camera {self._camera_id} connected")

    def disconnect(self) -> None:
        self._connected = False
        self._stop_event.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._face_landmarker.close()
        print("[WebcamGaze] Camera disconnected")

    def read_frame(self) -> GazeSignal:
        if not self._connected:
            raise RuntimeError("Camera not connected – call connect() first")

        with self._frame_lock:
            frame = self._latest_frame.copy() if self._latest_frame is not None else None

        if frame is None:
            return GazeSignal(
                timestamp=time.time(),
                channels=np.array([12.0, 12.0, 12.0, 12.0]),
                impedance_ok=True,
            )

        direction = self._estimate_gaze(frame)
        self._prev_direction = self._current_direction
        self._current_direction = direction

        channels = self._direction_to_channels(direction, self._prev_direction)
        return GazeSignal(
            timestamp=time.time(),
            channels=channels,
            impedance_ok=True,
        )

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Return the most recent camera frame for display (thread-safe)."""
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    # ── Internals ────────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        """Background thread: keep grabbing the latest frame."""
        while not self._stop_event.is_set():
            if self._cap is None:
                break
            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._frame_lock:
                    self._latest_frame = frame
            time.sleep(0.001)

    def _estimate_gaze(self, frame: np.ndarray) -> str:
        """Estimate gaze direction from a BGR frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._face_landmarker.detect_for_video(mp_image, self._frame_idx)
        self._frame_idx += 1

        if not results.face_landmarks:
            self._no_face_frames += 1
            if self._no_face_frames > 10:
                return "center"
            return self._current_direction

        self._no_face_frames = 0
        landmarks = results.face_landmarks[0]
        img_h, img_w, _ = frame.shape

        # ── Blink detection (EAR - Eye Aspect Ratio) ────────────
        def _ear(vert_ids, horiz_ids):
            """Calculate Eye Aspect Ratio for one eye."""
            v = [landmarks[i] for i in vert_ids]
            h = [landmarks[i] for i in horiz_ids]
            # Vertical distances (top-bottom pairs)
            v1 = math.dist((v[0].x * img_w, v[0].y * img_h), (v[1].x * img_w, v[1].y * img_h))
            v2 = math.dist((v[2].x * img_w, v[2].y * img_h), (v[3].x * img_w, v[3].y * img_h))
            # Horizontal distance
            h1 = math.dist((h[0].x * img_w, h[0].y * img_h), (h[1].x * img_w, h[1].y * img_h))
            if h1 < 1:
                return 1.0
            return (v1 + v2) / (2.0 * h1)

        left_ear = _ear(self._LEFT_EYE_VERT, self._LEFT_EYE_HORIZ)
        right_ear = _ear(self._RIGHT_EYE_VERT, self._RIGHT_EYE_HORIZ)
        avg_ear = (left_ear + right_ear) / 2.0

        # Periodic EAR debug output (every 30 frames)
        if self._frame_idx % 30 == 0:
            print(f"[WebcamGaze] EAR: L={left_ear:.3f} R={right_ear:.3f} avg={avg_ear:.3f} threshold={self._EAR_THRESHOLD} closed={self._eyes_closed_frames}/{self._BLINK_FRAMES}")

        if avg_ear < self._EAR_THRESHOLD:
            self._eyes_closed_frames += 1
            if self._eyes_closed_frames >= self._BLINK_FRAMES:
                if not self._blink_detected:
                    self._blink_detected = True
                    self._current_direction = "blink"
                    if self._debug:
                        print(f"[WebcamGaze] BLINK detected (EAR={avg_ear:.3f})")
                return "blink"
        else:
            # Eyes open — reset blink tracking
            if self._eyes_closed_frames > 0:
                self._eyes_closed_frames = 0
                self._blink_detected = False

        left_iris = self._avg_landmark(landmarks, self._LEFT_IRIS, img_w, img_h)
        right_iris = self._avg_landmark(landmarks, self._RIGHT_IRIS, img_w, img_h)
        left_eye_center = self._avg_landmark(landmarks, self._LEFT_EYE, img_w, img_h)
        right_eye_center = self._avg_landmark(landmarks, self._RIGHT_EYE, img_w, img_h)

        left_offset = self._normalized_offset(
            left_iris, left_eye_center, landmarks,
            self._LEFT_EYE[0], self._LEFT_EYE[1],
            self._LEFT_EYE[2], self._LEFT_EYE[3],
            img_w, img_h,
        )
        right_offset = self._normalized_offset(
            right_iris, right_eye_center, landmarks,
            self._RIGHT_EYE[0], self._RIGHT_EYE[1],
            self._RIGHT_EYE[2], self._RIGHT_EYE[3],
            img_w, img_h,
        )

        raw_x = (left_offset[0] + right_offset[0]) / 2.0
        raw_y = (left_offset[1] + right_offset[1]) / 2.0

        # ── Head pose estimation (nose relative to eye midpoint) ────
        # Detects head turns even when iris doesn't move much
        nose_tip = (landmarks[self._NOSE_TIP].x * img_w,
                    landmarks[self._NOSE_TIP].y * img_h)
        eye_mid_x = (left_eye_center[0] + right_eye_center[0]) / 2.0
        eye_mid_y = (left_eye_center[1] + right_eye_center[1]) / 2.0

        # Face width for normalization
        left_ear = self._avg_landmark(landmarks, self._LEFT_EAR, img_w, img_h)
        right_ear = self._avg_landmark(landmarks, self._RIGHT_EAR, img_w, img_h)
        face_width = math.dist(left_ear, right_ear)
        if face_width < 1:
            face_width = 1.0

        # Nose offset from eye midpoint (normalized by face width)
        # In unmirrored camera: nose right in image = user turns left
        head_x = (nose_tip[0] - eye_mid_x) / face_width
        head_y = (eye_mid_y - nose_tip[1]) / face_width  # positive = nose up

        # Combine iris + head pose (weighted average)
        # Head pose gets higher weight since it's more reliable for natural looking
        combined_x = 0.4 * raw_x + 0.6 * head_x
        combined_y = 0.4 * raw_y + 0.6 * head_y

        self._offset_history.append((combined_x, combined_y))

        avg_x = sum(o[0] for o in self._offset_history) / len(self._offset_history)
        avg_y = sum(o[1] for o in self._offset_history) / len(self._offset_history)

        # ── Auto-calibration: establish baseline during first N frames ──
        if not self._calibrated:
            self._calibration_samples.append((avg_x, avg_y))
            if len(self._calibration_samples) >= self._calibration_frames:
                self._baseline_x = sum(s[0] for s in self._calibration_samples) / len(self._calibration_samples)
                self._baseline_y = sum(s[1] for s in self._calibration_samples) / len(self._calibration_samples)
                self._calibrated = True
                print(f"[WebcamGaze] Calibration complete: baseline x={self._baseline_x:+.4f} y={self._baseline_y:+.4f}")
                print("[WebcamGaze] Now look LEFT / RIGHT / UP / DOWN to test gaze detection.")
            return "center"  # always center during calibration

        # Subtract baseline (calibrated offsets)
        avg_x -= self._baseline_x
        avg_y -= self._baseline_y

        if self._debug and (abs(avg_x) > 0.02 or abs(avg_y) > 0.02):
            print(f"[WebcamGaze] iris=({raw_x:+.3f},{raw_y:+.3f}) head=({head_x:+.3f},{head_y:+.3f}) combined=({avg_x:+.3f},{avg_y:+.3f})")

        # Periodic status (every ~30 frames ≈ 1 sec)
        if self._frame_idx % 30 == 0:
            y_dir = "up" if avg_y > 0 else "down" if avg_y < 0 else "center"
            print(f"[WebcamGaze] x={avg_x:+.4f} y={avg_y:+.4f} (thresh_y={self._y_threshold}) → {self._current_direction} | iris_y={raw_y:+.3f} head_y={head_y:+.3f}")

        # Note: webcam image is NOT mirrored for landmark detection,
        # so iris x-offset is inverted relative to user's perspective.
        # Flip x direction to match user's view.
        #
        # Independent axis detection: check X and Y independently
        # instead of comparing abs(x) vs abs(y). This fixes the issue
        # where slight vertical drift prevents horizontal detection.
        x_active = abs(avg_x) > self._x_threshold
        y_active = abs(avg_y) > self._y_threshold

        # If both axes active, pick the stronger one
        if x_active and y_active:
            # Normalize by threshold to compare fairly
            x_strength = abs(avg_x) / self._x_threshold
            y_strength = abs(avg_y) / self._y_threshold
            if x_strength >= y_strength:
                y_active = False
            else:
                x_active = False

        if x_active:
            if avg_x > 0:
                return "left"     # iris right in image = user looks left
            else:
                return "right"    # iris left in image = user looks right

        if y_active:
            if avg_y > 0:
                return "up"
            else:
                return "down"

        return "center"

    @staticmethod
    def _avg_landmark(landmarks, indices: list[int], img_w: int, img_h: int) -> Tuple[float, float]:
        xs = [landmarks[i].x * img_w for i in indices]
        ys = [landmarks[i].y * img_h for i in indices]
        return sum(xs) / len(xs), sum(ys) / len(ys)

    @staticmethod
    def _normalized_offset(
        iris: Tuple[float, float],
        eye_center: Tuple[float, float],
        landmarks,
        h_inner: int,
        h_outer: int,
        v_top: int,
        v_bottom: int,
        img_w: int,
        img_h: int,
    ) -> Tuple[float, float]:
        h1 = (landmarks[h_inner].x * img_w, landmarks[h_inner].y * img_h)
        h2 = (landmarks[h_outer].x * img_w, landmarks[h_outer].y * img_h)
        eye_width = math.dist(h1, h2)

        v1 = (landmarks[v_top].x * img_w, landmarks[v_top].y * img_h)
        v2 = (landmarks[v_bottom].x * img_w, landmarks[v_bottom].y * img_h)
        eye_height = math.dist(v1, v2)

        if eye_width == 0 or eye_height == 0:
            return 0.0, 0.0

        dx = (iris[0] - eye_center[0]) / eye_width
        # Positive dy = iris above eye centre = looking up
        dy = (eye_center[1] - iris[1]) / eye_height
        return dx, dy

    @staticmethod
    def _direction_to_channels(direction: str, prev_direction: str) -> np.ndarray:
        """
        Convert a gaze direction to a 4-channel EOG-like signal.
        Channels are [h_left, h_right, v_up, v_down] in micro-volts.
        """
        noise = np.random.normal(0, 2.5, size=4)
        channels = np.array([12.0, 12.0, 12.0, 12.0]) + noise

        amp = 120.0
        if direction == "left":
            channels[0] += amp
            channels[1] -= amp * 0.5
        elif direction == "right":
            channels[1] += amp
            channels[0] -= amp * 0.5
        elif direction == "up":
            channels[2] += amp
            channels[3] -= amp * 0.5
        elif direction == "down":
            channels[3] += amp
            channels[2] -= amp * 0.5

        if direction != prev_direction and direction != "center":
            channels += np.random.normal(40.0, 20.0, size=4)

        return channels


# ── Hand Gesture Source ──────────────────────────────────────────────────────

class HandGestureSource:
    """
    Detects predefined hand gestures using MediaPipe GestureRecognizer and
    maps them to Urdu phrases.

    Gesture vocabulary
    ──────────────────
    Closed_Fist   → "رکو"                    (stop)
    Pointing_Up   → "ڈاکٹر کو بلائیں"        (call doctor)
    Thumb_Up      → "سب ٹھیک ہے"              (all fine)
    Open_Palm     → "مجھے مدد چاہیے"         (help)
    ILoveYou      → "مجھے پانی چاہیے"        (water)
    Victory       → "مجھے بھوک لگی ہے"       (hungry)
    Thumbs_Down   → "مجھے تکلیف ہے"          (in pain)

    Finger counting (fallback when gesture not recognized):
    Only pinky   → "مجھے واش روم جانا ہے"    (need washroom)
    3 fingers    → "شکریہ"                    (thank you)
    4 fingers    → "گھر جانا ہے"              (want to go home)
    """

    _GESTURE_PHRASES = {
        "Closed_Fist": "رکو",
        "Pointing_Up": "ڈاکٹر کو بلائیں",
        "Thumb_Up": "سب ٹھیک ہے",
        "Open_Palm": "مجھے مدد چاہیے",
        "ILoveYou": "مجھے پانی چاہیے",
        "Victory": "مجھے بھوک لگی ہے",
        "Thumbs_Down": "مجھے تکلیف ہے",
    }

    _FINGER_PHRASES = {
        3: "شکریہ",
        4: "گھر جانا ہے",
    }

    # Special phrase when ONLY pinky is extended
    _PINKY_PHRASE = "مجھے واش روم جانا ہے"

    def __init__(self) -> None:
        # Gesture recognizer for named gestures
        gesture_model = BaseOptions(model_asset_path=_ensure_model("gesture_recognizer.task"))
        options = vision.GestureRecognizerOptions(
            base_options=gesture_model,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
        )
        self._gesture_recognizer = vision.GestureRecognizer.create_from_options(options)

        # Hand landmarker for finger counting (separate model)
        hand_model = BaseOptions(model_asset_path=_ensure_model("hand_landmarker.task"))
        hand_options = vision.HandLandmarkerOptions(
            base_options=hand_model,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
        )
        self._hand_landmarker = HandLandmarker.create_from_options(hand_options)

        self._frame_idx = 0
        self._last_phrase: Optional[str] = None

    def close(self) -> None:
        self._gesture_recognizer.close()
        self._hand_landmarker.close()

    def _count_fingers(self, frame: np.ndarray) -> tuple:
        """
        Count extended fingers using hand landmarks.
        Returns (count, extended_fingers) where extended_fingers is a set of finger names.
        Finger names: 'thumb', 'index', 'middle', 'ring', 'pinky'
        Returns (None, None) if no hand detected.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._hand_landmarker.detect_for_video(mp_image, self._frame_idx)

        if not results.hand_landmarks:
            return None, None

        hand = results.hand_landmarks[0]

        # MediaPipe hand landmark indices:
        # Fingertips: Thumb=4, Index=8, Middle=12, Ring=16, Pinky=20
        # PIP joints: Thumb=3, Index=6, Middle=10, Ring=14, Pinky=18

        tips = [4, 8, 12, 16, 20]
        pips = [3, 6, 10, 14, 18]
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

        # For handedness (left/right hand)
        handedness = results.handedness[0][0].category_name  # "Left" or "Right"
        is_left_hand = (handedness == "Left")

        count = 0
        extended = set()

        for i, (tip, pip, name) in enumerate(zip(tips, pips, finger_names)):
            is_extended = False
            if i == 0:  # Thumb - use x distance
                if is_left_hand:
                    if hand[tip].x < hand[pip].x:
                        is_extended = True
                else:
                    if hand[tip].x > hand[pip].x:
                        is_extended = True
            else:
                # Other fingers: tip y < pip y means extended
                if hand[tip].y < hand[pip].y:
                    is_extended = True

            if is_extended:
                count += 1
                extended.add(name)

        return count, extended

    def detect(self, frame: np.ndarray) -> tuple:
        """
        Process a BGR frame.
        Returns (phrase, hand_present, gesture_name):
          - phrase: Urdu phrase when a NEW gesture is detected, else None
          - hand_present: True if any hand is visible in the frame
          - gesture_name: current gesture category name (e.g. 'Thumbs_Down')

        Detection priority:
          1. Named gestures (fist, thumb up, palm, etc.)
          2. Finger counting fallback (3 fingers, 4 fingers, pinky only)

        Each gesture phrase is spoken only ONCE until:
          - A different gesture is shown, OR
          - Hand leaves the frame and comes back
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._gesture_recognizer.recognize_for_video(mp_image, self._frame_idx)
        self._frame_idx += 1

        if not results.gestures:
            # No hand visible — reset so same gesture can fire again later
            self._last_phrase = None
            return None, False, None

        # Hand IS visible — check named gesture first
        gesture = results.gestures[0][0].category_name
        phrase = self._GESTURE_PHRASES.get(gesture)

        # If named gesture not found, try finger counting
        if not phrase:
            finger_count, extended_fingers = self._count_fingers(frame)
            if finger_count is not None:
                # Check for pinky-only gesture (only pinky extended)
                if extended_fingers == {'pinky'}:
                    phrase = self._PINKY_PHRASE
                    gesture = "Pinky_Only"
                else:
                    # Use regular finger count mapping
                    finger_phrase = self._FINGER_PHRASES.get(finger_count)
                    if finger_phrase:
                        phrase = finger_phrase
                        gesture = f"{finger_count}_Fingers"

        # Only return phrase if it's DIFFERENT from last spoken gesture
        if phrase and phrase != self._last_phrase:
            self._last_phrase = phrase
            return phrase, True, gesture

        return None, True, gesture  # hand visible but same gesture (no repeat)


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("\n--- Vision Input – Gaze Smoke Test ---")
    print("Look left/right/up/down. Press Ctrl+C to quit.\n")

    source = WebcamGazeSource(camera_id=0, rate=30.0)
    source.connect()

    gaze_classifier = __import__("gaze_classifier")
    classifier = gaze_classifier.GazeIntentClassifier(window_size=30, sample_rate=source.sample_rate)
    buffer = gaze_classifier.GazeIntentBuffer(hold_count=2)

    try:
        while True:
            sig = source.read_frame()
            intent = classifier.push(sig)
            stable = buffer.update(intent)

            if stable:
                token = gaze_classifier.intent_to_token(stable)
                print(f"  INTENT  {stable.action:10s} {stable.direction:6s}  -> {token}")
    except KeyboardInterrupt:
        pass
    finally:
        source.disconnect()
        print("--- Test complete ---\n")
