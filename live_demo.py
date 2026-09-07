"""
mindgrid/live_demo.py
──────────────────────
MindGrid End-to-End Interactive Demo.

Launches the full pipeline with simulated gaze sensor and system
microphone, showing real-time output in:

  • A camera window ("wearer's view") with overlay indicators
  • Console log with timestamps
  • pygame lens HUD (if available)

Keyboard Controls
─────────────────
  ← ↑ → ↓   Simulate gaze direction (override sensor)
  SPACE       Simulate blink-confirm
  D           Simulate dwell-select on current direction
  M           Toggle microphone listening
  T           Type custom Urdu text for TTS
  Q / ESC     Quit

"""

from __future__ import annotations

import argparse
import os
import cv2
import sys
import time
import threading
from typing import Tuple, Optional
import numpy as np

from PIL import Image, ImageDraw, ImageFont

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    _HAS_URDU_RENDER = True
except ImportError:
    _HAS_URDU_RENDER = False

# ── MindGrid modules ─────────────────────────────────────────────────────────

from gaze_sensor import SiliconPadSensor, GazeSignal
from gaze_classifier import GazeIntentClassifier, GazeIntentBuffer, intent_to_token, GazeIntent
from mic_input import MicCapture
from modelscope_stt import UrduTranscriber
from urdu_tts import UrduVoice
from lens_display import create_hud
from cloud import reconstruct_phrase, lookup_token
from stream import app, socketio, push_subtitle, push_gaze, push_transcript
from vision_input import WebcamGazeSource, HandGestureSource

# ── State ────────────────────────────────────────────────────────────────────

class DemoState:
    """Thread-safe shared state for the demo."""
    def __init__(self):
        self.lock = threading.Lock()
        self.gaze_direction = "center"
        self.gaze_action = "idle"
        self.gaze_confidence = 0.0
        self.current_subtitle = ""
        self.current_transcript = ""
        self.mic_active = True
        self.manual_direction = None  # keyboard override
        self.hand_active = False      # True when hand visible in frame
        self.thumb_down_active = False # True when Thumb Down gesture shown
        self.running = True


state = DemoState()

# ── Gaze Pipeline Thread ─────────────────────────────────────────────────────

def gaze_thread_fn(sensor: SiliconPadSensor, voice: UrduVoice, hud) -> None:
    """Process gaze signals and emit Urdu phrases."""
    classifier = GazeIntentClassifier(window_size=30, sample_rate=sensor.sample_rate)
    buffer = GazeIntentBuffer(hold_count=2)

    # Webcam mode: use direct direction-based dwell detection
    # (bandpass filter removes DC-level direction info from synthetic EOG)
    is_webcam = isinstance(sensor, WebcamGazeSource)
    _direction_counter = 0
    _prev_dir = "center"
    _dwell_cooldown_until = 0.0
    _last_spoken_dir = None  # Track last spoken direction
    _DWELL_HOLD_FRAMES = 10   # ~0.3 sec at 30fps — very easy
    _DWELL_COOLDOWN_SEC = 3.0  # 3 sec cooldown between phrases

    # Double-blink detection for washroom
    _blink_count = 0           # Count blinks in current sequence
    _first_blink_time = 0.0    # Time of first blink
    _DOUBLE_BLINK_WINDOW = 2.0 # 2 seconds to complete double blink
    _WASHROOM_PHRASE = "مجھے واش روم جانا ہے"

    print(f"[GAZE] Thread started, is_webcam={is_webcam}")

    while state.running:
        # Check for manual keyboard override
        with state.lock:
            manual = state.manual_direction
            if manual:
                state.manual_direction = None

        if manual:
            # Inject a synthetic dwell event from keyboard
            intent = GazeIntent(
                direction=manual,
                action="dwell",
                confidence=0.95,
                timestamp=time.time(),
            )
            token = intent_to_token(intent)

            with state.lock:
                state.gaze_direction = manual
                state.gaze_action = "dwell"
                state.gaze_confidence = 0.95

            push_gaze(manual, "dwell", 0.95)

            if token:
                entry = lookup_token(token)
                if entry:
                    urdu_text = entry["display"]
                    with state.lock:
                        state.current_subtitle = urdu_text
                    push_subtitle(urdu_text, 0.95)
                    voice.speak(urdu_text)
                    print(f"  [GAZE→TTS] {manual} dwell → {urdu_text}")
        elif is_webcam:
            # ── Webcam direct dwell detection ──────────────────────
            # Check if hand is visible — if so, skip DWELL but still check blinks
            hand_visible = False
            thumb_down = False
            with state.lock:
                hand_visible = state.hand_active
                thumb_down = state.thumb_down_active

            # Read direction directly from WebcamGazeSource (bypasses bandpass)
            sig = sensor.read_frame()
            # Get current direction from the webcam source
            current_dir = sensor._current_direction

            with state.lock:
                state.gaze_direction = current_dir

            # Track stable direction for dwell
            if current_dir == _prev_dir:
                _direction_counter += 1
            else:
                _direction_counter = 0
                _prev_dir = current_dir

            now = time.time()

            # ── Blink + Double-Blink detection ─────────────────────
            if current_dir == "blink":
                # Track blink count for double-blink detection
                if _blink_count == 0:
                    # First blink
                    _first_blink_time = now
                    _blink_count = 1
                elif now - _first_blink_time <= _DOUBLE_BLINK_WINDOW:
                    # Second blink within window → washroom!
                    _blink_count = 0
                    if now >= _dwell_cooldown_until:
                        _dwell_cooldown_until = now + _DWELL_COOLDOWN_SEC
                        _last_spoken_dir = "double_blink"
                        _direction_counter = 0
                        with state.lock:
                            state.gaze_direction = "center"
                            state.gaze_action = "double_blink"
                            state.current_subtitle = _WASHROOM_PHRASE
                        push_gaze("center", "double_blink", 0.95)
                        push_subtitle(_WASHROOM_PHRASE, 0.95)
                        voice.speak(_WASHROOM_PHRASE)
                        print(f"  [GAZE→TTS] double-blink → {_WASHROOM_PHRASE}")
                        time.sleep(1 / sensor.sample_rate)
                        continue
                else:
                    # Too slow for double blink — reset
                    _blink_count = 1
                    _first_blink_time = now

                # Single blink → "مجھے نیند آئی ہے" (with cooldown)
                if _last_spoken_dir != "blink" and now >= _dwell_cooldown_until:
                    _dwell_cooldown_until = now + _DWELL_COOLDOWN_SEC
                    _last_spoken_dir = "blink"
                    _direction_counter = 0
                    blink_text = "مجھے نیند آئی ہے"
                    with state.lock:
                        state.gaze_direction = "blink"
                        state.gaze_action = "blink"
                        state.current_subtitle = blink_text
                    push_gaze("blink", "blink", 0.95)
                    push_subtitle(blink_text, 0.95)
                    voice.speak(blink_text)
                    print(f"  [GAZE→TTS] blink → {blink_text}")
                time.sleep(1 / sensor.sample_rate)
                continue
            else:
                # Not blinking — reset double-blink timer if window expired
                if _blink_count > 0 and now - _first_blink_time > _DOUBLE_BLINK_WINDOW:
                    _blink_count = 0

            # Skip DWELL detection if hand is visible (but blink still works above)
            if hand_visible:
                _direction_counter = 0  # reset so gaze doesn't accumulate
                time.sleep(1 / sensor.sample_rate)
                continue

            # Dwell trigger: stable direction for N frames + cooldown elapsed + new direction
            now = time.time()
            if (_direction_counter >= _DWELL_HOLD_FRAMES
                    and current_dir != "center"
                    and current_dir != _last_spoken_dir  # only speak NEW direction
                    and now >= _dwell_cooldown_until):
                _dwell_cooldown_until = now + _DWELL_COOLDOWN_SEC
                _direction_counter = 0
                _last_spoken_dir = current_dir  # remember what we spoke

                intent = GazeIntent(
                    direction=current_dir,
                    action="dwell",
                    confidence=0.90,
                    timestamp=now,
                )
                token = intent_to_token(intent)

                with state.lock:
                    state.gaze_action = "dwell"
                    state.gaze_confidence = 0.90

                push_gaze(current_dir, "dwell", 0.90)

                if token:
                    entry = lookup_token(token)
                    if entry:
                        urdu_text = entry["display"]
                        with state.lock:
                            state.current_subtitle = urdu_text
                        push_subtitle(urdu_text, 0.90)
                        voice.speak(urdu_text)
                        print(f"  [GAZE→TTS] {current_dir} dwell → {urdu_text}")

        else:
            # Read from simulated sensor (uses classifier with bandpass)
            sig = sensor.read_frame()
            raw_intent = classifier.push(sig)
            stable = buffer.update(raw_intent)

            if stable:
                token = intent_to_token(stable)
                with state.lock:
                    state.gaze_direction = stable.direction
                    state.gaze_action = stable.action
                    state.gaze_confidence = stable.confidence

                push_gaze(stable.direction, stable.action, stable.confidence)

                if token:
                    entry = lookup_token(token)
                    if entry:
                        urdu_text = entry["display"]
                        with state.lock:
                            state.current_subtitle = urdu_text
                        push_subtitle(urdu_text, stable.confidence)
                        voice.speak(urdu_text)
                        print(f"  [GAZE→TTS] {stable.direction} {stable.action} → {urdu_text}")

        time.sleep(1 / sensor.sample_rate)


# ── Mic Pipeline Thread ──────────────────────────────────────────────────────

def mic_thread_fn(mic: MicCapture, transcriber: UrduTranscriber) -> None:
    """Capture audio and transcribe Urdu speech."""
    while state.running:
        if not state.mic_active:
            time.sleep(0.5)
            continue

        try:
            audio = mic.get_chunk(duration_sec=3.0)
            text = transcriber.transcribe(audio)
            if text:
                with state.lock:
                    state.current_transcript = text
                push_transcript(text)
                print(f"  [MIC→TEXT] {text}")
        except Exception:
            time.sleep(0.5)


# ── Hand Gesture Thread ──────────────────────────────────────────────────────

def hand_thread_fn(source: WebcamGazeSource, voice: UrduVoice) -> None:
    """Detect hand gestures from the shared camera feed and speak Urdu phrases."""
    print("[HAND] Gesture detection thread started")
    detector = HandGestureSource()
    frame_count = 0

    try:
        while state.running:
            frame = source.get_latest_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            phrase, hand_present, gesture_name = detector.detect(frame)
            frame_count += 1

            # Update hand presence + thumb_down in shared state
            with state.lock:
                state.hand_active = hand_present
                state.thumb_down_active = (gesture_name == "Thumbs_Down")

            if frame_count % 100 == 0:
                print(f"[HAND] processed {frame_count} frames, hand={hand_present}, gesture={gesture_name}, last={detector._last_phrase}")

            # Speak all detected gestures (including Thumb_Down)
            if phrase:
                with state.lock:
                    state.current_subtitle = phrase
                push_subtitle(phrase, 1.0)
                voice.speak(phrase)
                print(f"  [HAND→TTS] {phrase}  (gaze paused while hand visible)")

            time.sleep(0.1)
    finally:
        detector.close()


# ── Urdu text helper for OpenCV overlay ─────────────────────────────────────

def _prepare_urdu(text: str) -> str:
    """Reshape and bidi-ize Urdu for proper display."""
    if not _HAS_URDU_RENDER or not text:
        return text
    try:
        return get_display(arabic_reshaper.reshape(text))
    except Exception:
        return text


def _find_urdu_font(size: int) -> Optional[ImageFont.FreeTypeFont]:
    """Find a Windows font that supports Urdu."""
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return None


def put_urdu_text(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_size: int = 24,
    color: Tuple[int, int, int] = (255, 255, 255),
) -> None:
    """Render Urdu text onto an OpenCV BGR frame using PIL."""
    if not text:
        return

    prepared = _prepare_urdu(text)
    font = _find_urdu_font(font_size)
    if font is None:
        # Fallback: skip rendering
        return

    # Convert OpenCV BGR to RGB PIL image
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text(position, prepared, font=font, fill=color[::-1])  # RGB for PIL
    # Put back into frame
    frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ── Camera Overlay Drawing ───────────────────────────────────────────────────

_DIRECTION_ARROWS = {
    "left": "◀", "right": "▶", "up": "▲", "down": "▼", "center": "●",
}

def draw_overlay(frame: np.ndarray) -> None:
    """Draw gaze indicator, subtitle, and transcript on the camera frame."""
    h, w = frame.shape[:2]

    with state.lock:
        direction = state.gaze_direction
        action = state.gaze_action
        confidence = state.gaze_confidence
        subtitle = state.current_subtitle
        transcript = state.current_transcript
        mic_on = state.mic_active

    # Semi-transparent overlay bar at bottom
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 80), (w, h), (20, 30, 50), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Gaze direction indicator (top-right)
    arrow = _DIRECTION_ARROWS.get(direction, "●")
    cv2.putText(frame, f"Gaze: {arrow} {direction}", (w - 250, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)

    # Action badge
    action_colors = {
        "fixation": (100, 200, 100), "saccade": (80, 200, 255),
        "blink": (255, 180, 80), "dwell": (200, 100, 255), "idle": (150, 150, 150),
    }
    color = action_colors.get(action, (150, 150, 150))
    cv2.putText(frame, f"{action} ({confidence:.0%})", (w - 250, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Microphone status
    mic_icon = "MIC ON" if mic_on else "MIC OFF"
    mic_color = (0, 255, 0) if mic_on else (0, 0, 255)
    cv2.putText(frame, mic_icon, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, mic_color, 1)

    # Subtitle (bottom bar) – render Urdu properly via PIL
    if subtitle:
        display_text = subtitle[:50]
        put_urdu_text(frame, display_text, (10, h - 55), font_size=26, color=(255, 255, 255))

    # Transcript (middle) – render Urdu properly via PIL
    if transcript:
        t_text = transcript[:50]
        put_urdu_text(frame, f"STT: {t_text}", (10, h - 90), font_size=18, color=(200, 220, 255))

    # Title
    cv2.putText(frame, "MindGrid Assistive Glass - Live Demo", (10, h - 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)


# ── Keyboard Input ───────────────────────────────────────────────────────────

def handle_key(key: int, voice: UrduVoice) -> bool:
    """Process keyboard input. Returns False to quit."""
    key_char = chr(key & 0xFF).lower() if key > 0 else ""

    key_map = {
        "left":  "left",
        "up":    "up",
        "right": "right",
        "down":  "down",
    }

    # Arrow keys → simulate gaze direction (OpenCV returns special codes)
    if key == 2424832:    # Left arrow (Windows)
        state.manual_direction = "left"
    elif key == 2490368:  # Up arrow
        state.manual_direction = "up"
    elif key == 2555904:  # Right arrow
        state.manual_direction = "right"
    elif key == 2621440:  # Down arrow
        state.manual_direction = "down"
    elif key_char == " ":
        state.manual_direction = "center"  # blink-confirm on center
    elif key_char == "d":
        with state.lock:
            state.manual_direction = state.gaze_direction  # dwell on current
    elif key_char == "m":
        state.mic_active = not state.mic_active
        print(f"  [MIC] {'Enabled' if state.mic_active else 'Disabled'}")
    elif key_char == "t":
        print("\n  Type Urdu text for TTS (press Enter):")
        try:
            text = input("  > ")
            if text.strip():
                voice.speak(text.strip())
                push_subtitle(text.strip(), 1.0)
        except Exception:
            pass
    elif key_char in ("q", "\x1b"):  # q or ESC
        return False

    return True


# ── Console Keyboard Input (fallback when no OpenCV window) ─────────────────

_ARROW_KEYS = {
    "K": "left",
    "H": "up",
    "M": "right",
    "P": "down",
}

def _read_console_key(voice: UrduVoice) -> bool:
    """
    Read one keystroke from the Windows console (msvcrt) or POSIX tty.
    Returns False to quit.
    """
    try:
        import msvcrt
        ch = msvcrt.getch()
        if ch == b'\xe0' or ch == b'\x00':   # extended key prefix
            ch2 = msvcrt.getch().decode('ascii', errors='ignore').upper()
            direction = _ARROW_KEYS.get(ch2)
            if direction:
                state.manual_direction = direction
                print(f"  [KEY] gaze → {direction}")
            return True
        key_char = ch.decode('ascii', errors='ignore').lower()
    except ImportError:
        # POSIX fallback
        import tty, termios, sys as _sys
        fd = _sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = _sys.stdin.read(1)
            if ch == '\x1b':   # ESC or arrow prefix
                ch2 = _sys.stdin.read(2)
                arrow_map = {'[D': 'left', '[A': 'up', '[C': 'right', '[B': 'down'}
                direction = arrow_map.get(ch2)
                if direction:
                    state.manual_direction = direction
                    print(f"  [KEY] gaze → {direction}")
                return True
            key_char = ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    if key_char == ' ':
        state.manual_direction = "center"
        print("  [KEY] blink-confirm")
    elif key_char == 'd':
        with state.lock:
            state.manual_direction = state.gaze_direction
        print(f"  [KEY] dwell on {state.gaze_direction}")
    elif key_char == 'm':
        state.mic_active = not state.mic_active
        print(f"  [MIC] {'Enabled' if state.mic_active else 'Disabled'}")
    elif key_char == 't':
        print("\n  Type Urdu text for TTS (press Enter):")
        try:
            text = input("  > ")
            if text.strip():
                voice.speak(text.strip())
                push_subtitle(text.strip(), 1.0)
        except Exception:
            pass
    elif key_char in ('q', '\x1b'):
        return False

    return True


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MindGrid Live Demo")
    parser.add_argument(
        "--camera",
        choices=["simulated", "webcam"],
        default="simulated",
        help="Gaze source: simulated EOG sensor or webcam eye tracking",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print gaze offset debug info (webcam mode only)",
    )
    args = parser.parse_args()

    print("\n==================================================")
    print("    MindGrid Live Demo – Assistive Glass Preview  ")
    print("==================================================")
    print("  Controls:")
    if args.camera == "webcam":
        print("    Webcam      → Real eye tracking (look left/right/up/down)")
        print("    Hand        → Fist/Point/Thumb/Palm/LoveYou/Victory")
        print("    SPACE       → Blink-confirm")
    else:
        print("    Arrow Keys  → Simulate gaze direction")
        print("    SPACE       → Blink-confirm")
    print("    D           → Dwell-select current direction")
    print("    M           → Toggle microphone")
    print("    T           → Type Urdu text for TTS")
    print("    Q / ESC     → Quit")
    print("==================================================\n")

    # ── Initialise ───────────────────────────────────────────────────

    if args.camera == "webcam":
        sensor = WebcamGazeSource(camera_id=0, rate=30.0, debug=args.debug)
    else:
        sensor = SiliconPadSensor()
    sensor.connect()

    voice = UrduVoice(engine_priority=["gtts", "pyttsx3"])

    transcriber = UrduTranscriber(stub_mode=True)
    transcriber.load_model()

    mic = MicCapture()
    mic.start_stream()

    hud = create_hud(headless=False)
    hud.start()

    # ── Launch pipeline threads ──────────────────────────────────────

    threads = [
        threading.Thread(target=gaze_thread_fn, args=(sensor, voice, hud), daemon=True),
        threading.Thread(target=mic_thread_fn, args=(mic, transcriber), daemon=True),
    ]
    if args.camera == "webcam":
        threads.append(
            threading.Thread(target=hand_thread_fn, args=(sensor, voice), daemon=True)
        )
    for t in threads:
        t.start()

    # ── Camera loop ──────────────────────────────────────────────────

    using_webcam_gaze = args.camera == "webcam"

    if using_webcam_gaze:
        print("[Demo] Webcam gaze tracking active – look left/right/up/down.\n")
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if sys.platform == "win32" else 0)
        if not cap.isOpened():
            print("[Demo] No camera found – running console-only mode")
            print("[Demo] Use keyboard controls in THIS terminal window.")
            print("[Demo] Press Q to stop.\n")
            try:
                while state.running:
                    if not _read_console_key(voice):
                        break
            except KeyboardInterrupt:
                pass
            state.running = False
            sensor.disconnect()
            mic.stop_stream()
            hud.stop()
            return
        print("[Demo] Camera active – open the window and use keyboard controls.\n")

    # ── Console input thread (works alongside OpenCV window) ─────────
    def _console_input_loop(v: UrduVoice) -> None:
        """Background thread: read keys from terminal via msvcrt/tty."""
        try:
            while state.running:
                if not _read_console_key(v):
                    state.running = False
                    break
        except Exception:
            pass

    console_thread = threading.Thread(target=_console_input_loop, args=(voice,), daemon=True)
    console_thread.start()

    while state.running:
        try:
            if using_webcam_gaze:
                frame = sensor.get_latest_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
            else:
                ret, frame = cap.read()
                if not ret:
                    break

            frame = cv2.flip(frame, 1)
            draw_overlay(frame)
            cv2.imshow("MindGrid Live Demo", frame)

            key = cv2.waitKey(1)
            if key != -1:
                if not handle_key(key, voice):
                    break
        except Exception as e:
            print(f"[Demo] Main loop error: {e}")
            time.sleep(0.1)

    # ── Cleanup ──────────────────────────────────────────────────────

    state.running = False
    if not using_webcam_gaze:
        cap.release()
    cv2.destroyAllWindows()
    sensor.disconnect()
    mic.stop_stream()
    hud.stop()
    print("\n[Demo] Shutdown complete.\n")


if __name__ == "__main__":
    main()
