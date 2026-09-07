"""
mindgrid/gaze_classifier.py
────────────────────────────
Converts raw micro-volt gaze signals from the silicon pad into
discrete gaze intents: direction (left/right/up/down/center),
action type (blink/dwell/saccade), and confidence score.

Signal processing pipeline
──────────────────────────
  1. Bandpass filter (0.5–30 Hz) to reject DC drift and mains hum
  2. Blink detection via vertical amplitude threshold (> 400 µV)
  3. Saccade detection via horizontal/vertical velocity threshold
  4. Gaze direction from steady-state channel ratios
  5. Temporal smoothing via GazeIntentBuffer
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
from scipy.signal import butter, sosfiltfilt

from gaze_sensor import GazeSignal


# ── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class GazeIntent:
    """A classified gaze event emitted by the classifier."""
    direction: str          # "left" | "right" | "up" | "down" | "center"
    action: str             # "fixation" | "saccade" | "blink" | "dwell"
    confidence: float       # 0.0 – 1.0
    timestamp: float        # epoch seconds


# ── Signal Processing ────────────────────────────────────────────────────────

def _bandpass_sos(low: float, high: float, fs: float, order: int = 4):
    """Return second-order-sections coefficients for a Butterworth bandpass."""
    nyq = fs / 2.0
    return butter(order, [low / nyq, high / nyq], btype="band", output="sos")


# ── Classifier ───────────────────────────────────────────────────────────────

# Thresholds (micro-volts unless noted)
_BLINK_THRESHOLD = 400.0      # vertical amplitude for blink
_SACCADE_VELOCITY = 60.0       # µV/frame velocity for saccade onset
_FIXATION_FRAMES = 8           # frames of low-activity to declare fixation
_DWELL_FRAMES = 15             # sustained fixation → dwell-select (was 30)
_DIRECTION_THRESHOLD = 25.0    # µV net signal to declare a direction


class GazeIntentClassifier:
    """
    Sliding-window gaze intent classifier.

    Parameters
    ──────────
    window_size : int
        Number of frames to buffer (default 30 @ 60 Hz ≈ 500 ms).
    sample_rate : float
        Expected sample rate of the gaze source (Hz).
    """

    def __init__(self, window_size: int = 30, sample_rate: float = 60.0) -> None:
        self._window_size = window_size
        self._fs = sample_rate
        self._buffer: deque[GazeSignal] = deque(maxlen=window_size)

        # Pre-compute bandpass filter coefficients
        # High cutoff must stay below Nyquist (fs/2); clamp to 90% of Nyquist
        high_cutoff = min(30.0, self._fs / 2.0 * 0.9)
        self._sos = _bandpass_sos(0.5, high_cutoff, self._fs)

        # State tracking
        self._fixation_counter: int = 0
        self._prev_horizontal: float = 0.0
        self._prev_vertical: float = 0.0

        # Cooldown counters (frames to suppress after an event)
        self._blink_cooldown: int = 0
        self._dwell_cooldown: int = 0
        self._fixation_cooldown: int = 0

    # ── Public API ───────────────────────────────────────────────────────

    def push(self, signal: GazeSignal) -> Optional[GazeIntent]:
        """
        Feed one GazeSignal frame.  Returns a GazeIntent when a meaningful
        event is detected, or None while accumulating.
        """
        self._buffer.append(signal)
        if len(self._buffer) < self._window_size:
            return None  # not enough data yet

        # Stack channels into (window, 4) matrix
        raw = np.array([s.channels for s in self._buffer])

        # Bandpass each channel
        filtered = np.column_stack([
            sosfiltfilt(self._sos, raw[:, ch]) for ch in range(4)
        ])

        # Latest filtered sample
        h_left, h_right, v_up, v_down = filtered[-1]
        net_h = float(h_right - h_left)
        net_v = float(v_up - v_down)

        # Decrement cooldowns
        if self._blink_cooldown > 0:
            self._blink_cooldown -= 1
        if self._dwell_cooldown > 0:
            self._dwell_cooldown -= 1
        if self._fixation_cooldown > 0:
            self._fixation_cooldown -= 1

        # ── Blink detection ──────────────────────────────────────────
        vert_peak = float(np.max(np.abs(filtered[:, 2] + filtered[:, 3])))
        if vert_peak > _BLINK_THRESHOLD and self._blink_cooldown == 0:
            self._fixation_counter = 0
            self._blink_cooldown = int(self._fs * 1.0)  # 1 second cooldown
            self._dwell_cooldown = max(self._dwell_cooldown, int(self._fs * 0.5))  # 0.5 sec after blink
            return GazeIntent(
                direction="center",
                action="blink",
                confidence=min(vert_peak / 600.0, 1.0),
                timestamp=signal.timestamp,
            )

        # ── Saccade detection (velocity) ─────────────────────────────
        vel_h = abs(net_h - self._prev_horizontal)
        vel_v = abs(net_v - self._prev_vertical)
        self._prev_horizontal = net_h
        self._prev_vertical = net_v

        if vel_h > _SACCADE_VELOCITY or vel_v > _SACCADE_VELOCITY:
            self._fixation_counter = 0
            direction = self._classify_direction(net_h, net_v)
            return GazeIntent(
                direction=direction,
                action="saccade",
                confidence=min(max(vel_h, vel_v) / 120.0, 1.0),
                timestamp=signal.timestamp,
            )

        # ── Fixation / Dwell ─────────────────────────────────────────
        activity = float(np.std(filtered[-6:], axis=0).mean())
        if activity < 15.0:
            self._fixation_counter += 1
        else:
            if self._fixation_counter > 3:
                print(f"[Classifier] fixation reset: activity={activity:.1f}, was at {self._fixation_counter}")
            self._fixation_counter = 0

        # Debug: show fixation progress every 5 frames
        if self._fixation_counter > 0 and self._fixation_counter % 5 == 0:
            _dir = self._classify_direction(net_h, net_v)
            print(f"[Classifier] fixation={self._fixation_counter}/{_DWELL_FRAMES} dir={_dir} dwell_cd={self._dwell_cooldown}")

        if self._fixation_counter >= _DWELL_FRAMES and self._dwell_cooldown == 0:
            self._fixation_counter = 0  # reset to avoid repeated dwell
            self._dwell_cooldown = int(self._fs * 1.5)  # 1.5 sec cooldown (was 3.0)
            self._fixation_cooldown = int(self._fs * 0.5)
            direction = self._classify_direction(net_h, net_v)
            return GazeIntent(
                direction=direction,
                action="dwell",
                confidence=0.85,
                timestamp=signal.timestamp,
            )

        if self._fixation_counter >= _FIXATION_FRAMES and self._fixation_cooldown == 0:
            self._fixation_cooldown = int(self._fs * 1.5)  # 1.5 sec cooldown
            direction = self._classify_direction(net_h, net_v)
            # Skip "center" fixations – they are noise when eyes are at rest
            if direction == "center":
                return None
            return GazeIntent(
                direction=direction,
                action="fixation",
                confidence=0.70,
                timestamp=signal.timestamp,
            )

        return None

    # ── Internals ──────────────────────────────────────────────────────

    @staticmethod
    def _classify_direction(net_h: float, net_v: float) -> str:
        if abs(net_h) > abs(net_v):
            if net_h > _DIRECTION_THRESHOLD:
                return "right"
            elif net_h < -_DIRECTION_THRESHOLD:
                return "left"
        else:
            if net_v > _DIRECTION_THRESHOLD:
                return "up"
            elif net_v < -_DIRECTION_THRESHOLD:
                return "down"
        return "center"


# ── Temporal Smoothing Buffer ────────────────────────────────────────────────

class GazeIntentBuffer:
    """
    Smooths classifier output by requiring N consecutive matching
    intents before emitting – analogous to GestureBuffer.

    Parameters
    ──────────
    hold_count : int
        Number of consecutive identical intents to confirm (default 2).
    """

    def __init__(self, hold_count: int = 2) -> None:
        self._hold = hold_count
        self._current: Optional[str] = None  # composite key: direction+action
        self._counter: int = 0

    def update(self, intent: Optional[GazeIntent]) -> Optional[GazeIntent]:
        if intent is None:
            return None

        key = f"{intent.direction}:{intent.action}"

        # Blinks and dwells emit immediately (they are rare, deliberate events)
        if intent.action in ("blink", "dwell"):
            self._current = None
            self._counter = 0
            return intent

        if key == self._current:
            self._counter += 1
        else:
            self._current = key
            self._counter = 1

        if self._counter >= self._hold:
            self._counter = 0
            self._current = None
            return intent

        return None


# ── Intent → Urdu Phrase Mapping ─────────────────────────────────────────────

# Maps (direction, action) tuples to Urdu vocabulary tokens.
# This is the gaze-to-language bridge; expand as needed.
GAZE_PHRASE_MAP: dict[tuple[str, str], str] = {
    # Dwell selects (most common intents)
    ("left",  "dwell"):  "مجھے پانی چاہیے",         # I need water
    ("right", "dwell"):  "مجھے مدد چاہیے",          # I need help
    ("up",    "dwell"):  "ڈاکٹر کو بلائیں",          # Call a doctor
    ("down",  "dwell"):  "مجھے بھوک لگی ہے",        # I am hungry
    ("center", "dwell"): "سب ٹھیک ہے",               # Everything is fine

    # Blink confirms
    ("center", "blink"): "آپ کا بہت شکریہ",          # Thank you

    # Fixation holds (shorter intent)
    ("left",  "fixation"): "مجھے دوا چاہیے",         # I need medicine
    ("right", "fixation"): "میں گھر جانا چاہتا ہوں", # I want to go home
    ("up",    "fixation"): "فوری مدد بھیجیں",        # Immediately / urgent
    ("down",  "fixation"): "براہ کرم رکیں",          # Wait / stop
}


def intent_to_token(intent: GazeIntent) -> Optional[str]:
    """Resolve a GazeIntent to a Urdu vocabulary token, or None."""
    return GAZE_PHRASE_MAP.get((intent.direction, intent.action))


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    from gaze_sensor import SiliconPadSensor

    print("\n--- MindGrid Gaze Classifier – Live Test ---")
    sensor = SiliconPadSensor()
    sensor.connect()

    classifier = GazeIntentClassifier(window_size=30, sample_rate=sensor.sample_rate)
    buffer = GazeIntentBuffer(hold_count=2)

    try:
        for i in range(300):
            sig = sensor.read_frame()
            raw_intent = classifier.push(sig)
            stable = buffer.update(raw_intent)

            if stable:
                token = intent_to_token(stable)
                tag = f" -> {token}" if token else ""
                print(f"  [{i:03d}] INTENT  {stable.action:10s} {stable.direction:6s}  "
                      f"(conf {stable.confidence:.2f}){tag}")

            time.sleep(1 / sensor.sample_rate)
    except KeyboardInterrupt:
        pass

    sensor.disconnect()
    print("--- Test complete ---\n")
