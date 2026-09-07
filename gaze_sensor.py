"""
mindgrid/gaze_sensor.py
────────────────────────
Eye-gaze signal acquisition from dry conductive silicon pad.

Abstracts the ADC hardware (ADS1299) behind a uniform interface.
Current build ships a SimulatedGazeSource that generates realistic
EOG-like waveforms (saccades, fixations, blink artifacts) so the
full software pipeline can be exercised on Windows without hardware.

Future: swap in RealGazeSource for SPI/I2C ADC on Raspberry Pi.
"""

from __future__ import annotations

import time
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class GazeSignal:
    """One frame of 4-channel EOG differential micro-volt readings."""
    timestamp: float                         # epoch seconds
    channels: np.ndarray                     # shape (4,) – [h_left, h_right, v_up, v_down] in µV
    impedance_ok: bool = True                # pad contact quality flag

    @property
    def horizontal(self) -> float:
        """Net horizontal signal (right minus left)."""
        return float(self.channels[1] - self.channels[0])

    @property
    def vertical(self) -> float:
        """Net vertical signal (up minus down)."""
        return float(self.channels[2] - self.channels[3])

    @property
    def amplitude(self) -> float:
        """Peak absolute amplitude across all channels."""
        return float(np.max(np.abs(self.channels)))


# ── Abstract Sensor Interface ────────────────────────────────────────────────

class GazeSource(ABC):
    """Abstract base for all gaze signal providers."""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def read_frame(self) -> GazeSignal: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @property
    @abstractmethod
    def sample_rate(self) -> float: ...


# ── Simulated EOG Source ─────────────────────────────────────────────────────

class SimulatedGazeSource(GazeSource):
    """
    Generates synthetic 4-channel EOG waveforms at a configurable sample rate.

    Signal characteristics
    ──────────────────────
    • Fixation baseline  : 10-20 µV (low-frequency drift)
    • Saccade pulse      : 50-200 µV (sharp transient, ~30-80 ms)
    • Blink artifact     : 300-600 µV (biphasic spike, ~200 ms)
    • Ambient noise      : 1-5 µV Gaussian
    """

    def __init__(self, rate: float = 60.0) -> None:
        self._rate = rate
        self._connected = False
        self._t0: float = 0.0
        self._frame_idx: int = 0

        # Saccade scheduling
        self._next_saccade_frame: int = 0
        self._saccade_direction: str = "center"
        self._saccade_remaining: int = 0

        # Blink scheduling
        self._next_blink_frame: int = 0
        self._blink_remaining: int = 0

        self._schedule_next_saccade(0)
        self._schedule_next_blink(0)

    # ── Public API ───────────────────────────────────────────────────────

    @property
    def sample_rate(self) -> float:
        return self._rate

    def connect(self) -> None:
        self._t0 = time.time()
        self._connected = True
        self._frame_idx = 0
        print("[GazeSensor] Simulated silicon pad connected (EOG mode)")

    def disconnect(self) -> None:
        self._connected = False
        print("[GazeSensor] Simulated silicon pad disconnected")

    # ── Internal helpers ─────────────────────────────────────────────────

    def _schedule_next_saccade(self, after_frame: int) -> None:
        gap = random.randint(int(self._rate * 1.5), int(self._rate * 4.0))
        self._next_saccade_frame = after_frame + gap
        self._saccade_direction = random.choice(["left", "right", "up", "down", "center"])
        self._saccade_remaining = 0

    def _schedule_next_blink(self, after_frame: int) -> None:
        gap = random.randint(int(self._rate * 3.0), int(self._rate * 8.0))
        self._next_blink_frame = after_frame + gap

    def _maybe_trigger(self) -> None:
        """Called implicitly – check scheduled events."""
        if self._frame_idx >= self._next_saccade_frame and self._saccade_remaining == 0:
            self._saccade_remaining = 3  # ~50 ms at 60 Hz
        if self._frame_idx >= self._next_blink_frame and self._blink_remaining == 0:
            self._blink_remaining = 6  # ~100 ms at 60 Hz

    def read_frame(self) -> GazeSignal:
        if not self._connected:
            raise RuntimeError("Sensor not connected – call connect() first")

        # Check scheduled events before generating signal
        self._maybe_trigger()

        ts = self._t0 + self._frame_idx / self._rate

        # Baseline drift (slow sine, 0.1 Hz)
        drift = 12.0 * math.sin(2 * math.pi * 0.1 * (self._frame_idx / self._rate))

        # Gaussian noise floor
        noise = np.random.normal(0, 2.5, size=4)

        channels = np.array([drift, drift, drift, drift]) + noise

        # ── Inject saccade pulse ─────────────────────────────────────────
        if self._saccade_remaining > 0:
            progress = 1.0 - (self._saccade_remaining / 3)
            pulse_amp = random.uniform(80, 160) * math.sin(math.pi * progress)
            channels += self._direction_vector(self._saccade_direction) * pulse_amp
            self._saccade_remaining -= 1
            if self._saccade_remaining == 0:
                self._schedule_next_saccade(self._frame_idx + 1)

        # ── Inject blink artifact ────────────────────────────────────────
        if self._blink_remaining > 0:
            blink_phase = 1.0 - (self._blink_remaining / 6)
            blink_amp = random.uniform(350, 550) * math.sin(math.pi * blink_phase)
            sign = 1.0 if blink_phase < 0.5 else -0.6
            channels[2] += blink_amp * sign
            channels[3] += blink_amp * sign
            self._blink_remaining -= 1
            if self._blink_remaining == 0:
                self._schedule_next_blink(self._frame_idx + 1)

        self._frame_idx += 1
        return GazeSignal(timestamp=ts, channels=channels, impedance_ok=True)

    @staticmethod
    def _direction_vector(direction: str) -> np.ndarray:
        """Unit-ish vector for saccade direction in [hL, hR, vU, vD] space."""
        vectors = {
            "left":   np.array([ 1.0, -1.0,  0.0,  0.0]),
            "right":  np.array([-1.0,  1.0,  0.0,  0.0]),
            "up":     np.array([ 0.0,  0.0,  1.0, -1.0]),
            "down":   np.array([ 0.0,  0.0, -1.0,  1.0]),
            "center": np.array([ 0.2,  0.2,  0.2,  0.2]),
        }
        return vectors.get(direction, vectors["center"])


# ── Sensor Facade ────────────────────────────────────────────────────────────

class SiliconPadSensor:
    """
    High-level facade wrapping a GazeSource implementation.

    Usage
    ─────
        sensor = SiliconPadSensor()          # defaults to SimulatedGazeSource
        sensor.connect()
        for _ in range(300):
            sig = sensor.read_frame()
        sensor.disconnect()
    """

    def __init__(self, source: Optional[GazeSource] = None) -> None:
        self._source = source or SimulatedGazeSource(rate=60.0)
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def sample_rate(self) -> float:
        return self._source.sample_rate

    def connect(self) -> None:
        self._source.connect()
        self._connected = True

    def read_frame(self) -> GazeSignal:
        return self._source.read_frame()

    def disconnect(self) -> None:
        self._source.disconnect()
        self._connected = False


# ── Standalone smoke test ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n--- MindGrid Gaze Sensor – Smoke Test ---")
    sensor = SiliconPadSensor()
    sensor.connect()

    for i in range(30):
        sig = sensor.read_frame()
        tag = ""
        if sig.amplitude > 200:
            tag = " << BLINK"
        elif abs(sig.horizontal) > 50:
            tag = f" << SACCADE-H ({sig.horizontal:+.1f} µV)"
        elif abs(sig.vertical) > 50:
            tag = f" << SACCADE-V ({sig.vertical:+.1f} µV)"
        print(f"  Frame {i:03d} | H={sig.horizontal:+7.1f}  V={sig.vertical:+7.1f}  "
              f"Amp={sig.amplitude:6.1f} µV{tag}")
        time.sleep(1 / sensor.sample_rate)

    sensor.disconnect()
    print("--- Test complete ---\n")
