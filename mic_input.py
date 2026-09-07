"""
mindgrid/mic_input.py
──────────────────────
I2S microphone audio capture abstraction.

On Windows (development): captures from the system default microphone
via the sounddevice library at 16 kHz mono PCM.

On Raspberry Pi (production): swap to pyaudio reading from ALSA I2S
device (INMP441 / SPH0645) without changing the public API.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

import numpy as np

try:
    import sounddevice as sd
    _HAS_SD = True
except ImportError:
    _HAS_SD = False


# ── Data Model ───────────────────────────────────────────────────────────────

SAMPLE_RATE = 16000       # 16 kHz – standard for speech ASR
CHANNELS = 1              # mono
DTYPE = "float32"         # normalised –1.0 to 1.0


# ── Abstract Interface ───────────────────────────────────────────────────────

class MicSource:
    """Abstract microphone source."""

    def start_stream(self) -> None:
        raise NotImplementedError

    def get_chunk(self, duration_sec: float = 3.0) -> np.ndarray:
        raise NotImplementedError

    def stop_stream(self) -> None:
        raise NotImplementedError

    @property
    def is_active(self) -> bool:
        raise NotImplementedError


# ── System Microphone (Windows / macOS / Linux desktop) ─────────────────────

class SystemMicSource(MicSource):
    """
    Captures audio from the OS default microphone using sounddevice.

    The stream callback pushes incoming frames into a thread-safe deque.
    get_chunk() drains the deque to return a contiguous numpy array of
    the requested duration.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, device: Optional[int] = None) -> None:
        self._sr = sample_rate
        self._device = device
        self._stream: Optional[sd.InputStream] = None
        self._buffer: deque[np.ndarray] = deque()
        self._lock = threading.Lock()
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    def start_stream(self) -> None:
        if not _HAS_SD:
            print("[MicInput] sounddevice not installed – running in stub mode")
            self._active = True
            return

        self._stream = sd.InputStream(
            samplerate=self._sr,
            channels=CHANNELS,
            dtype=DTYPE,
            device=self._device,
            callback=self._audio_callback,
            blocksize=1024,
        )
        self._stream.start()
        self._active = True
        print(f"[MicInput] System microphone active @ {self._sr} Hz")

    def get_chunk(self, duration_sec: float = 3.0) -> np.ndarray:
        """Return ~duration_sec of audio as a 1-D float32 numpy array."""
        if not _HAS_SD or not self._active:
            # Stub: return silence
            return np.zeros(int(self._sr * duration_sec), dtype=np.float32)

        target_samples = int(self._sr * duration_sec)
        collected: list[np.ndarray] = []
        total = 0

        deadline = time.time() + duration_sec + 0.5  # generous timeout
        while total < target_samples and time.time() < deadline:
            with self._lock:
                if self._buffer:
                    chunk = self._buffer.popleft()
                    collected.append(chunk)
                    total += len(chunk)
                else:
                    pass
            if total < target_samples:
                time.sleep(0.01)

        if not collected:
            return np.zeros(target_samples, dtype=np.float32)

        audio = np.concatenate(collected)[:target_samples]
        # Pad if we fell short
        if len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)), mode="constant")
        return audio

    def stop_stream(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._active = False
        self._buffer.clear()
        print("[MicInput] Microphone stream stopped")

    # ── Callback ───────────────────────────────────────────────────────

    def _audio_callback(self, indata: np.ndarray, frames: int,
                        time_info, status) -> None:
        if status:
            pass  # ignore xruns etc.
        with self._lock:
            self._buffer.append(indata[:, 0].copy())


# ── Stub / Silent Source (no hardware) ───────────────────────────────────────

class SilentMicSource(MicSource):
    """
    Produces silence or synthetic tone for testing without any microphone.
    Useful when sounddevice is unavailable or the system has no audio input.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self._sr = sample_rate
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    def start_stream(self) -> None:
        self._active = True
        print("[MicInput] Silent stub source active (no real audio)")

    def get_chunk(self, duration_sec: float = 3.0) -> np.ndarray:
        return np.zeros(int(self._sr * duration_sec), dtype=np.float32)

    def stop_stream(self) -> None:
        self._active = False


# ── Convenience Facade ───────────────────────────────────────────────────────

class MicCapture:
    """
    High-level microphone facade.

    Automatically selects SystemMicSource when sounddevice is available,
    otherwise falls back to SilentMicSource.
    """

    def __init__(self, source: Optional[MicSource] = None) -> None:
        if source is not None:
            self._source = source
        elif _HAS_SD:
            self._source = SystemMicSource()
        else:
            self._source = SilentMicSource()

    @property
    def is_active(self) -> bool:
        return self._source.is_active

    def start_stream(self) -> None:
        self._source.start_stream()

    def get_chunk(self, duration_sec: float = 3.0) -> np.ndarray:
        return self._source.get_chunk(duration_sec)

    def stop_stream(self) -> None:
        self._source.stop_stream()


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n--- MindGrid Mic Input – Smoke Test ---")
    mic = MicCapture()
    mic.start_stream()

    for i in range(3):
        chunk = mic.get_chunk(duration_sec=1.0)
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        print(f"  Chunk {i+1}: {len(chunk)} samples, RMS={rms:.6f}")

    mic.stop_stream()
    print("--- Test complete ---\n")
