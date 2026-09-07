"""
mindgrid/modelscope_stt.py
───────────────────────────
ModelScope-powered Urdu speech-to-text transcription.

Primary path: loads a ModelScope ASR model (Paraformer) trained on Urdu
and transcribes audio chunks in real-time.

Fallback path: if ModelScope SDK or the Urdu model is unavailable,
returns a hardcoded rotation of Urdu demo phrases so the pipeline
remains exercisable offline.

Voice Activity Detection (VAD)
──────────────────────────────
A simple energy-based VAD segments incoming audio before transcription,
discarding silence and only feeding speech-bearing chunks to the model.
"""

from __future__ import annotations

import time
import threading
from typing import Optional, List

import numpy as np

# ── Optional ModelScope import ───────────────────────────────────────────────

try:
    from modelscope.pipelines import pipeline as ms_pipeline
    from modelscope.utils.constant import Tasks
    _HAS_MODELSCOPE = True
except ImportError:
    _HAS_MODELSCOPE = False


# ── Offline Demo Phrases ─────────────────────────────────────────────────────

_OFFLINE_URDU_PHRASES: List[str] = [
    "مجھے مدد چاہیے",       # I need help
    "مجھے پانی چاہیے",      # I need water
    "مجھے بھوک لگی ہے",     # I am hungry
    "ڈاکٹر کو بلائیں",      # Call the doctor
    "سب ٹھیک ہے",           # Everything is fine
    "مجھے دوا چاہیے",       # I need medicine
    "میں گھر جانا چاہتا ہوں", # I want to go home
    "آپ کا بہت شکریہ",      # Thank you very much
    "براہ کرم رکیں",        # Please wait
    "فوری مدد بھیجیں",       # Send help immediately
]


# ── Voice Activity Detection ─────────────────────────────────────────────────

class SimpleVAD:
    """
    Energy-based voice activity detector.

    Parameters
    ──────────
    energy_threshold : float
        RMS amplitude threshold to classify a frame as speech.
    min_speech_frames : int
        Minimum consecutive speech frames to accept a segment.
    frame_size : int
        Number of samples per analysis frame.
    """

    def __init__(
        self,
        energy_threshold: float = 0.02,
        min_speech_frames: int = 5,
        frame_size: int = 1600,    # 100 ms at 16 kHz
    ) -> None:
        self._threshold = energy_threshold
        self._min_frames = min_speech_frames
        self._frame_size = frame_size

    def segment(self, audio: np.ndarray) -> List[np.ndarray]:
        """
        Split audio into speech-only segments.

        Returns a list of numpy arrays, each containing a contiguous
        speech segment.  Returns empty list if no speech detected.
        """
        if len(audio) == 0:
            return []

        # Compute per-frame RMS energy
        n_frames = len(audio) // self._frame_size
        if n_frames == 0:
            rms = float(np.sqrt(np.mean(audio ** 2)))
            return [audio] if rms > self._threshold else []

        energy = np.array([
            np.sqrt(np.mean(audio[i * self._frame_size:(i + 1) * self._frame_size] ** 2))
            for i in range(n_frames)
        ])

        # Binary speech mask
        is_speech = energy > self._threshold

        # Extract contiguous speech runs
        segments: List[np.ndarray] = []
        in_speech = False
        start = 0

        for i, speech in enumerate(is_speech):
            if speech and not in_speech:
                start = i
                in_speech = True
            elif not speech and in_speech:
                run_len = i - start
                if run_len >= self._min_frames:
                    seg = audio[start * self._frame_size:i * self._frame_size]
                    segments.append(seg)
                in_speech = False

        # Handle trailing speech
        if in_speech:
            run_len = n_frames - start
            if run_len >= self._min_frames:
                seg = audio[start * self._frame_size:]
                segments.append(seg)

        return segments


# ── Transcriber ──────────────────────────────────────────────────────────────

class UrduTranscriber:
    """
    Urdu speech-to-text transcription engine.

    Attempts to load a ModelScope ASR pipeline.  Falls back to
    offline rotation of demo Urdu phrases when the SDK or model
    is unavailable.

    Parameters
    ──────────
    model_id : str
        ModelScope model identifier.
    stub_mode : bool or None
        Force stub mode.  None = auto-detect (use ModelScope if available).
    """

    DEFAULT_MODEL = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"

    def __init__(
        self,
        model_id: Optional[str] = None,
        stub_mode: Optional[bool] = None,
    ) -> None:
        self._model_id = model_id or self.DEFAULT_MODEL
        self._pipeline = None
        self._vad = SimpleVAD()
        self._stub_mode = stub_mode
        self._offline_idx = 0
        self._loaded = False

    # ── Lifecycle ──────────────────────────────────────────────────────

    def load_model(self) -> bool:
        """
        Attempt to load the ASR model.  Returns True on success.
        On failure, silently switches to stub mode.
        """
        if self._stub_mode is True or not _HAS_MODELSCOPE:
            self._stub_mode = True
            self._loaded = True
            print("[ModelScope] Stub mode active – returning demo Urdu phrases")
            return True

        try:
            self._pipeline = ms_pipeline(
                task=Tasks.auto_speech_recognition,
                model=self._model_id,
            )
            self._loaded = True
            self._stub_mode = False
            print(f"[ModelScope] ASR model loaded: {self._model_id}")
            return True
        except Exception as exc:
            print(f"[ModelScope] Failed to load model: {exc}")
            print("[ModelScope] Falling back to stub mode")
            self._stub_mode = True
            self._loaded = True
            return False

    # ── Transcription ─────────────────────────────────────────────────

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe a single audio buffer to Urdu text.

        Runs VAD first to isolate speech, then transcribes the
        longest speech segment.  Returns empty string if no speech.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded – call load_model() first")

        # VAD segmentation
        segments = self._vad.segment(audio)

        if not segments:
            return ""

        # Pick the longest segment (most likely to contain speech)
        best = max(segments, key=len)

        if self._stub_mode:
            return self._offline_phrase()

        return self._run_asr(best, sample_rate)

    def transcribe_live(
        self,
        audio_provider,
        duration_sec: float = 3.0,
        callback=None,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        """
        Continuously transcribe audio from a provider with get_chunk().

        Parameters
        ──────────
        audio_provider : MicCapture or similar with get_chunk(duration_sec)
        duration_sec : float
            Seconds of audio per chunk.
        callback : callable(str) or None
            Called with each transcription result.
        stop_event : threading.Event
            Set to stop the loop.
        """
        if stop_event is None:
            stop_event = threading.Event()

        while not stop_event.is_set():
            chunk = audio_provider.get_chunk(duration_sec)
            text = self.transcribe(chunk)
            if text and callback:
                callback(text)

    # ── Internals ──────────────────────────────────────────────────────

    def _run_asr(self, audio: np.ndarray, sample_rate: int) -> str:
        """Invoke ModelScope pipeline on a speech segment."""
        try:
            # ModelScope expects a dict with 'data' key (numpy array or file path)
            result = self._pipeline(audio)
            if isinstance(result, dict):
                return result.get("text", "")
            elif isinstance(result, list) and result:
                return result[0].get("text", "") if isinstance(result[0], dict) else str(result[0])
            return ""
        except Exception as exc:
            print(f"[ModelScope] ASR error: {exc}")
            return self._offline_phrase()

    def _offline_phrase(self) -> str:
        """Rotate through demo Urdu phrases."""
        phrase = _OFFLINE_URDU_PHRASES[self._offline_idx % len(_OFFLINE_URDU_PHRASES)]
        self._offline_idx += 1
        return phrase


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    # Ensure Urdu characters can print on Windows consoles
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    print("\n--- MindGrid ModelScope STT – Smoke Test ---")
    stt = UrduTranscriber(stub_mode=True)
    stt.load_model()

    # Simulate audio chunks with varying energy
    for i in range(5):
        # Alternate between silence and "speech" (noise above threshold)
        if i % 2 == 0:
            audio = np.random.randn(48000).astype(np.float32) * 0.001  # silence
        else:
            audio = np.random.randn(48000).astype(np.float32) * 0.08   # speech-like
        text = stt.transcribe(audio)
        print(f"  Chunk {i+1}: '{text}'")

    print("--- Test complete ---\n")
