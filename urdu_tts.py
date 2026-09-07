"""
mindgrid/urdu_tts.py
─────────────────────
Urdu text-to-speech synthesis for MindGrid audio output.

Primary engine : Alibaba Cloud DashScope TTS (CosyVoice / Sambert)
Fallback 1     : gTTS (Google Text-to-Speech) with lang='ur'
Fallback 2     : pyttsx3 offline engine with rate/voice tuning

Caches recently spoken phrases to avoid redundant API calls.
All playback is non-blocking (threaded).
"""

from __future__ import annotations

import io
import os
import sys
import time
import hashlib
import platform
import tempfile
import threading
import subprocess
from collections import OrderedDict
from typing import Optional

import numpy as np

# ── Optional imports ─────────────────────────────────────────────────────────

try:
    import pyttsx3
    _HAS_PYTTSX3 = True
except ImportError:
    _HAS_PYTTSX3 = False

try:
    from gtts import gTTS
    _HAS_GTTS = True
except ImportError:
    _HAS_GTTS = False

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


# ── Audio Playback ───────────────────────────────────────────────────────────

# Try to import playback libraries once at module level
try:
    import sounddevice as _sd
    import soundfile as _sf
    _HAS_SOUNDDEVICE = True
except ImportError:
    _HAS_SOUNDDEVICE = False

try:
    import pygame
    _HAS_PYGAME = True
except ImportError:
    _HAS_PYGAME = False


def _play_audio_file(path: str) -> None:
    """Play a WAV/MP3 file silently (no media player popup)."""
    # Method 1: sounddevice + soundfile (best, no external player)
    if _HAS_SOUNDDEVICE:
        try:
            data, sr = _sf.read(path)
            _sd.play(data, sr)
            _sd.wait()
            return
        except Exception:
            pass

    # Method 2: pygame mixer (already imported for HUD, no popup)
    if _HAS_PYGAME:
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
            return
        except Exception:
            pass

    # Method 3: Windows hidden playback via PowerShell (no visible window)
    system = platform.system()
    try:
        if system == "Windows":
            # Use PowerShell with hidden window style
            ps_script = (
                f'$p = New-Object System.Media.SoundPlayer;'
                f'$p.SoundLocation = "{path}";'
                f'$p.PlaySync()'
            )
            subprocess.Popen(
                ["powershell", "-WindowStyle", "Hidden", "-Command", ps_script],
                creationflags=subprocess.CREATE_NO_WINDOW,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(2.0)
        elif system == "Darwin":
            subprocess.Popen(["afplay", path])
        else:
            subprocess.Popen(["aplay", path], stderr=subprocess.DEVNULL)
    except Exception:
        pass


# ── TTS Engine ───────────────────────────────────────────────────────────────

class UrduVoice:
    """
    Multi-backend Urdu text-to-speech synthesiser.

    Parameters
    ──────────
    dashscope_api_key : str or None
        Alibaba Cloud DashScope API key for CosyVoice / Sambert TTS.
    cache_size : int
        Number of recently spoken phrases to cache (LRU).
    engine_priority : list[str]
        Order of TTS engines to try: 'dashscope', 'gtts', 'pyttsx3'.
    """

    DEFAULT_VOICE = "longxiaochun"   # DashScope Sambert voice (supports multilingual)
    DASHSCOPE_TTS_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

    def __init__(
        self,
        dashscope_api_key: Optional[str] = None,
        cache_size: int = 50,
        engine_priority: Optional[list[str]] = None,
    ) -> None:
        self._api_key = dashscope_api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._cache_size = cache_size
        self._priority = engine_priority or ["dashscope", "gtts", "pyttsx3"]
        self._speaking = False

    # ── Public API ─────────────────────────────────────────────────────

    def speak(self, urdu_text: str) -> None:
        """Synthesize and play Urdu text. Non-blocking (threaded)."""
        if not urdu_text or not urdu_text.strip():
            return
        threading.Thread(target=self._speak_worker, args=(urdu_text,), daemon=True).start()

    def synthesize_to_file(self, urdu_text: str, path: str) -> bool:
        """Synthesize Urdu text and save to an audio file. Returns True on success."""
        for engine_name in self._priority:
            try:
                if engine_name == "dashscope" and self._api_key:
                    return self._tts_dashscope(urdu_text, path)
                elif engine_name == "gtts" and _HAS_GTTS:
                    return self._tts_gtts(urdu_text, path)
                elif engine_name == "pyttsx3" and _HAS_PYTTSX3:
                    return self._tts_pyttsx3(urdu_text, path)
            except Exception as exc:
                print(f"[TTS] {engine_name} failed: {exc}")
                continue

        print("[TTS] All engines failed")
        return False

    # ── Worker Thread ──────────────────────────────────────────────────

    def _speak_worker(self, urdu_text: str) -> None:
        while self._speaking:
            time.sleep(0.05)

        self._speaking = True
        try:
            # Check cache
            cache_key = self._hash(urdu_text)
            cached_path = self._cache.get(cache_key)
            if cached_path and os.path.isfile(cached_path):
                _play_audio_file(cached_path)
                return

            # Synthesize to temp file
            tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            tmp_path = tmp.name
            tmp.close()

            if self.synthesize_to_file(urdu_text, tmp_path):
                # Cache it
                self._cache[cache_key] = tmp_path
                if len(self._cache) > self._cache_size:
                    oldest_key, oldest_path = self._cache.popitem(last=False)
                    try:
                        os.unlink(oldest_path)
                    except OSError:
                        pass

                _play_audio_file(tmp_path)
        except Exception as exc:
            print(f"[TTS] Speak error: {exc}")
        finally:
            self._speaking = False

    # ── Engine Implementations ─────────────────────────────────────────

    def _tts_dashscope(self, text: str, path: str) -> bool:
        """Alibaba Cloud DashScope CosyVoice / Sambert TTS."""
        if not _HAS_REQUESTS or not self._api_key:
            return False

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "sambert-zhichu-v1",
            "input": {"text": text},
            "parameters": {"sample_rate": 16000, "format": "mp3"},
        }

        resp = _requests.post(self.DASHSCOPE_TTS_URL, json=payload, headers=headers, timeout=15)
        if resp.status_code != 200:
            print(f"[TTS] DashScope HTTP {resp.status_code}: {resp.text[:200]}")
            return False

        # DashScope may return audio URL or binary
        content_type = resp.headers.get("Content-Type", "")
        if "audio" in content_type:
            with open(path, "wb") as f:
                f.write(resp.content)
            return True

        # JSON response with audio URL
        data = resp.json()
        audio_url = data.get("output", {}).get("audio", "")
        if audio_url:
            audio_resp = _requests.get(audio_url, timeout=15)
            with open(path, "wb") as f:
                f.write(audio_resp.content)
            return True

        return False

    def _tts_gtts(self, text: str, path: str) -> bool:
        """Google Text-to-Speech fallback."""
        if not _HAS_GTTS:
            return False
        tts = gTTS(text=text, lang="ur", slow=False)
        tts.save(path)
        return os.path.isfile(path) and os.path.getsize(path) > 0

    def _tts_pyttsx3(self, text: str, path: str) -> bool:
        """Offline pyttsx3 engine (limited Urdu support)."""
        if not _HAS_PYTTSX3:
            return False
        engine = pyttsx3.init()
        engine.setProperty("rate", 130)
        engine.save_to_file(text, path)
        engine.runAndWait()
        engine.stop()
        return os.path.isfile(path) and os.path.getsize(path) > 0

    # ── Utilities ──────────────────────────────────────────────────────

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys as _sys
    if hasattr(_sys.stdout, 'reconfigure'):
        _sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    print("\n--- MindGrid Urdu TTS – Smoke Test ---")

    voice = UrduVoice(engine_priority=["gtts", "pyttsx3"])

    test_phrases = [
        "مجھے مدد چاہیے",
        "سب ٹھیک ہے",
    ]

    for phrase in test_phrases:
        print(f"  Speaking: {phrase}")
        voice.speak(phrase)
        time.sleep(3.0)

    print("--- Test complete ---\n")
