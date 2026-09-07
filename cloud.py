"""
mindgrid/cloud.py
─────────────────
Alibaba Cloud integration hub for MindGrid.

Modules
───────
• MindGridConfig   – loads credentials from .env via python-dotenv
• URDU_VOCABULARY_DB – expanded Urdu phrase database (50+ entries)
• reconstruct_phrase – offline fallback phrase lookup
• AlibabaCloudClient – OSS upload, DashScope API proxy
"""

from __future__ import annotations

import io
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class MindGridConfig:
    """Centralised configuration loaded from environment / .env file."""

    # Alibaba Cloud OSS
    OSS_SYNC_ENABLED: bool = False
    OSS_ACCESS_KEY_ID: str = ""
    OSS_ACCESS_KEY_SECRET: str = ""
    OSS_ENDPOINT: str = ""
    OSS_BUCKET: str = ""
    OSS_PREFIX: str = "mindgrid/"

    # DashScope (TTS / NLP)
    DASHSCOPE_API_KEY: str = ""
    DASHSCOPE_TTS_MODEL: str = "sambert-zhichu-v1"
    DASHSCOPE_TTS_VOICE: str = "longxiaochun"

    # ModelScope
    MODELSCOPE_ASR_MODEL: str = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"

    @classmethod
    def from_env(cls) -> "MindGridConfig":
        return cls(
            OSS_SYNC_ENABLED=os.getenv("OSS_SYNC_ENABLED", "false").lower() == "true",
            OSS_ACCESS_KEY_ID=os.getenv("OSS_ACCESS_KEY_ID", ""),
            OSS_ACCESS_KEY_SECRET=os.getenv("OSS_ACCESS_KEY_SECRET", ""),
            OSS_ENDPOINT=os.getenv("OSS_ENDPOINT", ""),
            OSS_BUCKET=os.getenv("OSS_BUCKET", ""),
            OSS_PREFIX=os.getenv("OSS_PREFIX", "mindgrid/"),
            DASHSCOPE_API_KEY=os.getenv("DASHSCOPE_API_KEY", ""),
            DASHSCOPE_TTS_MODEL=os.getenv("DASHSCOPE_TTS_MODEL", "sambert-zhichu-v1"),
            DASHSCOPE_TTS_VOICE=os.getenv("DASHSCOPE_TTS_VOICE", "longxiaochun"),
            MODELSCOPE_ASR_MODEL=os.getenv("MODELSCOPE_ASR_MODEL",
                "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"),
        )


# Singleton config
config = MindGridConfig.from_env()


# ── Urdu Vocabulary Database ────────────────────────────────────────────────

URDU_VOCABULARY_DB: Dict[str, dict] = {
    # ── Emergency / Medical ────────────────────────────────────────────
    "مجھے مدد چاہیے":    {"display": "مجھے مدد چاہیے!",        "speech": "I need help!",              "category": "emergency"},
    "فوری مدد بھیجیں":   {"display": "فوری مدد بھیجیں!",        "speech": "Send help immediately!",    "category": "emergency"},
    "ڈاکٹر کو بلائیں":   {"display": "ڈاکٹر کو بلائیں!",        "speech": "Please call a doctor!",     "category": "medical"},
    "مجھے دوا چاہیے":    {"display": "مجھے دوا چاہیے۔",         "speech": "I need medicine.",           "category": "medical"},
    "dard":     {"display": "مجھے درد ہو رہا ہے۔",     "speech": "I am in pain.",              "category": "medical"},
    "saans":    {"display": "مجھے سانس لینے میں مشکل ہے۔", "speech": "I have difficulty breathing.", "category": "medical"},
    "ambulance":{"display": "ایمبولینس بلائیں!",       "speech": "Call an ambulance!",         "category": "emergency"},
    "behosh":   {"display": "میں بے ہوش ہونے والا ہوں۔","speech": "I am about to faint.",      "category": "medical"},
    "khon":     {"display": "خون نکل رہا ہے۔",         "speech": "I am bleeding.",             "category": "medical"},
    "bukhar":   {"display": "مجھے بخار ہے۔",           "speech": "I have a fever.",            "category": "medical"},

    # ── Basic Needs ────────────────────────────────────────────────────
    "مجھے پانی چاہیے":    {"display": "مجھے پانی چاہیے!",        "speech": "I need water!",              "category": "needs"},
    "مجھے بھوک لگی ہے":   {"display": "مجھے بھوک لگی ہے!",       "speech": "I am hungry!",               "category": "needs"},
    "pyas":     {"display": "مجھے پیاس لگی ہے۔",       "speech": "I am thirsty.",              "category": "needs"},
    "khana":    {"display": "مجھے کھانا چاہیے۔",       "speech": "I need food.",               "category": "needs"},
    "chai":     {"display": "مجھے چائے چاہیے۔",        "speech": "I would like tea.",          "category": "needs"},
    "bathroom": {"display": "مجھے باتھ روم جانا ہے۔",  "speech": "I need to use the bathroom.","category": "needs"},
    "neend":    {"display": "مجھے نیند آ رہی ہے۔",     "speech": "I am feeling sleepy.",       "category": "needs"},
    "kapray":   {"display": "مجھے کپڑے چاہیے۔",        "speech": "I need clothes.",            "category": "needs"},
    "garam":    {"display": "مجھے گرمی لگ رہی ہے۔",    "speech": "I am feeling hot.",          "category": "needs"},
    "sardi":    {"display": "مجھے سردی لگ رہی ہے۔",    "speech": "I am feeling cold.",         "category": "needs"},

    # ── Location / Movement ────────────────────────────────────────────
    "میں گھر جانا چاہتا ہوں": {"display": "میں گھر جانا چاہتا ہوں۔", "speech": "I want to go home.",         "category": "location"},
    "bahar":    {"display": "مجھے باہر جانا ہے۔",      "speech": "I want to go outside.",      "category": "location"},
    "andar":    {"display": "مجھے اندر جانا ہے۔",      "speech": "I want to go inside.",       "category": "location"},
    "wapis":    {"display": "واپس چلیں۔",               "speech": "Let's go back.",             "category": "location"},
    "براہ کرم رکیں":     {"display": "براہ کرم رکیں۔",          "speech": "Please wait.",               "category": "location"},
    "chalo":    {"display": "چلو!",                     "speech": "Let's go!",                  "category": "location"},
    "bistara":  {"display": "مجھے بستر پر لٹائیں۔",    "speech": "Please put me to bed.",      "category": "location"},
    "kursi":    {"display": "مجھے کرسی پر بٹھائیں۔",   "speech": "Please seat me in a chair.", "category": "location"},

    # ── Emotional / Social ─────────────────────────────────────────────
    "آپ کا بہت شکریہ": {"display": "آپ کا بہت شکریہ!",        "speech": "Thank you very much!",       "category": "social"},
    "سب ٹھیک ہے":    {"display": "سب ٹھیک ہے!",             "speech": "Everything is fine!",        "category": "social"},
    "salaam":   {"display": "السلام علیکم!",            "speech": "Peace be upon you!",         "category": "social"},
    "maafi":    {"display": "مجھے معاف کریں۔",         "speech": "Please forgive me.",         "category": "social"},
    "khush":    {"display": "میں خوش ہوں!",             "speech": "I am happy!",                "category": "social"},
    "gham":     {"display": "میں اداس ہوں۔",           "speech": "I am sad.",                  "category": "social"},
    "dar":      {"display": "مجھے ڈر لگ رہا ہے۔",      "speech": "I am scared.",               "category": "social"},
    "ghussa":   {"display": "مجھے غصہ آ رہا ہے۔",      "speech": "I am angry.",                "category": "social"},
    "haan":     {"display": "ہاں!",                     "speech": "Yes!",                       "category": "social"},
    "nahi":     {"display": "نہیں!",                    "speech": "No!",                        "category": "social"},

    # ── Communication ──────────────────────────────────────────────────
    "baat":     {"display": "مجھ سے بات کریں۔",         "speech": "Talk to me.",                "category": "communication"},
    "phone":    {"display": "فون ملائیں۔",              "speech": "Make a phone call.",         "category": "communication"},
    "family":   {"display": "میرے گھر والوں کو بلائیں۔","speech": "Call my family.",            "category": "communication"},
    "sunao":    {"display": "مجھے سنائیں۔",             "speech": "Let me listen.",             "category": "communication"},
    "dikhao":   {"display": "مجھے دکھائیں۔",            "speech": "Show me.",                   "category": "communication"},
    "samjhao":  {"display": "مجھے سمجھائیں۔",           "speech": "Explain to me.",             "category": "communication"},
    "dobara":   {"display": "دوبارہ کہیں۔",             "speech": "Say that again.",            "category": "communication"},

    # ── Time / Urgency ─────────────────────────────────────────────────
    "jaldi":    {"display": "جلدی کریں!",               "speech": "Hurry up!",                  "category": "time"},
    "intezaar": {"display": "کتنی دیر لگے گی؟",         "speech": "How long will it take?",     "category": "time"},
    "abhi":     {"display": "ابھی!",                    "speech": "Right now!",                 "category": "time"},
    "baad":     {"display": "بعد میں۔",                 "speech": "Later.",                     "category": "time"},
    "subah":    {"display": "صبح ہو گئی ہے۔",           "speech": "It is morning.",             "category": "time"},
    "raat":     {"display": "رات ہو گئی ہے۔",           "speech": "It is night.",               "category": "time"},
}


# ── Phrase Reconstruction (Offline Fallback) ────────────────────────────────

def reconstruct_phrase(tokens: List[str]) -> tuple[str, str]:
    """
    Maps a list of Urdu vocabulary tokens to (display_text, speech_text).

    This is the offline fallback used when ModelScope is unavailable.
    Concatenates multiple tokens with natural separators.
    """
    if not tokens:
        return ("", "")

    displays: list[str] = []
    speeches: list[str] = []

    for token in tokens:
        key = token.lower().strip()
        entry = URDU_VOCABULARY_DB.get(key)
        if entry:
            displays.append(entry["display"])
            speeches.append(entry["speech"])
        else:
            # Unknown token – pass through as-is
            displays.append(token)
            speeches.append(token)

    return (" ".join(displays), " ".join(speeches))


def lookup_token(token: str) -> Optional[dict]:
    """Look up a single token in the vocabulary. Returns None if not found."""
    return URDU_VOCABULARY_DB.get(token.lower().strip())


# ── Alibaba Cloud Client ─────────────────────────────────────────────────────

class AlibabaCloudClient:
    """
    Manages all Alibaba Cloud interactions:
    • OSS object storage for session data and audio uploads
    • DashScope API proxy for advanced NLP/TTS
    """

    def __init__(self, cfg: Optional[MindGridConfig] = None, max_workers: int = 4) -> None:
        self._cfg = cfg or config
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._session_id = _generate_session_id()
        self._oss_bucket = None

        # Attempt to initialise OSS
        if self._cfg.OSS_SYNC_ENABLED and self._cfg.OSS_ACCESS_KEY_ID:
            self._init_oss()

    # ── OSS ────────────────────────────────────────────────────────────

    def _init_oss(self) -> None:
        try:
            import oss2
            auth = oss2.Auth(self._cfg.OSS_ACCESS_KEY_ID, self._cfg.OSS_ACCESS_KEY_SECRET)
            self._oss_bucket = oss2.Bucket(auth, self._cfg.OSS_ENDPOINT, self._cfg.OSS_BUCKET)
            print(f"[Cloud] OSS bucket connected: {self._cfg.OSS_BUCKET}")
        except ImportError:
            print("[Cloud] oss2 not installed – OSS uploads disabled")
        except Exception as exc:
            print(f"[Cloud] OSS init failed: {exc}")

    def upload_gaze_session(self, data: List[dict]) -> Optional[str]:
        """Upload a gaze session log to OSS. Returns the object key on success."""
        if self._oss_bucket is None:
            logger.debug("[Cloud] OSS disabled – gaze session not uploaded")
            return None

        key = f"{self._cfg.OSS_PREFIX}gaze/{self._session_id}.json"
        payload = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")

        def _do_upload():
            try:
                self._oss_bucket.put_object(key, payload)
                print(f"[Cloud] Uploaded gaze session: {key}")
                return key
            except Exception as exc:
                logger.error(f"[Cloud] OSS upload failed: {exc}")
                return None

        future = self._executor.submit(_do_upload)
        return future.result(timeout=10)

    def upload_audio_chunk(self, audio: np.ndarray, timestamp: float) -> Optional[str]:
        """Upload a raw audio segment to OSS for archival / offline ASR."""
        if self._oss_bucket is None:
            return None

        ts_str = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        key = f"{self._cfg.OSS_PREFIX}audio/{self._session_id}_{ts_str}.raw"

        def _do_upload():
            try:
                raw_bytes = audio.astype(np.float32).tobytes()
                self._oss_bucket.put_object(key, raw_bytes)
                logger.debug(f"[Cloud] Uploaded audio chunk: {key}")
                return key
            except Exception as exc:
                logger.error(f"[Cloud] Audio upload failed: {exc}")
                return None

        future = self._executor.submit(_do_upload)
        return future.result(timeout=10)

    def upload_metrics(self, metrics: dict) -> None:
        """Upload aggregated session metrics."""
        if self._oss_bucket is None:
            return

        key = f"{self._cfg.OSS_PREFIX}metrics/{self._session_id}_metrics.json"
        payload = json.dumps(metrics, ensure_ascii=False, default=str).encode("utf-8")

        def _do_upload():
            try:
                self._oss_bucket.put_object(key, payload)
                print(f"[Cloud] Uploaded metrics: {key}")
            except Exception as exc:
                logger.error(f"[Cloud] Metrics upload failed: {exc}")

        self._executor.submit(_do_upload)

    # ── Shutdown ───────────────────────────────────────────────────────

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)


# ── Utilities ────────────────────────────────────────────────────────────────

def _generate_session_id() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
