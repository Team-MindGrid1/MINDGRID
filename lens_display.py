"""
mindgrid/lens_display.py
─────────────────────────
Transparent lens HUD renderer for the MindGrid smart glasses.

On Windows (development): renders a borderless, semi-transparent
pygame overlay window that simulates the transparent lens display.
Urdu text is rendered right-to-left using a Nastaliq-compatible font.

On real hardware: this module would drive an OLED micro-display
module (e.g., SSD1306 or HDMI-to-glass projector) via SPI/HDMI.
The public API remains identical.
"""

from __future__ import annotations

import os
import sys
import time
import threading
from typing import Optional

import numpy as np

try:
    import pygame
    _HAS_PYGAME = True
except ImportError:
    _HAS_PYGAME = False

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    _HAS_URDU_RENDER = True
except ImportError:
    _HAS_URDU_RENDER = False


# ── Data Models ──────────────────────────────────────────────────────────────

# Re-export so callers can use `from lens_display import GazeIntent`
from gaze_classifier import GazeIntent


# ── Direction Arrow Mapping ──────────────────────────────────────────────────

_DIRECTION_ARROWS = {
    "left":   "◀",
    "right":  "▶",
    "up":     "▲",
    "down":   "▼",
    "center": "●",
}

_ACTION_COLORS = {
    "fixation": (100, 200, 100),   # soft green
    "saccade":  (255, 200, 80),    # amber
    "blink":    (80, 180, 255),    # cyan
    "dwell":    (255, 120, 200),   # magenta
}


# ── Lens HUD ─────────────────────────────────────────────────────────────────

class LensHUD:
    """
    Transparent overlay HUD that simulates a smart-glasses lens display.

    Parameters
    ──────────
    width, height : int
        Window dimensions in pixels (default 640×200, landscape strip).
    opacity : int
        Window transparency 0-255 (default 200).
    headless : bool
        If True, all rendering is suppressed (for server / CI use).
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 200,
        opacity: int = 200,
        headless: bool = False,
    ) -> None:
        self._w = width
        self._h = height
        self._opacity = opacity
        self._headless = headless or not _HAS_PYGAME

        self._screen = None
        self._font_large = None
        self._font_small = None
        self._running = False
        self._lock = threading.Lock()

        # State
        self._subtitle: str = ""
        self._subtitle_confidence: float = 0.0
        self._gaze_arrow: str = "●"
        self._gaze_color: tuple = (100, 200, 100)
        self._transcript: str = ""
        self._status: str = "MindGrid Ready"

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Initialise pygame and create the overlay window."""
        if self._headless:
            print("[LensHUD] Headless mode – rendering suppressed")
            self._running = True
            return

        os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "100,100")
        pygame.init()
        pygame.display.set_caption("MindGrid Lens Display")

        flags = pygame.NOFRAME
        self._screen = pygame.display.set_mode((self._w, self._h), flags)

        # Try to load a Unicode-capable font
        self._font_large = self._load_font(28)
        self._font_small = self._load_font(18)

        self._running = True
        print("[LensHUD] Transparent lens display active")

    def stop(self) -> None:
        self._running = False
        if not self._headless and _HAS_PYGAME:
            pygame.quit()
        print("[LensHUD] Display stopped")

    # ── Render Commands ────────────────────────────────────────────────

    def render_subtitle(self, urdu_text: str, confidence: float = 1.0) -> None:
        """Display Urdu subtitle text at the bottom of the lens."""
        with self._lock:
            self._subtitle = urdu_text
            self._subtitle_confidence = confidence
            self._status = "Subtitle"

    def render_transcript(self, urdu_text: str) -> None:
        """Display ModelScope transcription at the top of the lens."""
        with self._lock:
            self._transcript = urdu_text
            self._status = "Live Transcript"

    def render_gaze_indicator(self, intent: GazeIntent) -> None:
        """Update the peripheral gaze direction indicator."""
        with self._lock:
            self._gaze_arrow = _DIRECTION_ARROWS.get(intent.direction, "●")
            self._gaze_color = _ACTION_COLORS.get(intent.action, (100, 200, 100))
            self._status = f"Gaze: {intent.action} {intent.direction}"

    def clear(self) -> None:
        with self._lock:
            self._subtitle = ""
            self._transcript = ""
            self._gaze_arrow = "●"
            self._status = "MindGrid Ready"

    # ── Main Render Loop ───────────────────────────────────────────────

    def run_loop(self, fps: int = 30) -> None:
        """
        Blocking render loop – call from a dedicated thread.
        Handles pygame events and redraws at the target frame rate.
        """
        if self._headless:
            while self._running:
                time.sleep(1 / fps)
            return

        clock = pygame.time.Clock()

        while self._running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                    break
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self._running = False
                    break

            self._draw_frame()
            clock.tick(fps)

    # ── Internal Drawing ───────────────────────────────────────────────

    def _draw_frame(self) -> None:
        if not self._screen:
            return

        with self._lock:
            subtitle = self._subtitle
            transcript = self._transcript
            arrow = self._gaze_arrow
            gaze_color = self._gaze_color
            status = self._status
            conf = self._subtitle_confidence

        # Semi-transparent dark background
        overlay = pygame.Surface((self._w, self._h), pygame.SRCALPHA)
        overlay.fill((10, 10, 20, 180))
        self._screen.blit(overlay, (0, 0))

        # ── Gaze indicator (top-right corner) ────────────────────────
        if self._font_large:
            gaze_surf = self._font_large.render(arrow, True, gaze_color)
            self._screen.blit(gaze_surf, (self._w - 50, 10))

        # ── Status bar (top-left) ────────────────────────────────────
        if self._font_small:
            status_surf = self._font_small.render(status, True, (150, 150, 150))
            self._screen.blit(status_surf, (10, 12))

        # ── Transcript (middle area) ─────────────────────────────────
        if transcript and self._font_small:
            # Truncate if too long and prepare for RTL rendering
            display_text = transcript[:60] + ("..." if len(transcript) > 60 else "")
            display_text = self._prepare_urdu(display_text)
            t_surf = self._font_small.render(display_text, True, (200, 220, 255))
            y = self._h // 2 - 15
            # Right-align for Urdu RTL
            x = self._w - t_surf.get_width() - 15
            self._screen.blit(t_surf, (max(10, x), y))

        # ── Subtitle bar (bottom) ────────────────────────────────────
        if subtitle and self._font_large:
            # Confidence-based color intensity
            alpha = int(180 + 75 * conf)
            bar = pygame.Surface((self._w, 50), pygame.SRCALPHA)
            bar.fill((20, 40, 80, alpha))
            self._screen.blit(bar, (0, self._h - 55))

            s_surf = self._font_large.render(self._prepare_urdu(subtitle), True, (255, 255, 255))
            # Right-align for Urdu RTL
            x = self._w - s_surf.get_width() - 15
            self._screen.blit(s_surf, (max(10, x), self._h - 45))

        pygame.display.flip()

    # ── Font Loading ───────────────────────────────────────────────────

    @staticmethod
    def _prepare_urdu(text: str) -> str:
        """Reshape Arabic/Urdu glyphs and apply bidirectional ordering."""
        if not _HAS_URDU_RENDER or not text:
            return text
        try:
            reshaped = arabic_reshaper.reshape(text)
            return get_display(reshaped)
        except Exception:
            return text

    @staticmethod
    def _load_font(size: int) -> Optional[object]:
        """Try to load a Unicode/Arabic-capable font, fall back to default."""
        # Common font paths on Windows
        candidates = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/tahoma.ttf",
            "C:/Windows/Fonts/NotoNastaliqUrdu-Regular.ttf",
        ]
        for path in candidates:
            if os.path.isfile(path):
                try:
                    return pygame.font.Font(path, size)
                except Exception:
                    continue
        # Fallback to pygame default
        try:
            return pygame.font.SysFont("arial", size)
        except Exception:
            return pygame.font.Font(None, size)


# ── Console Fallback (no pygame) ────────────────────────────────────────────

class ConsoleHUD:
    """Prints HUD state to stdout when pygame is unavailable."""

    def start(self) -> None:
        print("[ConsoleHUD] Text-only display active")

    def stop(self) -> None:
        pass

    def render_subtitle(self, urdu_text: str, confidence: float = 1.0) -> None:
        print(f"[HUD Subtitle] {urdu_text}  (conf: {confidence:.0%})")

    def render_transcript(self, urdu_text: str) -> None:
        print(f"[HUD Transcript] {urdu_text}")

    def render_gaze_indicator(self, intent: GazeIntent) -> None:
        arrow = _DIRECTION_ARROWS.get(intent.direction, "●")
        print(f"[HUD Gaze] {arrow} {intent.action} {intent.direction}")

    def clear(self) -> None:
        print("[HUD] Cleared")

    def run_loop(self, fps: int = 30) -> None:
        pass  # nothing to render in console mode


# ── Factory ──────────────────────────────────────────────────────────────────

def create_hud(headless: bool = False) -> LensHUD | ConsoleHUD:
    """Create the best available HUD implementation."""
    if _HAS_PYGAME and not headless:
        return LensHUD()
    return ConsoleHUD()


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n--- MindGrid Lens Display – Smoke Test ---")
    hud = create_hud()
    hud.start()

    # Simulate events
    hud.render_subtitle("مجھے مدد چاہیے!", confidence=0.92)
    time.sleep(2.0)

    intent = GazeIntent(direction="right", action="dwell", confidence=0.88, timestamp=time.time())
    hud.render_gaze_indicator(intent)
    time.sleep(1.0)

    hud.render_transcript("آپ کا بہت شکریہ")
    time.sleep(2.0)

    hud.clear()
    hud.stop()
    print("--- Test complete ---\n")
