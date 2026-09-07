"""
mindgrid/stream.py
───────────────────
Flask + Socket.IO HUD relay server for MindGrid smart glasses.

Endpoints
─────────
  GET  /health          – system health check
  GET  /hud             – latest HUD subtitle (JSON)
  POST /hud             – push new subtitle text
  GET  /gaze            – current gaze intent
  GET  /transcript      – latest ModelScope transcription
  POST /speak           – trigger Urdu TTS from remote client

WebSocket Events (via flask-socketio)
──────────────────────────────────────
  'subtitle_update'     – server pushes new Urdu subtitle to lens display
  'gaze_update'         – server pushes gaze direction change
  'transcript_update'   – server pushes live transcription
"""

from __future__ import annotations

import time
import json
import logging
from typing import Optional

from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit

logger = logging.getLogger(__name__)

# ── App Setup ────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── Shared State ─────────────────────────────────────────────────────────────

shared_hud_state = {
    "latest_text": "MindGrid HUD Ready",
    "timestamp": time.time(),
}

shared_gaze_state = {
    "direction": "center",
    "action": "idle",
    "confidence": 0.0,
    "timestamp": time.time(),
}

shared_transcript_state = {
    "text": "",
    "timestamp": time.time(),
}


# ── REST Endpoints ───────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "active",
        "system": "MindGrid Assistive HUD",
        "uptime": time.time(),
    })


@app.route("/hud", methods=["GET"])
def get_hud():
    """Smart glasses query this endpoint to render Urdu text on the lens."""
    return app.response_class(
        response=json.dumps(shared_hud_state, ensure_ascii=False),
        status=200,
        mimetype="application/json",
    )


@app.route("/hud", methods=["POST"])
def update_hud():
    """Push new subtitle text to the HUD (from pipeline or remote client)."""
    data = request.get_json(silent=True)
    if data and "text" in data:
        confidence = float(data.get("confidence", 1.0))
        shared_hud_state["latest_text"] = data["text"]
        shared_hud_state["timestamp"] = time.time()
        shared_hud_state["confidence"] = confidence

        # Broadcast to all WebSocket clients
        socketio.emit("subtitle_update", {
            "text": data["text"],
            "confidence": confidence,
            "timestamp": shared_hud_state["timestamp"],
        })
        return jsonify({"status": "updated", "text": data["text"]})
    return jsonify({"error": "Invalid payload – 'text' field required"}), 400


@app.route("/gaze", methods=["GET"])
def get_gaze():
    """Returns the latest gaze intent for debugging / monitoring."""
    return jsonify(shared_gaze_state)


@app.route("/gaze", methods=["POST"])
def update_gaze():
    """Update gaze state from the classifier pipeline."""
    data = request.get_json(silent=True)
    if data and "direction" in data:
        shared_gaze_state["direction"] = data["direction"]
        shared_gaze_state["action"] = data.get("action", "unknown")
        shared_gaze_state["confidence"] = float(data.get("confidence", 0.0))
        shared_gaze_state["timestamp"] = time.time()

        socketio.emit("gaze_update", shared_gaze_state)
        return jsonify({"status": "updated"})
    return jsonify({"error": "Invalid payload – 'direction' field required"}), 400


@app.route("/transcript", methods=["GET"])
def get_transcript():
    """Returns the latest ModelScope transcription."""
    return app.response_class(
        response=json.dumps(shared_transcript_state, ensure_ascii=False),
        status=200,
        mimetype="application/json",
    )


@app.route("/transcript", methods=["POST"])
def update_transcript():
    """Push new transcription from ModelScope ASR."""
    data = request.get_json(silent=True)
    if data and "text" in data:
        shared_transcript_state["text"] = data["text"]
        shared_transcript_state["timestamp"] = time.time()

        socketio.emit("transcript_update", {
            "text": data["text"],
            "timestamp": shared_transcript_state["timestamp"],
        })
        return jsonify({"status": "updated", "text": data["text"]})
    return jsonify({"error": "Invalid payload – 'text' field required"}), 400


@app.route("/speak", methods=["POST"])
def speak_remote():
    """
    Trigger Urdu TTS from a remote client.
    Expects JSON: {"text": "<Urdu text>"}
    """
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Invalid payload – 'text' field required"}), 400

    urdu_text = data["text"]

    # Import here to avoid circular imports at module level
    try:
        from urdu_tts import UrduVoice
        voice = UrduVoice()
        voice.speak(urdu_text)
        return jsonify({"status": "speaking", "text": urdu_text})
    except Exception as exc:
        return jsonify({"error": f"TTS failed: {exc}"}), 500


# ── WebSocket Event Handlers ────────────────────────────────────────────────

@socketio.on("connect")
def ws_connect():
    logger.info("[Stream] WebSocket client connected")
    emit("connected", {"status": "ok", "system": "MindGrid HUD"})


@socketio.on("disconnect")
def ws_disconnect():
    logger.info("[Stream] WebSocket client disconnected")


@socketio.on("request_state")
def ws_request_state():
    """Client requests full current state snapshot."""
    emit("full_state", {
        "hud": shared_hud_state,
        "gaze": shared_gaze_state,
        "transcript": shared_transcript_state,
    })


# ── Helper: push state from pipeline threads ────────────────────────────────

def push_subtitle(text: str, confidence: float = 1.0) -> None:
    """Thread-safe subtitle push (callable from any pipeline thread)."""
    shared_hud_state["latest_text"] = text
    shared_hud_state["timestamp"] = time.time()
    shared_hud_state["confidence"] = confidence
    socketio.emit("subtitle_update", {
        "text": text,
        "confidence": confidence,
        "timestamp": shared_hud_state["timestamp"],
    })


def push_gaze(direction: str, action: str, confidence: float) -> None:
    """Thread-safe gaze state push."""
    shared_gaze_state["direction"] = direction
    shared_gaze_state["action"] = action
    shared_gaze_state["confidence"] = confidence
    shared_gaze_state["timestamp"] = time.time()
    socketio.emit("gaze_update", dict(shared_gaze_state))


def push_transcript(text: str) -> None:
    """Thread-safe transcript push."""
    shared_transcript_state["text"] = text
    shared_transcript_state["timestamp"] = time.time()
    socketio.emit("transcript_update", {
        "text": text,
        "timestamp": shared_transcript_state["timestamp"],
    })


# ── Standalone Server ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n--- Starting MindGrid HUD Relay Server ---")
    socketio.run(app, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)
