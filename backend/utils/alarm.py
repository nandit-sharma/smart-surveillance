"""
backend/utils/alarm.py
───────────────────────
Handles:
  - Audio beep alarm (via pygame or system beep fallback)
  - Snapshot frame saving
  - HTTP POST to backend alert endpoint
"""
from __future__ import annotations

import asyncio
import base64
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from backend.models.schemas import AlertEvent
from backend.utils.logger import log
from config.settings import settings

# Track last alert time per event type to enforce cooldown
_last_alert_time: dict[str, float] = {}

# ── pygame alarm (optional) ─────────────────────────────────────────────────
_pygame_ready = False
try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
    _pygame_ready = True
except Exception:
    pass


def _generate_beep_sound(freq: int = 880, duration_ms: int = 500) -> bytes:
    """Generate a raw PCM sine-wave beep."""
    import math, struct
    sample_rate = 44100
    n_samples = int(sample_rate * duration_ms / 1000)
    data = []
    for i in range(n_samples):
        val = int(32767 * math.sin(2 * math.pi * freq * i / sample_rate))
        data.append(struct.pack("<h", val))
    return b"".join(data)


def play_alarm(duration_ms: Optional[int] = None) -> None:
    """Play an audible alarm beep (non-blocking)."""
    if not settings.enable_alarm:
        return
    dur = duration_ms or settings.alarm_duration_ms

    if _pygame_ready:
        try:
            pcm = _generate_beep_sound(880, dur)
            sound = pygame.sndarray.make_sound(
                np.frombuffer(pcm, dtype=np.int16)
            )
            sound.play()
            return
        except Exception as e:
            log.debug(f"pygame alarm failed: {e}")

    # Fallback: ASCII bell
    print("\a", end="", flush=True)


def save_snapshot(frame: np.ndarray, event: AlertEvent) -> Optional[str]:
    """Save a JPEG snapshot of the alert frame, returns file path."""
    if not settings.save_snapshots:
        return None
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{event.event_type.value}_{ts}_{event.event_id}.jpg"
    filepath = settings.snapshot_path / filename
    try:
        cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        log.info(f"Snapshot saved: {filepath}")
        return str(filepath)
    except Exception as e:
        log.error(f"Snapshot save failed: {e}")
        return None


def is_cooldown_active(event_type: str) -> bool:
    """Returns True if the same event type fired too recently."""
    last = _last_alert_time.get(event_type, 0.0)
    return (time.time() - last) < settings.alert_cooldown_seconds


def mark_alert_fired(event_type: str) -> None:
    _last_alert_time[event_type] = time.time()


async def dispatch_alert(event: AlertEvent, frame: Optional[np.ndarray] = None) -> None:
    """
    Full alert pipeline:
      1. Cooldown check
      2. Play alarm
      3. Save snapshot
      4. POST to backend REST endpoint
    """
    if is_cooldown_active(event.event_type.value):
        return

    mark_alert_fired(event.event_type.value)
    log.warning(f"ALERT [{event.severity.upper()}] {event.event_type}: {event.description}")

    # Play alarm in thread-pool to avoid blocking
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, play_alarm)

    # Snapshot
    if frame is not None:
        snap_path = await loop.run_in_executor(None, save_snapshot, frame, event)
        if snap_path:
            event.snapshot_path = snap_path

    # HTTP POST to backend
    await _post_alert(event, frame)


async def _post_alert(event: AlertEvent, frame: Optional[np.ndarray]) -> None:
    """POST JSON alert to the FastAPI backend endpoint."""
    import aiohttp

    payload = event.model_dump(mode="json")

    # Optionally embed a base64 snapshot
    if frame is not None:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        payload["snapshot_b64"] = base64.b64encode(buf).decode()

    url = f"http://127.0.0.1:{settings.backend_port}/api/alerts"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                if resp.status != 200:
                    log.warning(f"Alert POST returned HTTP {resp.status}")
    except Exception as e:
        log.debug(f"Alert POST failed (backend may not be running): {e}")
