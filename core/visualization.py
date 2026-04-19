"""
core/visualization.py
──────────────────────
Draws bounding boxes, track IDs, alert overlays, motion arrows,
and restricted zone polygons on video frames.
"""
from __future__ import annotations

import time
from typing import Dict, List

import cv2
import numpy as np

from backend.models.schemas import AlertEvent, DetectedObject, EventType, Severity

# ── Color palette (BGR) ────────────────────────────────────────────────────
_COLORS: Dict[str, tuple] = {
    "person":         (0, 220, 120),
    "vehicle":        (255, 180, 0),
    "object":         (180, 180, 180),
    "alert_CRITICAL": (0, 30, 220),
    "alert_HIGH":     (0, 120, 255),
    "alert_MEDIUM":   (0, 200, 255),
    "alert_LOW":      (0, 240, 200),
    "zone":           (80, 0, 200),
    "text_bg":        (15, 15, 15),
    "hud_bg":         (20, 20, 20),
}

_SEVERITY_EMOJI = {
    Severity.CRITICAL: "⚠ CRITICAL",
    Severity.HIGH:     "! HIGH",
    Severity.MEDIUM:   "~ MEDIUM",
    Severity.LOW:      "· LOW",
}

_EVENT_ICON = {
    EventType.FIGHT:         "👊 FIGHT",
    EventType.ROBBERY:       "🔓 ROBBERY",
    EventType.FALL:          "⬇ FALL",
    EventType.INTRUSION:     "🚫 INTRUSION",
    EventType.CROWD_PANIC:   "🏃 PANIC",
    EventType.LOITERING:     "⏱ LOITER",
    EventType.VEHICLE_CRASH: "🚗 CRASH",
    EventType.MOTION:        "〰 MOTION",
    EventType.UNKNOWN:       "? UNKNOWN",
}


class FrameVisualizer:
    """Draws all surveillance overlays onto a frame in-place."""

    def __init__(self, width: int, height: int) -> None:
        self.width  = width
        self.height = height
        self._alert_display_until: float = 0.0
        self._current_alerts: List[AlertEvent] = []

    def draw(
        self,
        frame:           np.ndarray,
        tracked_persons: List[DetectedObject],
        other_objects:   List[DetectedObject],
        new_alerts:      List[AlertEvent],
        motion_level:    float,
        fps:             float,
        zones:           list,
        frame_number:    int,
    ) -> np.ndarray:
        out = frame.copy()

        # Persist alerts for a few seconds
        if new_alerts:
            self._current_alerts = new_alerts
            self._alert_display_until = time.time() + 4.0

        if time.time() > self._alert_display_until:
            self._current_alerts = []

        self._draw_zones(out, zones)
        self._draw_persons(out, tracked_persons)
        self._draw_others(out, other_objects)
        self._draw_hud(out, fps, motion_level, len(tracked_persons), frame_number)

        if self._current_alerts:
            self._draw_alert_banner(out, self._current_alerts)

        return out

    # ── Internal drawing helpers ───────────────────────────────────────────

    def _draw_persons(self, frame: np.ndarray, persons: List[DetectedObject]) -> None:
        for p in persons:
            b = p.bbox
            color = _COLORS["person"]
            cv2.rectangle(frame, (b.x1, b.y1), (b.x2, b.y2), color, 2)
            label = f"P#{p.track_id}"
            self._draw_label(frame, label, b.x1, b.y1 - 4, color)

    def _draw_others(self, frame: np.ndarray, objects: List[DetectedObject]) -> None:
        for o in objects:
            b = o.bbox
            color = _COLORS.get("vehicle" if "car" in o.class_name else "object",
                                 _COLORS["object"])
            cv2.rectangle(frame, (b.x1, b.y1), (b.x2, b.y2), color, 1)
            label = f"{o.class_name} {o.confidence:.0%}"
            self._draw_label(frame, label, b.x1, b.y1 - 4, color, scale=0.45)

    def _draw_zones(self, frame: np.ndarray, zones: list) -> None:
        for poly in zones:
            pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=_COLORS["zone"], thickness=2)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], _COLORS["zone"])
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            # Label
            cx = int(np.mean([p[0] for p in poly]))
            cy = int(np.mean([p[1] for p in poly]))
            cv2.putText(frame, "RESTRICTED", (cx - 50, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, _COLORS["zone"], 2)

    def _draw_hud(
        self,
        frame:        np.ndarray,
        fps:          float,
        motion_level: float,
        person_count: int,
        frame_number: int,
    ) -> None:
        """Draw semi-transparent HUD bar at the top of the frame."""
        bar_h = 36
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, bar_h), _COLORS["hud_bg"], -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        hud_items = [
            f"FPS: {fps:.1f}",
            f"Persons: {person_count}",
            f"Motion: {motion_level:.0%}",
            f"Frame: {frame_number}",
        ]
        x = 10
        for item in hud_items:
            cv2.putText(frame, item, (x, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
            x += len(item) * 9 + 20

        # Motion bar
        bar_w = int(motion_level * 150)
        cv2.rectangle(frame, (self.width - 165, 10), (self.width - 15, 28), (40, 40, 40), -1)
        bar_color = (0, 200, 100) if motion_level < 0.3 else (0, 180, 255) if motion_level < 0.6 else (0, 30, 220)
        cv2.rectangle(frame, (self.width - 165, 10), (self.width - 165 + bar_w, 28), bar_color, -1)

    def _draw_alert_banner(self, frame: np.ndarray, alerts: List[AlertEvent]) -> None:
        """Full-width alert banner at the bottom of the frame."""
        top_alert = max(alerts, key=lambda a: list(Severity).index(a.severity))
        color = _COLORS.get(f"alert_{top_alert.severity.upper()}", (0, 0, 200))
        bh = 60
        by = self.height - bh

        # Pulsing opacity based on time
        pulse = abs(np.sin(time.time() * 4)) * 0.4 + 0.55
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, by), (self.width, self.height), color, -1)
        cv2.addWeighted(overlay, pulse, frame, 1 - pulse, 0, frame)

        # Event label
        icon = _EVENT_ICON.get(top_alert.event_type, "⚠")
        cv2.putText(frame, icon, (20, by + 22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, top_alert.description[:100], (20, by + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Confidence badge
        conf_txt = f"{top_alert.confidence:.0%}"
        cv2.putText(frame, conf_txt, (self.width - 90, by + 38),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    @staticmethod
    def _draw_label(
        frame: np.ndarray,
        text: str,
        x: int, y: int,
        color: tuple,
        scale: float = 0.55,
    ) -> None:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        y = max(y, th + 4)
        cv2.rectangle(frame, (x, y - th - 4), (x + tw + 4, y + 2), _COLORS["text_bg"], -1)
        cv2.putText(frame, text, (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
