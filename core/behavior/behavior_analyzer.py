"""
core/behavior/behavior_analyzer.py
────────────────────────────────────
Rule-based behavior analysis that inspects tracked persons and motion
signals to detect suspicious events:

  • Fight / violence
  • Fall detection
  • Loitering
  • Restricted zone intrusion
  • Crowd panic / abnormal running
  • Robbery / theft (person-object proximity + speed burst)
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from backend.models.schemas import AlertEvent, BoundingBox, DetectedObject, EventType, Severity
from backend.utils.logger import log
from config.settings import settings
from core.tracking.tracker import PersonTracker, TrackState


# ── Utility helpers ────────────────────────────────────────────────────────

def _iou(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection-over-Union of two bounding boxes."""
    ix1 = max(a.x1, b.x1); iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2); iy2 = min(a.y2, b.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / (a.area + b.area - inter)


def _point_in_polygon(px: int, py: int, poly: List[Tuple[int, int]]) -> bool:
    """Ray-casting polygon containment test."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-9) + xi):
            inside = not inside
        j = i
    return inside


class BehaviorAnalyzer:
    """
    Stateful behavior analyzer.
    Call `analyze()` once per frame to get a list of AlertEvent objects.
    """

    def __init__(self, tracker: PersonTracker) -> None:
        self._tracker  = tracker
        self._zones    = settings.zones        # list of polygon vertex lists
        self._loiter_timers: Dict[int, float]  = {}   # track_id -> entry time
        self._speed_burst:   Dict[int, int]    = {}   # track_id -> high-speed frame count

    # ── Public interface ───────────────────────────────────────────────────

    def analyze(
        self,
        tracked_persons: List[DetectedObject],
        other_objects:   List[DetectedObject],
        motion_level:    float,
        frame_number:    int,
    ) -> List[AlertEvent]:
        """Run all behavior checks; return fired alerts this frame."""
        alerts: List[AlertEvent] = []
        states = [
            (p, self._tracker.get_state(p.track_id))
            for p in tracked_persons
            if p.track_id is not None
        ]
        states = [(p, s) for p, s in states if s is not None]

        # ── 1. Fight / violence detection ─────────────────────────────────
        alerts.extend(self._check_fight(states))

        # ── 2. Fall detection ──────────────────────────────────────────────
        alerts.extend(self._check_fall(states))

        # ── 3. Loitering ──────────────────────────────────────────────────
        alerts.extend(self._check_loitering(states))

        # ── 4. Restricted zone intrusion ──────────────────────────────────
        if settings.intrusion_check and self._zones:
            alerts.extend(self._check_intrusion(states))

        # ── 5. Crowd panic / abnormal running ─────────────────────────────
        alerts.extend(self._check_crowd_panic(states))

        # ── 6. Robbery / theft heuristic ──────────────────────────────────
        alerts.extend(self._check_robbery(states, other_objects))

        return alerts

    # ── Individual detectors ───────────────────────────────────────────────

    def _check_fight(
        self, states: List[Tuple[DetectedObject, TrackState]]
    ) -> List[AlertEvent]:
        """
        Fight heuristic:
          - Two or more persons within IoU > threshold AND
          - Both have speed > fight_speed_threshold
        """
        alerts = []
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                pi, si = states[i]
                pj, sj = states[j]
                overlap = _iou(pi.bbox, pj.bbox)
                if overlap < settings.fight_overlap_threshold:
                    continue
                speed_i = si.current_speed
                speed_j = sj.current_speed
                if (speed_i > settings.fight_speed_threshold and
                        speed_j > settings.fight_speed_threshold):
                    conf = min(0.99, overlap * 1.5 + (speed_i + speed_j) / 200)
                    alerts.append(AlertEvent(
                        event_type=EventType.FIGHT,
                        severity=Severity.CRITICAL,
                        confidence=round(conf, 3),
                        description=(
                            f"Possible fight between persons "
                            f"#{pi.track_id} and #{pj.track_id} "
                            f"(IoU={overlap:.2f}, speeds={speed_i:.1f},{speed_j:.1f})"
                        ),
                        involved_ids=[pi.track_id, pj.track_id],
                        frame_number=self._tracker.frame_number,
                    ))
        return alerts

    def _check_fall(
        self, states: List[Tuple[DetectedObject, TrackState]]
    ) -> List[AlertEvent]:
        """
        Fall heuristic:
          - Bounding box aspect ratio (w/h) exceeds threshold (person lying down)
          - Followed by near-zero velocity (stationary after fall)
        """
        alerts = []
        threshold = settings.fall_aspect_ratio_threshold
        for p, s in states:
            b = p.bbox
            w = b.x2 - b.x1
            h = b.y2 - b.y1
            if h == 0:
                continue
            ratio = w / h
            if ratio > threshold and s.current_speed < 3.0:
                conf = min(0.95, (ratio - threshold) / threshold + 0.5)
                alerts.append(AlertEvent(
                    event_type=EventType.FALL,
                    severity=Severity.HIGH,
                    confidence=round(conf, 3),
                    description=(
                        f"Possible fall – person #{p.track_id} "
                        f"horizontal (w/h={ratio:.2f})"
                    ),
                    involved_ids=[p.track_id],
                    frame_number=self._tracker.frame_number,
                ))
        return alerts

    def _check_loitering(
        self, states: List[Tuple[DetectedObject, TrackState]]
    ) -> List[AlertEvent]:
        """
        Loitering heuristic:
          - Person stays within a small bounding circle for > loiter_time_threshold seconds
        """
        alerts = []
        now = time.time()
        for p, s in states:
            tid = p.track_id
            if s.current_speed < 5.0:          # nearly stationary
                if tid not in self._loiter_timers:
                    self._loiter_timers[tid] = now
                else:
                    elapsed = now - self._loiter_timers[tid]
                    if elapsed > settings.loiter_time_threshold:
                        conf = min(0.9, 0.5 + elapsed / (settings.loiter_time_threshold * 2))
                        alerts.append(AlertEvent(
                            event_type=EventType.LOITERING,
                            severity=Severity.MEDIUM,
                            confidence=round(conf, 3),
                            description=(
                                f"Person #{tid} loitering for {int(elapsed)}s"
                            ),
                            involved_ids=[tid],
                            frame_number=self._tracker.frame_number,
                            metadata={"elapsed_seconds": int(elapsed)},
                        ))
            else:
                self._loiter_timers.pop(tid, None)
        return alerts

    def _check_intrusion(
        self, states: List[Tuple[DetectedObject, TrackState]]
    ) -> List[AlertEvent]:
        """Check if any person's center is inside a restricted polygon."""
        alerts = []
        for p, s in states:
            cx, cy = p.bbox.center
            for zone_idx, poly in enumerate(self._zones):
                if _point_in_polygon(cx, cy, poly):
                    if not s.in_restricted:
                        s.in_restricted = True
                        alerts.append(AlertEvent(
                            event_type=EventType.INTRUSION,
                            severity=Severity.HIGH,
                            confidence=0.95,
                            description=(
                                f"Person #{p.track_id} entered restricted zone {zone_idx}"
                            ),
                            involved_ids=[p.track_id],
                            frame_number=self._tracker.frame_number,
                            metadata={"zone_index": zone_idx},
                        ))
                else:
                    s.in_restricted = False
        return alerts

    def _check_crowd_panic(
        self, states: List[Tuple[DetectedObject, TrackState]]
    ) -> List[AlertEvent]:
        """
        Crowd panic heuristic:
          - >= crowd_min_persons all moving fast simultaneously
          - Average speed > crowd_panic_speed
        """
        alerts = []
        if len(states) < settings.crowd_min_persons:
            return alerts
        speeds = [s.current_speed for _, s in states]
        avg_speed = float(np.mean(speeds))
        fast_count = sum(1 for sp in speeds if sp > settings.crowd_panic_speed)

        if avg_speed > settings.crowd_panic_speed and fast_count >= settings.crowd_min_persons:
            conf = min(0.92, avg_speed / (settings.crowd_panic_speed * 2))
            alerts.append(AlertEvent(
                event_type=EventType.CROWD_PANIC,
                severity=Severity.CRITICAL,
                confidence=round(conf, 3),
                description=(
                    f"Crowd panic: {fast_count}/{len(states)} persons moving fast "
                    f"(avg speed={avg_speed:.1f})"
                ),
                involved_ids=[p.track_id for p, _ in states],
                frame_number=self._tracker.frame_number,
                metadata={"avg_speed": round(avg_speed, 2), "fast_count": fast_count},
            ))
        return alerts

    def _check_robbery(
        self,
        states: List[Tuple[DetectedObject, TrackState]],
        other_objects: List[DetectedObject],
    ) -> List[AlertEvent]:
        """
        Robbery heuristic:
          - A person is in close proximity to a high-value object (handbag, backpack, etc.)
          - Then both person AND object suddenly disappear or person has a speed burst
        """
        alerts = []
        target_classes = {"handbag", "backpack", "suitcase", "cell phone", "laptop"}
        valuables = [o for o in other_objects if o.class_name in target_classes]

        for p, s in states:
            for val in valuables:
                overlap = _iou(p.bbox, val.bbox)
                if overlap > 0.05 and s.current_speed > settings.fight_speed_threshold * 0.8:
                    tid = p.track_id
                    self._speed_burst[tid] = self._speed_burst.get(tid, 0) + 1
                    if self._speed_burst[tid] >= 5:   # sustained burst
                        conf = min(0.85, overlap * 5 + s.current_speed / 100)
                        alerts.append(AlertEvent(
                            event_type=EventType.ROBBERY,
                            severity=Severity.CRITICAL,
                            confidence=round(conf, 3),
                            description=(
                                f"Possible robbery – person #{tid} near "
                                f"'{val.class_name}' moving fast"
                            ),
                            involved_ids=[tid],
                            frame_number=self._tracker.frame_number,
                            metadata={"target_object": val.class_name},
                        ))
                        self._speed_burst[tid] = 0
                else:
                    self._speed_burst.pop(p.track_id, None)
        return alerts
