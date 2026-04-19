"""
core/tracking/tracker.py
─────────────────────────
DeepSORT-based multi-object tracker.

Maintains persistent track IDs across frames for all detected persons.
Also stores per-track velocity history for behavior analysis.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from backend.models.schemas import BoundingBox, DetectedObject
from backend.utils.logger import log
from config.settings import settings


@dataclass
class TrackState:
    """Per-person tracking state accumulated across frames."""
    track_id:        int
    bbox:            BoundingBox
    center_history:  deque = field(default_factory=lambda: deque(maxlen=60))
    velocity_history: deque = field(default_factory=lambda: deque(maxlen=30))
    first_seen:      int   = 0         # frame number
    last_seen:       int   = 0
    zone_entry_time: Optional[float] = None   # for loitering
    in_restricted:   bool = False

    @property
    def current_speed(self) -> float:
        """Euclidean speed in px/frame (avg over last 5 frames)."""
        if len(self.velocity_history) < 2:
            return 0.0
        recent = list(self.velocity_history)[-5:]
        return float(np.mean(recent))

    @property
    def avg_position(self) -> Tuple[float, float]:
        if not self.center_history:
            return (0.0, 0.0)
        pts = list(self.center_history)
        return (np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts]))


class PersonTracker:
    """Wraps deep_sort_realtime for persistent person tracking."""

    def __init__(self) -> None:
        from deep_sort_realtime.deepsort_tracker import DeepSort
        self._ds = DeepSort(
            max_age=settings.max_track_age,
            n_init=3,
            nms_max_overlap=1.0,
            max_cosine_distance=settings.max_cosine_distance,
            nn_budget=settings.nn_budget,
        )
        self.states: Dict[int, TrackState] = {}
        self._frame_num = 0
        log.info("DeepSORT tracker initialized OK")

    def update(
        self,
        frame: np.ndarray,
        raw_detections: List[List[float]],   # [[x1,y1,x2,y2,conf], ...]
    ) -> List[DetectedObject]:
        """
        Feed detections into DeepSORT and return tracked persons with IDs.
        """
        self._frame_num += 1

        # DeepSORT expects [[left,top,w,h], conf, class] per detection
        ds_input = []
        for det in raw_detections:
            x1, y1, x2, y2, conf = det
            w, h = x2 - x1, y2 - y1
            ds_input.append(([x1, y1, w, h], conf, "person"))

        tracks = self._ds.update_tracks(ds_input, frame=frame)

        tracked_persons: List[DetectedObject] = []
        active_ids = set()

        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = int(track.track_id)
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
            center = bbox.center
            active_ids.add(tid)

            # Update or create track state
            if tid not in self.states:
                self.states[tid] = TrackState(
                    track_id=tid, bbox=bbox,
                    first_seen=self._frame_num, last_seen=self._frame_num
                )
            st = self.states[tid]
            st.bbox = bbox
            st.last_seen = self._frame_num

            # Update velocity
            if st.center_history:
                prev = st.center_history[-1]
                vel = np.linalg.norm(
                    np.array(center) - np.array(prev)
                )
                st.velocity_history.append(vel)
            st.center_history.append(center)

            tracked_persons.append(
                DetectedObject(
                    track_id=tid,
                    class_name="person",
                    confidence=1.0,
                    bbox=bbox,
                )
            )

        # Prune stale states
        stale = [tid for tid in self.states if tid not in active_ids
                 and self._frame_num - self.states[tid].last_seen > settings.max_track_age]
        for tid in stale:
            del self.states[tid]

        return tracked_persons

    def get_state(self, track_id: int) -> Optional[TrackState]:
        return self.states.get(track_id)

    @property
    def frame_number(self) -> int:
        return self._frame_num
