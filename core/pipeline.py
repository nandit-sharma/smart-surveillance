"""
core/pipeline.py
─────────────────
Main surveillance pipeline.

Orchestrates: Camera → Detection → Tracking → Motion → Behavior → Alerts

Runs in a background thread; exposes:
  - latest_frame   (np.ndarray) – annotated frame ready for streaming
  - latest_stats   (dict)       – per-frame statistics for WebSocket
  - alert_queue    (asyncio.Queue) – fired alert events
"""
from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from backend.models.schemas import AlertEvent, FrameAnalysis
from backend.utils.alarm import dispatch_alert
from backend.utils.logger import log
from config.settings import settings
from core.behavior.behavior_analyzer import BehaviorAnalyzer
from core.detection.detector import ObjectDetector
from core.motion.motion_analyzer import MotionAnalyzer
from core.tracking.tracker import PersonTracker
from core.visualization import FrameVisualizer


class SurveillancePipeline:
    """Thread-safe, real-time surveillance pipeline."""

    def __init__(self) -> None:
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._start_time = 0.0

        # Shared state (thread-safe via lock for frame, queue for alerts)
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_stats: dict = {}
        self.alert_queue: asyncio.Queue = asyncio.Queue(maxsize=100)

        # FPS tracking
        self._fps_buffer: deque = deque(maxlen=30)
        self._last_frame_time = time.time()

        # Components (lazy-init in thread)
        self._detector: Optional[ObjectDetector] = None
        self._tracker:  Optional[PersonTracker]  = None
        self._motion:   Optional[MotionAnalyzer] = None
        self._behavior: Optional[BehaviorAnalyzer] = None
        self._viz:      Optional[FrameVisualizer]  = None

        # Alert totals
        self.total_alerts = 0

    # ── Public API ─────────────────────────────────────────────────────────

    def start(self) -> None:
        if self.running:
            log.warning("Pipeline already running")
            return
        self.running = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="SurveillanceLoop")
        self._thread.start()
        log.info("Surveillance pipeline started OK")

    def stop(self) -> None:
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        log.info("Surveillance pipeline stopped")

    @property
    def latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._latest_frame

    @property
    def latest_stats(self) -> dict:
        with self._lock:
            return self._latest_stats.copy()

    @property
    def uptime(self) -> float:
        return time.time() - self._start_time if self._start_time else 0.0

    # ── Internal loop ──────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        # Initialise heavy components inside the worker thread
        try:
            self._detector = ObjectDetector()
            self._tracker  = PersonTracker()
            cap = self._open_camera()
            w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._motion   = MotionAnalyzer(w, h)
            self._behavior = BehaviorAnalyzer(self._tracker)
            self._viz      = FrameVisualizer(w, h)
        except Exception as e:
            log.error(f"Pipeline init failed: {e}")
            self.running = False
            return

        frame_num = 0
        try:
            while self.running:
                ok, raw_frame = cap.read()
                if not ok:
                    log.warning("Camera read failed – retrying in 1s")
                    time.sleep(1)
                    continue

                frame_num += 1
                fps = self._calc_fps()

                # ── Detection ─────────────────────────────────────────────
                persons_det, others_det = self._detector.detect(raw_frame)
                raw_dets = self._detector.get_raw_detections_for_tracker(persons_det)

                # ── Tracking ──────────────────────────────────────────────
                tracked = self._tracker.update(raw_frame, raw_dets)

                # ── Motion analysis ───────────────────────────────────────
                motion_level, _ = self._motion.process(raw_frame)

                # ── Behavior analysis ─────────────────────────────────────
                alerts = self._behavior.analyze(
                    tracked, others_det, motion_level, frame_num
                )

                # ── Fire alerts asynchronously ────────────────────────────
                if alerts:
                    self.total_alerts += len(alerts)
                    for alert in alerts:
                        self._fire_alert(alert, raw_frame)

                # ── Visualise ─────────────────────────────────────────────
                annotated = self._viz.draw(
                    raw_frame, tracked, others_det, alerts,
                    motion_level, fps, settings.zones, frame_num
                )

                # ── Display ───────────────────────────────────────────────
                cv2.imshow("Smart Surveillance", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    log.info("Quit key 'q' pressed. Stopping pipeline.")
                    self.running = False

                # ── Update shared state ───────────────────────────────────
                obj_counts: dict = {}
                for o in others_det:
                    obj_counts[o.class_name] = obj_counts.get(o.class_name, 0) + 1

                stats = {
                    "frame_number":    frame_num,
                    "fps":             fps,
                    "person_count":    len(tracked),
                    "object_counts":   obj_counts,
                    "active_alerts":   [a.model_dump(mode="json") for a in alerts],
                    "motion_level":    motion_level,
                }
                with self._lock:
                    self._latest_frame = annotated
                    self._latest_stats = stats

        finally:
            cap.release()
            cv2.destroyAllWindows()
            log.info("Camera released and windows closed")

    def _open_camera(self) -> cv2.VideoCapture:
        src = settings.camera_index
        log.info(f"Opening camera: {src}")
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {src}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  settings.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.camera_height)
        cap.set(cv2.CAP_PROP_FPS,          settings.camera_fps)
        return cap

    def _calc_fps(self) -> float:
        now = time.time()
        dt  = now - self._last_frame_time
        self._last_frame_time = now
        self._fps_buffer.append(1.0 / dt if dt > 0 else 0.0)
        return round(float(np.mean(self._fps_buffer)), 1)

    def _fire_alert(self, alert: AlertEvent, frame: np.ndarray) -> None:
        """Put alert on async queue (non-blocking, drop if full)."""
        try:
            self.alert_queue.put_nowait(alert)
        except asyncio.QueueFull:
            pass

        # Run async dispatch in a separate thread to avoid blocking the loop
        def _dispatch():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(dispatch_alert(alert, frame.copy()))
            loop.close()

        threading.Thread(target=_dispatch, daemon=True).start()


# Singleton
pipeline = SurveillancePipeline()
