"""
core/motion/motion_analyzer.py
───────────────────────────────
Provides three complementary motion signals:

1. Frame differencing  – fast change detection, triggers MOTION alerts
2. Lucas-Kanade optical flow – per-region velocity field
3. Accumulating motion heatmap – shows where activity concentrates over time
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from config.settings import settings


class MotionAnalyzer:
    """Real-time motion analysis for a fixed-resolution video stream."""

    # Lucas-Kanade parameters
    _LK_PARAMS = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    _FEATURE_PARAMS = dict(
        maxCorners=200,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7,
    )

    def __init__(self, width: int, height: int) -> None:
        self.width  = width
        self.height = height
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_pts:  Optional[np.ndarray] = None
        self._heatmap    = np.zeros((height, width), dtype=np.float32)
        self._frame_cnt  = 0
        self._bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=50, detectShadows=False
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Analyse one BGR frame.

        Returns
        -------
        motion_level : float   – 0.0 (no motion) to 1.0 (full-frame motion)
        flow_vis     : ndarray – HSV optical-flow visualisation (BGR)
        """
        self._frame_cnt += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        motion_level = self._background_subtraction(gray)
        flow_vis     = self._optical_flow(gray, frame)
        self._update_heatmap(gray)

        self._prev_gray = gray
        return motion_level, flow_vis

    def get_heatmap_overlay(self, frame: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        """Overlay the accumulated motion heatmap on a copy of frame."""
        hm_norm = cv2.normalize(self._heatmap, None, 0, 255, cv2.NORM_MINMAX)
        hm_u8   = hm_norm.astype(np.uint8)
        hm_col  = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 1 - alpha, hm_col, alpha, 0)

    def detect_sudden_motion(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Quick frame-difference check for sudden large motion.
        Returns (detected, fraction_of_frame_changed).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return False, 0.0

        diff  = cv2.absdiff(self._prev_gray, gray)
        _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        changed_frac = np.count_nonzero(th) / th.size
        return changed_frac > 0.12, changed_frac

    # ── Internal helpers ───────────────────────────────────────────────────

    def _background_subtraction(self, gray: np.ndarray) -> float:
        """Return fraction of frame with foreground motion."""
        fg_mask = self._bg_sub.apply(gray)
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        motion_area = sum(
            cv2.contourArea(c)
            for c in contours
            if cv2.contourArea(c) > settings.motion_sensitivity
        )
        return min(motion_area / (self.width * self.height), 1.0)

    def _optical_flow(self, gray: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """
        Compute sparse Lucas-Kanade optical flow and return HSV visualisation.
        """
        vis = frame.copy()
        if self._prev_gray is None:
            return vis

        # Refresh feature points every 10 frames
        if self._frame_cnt % 10 == 0 or self._prev_pts is None:
            self._prev_pts = cv2.goodFeaturesToTrack(
                self._prev_gray, mask=None, **self._FEATURE_PARAMS
            )

        if self._prev_pts is None:
            return vis

        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_pts, None, **self._LK_PARAMS
        )

        if new_pts is None:
            return vis

        good_new  = new_pts[status == 1]
        good_prev = self._prev_pts[status == 1]

        for new, old in zip(good_new, good_prev):
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)
            dx, dy = a - c, b - d
            magnitude = np.sqrt(dx**2 + dy**2)
            if magnitude > 2:          # ignore micro-jitter
                cv2.arrowedLine(vis, (c, d), (a, b), (0, 255, 100), 1, tipLength=0.4)

        self._prev_pts = good_new.reshape(-1, 1, 2)
        return vis

    def _update_heatmap(self, gray: np.ndarray) -> None:
        """Accumulate motion energy into the heatmap with temporal decay."""
        fg = self._bg_sub.apply(gray, learningRate=0)
        self._heatmap *= 0.97                       # exponential decay
        self._heatmap += fg.astype(np.float32) / 255.0
