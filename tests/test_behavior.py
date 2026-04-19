"""
tests/test_behavior.py
───────────────────────
Unit tests for the behavior analysis engine.
Uses mocked tracker states — no camera or GPU required.
"""
from __future__ import annotations

import time
from collections import deque
from unittest.mock import MagicMock, patch

import pytest

from backend.models.schemas import BoundingBox, DetectedObject, EventType
from core.behavior.behavior_analyzer import BehaviorAnalyzer, _iou, _point_in_polygon


# ── Helpers ────────────────────────────────────────────────────────────────

def make_person(track_id: int, x1: int, y1: int, x2: int, y2: int) -> DetectedObject:
    return DetectedObject(
        track_id=track_id,
        class_name="person",
        confidence=0.9,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


def make_state(speed: float = 0.0, speed_history=None, in_restricted=False):
    """Create a mock TrackState."""
    st = MagicMock()
    st.current_speed = speed
    st.in_restricted = in_restricted
    st.center_history = deque([(100, 100)], maxlen=60)
    return st


# ── IoU tests ──────────────────────────────────────────────────────────────

class TestIoU:
    def test_perfect_overlap(self):
        b = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        assert _iou(b, b) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = BoundingBox(x1=0, y1=0, x2=50, y2=50)
        b = BoundingBox(x1=100, y1=100, x2=150, y2=150)
        assert _iou(a, b) == 0.0

    def test_partial_overlap(self):
        a = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        b = BoundingBox(x1=50, y1=0, x2=150, y2=100)
        iou = _iou(a, b)
        assert 0.0 < iou < 1.0
        assert iou == pytest.approx(1/3, abs=0.01)


# ── Polygon containment tests ──────────────────────────────────────────────

class TestPolygon:
    _square = [(0, 0), (200, 0), (200, 200), (0, 200)]

    def test_inside(self):
        assert _point_in_polygon(100, 100, self._square) is True

    def test_outside(self):
        assert _point_in_polygon(300, 300, self._square) is False

    def test_edge_case(self):
        # Points on/near edge should not crash
        result = _point_in_polygon(0, 0, self._square)
        assert isinstance(result, bool)


# ── BehaviorAnalyzer tests ─────────────────────────────────────────────────

class TestBehaviorAnalyzer:

    def _make_analyzer(self, zones=None):
        tracker = MagicMock()
        tracker.frame_number = 100
        analyzer = BehaviorAnalyzer(tracker)
        if zones is not None:
            analyzer._zones = zones
        return analyzer, tracker

    def test_fight_detected(self):
        analyzer, tracker = self._make_analyzer()

        p1 = make_person(1, 100, 100, 200, 300)
        p2 = make_person(2, 110, 100, 210, 300)   # high overlap

        s1 = make_state(speed=40.0)
        s2 = make_state(speed=42.0)
        tracker.get_state.side_effect = lambda tid: {1: s1, 2: s2}.get(tid)

        alerts = analyzer._check_fight([(p1, s1), (p2, s2)])
        assert any(a.event_type == EventType.FIGHT for a in alerts), \
            "Should detect FIGHT when two persons overlap with high speed"

    def test_fight_not_detected_low_speed(self):
        analyzer, tracker = self._make_analyzer()

        p1 = make_person(1, 100, 100, 200, 300)
        p2 = make_person(2, 110, 100, 210, 300)
        s1 = make_state(speed=5.0)
        s2 = make_state(speed=5.0)

        alerts = analyzer._check_fight([(p1, s1), (p2, s2)])
        assert not any(a.event_type == EventType.FIGHT for a in alerts), \
            "Should NOT detect fight at low speed"

    def test_fall_detected(self):
        analyzer, tracker = self._make_analyzer()
        # Wide bbox = person lying down  (w > h * threshold)
        p = make_person(1, 50, 200, 350, 280)     # w=300, h=80 → ratio=3.75
        s = make_state(speed=0.5)

        alerts = analyzer._check_fall([(p, s)])
        assert any(a.event_type == EventType.FALL for a in alerts), \
            "Should detect FALL for wide bbox with near-zero speed"

    def test_fall_not_detected_moving(self):
        analyzer, tracker = self._make_analyzer()
        p = make_person(1, 50, 200, 350, 280)
        s = make_state(speed=20.0)     # still moving fast → not a fall

        alerts = analyzer._check_fall([(p, s)])
        assert not any(a.event_type == EventType.FALL for a in alerts)

    def test_loitering_triggers_after_threshold(self):
        analyzer, tracker = self._make_analyzer()
        p = make_person(1, 100, 100, 150, 300)
        s = make_state(speed=0.5)

        # Manually plant a stale timer
        analyzer._loiter_timers[1] = time.time() - 120   # 120s ago

        alerts = analyzer._check_loitering([(p, s)])
        assert any(a.event_type == EventType.LOITERING for a in alerts), \
            "Should fire LOITERING after threshold exceeded"

    def test_loitering_resets_when_moving(self):
        analyzer, tracker = self._make_analyzer()
        p = make_person(1, 100, 100, 150, 300)
        s = make_state(speed=25.0)
        analyzer._loiter_timers[1] = time.time() - 120

        analyzer._check_loitering([(p, s)])
        assert 1 not in analyzer._loiter_timers, \
            "Timer should clear when person starts moving"

    def test_intrusion_detected(self):
        zone = [(0, 0), (400, 0), (400, 400), (0, 400)]
        analyzer, tracker = self._make_analyzer(zones=[zone])

        p = make_person(1, 150, 150, 250, 350)   # center inside zone
        s = make_state(in_restricted=False)

        alerts = analyzer._check_intrusion([(p, s)])
        assert any(a.event_type == EventType.INTRUSION for a in alerts), \
            "Should detect INTRUSION when person enters restricted zone"

    def test_intrusion_not_refired(self):
        zone = [(0, 0), (400, 0), (400, 400), (0, 400)]
        analyzer, tracker = self._make_analyzer(zones=[zone])

        p = make_person(1, 150, 150, 250, 350)
        s = make_state(in_restricted=True)        # already inside

        alerts = analyzer._check_intrusion([(p, s)])
        assert not any(a.event_type == EventType.INTRUSION for a in alerts), \
            "Should NOT re-fire INTRUSION if already inside"

    def test_crowd_panic(self):
        analyzer, tracker = self._make_analyzer()

        persons_and_states = [
            (make_person(i, i*60, 100, i*60+50, 300), make_state(speed=35.0))
            for i in range(6)
        ]

        alerts = analyzer._check_crowd_panic(persons_and_states)
        assert any(a.event_type == EventType.CROWD_PANIC for a in alerts), \
            "Should detect CROWD_PANIC with 6 fast-moving persons"

    def test_crowd_panic_not_triggered_too_few(self):
        analyzer, tracker = self._make_analyzer()
        persons_and_states = [
            (make_person(i, i*60, 100, i*60+50, 300), make_state(speed=35.0))
            for i in range(2)
        ]
        alerts = analyzer._check_crowd_panic(persons_and_states)
        assert not any(a.event_type == EventType.CROWD_PANIC for a in alerts)


# ── Full analyze() integration ─────────────────────────────────────────────

class TestAnalyzePipeline:
    def test_analyze_returns_list(self):
        tracker = MagicMock()
        tracker.frame_number = 1
        tracker.get_state.return_value = make_state(speed=0.0)

        analyzer = BehaviorAnalyzer(tracker)
        alerts = analyzer.analyze([], [], 0.0, 1)
        assert isinstance(alerts, list)

    def test_analyze_no_crash_empty(self):
        tracker = MagicMock()
        tracker.frame_number = 1
        tracker.get_state.return_value = None

        analyzer = BehaviorAnalyzer(tracker)
        alerts = analyzer.analyze([], [], 0.1, 1)
        assert alerts == []
