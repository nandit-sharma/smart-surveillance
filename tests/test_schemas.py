"""
tests/test_schemas.py
──────────────────────
Tests for Pydantic schemas and alarm utility helpers.
"""
import pytest
from backend.models.schemas import (
    AlertEvent, BoundingBox, DetectedObject, EventType, Severity
)
from backend.utils.alarm import is_cooldown_active, mark_alert_fired
import time


class TestBoundingBox:
    def test_center(self):
        b = BoundingBox(x1=0, y1=0, x2=100, y2=200)
        assert b.center == (50, 100)

    def test_area(self):
        b = BoundingBox(x1=10, y1=10, x2=110, y2=60)
        assert b.area == 5000

    def test_zero_area(self):
        b = BoundingBox(x1=50, y1=50, x2=50, y2=50)
        assert b.area == 0


class TestDetectedObject:
    def test_valid_object(self):
        obj = DetectedObject(
            track_id=1,
            class_name="person",
            confidence=0.9,
            bbox=BoundingBox(x1=0, y1=0, x2=100, y2=200),
        )
        assert obj.track_id == 1
        assert obj.class_name == "person"

    def test_no_track_id(self):
        obj = DetectedObject(
            class_name="car",
            confidence=0.75,
            bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100),
        )
        assert obj.track_id is None


class TestAlertEvent:
    def test_auto_event_id(self):
        alert = AlertEvent(
            event_type=EventType.FIGHT,
            severity=Severity.CRITICAL,
            confidence=0.88,
            description="test",
        )
        assert alert.event_id is not None
        assert len(alert.event_id) == 12

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            AlertEvent(
                event_type=EventType.FALL,
                severity=Severity.HIGH,
                confidence=1.5,   # > 1.0 → invalid
                description="bad",
            )

    def test_serialization(self):
        alert = AlertEvent(
            event_type=EventType.INTRUSION,
            severity=Severity.MEDIUM,
            confidence=0.75,
            description="zone breach",
            involved_ids=[3, 7],
        )
        data = alert.model_dump(mode="json")
        assert data["event_type"] == "intrusion"
        assert data["involved_ids"] == [3, 7]

    def test_all_event_types(self):
        for et in EventType:
            a = AlertEvent(
                event_type=et,
                severity=Severity.LOW,
                confidence=0.5,
                description=f"test {et}",
            )
            assert a.event_type == et


class TestCooldown:
    def test_no_cooldown_initially(self):
        unique = f"test_event_{time.time()}"
        assert is_cooldown_active(unique) is False

    def test_cooldown_active_after_fire(self):
        unique = f"test_event_{time.time()}_2"
        mark_alert_fired(unique)
        assert is_cooldown_active(unique) is True
