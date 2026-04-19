"""
backend/models/schemas.py
─────────────────────────
Pydantic schemas for alert events, API responses, and WebSocket messages.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EventType(str, Enum):
    FIGHT         = "fight"
    ROBBERY       = "robbery"
    FALL          = "fall"
    INTRUSION     = "intrusion"
    CROWD_PANIC   = "crowd_panic"
    LOITERING     = "loitering"
    VEHICLE_CRASH = "vehicle_crash"
    MOTION        = "motion"
    UNKNOWN       = "unknown"


class Severity(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center(self):
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def area(self):
        return (self.x2 - self.x1) * (self.y2 - self.y1)


class DetectedObject(BaseModel):
    track_id:   Optional[int]  = None
    class_name: str
    confidence: float
    bbox:       BoundingBox


class AlertEvent(BaseModel):
    """Primary schema for suspicious activity alerts."""
    event_id:       str         = Field(default_factory=lambda: __import__('uuid').uuid4().hex[:12])
    event_type:     EventType
    severity:       Severity
    timestamp:      datetime    = Field(default_factory=datetime.utcnow)
    confidence:     float       = Field(ge=0.0, le=1.0)
    description:    str
    involved_ids:   List[int]   = Field(default_factory=list)   # track IDs
    frame_number:   int         = 0
    snapshot_path:  Optional[str] = None
    metadata:       Dict[str, Any] = Field(default_factory=dict)


class AlertResponse(BaseModel):
    status:   str = "received"
    event_id: str
    message:  str = ""


class FrameAnalysis(BaseModel):
    """Per-frame summary sent over WebSocket to dashboard."""
    frame_number:    int
    timestamp:       datetime = Field(default_factory=datetime.utcnow)
    fps:             float
    person_count:    int
    object_counts:   Dict[str, int]
    active_alerts:   List[AlertEvent]
    motion_level:    float        # 0.0–1.0 normalised
    tracked_persons: List[DetectedObject]


class SystemStatus(BaseModel):
    running:        bool
    camera_source:  str
    model:          str
    device:         str
    uptime_seconds: float
    total_alerts:   int
    fps:            float
