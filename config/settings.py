"""
config/settings.py
──────────────────
Centralised configuration using Pydantic-Settings.
All values load from .env (or environment variables).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Camera ────────────────────────────────────────────────
    camera_source: str = "0"
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30

    # ── Detection ─────────────────────────────────────────────
    yolo_model: str = "yolov8n.pt"
    detection_confidence: float = 0.45
    detection_iou: float = 0.45
    device: str = "cpu"

    # ── Tracking ──────────────────────────────────────────────
    max_track_age: int = 30
    max_iou_distance: float = 0.7
    max_cosine_distance: float = 0.3
    nn_budget: int = 100

    # ── Behavior ──────────────────────────────────────────────
    fight_speed_threshold: float = 35.0
    fight_overlap_threshold: float = 0.25
    loiter_time_threshold: int = 60
    intrusion_check: bool = True
    fall_aspect_ratio_threshold: float = 1.5
    crowd_panic_speed: float = 28.0
    crowd_min_persons: int = 5

    # ── Motion ────────────────────────────────────────────────
    motion_sensitivity: int = 500
    optical_flow_quality: float = 0.3

    # ── Alert ─────────────────────────────────────────────────
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    alert_cooldown_seconds: int = 5
    save_snapshots: bool = True
    snapshot_dir: str = "./snapshots"
    enable_alarm: bool = True
    alarm_duration_ms: int = 1500

    # ── Zones (stored as JSON string, decoded below) ───────────
    restricted_zones: str = "[]"

    # ── Logging ───────────────────────────────────────────────
    log_level: str = "INFO"
    log_file: str = "./logs/surveillance.log"

    # ── Derived ───────────────────────────────────────────────
    @property
    def camera_index(self) -> int | str:
        """Return int for webcam index, string for RTSP/file paths."""
        try:
            return int(self.camera_source)
        except ValueError:
            return self.camera_source

    @property
    def zones(self) -> List[List[Tuple[int, int]]]:
        """Parse restricted zones from JSON string."""
        raw = json.loads(self.restricted_zones)
        return [list(map(tuple, zone)) for zone in raw]

    @property
    def snapshot_path(self) -> Path:
        p = Path(self.snapshot_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


# Singleton instance
settings = Settings()
