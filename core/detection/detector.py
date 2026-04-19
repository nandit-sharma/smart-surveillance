"""
core/detection/detector.py
───────────────────────────
YOLOv8-based real-time object detector.

Outputs a list of DetectedObject per frame. Separates persons from
other objects so the tracker can focus on human targets.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from backend.models.schemas import BoundingBox, DetectedObject
from backend.utils.logger import log
from config.settings import settings


class ObjectDetector:
    """Wraps Ultralytics YOLOv8 for frame-level detection."""

    # COCO class IDs of interest for surveillance
    _PERSON_ID = 0
    _VEHICLE_IDS = {2, 3, 5, 7}       # car, motorcycle, bus, truck

    def __init__(self) -> None:
        from ultralytics import YOLO
        log.info(f"Loading YOLO model: {settings.yolo_model} on {settings.device}")
        self.model = YOLO(settings.yolo_model)
        self.model.to(settings.device)
        self.conf  = settings.detection_confidence
        self.iou   = settings.detection_iou
        log.info("YOLO model loaded OK")

    def detect(self, frame: np.ndarray) -> Tuple[List[DetectedObject], List[DetectedObject]]:
        """
        Run inference on a BGR frame.

        Returns
        -------
        persons  : List[DetectedObject]  – all detected people
        others   : List[DetectedObject]  – all other detected objects
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            device=settings.device,
            verbose=False,
        )

        persons: List[DetectedObject] = []
        others:  List[DetectedObject] = []

        if not results or results[0].boxes is None:
            return persons, others

        boxes = results[0].boxes
        names = self.model.names

        for box in boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            obj = DetectedObject(
                class_name=names.get(cls_id, str(cls_id)),
                confidence=round(conf, 3),
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
            )

            if cls_id == self._PERSON_ID:
                persons.append(obj)
            else:
                others.append(obj)

        return persons, others

    def get_raw_detections_for_tracker(
        self, persons: List[DetectedObject]
    ) -> List[List[float]]:
        """
        Convert DetectedObject list to [[x1,y1,x2,y2,conf], ...] format
        expected by DeepSORT / ByteTrack.
        """
        raw = []
        for p in persons:
            b = p.bbox
            raw.append([b.x1, b.y1, b.x2, b.y2, p.confidence])
        return raw
