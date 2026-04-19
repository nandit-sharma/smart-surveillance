"""
backend/api/server.py
──────────────────────
FastAPI application exposing:

  GET  /               → Live dashboard HTML
  GET  /video_feed     → MJPEG live stream
  GET  /ws             → WebSocket: real-time stats JSON
  POST /api/alerts     → Receive & store alert events
  GET  /api/alerts     → List stored alert events (paginated)
  GET  /api/status     → System health & stats
  POST /api/control    → start / stop pipeline
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from backend.models.schemas import AlertEvent, AlertResponse, SystemStatus
from backend.utils.logger import log
from config.settings import settings
from core.pipeline import pipeline

app = FastAPI(title="Smart Surveillance System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files & templates
import os, pathlib
BASE = pathlib.Path(__file__).parent.parent
templates = Jinja2Templates(directory=str(BASE / "templates"))
try:
    app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")
except Exception:
    pass

# In-memory alert store (last 500 events)
_alert_store: deque[Dict[str, Any]] = deque(maxlen=500)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.connections: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)
        log.info(f"WebSocket connected ({len(self.connections)} total)")

    def disconnect(self, ws: WebSocket):
        if ws in self.connections:
            self.connections.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.connections:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.connections.remove(ws)


manager = ConnectionManager()


# ── Startup / Shutdown ─────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    log.info("FastAPI startup – launching surveillance pipeline")
    pipeline.start()
    asyncio.create_task(_stats_broadcaster())
    asyncio.create_task(_alert_queue_consumer())


@app.on_event("shutdown")
async def shutdown():
    pipeline.stop()


# ── Background tasks ───────────────────────────────────────────────────────

async def _stats_broadcaster():
    """Push per-frame stats to all WebSocket clients at ~10 Hz."""
    while True:
        await asyncio.sleep(0.1)
        stats = pipeline.latest_stats
        if stats:
            await manager.broadcast(stats)


async def _alert_queue_consumer():
    """Drain the pipeline's alert queue and store events."""
    while True:
        try:
            alert: AlertEvent = await asyncio.wait_for(
                pipeline.alert_queue.get(), timeout=1.0
            )
            _alert_store.append(alert.model_dump(mode="json"))
        except asyncio.TimeoutError:
            pass


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/video_feed")
async def video_feed():
    """MJPEG stream of the annotated surveillance video."""
    def generate():
        while True:
            frame = pipeline.latest_frame
            if frame is None:
                time.sleep(0.05)
                continue
            ret, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
            if not ret:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buf.tobytes()
                + b"\r\n"
            )
            time.sleep(1 / 25)  # cap at 25 fps for stream

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()   # keep alive
    except WebSocketDisconnect:
        manager.disconnect(ws)


@app.post("/api/alerts", response_model=AlertResponse)
async def receive_alert(alert: AlertEvent):
    """Accept an alert event (can be called externally too)."""
    _alert_store.append(alert.model_dump(mode="json"))
    log.info(f"Alert stored: {alert.event_type} (id={alert.event_id})")
    return AlertResponse(event_id=alert.event_id, message="Stored")


@app.get("/api/alerts")
async def list_alerts(
    limit: int = 50,
    event_type: Optional[str] = None,
):
    """Return stored alerts (newest first)."""
    alerts = list(_alert_store)
    if event_type:
        alerts = [a for a in alerts if a.get("event_type") == event_type]
    return {"alerts": list(reversed(alerts))[:limit], "total": len(alerts)}


@app.get("/api/status", response_model=SystemStatus)
async def system_status():
    stats = pipeline.latest_stats
    return SystemStatus(
        running=pipeline.running,
        camera_source=settings.camera_source,
        model=settings.yolo_model,
        device=settings.device,
        uptime_seconds=round(pipeline.uptime, 1),
        total_alerts=pipeline.total_alerts,
        fps=stats.get("fps", 0.0),
    )


@app.post("/api/control")
async def control(action: str):
    """Start or stop the pipeline."""
    if action == "start":
        pipeline.start()
        return {"status": "started"}
    elif action == "stop":
        pipeline.stop()
        return {"status": "stopped"}
    raise HTTPException(400, "action must be 'start' or 'stop'")
