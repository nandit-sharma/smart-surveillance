# 🎯 Real-Time Smart Surveillance System
### Computer Vision · Deep Learning · Multi-Object Tracking

A production-grade surveillance system that ingests live camera/CCTV feeds and performs real-time object detection, human tracking, motion analysis, and behavior classification — all served through a live web dashboard with WebSocket-pushed alerts.

---

## 🏗️ Architecture Overview

```
Camera / RTSP / File
        │
        ▼
┌─────────────────────────────────────────────────────┐
│                 SurveillancePipeline                │
│                                                     │
│  ┌──────────────┐   ┌─────────────────────────┐    │
│  │ ObjectDetector│──▶│    PersonTracker         │   │
│  │  (YOLOv8)    │   │    (DeepSORT)            │   │
│  └──────────────┘   └───────────┬─────────────┘    │
│                                 │ TrackStates       │
│  ┌──────────────┐               ▼                   │
│  │MotionAnalyzer│   ┌─────────────────────────┐    │
│  │ (Optical Flow│──▶│   BehaviorAnalyzer       │   │
│  │  + BG Sub)   │   │ Fight│Fall│Loiter│Panic  │   │
│  └──────────────┘   │ Robbery│Intrusion         │   │
│                     └───────────┬─────────────┘    │
│                                 │ AlertEvent[]      │
│  ┌──────────────────────────────▼──────────────┐   │
│  │  FrameVisualizer  (bboxes, HUD, banners)    │   │
│  └──────────────────────────────┬──────────────┘   │
└─────────────────────────────────┼───────────────────┘
                                  │
                    ┌─────────────▼─────────────────┐
                    │         FastAPI Server         │
                    │  /video_feed (MJPEG stream)   │
                    │  /ws (WebSocket stats)         │
                    │  /api/alerts (REST)            │
                    │  /  (Live Dashboard HTML)      │
                    └───────────────────────────────┘
```

---

## 📁 Folder Structure

```
smart-surveillance/
├── main.py                          # ← Run this
├── requirements.txt
├── .env.example                     # Copy to .env and configure
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
│
├── config/
│   ├── __init__.py
│   └── settings.py                  # Pydantic-Settings config loader
│
├── core/
│   ├── __init__.py
│   ├── pipeline.py                  # Main orchestrator thread
│   ├── visualization.py             # Frame annotation / HUD / alerts overlay
│   │
│   ├── detection/
│   │   ├── __init__.py
│   │   └── detector.py              # YOLOv8 object detector
│   │
│   ├── tracking/
│   │   ├── __init__.py
│   │   └── tracker.py               # DeepSORT multi-object tracker + TrackState
│   │
│   ├── motion/
│   │   ├── __init__.py
│   │   └── motion_analyzer.py       # Optical flow + BG subtraction + heatmap
│   │
│   └── behavior/
│       ├── __init__.py
│       └── behavior_analyzer.py     # Fight/fall/loiter/intrusion/panic/robbery
│
├── backend/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── server.py                # FastAPI: REST + WebSocket + MJPEG
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py               # Pydantic data models
│   │
│   └── utils/
│       ├── __init__.py
│       ├── alarm.py                 # Audio alarm + snapshot + HTTP dispatch
│       └── logger.py                # Structured logger
│
├── frontend/
│   └── templates/
│       └── dashboard.html           # Live surveillance dashboard
│
├── tests/
│   ├── __init__.py
│   ├── test_behavior.py             # Behavior analyzer unit tests
│   ├── test_motion.py               # Motion analyzer unit tests
│   └── test_schemas.py              # Schema + cooldown tests
│
├── logs/                            # Auto-created at runtime
└── snapshots/                       # Auto-created alert snapshots
```

---

## ⚙️ Configuration (`.env`)

Copy `.env.example` to `.env` and adjust:

| Variable | Default | Description |
|---|---|---|
| `CAMERA_SOURCE` | `0` | Webcam index, RTSP URL, or video file path |
| `YOLO_MODEL` | `yolov8m.pt` | nano / small / medium / large / xlarge |
| `DEVICE` | `cpu` | `cpu`, `cuda`, or `mps` (Apple Silicon) |
| `DETECTION_CONFIDENCE` | `0.45` | YOLO detection threshold |
| `FIGHT_SPEED_THRESHOLD` | `35.0` | px/frame speed to flag a fight |
| `LOITER_TIME_THRESHOLD` | `60` | Seconds before loitering alert |
| `RESTRICTED_ZONES` | `[]` | JSON list of `[[x1,y1],[x2,y2],...]` polygons |
| `ALERT_COOLDOWN_SECONDS` | `5` | Min seconds between same-type alerts |
| `SAVE_SNAPSHOTS` | `true` | Save JPEG snapshots on alert |
| `ENABLE_ALARM` | `true` | Audible beep on alert |
| `BACKEND_PORT` | `8000` | FastAPI server port |

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure
```bash
cp .env.example .env
# Edit .env — set CAMERA_SOURCE, DEVICE, etc.
```

### 3. Run
```bash
python main.py
```

Open **http://localhost:8000** in your browser for the live dashboard.

### 4. Test (no camera/GPU required)
```bash
pytest tests/ -v
```

---

## 🐳 Docker

```bash
# Build & run (CPU, webcam passthrough)
docker-compose up --build

# With GPU
# Edit docker-compose.yml and uncomment the deploy.resources.reservations block
docker-compose up --build
```

---

## 🌐 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Live dashboard HTML |
| `GET` | `/video_feed` | MJPEG video stream |
| `WS` | `/ws` | Real-time stats JSON (10 Hz) |
| `GET` | `/api/alerts` | List stored alerts (`?limit=50&event_type=fight`) |
| `POST`| `/api/alerts` | Submit an alert event (JSON body = AlertEvent) |
| `GET` | `/api/status` | System health & FPS |
| `POST`| `/api/control` | `?action=start` or `?action=stop` |

---

## 🧠 Detected Event Types

| Event | Trigger Heuristic |
|---|---|
| **Fight / Violence** | Two persons with high bounding-box overlap AND both moving fast |
| **Fall Detection** | Person bbox width/height ratio > threshold AND near-zero velocity |
| **Loitering** | Person stationary for > `LOITER_TIME_THRESHOLD` seconds |
| **Intrusion** | Person center inside a defined restricted-zone polygon |
| **Crowd Panic** | ≥N persons all moving above panic-speed simultaneously |
| **Robbery / Theft** | Person in proximity to valuables (bag, phone, etc.) + speed burst |

---

## 🔧 Extending the System

### Add a new behavior detector
1. Add a new `EventType` enum value in `backend/models/schemas.py`
2. Implement a `_check_<name>` method in `core/behavior/behavior_analyzer.py`
3. Call it inside `analyze()` and append results to `alerts`

### Switch tracker to ByteTrack
Replace `deep_sort_realtime` with `supervision` + `ByteTrack` in `core/tracking/tracker.py`.  
The `update()` method interface stays the same.

### Add RTSP / IP Camera
```ini
# .env
CAMERA_SOURCE=rtsp://admin:password@192.168.1.100:554/stream
```

### Add restricted zones via the API (future)
Currently zones are set in `.env`. A `POST /api/zones` endpoint can be added to `backend/api/server.py` to hot-reload them.

---

## 📊 Performance Tips

| Setting | Recommended for speed |
|---|---|
| `YOLO_MODEL=yolov8n.pt` | Nano model — fastest, slightly less accurate |
| `DEVICE=cuda` | GPU inference — 10× faster than CPU |
| `DETECTION_CONFIDENCE=0.55` | Higher threshold → fewer detections → faster |
| Resize input frames | Add `cv2.resize(frame, (640, 480))` before detection |

---

## 📄 License
MIT — free to use and modify.
