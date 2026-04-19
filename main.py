"""
main.py
────────
Entry point for the Smart Surveillance System.

Usage
-----
  # Run with auto-reload (dev)
  python main.py

  # Run via uvicorn directly (production)
  uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
"""
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

import uvicorn
from backend.api.server import app
from backend.utils.logger import log
from config.settings import settings


def main():
    log.info("=" * 60)
    log.info(" Smart Surveillance System v1.0")
    log.info(f" Camera  : {settings.camera_source}")
    log.info(f" Model   : {settings.yolo_model}  (device={settings.device})")
    log.info(f" Server  : http://{settings.backend_host}:{settings.backend_port}")
    log.info("=" * 60)

    uvicorn.run(
        "backend.api.server:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=False,
        workers=1,          # Must be 1 – OpenCV camera cannot share across processes
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
