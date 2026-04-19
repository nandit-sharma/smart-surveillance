# ── Stage 1: Base with CUDA support (switch to cpu-only if no GPU) ──────────
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies for OpenCV + audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libopencv-dev \
    ffmpeg \
    v4l-utils \
    alsa-utils \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Stage 2: Install Python deps ────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ── Stage 3: Copy application ────────────────────────────────────────────────
COPY . .

# Create runtime directories
RUN mkdir -p logs snapshots

# ── Runtime ──────────────────────────────────────────────────────────────────
EXPOSE 8000

# Allow camera device access via --device /dev/video0
CMD ["python", "main.py"]
