# Middleware container for the camera_sensor_fusion Python stack.
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        gstreamer1.0-tools \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for better layer caching.
WORKDIR /app
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the remainder of the project into /app.
COPY . /app

# Default command matches the existing manual invocation.
CMD ["python", "-m", "camera_fusion.launch", "configs/cam1.json", "configs/cam2.json"]
