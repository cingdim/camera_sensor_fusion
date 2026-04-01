#!/usr/bin/env bash
# Lightweight ffmpeg launcher that streams V4L2 video to RTP based on env vars.
set -euo pipefail

log() {
  printf '[%s] %s\n' "${CAMERA_NAME:-camera}" "$1"
}

: "${DEST_IP:?DEST_IP must be set}"
: "${DEST_PORT:?DEST_PORT must be set}"
VIDEO_DEVICE="${VIDEO_DEVICE:-/dev/video0}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"
FPS="${FPS:-30}"
CAMERA_NAME="${CAMERA_NAME:-pi-cam}"

log "Starting ffmpeg pipeline from ${VIDEO_DEVICE} to ${DEST_IP}:${DEST_PORT}"
log "Resolution ${WIDTH}x${HEIGHT} @ ${FPS}fps"

exec ffmpeg \
  -f v4l2 \
  -input_format mjpeg \
  -video_size "${WIDTH}x${HEIGHT}" \
  -framerate "${FPS}" \
  -i "${VIDEO_DEVICE}" \
  -an \
  -vf "scale=${WIDTH}:${HEIGHT},format=yuv420p" \
  -c:v libx264 \
  -preset ultrafast \
  -tune zerolatency \
  -payload_type 96 \
  -f rtp "rtp://${DEST_IP}:${DEST_PORT}"
