# =============================================================================
# config.py — Central configuration for Edge AI (YOLOv8 + NLP) on Jetson Nano
# =============================================================================

import os

# --- Video Source ---
# Accepts:  RTSP URL      rtsp://user:pass@host/stream
#           HTTP MJPEG    http://host/mjpeg
#           YouTube URL   https://www.youtube.com/watch?v=...  (needs yt-dlp)
#           Local file    /path/to/traffic.mp4  (loops automatically)
#           Webcam        0  (first USB camera)
RTSP_URL = os.getenv(
    "RTSP_URL",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "WhatsApp Video 2026-04-18 at 11.35.33 AM.mp4"),
)
CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS    = 30

# Loop local video files when they reach the end (set False to stop at EOF)
VIDEO_LOOP = os.getenv("VIDEO_LOOP", "true").lower() != "false"

# --- YOLOv8 Model ---
# Use a lightweight model for Jetson Nano: yolov8n (nano) or yolov8s (small)
# Export to TensorRT (.engine) for maximum performance on Jetson
MODEL_PATH       = os.getenv("MODEL_PATH", "yolov8n.pt")   # swap to yolov8n.engine after export
CONFIDENCE_THRESH = float(os.getenv("CONF_THRESH", "0.45"))
IOU_THRESH        = float(os.getenv("IOU_THRESH",  "0.50"))
DEVICE            = os.getenv("DEVICE", "cpu")              # 'cuda' on Jetson Nano, 'cpu' for local testing
INFERENCE_IMG_SIZE = 640                                     # YOLOv8 input resolution

# --- Detection Processing ---
FRAME_SKIP = 2    # Run inference every N frames to reduce Jetson load
MAX_DETECTIONS_PER_FRAME = 50

# --- Classes of Interest for Alerts (COCO class names) ---
# Detections of these classes trigger NLP-generated alerts
ALERT_CLASSES = {
    "person":     {"min_confidence": 0.55, "severity": "HIGH"},
    "car":        {"min_confidence": 0.50, "severity": "MEDIUM"},
    "truck":      {"min_confidence": 0.50, "severity": "MEDIUM"},
    "motorcycle": {"min_confidence": 0.50, "severity": "MEDIUM"},
    "bicycle":    {"min_confidence": 0.45, "severity": "LOW"},
    "fire":       {"min_confidence": 0.60, "severity": "CRITICAL"},
    "knife":      {"min_confidence": 0.65, "severity": "CRITICAL"},
}

# Minimum seconds between repeated alerts for the same class
ALERT_COOLDOWN_SECONDS = 10

# --- Database ---
DB_PATH = os.getenv("DB_PATH", "detections.db")

# How long to keep detection records (days). 0 = keep forever
DB_RETENTION_DAYS = 7

# --- REST API / Dashboard ---
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "5000"))
API_DEBUG = False

# Number of recent detections returned by /detections endpoint
API_DETECTIONS_LIMIT = 100

# Number of recent alerts returned by /alerts endpoint
API_ALERTS_LIMIT = 50

# --- NLP ---
# Max objects to include in a scene caption before saying "and more"
NLP_MAX_OBJECTS_IN_CAPTION = 8

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE  = os.getenv("LOG_FILE",  "edge_ai.log")