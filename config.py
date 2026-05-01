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
RTSP_URL = os.getenv("RTSP_URL", "0")
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# Loop local video files when they reach the end (set False to stop at EOF)
VIDEO_LOOP = os.getenv("VIDEO_LOOP", "true").lower() != "false"

# --- YOLOv8 Model ---
# Use a lightweight model for Jetson Nano: yolov8n (nano) or yolov8s (small)
# Export to TensorRT (.engine) for maximum performance on Jetson
MODEL_PATH = os.getenv(
    "MODEL_PATH", "yolov8n.pt"
)  # swap to yolov8n.engine after export
CONFIDENCE_THRESH = float(os.getenv("CONF_THRESH", "0.45"))
IOU_THRESH = float(os.getenv("IOU_THRESH", "0.50"))
DEVICE = os.getenv("DEVICE", "cpu")  # 'cuda' on Jetson Nano, 'cpu' for local testing
INFERENCE_IMG_SIZE = 640  # YOLOv8 input resolution

# --- Detection Processing ---
FRAME_SKIP = 2  # Run inference every N frames to reduce Jetson load
MAX_DETECTIONS_PER_FRAME = 50

# --- Classes of Interest for Alerts (COCO class names) ---
# Detections of these classes trigger NLP-generated alerts
ALERT_CLASSES = {
    "person": {"min_confidence": 0.55, "severity": "HIGH"},
    "car": {"min_confidence": 0.50, "severity": "MEDIUM"},
    "truck": {"min_confidence": 0.50, "severity": "MEDIUM"},
    "motorcycle": {"min_confidence": 0.50, "severity": "MEDIUM"},
    "bicycle": {"min_confidence": 0.45, "severity": "LOW"},
    "fire": {"min_confidence": 0.60, "severity": "CRITICAL"},
    "knife": {"min_confidence": 0.65, "severity": "CRITICAL"},
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

# --- Lane Configuration (4-way intersection) ---
# Define normalized polygon coordinates for each lane (0-1 range)
LANES = {
    "north": {  # approaching from top
        "polygon": [(0.35, 0.0), (0.65, 0.0), (0.65, 0.5), (0.35, 0.5)],
        "direction": "N->S",
        "stop_line_y": 0.48,
    },
    "south": {  # approaching from bottom
        "polygon": [(0.35, 0.5), (0.65, 0.5), (0.65, 1.0), (0.35, 1.0)],
        "direction": "S->N",
        "stop_line_y": 0.52,
    },
    "east": {  # approaching from right
        "polygon": [(0.5, 0.35), (1.0, 0.35), (1.0, 0.65), (0.5, 0.65)],
        "direction": "E->W",
        "stop_line_x": 0.52,
    },
    "west": {  # approaching from left
        "polygon": [(0.0, 0.35), (0.5, 0.35), (0.5, 0.65), (0.0, 0.65)],
        "direction": "W->E",
        "stop_line_x": 0.48,
    },
}

# --- Vehicle Tracker Configuration ---
TRACKER_WINDOW_FRAMES = int(
    os.getenv("TRACKER_WINDOW", "150")
)  # Rolling history window
TRACKER_MAX_AGE = int(os.getenv("TRACKER_MAX_AGE", "30"))  # Frames before track dies
TRACKER_MIN_HITS = int(
    os.getenv("TRACKER_MIN_HITS", "3")
)  # Detections before track confirmed
TRACKER_MAX_DISTANCE = float(
    os.getenv("TRACKER_MAX_DIST", "100")
)  # Max pixels for centroid matching
TRACKER_IOU_THRESHOLD = float(
    os.getenv("TRACKER_IOU", "0.3")
)  # Min IoU for bounding box match
TRACKER_VELOCITY_ALPHA = float(
    os.getenv("TRACKER_VEL_ALPHA", "0.8")
)  # Exponential smoothing factor

# --- Accident Detector Configuration ---
ACCIDENT_WINDOW_SEC = float(os.getenv("ACCIDENT_WINDOW", "5.0"))
ACCIDENT_THRESHOLD = float(os.getenv("ACCIDENT_THRESHOLD", "0.7"))
ACCIDENT_COLLISION_IOU = float(os.getenv("ACCIDENT_COLLISION_IOU", "0.2"))
ACCIDENT_STATIONARY_SEC = float(os.getenv("ACCIDENT_STATIONARY", "3.0"))
ACCIDENT_SUDDEN_STOP_THRESHOLD = float(os.getenv("ACCIDENT_STOP_THRESH", "0.5"))

# --- Emergency Responder Configuration ---
EMERGENCY_WEBHOOK_URL = os.getenv(
    "EMERGENCY_WEBHOOK_URL", None
)  # Optional webhook for emergency events
EMERGENCY_WEBHOOK_TIMEOUT = float(os.getenv("EMERGENCY_WEBHOOK_TIMEOUT", "3.0"))
EMERGENCY_PREEMPTION_DURATION = float(os.getenv("EMERGENCY_PREEMPTION", "30.0"))

# --- Traffic Simulator Configuration (for demo/testing) ---
SIMULATOR_ENABLED = os.getenv("SIMULATOR_ENABLED", "false").lower() == "true"
SIMULATOR_SCENARIO = os.getenv(
    "SIMULATOR_SCENARIO", "normal"
)  # normal, congestion, collision, stalled, emergency
SIMULATOR_FPS = int(os.getenv("SIMULATOR_FPS", "30"))
SIMULATOR_WIDTH = int(os.getenv("SIMULATOR_WIDTH", "1280"))
SIMULATOR_HEIGHT = int(os.getenv("SIMULATOR_HEIGHT", "720"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "edge_ai.log")
