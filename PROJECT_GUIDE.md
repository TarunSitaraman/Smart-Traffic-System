# Smart Traffic System — Complete Project Guide

> A comprehensive walkthrough of every file, its purpose, and its key code snippets.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [File-by-File Breakdown](#3-file-by-file-breakdown)
   - [config.py](#31-configpy)
   - [detector.py](#32-detectorpy)
   - [nlp_engine.py](#33-nlp_enginepy)
   - [database.py](#34-databasepy)
   - [api.py](#35-apipy)
   - [main.py](#36-mainpy)
   - [requirements.txt](#37-requirementstxt)
4. [Data Flow Summary](#4-data-flow-summary)
5. [Key Design Patterns](#5-key-design-patterns)
6. [Running the Project](#6-running-the-project)
7. [Configuration Reference](#7-configuration-reference)
8. [API Endpoint Reference](#8-api-endpoint-reference)

---

## 1. Project Overview

This is an **Edge AI Smart Traffic / Surveillance System** designed to run on an **NVIDIA Jetson Nano** (a low-power embedded GPU board), though it also works on any machine with Python 3.8+.

**What it does:**

- Reads a live video stream from an RTSP IP camera (e.g. a traffic camera or CCTV)
- Runs **YOLOv8** real-time object detection on each frame to identify people, cars, trucks, motorcycles, bicycles, fire, knives, etc.
- Uses a **lightweight NLP engine** (no cloud, no large LLM required) to generate human-readable scene descriptions and alert messages from the raw detection data
- Persists everything to a local **SQLite database**
- Exposes a **Flask REST API** and a self-contained **live web dashboard** that auto-refreshes every 3 seconds

**Primary use case:** Smart traffic monitoring, edge security surveillance, or any real-time object detection scenario where you need on-device inference with a human-readable overlay and web dashboard.

---

## 2. Architecture Diagram

The system runs as a **single Python process with three threads**:

```
┌────────────────────────────────────────────────────────────┐
│                      main.py (orchestrator)                 │
│                                                            │
│  Thread 1: YOLOv8Detector (detector.py)                    │
│    RTSP Camera → read frame → YOLOv8 inference             │
│                    ↓  put()                                │
│             FrameBuffer  (shared, thread-safe)             │
│                    ↑  get()                                │
│  Thread 2: ProcessingLoop (main.py)                        │
│    DetectionResult → SceneCaptioner (NLP) → DB insert      │
│                    → AlertGenerator (NLP) → DB insert      │
│                                                            │
│  Thread 3: Flask API (api.py)                              │
│    HTTP /        → HTML dashboard                          │
│    HTTP /live    → MJPEG stream (reads FrameBuffer)        │
│    HTTP /api/*   → JSON data (reads SQLite DB)             │
└────────────────────────────────────────────────────────────┘
         ↕
   detections.db  (SQLite, WAL mode, 7-day retention)
```

---

## 3. File-by-File Breakdown

---

### 3.1 `config.py`

**Purpose:** Central configuration hub. Every tunable parameter for the entire system lives here. No magic numbers are scattered through other files — they all read from this module.

**Usage:** Imported by every other module. Settings can be overridden at runtime via environment variables (e.g. `RTSP_URL=rtsp://... python main.py`).

#### Key Code Snippets

**Camera and stream settings:**
```python
RTSP_URL      = os.getenv("RTSP_URL", "rtsp://admin:password@192.168.1.100:554/stream1")
CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS    = 30
```
The RTSP URL is read from an environment variable first, falling back to a default. Set `RTSP_URL=0` to use a USB webcam locally.

**YOLOv8 model settings:**
```python
MODEL_PATH        = os.getenv("MODEL_PATH", "yolov8n.pt")
CONFIDENCE_THRESH = float(os.getenv("CONF_THRESH", "0.45"))
IOU_THRESH        = float(os.getenv("IOU_THRESH",  "0.50"))
DEVICE            = os.getenv("DEVICE", "cuda")
FRAME_SKIP        = 2    # Run inference every N frames
```
`FRAME_SKIP = 2` means only every 2nd frame is processed — a critical optimisation for the Jetson Nano's limited GPU. The model defaults to `yolov8n.pt` (nano, fastest), and can be swapped for `yolov8n.engine` (TensorRT, ~3-4× faster) after a one-time export.

**Alert class rules:**
```python
ALERT_CLASSES = {
    "person":     {"min_confidence": 0.55, "severity": "HIGH"},
    "car":        {"min_confidence": 0.50, "severity": "MEDIUM"},
    "fire":       {"min_confidence": 0.60, "severity": "CRITICAL"},
    "knife":      {"min_confidence": 0.65, "severity": "CRITICAL"},
    ...
}
ALERT_COOLDOWN_SECONDS = 10
```
Each class has its own minimum confidence threshold and severity level. The cooldown prevents the system from flooding the alert log when the same object persists in frame.

---

### 3.2 `detector.py`

**Purpose:** Handles everything related to video capture and YOLOv8 inference. Reads frames from the RTSP stream, runs the model, draws annotated bounding boxes, and publishes results to a shared `FrameBuffer`.

**Usage:** Instantiated in `main.py` and run in its own thread: `detector.run()` blocks indefinitely, reconnecting automatically if the stream drops.

#### Key Code Snippets

**`DetectionResult` dataclass** — the data contract between the detector and the rest of the system:
```python
@dataclass
class DetectionResult:
    frame_id:     int
    frame:        np.ndarray    # BGR image WITH bounding boxes drawn on it
    raw_frame:    np.ndarray    # original BGR image, unmodified
    detections:   List[Dict]    # list of dicts: class_name, confidence, x1/y1/x2/y2
    inference_ms: float
    timestamp:    float = field(default_factory=time.time)
```

**`FrameBuffer`** — thread-safe single-slot buffer. The detector writes the latest result; the API and processing loop both read it independently:
```python
class FrameBuffer:
    def put(self, result: DetectionResult) -> None:
        with self._lock:
            self._result = result

    def get(self) -> Optional[DetectionResult]:
        with self._lock:
            return self._result
```
Only one result is ever stored — this ensures the API always shows the *most recent* frame without queuing delay.

**GStreamer pipeline** — hardware-accelerated H.264 decoding on the Jetson Nano:
```python
pipeline = (
    f"rtspsrc location={rtsp_url} latency=100 ! "
    "rtph264depay ! h264parse ! "
    "nvv4l2decoder ! "       # NVIDIA hardware decoder
    "nvvidconv ! "
    f"video/x-raw,format=BGRx,width={width},height={height} ! "
    "videoconvert ! appsink drop=1"
)
```
If GStreamer is unavailable (e.g. running on a desktop PC), it automatically falls back to standard OpenCV RTSP.

**YOLOv8 warm-up** — prevents the first real frame from having a slow inference time due to JIT/CUDA initialisation:
```python
dummy = np.zeros((640, 640, 3), dtype=np.uint8)
self._model.predict(source=dummy, device=config.DEVICE, verbose=False, conf=...)
```

**`_infer()` — the core inference call** — runs YOLO and converts results to normalised dicts:
```python
results = self._model.predict(
    source=frame, device=config.DEVICE,
    conf=config.CONFIDENCE_THRESH, iou=config.IOU_THRESH,
    imgsz=640, max_det=50, verbose=False,
)
for box, conf, cls_id in zip(boxes, confs, cls_ids):
    detections.append({
        "class_name": names[cls_id],
        "confidence": float(conf),
        "x1": x1 / w,   # normalised to [0,1]
        "y1": y1 / h,
        "x2": x2 / w,
        "y2": y2 / h,
    })
```
Bounding box coordinates are stored as normalised fractions (0.0–1.0) rather than absolute pixels, making them resolution-independent.

**Auto-reconnect loop:**
```python
while not self._stop_evt.is_set():
    try:
        cap = _open_capture(config.RTSP_URL)
        self._detection_loop(cap)
    except Exception as exc:
        logger.error("Stream error: %s — reconnecting in 5 s", exc)
        time.sleep(5)
```

---

### 3.3 `nlp_engine.py`

**Purpose:** Converts raw detection data (lists of class names and bounding boxes) into readable English. Two classes handle this: `SceneCaptioner` generates a scene description per frame, and `AlertGenerator` produces alert notifications when high-priority objects are detected.

**Usage:** Both classes are instantiated in `main.py` and called by the `ProcessingLoop` for every processed frame.

**Design philosophy:** Entirely rule/template-based — no large language model required. This keeps RAM usage low enough to run on the Jetson Nano (which only has 4 GB of shared RAM). The code includes notes on where you could swap in a quantised TinyLlama model if you have the resources.

#### Key Code Snippets

**Template system** — randomised templates prevent repetitive output:
```python
_CAPTION_INTRO = [
    "The scene shows",
    "The camera captures",
    "Currently visible:",
    "In the frame:",
]

_ALERT_TEMPLATES = {
    "CRITICAL": [
        "⚠️  CRITICAL ALERT: {count} {obj} detected with {conf:.0%} confidence at {time}.",
        "🚨  CRITICAL: {obj} identified — confidence {conf:.0%}. Immediate attention required.",
    ],
    "HIGH":   [...],
    "MEDIUM": [...],
    "LOW":    [...],
}
```

**`SceneCaptioner.caption()`** — the main NLP generation method:
```python
def caption(self, detections, frame_width=1280, frame_height=720) -> str:
    counts = Counter(d["class_name"] for d in detections)
    top = counts.most_common(config.NLP_MAX_OBJECTS_IN_CAPTION)
    object_phrases = [self._count_phrase(obj, cnt) for obj, cnt in top]
    primary = max(detections, key=lambda d: d["confidence"])
    position = self._spatial_label(primary, frame_width, frame_height)
    intro = random.choice(_CAPTION_INTRO)
    caption = f"{intro} {', '.join(object_phrases)}."
    if position:
        caption += f" The {primary['class_name']} appears {position} of the frame."
    return caption
```
Example output: *"The camera captures two cars, a person and a bicycle. The person appears at the top-left of the frame."*

**Spatial position mapping:**
```python
h_zone = "left"   if cx < 0.35 else ("right" if cx > 0.65 else "centre")
v_zone = "top"    if cy < 0.35 else ("bottom" if cy > 0.65 else "middle")
```
The frame is divided into a 3×3 grid and the most confident detection's centre point is mapped to a human-readable position string.

**`AlertGenerator.evaluate()`** — cooldown-aware alert generation:
```python
def evaluate(self, detections, frame_id) -> List[Dict]:
    best = {}  # Most confident detection per class
    for d in detections:
        cls = d["class_name"]
        if cls in config.ALERT_CLASSES:
            if cls not in best or d["confidence"] > best[cls]["confidence"]:
                best[cls] = d

    for cls, det in best.items():
        rule = config.ALERT_CLASSES[cls]
        if det["confidence"] < rule["min_confidence"]: continue
        if now - self._last_alert[cls] < cooldown: continue  # Suppress during cooldown
        # ...generate and return alert
```
For each alert-worthy class, the system only fires once per `ALERT_COOLDOWN_SECONDS` (default 10 s), preventing the database from being flooded when a car sits in frame for minutes.

**English pluralisation:**
```python
@staticmethod
def _pluralise(word: str, count: int) -> str:
    if count == 1: return word
    if word.endswith(("s", "sh", "ch", "x", "z")): return word + "es"
    if word.endswith("y") and word[-2] not in "aeiou": return word[:-1] + "ies"
    if word.endswith("f"):   return word[:-1] + "ves"
    return word + "s"
```

---

### 3.4 `database.py`

**Purpose:** All SQLite read/write operations. Provides a clean interface for inserting detections, inserting alerts, querying recent data, getting aggregate statistics, and purging old records.

**Usage:** Called by `ProcessingLoop` (inserts) and `api.py` (queries). Never accessed directly from the detector thread.

#### Key Code Snippets

**Thread-local connections** — SQLite connections are not safe to share across threads, so each thread gets its own:
```python
_local = threading.local()

def _get_conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row     # rows behave like dicts
        _local.conn.execute("PRAGMA journal_mode=WAL")   # allows concurrent reads + writes
        _local.conn.execute("PRAGMA synchronous=NORMAL")
    return _local.conn
```
`WAL` (Write-Ahead Logging) mode is critical here — it allows the Flask API thread to read the database at the same time the processing thread is writing to it, without locking.

**Schema:**
```sql
CREATE TABLE detections (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT    NOT NULL,
    frame_id      INTEGER NOT NULL,
    class_name    TEXT    NOT NULL,
    confidence    REAL    NOT NULL,
    x1, y1, x2, y2  REAL  NOT NULL,    -- normalised bounding box
    scene_caption TEXT,
    inference_ms  REAL
);

CREATE TABLE alerts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT  NOT NULL,
    class_name  TEXT  NOT NULL,
    severity    TEXT  NOT NULL,        -- LOW / MEDIUM / HIGH / CRITICAL
    confidence  REAL  NOT NULL,
    message     TEXT  NOT NULL,
    frame_id    INTEGER
);
```

**Bulk insert** — all detections from a single frame are inserted in one `executemany` call:
```python
conn.executemany(
    "INSERT INTO detections (timestamp, frame_id, class_name, confidence, "
    "x1, y1, x2, y2, scene_caption, inference_ms) VALUES (?,?,?,?,?,?,?,?,?,?)",
    rows,
)
```

**Statistics query** — aggregates used by the dashboard:
```python
def get_detection_stats() -> Dict:
    total_det    = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
    det_1h       = conn.execute("SELECT COUNT(*) FROM detections WHERE timestamp >= ?", (one_hour_ago,)).fetchone()[0]
    class_counts = {r["class_name"]: r["cnt"] for r in class_rows}
    # ... returns total counts, last-hour counts, per-class breakdown
```

**Automatic purge** — called every ~1000 frames to keep the database from growing indefinitely:
```python
def purge_old_records() -> None:
    if config.DB_RETENTION_DAYS <= 0: return   # 0 = keep forever
    cutoff = (datetime.utcnow() - timedelta(days=config.DB_RETENTION_DAYS)).isoformat()
    conn.execute("DELETE FROM detections WHERE timestamp < ?", (cutoff,))
    conn.execute("DELETE FROM alerts     WHERE timestamp < ?", (cutoff,))
```

---

### 3.5 `api.py`

**Purpose:** Flask web server that provides both a live visual dashboard and a JSON REST API. All endpoints are read-only — the API never modifies data.

**Usage:** Instantiated in `main.py` and run in its own thread. The `set_frame_buffer()` function must be called before starting the thread to wire up the live video stream.

#### Key Code Snippets

**Flask app and FrameBuffer injection:**
```python
app = Flask(__name__)
_frame_buffer: Optional[FrameBuffer] = None

def set_frame_buffer(buf: FrameBuffer) -> None:
    global _frame_buffer
    _frame_buffer = buf
```
`main.py` calls `flask_api.set_frame_buffer(frame_buffer)` before starting threads, connecting the live frame data into the API.

**MJPEG live stream** — continuously encodes frames as JPEG and streams them over HTTP:
```python
@app.route("/live")
def live_stream():
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

def _mjpeg_generator():
    while True:
        result = _frame_buffer.get() if _frame_buffer else None
        img = result.frame if result else _make_placeholder()
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        time.sleep(1.0 / config.CAMERA_FPS)
```
The `multipart/x-mixed-replace` MIME type is the MJPEG streaming standard — browsers render it as a continuously updating image. A grey "No Signal" placeholder is shown when no frame is available.

**REST API routes:**
```python
@app.route("/api/detections")
def api_detections():
    limit = _safe_int("limit", config.API_DETECTIONS_LIMIT)
    return jsonify(db.get_recent_detections(limit=limit))

@app.route("/api/alerts")
def api_alerts():
    return jsonify(db.get_recent_alerts(limit=_safe_int("limit", config.API_ALERTS_LIMIT)))

@app.route("/api/stats")
def api_stats():
    return jsonify(db.get_detection_stats())

@app.route("/api/latest_caption")
def api_latest_caption():
    result = _frame_buffer.get() if _frame_buffer else None
    recent = db.get_recent_detections(limit=1)
    caption = recent[0]["scene_caption"] if recent else "Awaiting caption…"
    return jsonify({"caption": caption, "frame_id": result.frame_id, "inference_ms": result.inference_ms})
```

**Dashboard auto-refresh** — the embedded JavaScript in `_DASHBOARD_HTML` polls all four API endpoints every 3 seconds and updates the DOM in place:
```javascript
async function refresh() {
    const [statsR, alertsR, detsR, capR] = await Promise.all([
        fetch('/api/stats'),
        fetch('/api/alerts?limit=10'),
        fetch('/api/detections?limit=20'),
        fetch('/api/latest_caption'),
    ]);
    // ...update DOM elements
}
setInterval(refresh, 3000);
```

---

### 3.6 `main.py`

**Purpose:** The application entry point and orchestrator. Initialises all components, wires them together, spawns the three threads, and handles graceful shutdown.

**Usage:** `python main.py` — this is the only file you run directly.

#### Key Code Snippets

**`ProcessingLoop`** — sits between the detector and the database/NLP:
```python
class ProcessingLoop:
    def run(self) -> None:
        while not self._stop.is_set():
            result = self._buf.get()
            if result is None or result.frame_id == self._last_processed_id:
                time.sleep(0.01)
                continue
            self._last_processed_id = result.frame_id
            self._process(result)

    def _process(self, result: DetectionResult) -> None:
        caption = self._captioner.caption(result.detections, ...)   # NLP
        db.insert_detections(result.frame_id, result.detections, caption, result.inference_ms)
        alerts = self._alert_gen.evaluate(result.detections, frame_id=result.frame_id)
        for alert in alerts:
            db.insert_alert(...)
```
The loop checks `frame_id` to avoid processing the same frame twice — important since the FrameBuffer only holds the latest frame and may be read faster than the detector produces new ones.

**Three-thread startup:**
```python
det_thread  = threading.Thread(target=detector.run,  name="DetectorThread",  daemon=True)
proc_thread = threading.Thread(target=proc_loop.run, name="ProcessingThread", daemon=True)
api_thread  = threading.Thread(
    target=lambda: flask_api.app.run(host=..., port=..., use_reloader=False, threaded=True),
    name="APIThread", daemon=True,
)
det_thread.start()
proc_thread.start()
api_thread.start()
```
All threads are `daemon=True`, meaning they automatically die when the main process exits.

**Graceful shutdown on Ctrl+C / SIGTERM:**
```python
def shutdown(signum, frame):
    stop_event.set()    # signals ProcessingLoop to exit
    detector.stop()     # signals YOLOv8Detector to exit
    sys.exit(0)

signal.signal(signal.SIGINT,  shutdown)
signal.signal(signal.SIGTERM, shutdown)
```

**DB purge scheduling** — runs every ~1000 processed frames rather than on a timer, to avoid blocking during high activity:
```python
purge_counter += 1
if purge_counter >= 1000:
    purge_counter = 0
    db.purge_old_records()
```

---

### 3.7 `requirements.txt`

**Purpose:** Lists all Python package dependencies. Also serves as an installation guide for the Jetson Nano, which requires special builds of PyTorch and torchvision (not available via standard pip).

**Key dependencies:**

| Package | Role |
|---|---|
| `ultralytics >= 8.0` | YOLOv8 model loading, inference, TensorRT export |
| `opencv-python-headless >= 4.7` | Video capture, frame decoding, bounding box drawing, JPEG encoding |
| `numpy >= 1.21` | Array operations on frames and bounding boxes |
| `flask >= 3.0` | Web server for dashboard and REST API |
| `Pillow >= 9.0` | Required by ultralytics for image I/O |
| `pyyaml >= 6.0` | ultralytics config parsing |

**Notable Jetson Nano caveat** — PyTorch must be installed from NVIDIA's special ARM wheel, not from PyPI:
```
pip3 install https://developer.download.nvidia.com/compute/redist/jp/v46/pytorch/
    torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl
```

**TensorRT export** — after the first run with `yolov8n.pt`, run this once to get a ~3-4× speedup:
```python
from ultralytics import YOLO
YOLO("yolov8n.pt").export(format="engine", half=True, device=0)
# Then set MODEL_PATH=yolov8n.engine in config.py
```

---

## 4. Data Flow Summary

```
RTSP Camera
    │
    ▼ (every Nth frame, where N = FRAME_SKIP)
YOLOv8Detector._infer()
    │  produces DetectionResult:
    │    - frame_id
    │    - annotated frame (BGR)
    │    - detections: [{class_name, confidence, x1,y1,x2,y2}, ...]
    │    - inference_ms
    ▼
FrameBuffer.put()           ←──────────────────────────────────┐
    │                                                           │
    ├──► ProcessingLoop._process()                              │
    │        │                                                  │
    │        ├──► SceneCaptioner.caption()                      │
    │        │        → "The scene shows two cars and a person" │
    │        │                                                  │
    │        ├──► db.insert_detections()                        │
    │        │        → SQLite: detections table                │
    │        │                                                  │
    │        └──► AlertGenerator.evaluate()                     │
    │                 → db.insert_alert() if threshold met      │
    │                   → SQLite: alerts table                  │
    │                                                           │
    └──► Flask API Thread reads FrameBuffer.get()              ─┘
             │
             ├──  GET /live          → MJPEG stream of annotated frame
             ├──  GET /api/stats     → JSON aggregate stats from SQLite
             ├──  GET /api/alerts    → JSON recent alerts from SQLite
             ├──  GET /api/detections → JSON recent detections from SQLite
             ├──  GET /api/latest_caption → JSON caption + inference time
             └──  GET /             → HTML dashboard (polls all above)
```

---

## 5. Key Design Patterns

**Thread safety via single-slot buffer:** Rather than using a queue (which would cause processing lag if the consumer is slower than the producer), `FrameBuffer` always holds only the *latest* frame. This means under load, frames are skipped rather than accumulated — the right behaviour for a real-time system.

**Environment variable overrides for every setting:** Every value in `config.py` reads from `os.getenv(...)` first. This makes the system deployable without code changes — just set environment variables.

**Template-based NLP instead of LLM:** The `nlp_engine.py` uses random selection from hand-written templates. This avoids the RAM and latency cost of loading a language model on a 4 GB embedded device, while still producing varied, natural-sounding output. The code is structured so that `SceneCaptioner` can be swapped out for an LLM call without changing anything else.

**WAL mode SQLite:** Using `PRAGMA journal_mode=WAL` means the write thread (ProcessingLoop) and read threads (Flask API) don't block each other — essential since both are active simultaneously.

**Daemon threads + signal handling:** All worker threads are daemon threads, so Python won't hang on exit. The `SIGINT`/`SIGTERM` handler additionally calls `stop()` on the detector to allow clean stream closure.

---

## 6. Running the Project

**Basic run (uses defaults from config.py):**
```bash
python main.py
```

**Override settings via environment variables:**
```bash
RTSP_URL=rtsp://192.168.1.50:554/stream \
MODEL_PATH=yolov8n.engine \
DEVICE=cuda \
CONF_THRESH=0.5 \
python main.py
```

**Local testing with a webcam:**
```bash
RTSP_URL=0 DEVICE=cpu python main.py
```

**Access the dashboard:**
Open `http://localhost:5000` in your browser once the system starts.

**Install dependencies:**
```bash
pip install -r requirements.txt
# On Jetson Nano: follow the special PyTorch install steps in requirements.txt first
```

---

## 7. Configuration Reference

| Variable | Default | Env Override | Description |
|---|---|---|---|
| `RTSP_URL` | `rtsp://admin:password@...` | `RTSP_URL` | IP camera stream URL |
| `CAMERA_WIDTH` | `1280` | — | Frame width |
| `CAMERA_HEIGHT` | `720` | — | Frame height |
| `CAMERA_FPS` | `30` | — | Target FPS |
| `MODEL_PATH` | `yolov8n.pt` | `MODEL_PATH` | YOLO model file |
| `CONFIDENCE_THRESH` | `0.45` | `CONF_THRESH` | Min detection confidence |
| `IOU_THRESH` | `0.50` | `IOU_THRESH` | NMS IoU threshold |
| `DEVICE` | `cuda` | `DEVICE` | `cuda` or `cpu` |
| `FRAME_SKIP` | `2` | — | Process every Nth frame |
| `ALERT_COOLDOWN_SECONDS` | `10` | — | Seconds between repeat alerts |
| `DB_PATH` | `detections.db` | `DB_PATH` | SQLite file location |
| `DB_RETENTION_DAYS` | `7` | — | Days to keep records (0 = forever) |
| `API_HOST` | `0.0.0.0` | `API_HOST` | Flask bind address |
| `API_PORT` | `5000` | `API_PORT` | Flask port |
| `NLP_MAX_OBJECTS_IN_CAPTION` | `8` | — | Max objects mentioned in caption |
| `LOG_LEVEL` | `INFO` | `LOG_LEVEL` | Python logging level |
| `LOG_FILE` | `edge_ai.log` | `LOG_FILE` | Log file path |

---

## 8. API Endpoint Reference

All endpoints are `GET` only.

| Endpoint | Returns | Notes |
|---|---|---|
| `GET /` | HTML page | Auto-refreshing dashboard, polls API every 3 s |
| `GET /live` | MJPEG stream | Live annotated video feed, embeddable in `<img>` tags |
| `GET /api/detections?limit=N` | JSON array | Recent detection rows from SQLite (default N=100) |
| `GET /api/alerts?limit=N` | JSON array | Recent alert rows from SQLite (default N=50) |
| `GET /api/stats` | JSON object | Aggregate counts: total detections, last-hour, per-class breakdown |
| `GET /api/latest_caption` | JSON object | `{caption, frame_id, inference_ms}` for most recent frame |

**Example `/api/stats` response:**
```json
{
  "total_detections": 15420,
  "detections_last_hour": 842,
  "total_alerts": 37,
  "alerts_last_hour": 4,
  "class_counts": {"car": 9200, "person": 4100, "truck": 2120},
  "top_alert_classes": [["person", 22], ["car", 10], ["fire", 5]]
}
```

**Example `/api/latest_caption` response:**
```json
{
  "caption": "The camera captures two cars, a person and a bicycle. The person appears at the top-left of the frame.",
  "frame_id": 4821,
  "inference_ms": 18.3
}
```
