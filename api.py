# =============================================================================
# api.py — Flask REST API + live HTML dashboard
#
# Endpoints:
#   GET  /                    → HTML dashboard (auto-refreshes every 3 s)
#   GET  /live                → MJPEG live video stream
#   GET  /api/detections      → JSON list of recent detections
#   GET  /api/alerts          → JSON list of recent alerts
#   GET  /api/stats           → JSON aggregate statistics
#   GET  /api/latest_caption  → JSON {caption, frame_id, inference_ms}
# =============================================================================

import logging
import os
import time
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request

import config
import database as db
from detector import FrameBuffer
from signal_controller import TrafficSignalController

_signal_ctrl = TrafficSignalController()

if TYPE_CHECKING:
    from detector import YOLOv8Detector

logger = logging.getLogger(__name__)

app = Flask(__name__)

_frame_buffer: Optional[FrameBuffer] = None
_detector: Optional["YOLOv8Detector"] = None
_signal_state_provider = None


def set_frame_buffer(buf: FrameBuffer) -> None:
    global _frame_buffer
    _frame_buffer = buf


def set_detector(det: "YOLOv8Detector") -> None:
    global _detector
    _detector = det


def set_signal_state_provider(provider) -> None:
    global _signal_state_provider
    _signal_state_provider = provider


def _scan_video_files() -> list[dict]:
    """Return all video files in the project directory as source descriptors."""
    project_dir = os.path.dirname(os.path.abspath(__file__))
    sources = []
    for fname in sorted(os.listdir(project_dir)):
        _, ext = os.path.splitext(fname.lower())
        if ext in _VIDEO_EXTENSIONS:
            sources.append(
                {
                    "filename": fname,
                    "path": os.path.join(project_dir, fname),
                    "label": os.path.splitext(fname)[0],
                }
            )
    return sources


# ---------------------------------------------------------------------------
# HTML Dashboard (single-file, no external templates)
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Smart Traffic System — 4-Way Intersection</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e2e8f0; }

    header {
      background: #1a1d2e; padding: 14px 24px;
      display: flex; align-items: center; justify-content: space-between;
      border-bottom: 1px solid #2d3148;
    }
    header h1 { font-size: 1.2rem; font-weight: 700; letter-spacing: .4px; }
    .badge { background: #22c55e; color: #000; font-size: .7rem;
             padding: 2px 10px; border-radius: 99px; font-weight: 700; }

    .layout { display: grid; grid-template-columns: 1fr 380px; gap: 20px; padding: 20px; }
    @media(max-width:1200px){ .layout { grid-template-columns: 1fr; } }

    .card { background: #1a1d2e; border-radius: 12px; border: 1px solid #2d3148; padding: 20px; }
    .card h2 { font-size: .8rem; text-transform: uppercase; letter-spacing: 1.2px;
               color: #64748b; margin-bottom: 16px; }

    .stream img { width: 100%; border-radius: 8px; display: block; border: 1px solid #2d3148; }

    .caption-box { background: #0f1117; border-radius: 8px; padding: 15px;
                   font-size: 1rem; line-height: 1.6; margin-top: 15px; min-height: 60px;
                   border-left: 4px solid #38bdf8; }

    /* 4-Way Intersection Grid */
    .intersection-grid {
      display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
    }

    .direction-box {
      background: #111421; border: 2px solid #2d3148; border-radius: 10px;
      padding: 12px; text-align: center; transition: all 0.3s ease;
    }
    .direction-box.active { border-color: #00ff66; background: #162521; }

    .dir-name { font-size: .7rem; font-weight: 700; text-transform: uppercase;
                color: #94a3b8; letter-spacing: 1px; margin-bottom: 10px; }

    .signal-lights {
      width: 34px; height: 80px; margin: 0 auto 10px; background: #000;
      border-radius: 17px; padding: 5px; display: flex; flex-direction: column; gap: 4px;
      border: 2px solid #222;
    }

    .light {
      width: 20px; height: 20px; border-radius: 50%; margin: 0 auto;
      opacity: .05; transition: opacity .2s, box-shadow .2s;
    }
    .light.red    { background: #ff3333; }
    .light.yellow { background: #ffcc00; }
    .light.green  { background: #00ff66; }

    .light.on { opacity: 1; }
    .light.red.on    { box-shadow: 0 0 15px #ff3333; }
    .light.yellow.on { box-shadow: 0 0 15px #ffcc00; }
    .light.green.on  { box-shadow: 0 0 15px #00ff66; }

    .countdown { font-size: 1.6rem; font-weight: 800; font-family: monospace; margin: 5px 0; }
    .countdown.green  { color: #00ff66; }
    .countdown.red    { color: #ff3333; }

    .density-meter {
      height: 4px; background: #000; border-radius: 2px; margin: 8px 0;
      overflow: hidden;
    }
    .density-fill { height: 100%; width: 0%; transition: width 0.5s ease; }

    .dir-stats { display: flex; justify-content: space-between; font-size: .6rem; color: #64748b; }

    .phase-indicator {
      padding: 12px; border-radius: 8px; font-size: .85rem; font-weight: 600;
      text-align: center; margin-bottom: 15px; border: 1px solid #2d3148;
    }
    .phase-ns { color: #00ff66; background: rgba(0, 255, 102, 0.05); }
    .phase-ew { color: #38bdf8; background: rgba(56, 189, 248, 0.05); }
    </style>
    </head>
    <body>
    <header>
    <h1>🚦 Smart Traffic Intelligence Platform</h1>
    <div style="display:flex; align-items:center; gap:15px;">
    <span id="clock" style="font-family:monospace; color:#64748b; font-size:1rem;"></span>
    <span class="badge" id="status">● ONLINE</span>
    </div>
    </header>

    <div class="layout">
    <!-- LEFT: Large stream feed -->
    <div style="display:flex;flex-direction:column;gap:16px;">
    <div class="card">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
        <h2 style="margin-bottom:0;">Live Intelligence Feed</h2>
        <div id="fps-counter" style="font-size:.7rem; color:#4ade80; font-family:monospace;">0 FPS</div>
      </div>
      <img src="/live" alt="Live Intelligence Feed"/>
      <div class="caption-box" id="caption">Initializing Intelligence Engine...</div>
      <div class="inf-tag" id="inf-time"></div>
    </div>
    </div>

    <!-- RIGHT: Signal Management -->
    <div style="display:flex;flex-direction:column;gap:16px;">
    <div class="card">
      <h2>Signal Synchronization</h2>

      <div id="phase-label" class="phase-indicator phase-ns">
        SYNCHRONIZING PHASES...
      </div>

      <div class="intersection-grid">
        <!-- North -->
        <div class="direction-box" id="dir-n">
          <div class="dir-name">North</div>
          <div class="signal-lights">
            <div class="light red"    id="light-n-red"></div>
            <div class="light yellow" id="light-n-yellow"></div>
            <div class="light green"  id="light-n-green"></div>
          </div>
          <div class="countdown" id="cd-n">--</div>
          <div class="density-meter"><div id="fill-n" class="density-fill"></div></div>
          <div class="dir-stats">
            <span id="pcu-n">0.0</span>
            <span id="veh-n">0 VEH</span>
          </div>
        </div>

        <!-- East -->
        <div class="direction-box" id="dir-e">
          <div class="dir-name">East</div>
          <div class="signal-lights">
            <div class="light red"    id="light-e-red"></div>
            <div class="light yellow" id="light-e-yellow"></div>
            <div class="light green"  id="light-e-green"></div>
          </div>
          <div class="countdown" id="cd-e">--</div>
          <div class="density-meter"><div id="fill-e" class="density-fill"></div></div>
          <div class="dir-stats">
            <span id="pcu-e">0.0</span>
            <span id="veh-e">0 VEH</span>
          </div>
        </div>

        <!-- South -->
        <div class="direction-box" id="dir-s">
          <div class="dir-name">South</div>
          <div class="signal-lights">
            <div class="light red"    id="light-s-red"></div>
            <div class="light yellow" id="light-s-yellow"></div>
            <div class="light green"  id="light-s-green"></div>
          </div>
          <div class="countdown" id="cd-s">--</div>
          <div class="density-meter"><div id="fill-s" class="density-fill"></div></div>
          <div class="dir-stats">
            <span id="pcu-s">0.0</span>
            <span id="veh-s">0 VEH</span>
          </div>
        </div>

        <!-- West -->
        <div class="direction-box" id="dir-w">
          <div class="dir-name">West</div>
          <div class="signal-lights">
            <div class="light red"    id="light-w-red"></div>
            <div class="light yellow" id="light-w-yellow"></div>
            <div class="light green"  id="light-w-green"></div>
          </div>
          <div class="countdown" id="cd-w">--</div>
          <div class="density-meter"><div id="fill-w" class="density-fill"></div></div>
          <div class="dir-stats">
            <span id="pcu-w">0.0</span>
            <span id="veh-w">0 VEH</span>
          </div>
        </div>
      </div>

      <div class="formula-box" id="formula-box" style="margin-top:20px; font-family:monospace; font-size:.65rem; color:#4b5563;">
        Webster Optimization: C = (1.5L + 5) / (1 - Y)
      </div>
    </div>
    </div>
    </div>

    <script>
    function updateClock() {
    document.getElementById('clock').textContent = new Date().toLocaleTimeString('en-GB', {hour12:false});
    }
    setInterval(updateClock, 1000);
    updateClock();

    function setLight(dir, state) {
    ['red','yellow','green'].forEach(c => {
      const el = document.getElementById(`light-${dir.charAt(0)}-${c}`);
      if (el) el.classList.toggle('on', c.toUpperCase() === state);
    });
    }

    async function refresh() {
    try {
      const d = await (await fetch('/api/traffic_timing')).json();

      const dirs = ['north', 'south', 'east', 'west'];
      const short = ['n', 's', 'e', 'w'];

      let nsActive = false;
      dirs.forEach((dir, i) => {
        const sd = short[i];
        const data = d[dir];
        const state = data.state || 'RED';
        if ((sd === 'n' || sd === 's') && state === 'GREEN') nsActive = true;

        setLight(sd, state);
        document.getElementById("dir-" + sd).classList.toggle("active", state === "GREEN");

        const cdEl = document.getElementById("cd-" + sd);
        cdEl.textContent = data.timer + "s";

        if (state === "GREEN") {
            cdEl.className = "countdown green";
        } else if (state === "YELLOW") {
            cdEl.className = "countdown";
            cdEl.style.color = "#ffcc00";
        } else {
            cdEl.className = "countdown red";
        }
        document.getElementById('pcu-'+sd).textContent = data.pcu.toFixed(1);
        document.getElementById('veh-'+sd).textContent = data.vehicle_count + ' VEH';

        const fill = document.getElementById('fill-'+sd);
        const pcu = data.pcu;
        fill.style.width = Math.min(100, (pcu / 12) * 100) + '%';
        fill.style.backgroundColor = pcu > 8 ? '#ff3333' : (pcu > 4 ? '#ffcc00' : '#00ff66');
      });

      const label = document.getElementById('phase-label');
      label.textContent = nsActive ? 'NORTH-SOUTH AXIS: ACTIVE' : 'EAST-WEST AXIS: ACTIVE';
      label.className = 'phase-indicator ' + (nsActive ? 'phase-ns' : 'phase-ew');

      document.getElementById('caption').textContent = d.caption || '';
      if (d.inference_ms) {
        document.getElementById('inf-time').textContent = 'LATENCY: ' + d.inference_ms.toFixed(1) + 'ms';
        document.getElementById('fps-counter').textContent = (1000 / d.inference_ms).toFixed(1) + ' FPS';
      }
      document.getElementById('status').textContent = '● SYSTEM ONLINE';
    } catch(e) {
      document.getElementById('status').textContent = '● OFFLINE';
    }
    }

    setInterval(refresh, 1000);
    refresh();
    </script>

</body>
</html>
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def dashboard():
    return render_template_string(_DASHBOARD_HTML)


@app.route("/live")
def live_stream():
    """MJPEG stream from the latest annotated frame."""
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def _mjpeg_generator():
    """
    Continuously yields MJPEG frames from the FrameBuffer.
    Falls back to a placeholder image when no frame is available.
    """
    _placeholder = _make_placeholder()

    while True:
        result = _frame_buffer.get() if _frame_buffer else None

        if result is not None:
            img = result.frame
        else:
            img = _placeholder

        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            time.sleep(0.05)
            continue

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(1.0 / config.CAMERA_FPS)


def _make_placeholder() -> np.ndarray:
    """Return a grey placeholder frame with 'No Signal' text."""
    img = np.full((480, 640, 3), 40, dtype=np.uint8)
    cv2.putText(
        img,
        "No Signal",
        (220, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (180, 180, 180),
        2,
        cv2.LINE_AA,
    )
    return img


@app.route("/api/sources")
def api_sources():
    """List all video files in the project directory."""
    return jsonify(_scan_video_files())


@app.route("/api/current_source")
def api_current_source():
    """Return the path and label of the currently active source."""
    if _detector is None:
        return jsonify({"path": config.RTSP_URL, "label": config.RTSP_URL})
    path = _detector.current_source
    label = (
        os.path.splitext(os.path.basename(path))[0] if os.path.isfile(path) else path
    )
    return jsonify({"path": path, "label": label})


@app.route("/api/source", methods=["POST"])
def api_switch_source():
    """Switch the detector to a different video source."""
    data = request.get_json(force=True, silent=True) or {}
    path = data.get("path", "").strip()
    if not path:
        return jsonify({"error": "path required"}), 400
    if not os.path.isfile(path):
        return jsonify({"error": f"File not found: {path}"}), 404
    if _detector is None:
        return jsonify({"error": "Detector not initialised"}), 503
    _detector.switch_source(path)
    logger.info("Source switch requested: %s", path)
    return jsonify({"status": "switching", "path": path})


@app.route("/api/traffic_timing")
def api_traffic_timing():
    """
    Webster-based adaptive signal timing from recent YOLOv8 detections.
    Returns 4-way intersection timings and current live phase data.
    """
    from signal_controller import PCU_WEIGHTS

    recent = db.get_recent_detections(limit=60)
    result = _frame_buffer.get() if _frame_buffer else None
    inf_ms = result.inference_ms if result else None
    caption = recent[0]["scene_caption"] if recent else "Awaiting detections…"

    lane_data = {
        "north": {"pcu": 0.0, "vehicle_count": 0, "breakdown": {}},
        "south": {"pcu": 0.0, "vehicle_count": 0, "breakdown": {}},
        "east": {"pcu": 0.0, "vehicle_count": 0, "breakdown": {}},
        "west": {"pcu": 0.0, "vehicle_count": 0, "breakdown": {}},
    }

    for det in recent:
        lane = det.get("lane", "unknown")
        if lane not in lane_data:
            continue
        cls_name = det.get("class_name", "unknown")
        pcu = PCU_WEIGHTS.get(cls_name, 0.0)
        lane_data[lane]["pcu"] += pcu
        lane_data[lane]["vehicle_count"] += 1
        lane_data[lane]["breakdown"][cls_name] = (
            lane_data[lane]["breakdown"].get(cls_name, 0.0) + pcu
        )

    cycle = _signal_ctrl.compute(lane_data)

    # Sync with live signal state if available
    live_states = _signal_state_provider() if _signal_state_provider else None

    # We need to get the remaining seconds from the PhaseManager
    # To do this safely, we'll assume the proc_loop can provide it
    rem_sec = 0
    timers = {"north": 0, "south": 0, "east": 0, "west": 0}
    from main import _proc_loop_ref

    if _proc_loop_ref and hasattr(_proc_loop_ref, "_phase_mgr"):
        rem_sec = _proc_loop_ref._phase_mgr.remaining_seconds
        timers = _proc_loop_ref._phase_mgr.get_timers()

    return jsonify(
        {
            "north": {
                "vehicle_count": cycle.north.vehicle_count,
                "pcu": cycle.north.pcu,
                "flow_ratio": cycle.north.flow_ratio,
                "suggested_green": cycle.north.green,
                "state": live_states.get("north", "RED") if live_states else "RED",
                "timer": timers.get("north", 0),
            },
            "south": {
                "vehicle_count": cycle.south.vehicle_count,
                "pcu": cycle.south.pcu,
                "flow_ratio": cycle.south.flow_ratio,
                "suggested_green": cycle.south.green,
                "state": live_states.get("south", "RED") if live_states else "RED",
                "timer": timers.get("south", 0),
            },
            "east": {
                "vehicle_count": cycle.east.vehicle_count,
                "pcu": cycle.east.pcu,
                "flow_ratio": cycle.east.flow_ratio,
                "suggested_green": cycle.east.green,
                "state": live_states.get("east", "RED") if live_states else "RED",
                "timer": timers.get("east", 0),
            },
            "west": {
                "vehicle_count": cycle.west.vehicle_count,
                "pcu": cycle.west.pcu,
                "flow_ratio": cycle.west.flow_ratio,
                "suggested_green": cycle.west.green,
                "state": live_states.get("west", "RED") if live_states else "RED",
                "timer": timers.get("west", 0),
            },
            "cycle_length": cycle.cycle_length,
            "lost_time": cycle.lost_time,
            "y_total": cycle.y_total,
            "remaining_seconds": rem_sec,
            "caption": caption,
            "inference_ms": inf_ms,
        }
    )


@app.route("/api/detections")
def api_detections():
    limit = _safe_int("limit", config.API_DETECTIONS_LIMIT)
    rows = db.get_recent_detections(limit=limit)
    return jsonify(rows)


@app.route("/api/alerts")
def api_alerts():
    limit = _safe_int("limit", config.API_ALERTS_LIMIT)
    rows = db.get_recent_alerts(limit=limit)
    return jsonify(rows)


@app.route("/api/stats")
def api_stats():
    stats = db.get_detection_stats()
    return jsonify(stats)


@app.route("/api/latest_caption")
def api_latest_caption():
    result = _frame_buffer.get() if _frame_buffer else None
    if result is None:
        return jsonify(
            {"caption": "No data yet.", "frame_id": None, "inference_ms": None}
        )

    # Retrieve the caption stored with the most recent frame from DB
    recent = db.get_recent_detections(limit=1)
    caption = recent[0]["scene_caption"] if recent else "Awaiting caption…"
    return jsonify(
        {
            "caption": caption,
            "frame_id": result.frame_id,
            "inference_ms": result.inference_ms,
        }
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_int(param_name: str, default: int) -> int:
    from flask import request

    try:
        return int(request.args.get(param_name, default))
    except (TypeError, ValueError):
        return default
