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
from typing import Optional, TYPE_CHECKING

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

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def set_frame_buffer(buf: FrameBuffer) -> None:
    global _frame_buffer
    _frame_buffer = buf


def set_detector(det: "YOLOv8Detector") -> None:
    global _detector
    _detector = det


def _scan_video_files() -> list[dict]:
    """Return all video files in the project directory as source descriptors."""
    project_dir = os.path.dirname(os.path.abspath(__file__))
    sources = []
    for fname in sorted(os.listdir(project_dir)):
        _, ext = os.path.splitext(fname.lower())
        if ext in _VIDEO_EXTENSIONS:
            sources.append({
                "filename": fname,
                "path": os.path.join(project_dir, fname),
                "label": os.path.splitext(fname)[0],
            })
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

    .layout { display: grid; grid-template-columns: 1fr 520px; gap: 16px; padding: 18px; }
    @media(max-width:1200px){ .layout { grid-template-columns: 1fr; } }

    .card { background: #1a1d2e; border-radius: 10px; border: 1px solid #2d3148; padding: 16px; }
    .card h2 { font-size: .75rem; text-transform: uppercase; letter-spacing: 1px;
               color: #64748b; margin-bottom: 14px; }

    .stream img { width: 100%; border-radius: 6px; display: block; }
    .stream-label { font-size: .8rem; color: #38bdf8; margin-bottom: 8px; }

    .caption-box { background: #0f1117; border-radius: 6px; padding: 11px 13px;
                   font-size: .88rem; line-height: 1.6; margin-top: 12px; min-height: 52px; }
    .inf-tag { font-size: .72rem; color: #4ade80; margin-top: 5px; text-align: right; }

    .src-list { display: flex; flex-direction: column; gap: 7px; }
    .src-btn {
      background: #0f1117; border: 1px solid #2d3148; border-radius: 7px;
      color: #cbd5e1; padding: 9px 13px; cursor: pointer; font-size: .8rem;
      display: flex; align-items: center; gap: 8px; transition: background .15s, border-color .15s;
    }
    .src-btn:hover { background: #1e2235; border-color: #38bdf8; }
    .src-btn.active { background: #0c2340; border-color: #38bdf8; color: #fff; }
    .src-btn::before { content: "○"; color: #4b5563; flex-shrink: 0; }
    .src-btn.active::before { content: "▶"; color: #38bdf8; }
    .src-label { overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
    .switch-note { font-size: .72rem; color: #facc15; margin-top: 8px; display: none; }

    /* 4-Way Intersection Grid */
    .intersection-grid {
      display: grid; grid-template-columns: 1fr 1fr; gap: 10px;
      margin-bottom: 14px; background: #0a0d18; padding: 12px; border-radius: 8px;
    }

    .direction-box {
      background: #1a1d2e; border: 2px solid #2d3148; border-radius: 8px;
      padding: 10px; text-align: center; transition: border-color .3s;
    }
    .direction-box.active { border-color: #22c55e; box-shadow: 0 0 12px #22c55e44; }

    .dir-name { font-size: .7rem; font-weight: 700; text-transform: uppercase;
                color: #94a3b8; letter-spacing: .6px; margin-bottom: 6px; }

    .signal-lights {
      width: 40px; height: 90px; margin: 0 auto 8px; background: #000;
      border-radius: 6px; padding: 6px; display: flex; flex-direction: column; gap: 4px;
    }

    .light {
      width: 28px; height: 24px; border-radius: 50%; margin: 0 auto;
      opacity: .15; transition: opacity .25s, box-shadow .25s;
    }
    .light.red    { background: #ef4444; }
    .light.yellow { background: #eab308; }
    .light.green  { background: #22c55e; }
    .light.on {
      opacity: 1;
    }
    .light.red.on    { box-shadow: 0 0 12px #ef4444, 0 0 4px #ef4444; }
    .light.yellow.on { box-shadow: 0 0 12px #eab308, 0 0 4px #eab308; }
    .light.green.on  { box-shadow: 0 0 12px #22c55e, 0 0 4px #22c55e; }

    .countdown { font-size: 1.4rem; font-weight: 800; line-height: 1; margin: 4px 0; }
    .countdown.green  { color: #22c55e; }
    .countdown.red    { color: #ef4444; }
    .countdown.yellow { color: #eab308; }

    .dir-pcu { font-size: .68rem; color: #64748b; margin-top: 4px; }
    .dir-veh { font-size: .65rem; color: #4b5563; }

    /* Phase indicator */
    .phase-info {
      background: #0a0d18; border-left: 3px solid #38bdf8; border-radius: 0 4px 4px 0;
      padding: 10px 12px; font-size: .8rem; color: #cbd5e1; line-height: 1.4;
      margin-bottom: 14px;
    }
    .phase-info strong { color: #38bdf8; }

    .formula-box {
      background: #0a0d18; border: 1px solid #2d3148; border-radius: 6px;
      padding: 9px 12px; font-size: .7rem; color: #64748b;
      font-family: 'Courier New', monospace; line-height: 1.6; margin-bottom: 12px;
      white-space: pre;
    }
    .formula-box .hl { color: #38bdf8; }

    .density-bar-wrap { background: #1e2235; border-radius: 3px; height: 4px; margin: 2px 0; }
    .density-bar { height: 4px; border-radius: 3px; transition: width .8s ease, background .4s; }

    .breakdown { display: flex; flex-wrap: wrap; gap: 3px; justify-content: center; margin-top: 4px; }
    .chip { background: #1e2235; border-radius: 3px; padding: 2px 5px; font-size: .6rem; color: #94a3b8; }
  </style>
</head>
<body>
<header>
  <h1>🚦 Smart Traffic Signal Control — 4-Way Intersection</h1>
  <span class="badge" id="status">● LIVE</span>
</header>

<div class="layout">
  <!-- LEFT: stream + caption -->
  <div style="display:flex;flex-direction:column;gap:16px;">
    <div class="card">
      <div class="stream-label" id="src-label">Loading source…</div>
      <img src="/live" alt="Live feed" style="width:100%;border-radius:6px;display:block;"/>
      <div class="caption-box" id="caption">Waiting for first frame…</div>
      <div class="inf-tag" id="inf-time"></div>
    </div>
  </div>

  <!-- RIGHT: 4-way control panel -->
  <div style="display:flex;flex-direction:column;gap:16px;">
    <!-- Source switcher -->
    <div class="card">
      <h2>Video Source</h2>
      <div class="src-list" id="src-list"><div style="color:#64748b;font-size:.8rem">Scanning…</div></div>
      <div class="switch-note" id="switch-note">⏳ Switching…</div>
    </div>

    <!-- Traffic Signal Control -->
    <div class="card">
      <h2>Webster Adaptive Signal Control</h2>

      <!-- Current phase -->
      <div class="phase-info" id="phase-info">
        <strong>N/S GREEN</strong> — 18s remaining. E/W holding on RED.
      </div>

      <!-- 4-way intersection visualization -->
      <div class="intersection-grid">
        <!-- North -->
        <div class="direction-box" id="dir-n" style="grid-column: 1 / 2; grid-row: 1;">
          <div class="dir-name">North ↓</div>
          <div class="signal-lights">
            <div class="light red"    id="light-n-red"></div>
            <div class="light yellow" id="light-n-yellow"></div>
            <div class="light green"  id="light-n-green"></div>
          </div>
          <div class="countdown" id="cd-n">—</div>
          <div class="dir-pcu" id="pcu-n">PCU: —</div>
          <div class="dir-veh" id="veh-n">— vehicles</div>
        </div>

        <!-- East (top right) -->
        <div class="direction-box" id="dir-e" style="grid-column: 2 / 3; grid-row: 1;">
          <div class="dir-name">East ←</div>
          <div class="signal-lights">
            <div class="light red"    id="light-e-red"></div>
            <div class="light yellow" id="light-e-yellow"></div>
            <div class="light green"  id="light-e-green"></div>
          </div>
          <div class="countdown" id="cd-e">—</div>
          <div class="dir-pcu" id="pcu-e">PCU: —</div>
          <div class="dir-veh" id="veh-e">— vehicles</div>
        </div>

        <!-- South (bottom left) -->
        <div class="direction-box" id="dir-s" style="grid-column: 1 / 2; grid-row: 2;">
          <div class="dir-name">South ↑</div>
          <div class="signal-lights">
            <div class="light red"    id="light-s-red"></div>
            <div class="light yellow" id="light-s-yellow"></div>
            <div class="light green"  id="light-s-green"></div>
          </div>
          <div class="countdown" id="cd-s">—</div>
          <div class="dir-pcu" id="pcu-s">PCU: —</div>
          <div class="dir-veh" id="veh-s">— vehicles</div>
        </div>

        <!-- West (bottom right) -->
        <div class="direction-box" id="dir-w" style="grid-column: 2 / 3; grid-row: 2;">
          <div class="dir-name">West →</div>
          <div class="signal-lights">
            <div class="light red"    id="light-w-red"></div>
            <div class="light yellow" id="light-w-yellow"></div>
            <div class="light green"  id="light-w-green"></div>
          </div>
          <div class="countdown" id="cd-w">—</div>
          <div class="dir-pcu" id="pcu-w">PCU: —</div>
          <div class="dir-veh" id="veh-w">— vehicles</div>
        </div>
      </div>

      <!-- Formula box -->
      <div class="formula-box" id="formula-box">Waiting for data…</div>
    </div>
  </div>
</div>

<script>
  let sources = [], currentSrc = '';
  let phaseEnd = 0, phaseTotal = 30;
  let nsGreen = 30, ewGreen = 30;
  let currentPhase = 'ns-green';

  async function loadSources() {
    const r = await fetch('/api/sources');
    sources = await r.json();
    renderSources();
  }

  function renderSources() {
    const el = document.getElementById('src-list');
    if (!sources.length) {
      el.innerHTML = '<div style="color:#64748b;font-size:.8rem">No video files found.</div>';
      return;
    }
    el.innerHTML = sources.map(s =>
      `<button class="src-btn${s.path===currentSrc?' active':''}"
               onclick="switchSrc('${s.path.replace(/\\/g,'\\\\')}')"
               title="${s.filename}">
         <span class="src-label">${s.label}</span>
       </button>`
    ).join('');
  }

  async function switchSrc(path) {
    document.getElementById('switch-note').style.display = 'block';
    await fetch('/api/source', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({path}),
    });
    currentSrc = path;
    renderSources();
    setTimeout(()=>{ document.getElementById('switch-note').style.display='none'; }, 3000);
  }

  async function pollSource() {
    try {
      const d = await (await fetch('/api/current_source')).json();
      currentSrc = d.path || '';
      document.getElementById('src-label').textContent = d.label || '—';
      renderSources();
    } catch(e){}
  }

  function setLight(dir, state) {
    ['red','yellow','green'].forEach(c =>
      document.getElementById(`light-${dir}-${c}`).classList.toggle('on', c === state)
    );
  }

  function advancePhase() {
    if (Date.now() < phaseEnd) return;
    currentPhase = currentPhase === 'ns-green' ? 'ew-green' : 'ns-green';
    const duration = currentPhase === 'ns-green' ? nsGreen : ewGreen;
    phaseTotal = duration;
    phaseEnd = Date.now() + duration * 1000;
  }

  function renderSignals() {
    const now = Date.now();
    const rem = Math.max(0, Math.ceil((phaseEnd - now) / 1000));
    const isNS = currentPhase === 'ns-green';

    // Set lights
    if (isNS) {
      setLight('n', 'green');
      setLight('s', 'green');
      setLight('e', 'red');
      setLight('w', 'red');
    } else {
      setLight('n', 'red');
      setLight('s', 'red');
      setLight('e', 'green');
      setLight('w', 'green');
    }

    // Update boxes
    document.getElementById('dir-n').classList.toggle('active', isNS);
    document.getElementById('dir-s').classList.toggle('active', isNS);
    document.getElementById('dir-e').classList.toggle('active', !isNS);
    document.getElementById('dir-w').classList.toggle('active', !isNS);

    // Countdowns
    const greenDirs = isNS ? ['n','s'] : ['e','w'];
    greenDirs.forEach(d => {
      const el = document.getElementById('cd-'+d);
      el.textContent = rem + 's';
      el.className = 'countdown green';
    });
    const redDirs = isNS ? ['e','w'] : ['n','s'];
    redDirs.forEach(d => {
      const el = document.getElementById('cd-'+d);
      el.textContent = '—';
      el.className = 'countdown red';
    });

    // Phase info
    const msg = isNS
      ? `<strong>N/S GREEN</strong> — ${rem}s remaining. E/W holding on RED.`
      : `<strong>E/W GREEN</strong> — ${rem}s remaining. N/S holding on RED.`;
    document.getElementById('phase-info').innerHTML = msg;
  }

  async function refreshTimings() {
    try {
      const d = await (await fetch('/api/traffic_timing')).json();

      nsGreen = d.north.suggested_green;
      ewGreen = d.east.suggested_green;

      const dirs = ['north', 'south', 'east', 'west'];
      const shortDirs = ['n', 's', 'e', 'w'];

      dirs.forEach((dir, i) => {
        const sd = shortDirs[i];
        const data = d[dir];
        document.getElementById('pcu-'+sd).textContent = `PCU: ${data.pcu.toFixed(2)}`;
        document.getElementById('veh-'+sd).textContent = `${data.vehicle_count} vehicles`;
      });

      document.getElementById('formula-box').innerHTML =
        `<span class="hl">N/S Green:</span> ${nsGreen}s (${d.north.flow_ratio.toFixed(3)})\n` +
        `<span class="hl">E/W Green:</span> ${ewGreen}s (${d.east.flow_ratio.toFixed(3)})\n` +
        `<span class="hl">Cycle:</span> ${d.cycle_length}s  Lost Time: ${d.lost_time}s  Y_total: ${d.y_total.toFixed(3)}\n` +
        `<span class="hl">Formula:</span> C = (1.5×${d.lost_time}+5)/(1−${d.y_total.toFixed(3)}) = ${d.cycle_length}s`;

      document.getElementById('caption').textContent = d.caption || '—';
      if (d.inference_ms) {
        document.getElementById('inf-time').textContent = `Inference: ${d.inference_ms.toFixed(1)} ms`;
      }

      document.getElementById('status').textContent = '● LIVE';
    } catch(e) {
      document.getElementById('status').textContent = '● OFFLINE';
    }
  }

  phaseEnd = Date.now() + nsGreen * 1000;
  phaseTotal = nsGreen;
  setInterval(() => { advancePhase(); renderSignals(); }, 1000);
  setInterval(refreshTimings, 3000);

  loadSources();
  pollSource();
  refreshTimings();
  renderSignals();
  setInterval(pollSource, 5000);
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

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buf.tobytes()
            + b"\r\n"
        )
        time.sleep(1.0 / config.CAMERA_FPS)


def _make_placeholder() -> np.ndarray:
    """Return a grey placeholder frame with 'No Signal' text."""
    img = np.full((480, 640, 3), 40, dtype=np.uint8)
    cv2.putText(
        img, "No Signal",
        (220, 240), cv2.FONT_HERSHEY_SIMPLEX,
        1.2, (180, 180, 180), 2, cv2.LINE_AA,
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
    label = os.path.splitext(os.path.basename(path))[0] if os.path.isfile(path) else path
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
    Returns 4-way intersection timings (north, south, east, west).
    See signal_controller.py for the full mathematical model.
    """
    from signal_controller import PCU_WEIGHTS

    recent = db.get_recent_detections(limit=60)
    result = _frame_buffer.get() if _frame_buffer else None
    inf_ms = result.inference_ms if result else None
    caption = recent[0]["scene_caption"] if recent else "Awaiting detections…"

    # Build lane_data from detections, matching signal_controller.compute() input
    lane_data = {
        "north": {"pcu": 0.0, "vehicle_count": 0, "breakdown": {}},
        "south": {"pcu": 0.0, "vehicle_count": 0, "breakdown": {}},
        "east":  {"pcu": 0.0, "vehicle_count": 0, "breakdown": {}},
        "west":  {"pcu": 0.0, "vehicle_count": 0, "breakdown": {}},
    }

    for det in recent:
        lane = det.get("lane", "unknown")
        if lane not in lane_data:
            continue

        cls_name = det.get("class_name", "unknown")
        pcu = PCU_WEIGHTS.get(cls_name, 0.0)
        lane_data[lane]["pcu"] += pcu
        lane_data[lane]["vehicle_count"] += 1
        lane_data[lane]["breakdown"][cls_name] = lane_data[lane]["breakdown"].get(cls_name, 0.0) + pcu

    cycle = _signal_ctrl.compute(lane_data)

    return jsonify({
        "north": {
            "vehicle_count":   cycle.north.vehicle_count,
            "pcu":             cycle.north.pcu,
            "flow_ratio":      cycle.north.flow_ratio,
            "suggested_green": cycle.north.green,
            "yellow":          cycle.north.yellow,
            "density_level":   cycle.north.density_level,
            "breakdown":       cycle.north.breakdown,
        },
        "south": {
            "vehicle_count":   cycle.south.vehicle_count,
            "pcu":             cycle.south.pcu,
            "flow_ratio":      cycle.south.flow_ratio,
            "suggested_green": cycle.south.green,
            "yellow":          cycle.south.yellow,
            "density_level":   cycle.south.density_level,
            "breakdown":       cycle.south.breakdown,
        },
        "east": {
            "vehicle_count":   cycle.east.vehicle_count,
            "pcu":             cycle.east.pcu,
            "flow_ratio":      cycle.east.flow_ratio,
            "suggested_green": cycle.east.green,
            "yellow":          cycle.east.yellow,
            "density_level":   cycle.east.density_level,
            "breakdown":       cycle.east.breakdown,
        },
        "west": {
            "vehicle_count":   cycle.west.vehicle_count,
            "pcu":             cycle.west.pcu,
            "flow_ratio":      cycle.west.flow_ratio,
            "suggested_green": cycle.west.green,
            "yellow":          cycle.west.yellow,
            "density_level":   cycle.west.density_level,
            "breakdown":       cycle.west.breakdown,
        },
        "cycle_length":  cycle.cycle_length,
        "lost_time":     cycle.lost_time,
        "total_pcu":     cycle.total_pcu,
        "y_total":       cycle.y_total,
        "formula_note":  cycle.formula_note,
        "caption":       caption,
        "inference_ms":  inf_ms,
    })


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
        return jsonify({"caption": "No data yet.", "frame_id": None, "inference_ms": None})

    # Retrieve the caption stored with the most recent frame from DB
    recent = db.get_recent_detections(limit=1)
    caption = recent[0]["scene_caption"] if recent else "Awaiting caption…"
    return jsonify({
        "caption":      caption,
        "frame_id":     result.frame_id,
        "inference_ms": result.inference_ms,
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_int(param_name: str, default: int) -> int:
    from flask import request
    try:
        return int(request.args.get(param_name, default))
    except (TypeError, ValueError):
        return default