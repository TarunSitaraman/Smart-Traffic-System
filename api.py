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

import io
import json
import logging
import time
from typing import Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string

import config
import database as db
from detector import FrameBuffer

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Injected by main.py after startup
_frame_buffer: Optional[FrameBuffer] = None


def set_frame_buffer(buf: FrameBuffer) -> None:
    global _frame_buffer
    _frame_buffer = buf


# ---------------------------------------------------------------------------
# HTML Dashboard (single-file, no external templates)
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Edge AI — YOLOv8 Dashboard</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e2e8f0; }
    header { background: #1a1d2e; padding: 14px 24px; display: flex;
             align-items: center; justify-content: space-between; border-bottom: 1px solid #2d3148; }
    header h1 { font-size: 1.25rem; font-weight: 600; letter-spacing: .5px; }
    .badge { background: #22c55e; color: #000; font-size: .7rem;
             padding: 2px 8px; border-radius: 99px; font-weight: 700; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 20px; }
    @media(max-width:900px){ .grid { grid-template-columns: 1fr; } }
    .card { background: #1a1d2e; border-radius: 10px; border: 1px solid #2d3148; padding: 16px; }
    .card h2 { font-size: .8rem; text-transform: uppercase; letter-spacing: 1px;
                color: #94a3b8; margin-bottom: 12px; }
    .stream img { width: 100%; border-radius: 6px; }
    .stat-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
    .stat { background: #0f1117; border-radius: 8px; padding: 12px 18px;
            flex: 1; min-width: 110px; text-align: center; }
    .stat .val { font-size: 1.8rem; font-weight: 700; color: #38bdf8; }
    .stat .lbl { font-size: .72rem; color: #64748b; margin-top: 2px; }
    table { width: 100%; border-collapse: collapse; font-size: .8rem; }
    th { text-align: left; padding: 6px 8px; color: #64748b;
         border-bottom: 1px solid #2d3148; font-weight: 500; }
    td { padding: 6px 8px; border-bottom: 1px solid #1e2235; }
    tr:last-child td { border-bottom: none; }
    .sev-CRITICAL { color: #f87171; font-weight: 700; }
    .sev-HIGH     { color: #fb923c; font-weight: 600; }
    .sev-MEDIUM   { color: #facc15; }
    .sev-LOW      { color: #4ade80; }
    .caption-box { background: #0f1117; border-radius: 6px; padding: 12px;
                   font-size: .9rem; line-height: 1.6; min-height: 60px; }
    .inf-time { font-size: .75rem; color: #4ade80; text-align: right; margin-top: 6px; }
  </style>
</head>
<body>
  <header>
    <h1>🎯 Edge AI · YOLOv8 on Jetson Nano</h1>
    <span class="badge" id="status">● LIVE</span>
  </header>

  <div class="grid">
    <!-- Live stream -->
    <div class="card stream" style="grid-column: span 2">
      <h2>Live Camera Feed (RTSP)</h2>
      <img src="/live" alt="Live stream" onerror="this.src='/static/no_signal.png'"/>
    </div>

    <!-- Stats -->
    <div class="card">
      <h2>Statistics</h2>
      <div class="stat-row" id="stats-row">
        <div class="stat"><div class="val" id="s-total-det">—</div><div class="lbl">Total Detections</div></div>
        <div class="stat"><div class="val" id="s-det-1h">—</div><div class="lbl">Last Hour</div></div>
        <div class="stat"><div class="val" id="s-total-alerts">—</div><div class="lbl">Total Alerts</div></div>
        <div class="stat"><div class="val" id="s-alerts-1h">—</div><div class="lbl">Alerts / Hour</div></div>
      </div>

      <h2 style="margin-top:12px">Scene Caption</h2>
      <div class="caption-box" id="caption">Waiting for first frame…</div>
      <div class="inf-time" id="inf-time"></div>
    </div>

    <!-- Recent Alerts -->
    <div class="card">
      <h2>Recent Alerts</h2>
      <table>
        <thead><tr>
          <th>Time</th><th>Class</th><th>Severity</th><th>Confidence</th>
        </tr></thead>
        <tbody id="alerts-body"></tbody>
      </table>
    </div>

    <!-- Recent Detections -->
    <div class="card" style="grid-column: span 2">
      <h2>Recent Detections</h2>
      <table>
        <thead><tr>
          <th>Time</th><th>Frame</th><th>Class</th><th>Confidence</th><th>Inference ms</th>
        </tr></thead>
        <tbody id="detections-body"></tbody>
      </table>
    </div>
  </div>

  <script>
    function fmtTime(ts){ return ts ? ts.replace('T',' ').slice(0,19) : ''; }
    function fmtConf(c){ return (c*100).toFixed(1)+'%'; }

    async function refresh(){
      try {
        const [statsR, alertsR, detsR, capR] = await Promise.all([
          fetch('/api/stats'),
          fetch('/api/alerts?limit=10'),
          fetch('/api/detections?limit=20'),
          fetch('/api/latest_caption'),
        ]);
        const stats = await statsR.json();
        const alerts = await alertsR.json();
        const dets   = await detsR.json();
        const cap    = await capR.json();

        document.getElementById('s-total-det').textContent   = stats.total_detections ?? '—';
        document.getElementById('s-det-1h').textContent      = stats.detections_last_hour ?? '—';
        document.getElementById('s-total-alerts').textContent = stats.total_alerts ?? '—';
        document.getElementById('s-alerts-1h').textContent   = stats.alerts_last_hour ?? '—';

        document.getElementById('caption').textContent = cap.caption ?? '—';
        document.getElementById('inf-time').textContent =
          cap.inference_ms ? `Inference: ${cap.inference_ms.toFixed(1)} ms` : '';

        const ab = document.getElementById('alerts-body');
        ab.innerHTML = alerts.map(a =>
          `<tr><td>${fmtTime(a.timestamp)}</td><td>${a.class_name}</td>
           <td class="sev-${a.severity}">${a.severity}</td>
           <td>${fmtConf(a.confidence)}</td></tr>`
        ).join('');

        const db = document.getElementById('detections-body');
        db.innerHTML = dets.map(d =>
          `<tr><td>${fmtTime(d.timestamp)}</td><td>${d.frame_id}</td>
           <td>${d.class_name}</td><td>${fmtConf(d.confidence)}</td>
           <td>${d.inference_ms ?? '—'}</td></tr>`
        ).join('');

        document.getElementById('status').textContent = '● LIVE';
      } catch(e) {
        document.getElementById('status').textContent = '● OFFLINE';
      }
    }

    refresh();
    setInterval(refresh, 3000);
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