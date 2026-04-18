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
  <title>Smart Traffic System</title>
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

    .layout { display: grid; grid-template-columns: 1fr 420px; gap: 16px; padding: 18px; }
    @media(max-width:1000px){ .layout { grid-template-columns: 1fr; } }

    .card { background: #1a1d2e; border-radius: 10px; border: 1px solid #2d3148; padding: 16px; }
    .card h2 { font-size: .75rem; text-transform: uppercase; letter-spacing: 1px;
               color: #64748b; margin-bottom: 14px; }

    /* Stream */
    .stream img { width: 100%; border-radius: 6px; display: block; }
    .stream-label { font-size: .8rem; color: #38bdf8; margin-bottom: 8px; }

    /* Caption */
    .caption-box { background: #0f1117; border-radius: 6px; padding: 11px 13px;
                   font-size: .88rem; line-height: 1.6; margin-top: 12px; min-height: 52px; }
    .inf-tag { font-size: .72rem; color: #4ade80; margin-top: 5px; text-align: right; }

    /* Source switcher */
    .src-list { display: flex; flex-direction: column; gap: 7px; }
    .src-btn {
      background: #0f1117; border: 1px solid #2d3148; border-radius: 7px;
      color: #cbd5e1; padding: 9px 13px; cursor: pointer; font-size: .8rem;
      display: flex; align-items: center; gap: 8px;
      transition: background .15s, border-color .15s;
    }
    .src-btn:hover { background: #1e2235; border-color: #38bdf8; }
    .src-btn.active { background: #0c2340; border-color: #38bdf8; color: #fff; }
    .src-btn::before { content: "○"; color: #4b5563; flex-shrink: 0; }
    .src-btn.active::before { content: "▶"; color: #38bdf8; }
    .src-label { overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
    .switch-note { font-size: .72rem; color: #facc15; margin-top: 8px; display: none; }

    /* ── Traffic Light Control ── */
    .tl-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 14px; }

    .tl-signal {
      background: #0f1117; border-radius: 10px; border: 2px solid #2d3148;
      padding: 14px 10px; text-align: center; transition: border-color .3s, box-shadow .3s;
    }
    .tl-signal.green-active  { border-color: #22c55e; box-shadow: 0 0 20px #22c55e55; }
    .tl-signal.red-active    { border-color: #ef4444; box-shadow: 0 0 10px #ef444422; }
    .tl-signal.yellow-active { border-color: #eab308; box-shadow: 0 0 16px #eab30844; }

    .tl-name { font-size: .72rem; text-transform: uppercase; letter-spacing: .8px;
               color: #94a3b8; margin-bottom: 8px; }

    /* Traffic light housing */
    .tl-housing {
      background: #111; border-radius: 10px; padding: 10px 8px;
      width: 50px; margin: 0 auto 10px; display: flex; flex-direction: column; gap: 7px;
    }
    .tl-bulb {
      width: 30px; height: 30px; border-radius: 50%; opacity: .12;
      transition: opacity .25s, box-shadow .25s;
    }
    .tl-bulb.red    { background: #ef4444; }
    .tl-bulb.yellow { background: #eab308; }
    .tl-bulb.green  { background: #22c55e; }
    .tl-bulb.on     { opacity: 1; }
    .tl-bulb.red.on    { box-shadow: 0 0 16px #ef4444, 0 0 6px #ef4444; }
    .tl-bulb.yellow.on { box-shadow: 0 0 16px #eab308, 0 0 6px #eab308; }
    .tl-bulb.green.on  { box-shadow: 0 0 16px #22c55e, 0 0 6px #22c55e; }

    /* Countdown — big visible number */
    .tl-countdown {
      font-size: 2rem; font-weight: 800; line-height: 1;
      margin: 4px 0 2px; letter-spacing: -1px;
    }
    .tl-countdown.col-green  { color: #22c55e; }
    .tl-countdown.col-yellow { color: #eab308; }
    .tl-countdown.col-red    { color: #ef4444; }

    /* Progress bar under countdown */
    .tl-progress-wrap { background: #1e2235; border-radius: 4px; height: 5px; margin: 4px 6px 8px; }
    .tl-progress { height: 5px; border-radius: 4px; transition: width 1s linear, background .3s; }

    .tl-suggest  { font-size: .78rem; font-weight: 700; color: #38bdf8; margin-top: 4px; }
    .tl-vehicles { font-size: .73rem; color: #94a3b8; margin-top: 2px; }
    .tl-pcu      { font-size: .68rem; color: #4b5563; margin-top: 2px; }

    /* Density bar */
    .density-bar-wrap { background: #1e2235; border-radius: 4px; height: 6px; margin: 6px 4px 2px; }
    .density-bar { height: 6px; border-radius: 4px; transition: width .8s ease, background .4s; }

    /* Density badge */
    .dlvl { display: inline-block; padding: 1px 7px; border-radius: 99px;
            font-size: .65rem; font-weight: 700; margin-top: 4px; }
    .dlvl-LOW    { background: #14532d; color: #4ade80; }
    .dlvl-MEDIUM { background: #713f12; color: #fbbf24; }
    .dlvl-HIGH   { background: #7f1d1d; color: #f87171; }

    /* PCU breakdown chips */
    .breakdown { display: flex; flex-wrap: wrap; gap: 4px; justify-content: center; margin-top: 6px; }
    .chip { background: #1e2235; border-radius: 4px; padding: 2px 6px; font-size: .65rem; color: #94a3b8; }

    /* Formula box */
    .formula-box {
      background: #0a0d18; border: 1px solid #2d3148; border-radius: 6px;
      padding: 9px 12px; font-size: .72rem; color: #64748b;
      font-family: 'Courier New', monospace; line-height: 1.8; margin-bottom: 12px;
      white-space: pre;
    }
    .formula-box .hl { color: #38bdf8; }

    /* Cycle timeline */
    .cycle-bar { display: flex; height: 20px; border-radius: 6px; overflow: hidden; margin-bottom: 12px; }
    .cb-seg {
      display: flex; align-items: center; justify-content: center;
      font-size: .6rem; font-weight: 700; color: #000; transition: flex 1s ease;
      min-width: 0;
    }
    .cb-green  { background: #22c55e; }
    .cb-yellow { background: #eab308; }
    .cb-red    { background: #ef4444; color: #fff; }
    .cb-allred { background: #374151; color: #9ca3af; }

    /* Reason bar */
    .reason-bar {
      background: #0f1117; border-left: 3px solid #38bdf8;
      border-radius: 0 6px 6px 0; padding: 10px 13px;
      font-size: .82rem; color: #cbd5e1; line-height: 1.5; margin-top: 12px;
    }
    .reason-bar .phase-label { font-weight: 700; color: #38bdf8; }
  </style>
</head>
<body>
<header>
  <h1>🚦 Smart Traffic Signal Control — YOLOv8 Edge AI</h1>
  <span class="badge" id="status">● LIVE</span>
</header>

<div class="layout">

  <!-- LEFT: stream + caption -->
  <div style="display:flex;flex-direction:column;gap:16px;">
    <div class="card stream">
      <div class="stream-label" id="src-label">Loading source…</div>
      <img src="/live" alt="Live feed"/>
      <div class="caption-box" id="caption">Waiting for first frame…</div>
      <div class="inf-tag" id="inf-time"></div>
    </div>
  </div>

  <!-- RIGHT: controls -->
  <div style="display:flex;flex-direction:column;gap:16px;">

    <!-- Source switcher -->
    <div class="card">
      <h2>Video Source</h2>
      <div class="src-list" id="src-list"><div style="color:#64748b;font-size:.8rem">Scanning…</div></div>
      <div class="switch-note" id="switch-note">⏳ Switching…</div>
    </div>

    <!-- Traffic Signal Control -->
    <div class="card">
      <h2>Adaptive Signal Timing — Webster's Method</h2>

      <!-- Formula display -->
      <div class="formula-box" id="formula-box">
        Waiting for detection data…
      </div>

      <!-- Cycle timeline bar -->
      <div class="cycle-bar" id="cycle-bar">
        <div class="cb-seg cb-green"  id="cb-ga" style="flex:30">A</div>
        <div class="cb-seg cb-yellow" style="flex:3">Y</div>
        <div class="cb-seg cb-allred" style="flex:1"></div>
        <div class="cb-seg cb-red"    id="cb-gb" style="flex:30">B</div>
        <div class="cb-seg cb-yellow" style="flex:3">Y</div>
        <div class="cb-seg cb-allred" style="flex:1"></div>
      </div>

      <div class="tl-grid">
        <!-- Direction A: Main Road -->
        <div class="tl-signal" id="sig-a">
          <div class="tl-name">Main Road</div>
          <div class="tl-housing">
            <div class="tl-bulb red"    id="a-red"></div>
            <div class="tl-bulb yellow" id="a-yellow"></div>
            <div class="tl-bulb green"  id="a-green"></div>
          </div>
          <div class="tl-countdown col-green" id="cd-a">—</div>
          <div class="tl-progress-wrap"><div class="tl-progress" id="prog-a" style="width:100%;background:#22c55e"></div></div>
          <div class="tl-suggest"  id="sug-a">— s green</div>
          <div class="tl-vehicles" id="veh-a">— vehicles</div>
          <div class="tl-pcu"      id="pcu-a">PCU: —</div>
          <div class="density-bar-wrap"><div class="density-bar" id="dbar-a" style="width:0%"></div></div>
          <div><span class="dlvl dlvl-LOW" id="dlvl-a">LOW</span></div>
          <div class="breakdown" id="bdwn-a"></div>
        </div>

        <!-- Direction B: Cross Road -->
        <div class="tl-signal" id="sig-b">
          <div class="tl-name">Cross Road</div>
          <div class="tl-housing">
            <div class="tl-bulb red"    id="b-red"></div>
            <div class="tl-bulb yellow" id="b-yellow"></div>
            <div class="tl-bulb green"  id="b-green"></div>
          </div>
          <div class="tl-countdown col-red" id="cd-b">—</div>
          <div class="tl-progress-wrap"><div class="tl-progress" id="prog-b" style="width:0%;background:#ef4444"></div></div>
          <div class="tl-suggest"  id="sug-b">— s green</div>
          <div class="tl-vehicles" id="veh-b">— vehicles</div>
          <div class="tl-pcu"      id="pcu-b">PCU: —</div>
          <div class="density-bar-wrap"><div class="density-bar" id="dbar-b" style="width:0%"></div></div>
          <div><span class="dlvl dlvl-LOW" id="dlvl-b">LOW</span></div>
          <div class="breakdown" id="bdwn-b"></div>
        </div>
      </div>

      <!-- Phase status -->
      <div class="reason-bar" id="reason-bar">Waiting for detection data…</div>
    </div>

  </div><!-- /right -->
</div>

<script>
  // ── Source switcher ────────────────────────────────────────────────
  let sources = [], currentSrc = '';

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

  // ── Traffic Signal Logic ────────────────────────────────────────────
  let phase      = 'a-green';
  let phaseEnd   = 0;
  let phaseTotal = 30;          // duration of current phase at the moment it started
  let sugA = 30, sugB = 30;
  let yellA = 3, yellB = 3;

  function setLight(dir, state) {
    ['red','yellow','green'].forEach(c =>
      document.getElementById(`${dir}-${c}`).classList.toggle('on', c === state)
    );
    document.getElementById(`sig-${dir}`).className = 'tl-signal ' + state + '-active';
  }

  function advancePhase() {
    if (Date.now() < phaseEnd) return;
    switch(phase) {
      case 'a-green':  phase = 'a-yellow'; phaseTotal = yellA; phaseEnd = Date.now() + yellA * 1000; break;
      case 'a-yellow': phase = 'b-green';  phaseTotal = sugB;  phaseEnd = Date.now() + sugB  * 1000; break;
      case 'b-green':  phase = 'b-yellow'; phaseTotal = yellB; phaseEnd = Date.now() + yellB * 1000; break;
      case 'b-yellow': phase = 'a-green';  phaseTotal = sugA;  phaseEnd = Date.now() + sugA  * 1000; break;
    }
  }

  function renderSignals() {
    const now  = Date.now();
    const rem  = Math.max(0, Math.ceil((phaseEnd - now) / 1000));
    const isA  = phase.startsWith('a');
    const isY  = phase.endsWith('yellow');
    const col  = isY ? '#eab308' : (isA ? '#22c55e' : '#ef4444');
    const colCls = isY ? 'col-yellow' : (isA ? 'col-green' : 'col-red');
    const pct  = (phaseTotal > 0 ? (rem / phaseTotal) * 100 : 0).toFixed(1) + '%';

    if      (phase === 'a-green')  { setLight('a','green');  setLight('b','red');    }
    else if (phase === 'a-yellow') { setLight('a','yellow'); setLight('b','red');    }
    else if (phase === 'b-green')  { setLight('a','red');    setLight('b','green');  }
    else                           { setLight('a','red');    setLight('b','yellow'); }

    const actDir = isA ? 'a' : 'b';
    const idlDir = isA ? 'b' : 'a';

    // Active direction: show countdown + filling progress bar
    const cdAct = document.getElementById('cd-' + actDir);
    cdAct.textContent = rem + 's';
    cdAct.className = 'tl-countdown ' + colCls;
    const progAct = document.getElementById('prog-' + actDir);
    progAct.style.width      = pct;
    progAct.style.background = col;

    // Idle direction: dash + empty bar
    const cdIdle = document.getElementById('cd-' + idlDir);
    cdIdle.textContent = '—';
    cdIdle.className = 'tl-countdown col-red';
    const progIdle = document.getElementById('prog-' + idlDir);
    progIdle.style.width      = '0%';
    progIdle.style.background = '#ef4444';

    const phaseLabels = {
      'a-green':  `<span class="phase-label">Main Road GREEN</span> — ${rem}s remaining. Cross Road holding on RED.`,
      'a-yellow': `<span class="phase-label">Main Road YELLOW</span> — clearing intersection (${rem}s). Cross Road next.`,
      'b-green':  `<span class="phase-label">Cross Road GREEN</span> — ${rem}s remaining. Main Road holding on RED.`,
      'b-yellow': `<span class="phase-label">Cross Road YELLOW</span> — clearing intersection (${rem}s). Main Road next.`,
    };
    document.getElementById('reason-bar').innerHTML = phaseLabels[phase];
  }

  function renderDensityBar(id, pcu, totalPcu) {
    const pct = totalPcu > 0 ? Math.min(100, (pcu / totalPcu) * 100) : 50;
    const el  = document.getElementById(id);
    el.style.width = pct.toFixed(1) + '%';
    el.style.background = pct > 65 ? '#ef4444' : pct > 35 ? '#eab308' : '#22c55e';
  }

  function renderDensityLevel(id, level) {
    const el = document.getElementById(id);
    el.textContent = level;
    el.className = `dlvl dlvl-${level}`;
  }

  function renderBreakdown(id, bdwn) {
    const el = document.getElementById(id);
    el.innerHTML = Object.entries(bdwn)
      .sort((a,b) => b[1]-a[1])
      .map(([cls, pcu]) => `<span class="chip">${cls} ${pcu.toFixed(1)}</span>`)
      .join('');
  }

  async function refreshTimings() {
    try {
      const d = await (await fetch('/api/traffic_timing')).json();
      const mr = d.main_road, cr = d.cross_road;

      sugA  = mr.suggested_green;
      sugB  = cr.suggested_green;
      yellA = mr.yellow || 3;
      yellB = cr.yellow || 3;

      // Signal cards
      document.getElementById('sug-a').textContent  = `${sugA}s green`;
      document.getElementById('sug-b').textContent  = `${sugB}s green`;
      document.getElementById('veh-a').textContent  = `${mr.vehicle_count} vehicles`;
      document.getElementById('veh-b').textContent  = `${cr.vehicle_count} vehicles`;
      document.getElementById('pcu-a').textContent  = `PCU: ${mr.pcu.toFixed(2)}  y=${mr.flow_ratio.toFixed(3)}`;
      document.getElementById('pcu-b').textContent  = `PCU: ${cr.pcu.toFixed(2)}  y=${cr.flow_ratio.toFixed(3)}`;

      renderDensityBar('dbar-a', mr.pcu, d.total_pcu);
      renderDensityBar('dbar-b', cr.pcu, d.total_pcu);
      renderDensityLevel('dlvl-a', mr.density_level);
      renderDensityLevel('dlvl-b', cr.density_level);
      renderBreakdown('bdwn-a', mr.breakdown || {});
      renderBreakdown('bdwn-b', cr.breakdown || {});

      // Cycle timeline bar proportions
      document.getElementById('cb-ga').style.flex = sugA;
      document.getElementById('cb-gb').style.flex = sugB;

      // Formula box
      document.getElementById('formula-box').innerHTML =
        `<span class="hl">PCU weights:</span> car=1.0 · truck/bus=2.5 · moto=0.5 · bicycle=0.3 · person=0.2\n` +
        `<span class="hl">Lane density:</span> Main=${mr.pcu.toFixed(2)} PCU  Cross=${cr.pcu.toFixed(2)} PCU\n` +
        `<span class="hl">Flow ratios:</span>  y_A=${mr.flow_ratio.toFixed(3)}  y_B=${cr.flow_ratio.toFixed(3)}  Y=${d.y_total.toFixed(3)}\n` +
        `<span class="hl">Webster:</span>      C = (1.5L+5)/(1−Y) = ${d.cycle_length}s  (L=${d.lost_time}s)\n` +
        `<span class="hl">Green split:</span>  Main=${sugA}s (${(mr.flow_ratio/d.y_total*100||50).toFixed(1)}%)  Cross=${sugB}s (${(cr.flow_ratio/d.y_total*100||50).toFixed(1)}%)`;

      // Caption
      document.getElementById('caption').textContent  = d.caption || '—';
      document.getElementById('inf-time').textContent = d.inference_ms
        ? `Inference: ${d.inference_ms.toFixed(1)} ms` : '';

      document.getElementById('status').textContent = '● LIVE';
    } catch(e) {
      document.getElementById('status').textContent = '● OFFLINE';
    }
  }

  phaseEnd   = Date.now() + sugA * 1000;
  phaseTotal = sugA;
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
    See signal_controller.py for the full mathematical model.
    """
    recent = db.get_recent_detections(limit=60)
    result = _frame_buffer.get() if _frame_buffer else None
    inf_ms = result.inference_ms if result else None
    caption = recent[0]["scene_caption"] if recent else "Awaiting detections…"

    cycle = _signal_ctrl.compute(recent)
    mr, cr = cycle.main_road, cycle.cross_road

    return jsonify({
        "main_road": {
            "vehicle_count":  mr.vehicle_count,
            "pcu":            mr.pcu,
            "flow_ratio":     mr.flow_ratio,
            "suggested_green": mr.green,
            "yellow":         mr.yellow,
            "density_level":  mr.density_level,
            "breakdown":      mr.breakdown,
        },
        "cross_road": {
            "vehicle_count":  cr.vehicle_count,
            "pcu":            cr.pcu,
            "flow_ratio":     cr.flow_ratio,
            "suggested_green": cr.green,
            "yellow":         cr.yellow,
            "density_level":  cr.density_level,
            "breakdown":      cr.breakdown,
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