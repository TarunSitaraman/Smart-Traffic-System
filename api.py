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
_signal_timing_provider = None
_traffic_snapshot_provider = None


def set_frame_buffer(buf: FrameBuffer) -> None:
    global _frame_buffer
    _frame_buffer = buf


def set_detector(det: "YOLOv8Detector") -> None:
    global _detector
    _detector = det


def set_signal_state_provider(provider) -> None:
    global _signal_state_provider
    _signal_state_provider = provider


def set_signal_timing_provider(provider) -> None:
    global _signal_timing_provider
    _signal_timing_provider = provider


def set_traffic_snapshot_provider(provider) -> None:
    global _traffic_snapshot_provider
    _traffic_snapshot_provider = provider


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
  <title>Smart Traffic System — Adaptive Demo</title>
  <style>
    :root {
      --bg: #071217;
      --panel: #0f1e25;
      --panel-alt: #142a33;
      --line: #274754;
      --text: #e9f7fa;
      --muted: #8daeb2;
      --green: #67f28b;
      --amber: #ffc857;
      --red: #ff6b57;
      --cyan: #5ed9ff;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      min-height: 100vh;
      font-family: "Bahnschrift", "Trebuchet MS", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(94, 217, 255, 0.12), transparent 26%),
        radial-gradient(circle at top right, rgba(255, 200, 87, 0.08), transparent 22%),
        linear-gradient(180deg, #081318 0%, #071217 100%);
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 18px 24px;
      border-bottom: 1px solid rgba(94, 217, 255, 0.12);
      background: rgba(7, 18, 23, 0.78);
      backdrop-filter: blur(14px);
    }
    header h1 {
      font-size: 1.16rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .header-meta {
      display: flex;
      align-items: center;
      gap: 14px;
      color: var(--muted);
      font-family: "Consolas", "Courier New", monospace;
      font-size: 0.92rem;
    }
    .badge {
      padding: 6px 12px;
      border-radius: 999px;
      background: rgba(102, 242, 139, 0.16);
      border: 1px solid rgba(102, 242, 139, 0.24);
      color: var(--green);
      font-family: "Bahnschrift", sans-serif;
      font-size: 0.74rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1.24fr) minmax(340px, 0.96fr);
      gap: 18px;
      padding: 18px;
    }
    .stack { display: flex; flex-direction: column; gap: 18px; }
    .card {
      background: linear-gradient(180deg, rgba(19, 39, 48, 0.94), rgba(11, 24, 30, 0.96));
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 18px;
      box-shadow: 0 16px 34px rgba(0, 0, 0, 0.24);
    }
    .section-title {
      font-size: 0.78rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 14px;
    }
    .hero { display: grid; gap: 16px; }
    .hero-head {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
    }
    .hero-copy h2 {
      font-size: 1.52rem;
      line-height: 1.08;
      margin-bottom: 6px;
    }
    .hero-copy p {
      color: var(--muted);
      line-height: 1.45;
      max-width: 46rem;
    }
    .demo-cap {
      min-width: 124px;
      padding: 14px;
      border-radius: 16px;
      text-align: right;
      background: rgba(255, 200, 87, 0.08);
      border: 1px solid rgba(255, 200, 87, 0.22);
    }
    .demo-cap .label {
      color: var(--muted);
      font-size: 0.72rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .demo-cap .value {
      margin-top: 4px;
      color: var(--amber);
      font-size: 1.8rem;
      font-weight: 700;
    }
    .insight-banner {
      min-height: 56px;
      padding: 14px 16px;
      border-radius: 16px;
      border: 1px solid rgba(94, 217, 255, 0.24);
      background: linear-gradient(90deg, rgba(94, 217, 255, 0.12), rgba(102, 242, 139, 0.08));
      line-height: 1.45;
    }
    .metric-ribbon {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }
    .metric-box, .control-tile {
      padding: 12px 14px;
      border-radius: 15px;
      background: rgba(7, 18, 23, 0.7);
      border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .metric-box span, .control-tile span {
      display: block;
      font-size: 0.72rem;
      color: var(--muted);
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 6px;
    }
    .metric-box strong { font-size: 1.4rem; }
    .control-tile strong { font-size: 1.3rem; }
    .stream-shell {
      position: relative;
      overflow: hidden;
      border-radius: 18px;
      border: 1px solid rgba(94, 217, 255, 0.16);
      background: #081015;
    }
    .stream-shell img {
      width: 100%;
      display: block;
      aspect-ratio: 16 / 9;
      object-fit: cover;
    }
    .stream-note {
      position: absolute;
      right: 14px;
      bottom: 14px;
      max-width: 260px;
      padding: 8px 10px;
      border-radius: 12px;
      background: rgba(7, 18, 23, 0.78);
      border: 1px solid rgba(255, 255, 255, 0.08);
      color: var(--muted);
      font-size: 0.72rem;
      line-height: 1.35;
    }
    .caption-box {
      margin-top: 14px;
      min-height: 62px;
      padding: 14px 16px;
      border-radius: 14px;
      border-left: 4px solid var(--cyan);
      background: rgba(7, 18, 23, 0.72);
      line-height: 1.5;
    }
    .stream-meta {
      display: flex;
      justify-content: space-between;
      margin-top: 10px;
      color: var(--muted);
      font-size: 0.8rem;
      font-family: "Consolas", "Courier New", monospace;
    }
    .source-bar {
      display: flex;
      gap: 10px;
      align-items: center;
      margin-top: 14px;
      flex-wrap: wrap;
    }
    .source-bar label {
      color: var(--muted);
      font-size: 0.76rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .source-select {
      flex: 1;
      min-width: 220px;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      background: rgba(7, 18, 23, 0.88);
      color: var(--text);
    }
    .source-status {
      color: var(--muted);
      font-size: 0.76rem;
    }
    .phase-banner {
      margin-bottom: 14px;
      padding: 14px 16px;
      border-radius: 16px;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .phase-ns { color: var(--green); background: rgba(102, 242, 139, 0.09); }
    .phase-ew { color: var(--cyan); background: rgba(94, 217, 255, 0.09); }
    .control-grid, .queue-board {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .control-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .legend {
      margin-bottom: 14px;
      color: var(--muted);
      font-size: 0.82rem;
      line-height: 1.45;
    }
    .lane-card {
      padding: 14px;
      border-radius: 18px;
      background: linear-gradient(180deg, rgba(10, 19, 25, 0.95), rgba(8, 15, 20, 0.94));
      border: 1px solid rgba(255, 255, 255, 0.07);
      transition: transform 0.25s ease, border-color 0.25s ease;
    }
    .lane-card.active {
      border-color: rgba(102, 242, 139, 0.32);
      transform: translateY(-2px);
    }
    .lane-card.queued { border-color: rgba(255, 200, 87, 0.28); }
    .lane-head, .lane-row, .pressure-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
    }
    .lane-head { margin-bottom: 10px; }
    .lane-head strong {
      font-size: 1rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }
    .state-pill {
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.7rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      border: 1px solid transparent;
    }
    .state-pill.green {
      color: var(--green);
      background: rgba(102, 242, 139, 0.12);
      border-color: rgba(102, 242, 139, 0.24);
    }
    .state-pill.yellow {
      color: var(--amber);
      background: rgba(255, 200, 87, 0.12);
      border-color: rgba(255, 200, 87, 0.24);
    }
    .state-pill.red {
      color: var(--red);
      background: rgba(255, 107, 87, 0.12);
      border-color: rgba(255, 107, 87, 0.24);
    }
    .lane-timer {
      font-size: 1.78rem;
      font-weight: 700;
      font-family: "Consolas", "Courier New", monospace;
    }
    .timer-cluster {
      display: flex;
      flex-direction: column;
      gap: 3px;
    }
    .timer-label, .lane-plan-label {
      color: var(--muted);
      font-size: 0.7rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .lane-green {
      text-align: right;
      color: var(--muted);
      font-size: 0.76rem;
    }
    .lane-green strong {
      display: block;
      margin-top: 3px;
      color: var(--text);
      font-size: 1.1rem;
    }
    .pressure-row {
      margin: 10px 0 8px;
      color: var(--muted);
      font-size: 0.78rem;
    }
    .queue-rail {
      height: 10px;
      margin-bottom: 10px;
      border-radius: 999px;
      overflow: hidden;
      background: rgba(255, 255, 255, 0.06);
    }
    .queue-fill {
      height: 100%;
      width: 0%;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--green), var(--amber), var(--red));
      transition: width 0.5s ease;
    }
    .queue-chips {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      min-height: 28px;
      margin-bottom: 10px;
    }
    .chip {
      min-width: 18px;
      padding: 4px 6px;
      border-radius: 8px;
      background: rgba(94, 217, 255, 0.12);
      color: #c5f6ff;
      font-size: 0.72rem;
      text-align: center;
      font-family: "Consolas", "Courier New", monospace;
    }
    .chip.more {
      background: rgba(255, 200, 87, 0.14);
      color: var(--amber);
    }
    .chip.live {
      background: rgba(94, 217, 255, 0.16);
      color: #d7f8ff;
    }
    .chip.hidden {
      background: rgba(255, 107, 87, 0.12);
      color: #ffc8bf;
    }
    .lane-stats {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      color: var(--muted);
      font-size: 0.74rem;
    }
    .lane-stats span {
      padding: 4px 8px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.04);
    }
    .formula-box, .event-item {
      padding: 12px 14px;
      border-radius: 14px;
      background: rgba(7, 18, 23, 0.72);
      border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .formula-box {
      margin-top: 14px;
      color: var(--muted);
      font-size: 0.76rem;
      line-height: 1.5;
      font-family: "Consolas", "Courier New", monospace;
    }
    .event-list { display: grid; gap: 10px; }
    .event-time {
      margin-bottom: 5px;
      color: var(--muted);
      font-size: 0.72rem;
      font-family: "Consolas", "Courier New", monospace;
    }
    .event-copy { line-height: 1.45; }
    @media (max-width: 1180px) {
      .layout { grid-template-columns: 1fr; }
    }
    @media (max-width: 760px) {
      .metric-ribbon, .control-grid, .queue-board { grid-template-columns: 1fr; }
      .hero-head { flex-direction: column; }
      .demo-cap { width: 100%; text-align: left; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Smart Traffic Intelligence Platform</h1>
    <div class="header-meta">
      <span id="clock"></span>
      <span class="badge" id="status">System Online</span>
    </div>
  </header>
  <main class="layout">
    <section class="stack">
      <div class="card">
        <div class="hero">
          <div class="hero-head">
            <div class="hero-copy">
              <h2>Adaptive demo view for traffic build-up and smart signal response</h2>
              <p>The camera only shows the stop-line slice. The backlog board combines visible vehicles, density, and queue pressure so you can still see why the controller changes timings.</p>
            </div>
            <div class="demo-cap">
              <div class="label">Demo Green Cap</div>
              <div class="value" id="demo-cap">20s</div>
            </div>
          </div>
          <div class="insight-banner" id="insight-banner">Waiting for adaptive signal insight...</div>
          <div class="metric-ribbon">
            <div class="metric-box"><span>Cycle Length</span><strong id="cycle-length">--s</strong></div>
            <div class="metric-box"><span>Current Phase</span><strong id="active-axis">--</strong></div>
            <div class="metric-box"><span>Peak Pressure</span><strong id="peak-pressure">0%</strong></div>
            <div class="metric-box"><span>Next Switch</span><strong id="next-switch">--s</strong></div>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="section-title">Live Feed</div>
        <div class="stream-shell">
          <img src="/live" alt="Live Intelligence Feed"/>
          <div class="stream-note">Camera shows the intersection throat only. Queue rails estimate upstream traffic outside the frame.</div>
        </div>
        <div class="source-bar">
          <label for="source-select">Video Source</label>
          <select id="source-select" class="source-select"></select>
          <div id="source-status" class="source-status">Using local traffic footage for real YOLOv8 detection.</div>
        </div>
        <div class="caption-box" id="caption">Initializing intelligence engine...</div>
        <div class="stream-meta">
          <span id="inf-time">Latency: --</span>
          <span id="fps-counter">0 FPS</span>
        </div>
      </div>

      <div class="card">
        <div class="section-title">Adaptive Event Log</div>
        <div class="event-list" id="change-log">
          <div class="event-item">
            <div class="event-time">--:--:--</div>
            <div class="event-copy">Waiting for adaptive signal updates...</div>
          </div>
        </div>
      </div>
    </section>

    <aside class="stack">
      <div class="card">
        <div class="section-title">Signal Control</div>
        <div id="phase-label" class="phase-banner phase-ns">Synchronizing live phase...</div>
        <div class="control-grid">
          <div class="control-tile"><span>Priority Axis</span><strong id="priority-axis">--</strong></div>
          <div class="control-tile"><span>North/South Green</span><strong id="ns-green">--s</strong></div>
          <div class="control-tile"><span>East/West Green</span><strong id="ew-green">--s</strong></div>
        </div>
        <div class="formula-box" id="formula-box">Adaptive controller warming up...</div>
      </div>

      <div class="card">
        <div class="section-title">Approach Backlog Board</div>
        <div class="legend">Each card shows live state, smart green time, visible vehicles, and an estimated backlog ribbon. This makes the demo readable even when the queue extends beyond the camera frame.</div>
        <div class="queue-board">
          <div class="lane-card" id="lane-n">
            <div class="lane-head"><strong>North</strong><span class="state-pill red" id="state-n">Red</span></div>
            <div class="lane-row"><div class="timer-cluster"><div class="timer-label" id="timer-label-n">Wait Time</div><div class="lane-timer" id="cd-n">--s</div></div><div class="lane-green"><div class="lane-plan-label" id="green-label-n">Next Green</div><strong id="green-n">--s</strong></div></div>
            <div class="pressure-row"><span id="reason-n">Balanced approach</span><span id="backlog-n">0 est. backlog</span></div>
            <div class="queue-rail"><div class="queue-fill" id="rail-n"></div></div>
            <div class="queue-chips" id="chips-n"></div>
            <div class="lane-stats"><span id="veh-n">0 visible</span><span id="density-n">Density 0.0</span><span id="queue-n">Queue 0.00</span><span id="pcu-n">PCU 0.0</span></div>
          </div>

          <div class="lane-card" id="lane-e">
            <div class="lane-head"><strong>East</strong><span class="state-pill red" id="state-e">Red</span></div>
            <div class="lane-row"><div class="timer-cluster"><div class="timer-label" id="timer-label-e">Wait Time</div><div class="lane-timer" id="cd-e">--s</div></div><div class="lane-green"><div class="lane-plan-label" id="green-label-e">Next Green</div><strong id="green-e">--s</strong></div></div>
            <div class="pressure-row"><span id="reason-e">Balanced approach</span><span id="backlog-e">0 est. backlog</span></div>
            <div class="queue-rail"><div class="queue-fill" id="rail-e"></div></div>
            <div class="queue-chips" id="chips-e"></div>
            <div class="lane-stats"><span id="veh-e">0 visible</span><span id="density-e">Density 0.0</span><span id="queue-e">Queue 0.00</span><span id="pcu-e">PCU 0.0</span></div>
          </div>

          <div class="lane-card" id="lane-s">
            <div class="lane-head"><strong>South</strong><span class="state-pill red" id="state-s">Red</span></div>
            <div class="lane-row"><div class="timer-cluster"><div class="timer-label" id="timer-label-s">Wait Time</div><div class="lane-timer" id="cd-s">--s</div></div><div class="lane-green"><div class="lane-plan-label" id="green-label-s">Next Green</div><strong id="green-s">--s</strong></div></div>
            <div class="pressure-row"><span id="reason-s">Balanced approach</span><span id="backlog-s">0 est. backlog</span></div>
            <div class="queue-rail"><div class="queue-fill" id="rail-s"></div></div>
            <div class="queue-chips" id="chips-s"></div>
            <div class="lane-stats"><span id="veh-s">0 visible</span><span id="density-s">Density 0.0</span><span id="queue-s">Queue 0.00</span><span id="pcu-s">PCU 0.0</span></div>
          </div>

          <div class="lane-card" id="lane-w">
            <div class="lane-head"><strong>West</strong><span class="state-pill red" id="state-w">Red</span></div>
            <div class="lane-row"><div class="timer-cluster"><div class="timer-label" id="timer-label-w">Wait Time</div><div class="lane-timer" id="cd-w">--s</div></div><div class="lane-green"><div class="lane-plan-label" id="green-label-w">Next Green</div><strong id="green-w">--s</strong></div></div>
            <div class="pressure-row"><span id="reason-w">Balanced approach</span><span id="backlog-w">0 est. backlog</span></div>
            <div class="queue-rail"><div class="queue-fill" id="rail-w"></div></div>
            <div class="queue-chips" id="chips-w"></div>
            <div class="lane-stats"><span id="veh-w">0 visible</span><span id="density-w">Density 0.0</span><span id="queue-w">Queue 0.00</span><span id="pcu-w">PCU 0.0</span></div>
          </div>
        </div>
      </div>
    </aside>
  </main>

  <script>
    const laneNames = { north: "North", south: "South", east: "East", west: "West" };
    const shortMap = { north: "n", south: "s", east: "e", west: "w" };
    const previousGreens = {};
    let previousPriority = null;
    let eventLog = [];
    let sourcePathsLoaded = false;

    function updateClock() {
      document.getElementById("clock").textContent = new Date().toLocaleTimeString("en-GB", { hour12: false });
    }

    function cap(value, minValue, maxValue) {
      return Math.max(minValue, Math.min(maxValue, value));
    }

    function backlogUnits(data) {
      if (typeof data.estimated_backlog === "number") {
        return data.estimated_backlog;
      }
      return Math.max(
        data.vehicle_count || 0,
        Math.round((data.demand_score || 0) * 9 + (data.queue_length || 0) * 8)
      );
    }

    function laneReason(data) {
      if ((data.demand_score || 0) >= 0.8) return "Heavy build-up detected";
      if ((data.queue_length || 0) >= 0.45) return "Queue rising upstream";
      if ((data.density || 0) >= 18) return "Dense approach flow";
      if ((data.vehicle_count || 0) >= 4) return "Visible pack at stop line";
      return "Balanced approach";
    }

    function setStatePill(el, state) {
      el.textContent = state;
      el.className = "state-pill " + state.toLowerCase();
    }

    function renderChips(targetId, totalCount, visibleCount) {
      const el = document.getElementById(targetId);
      const live = Math.min(visibleCount || 0, 5);
      const hidden = Math.max(0, totalCount - live);
      let html = "";
      for (let i = 0; i < live; i += 1) {
        html += '<span class="chip live">LIVE</span>';
      }
      if (hidden > 0) {
        html += '<span class="chip hidden">QUEUE</span>';
        html += '<span class="chip more">+' + hidden + "</span>";
      }
      el.innerHTML = html || '<span class="chip">CLEAR</span>';
    }

    function addEvent(message) {
      const stamp = new Date().toLocaleTimeString("en-GB", { hour12: false });
      eventLog.unshift({ stamp, message });
      eventLog = eventLog.slice(0, 6);
      const host = document.getElementById("change-log");
      host.innerHTML = eventLog.map((item) =>
        '<div class="event-item"><div class="event-time">' + item.stamp + '</div><div class="event-copy">' + item.message + "</div></div>"
      ).join("");
    }

    async function loadSources() {
      if (sourcePathsLoaded) return;
      try {
        const [sourcesResp, currentResp] = await Promise.all([
          fetch("/api/sources"),
          fetch("/api/current_source")
        ]);
        const sources = await sourcesResp.json();
        const current = await currentResp.json();
        const select = document.getElementById("source-select");
        select.innerHTML = sources.map((item) =>
          '<option value="' + item.path.replace(/"/g, "&quot;") + '">' + item.label + "</option>"
        ).join("");
        select.value = current.path;
        document.getElementById("source-status").textContent = "Current source: " + current.label;
        select.onchange = async (event) => {
          const path = event.target.value;
          document.getElementById("source-status").textContent = "Switching source...";
          const res = await fetch("/api/source", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ path })
          });
          const payload = await res.json();
          document.getElementById("source-status").textContent = payload.path ? "Current source: " + payload.path.split(/[\\\\/]/).pop().replace(/\\.[^.]+$/, "") : "Source switch requested";
          addEvent("Source switched to " + (payload.path ? payload.path.split(/[\\\\/]/).pop() : "selected clip") + ".");
        };
        sourcePathsLoaded = true;
      } catch (e) {
        document.getElementById("source-status").textContent = "Source list unavailable.";
      }
    }

    function updateLaneCard(dir, data) {
      const sd = shortMap[dir];
      const state = (data.state || "RED").toUpperCase();
      const card = document.getElementById("lane-" + sd);
      const pill = document.getElementById("state-" + sd);
      const backlog = backlogUnits(data);
      const pressure = cap(data.demand_score || 0, 0, 1);
      const timerEl = document.getElementById("cd-" + sd);
      const timerLabel = document.getElementById("timer-label-" + sd);
      const greenLabel = document.getElementById("green-label-" + sd);

      card.classList.toggle("active", state === "GREEN");
      card.classList.toggle("queued", state !== "GREEN" && pressure > 0.45);
      setStatePill(pill, state);
      timerEl.textContent = (data.timer || 0) + "s";
      timerEl.style.color = state === "GREEN" ? "var(--green)" : (state === "YELLOW" ? "var(--amber)" : "var(--red)");
      timerLabel.textContent = state === "GREEN" ? "Green Ends In" : (state === "YELLOW" ? "Clearance Ends In" : "Wait Time");
      greenLabel.textContent = state === "GREEN" ? "Current Green" : "Next Green";

      document.getElementById("green-" + sd).textContent = (data.suggested_green || 0) + "s";
      document.getElementById("reason-" + sd).textContent = laneReason(data);
      document.getElementById("backlog-" + sd).textContent = backlog + " est. backlog";
      document.getElementById("veh-" + sd).textContent = (data.vehicle_count || 0) + " visible";
      document.getElementById("density-" + sd).textContent = "Density " + (data.density || 0).toFixed(1);
      document.getElementById("queue-" + sd).textContent = "Queue " + (data.queue_length || 0).toFixed(2);
      document.getElementById("pcu-" + sd).textContent = "PCU " + (data.pcu || 0).toFixed(1);
      document.getElementById("rail-" + sd).style.width = cap(backlog / 10, 0.08, 1) * 100 + "%";
      renderChips("chips-" + sd, backlog, data.vehicle_count || 0);

      const prev = previousGreens[dir];
      if (typeof prev === "number" && prev !== data.suggested_green) {
        addEvent(laneNames[dir] + " smart green adjusted from " + prev + "s to " + data.suggested_green + "s as traffic pressure changed.");
      }
      previousGreens[dir] = data.suggested_green;
      return pressure;
    }

    async function refresh() {
      try {
        const d = await (await fetch("/api/traffic_timing")).json();
        const dirs = ["north", "south", "east", "west"];
        let nsActive = false;
        let peakPressure = 0;

        dirs.forEach((dir) => {
          const pressure = updateLaneCard(dir, d[dir]);
          peakPressure = Math.max(peakPressure, pressure);
          if ((dir === "north" || dir === "south") && d[dir].state === "GREEN") {
            nsActive = true;
          }
        });

        const activeAxis = nsActive ? "North / South" : "East / West";
        const priorityAxis = d.dominant_phase === "NS" ? "North / South" : "East / West";
        const phaseEl = document.getElementById("phase-label");
        phaseEl.textContent = activeAxis.toUpperCase() + " MOVEMENT ACTIVE";
        phaseEl.className = "phase-banner " + (nsActive ? "phase-ns" : "phase-ew");

        if (previousPriority && previousPriority !== priorityAxis) {
          addEvent(priorityAxis + " pressure has overtaken the other axis, so green priority shifted.");
        }
        previousPriority = priorityAxis;

        document.getElementById("active-axis").textContent = activeAxis;
        document.getElementById("priority-axis").textContent = priorityAxis;
        document.getElementById("cycle-length").textContent = (d.cycle_length || 0) + "s";
        document.getElementById("peak-pressure").textContent = Math.round(peakPressure * 100) + "%";
        document.getElementById("next-switch").textContent = (d.remaining_seconds || 0) + "s";
        document.getElementById("ns-green").textContent = (d.north.suggested_green || 0) + "s";
        document.getElementById("ew-green").textContent = (d.east.suggested_green || 0) + "s";
        document.getElementById("demo-cap").textContent = (d.demo_cap || 20) + "s";
        document.getElementById("insight-banner").textContent = d.adaptive_insight || "Adaptive timing active.";
        document.getElementById("formula-box").textContent = d.formula_note || "Adaptive controller active.";
        document.getElementById("caption").textContent = d.caption || "";

        if (d.inference_ms) {
          document.getElementById("inf-time").textContent = "Latency: " + d.inference_ms.toFixed(1) + "ms";
          document.getElementById("fps-counter").textContent = (1000 / d.inference_ms).toFixed(1) + " FPS";
        }
        document.getElementById("status").textContent = "System Online";
      } catch (e) {
        document.getElementById("status").textContent = "Offline";
      }
    }

    updateClock();
    loadSources();
    setInterval(updateClock, 1000);
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
    from signal_controller import MAX_GREEN, PCU_WEIGHTS

    recent = db.get_recent_detections(limit=60)
    result = _frame_buffer.get() if _frame_buffer else None
    inf_ms = result.inference_ms if result else None
    caption = recent[0]["scene_caption"] if recent else "Awaiting detections…"

    snapshot = _traffic_snapshot_provider() if _traffic_snapshot_provider else None
    cycle = snapshot.get("cycle_result") if snapshot else None

    if cycle is None:
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

    rem_sec = 0
    timers = {"north": 0, "south": 0, "east": 0, "west": 0}
    if _signal_timing_provider:
        timing = _signal_timing_provider() or {}
        rem_sec = timing.get("remaining_seconds", 0)
        timers = timing.get("timers", timers)

    dominant_phase = "NS" if cycle.north.green >= cycle.east.green else "EW"
    adaptive_insight = _build_adaptive_insight(cycle, dominant_phase)

    return jsonify(
        {
            "north": {
                "vehicle_count": cycle.north.vehicle_count,
                "pcu": cycle.north.pcu,
                "density": cycle.north.density,
                "queue_length": cycle.north.queue_length,
                "demand_score": cycle.north.demand_score,
                "estimated_backlog": _estimate_backlog(cycle.north),
                "flow_ratio": cycle.north.flow_ratio,
                "suggested_green": cycle.north.green,
                "state": live_states.get("north", "RED") if live_states else "RED",
                "timer": timers.get("north", 0),
            },
            "south": {
                "vehicle_count": cycle.south.vehicle_count,
                "pcu": cycle.south.pcu,
                "density": cycle.south.density,
                "queue_length": cycle.south.queue_length,
                "demand_score": cycle.south.demand_score,
                "estimated_backlog": _estimate_backlog(cycle.south),
                "flow_ratio": cycle.south.flow_ratio,
                "suggested_green": cycle.south.green,
                "state": live_states.get("south", "RED") if live_states else "RED",
                "timer": timers.get("south", 0),
            },
            "east": {
                "vehicle_count": cycle.east.vehicle_count,
                "pcu": cycle.east.pcu,
                "density": cycle.east.density,
                "queue_length": cycle.east.queue_length,
                "demand_score": cycle.east.demand_score,
                "estimated_backlog": _estimate_backlog(cycle.east),
                "flow_ratio": cycle.east.flow_ratio,
                "suggested_green": cycle.east.green,
                "state": live_states.get("east", "RED") if live_states else "RED",
                "timer": timers.get("east", 0),
            },
            "west": {
                "vehicle_count": cycle.west.vehicle_count,
                "pcu": cycle.west.pcu,
                "density": cycle.west.density,
                "queue_length": cycle.west.queue_length,
                "demand_score": cycle.west.demand_score,
                "estimated_backlog": _estimate_backlog(cycle.west),
                "flow_ratio": cycle.west.flow_ratio,
                "suggested_green": cycle.west.green,
                "state": live_states.get("west", "RED") if live_states else "RED",
                "timer": timers.get("west", 0),
            },
            "cycle_length": cycle.cycle_length,
            "lost_time": cycle.lost_time,
            "y_total": cycle.y_total,
            "remaining_seconds": rem_sec,
            "formula_note": cycle.formula_note,
            "dominant_phase": dominant_phase,
            "demo_cap": MAX_GREEN,
            "adaptive_insight": adaptive_insight,
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


def _estimate_backlog(direction) -> int:
    if (
        direction.vehicle_count <= 1
        and direction.queue_length < 0.08
        and direction.demand_score < 0.25
    ):
        return direction.vehicle_count

    extra_backlog = max(
        0,
        round((direction.demand_score - 0.3) * 6 + max(0.0, direction.queue_length - 0.08) * 10),
    )
    return max(
        direction.vehicle_count,
        min(12, direction.vehicle_count + extra_backlog),
    )


def _build_adaptive_insight(cycle, dominant_phase: str) -> str:
    if dominant_phase == "NS":
        lead_name = "North / South"
        lead_green = cycle.north.green
        trailing_green = cycle.east.green
        lead_pressure = max(cycle.north.demand_score, cycle.south.demand_score)
    else:
        lead_name = "East / West"
        lead_green = cycle.east.green
        trailing_green = cycle.north.green
        lead_pressure = max(cycle.east.demand_score, cycle.west.demand_score)

    if lead_pressure >= 0.8:
        return (
            f"Traffic is piling up on {lead_name}, so the controller pushes green to "
            f"{lead_green}s while keeping the opposite axis at {trailing_green}s."
        )
    if lead_pressure >= 0.55:
        return (
            f"Moderate build-up on {lead_name} is being cleared with a longer "
            f"{lead_green}s green and a shorter opposing phase."
        )
    return (
        f"Approaches are balanced, so the controller keeps both directions close to "
        f"their demo baseline while still adapting in real time."
    )


def _safe_int(param_name: str, default: int) -> int:
    from flask import request

    try:
        return int(request.args.get(param_name, default))
    except (TypeError, ValueError):
        return default
