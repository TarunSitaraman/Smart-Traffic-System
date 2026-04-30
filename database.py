# =============================================================================
# database.py — SQLite persistence layer for detections and alerts
# =============================================================================

import sqlite3
import threading
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import config

logger = logging.getLogger(__name__)

# Thread-local SQLite connections (SQLite is not thread-safe with shared connections)
_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Return a per-thread SQLite connection, creating it if needed."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row   # dict-like rows
        _local.conn.execute("PRAGMA journal_mode=WAL")   # better concurrent reads
        _local.conn.execute("PRAGMA synchronous=NORMAL")
    return _local.conn


def init_db() -> None:
    """Create tables if they don't exist."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS detections (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT    NOT NULL,
            frame_id      INTEGER NOT NULL,
            class_name    TEXT    NOT NULL,
            confidence    REAL    NOT NULL,
            x1            REAL    NOT NULL,
            y1            REAL    NOT NULL,
            x2            REAL    NOT NULL,
            y2            REAL    NOT NULL,
            scene_caption TEXT,
            inference_ms  REAL
        );

        CREATE TABLE IF NOT EXISTS alerts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT  NOT NULL,
            class_name  TEXT  NOT NULL,
            severity    TEXT  NOT NULL,
            confidence  REAL  NOT NULL,
            message     TEXT  NOT NULL,
            frame_id    INTEGER
        );

        CREATE TABLE IF NOT EXISTS accidents (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id        INTEGER UNIQUE NOT NULL,
            timestamp       TEXT    NOT NULL,
            lane            TEXT    NOT NULL,
            track_id        INTEGER,
            secondary_track_id INTEGER,
            accident_type   TEXT    NOT NULL,
            confidence      REAL    NOT NULL,
            score_breakdown TEXT
        );

        CREATE TABLE IF NOT EXISTS signal_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            lane        TEXT    NOT NULL,
            state       TEXT    NOT NULL,
            duration_sec REAL
        );

        CREATE TABLE IF NOT EXISTS lane_metrics (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp      TEXT    NOT NULL,
            lane           TEXT    NOT NULL,
            vehicle_count  INTEGER NOT NULL,
            queue_length   REAL,
            density        REAL
        );

        CREATE INDEX IF NOT EXISTS idx_detections_ts    ON detections(timestamp);
        CREATE INDEX IF NOT EXISTS idx_detections_class ON detections(class_name);
        CREATE INDEX IF NOT EXISTS idx_alerts_ts        ON alerts(timestamp);
        CREATE INDEX IF NOT EXISTS idx_alerts_severity  ON alerts(severity);
        CREATE INDEX IF NOT EXISTS idx_accidents_ts     ON accidents(timestamp);
        CREATE INDEX IF NOT EXISTS idx_accidents_lane   ON accidents(lane);
        CREATE INDEX IF NOT EXISTS idx_signals_ts       ON signal_events(timestamp);
        CREATE INDEX IF NOT EXISTS idx_signals_lane     ON signal_events(lane);
        CREATE INDEX IF NOT EXISTS idx_lanes_ts         ON lane_metrics(timestamp);
        CREATE INDEX IF NOT EXISTS idx_lanes_lane       ON lane_metrics(lane);
    """)
    conn.commit()
    logger.info("Database initialised at %s", config.DB_PATH)


def insert_detections(
    frame_id: int,
    detections: List[Dict[str, Any]],
    scene_caption: str,
    inference_ms: float,
) -> None:
    """
    Bulk-insert all detections from a single frame.

    Each detection dict must have keys:
        class_name, confidence, x1, y1, x2, y2
    """
    if not detections:
        return

    ts = datetime.utcnow().isoformat(timespec="milliseconds")
    rows = [
        (
            ts,
            frame_id,
            d["class_name"],
            round(d["confidence"], 4),
            d["x1"], d["y1"], d["x2"], d["y2"],
            scene_caption,
            round(inference_ms, 2),
        )
        for d in detections
    ]

    conn = _get_conn()
    conn.executemany(
        """INSERT INTO detections
           (timestamp, frame_id, class_name, confidence,
            x1, y1, x2, y2, scene_caption, inference_ms)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    conn.commit()


def insert_alert(
    class_name: str,
    severity: str,
    confidence: float,
    message: str,
    frame_id: int,
) -> None:
    """Persist a single alert record."""
    ts = datetime.utcnow().isoformat(timespec="milliseconds")
    conn = _get_conn()
    conn.execute(
        """INSERT INTO alerts (timestamp, class_name, severity, confidence, message, frame_id)
           VALUES (?,?,?,?,?,?)""",
        (ts, class_name, severity, round(confidence, 4), message, frame_id),
    )
    conn.commit()


def get_recent_detections(limit: int = 100) -> List[Dict]:
    """Return the most recent detection rows as dicts."""
    conn = _get_conn()
    cur = conn.execute(
        "SELECT * FROM detections ORDER BY id DESC LIMIT ?", (limit,)
    )
    return [dict(row) for row in cur.fetchall()]


def get_recent_alerts(limit: int = 50) -> List[Dict]:
    """Return the most recent alert rows as dicts."""
    conn = _get_conn()
    cur = conn.execute(
        "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)
    )
    return [dict(row) for row in cur.fetchall()]


def get_detection_stats() -> Dict[str, Any]:
    """
    Return aggregate statistics:
      - total_detections (all time)
      - detections_last_hour
      - total_alerts
      - alerts_last_hour
      - class_counts: {class_name: count} for all time
      - top_alert_classes: [(class, count), ...] top 5
    """
    conn = _get_conn()
    one_hour_ago = (datetime.utcnow() - timedelta(hours=1)).isoformat()

    total_det   = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
    det_1h      = conn.execute(
        "SELECT COUNT(*) FROM detections WHERE timestamp >= ?", (one_hour_ago,)
    ).fetchone()[0]
    total_alerts = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
    alerts_1h    = conn.execute(
        "SELECT COUNT(*) FROM alerts WHERE timestamp >= ?", (one_hour_ago,)
    ).fetchone()[0]

    class_rows = conn.execute(
        "SELECT class_name, COUNT(*) as cnt FROM detections GROUP BY class_name ORDER BY cnt DESC"
    ).fetchall()
    class_counts = {r["class_name"]: r["cnt"] for r in class_rows}

    alert_rows = conn.execute(
        "SELECT class_name, COUNT(*) as cnt FROM alerts GROUP BY class_name ORDER BY cnt DESC LIMIT 5"
    ).fetchall()
    top_alert_classes = [(r["class_name"], r["cnt"]) for r in alert_rows]

    return {
        "total_detections":    total_det,
        "detections_last_hour": det_1h,
        "total_alerts":        total_alerts,
        "alerts_last_hour":    alerts_1h,
        "class_counts":        class_counts,
        "top_alert_classes":   top_alert_classes,
    }


def insert_accident(
    event_id: int,
    timestamp: float,
    lane: str,
    track_id: int,
    secondary_track_id: Optional[int],
    accident_type: str,
    confidence: float,
    score_breakdown: Optional[str] = None,
) -> None:
    """Persist an accident event."""
    ts = datetime.utcfromtimestamp(timestamp).isoformat(timespec="milliseconds")
    conn = _get_conn()
    conn.execute(
        """INSERT OR IGNORE INTO accidents
           (event_id, timestamp, lane, track_id, secondary_track_id, accident_type, confidence, score_breakdown)
           VALUES (?,?,?,?,?,?,?,?)""",
        (event_id, ts, lane, track_id, secondary_track_id, accident_type, round(confidence, 4), score_breakdown),
    )
    conn.commit()


def insert_signal_event(
    timestamp: float,
    lane: str,
    state: str,
    duration_sec: float,
) -> None:
    """Persist a traffic signal state change."""
    ts = datetime.utcfromtimestamp(timestamp).isoformat(timespec="milliseconds")
    conn = _get_conn()
    conn.execute(
        """INSERT INTO signal_events (timestamp, lane, state, duration_sec)
           VALUES (?,?,?,?)""",
        (ts, lane, state, round(duration_sec, 2)),
    )
    conn.commit()


def insert_lane_metric(
    timestamp: float,
    lane: str,
    vehicle_count: int,
    queue_length: Optional[float] = None,
    density: Optional[float] = None,
) -> None:
    """Persist a lane traffic metric."""
    ts = datetime.utcfromtimestamp(timestamp).isoformat(timespec="milliseconds")
    conn = _get_conn()
    conn.execute(
        """INSERT INTO lane_metrics (timestamp, lane, vehicle_count, queue_length, density)
           VALUES (?,?,?,?,?)""",
        (ts, lane, vehicle_count, queue_length, density),
    )
    conn.commit()


def get_accidents_in_range(start_iso: str, end_iso: str) -> List[Dict]:
    """Get accidents between timestamps (ISO format)."""
    conn = _get_conn()
    cur = conn.execute(
        "SELECT * FROM accidents WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp",
        (start_iso, end_iso),
    )
    return [dict(row) for row in cur.fetchall()]


def get_signal_events_in_range(start_iso: str, end_iso: str) -> List[Dict]:
    """Get signal events between timestamps (ISO format)."""
    conn = _get_conn()
    cur = conn.execute(
        "SELECT * FROM signal_events WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp",
        (start_iso, end_iso),
    )
    return [dict(row) for row in cur.fetchall()]


def get_lane_metrics_in_range(start_iso: str, end_iso: str) -> List[Dict]:
    """Get lane metrics between timestamps (ISO format)."""
    conn = _get_conn()
    cur = conn.execute(
        "SELECT * FROM lane_metrics WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp",
        (start_iso, end_iso),
    )
    return [dict(row) for row in cur.fetchall()]


def purge_old_records() -> None:
    """Delete records older than DB_RETENTION_DAYS. No-op if retention is 0."""
    if config.DB_RETENTION_DAYS <= 0:
        return
    cutoff = (datetime.utcnow() - timedelta(days=config.DB_RETENTION_DAYS)).isoformat()
    conn = _get_conn()
    conn.execute("DELETE FROM detections WHERE timestamp < ?", (cutoff,))
    conn.execute("DELETE FROM alerts     WHERE timestamp < ?", (cutoff,))
    conn.execute("DELETE FROM accidents  WHERE timestamp < ?", (cutoff,))
    conn.execute("DELETE FROM signal_events WHERE timestamp < ?", (cutoff,))
    conn.execute("DELETE FROM lane_metrics WHERE timestamp < ?", (cutoff,))
    conn.commit()
    logger.debug("Purged records older than %s days", config.DB_RETENTION_DAYS)