# =============================================================================
# main.py — Orchestrator: ties detection, NLP, DB logging, and API together
#
# Architecture (all threads run in the same process):
#
#   ┌──────────────────────────────────────────────────────┐
#   │  Thread 1: YOLOv8Detector (detector.py)              │
#   │    RTSP → frame → infer → DetectionResult            │
#   │             ↓ put()                                  │
#   │         FrameBuffer (shared)                         │
#   │             ↑ get()                                  │
#   │  Thread 2: Processing Loop (this file)               │
#   │    DetectionResult → NLP → DB → alert                │
#   │                                                      │
#   │  Thread 3: Flask API (api.py)                        │
#   │    /api/*, /live, / — reads DB + FrameBuffer         │
#   └──────────────────────────────────────────────────────┘
#
# Usage:
#   python main.py
#   # or override settings via env vars:
#   RTSP_URL=rtsp://... MODEL_PATH=yolov8n.engine python main.py
# =============================================================================

import logging
import signal
import sys
import threading
import time
from datetime import datetime

import config
import database as db
import api as flask_api
from detector import YOLOv8Detector, FrameBuffer, DetectionResult
from nlp_engine import SceneCaptioner, AlertGenerator


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_FILE),
    ],
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Processing loop
# ---------------------------------------------------------------------------

class ProcessingLoop:
    """
    Sits between the detector and the database/NLP layer.
    Polls the FrameBuffer for new results and processes each unique frame.
    """

    def __init__(
        self,
        frame_buffer: FrameBuffer,
        captioner: SceneCaptioner,
        alert_gen: AlertGenerator,
        stop_event: threading.Event,
    ) -> None:
        self._buf       = frame_buffer
        self._captioner = captioner
        self._alert_gen = alert_gen
        self._stop      = stop_event
        self._last_processed_id = -1

    def run(self) -> None:
        logger.info("Processing loop started.")
        purge_counter = 0

        while not self._stop.is_set():
            result: DetectionResult = self._buf.get()

            if result is None or result.frame_id == self._last_processed_id:
                time.sleep(0.01)
                continue

            self._last_processed_id = result.frame_id
            self._process(result)

            # Purge old DB records every ~1000 processed frames
            purge_counter += 1
            if purge_counter >= 1000:
                purge_counter = 0
                db.purge_old_records()

        logger.info("Processing loop stopped.")

    def _process(self, result: DetectionResult) -> None:
        # 1. Generate scene caption
        caption = self._captioner.caption(
            result.detections,
            frame_width=config.CAMERA_WIDTH,
            frame_height=config.CAMERA_HEIGHT,
        )

        # 2. Persist detections to DB
        db.insert_detections(
            frame_id=result.frame_id,
            detections=result.detections,
            scene_caption=caption,
            inference_ms=result.inference_ms,
        )

        # 3. Evaluate alerts
        alerts = self._alert_gen.evaluate(result.detections, frame_id=result.frame_id)
        for alert in alerts:
            db.insert_alert(
                class_name=alert["class_name"],
                severity=alert["severity"],
                confidence=alert["confidence"],
                message=alert["message"],
                frame_id=alert["frame_id"],
            )

        if result.detections:
            logger.info(
                "Frame %d | %d objects | %.1f ms | Caption: %s",
                result.frame_id,
                len(result.detections),
                result.inference_ms,
                caption[:80],
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=" * 60)
    logger.info("Edge AI System starting  —  %s", datetime.utcnow().isoformat())
    logger.info("RTSP URL   : %s", config.RTSP_URL)
    logger.info("Model      : %s", config.MODEL_PATH)
    logger.info("Device     : %s", config.DEVICE)
    logger.info("API        : http://%s:%s", config.API_HOST, config.API_PORT)
    logger.info("=" * 60)

    # Initialise database
    db.init_db()

    # Shared objects
    frame_buffer = FrameBuffer()
    stop_event   = threading.Event()

    # NLP components
    captioner  = SceneCaptioner()
    alert_gen  = AlertGenerator()

    # Wire the FrameBuffer into the Flask API
    flask_api.set_frame_buffer(frame_buffer)

    # ----- Thread 1: YOLOv8 detection -----
    detector = YOLOv8Detector(frame_buffer)
    flask_api.set_detector(detector)
    det_thread = threading.Thread(
        target=detector.run, name="DetectorThread", daemon=True
    )

    # ----- Thread 2: Processing (NLP + DB) -----
    proc_loop = ProcessingLoop(frame_buffer, captioner, alert_gen, stop_event)
    proc_thread = threading.Thread(
        target=proc_loop.run, name="ProcessingThread", daemon=True
    )

    # ----- Thread 3: Flask API -----
    api_thread = threading.Thread(
        target=lambda: flask_api.app.run(
            host=config.API_HOST,
            port=config.API_PORT,
            debug=config.API_DEBUG,
            use_reloader=False,
            threaded=True,
        ),
        name="APIThread",
        daemon=True,
    )

    # Graceful shutdown on SIGINT / SIGTERM
    def shutdown(signum, frame):
        logger.info("Shutdown signal received — stopping…")
        stop_event.set()
        detector.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Start all threads
    det_thread.start()
    proc_thread.start()
    api_thread.start()

    logger.info("All threads started. Dashboard at http://%s:%s", config.API_HOST, config.API_PORT)

    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown(None, None)


if __name__ == "__main__":
    main()  