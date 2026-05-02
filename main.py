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
from datetime import datetime, timezone

import api as flask_api
import config
import database as db
from accident_detector import AccidentDetector
from detector import DetectionResult, FrameBuffer, YOLOv8Detector
from emergency_responder import EmergencyResponder
from lane_manager import LaneManager
from metrics import MetricsEngine
from nlp_engine import AlertGenerator, SceneCaptioner
from signal_controller import PCU_WEIGHTS, PhaseManager, TrafficSignalController
from tracker import VehicleTracker

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.CRITICAL,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_FILE),
    ],
)
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

# Silence everything else
logging.getLogger("tracker").setLevel(logging.CRITICAL)
logging.getLogger("lane_manager").setLevel(logging.CRITICAL)
logging.getLogger("accident_detector").setLevel(logging.CRITICAL)
logging.getLogger("detector").setLevel(logging.CRITICAL)
logging.getLogger("nlp_engine").setLevel(logging.CRITICAL)
logging.getLogger("database").setLevel(logging.CRITICAL)
logging.getLogger("simulator").setLevel(logging.CRITICAL)
logging.getLogger("flask").setLevel(logging.CRITICAL)
logging.getLogger("werkzeug").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)


# ---------------------------------------------------------------------------
# Processing loop
# ---------------------------------------------------------------------------


class ProcessingLoop:
    """
    Sits between the detector and the database/NLP layer.
    Polls the FrameBuffer for new results and processes each unique frame.
    Integrates tracking, lane management, accident detection, and signal control.
    """

    def __init__(
        self,
        frame_buffer: FrameBuffer,
        captioner: SceneCaptioner,
        alert_gen: AlertGenerator,
        tracker: VehicleTracker,
        lane_manager: LaneManager,
        accident_detector: AccidentDetector,
        emergency_responder: EmergencyResponder,
        metrics_engine: MetricsEngine,
        signal_controller: TrafficSignalController,
        stop_event: threading.Event,
    ) -> None:
        self._buf = frame_buffer
        self._captioner = captioner
        self._alert_gen = alert_gen
        self._tracker = tracker
        self._lane_manager = lane_manager
        self._accident_det = accident_detector
        self._emergency_resp = emergency_responder
        self._metrics = metrics_engine
        self._signal_ctrl = signal_controller
        self._phase_mgr = PhaseManager()  # NEW
        self._detector = None  # Will be set in main()
        self._stop = stop_event
        self._last_processed_id = -1
        self._latest_cycle_result = None
        self._current_signal_state = {
            "north": "RED",
            "south": "RED",
            "east": "GREEN",
            "west": "GREEN",
        }

    def set_detector(self, detector):
        self._detector = detector

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
        # 1. Track vehicles across frames
        tracked_dets = self._tracker.update(result.detections)

        # 2. Assign lanes and compute metrics
        for det in tracked_dets:
            det["lane"] = self._lane_manager.assign_lane(det)

        lane_densities = self._lane_manager.density_per_lane(tracked_dets)
        lane_queues = self._lane_manager.queue_length_per_lane(tracked_dets)

        # Store lane metrics
        now = time.time()
        for lane in ["north", "south", "east", "west"]:
            density = lane_densities.get(lane, {}).get("density", 0.0)
            vehicle_count = sum(1 for d in tracked_dets if d.get("lane") == lane)
            queue_length = lane_queues.get(lane, 0.0)
            db.insert_lane_metric(
                timestamp=now,
                lane=lane,
                vehicle_count=vehicle_count,
                queue_length=queue_length,
                density=density,
            )

        # 3. Detect accidents
        accidents = self._accident_det.update(
            tracked_dets, signal_state=self._current_signal_state
        )
        for accident in accidents:
            db.insert_accident(
                event_id=accident.event_id,
                timestamp=accident.timestamp,
                lane=accident.lane,
                track_id=accident.primary_track_id,
                secondary_track_id=accident.secondary_track_id,
                accident_type=accident.accident_type,
                confidence=accident.confidence,
                score_breakdown=str(accident.score_breakdown),
            )
            # Trigger emergency response
            self._emergency_resp.handle(accident)

        # 4. Compute signal timings (4-way intersection with Webster method)
        lane_data = {}
        for lane in ["north", "south", "east", "west"]:
            lane_vehicles = [d for d in tracked_dets if d.get("lane") == lane]
            pcu_total = 0.0
            breakdown = {}
            for det in lane_vehicles:
                cls = det.get("class_name", "unknown")
                pcu = PCU_WEIGHTS.get(cls, 0.0)
                pcu_total += pcu
                breakdown[cls] = breakdown.get(cls, 0.0) + pcu
            lane_data[lane] = {
                "pcu": pcu_total,
                "vehicle_count": len(lane_vehicles),
                "density": lane_densities.get(lane, {}).get("density", 0.0),
                "queue_length": lane_queues.get(lane, 0.0),
                "breakdown": breakdown,
            }

        cycle_result = self._signal_ctrl.compute(lane_data)
        self._latest_cycle_result = cycle_result
        self._current_signal_state = self._phase_mgr.update(
            cycle_result
        )  # CORRECT LOGIC

        # Push signal state back to detector if it's the simulator
        if self._detector and hasattr(self._detector, "signal_state"):
            self._detector.signal_state = self._current_signal_state

        # 5. Generate scene caption
        caption = self._captioner.caption(
            result.detections,
            frame_width=config.CAMERA_WIDTH,
            frame_height=config.CAMERA_HEIGHT,
        )

        # 6. Persist detections to DB
        db.insert_detections(
            frame_id=result.frame_id,
            detections=result.detections,
            scene_caption=caption,
            inference_ms=result.inference_ms,
        )

        # 7. Evaluate alerts
        alerts = self._alert_gen.evaluate(result.detections, frame_id=result.frame_id)
        for alert in alerts:
            db.insert_alert(
                class_name=alert["class_name"],
                severity=alert["severity"],
                confidence=alert["confidence"],
                message=alert["message"],
                frame_id=alert["frame_id"],
            )

        # Store signal state in DB
        for lane, state in self._current_signal_state.items():
            db.insert_signal_event(
                timestamp=now,
                lane=lane,
                state=state,
                duration_sec=cycle_result.cycle_length,
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_proc_loop_ref = None


def main() -> None:
    logger.info("=" * 60)
    logger.info(
        "Edge AI System starting  —  %s", datetime.now(timezone.utc).isoformat()
    )
    logger.info("RTSP URL   : %s", config.RTSP_URL)
    logger.info("Model      : %s", config.MODEL_PATH)
    logger.info("Device     : %s", config.DEVICE)
    logger.info("API        : http://%s:%s", config.API_HOST, config.API_PORT)
    logger.info("=" * 60)

    # Initialise database
    db.init_db()

    # Shared objects
    frame_buffer = FrameBuffer()
    stop_event = threading.Event()

    # NLP components
    captioner = SceneCaptioner()
    alert_gen = AlertGenerator()

    # Traffic system components
    tracker = VehicleTracker(
        max_age=config.TRACKER_MAX_AGE,
        min_hits=config.TRACKER_MIN_HITS,
        max_distance=config.TRACKER_MAX_DISTANCE,
        iou_threshold=config.TRACKER_IOU_THRESHOLD,
    )
    lane_manager = LaneManager(config.LANES)
    accident_detector = AccidentDetector(
        window_sec=config.ACCIDENT_WINDOW_SEC,
        accident_threshold=config.ACCIDENT_THRESHOLD,
        collision_iou_threshold=config.ACCIDENT_COLLISION_IOU,
        stationary_sec=config.ACCIDENT_STATIONARY_SEC,
        sudden_stop_threshold=config.ACCIDENT_SUDDEN_STOP_THRESHOLD,
    )
    emergency_responder = EmergencyResponder(
        webhook_url=config.EMERGENCY_WEBHOOK_URL,
        webhook_timeout=config.EMERGENCY_WEBHOOK_TIMEOUT,
        signal_preemption_duration=config.EMERGENCY_PREEMPTION_DURATION,
    )
    metrics_engine = MetricsEngine(db=db)
    signal_controller = TrafficSignalController()

    # Wire the FrameBuffer and signal state into the Flask API
    flask_api.set_frame_buffer(frame_buffer)

    def get_signal_state():
        if "proc_loop" in locals():
            return proc_loop._current_signal_state
        return None

    def get_signal_timing():
        if "proc_loop" in locals():
            return {
                "remaining_seconds": proc_loop._phase_mgr.remaining_seconds,
                "timers": proc_loop._phase_mgr.get_timers(),
            }
        return None

    def get_traffic_snapshot():
        if "proc_loop" in locals() and proc_loop._latest_cycle_result is not None:
            return {"cycle_result": proc_loop._latest_cycle_result}
        return None

    flask_api.set_signal_state_provider(get_signal_state)
    flask_api.set_signal_timing_provider(get_signal_timing)
    flask_api.set_traffic_snapshot_provider(get_traffic_snapshot)

    # ----- Thread 1: YOLOv8 detection (or simulator) -----
    if config.SIMULATOR_ENABLED:
        from traffic_simulator import TrafficSimulator

        detector = TrafficSimulator(
            width=config.SIMULATOR_WIDTH,
            height=config.SIMULATOR_HEIGHT,
            fps=config.SIMULATOR_FPS,
            scenario=config.SIMULATOR_SCENARIO,
        )
        detector._stop_evt = threading.Event()  # For compatibility
        logger.info("Using traffic simulator (scenario: %s)", config.SIMULATOR_SCENARIO)
        det_thread = threading.Thread(
            target=lambda: detector.run(frame_buffer),
            name="DetectorThread",
            daemon=True,
        )
    else:
        detector = YOLOv8Detector(frame_buffer)
        det_thread = threading.Thread(
            target=detector.run, name="DetectorThread", daemon=True
        )

    flask_api.set_detector(detector)

    # ----- Thread 2: Processing (NLP + DB + tracking + signals) -----
    global _proc_loop_ref
    proc_loop = ProcessingLoop(
        frame_buffer,
        captioner,
        alert_gen,
        tracker,
        lane_manager,
        accident_detector,
        emergency_responder,
        metrics_engine,
        signal_controller,
        stop_event,
    )
    proc_loop.set_detector(detector)
    _proc_loop_ref = proc_loop
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

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Start all threads
    det_thread.start()
    proc_thread.start()
    api_thread.start()

    logger.info(
        "All threads started. Dashboard at http://%s:%s",
        config.API_HOST,
        config.API_PORT,
    )
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown(None, None)


if __name__ == "__main__":
    main()
