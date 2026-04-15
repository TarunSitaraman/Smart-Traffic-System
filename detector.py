# =============================================================================
# detector.py — YOLOv8 inference on an RTSP stream (Jetson Nano optimised)
#
# Key Jetson Nano considerations:
#   • GStreamer pipeline is used for hardware-accelerated RTSP decoding
#   • TensorRT (.engine) export is recommended for ~3-4× speedup vs PyTorch
#   • Frame skip reduces compute load while keeping real-time latency low
#   • Thread-safe FrameBuffer ensures the API always reads the latest frame
# =============================================================================

import logging
import time
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """One frame's worth of inference output."""
    frame_id:      int
    frame:         np.ndarray          # BGR image (annotated)
    raw_frame:     np.ndarray          # BGR image (unannotated, for saving)
    detections:    List[Dict[str, Any]]  # list of detection dicts
    inference_ms:  float
    timestamp:     float = field(default_factory=time.time)

    @property
    def is_empty(self) -> bool:
        return len(self.detections) == 0


# ---------------------------------------------------------------------------
# Thread-safe latest-frame buffer
# ---------------------------------------------------------------------------

class FrameBuffer:
    """
    Holds the single most-recent DetectionResult.
    The API thread reads from this; the detector thread writes to it.
    """

    def __init__(self) -> None:
        self._lock   = threading.Lock()
        self._result: Optional[DetectionResult] = None

    def put(self, result: DetectionResult) -> None:
        with self._lock:
            self._result = result

    def get(self) -> Optional[DetectionResult]:
        with self._lock:
            return self._result


# ---------------------------------------------------------------------------
# GStreamer RTSP pipeline builder
# ---------------------------------------------------------------------------

def _build_gstreamer_pipeline(rtsp_url: str, width: int, height: int, fps: int) -> str:
    """
    Build an optimised GStreamer pipeline for the Jetson Nano.
    Uses nvv4l2decoder for hardware H.264/H.265 decoding and
    nvvidconv for colour-space conversion.

    Falls back to a software pipeline if GStreamer is unavailable.
    """
    pipeline = (
        f"rtspsrc location={rtsp_url} latency=100 ! "
        "rtph264depay ! h264parse ! "
        "nvv4l2decoder ! "                          # HW decoder on Jetson
        f"nvvidconv ! "
        f"video/x-raw,format=BGRx,width={width},height={height},framerate={fps}/1 ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink drop=1"
    )
    return pipeline


def _open_capture(rtsp_url: str) -> cv2.VideoCapture:
    """
    Attempt to open the video capture with a GStreamer pipeline (Jetson).
    Falls back to standard OpenCV RTSP handling for development environments.
    """
    # Try GStreamer first (Jetson Nano with JetPack)
    gst_pipe = _build_gstreamer_pipeline(
        rtsp_url,
        config.CAMERA_WIDTH,
        config.CAMERA_HEIGHT,
        config.CAMERA_FPS,
    )
    cap = cv2.VideoCapture(gst_pipe, cv2.CAP_GSTREAMER)

    if cap.isOpened():
        logger.info("Opened stream via GStreamer pipeline")
        return cap

    # Fallback: plain OpenCV (works on desktops / testing)
    logger.warning(
        "GStreamer pipeline failed, falling back to cv2 direct RTSP (%s)", rtsp_url
    )
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {rtsp_url}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          config.CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # keep latency minimal
    return cap


# ---------------------------------------------------------------------------
# YOLOv8 Detector
# ---------------------------------------------------------------------------

class YOLOv8Detector:
    """
    Loads YOLOv8 and runs inference on frames from an RTSP stream.

    Exports to TensorRT on first run when model_path ends with '.engine',
    otherwise uses the PyTorch model directly.

    Usage (from another thread):
        detector = YOLOv8Detector(frame_buffer)
        detector.run()   # blocking loop — call from a dedicated thread
    """

    def __init__(self, frame_buffer: FrameBuffer) -> None:
        self._buffer    = frame_buffer
        self._stop_evt  = threading.Event()
        self._frame_id  = 0

        logger.info("Loading YOLOv8 model from: %s", config.MODEL_PATH)
        self._model = YOLO(config.MODEL_PATH)

        # Warm up the model with a dummy frame
        dummy = np.zeros(
            (config.INFERENCE_IMG_SIZE, config.INFERENCE_IMG_SIZE, 3), dtype=np.uint8
        )
        self._model.predict(
            source=dummy,
            device=config.DEVICE,
            verbose=False,
            conf=config.CONFIDENCE_THRESH,
        )
        logger.info("YOLOv8 warm-up complete (device=%s)", config.DEVICE)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Main detection loop. Opens the RTSP stream, reads frames,
        runs YOLOv8 inference, and pushes results into the FrameBuffer.

        Automatically reconnects on stream loss.
        """
        while not self._stop_evt.is_set():
            cap = None
            try:
                cap = _open_capture(config.RTSP_URL)
                logger.info("Stream opened. Starting detection loop.")
                self._detection_loop(cap)
            except Exception as exc:
                logger.error("Stream error: %s — reconnecting in 5 s", exc)
                time.sleep(5)
            finally:
                if cap:
                    cap.release()

    def stop(self) -> None:
        """Signal the detection loop to exit."""
        self._stop_evt.set()

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _detection_loop(self, cap: cv2.VideoCapture) -> None:
        skip_counter = 0

        while not self._stop_evt.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("Frame read failed — stream may have dropped.")
                break

            skip_counter += 1
            if skip_counter < config.FRAME_SKIP:
                continue
            skip_counter = 0

            self._frame_id += 1
            raw_frame = frame.copy()

            t_start = time.perf_counter()
            detections, annotated = self._infer(frame)
            inference_ms = (time.perf_counter() - t_start) * 1000

            result = DetectionResult(
                frame_id=self._frame_id,
                frame=annotated,
                raw_frame=raw_frame,
                detections=detections,
                inference_ms=inference_ms,
            )
            self._buffer.put(result)

            logger.debug(
                "Frame %d | %d detections | %.1f ms",
                self._frame_id, len(detections), inference_ms,
            )

    def _infer(
        self, frame: np.ndarray
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Run YOLOv8 on a BGR frame.
        Returns (list_of_detection_dicts, annotated_bgr_frame).
        """
        results = self._model.predict(
            source=frame,
            device=config.DEVICE,
            conf=config.CONFIDENCE_THRESH,
            iou=config.IOU_THRESH,
            imgsz=config.INFERENCE_IMG_SIZE,
            max_det=config.MAX_DETECTIONS_PER_FRAME,
            verbose=False,
            stream=False,
        )

        detections: List[Dict[str, Any]] = []
        annotated = frame.copy()

        for r in results:
            if r.boxes is None:
                continue

            # Normalise bounding boxes to [0, 1]
            h, w = frame.shape[:2]
            boxes  = r.boxes.xyxy.cpu().numpy()      # (N, 4) pixel coords
            confs  = r.boxes.conf.cpu().numpy()       # (N,)
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)  # (N,)
            names  = r.names                          # {id: name}

            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                x1, y1, x2, y2 = box.tolist()
                class_name = names[cls_id]

                detections.append({
                    "class_name": class_name,
                    "confidence": float(conf),
                    "x1": x1 / w,
                    "y1": y1 / h,
                    "x2": x2 / w,
                    "y2": y2 / h,
                })

                # Draw bounding box and label on the annotated frame
                annotated = _draw_box(
                    annotated,
                    int(x1), int(y1), int(x2), int(y2),
                    class_name, float(conf),
                )

        return detections, annotated


# ---------------------------------------------------------------------------
# Drawing helper
# ---------------------------------------------------------------------------

_PALETTE: Dict[str, Tuple[int, int, int]] = {}


def _class_colour(class_name: str) -> Tuple[int, int, int]:
    """Return a consistent BGR colour for a class name."""
    if class_name not in _PALETTE:
        h = hash(class_name) & 0xFFFFFF
        r, g, b = (h >> 16) & 0xFF, (h >> 8) & 0xFF, h & 0xFF
        # Brighten to avoid very dark colours
        r, g, b = max(r, 80), max(g, 80), max(b, 80)
        _PALETTE[class_name] = (b, g, r)  # OpenCV BGR
    return _PALETTE[class_name]


def _draw_box(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    class_name: str,
    confidence: float,
) -> np.ndarray:
    colour    = _class_colour(class_name)
    label     = f"{class_name} {confidence:.0%}"
    font      = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)

    (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
    label_y = max(y1 - 6, th + 4)
    cv2.rectangle(frame, (x1, label_y - th - 4), (x1 + tw + 4, label_y + 2), colour, -1)
    cv2.putText(
        frame, label,
        (x1 + 2, label_y - 2),
        font, font_scale,
        (255, 255, 255), 1, cv2.LINE_AA,
    )
    return frame