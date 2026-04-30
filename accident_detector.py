# =============================================================================
# accident_detector.py — Multi-signal accident detection engine
#
# Detects traffic incidents using motion analysis, collision detection,
# and abnormal behavior recognition with weighted scoring
# =============================================================================

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
import time
import numpy as np

logger = logging.getLogger("accident_detector")


@dataclass
class AccidentEvent:
    """Represents a detected traffic incident."""
    event_id: int
    timestamp: float
    lane: str
    primary_track_id: int
    secondary_track_id: Optional[int]
    accident_type: str  # 'collision', 'sudden_stop', 'stationary', 'abnormal_motion'
    confidence: float  # 0.0 - 1.0
    score_breakdown: Dict[str, float] = field(default_factory=dict)


class AccidentDetector:
    """Detects traffic accidents using multi-signal analysis."""

    def __init__(
        self,
        window_sec: float = 5.0,
        accident_threshold: float = 0.7,
        collision_iou_threshold: float = 0.2,
        stationary_sec: float = 3.0,
        sudden_stop_threshold: float = 0.5,  # pixels/frame²
    ):
        """
        Args:
            window_sec: Rolling window size for motion history
            accident_threshold: Score threshold to trigger accident event
            collision_iou_threshold: IoU threshold to detect collision
            stationary_sec: Seconds to consider vehicle 'stationary'
            sudden_stop_threshold: Acceleration threshold for sudden stop
        """
        self.window_sec = window_sec
        self.accident_threshold = accident_threshold
        self.collision_iou_threshold = collision_iou_threshold
        self.stationary_sec = stationary_sec
        self.sudden_stop_threshold = sudden_stop_threshold

        # Rolling window of track states
        self.track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=int(window_sec * 30)))
        self.next_event_id = 1
        self.last_alert = {}  # track_id -> timestamp of last alert

    def update(
        self,
        detections: List[Dict],
        signal_state: Optional[Dict[str, str]] = None,
    ) -> List[AccidentEvent]:
        """
        Process new detections and return accident events.

        Args:
            detections: List of detection dicts with track_id, cx, cy, vx, vy
            signal_state: Dict mapping lane -> 'RED' | 'GREEN' | 'YELLOW'

        Returns:
            List of AccidentEvent objects
        """
        signal_state = signal_state or {}
        now = time.time()

        # Update track history
        for det in detections:
            track_id = det.get("track_id")
            if track_id is not None:
                state = {
                    "timestamp": now,
                    "x1": det.get("x1", 0),
                    "y1": det.get("y1", 0),
                    "x2": det.get("x2", 0),
                    "y2": det.get("y2", 0),
                    "cx": det.get("cx", 0),
                    "cy": det.get("cy", 0),
                    "vx": det.get("vx", 0.0),
                    "vy": det.get("vy", 0.0),
                    "class_name": det.get("class_name", "unknown"),
                }
                self.track_history[track_id].append(state)

        # Evaluate accidents
        accidents = []
        evaluated_pairs = set()

        # Check for collisions (overlapping vehicles)
        for i, det1 in enumerate(detections):
            for j, det2 in enumerate(detections[i + 1 :], start=i + 1):
                tid1 = det1.get("track_id")
                tid2 = det2.get("track_id")
                if tid1 is None or tid2 is None:
                    continue

                pair = tuple(sorted([tid1, tid2]))
                if pair in evaluated_pairs:
                    continue
                evaluated_pairs.add(pair)

                collision_score = self._collision_score(det1, det2)
                if collision_score > 0.6:
                    accident = AccidentEvent(
                        event_id=self.next_event_id,
                        timestamp=now,
                        lane=det1.get("lane", "unknown"),
                        primary_track_id=tid1,
                        secondary_track_id=tid2,
                        accident_type="collision",
                        confidence=collision_score,
                        score_breakdown={"collision": collision_score},
                    )
                    self.next_event_id += 1
                    accidents.append(accident)
                    logger.warning("Collision detected: tracks %d and %d (score=%.2f)", tid1, tid2, collision_score)

        # Check for sudden stops and stationary vehicles
        for det in detections:
            track_id = det.get("track_id")
            if track_id is None:
                continue

            lane = det.get("lane", "unknown")
            signal = signal_state.get(lane, "RED")

            # Get score components
            sudden_stop = self._sudden_stop_score(track_id)
            stationary = self._stationary_score(track_id)
            motion_anomaly = self._motion_anomaly_score(track_id)

            # Combine scores (suppress some signals when RED light)
            if signal == "RED":
                # Down-weight stationary and sudden-stop at red lights
                combined_score = 0.3 * sudden_stop + 0.2 * stationary + 0.7 * motion_anomaly
            else:
                combined_score = 0.3 * sudden_stop + 0.4 * stationary + 0.3 * motion_anomaly

            # Cooldown: don't re-alert the same track within 10 seconds
            if track_id in self.last_alert and now - self.last_alert[track_id] < 10.0:
                continue

            if combined_score > self.accident_threshold:
                accident = AccidentEvent(
                    event_id=self.next_event_id,
                    timestamp=now,
                    lane=lane,
                    primary_track_id=track_id,
                    secondary_track_id=None,
                    accident_type=self._determine_type(sudden_stop, stationary, motion_anomaly),
                    confidence=combined_score,
                    score_breakdown={
                        "sudden_stop": sudden_stop,
                        "stationary": stationary,
                        "motion_anomaly": motion_anomaly,
                    },
                )
                self.next_event_id += 1
                self.last_alert[track_id] = now
                accidents.append(accident)
                logger.warning("Accident detected: track %d (type=%s, score=%.2f)", track_id, accident.accident_type, combined_score)

        # Cleanup old history
        self._cleanup_old_tracks(now)

        return accidents

    def _collision_score(self, det1: Dict, det2: Dict) -> float:
        """Compute collision likelihood based on bounding box overlap."""
        box1 = (det1.get("x1", 0), det1.get("y1", 0), det1.get("x2", 1), det1.get("y2", 1))
        box2 = (det2.get("x1", 0), det2.get("y1", 0), det2.get("x2", 1), det2.get("y2", 1))

        iou = self._iou(box1, box2)

        if iou > self.collision_iou_threshold:
            # Check if both are in motion or one is much larger (heavy collision)
            v1_sq = det1.get("vx", 0) ** 2 + det1.get("vy", 0) ** 2
            v2_sq = det2.get("vx", 0) ** 2 + det2.get("vy", 0) ** 2
            combined_speed = np.sqrt(v1_sq + v2_sq)
            return min(1.0, 0.5 * iou / self.collision_iou_threshold + 0.5 * min(combined_speed / 2.0, 1.0))

        return 0.0

    def _sudden_stop_score(self, track_id: int) -> float:
        """Detect sudden deceleration (hard braking)."""
        history = self.track_history.get(track_id)
        if not history or len(history) < 3:
            return 0.0

        states = list(history)
        if len(states) < 3:
            return 0.0

        # Recent velocity (last frame)
        v_recent = np.sqrt(states[-1]["vx"] ** 2 + states[-1]["vy"] ** 2)

        # Velocity N frames ago
        v_past = np.sqrt(states[-3]["vx"] ** 2 + states[-3]["vy"] ** 2)

        # Acceleration (negative = braking)
        if v_past > 0.1:
            accel = (v_recent - v_past) / 3.0  # 3 frames
            if accel < -self.sudden_stop_threshold:
                return min(1.0, abs(accel) / self.sudden_stop_threshold)

        return 0.0

    def _stationary_score(self, track_id: int) -> float:
        """Detect vehicles stopped in roadway for too long."""
        history = self.track_history.get(track_id)
        if not history:
            return 0.0

        states = list(history)
        if not states:
            return 0.0

        # Count frames with near-zero velocity
        stationary_frames = 0
        vel_threshold = 0.1  # pixels/frame

        for state in states[-int(self.stationary_sec * 30) :]:
            v = np.sqrt(state["vx"] ** 2 + state["vy"] ** 2)
            if v < vel_threshold:
                stationary_frames += 1

        stationary_ratio = stationary_frames / max(1, len(states))

        # Only penalize if truly stationary for the full window
        if stationary_ratio > 0.9:
            return 1.0 if list(history)[0]["timestamp"] < time.time() - self.stationary_sec else 0.0

        return 0.0

    def _motion_anomaly_score(self, track_id: int) -> float:
        """Detect erratic trajectory or sudden direction changes."""
        history = self.track_history.get(track_id)
        if not history or len(history) < 5:
            return 0.0

        states = list(history)[-10:]  # Last 10 frames
        if len(states) < 5:
            return 0.0

        # Compute heading changes
        headings = []
        for state in states:
            vx, vy = state["vx"], state["vy"]
            if np.sqrt(vx ** 2 + vy ** 2) > 0.1:
                heading = np.arctan2(vy, vx)
                headings.append(heading)

        if len(headings) < 2:
            return 0.0

        # Detect sharp turns (heading changes > 90°)
        sharp_turns = 0
        for i in range(1, len(headings)):
            delta_heading = abs(headings[i] - headings[i - 1])
            # Normalize to [0, π]
            if delta_heading > np.pi:
                delta_heading = 2 * np.pi - delta_heading
            if delta_heading > np.pi / 2:  # 90°
                sharp_turns += 1

        anomaly_ratio = sharp_turns / max(1, len(headings) - 1)
        return min(1.0, anomaly_ratio)

    def _determine_type(self, sudden_stop: float, stationary: float, motion_anomaly: float) -> str:
        """Determine accident type based on score breakdown."""
        if motion_anomaly > 0.7:
            return "abnormal_motion"
        elif stationary > 0.7:
            return "stationary"
        elif sudden_stop > 0.7:
            return "sudden_stop"
        else:
            return "collision"

    @staticmethod
    def _iou(box1: Tuple, box2: Tuple) -> float:
        """Compute IoU of two boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        inter_min_x = max(x1_min, x2_min)
        inter_min_y = max(y1_min, y2_min)
        inter_max_x = min(x1_max, x2_max)
        inter_max_y = min(y1_max, y2_max)

        if inter_max_x < inter_min_x or inter_max_y < inter_min_y:
            return 0.0

        inter_area = (inter_max_x - inter_min_x) * (inter_max_y - inter_min_y)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _cleanup_old_tracks(self, now: float):
        """Remove tracks older than the window."""
        cutoff = now - self.window_sec
        for track_id in list(self.track_history.keys()):
            history = self.track_history[track_id]
            while history and history[0]["timestamp"] < cutoff:
                history.popleft()
            if not history:
                del self.track_history[track_id]
