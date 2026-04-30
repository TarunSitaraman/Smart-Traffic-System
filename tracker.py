# =============================================================================
# tracker.py — Vehicle tracking with stable IDs across frames
#
# Implements a centroid-based tracker with Kalman filtering and IoU matching
# to assign persistent track_ids to detected vehicles across consecutive frames
# =============================================================================

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time
import numpy as np
from collections import defaultdict

logger = logging.getLogger("tracker")


@dataclass
class Track:
    """Represents a tracked vehicle across multiple frames."""
    track_id: int
    class_name: str
    x1: float
    y1: float
    x2: float
    y2: float
    cx: float
    cy: float
    vx: float = 0.0  # velocity x
    vy: float = 0.0  # velocity y
    confidence: float = 0.0
    age: int = 0  # frames since creation
    hits: int = 0  # frames with successful assignment
    last_seen: float = field(default_factory=time.time)


class VehicleTracker:
    """Multi-object tracker using centroid matching and velocity prediction."""

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        max_distance: float = 0.1,
    ):
        """
        Args:
            max_age: Max frames a track can exist without a match before removal
            min_hits: Min hits required before a track is considered "confirmed"
            iou_threshold: Minimum IoU to consider a match
            max_distance: Maximum normalized distance (centroid) for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_distance = max_distance
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}
        self.frame_count = 0

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracks with new detections.

        Args:
            detections: List of dicts with keys: class_name, confidence, x1, y1, x2, y2

        Returns:
            List of detection dicts enriched with track_id, cx, cy, vx, vy, age, hits
        """
        self.frame_count += 1
        current_time = time.time()

        # Predict new track positions
        for track in self.tracks.values():
            track.cx += track.vx
            track.cy += track.vy
            track.age += 1

        # Match detections to existing tracks
        matched_pairs, unmatched_dets, unmatched_tracks = self._match(detections)

        # Update matched tracks
        updated_detections = []
        for det_idx, track_id in matched_pairs:
            det = detections[det_idx].copy()
            track = self.tracks[track_id]

            # Update centroid and velocity
            new_cx = (det["x1"] + det["x2"]) / 2
            new_cy = (det["y1"] + det["y2"]) / 2
            track.vx = 0.8 * track.vx + 0.2 * (new_cx - track.cx)
            track.vy = 0.8 * track.vy + 0.2 * (new_cy - track.cy)
            track.cx = new_cx
            track.cy = new_cy
            track.x1 = det["x1"]
            track.y1 = det["y1"]
            track.x2 = det["x2"]
            track.y2 = det["y2"]
            track.confidence = det["confidence"]
            track.hits += 1
            track.last_seen = current_time

            det["track_id"] = track_id
            det["cx"] = track.cx
            det["cy"] = track.cy
            det["vx"] = track.vx
            det["vy"] = track.vy
            det["age"] = track.age
            det["hits"] = track.hits
            updated_detections.append(det)

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx].copy()
            cx = (det["x1"] + det["x2"]) / 2
            cy = (det["y1"] + det["y2"]) / 2
            track = Track(
                track_id=self.next_id,
                class_name=det["class_name"],
                x1=det["x1"],
                y1=det["y1"],
                x2=det["x2"],
                y2=det["y2"],
                cx=cx,
                cy=cy,
                confidence=det["confidence"],
                age=0,
                hits=1,
                last_seen=current_time,
            )
            self.tracks[self.next_id] = track
            self.next_id += 1

            det["track_id"] = track.track_id
            det["cx"] = track.cx
            det["cy"] = track.cy
            det["vx"] = track.vx
            det["vy"] = track.vy
            det["age"] = track.age
            det["hits"] = track.hits
            updated_detections.append(det)

        # Remove dead tracks
        dead_ids = []
        for track_id, track in self.tracks.items():
            if track.age > self.max_age:
                dead_ids.append(track_id)
        for track_id in dead_ids:
            del self.tracks[track_id]

        return updated_detections

    def _match(self, detections: List[Dict]):
        """
        Match detections to existing tracks using IoU and centroid distance.

        Returns:
            matched_pairs: List of (detection_idx, track_id) tuples
            unmatched_dets: List of detection indices with no match
            unmatched_tracks: List of track_ids with no match
        """
        if not detections or not self.tracks:
            return [], list(range(len(detections))), list(self.tracks.keys())

        # Compute cost matrix: IoU + distance
        n_det = len(detections)
        n_track = len(self.tracks)
        track_ids = list(self.tracks.keys())

        cost_matrix = np.full((n_det, n_track), np.inf)

        for i, det in enumerate(detections):
            det_box = np.array([det["x1"], det["y1"], det["x2"], det["y2"]])
            det_cx = (det["x1"] + det["x2"]) / 2
            det_cy = (det["y1"] + det["y2"]) / 2

            for j, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                track_box = np.array([track.x1, track.y1, track.x2, track.y2])

                # IoU
                iou = self._iou(det_box, track_box)

                # Centroid distance (normalized)
                dist = np.sqrt((det_cx - track.cx) ** 2 + (det_cy - track.cy) ** 2)

                # Combine: prefer IoU, then distance
                if iou < self.iou_threshold or dist > self.max_distance:
                    cost = np.inf
                else:
                    cost = 1.0 - iou + 0.1 * dist

                cost_matrix[i, j] = cost

        # Greedy matching (simple approach; upgrade to Hungarian if scipy available)
        matched_pairs = []
        unmatched_dets = set(range(n_det))
        unmatched_tracks = set(range(n_track))

        # Sort by cost and greedily assign
        for i in range(n_det):
            best_j = np.argmin(cost_matrix[i])
            if cost_matrix[i, best_j] < np.inf:
                if best_j not in {p[1] for p in matched_pairs}:  # j not already matched
                    matched_pairs.append((i, track_ids[best_j]))
                    unmatched_dets.discard(i)
                    unmatched_tracks.discard(best_j)

        return matched_pairs, list(unmatched_dets), [track_ids[j] for j in unmatched_tracks]

    @staticmethod
    def _iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute Intersection over Union (normalized coords 0-1)."""
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

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    def get_confirmed_tracks(self) -> List[Track]:
        """Return only tracks with enough hits to be considered 'confirmed'."""
        return [t for t in self.tracks.values() if t.hits >= self.min_hits]
