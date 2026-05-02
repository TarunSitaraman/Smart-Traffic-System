# =============================================================================
# lane_manager.py — Lane detection and density calculation for intersections
#
# Manages lane polygons, assigns vehicles to lanes, calculates density and
# queue lengths per lane and direction
# =============================================================================

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np

logger = logging.getLogger("lane_manager")


class LaneManager:
    """Manages lane-based vehicle assignment and density metrics."""
    EXPECTED_VISIBLE_CAPACITY = 6
    MAX_QUEUE_GAP = 0.12

    # Default 4-way intersection polygon configuration (normalized 0-1)
    DEFAULT_LANES = {
        "north": {  # approaching from top (coming downward)
            "polygon": [(0.35, 0.0), (0.65, 0.0), (0.65, 0.5), (0.35, 0.5)],
            "direction": "N->S",
            "stop_line_y": 0.48,
        },
        "south": {  # approaching from bottom (coming upward)
            "polygon": [(0.35, 0.5), (0.65, 0.5), (0.65, 1.0), (0.35, 1.0)],
            "direction": "S->N",
            "stop_line_y": 0.52,
        },
        "east": {  # approaching from right (coming leftward)
            "polygon": [(0.5, 0.35), (1.0, 0.35), (1.0, 0.65), (0.5, 0.65)],
            "direction": "E->W",
            "stop_line_x": 0.52,
        },
        "west": {  # approaching from left (coming rightward)
            "polygon": [(0.0, 0.35), (0.5, 0.35), (0.5, 0.65), (0.0, 0.65)],
            "direction": "W->E",
            "stop_line_x": 0.48,
        },
    }

    def __init__(self, lanes: Optional[Dict] = None, frame_width: int = 1280, frame_height: int = 720):
        """
        Args:
            lanes: Dict of lane configs (key -> {polygon, direction, stop_line_*})
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
        """
        self.lanes = lanes or self.DEFAULT_LANES
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Convert polygons to numpy arrays for faster point-in-polygon tests
        self.polygons = {}
        for lane_name, config in self.lanes.items():
            self.polygons[lane_name] = np.array(config["polygon"], dtype=np.float32)

    def assign_lane(self, detection: Dict) -> Optional[str]:
        """
        Assign a detection to a lane based on its centroid.

        Args:
            detection: Detection dict with cx, cy (normalized 0-1)

        Returns:
            Lane name or None if not in any lane
        """
        cx = detection.get("cx", (detection.get("x1", 0) + detection.get("x2", 0)) / 2)
        cy = detection.get("cy", (detection.get("y1", 0) + detection.get("y2", 0)) / 2)

        point = np.array([cx, cy], dtype=np.float32)

        for lane_name, polygon in self.polygons.items():
            if self._point_in_polygon(point, polygon):
                return lane_name

        return None

    def density_per_lane(self, detections: List[Dict]) -> Dict[str, Dict]:
        """
        Calculate density and vehicle counts per lane.

        Args:
            detections: List of detection dicts

        Returns:
            Dict mapping lane_name -> {count, density, class_breakdown}
        """
        lane_data = {ln: {"count": 0, "density": 0.0, "classes": {}} for ln in self.lanes.keys()}

        for det in detections:
            lane = self.assign_lane(det)
            if lane:
                lane_data[lane]["count"] += 1
                cls = det.get("class_name", "unknown")
                lane_data[lane]["classes"][cls] = lane_data[lane]["classes"].get(cls, 0) + 1

        # Demo-friendly density: normalize by expected visible lane capacity
        # instead of polygon area, which overstates density for 1-2 vehicles.
        for lane_name in self.polygons.keys():
            count = lane_data[lane_name]["count"]
            lane_data[lane_name]["density"] = min(
                10.0, (count / self.EXPECTED_VISIBLE_CAPACITY) * 10.0
            )

        return lane_data

    def queue_length_per_lane(self, detections: List[Dict]) -> Dict[str, float]:
        """
        Estimate queue length per lane based on vehicle spacing.

        Args:
            detections: List of detection dicts with x1, y1, x2, y2, track_id (optional)

        Returns:
            Dict mapping lane_name -> estimated_queue_length (in normalized frame units)
        """
        queue_lengths = {}

        for lane_name, config in self.lanes.items():
            lane_vehicles = [d for d in detections if self.assign_lane(d) == lane_name]

            if not lane_vehicles:
                queue_lengths[lane_name] = 0.0
                continue

            # Sort by position along the direction
            direction = config["direction"]
            if direction.startswith("N") or direction.startswith("S"):
                # Sort by y coordinate
                lane_vehicles.sort(key=lambda d: d.get("cy", (d.get("y1", 0) + d.get("y2", 0)) / 2))
            else:
                # Sort by x coordinate
                lane_vehicles.sort(key=lambda d: d.get("cx", (d.get("x1", 0) + d.get("x2", 0)) / 2))

            stop_line = config.get("stop_line_y", config.get("stop_line_x", 0.5))

            def front_distance(det: Dict) -> float:
                if direction == "N->S":
                    return max(0.0, stop_line - det.get("y2", 0.0))
                if direction == "S->N":
                    return max(0.0, det.get("y1", 0.0) - stop_line)
                if direction == "E->W":
                    return max(0.0, det.get("x1", 0.0) - stop_line)
                return max(0.0, stop_line - det.get("x2", 0.0))

            def gap_between(front: Dict, back: Dict) -> float:
                if direction == "N->S":
                    return max(0.0, back.get("y1", 0.0) - front.get("y2", 0.0))
                if direction == "S->N":
                    return max(0.0, front.get("y1", 0.0) - back.get("y2", 0.0))
                if direction == "E->W":
                    return max(0.0, front.get("x1", 0.0) - back.get("x2", 0.0))
                return max(0.0, back.get("x1", 0.0) - front.get("x2", 0.0))

            queue_group = []
            if front_distance(lane_vehicles[0]) <= self.MAX_QUEUE_GAP:
                queue_group.append(lane_vehicles[0])
                for i in range(1, len(lane_vehicles)):
                    if gap_between(lane_vehicles[i - 1], lane_vehicles[i]) <= self.MAX_QUEUE_GAP:
                        queue_group.append(lane_vehicles[i])
                    else:
                        break

            if not queue_group:
                queue_lengths[lane_name] = 0.0
                continue

            front = queue_group[0]
            back = queue_group[-1]
            if direction.startswith("N") or direction.startswith("S"):
                queue_lengths[lane_name] = max(0.0, back.get("y2", 0.0) - front.get("y1", 0.0))
            else:
                queue_lengths[lane_name] = max(0.0, back.get("x2", 0.0) - front.get("x1", 0.0))

        return queue_lengths

    def heatmap(self, detections: List[Dict], grid_size: int = 24) -> np.ndarray:
        """
        Generate a 2D heatmap of vehicle density across the frame.

        Args:
            detections: List of detection dicts
            grid_size: Grid resolution (grid_size x grid_size)

        Returns:
            2D numpy array (grid_size, grid_size) with vehicle counts per cell
        """
        heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)

        for det in detections:
            cx = det.get("cx", (det.get("x1", 0) + det.get("x2", 0)) / 2)
            cy = det.get("cy", (det.get("y1", 0) + det.get("y2", 0)) / 2)

            # Clamp to [0, 1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))

            # Map to grid cell
            col = int(cx * grid_size)
            row = int(cy * grid_size)

            # Handle boundary
            col = min(col, grid_size - 1)
            row = min(row, grid_size - 1)

            heatmap[row, col] += 1.0

        # Normalize for visualization (optional smoothing could be added)
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap

    @staticmethod
    def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
        """Check if a point is inside a polygon using ray casting."""
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    @staticmethod
    def _polygon_area(polygon: np.ndarray) -> float:
        """Compute polygon area using the shoelace formula."""
        n = len(polygon)
        if n < 3:
            return 0.0
        area = 0.0
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0
