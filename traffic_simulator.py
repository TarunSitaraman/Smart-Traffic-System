# =============================================================================
# traffic_simulator.py — Synthetic traffic scenario generator for demos
#
# Generates realistic top-down intersection views with animated vehicles,
# respecting traffic signals and producing synthetic detection data
# =============================================================================

import logging
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("simulator")


@dataclass
class SimulatedVehicle:
    """Represents a vehicle in the simulation."""

    vehicle_id: int
    class_name: str  # 'car', 'truck', 'motorcycle', 'bicycle'
    x: float  # position x (0-1280)
    y: float  # position y (0-720)
    width: float  # vehicle width in pixels
    height: float  # vehicle height in pixels
    lane: str  # 'north', 'south', 'east', 'west'
    velocity: float  # pixels per frame
    stopped: bool = False
    spawn_time: float = 0.0

    def get_bbox_normalized(self) -> Dict:
        """Get bounding box in normalized coordinates."""
        x1 = max(0.0, (self.x - self.width / 2) / 1280)
        y1 = max(0.0, (self.y - self.height / 2) / 720)
        x2 = min(1.0, (self.x + self.width / 2) / 1280)
        y2 = min(1.0, (self.y + self.height / 2) / 720)
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


class TrafficSimulator:
    """Generates synthetic traffic scenarios for testing."""

    VEHICLE_CLASSES = ["car", "truck", "motorcycle"]
    VEHICLE_CLASS_WEIGHTS = [0.72, 0.13, 0.15]
    VEHICLE_SIZES = {
        "car": (26, 48),
        "truck": (32, 80),
        "motorcycle": (18, 32),
    }
    SPAWN_PROFILES = {
        "normal": {
            "rate_range": {
                "north": (0.16, 0.26),
                "south": (0.14, 0.22),
                "east": (0.10, 0.18),
                "west": (0.10, 0.16),
            },
            "lane_cap": 6,
            "min_gap_s": 0.9,
            "spawn_buffer_px": 120,
            "visible_fraction": 0.45,
        },
        "congestion": {
            "rate_range": {
                "north": (0.28, 0.42),
                "south": (0.24, 0.36),
                "east": (0.20, 0.30),
                "west": (0.18, 0.28),
            },
            "lane_cap": 9,
            "min_gap_s": 0.55,
            "spawn_buffer_px": 105,
            "visible_fraction": 0.5,
        },
        "collision": {
            "rate_range": {
                "north": (0.10, 0.16),
                "south": (0.10, 0.16),
                "east": (0.10, 0.16),
                "west": (0.10, 0.16),
            },
            "lane_cap": 5,
            "min_gap_s": 1.0,
            "spawn_buffer_px": 120,
            "visible_fraction": 0.45,
        },
        "stalled": {
            "rate_range": {
                "north": (0.15, 0.24),
                "south": (0.12, 0.20),
                "east": (0.10, 0.18),
                "west": (0.10, 0.16),
            },
            "lane_cap": 6,
            "min_gap_s": 0.9,
            "spawn_buffer_px": 120,
            "visible_fraction": 0.45,
        },
        "emergency": {
            "rate_range": {
                "north": (0.08, 0.14),
                "south": (0.08, 0.14),
                "east": (0.08, 0.14),
                "west": (0.08, 0.14),
            },
            "lane_cap": 4,
            "min_gap_s": 1.1,
            "spawn_buffer_px": 130,
            "visible_fraction": 0.45,
        },
    }

    # Lane configurations: spawn and goal positions
    LANE_CONFIG = {
        "north": {
            "start": (540, -30),
            "stop_y": 260,
            "goal": (540, 750),
            "axis": "y",
            "direction": 1,
        },
        "south": {
            "start": (740, 750),
            "stop_y": 460,
            "goal": (740, -30),
            "axis": "y",
            "direction": -1,
        },
        "east": {
            "start": (1310, 400),
            "stop_x": 860,
            "goal": (-30, 400),
            "axis": "x",
            "direction": -1,
        },
        "west": {
            "start": (-30, 320),
            "stop_x": 420,
            "goal": (1310, 320),
            "axis": "x",
            "direction": 1,
        },
    }

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        scenario: str = "normal",
    ):
        """
        Args:
            width: Canvas width
            height: Canvas height
            fps: Target FPS (affects vehicle speed scaling)
            scenario: 'normal', 'congestion', 'collision', 'stalled', 'emergency'
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.scenario = scenario
        self.frame_count = 0
        self.vehicles: Dict[int, SimulatedVehicle] = {}
        self.next_vehicle_id = 1
        self.last_spawn_time = {lane: 0.0 for lane in self.LANE_CONFIG}
        self.signal_state = {
            "north": "RED",
            "south": "RED",
            "east": "GREEN",
            "west": "GREEN",
        }
        self.frame_times = deque(maxlen=100)

        # Scenario-specific state
        self.collision_triggered = False
        self.stalled_vehicle_id = None
        self.emergency_vehicle_spawned = False

    def next_frame(
        self, signal_state: Optional[Dict[str, str]] = None
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Generate the next frame of simulation.

        Args:
            signal_state: Dict mapping lane -> 'RED' | 'GREEN' | 'YELLOW'

        Returns:
            (frame: BGR image, detections: list of detection dicts)
        """
        start = time.time()

        # Update signal state
        if signal_state:
            self.signal_state = signal_state

        # Create canvas
        canvas = (
            np.ones((self.height, self.width, 3), dtype=np.uint8) * 240
        )  # light gray background

        # Draw intersection
        self._draw_intersection(canvas)
        self._draw_signals(canvas)

        # Spawn vehicles based on scenario
        self._spawn_vehicles()

        # Update vehicle positions
        self._update_vehicles()

        # Handle collision scenario
        if self.scenario == "collision" and not self.collision_triggered:
            self._trigger_collision()

        # Handle stalled vehicle scenario
        if self.scenario == "stalled" and self.stalled_vehicle_id is None:
            self._trigger_stalled()

        # Handle emergency scenario
        if self.scenario == "emergency" and not self.emergency_vehicle_spawned:
            self._spawn_emergency_vehicle()

        # Draw vehicles on canvas
        detections = self._draw_vehicles(canvas)

        # Cleanup
        self._cleanup_vehicles()

        self.frame_count += 1
        elapsed = time.time() - start
        self.frame_times.append(elapsed)

        return canvas, detections

    def _draw_intersection(self, canvas: np.ndarray):
        """Draw a more realistic intersection structure."""
        # Road colors
        road_color = (60, 60, 60)
        marking_color = (200, 200, 200)
        yellow_marking = (0, 200, 220)

        # Draw main roads
        # Horizontal
        cv2.rectangle(canvas, (0, 280), (self.width, 440), road_color, -1)
        # Vertical
        cv2.rectangle(canvas, (440, 0), (840, self.height), road_color, -1)

        # Draw curbs/sidewalks
        sidewalk_color = (180, 180, 180)
        # Top-left
        cv2.rectangle(canvas, (0, 0), (440, 280), sidewalk_color, -1)
        # Top-right
        cv2.rectangle(canvas, (840, 0), (self.width, 280), sidewalk_color, -1)
        # Bottom-left
        cv2.rectangle(canvas, (0, 440), (440, self.height), sidewalk_color, -1)
        # Bottom-right
        cv2.rectangle(canvas, (840, 440), (self.width, self.height), sidewalk_color, -1)

        # Lane markings (dashed white)
        # Horizontal middle
        for x in range(0, self.width, 40):
            cv2.line(canvas, (x, 360), (x + 20, 360), marking_color, 2)
        # Vertical middle
        for y in range(0, self.height, 40):
            cv2.line(canvas, (640, y), (640, y + 20), marking_color, 2)

        # Lane separators (solid yellow)
        # North/South lanes
        cv2.line(canvas, (540, 0), (540, 280), yellow_marking, 2)
        cv2.line(canvas, (740, 0), (740, 280), yellow_marking, 2)
        cv2.line(canvas, (540, 440), (540, self.height), yellow_marking, 2)
        cv2.line(canvas, (740, 440), (740, self.height), yellow_marking, 2)

        # East/West lanes
        cv2.line(canvas, (0, 320), (440, 320), yellow_marking, 2)
        cv2.line(canvas, (0, 400), (440, 400), yellow_marking, 2)
        cv2.line(canvas, (840, 320), (self.width, 320), yellow_marking, 2)
        cv2.line(canvas, (840, 400), (self.width, 400), yellow_marking, 2)

        # Stop lines (thick solid white)
        # North
        cv2.rectangle(canvas, (440, 275), (840, 280), (255, 255, 255), -1)
        # South
        cv2.rectangle(canvas, (440, 440), (840, 445), (255, 255, 255), -1)
        # West
        cv2.rectangle(canvas, (435, 280), (440, 440), (255, 255, 255), -1)
        # East
        cv2.rectangle(canvas, (840, 280), (845, 440), (255, 255, 255), -1)

        # Crosswalks
        for i in range(450, 830, 30):
            # Top crosswalk
            cv2.rectangle(canvas, (i, 240), (i + 15, 270), (240, 240, 240), -1)
            # Bottom crosswalk
            cv2.rectangle(canvas, (i, 450), (i + 15, 480), (240, 240, 240), -1)

        for i in range(290, 430, 30):
            # Left crosswalk
            cv2.rectangle(canvas, (400, i), (430, i + 15), (240, 240, 240), -1)
            # Right crosswalk
            cv2.rectangle(canvas, (850, i), (880, i + 15), (240, 240, 240), -1)

    def _draw_signals(self, canvas: np.ndarray):
        """Draw realistic traffic signal assemblies with labels."""
        # North signal (Incoming from top)
        self._draw_signal_box(
            canvas, (590, 160), self.signal_state.get("north", "RED"), "NORTH"
        )
        # South signal (Incoming from bottom)
        self._draw_signal_box(
            canvas, (650, 480), self.signal_state.get("south", "RED"), "SOUTH"
        )
        # West signal (Incoming from left)
        self._draw_signal_box(
            canvas, (330, 200), self.signal_state.get("west", "RED"), "WEST"
        )
        # East signal (Incoming from right)
        self._draw_signal_box(
            canvas, (910, 430), self.signal_state.get("east", "RED"), "EAST"
        )

    def _draw_signal_box(
        self, canvas: np.ndarray, top_left: Tuple[int, int], state: str, label: str = ""
    ):
        """Draw a 3-light traffic signal box with a label."""
        x, y = top_left
        w, h = 34, 86

        # Box
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (20, 20, 20), -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (100, 100, 100), 1)

        # Label
        if label:
            cv2.putText(
                canvas,
                label,
                (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1,
            )

        # Lights
        # Red
        r_color = (0, 0, 255) if state == "RED" else (0, 0, 80)
        cv2.circle(canvas, (x + w // 2, y + 16), 10, r_color, -1)
        # Yellow
        y_color = (0, 220, 255) if state == "YELLOW" else (0, 80, 100)
        cv2.circle(canvas, (x + w // 2, y + 43), 10, y_color, -1)
        # Green
        g_color = (0, 255, 0) if state == "GREEN" else (0, 80, 0)
        cv2.circle(canvas, (x + w // 2, y + 70), 10, g_color, -1)

    def _state_to_color(self, state: str) -> Tuple[int, int, int]:
        """Convert signal state to BGR color."""
        if state == "GREEN":
            return (0, 255, 0)
        elif state == "YELLOW":
            return (0, 255, 255)
        else:  # RED
            return (0, 0, 255)

    def _spawn_vehicles(self):
        """Spawn vehicles using randomized flow plus queue backpressure."""
        profile = self._scenario_profile()
        now = time.time()

        for lane, rate_range in profile["rate_range"].items():
            if not self._can_spawn_in_lane(lane, now, profile):
                continue

            lane_load = self._approach_vehicle_count(lane)
            load_ratio = lane_load / max(1, profile["lane_cap"])
            spawn_rate = random.uniform(*rate_range) * max(0.2, 1.0 - 0.7 * load_ratio)

            # When a lane is backed up on red, ease off new arrivals so the
            # visible queue reflects what the controller can reasonably clear.
            if self.signal_state.get(lane) != "GREEN" and lane_load >= max(
                2, profile["lane_cap"] // 2
            ):
                spawn_rate *= 0.45

            spawn_probability = min(0.95, spawn_rate / max(1, self.fps))
            if random.random() < spawn_probability:
                self._spawn_vehicle_in_lane(lane)
                self.last_spawn_time[lane] = now

    def _scenario_profile(self) -> Dict:
        return self.SPAWN_PROFILES.get(self.scenario, self.SPAWN_PROFILES["normal"])

    def _can_spawn_in_lane(self, lane: str, now: float, profile: Dict) -> bool:
        if now - self.last_spawn_time.get(lane, 0.0) < profile["min_gap_s"]:
            return False
        if self._approach_vehicle_count(lane) >= profile["lane_cap"]:
            return False

        nearest = self._distance_from_spawn_to_nearest_vehicle(lane)
        if nearest is not None and nearest < profile["spawn_buffer_px"]:
            return False
        return True

    def _distance_from_spawn_to_nearest_vehicle(self, lane: str) -> Optional[float]:
        config = self.LANE_CONFIG[lane]
        axis = config["axis"]
        direction = config["direction"]
        spawn_pos = config["start"][0] if axis == "x" else config["start"][1]
        distances = []

        for vehicle in self.vehicles.values():
            if vehicle.lane != lane:
                continue
            pos = vehicle.x if axis == "x" else vehicle.y
            distance = (pos - spawn_pos) * direction
            if distance >= 0:
                distances.append(distance)

        return min(distances) if distances else None

    def _approach_vehicle_count(self, lane: str) -> int:
        config = self.LANE_CONFIG[lane]
        axis = config["axis"]
        direction = config["direction"]
        stop_pos = config["stop_x"] if axis == "x" else config["stop_y"]
        count = 0

        for vehicle in self.vehicles.values():
            if vehicle.lane != lane:
                continue
            pos = vehicle.x if axis == "x" else vehicle.y
            distance_to_stop = (stop_pos - pos) * direction
            if distance_to_stop > -180:
                count += 1

        return count

    def _spawn_vehicle_in_lane(self, lane: str):
        """Spawn a single vehicle in a lane with increased speed."""
        cls = random.choices(self.VEHICLE_CLASSES, weights=self.VEHICLE_CLASS_WEIGHTS)[
            0
        ]
        width, height = self.VEHICLE_SIZES[cls]
        config = self.LANE_CONFIG[lane]

        vehicle = SimulatedVehicle(
            vehicle_id=self.next_vehicle_id,
            class_name=cls,
            x=float(config["start"][0]),
            y=float(config["start"][1]),
            width=width,
            height=height,
            lane=lane,
            velocity=4.0 + random.uniform(-1.0, 1.0),  # Increased base speed (4px/f)
            spawn_time=time.time(),
        )
        self.vehicles[self.next_vehicle_id] = vehicle
        self.next_vehicle_id += 1

    def _update_vehicles(self):
        """Update vehicle positions with faster acceleration/deceleration."""
        # Sort vehicles in each lane by distance to process them from front to back
        lane_queues = {lane: [] for lane in self.LANE_CONFIG.keys()}
        for v in self.vehicles.values():
            lane_queues[v.lane].append(v)

        for lane, lane_vehicles in lane_queues.items():
            config = self.LANE_CONFIG[lane]
            direction = config["direction"]
            axis = config["axis"]

            # Sort: front-most vehicle first
            if axis == "y":
                lane_vehicles.sort(key=lambda v: v.y * direction, reverse=True)
            else:
                lane_vehicles.sort(key=lambda v: v.x * direction, reverse=True)

            for i, vehicle in enumerate(lane_vehicles):
                stop_pos = config["stop_x"] if axis == "x" else config["stop_y"]

                # 1. Check distance to stop line if signal is RED or YELLOW
                signal = self.signal_state.get(lane, "RED")
                at_stop = False
                if axis == "x":
                    dist_to_stop = (stop_pos - vehicle.x) * direction
                else:
                    dist_to_stop = (stop_pos - vehicle.y) * direction

                # Stop if Red or if Yellow and far enough from stop line to stop safely
                if signal == "RED" and 0 < dist_to_stop < 25:
                    at_stop = True
                elif signal == "YELLOW" and 30 < dist_to_stop < 60:
                    at_stop = True

                # 2. Check distance to vehicle in front
                vehicle_ahead = lane_vehicles[i - 1] if i > 0 else None
                blocked_by_vehicle = False
                if vehicle_ahead:
                    if axis == "x":
                        dist = (vehicle_ahead.x - vehicle.x) * direction
                    else:
                        dist = (vehicle_ahead.y - vehicle.y) * direction

                    safe_dist = vehicle_ahead.height + 20
                    if 0 < dist < safe_dist:
                        blocked_by_vehicle = True

                # 3. Decision logic
                if (signal == "RED" and at_stop) or blocked_by_vehicle:
                    vehicle.stopped = True
                    vehicle.velocity = max(0, vehicle.velocity - 0.5)  # Fast braking
                elif signal == "GREEN" or not at_stop:
                    if not blocked_by_vehicle:
                        vehicle.stopped = False
                        # Gradual acceleration
                        target_vel = 4.0 + random.uniform(-0.5, 0.5)
                        vehicle.velocity = min(target_vel, vehicle.velocity + 0.3)

                if signal == "YELLOW" and not blocked_by_vehicle and not at_stop:
                    vehicle.velocity = max(1.5, vehicle.velocity * 0.95)

                # Update position
                if vehicle.velocity > 0:
                    if axis == "x":
                        vehicle.x += vehicle.velocity * direction
                    else:
                        vehicle.y += vehicle.velocity * direction

    def _draw_vehicles(self, canvas: np.ndarray) -> List[Dict]:
        """Draw vehicles with more detail and return detections."""
        detections = []
        min_visible_fraction = self._scenario_profile()["visible_fraction"]

        for v_id, vehicle in self.vehicles.items():
            # Calculate corners
            if vehicle.lane in ["north", "south"]:
                w, h = vehicle.width, vehicle.height
            else:
                w, h = vehicle.height, vehicle.width  # Rotate for horizontal lanes

            x1 = int(vehicle.x - w / 2)
            y1 = int(vehicle.y - h / 2)
            x2 = int(vehicle.x + w / 2)
            y2 = int(vehicle.y + h / 2)

            vis_x1 = max(0, x1)
            vis_y1 = max(0, y1)
            vis_x2 = min(self.width, x2)
            vis_y2 = min(self.height, y2)
            if vis_x1 >= vis_x2 or vis_y1 >= vis_y2:
                continue

            bbox_area = max(1, (x2 - x1) * (y2 - y1))
            visible_area = (vis_x2 - vis_x1) * (vis_y2 - vis_y1)
            if (visible_area / bbox_area) < min_visible_fraction:
                continue

            # Color by class
            base_colors = {
                "car": (180, 100, 40),  # Blueish
                "truck": (40, 100, 180),  # Oranges
                "motorcycle": (150, 50, 150),  # Purple
                "emergency": (255, 255, 255),  # White
            }
            color = base_colors.get(vehicle.class_name, (128, 128, 128))

            # Draw body
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (20, 20, 20), 1)

            # Add "windshield" for directionality
            ws_color = (240, 240, 240)
            if vehicle.lane == "north":  # Moving down
                cv2.rectangle(
                    canvas, (x1 + 2, y1 + h - 8), (x2 - 2, y1 + h - 4), ws_color, -1
                )
            elif vehicle.lane == "south":  # Moving up
                cv2.rectangle(canvas, (x1 + 2, y1 + 4), (x2 - 2, y1 + 8), ws_color, -1)
            elif vehicle.lane == "west":  # Moving right
                cv2.rectangle(
                    canvas, (x1 + w - 8, y1 + 2), (x1 + w - 4, y2 - 2), ws_color, -1
                )
            elif vehicle.lane == "east":  # Moving left
                cv2.rectangle(canvas, (x1 + 4, y1 + 2), (x1 + 8, y2 - 2), ws_color, -1)

            # Special treatment for emergency vehicle (red/blue lights)
            if vehicle.class_name == "emergency":
                if (self.frame_count // 5) % 2 == 0:
                    cv2.circle(canvas, (x1 + 5, y1 + 5), 3, (255, 0, 0), -1)
                    cv2.circle(canvas, (x2 - 5, y2 - 5), 3, (0, 0, 255), -1)
                else:
                    cv2.circle(canvas, (x1 + 5, y1 + 5), 3, (0, 0, 255), -1)
                    cv2.circle(canvas, (x2 - 5, y2 - 5), 3, (255, 0, 0), -1)

            # Add detection
            det = {
                "class_name": vehicle.class_name,
                "confidence": 0.98,
                "x1": vis_x1 / self.width,
                "y1": vis_y1 / self.height,
                "x2": vis_x2 / self.width,
                "y2": vis_y2 / self.height,
            }
            detections.append(det)

        # Draw a small legend in the corner
        self._draw_legend(canvas)

        return detections

    def _draw_legend(self, canvas: np.ndarray):
        """Draw a legend for vehicle types."""
        overlay = canvas.copy()
        cv2.rectangle(overlay, (10, 10), (180, 130), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)

        cv2.putText(
            canvas,
            "Vehicle Types",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        types = [
            ("Car", (180, 100, 40)),
            ("Truck", (40, 100, 180)),
            ("Motorcycle", (150, 50, 150)),
            ("Emergency", (255, 255, 255)),
        ]

        for i, (name, color) in enumerate(types):
            y = 55 + i * 20
            cv2.rectangle(canvas, (20, y - 10), (40, y), color, -1)
            cv2.putText(
                canvas,
                name,
                (50, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1,
            )

    def _cleanup_vehicles(self):
        """Remove vehicles that have exited the visible canvas."""
        to_remove = []
        margin = 100

        for v_id, vehicle in self.vehicles.items():
            if (
                vehicle.x < -margin
                or vehicle.x > self.width + margin
                or vehicle.y < -margin
                or vehicle.y > self.height + margin
            ):
                to_remove.append(v_id)

        for v_id in to_remove:
            del self.vehicles[v_id]

    def _trigger_collision(self):
        """Trigger a collision in the collision scenario."""
        if len(self.vehicles) < 2:
            return

        # Find two vehicles approaching from different directions in the same area
        north_vehicles = [v for v in self.vehicles.values() if v.lane == "north"]
        east_vehicles = [v for v in self.vehicles.values() if v.lane == "east"]

        if north_vehicles and east_vehicles:
            v1 = north_vehicles[0]
            v2 = east_vehicles[0]

            # Force them to collide at the intersection center (640, 360)
            v1.x = 635
            v1.y = 355
            v2.x = 645
            v2.y = 365

            v1.stopped = True
            v2.stopped = True
            self.collision_triggered = True
            logger.info("Collision triggered at intersection center")

    def _trigger_stalled(self):
        """Trigger a stalled vehicle."""
        if self.vehicles:
            stalled = list(self.vehicles.values())[0]
            stalled.stopped = True
            stalled.velocity = 0.0
            self.stalled_vehicle_id = stalled.vehicle_id
            logger.info(
                "Stalled vehicle: %d in lane %s", stalled.vehicle_id, stalled.lane
            )

    def _spawn_emergency_vehicle(self):
        """Spawn an ambulance to test emergency response."""
        vehicle = SimulatedVehicle(
            vehicle_id=self.next_vehicle_id,
            class_name="emergency",
            x=540,
            y=-30,
            width=50,
            height=30,
            lane="north",
            velocity=3.0,
            spawn_time=time.time(),
        )
        self.vehicles[self.next_vehicle_id] = vehicle
        self.next_vehicle_id += 1
        self.emergency_vehicle_spawned = True
        logger.info("Emergency vehicle spawned")

    def run(self, frame_buffer=None) -> None:
        """
        Main loop for the simulator (compatible with YOLOv8Detector interface).
        If frame_buffer is provided, puts DetectionResult into it.
        Otherwise, just generates frames indefinitely.
        """
        import threading

        from detector import DetectionResult

        logger.info("Traffic simulator started (scenario: %s)", self.scenario)

        # For use as a detector replacement, we need a stop event
        if not hasattr(self, "_stop_evt"):
            self._stop_evt = threading.Event()

        frame_id = 0
        while not self._stop_evt.is_set():
            start = time.time()
            canvas, detections = self.next_frame(self.signal_state)
            inference_ms = (time.time() - start) * 1000

            if frame_buffer is not None:
                result = DetectionResult(
                    frame_id=frame_id,
                    frame=canvas,
                    raw_frame=canvas.copy(),
                    detections=detections,
                    inference_ms=inference_ms,
                )
                frame_buffer.put(result)

            frame_id += 1

            # Maintain FPS
            elapsed = time.time() - start
            target_frame_time = 1.0 / self.fps
            if elapsed < target_frame_time:
                time.sleep(target_frame_time - elapsed)

    def stop(self) -> None:
        """Stop the simulator loop."""
        if hasattr(self, "_stop_evt"):
            self._stop_evt.set()
        logger.info("Traffic simulator stopped")
