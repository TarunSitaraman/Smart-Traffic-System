# =============================================================================
# traffic_simulator.py — Synthetic traffic scenario generator for demos
#
# Generates realistic top-down intersection views with animated vehicles,
# respecting traffic signals and producing synthetic detection data
# =============================================================================

import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
import random
from collections import deque
import time

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
    VEHICLE_SIZES = {
        "car": (40, 25),
        "truck": (60, 30),
        "motorcycle": (25, 15),
    }

    # Lane configurations: spawn and goal positions
    LANE_CONFIG = {
        "north": {"start": (640, -30), "stop_y": 340, "goal": (640, 720), "axis": "y", "direction": 1},
        "south": {"start": (640, 750), "stop_y": 380, "goal": (640, -30), "axis": "y", "direction": -1},
        "east": {"start": (1310, 360), "stop_x": 940, "goal": (-30, 360), "axis": "x", "direction": -1},
        "west": {"start": (-30, 360), "stop_x": 340, "goal": (1310, 360), "axis": "x", "direction": 1},
    }

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30, scenario: str = "normal"):
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
        self.last_spawn_time = {}
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

    def next_frame(self, signal_state: Optional[Dict[str, str]] = None) -> Tuple[np.ndarray, List[Dict]]:
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
        canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 240  # light gray background

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
        """Draw the intersection structure (roads, lanes, markings)."""
        # Horizontal road
        cv2.rectangle(canvas, (0, 300), (self.width, 420), (200, 200, 200), -1)

        # Vertical road
        cv2.rectangle(canvas, (460, 0), (820, self.height), (200, 200, 200), -1)

        # Lane markings (dashed white lines)
        for y in [330, 390]:
            for x in range(0, self.width, 40):
                cv2.line(canvas, (x, y), (x + 20, y), (255, 255, 255), 2)

        for x in [540, 740]:
            for y in range(0, self.height, 40):
                cv2.line(canvas, (x, y), (x, y + 20), (255, 255, 255), 2)

        # Stop lines (solid yellow)
        cv2.line(canvas, (460, 340), (820, 340), (0, 255, 255), 3)  # north stop
        cv2.line(canvas, (460, 380), (820, 380), (0, 255, 255), 3)  # south stop
        cv2.line(canvas, (540, 0), (540, 720), (0, 255, 255), 3)   # west stop
        cv2.line(canvas, (740, 0), (740, 720), (0, 255, 255), 3)   # east stop

    def _draw_signals(self, canvas: np.ndarray):
        """Draw traffic signal lights."""
        signal_positions = {
            "north": (420, 280),
            "south": (420, 500),
            "west": (380, 320),
            "east": (900, 320),
        }

        for lane, pos in signal_positions.items():
            color = self._state_to_color(self.signal_state.get(lane, "RED"))
            cv2.circle(canvas, pos, 15, color, -1)
            cv2.circle(canvas, pos, 15, (0, 0, 0), 2)

    def _state_to_color(self, state: str) -> Tuple[int, int, int]:
        """Convert signal state to BGR color."""
        if state == "GREEN":
            return (0, 255, 0)
        elif state == "YELLOW":
            return (0, 255, 255)
        else:  # RED
            return (0, 0, 255)

    def _spawn_vehicles(self):
        """Spawn new vehicles based on arrival rates."""
        now = time.time()

        # Define arrival rates per scenario
        rates = {
            "normal": {"north": 3.0, "south": 2.5, "east": 2.0, "west": 1.5},
            "congestion": {"north": 8.0, "south": 7.0, "east": 6.0, "west": 5.0},
            "collision": {"north": 2.0, "south": 2.0, "east": 2.0, "west": 2.0},
            "stalled": {"north": 3.0, "south": 2.5, "east": 2.0, "west": 1.5},
            "emergency": {"north": 2.0, "south": 2.0, "east": 2.0, "west": 2.0},
        }

        target_rates = rates.get(self.scenario, rates["normal"])

        for lane, rate in target_rates.items():
            last_spawn = self.last_spawn_time.get(lane, 0)
            spawn_interval = 1.0 / rate
            if now - last_spawn > spawn_interval:
                self._spawn_vehicle_in_lane(lane)
                self.last_spawn_time[lane] = now

    def _spawn_vehicle_in_lane(self, lane: str):
        """Spawn a single vehicle in a lane."""
        cls = random.choice(self.VEHICLE_CLASSES)
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
            velocity=2.0 + random.uniform(-0.5, 0.5),  # 2 px/frame ≈ 20 km/h
            spawn_time=time.time(),
        )
        self.vehicles[self.next_vehicle_id] = vehicle
        self.next_vehicle_id += 1

    def _update_vehicles(self):
        """Update vehicle positions, respecting signals and stop lines."""
        for v_id, vehicle in list(self.vehicles.items()):
            config = self.LANE_CONFIG[vehicle.lane]
            axis = config["axis"]
            direction = config["direction"]
            stop_pos = config["stop_x"] if axis == "x" else config["stop_y"]

            # Check if approaching stop line
            if axis == "x":
                approaching_stop = (
                    (direction > 0 and vehicle.x > stop_pos - 50 and vehicle.x < stop_pos) or
                    (direction < 0 and vehicle.x < stop_pos + 50 and vehicle.x > stop_pos)
                )
                at_stop = abs(vehicle.x - stop_pos) < 10
            else:
                approaching_stop = (
                    (direction > 0 and vehicle.y > stop_pos - 50 and vehicle.y < stop_pos) or
                    (direction < 0 and vehicle.y < stop_pos + 50 and vehicle.y > stop_pos)
                )
                at_stop = abs(vehicle.y - stop_pos) < 10

            # Respect signal
            signal = self.signal_state.get(vehicle.lane, "RED")
            if signal == "RED" and at_stop:
                vehicle.stopped = True
                vehicle.velocity = 0.0
            elif signal == "GREEN" and vehicle.stopped:
                vehicle.stopped = False
                vehicle.velocity = 2.0 + random.uniform(-0.5, 0.5)
            elif signal == "YELLOW":
                vehicle.velocity = max(0.5, vehicle.velocity * 0.8)

            # Update position
            if not vehicle.stopped:
                if axis == "x":
                    vehicle.x += vehicle.velocity * direction
                else:
                    vehicle.y += vehicle.velocity * direction

    def _draw_vehicles(self, canvas: np.ndarray) -> List[Dict]:
        """Draw vehicles on canvas and return detections."""
        detections = []

        for v_id, vehicle in self.vehicles.items():
            x1 = int(vehicle.x - vehicle.width / 2)
            y1 = int(vehicle.y - vehicle.height / 2)
            x2 = int(vehicle.x + vehicle.width / 2)
            y2 = int(vehicle.y + vehicle.height / 2)

            # Clamp to canvas
            x1 = max(0, min(x1, self.width - 1))
            y1 = max(0, min(y1, self.height - 1))
            x2 = max(0, min(x2, self.width - 1))
            y2 = max(0, min(y2, self.height - 1))

            # Color by class
            color = {"car": (0, 0, 200), "truck": (0, 100, 200), "motorcycle": (200, 100, 0)}
            color = color.get(vehicle.class_name, (128, 128, 128))

            # Draw rectangle
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 0), 1)

            # Add detection
            det = {
                "class_name": vehicle.class_name,
                "confidence": 0.98,
                "x1": x1 / self.width,
                "y1": y1 / self.height,
                "x2": x2 / self.width,
                "y2": y2 / self.height,
            }
            detections.append(det)

        return detections

    def _cleanup_vehicles(self):
        """Remove vehicles that have exited the frame."""
        margin = 100
        to_remove = []

        for v_id, vehicle in self.vehicles.items():
            config = self.LANE_CONFIG[vehicle.lane]
            goal_x, goal_y = config["goal"]

            if vehicle.lane in ["north", "south"]:
                if abs(vehicle.y - goal_y) < margin:
                    to_remove.append(v_id)
            else:
                if abs(vehicle.x - goal_x) < margin:
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

            # Force them to collide at the intersection center
            v1.x = 640
            v1.y = 350
            v2.x = 640
            v2.y = 360

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
            logger.info("Stalled vehicle: %d in lane %s", stalled.vehicle_id, stalled.lane)

    def _spawn_emergency_vehicle(self):
        """Spawn an ambulance to test emergency response."""
        vehicle = SimulatedVehicle(
            vehicle_id=self.next_vehicle_id,
            class_name="emergency",
            x=640,
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
        if not hasattr(self, '_stop_evt'):
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
        if hasattr(self, '_stop_evt'):
            self._stop_evt.set()
        logger.info("Traffic simulator stopped")
