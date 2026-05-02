# =============================================================================
# signal_controller.py — Webster-based Adaptive Traffic Signal Controller (4-way)
#
# Adaptive signal control for 4-direction intersection using Webster's method.
# Phases: (North+South) vs (East+West) due to physical conflict.
#
# Key concepts:
#  - PCU (Passenger Car Unit): Normalized weight for vehicle type
#  - Flow ratio: PCU demand / saturation flow
#  - Webster's cycle: C_opt = (1.5×L + 5) / (1−Y)
#  - Conflicting phases: N/S cannot both be green with E/W simultaneously
# =============================================================================

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# PCU (Passenger Car Unit) weights
# ---------------------------------------------------------------------------

PCU_WEIGHTS: Dict[str, float] = {
    "car": 1.0,
    "van": 1.5,
    "truck": 2.5,
    "bus": 2.5,
    "motorcycle": 0.5,
    "bicycle": 0.3,
    "person": 0.2,
}

# ---------------------------------------------------------------------------
# Timing constants (all in seconds)
# ---------------------------------------------------------------------------

SATURATION_FLOW = 20.0  # max PCU/s that can pass in ideal conditions
LOST_TIME_PER_PHASE = 4  # startup delay + clearance lost time per phase
YELLOW = 3  # yellow/amber interval
ALL_RED = 1  # all-red clearance between phases
MIN_GREEN = 8  # absolute minimum green
MAX_GREEN = 20  # demo cap for adaptive green
MIN_CYCLE = 40  # minimum cycle length
MAX_CYCLE = 180  # maximum cycle length
DENSITY_REFERENCE = 8.0  # demo-friendly "high" lane density on a 0-10 scale
QUEUE_REFERENCE = 0.3  # normalized queue length considered significant
PHASE_PRESSURE_ALPHA = 0.35  # smoothing factor for changing demand
PCU_PRESSURE_WEIGHT = 0.5
DENSITY_PRESSURE_WEIGHT = 0.3
QUEUE_PRESSURE_WEIGHT = 0.2
PRESSURE_TO_CYCLE_GAIN = 0.18
PRESSURE_TO_SPLIT_GAIN = 0.35

# Congestion thresholds (PCU)
LEVEL_HIGH_THRESH = 10.0
LEVEL_MEDIUM_THRESH = 4.0


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DirectionTiming:
    direction: str  # 'north', 'south', 'east', 'west'
    pcu: float  # effective PCU count
    flow_ratio: float  # y_i = pcu / saturation_flow
    green: int  # suggested green seconds
    yellow: int = YELLOW
    all_red: int = ALL_RED
    density_level: str = "LOW"  # LOW / MEDIUM / HIGH
    vehicle_count: int = 0  # raw vehicle detections used
    state: str = "RED"  # RED / GREEN / YELLOW
    density: float = 0.0
    queue_length: float = 0.0
    demand_score: float = 0.0
    breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class CycleResult:
    north: DirectionTiming
    south: DirectionTiming
    east: DirectionTiming
    west: DirectionTiming
    cycle_length: int
    lost_time: int
    total_pcu: float
    y_total: float
    formula_note: str


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class TrafficSignalController:
    """Computes adaptive 4-direction signal timings using Webster's method."""

    def __init__(self) -> None:
        self._phase_pressure = {"NS": 0.0, "EW": 0.0}

    def compute(self, lane_data: Dict[str, Dict]) -> CycleResult:
        """
        Compute optimal cycle and green times for 4-way intersection.

        Args:
            lane_data: Dict with keys 'north', 'south', 'east', 'west'
                      Each contains: pcu (float), vehicle_count (int), breakdown (Dict)

        Returns:
            CycleResult with timings for all 4 directions
        """
        pcu_ns = lane_data.get("north", {}).get("pcu", 0.0) + lane_data.get(
            "south", {}
        ).get("pcu", 0.0)
        pcu_ew = lane_data.get("east", {}).get("pcu", 0.0) + lane_data.get(
            "west", {}
        ).get("pcu", 0.0)
        total_pcu = pcu_ns + pcu_ew

        if total_pcu < 0.1:
            return self._default_result()

        lane_scores = {
            lane: _lane_pressure_score(lane_data.get(lane, {}))
            for lane in ("north", "south", "east", "west")
        }

        phase_pressure_raw = {
            "NS": (lane_scores["north"] + lane_scores["south"]) / 2.0,
            "EW": (lane_scores["east"] + lane_scores["west"]) / 2.0,
        }
        for phase, raw_score in phase_pressure_raw.items():
            previous = self._phase_pressure.get(phase, raw_score)
            self._phase_pressure[phase] = (
                (1.0 - PHASE_PRESSURE_ALPHA) * previous
                + PHASE_PRESSURE_ALPHA * raw_score
            )

        # --- Base flow ratios (critical volume method) ---
        y_ns_base = pcu_ns / SATURATION_FLOW
        y_ew_base = pcu_ew / SATURATION_FLOW
        phase_pressure_ns = y_ns_base + PRESSURE_TO_SPLIT_GAIN * self._phase_pressure["NS"]
        phase_pressure_ew = y_ew_base + PRESSURE_TO_SPLIT_GAIN * self._phase_pressure["EW"]
        Y = (
            y_ns_base
            + y_ew_base
            + PRESSURE_TO_CYCLE_GAIN
            * (self._phase_pressure["NS"] + self._phase_pressure["EW"])
        )

        # --- Webster's optimal cycle ---
        num_phases = 2  # (N/S) and (E/W) due to conflict
        L = LOST_TIME_PER_PHASE * num_phases
        if Y >= 1.0:
            Y = 0.95
        C_opt = (1.5 * L + 5) / (1.0 - Y)
        C_opt = max(MIN_CYCLE, min(MAX_CYCLE, math.ceil(C_opt)))

        # --- Effective green split (proportional to phase pressure) ---
        effective = C_opt - L
        total_phase_pressure = max(0.001, phase_pressure_ns + phase_pressure_ew)
        g_ns = (phase_pressure_ns / total_phase_pressure) * effective
        g_ew = effective - g_ns

        g_ns = max(MIN_GREEN, min(MAX_GREEN, round(g_ns)))
        g_ew = max(MIN_GREEN, min(MAX_GREEN, round(g_ew)))

        # Recalculate actual cycle
        cycle = g_ns + YELLOW + ALL_RED + g_ew + YELLOW + ALL_RED

        # Extract per-lane PCU from lane_data
        pcu_n = lane_data.get("north", {}).get("pcu", 0.0)
        pcu_s = lane_data.get("south", {}).get("pcu", 0.0)
        pcu_e = lane_data.get("east", {}).get("pcu", 0.0)
        pcu_w = lane_data.get("west", {}).get("pcu", 0.0)
        density_n = lane_data.get("north", {}).get("density", 0.0)
        density_s = lane_data.get("south", {}).get("density", 0.0)
        density_e = lane_data.get("east", {}).get("density", 0.0)
        density_w = lane_data.get("west", {}).get("density", 0.0)
        queue_n = lane_data.get("north", {}).get("queue_length", 0.0)
        queue_s = lane_data.get("south", {}).get("queue_length", 0.0)
        queue_e = lane_data.get("east", {}).get("queue_length", 0.0)
        queue_w = lane_data.get("west", {}).get("queue_length", 0.0)

        north = DirectionTiming(
            direction="north",
            pcu=round(pcu_n, 2),
            flow_ratio=round(pcu_n / SATURATION_FLOW, 4),
            green=g_ns,
            density_level=_demand_level(lane_scores["north"]),
            vehicle_count=lane_data.get("north", {}).get("vehicle_count", 0),
            density=round(density_n, 2),
            queue_length=round(queue_n, 3),
            demand_score=round(lane_scores["north"], 3),
            breakdown=lane_data.get("north", {}).get("breakdown", {}),
        )
        south = DirectionTiming(
            direction="south",
            pcu=round(pcu_s, 2),
            flow_ratio=round(pcu_s / SATURATION_FLOW, 4),
            green=g_ns,
            density_level=_demand_level(lane_scores["south"]),
            vehicle_count=lane_data.get("south", {}).get("vehicle_count", 0),
            density=round(density_s, 2),
            queue_length=round(queue_s, 3),
            demand_score=round(lane_scores["south"], 3),
            breakdown=lane_data.get("south", {}).get("breakdown", {}),
        )
        east = DirectionTiming(
            direction="east",
            pcu=round(pcu_e, 2),
            flow_ratio=round(pcu_e / SATURATION_FLOW, 4),
            green=g_ew,
            density_level=_demand_level(lane_scores["east"]),
            vehicle_count=lane_data.get("east", {}).get("vehicle_count", 0),
            density=round(density_e, 2),
            queue_length=round(queue_e, 3),
            demand_score=round(lane_scores["east"], 3),
            breakdown=lane_data.get("east", {}).get("breakdown", {}),
        )
        west = DirectionTiming(
            direction="west",
            pcu=round(pcu_w, 2),
            flow_ratio=round(pcu_w / SATURATION_FLOW, 4),
            green=g_ew,
            density_level=_demand_level(lane_scores["west"]),
            vehicle_count=lane_data.get("west", {}).get("vehicle_count", 0),
            density=round(density_w, 2),
            queue_length=round(queue_w, 3),
            demand_score=round(lane_scores["west"], 3),
            breakdown=lane_data.get("west", {}).get("breakdown", {}),
        )

        note = (
            f"Webster C_opt = (1.5×{L}+5)/(1−{round(Y, 3)}) = {C_opt}s  |  "
            f"phase pressure NS={self._phase_pressure['NS']:.2f} EW={self._phase_pressure['EW']:.2f}  |  "
            f"N/S {g_ns}s  E/W {g_ew}s"
        )

        return CycleResult(
            north=north,
            south=south,
            east=east,
            west=west,
            cycle_length=cycle,
            lost_time=L,
            total_pcu=round(total_pcu, 2),
            y_total=round(Y, 4),
            formula_note=note,
        )

    def _default_result(self) -> CycleResult:
        return CycleResult(
            north=DirectionTiming("north", 0.0, 0.0, 12, density_level="LOW"),
            south=DirectionTiming("south", 0.0, 0.0, 12, density_level="LOW"),
            east=DirectionTiming("east", 0.0, 0.0, 12, density_level="LOW"),
            west=DirectionTiming("west", 0.0, 0.0, 12, density_level="LOW"),
            cycle_length=32,
            lost_time=8,
            total_pcu=0.0,
            y_total=0.0,
            formula_note="No vehicles detected — using demo baseline 12s N/S, 12s E/W.",
        )


class PhaseManager:
    """Manages the active signal phase and state transitions."""

    def __init__(self):
        self.current_phase = "NS"  # "NS" or "EW"
        self.current_state = "GREEN"  # "GREEN", "YELLOW", "RED" (all-red)
        self.phase_start_time = time.time()
        self.active_cycle_result: Optional[CycleResult] = None

    def update(self, cycle_result: CycleResult) -> Dict[str, str]:
        """Update and return the current state for all 4 lanes."""
        if self.active_cycle_result is None:
            self.active_cycle_result = cycle_result

        now = time.time()
        elapsed = now - self.phase_start_time

        # Lock timings for the current cycle so the countdown does not jump
        # around as fresh detections arrive every frame.
        g_ns = self.active_cycle_result.north.green
        g_ew = self.active_cycle_result.east.green

        # Phase sequence for one axis: GREEN -> YELLOW -> ALL_RED
        if self.current_phase == "NS":
            if elapsed < g_ns:
                self.current_state = "GREEN"
            elif elapsed < g_ns + YELLOW:
                self.current_state = "YELLOW"
            elif elapsed < g_ns + YELLOW + ALL_RED:
                self.current_state = "RED"
            else:
                # Switch to EW
                self.current_phase = "EW"
                self.phase_start_time = now
                self.current_state = "GREEN"
        else:  # EW
            if elapsed < g_ew:
                self.current_state = "GREEN"
            elif elapsed < g_ew + YELLOW:
                self.current_state = "YELLOW"
            elif elapsed < g_ew + YELLOW + ALL_RED:
                self.current_state = "RED"
            else:
                # Switch back to NS and adopt the newest computed timings
                # for the next full cycle.
                self.current_phase = "NS"
                self.phase_start_time = now
                self.current_state = "GREEN"
                self.active_cycle_result = cycle_result

        # Build final state map
        states = {"north": "RED", "south": "RED", "east": "RED", "west": "RED"}
        if self.current_phase == "NS":
            states["north"] = self.current_state
            states["south"] = self.current_state
        else:
            states["east"] = self.current_state
            states["west"] = self.current_state

        return states

    @property
    def remaining_seconds(self) -> int:
        if not self.active_cycle_result:
            return 0
        now = time.time()
        elapsed = now - self.phase_start_time

        if self.current_phase == "NS":
            total = self.active_cycle_result.north.green + YELLOW + ALL_RED
        else:
            total = self.active_cycle_result.east.green + YELLOW + ALL_RED

        return max(0, int(total - elapsed))

    def get_timers(self) -> Dict[str, int]:
        """Return the countdown timer (seconds) for each lane."""
        if not self.active_cycle_result:
            return {"north": 0, "south": 0, "east": 0, "west": 0}

        now = time.time()
        elapsed = now - self.phase_start_time
        g_ns = self.active_cycle_result.north.green
        g_ew = self.active_cycle_result.east.green

        timers = {}
        if self.current_phase == "NS":
            # NS is active: count down to next transition (Yellow or Red)
            if elapsed < g_ns:
                ns_t = int(g_ns - elapsed)
            elif elapsed < g_ns + YELLOW:
                ns_t = int(g_ns + YELLOW - elapsed)
            else:
                # All-Red: count down to Green (after EW phase)
                ns_t = int(
                    (g_ns + YELLOW + ALL_RED - elapsed) + g_ew + YELLOW + ALL_RED
                )

            timers["north"] = timers["south"] = max(0, ns_t)
            # EW is waiting for NS phase to end
            timers["east"] = timers["west"] = max(
                0, int(g_ns + YELLOW + ALL_RED - elapsed)
            )
        else:
            # EW is active
            if elapsed < g_ew:
                ew_t = int(g_ew - elapsed)
            elif elapsed < g_ew + YELLOW:
                ew_t = int(g_ew + YELLOW - elapsed)
            else:
                # All-Red: count down to Green (after NS phase)
                ew_t = int(
                    (g_ew + YELLOW + ALL_RED - elapsed) + g_ns + YELLOW + ALL_RED
                )

            timers["east"] = timers["west"] = max(0, ew_t)
            # NS is waiting for EW phase to end
            timers["north"] = timers["south"] = max(
                0, int(g_ew + YELLOW + ALL_RED - elapsed)
            )

        return timers

    def _default_result(self) -> CycleResult:
        return CycleResult(
            north=DirectionTiming("north", 0.0, 0.0, 12, density_level="LOW"),
            south=DirectionTiming("south", 0.0, 0.0, 12, density_level="LOW"),
            east=DirectionTiming("east", 0.0, 0.0, 12, density_level="LOW"),
            west=DirectionTiming("west", 0.0, 0.0, 12, density_level="LOW"),
            cycle_length=32,
            lost_time=8,
            total_pcu=0.0,
            y_total=0.0,
            formula_note="No vehicles detected — using demo baseline 12s N/S, 12s E/W.",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _density_level(pcu: float) -> str:
    if pcu >= LEVEL_HIGH_THRESH:
        return "HIGH"
    if pcu >= LEVEL_MEDIUM_THRESH:
        return "MEDIUM"
    return "LOW"


def _demand_level(score: float) -> str:
    if score >= 0.7:
        return "HIGH"
    if score >= 0.4:
        return "MEDIUM"
    return "LOW"


def _lane_pressure_score(lane_metrics: Dict) -> float:
    pcu = float(lane_metrics.get("pcu", 0.0))
    density = float(lane_metrics.get("density", 0.0))
    queue_length = float(lane_metrics.get("queue_length", 0.0))

    pcu_score = _clamp(pcu / LEVEL_HIGH_THRESH)
    density_score = _clamp(density / DENSITY_REFERENCE)
    queue_score = _clamp(queue_length / QUEUE_REFERENCE)

    return (
        PCU_PRESSURE_WEIGHT * pcu_score
        + DENSITY_PRESSURE_WEIGHT * density_score
        + QUEUE_PRESSURE_WEIGHT * queue_score
    )


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))
