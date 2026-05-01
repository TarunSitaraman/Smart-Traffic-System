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
MIN_GREEN = 10  # absolute minimum green
MAX_GREEN = 90  # absolute maximum green
MIN_CYCLE = 40  # minimum cycle length
MAX_CYCLE = 180  # maximum cycle length

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

        # --- Flow ratios (critical volume method) ---
        y_ns = pcu_ns / SATURATION_FLOW
        y_ew = pcu_ew / SATURATION_FLOW
        Y = y_ns + y_ew

        # --- Webster's optimal cycle ---
        num_phases = 2  # (N/S) and (E/W) due to conflict
        L = LOST_TIME_PER_PHASE * num_phases
        if Y >= 1.0:
            Y = 0.95
        C_opt = (1.5 * L + 5) / (1.0 - Y)
        C_opt = max(MIN_CYCLE, min(MAX_CYCLE, math.ceil(C_opt)))

        # --- Effective green split (proportional to demand) ---
        effective = C_opt - L
        g_ns = (y_ns / Y) * effective
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

        north = DirectionTiming(
            direction="north",
            pcu=round(pcu_n, 2),
            flow_ratio=round(pcu_n / SATURATION_FLOW, 4),
            green=g_ns,
            density_level=_density_level(pcu_n),
            vehicle_count=lane_data.get("north", {}).get("vehicle_count", 0),
            breakdown=lane_data.get("north", {}).get("breakdown", {}),
        )
        south = DirectionTiming(
            direction="south",
            pcu=round(pcu_s, 2),
            flow_ratio=round(pcu_s / SATURATION_FLOW, 4),
            green=g_ns,
            density_level=_density_level(pcu_s),
            vehicle_count=lane_data.get("south", {}).get("vehicle_count", 0),
            breakdown=lane_data.get("south", {}).get("breakdown", {}),
        )
        east = DirectionTiming(
            direction="east",
            pcu=round(pcu_e, 2),
            flow_ratio=round(pcu_e / SATURATION_FLOW, 4),
            green=g_ew,
            density_level=_density_level(pcu_e),
            vehicle_count=lane_data.get("east", {}).get("vehicle_count", 0),
            breakdown=lane_data.get("east", {}).get("breakdown", {}),
        )
        west = DirectionTiming(
            direction="west",
            pcu=round(pcu_w, 2),
            flow_ratio=round(pcu_w / SATURATION_FLOW, 4),
            green=g_ew,
            density_level=_density_level(pcu_w),
            vehicle_count=lane_data.get("west", {}).get("vehicle_count", 0),
            breakdown=lane_data.get("west", {}).get("breakdown", {}),
        )

        note = (
            f"Webster C_opt = (1.5×{L}+5)/(1−{round(Y, 3)}) = {C_opt}s  |  "
            f"N/S {g_ns}s ({round(y_ns / Y * 100, 1)}%)  E/W {g_ew}s ({round(y_ew / Y * 100, 1)}%)"
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
            north=DirectionTiming("north", 0.0, 0.0, 30, density_level="LOW"),
            south=DirectionTiming("south", 0.0, 0.0, 30, density_level="LOW"),
            east=DirectionTiming("east", 0.0, 0.0, 30, density_level="LOW"),
            west=DirectionTiming("west", 0.0, 0.0, 30, density_level="LOW"),
            cycle_length=66,
            lost_time=8,
            total_pcu=0.0,
            y_total=0.0,
            formula_note="No vehicles detected — using default 30s N/S, 30s E/W.",
        )


class PhaseManager:
    """Manages the active signal phase and state transitions."""

    def __init__(self):
        self.current_phase = "NS"  # "NS" or "EW"
        self.current_state = "GREEN"  # "GREEN", "YELLOW", "RED" (all-red)
        self.phase_start_time = time.time()
        self.last_cycle_result: Optional[CycleResult] = None

    def update(self, cycle_result: CycleResult) -> Dict[str, str]:
        """Update and return the current state for all 4 lanes."""
        self.last_cycle_result = cycle_result
        now = time.time()
        elapsed = now - self.phase_start_time

        # Get durations for current cycle
        g_ns = cycle_result.north.green
        g_ew = cycle_result.east.green

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
                # Switch back to NS
                self.current_phase = "NS"
                self.phase_start_time = now
                self.current_state = "GREEN"

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
        if not self.last_cycle_result:
            return 0
        now = time.time()
        elapsed = now - self.phase_start_time

        if self.current_phase == "NS":
            total = self.last_cycle_result.north.green + YELLOW + ALL_RED
        else:
            total = self.last_cycle_result.east.green + YELLOW + ALL_RED

        return max(0, int(total - elapsed))

    def get_timers(self) -> Dict[str, int]:
        """Return the countdown timer (seconds) for each lane."""
        if not self.last_cycle_result:
            return {"north": 0, "south": 0, "east": 0, "west": 0}

        now = time.time()
        elapsed = now - self.phase_start_time
        g_ns = self.last_cycle_result.north.green
        g_ew = self.last_cycle_result.east.green

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
            north=DirectionTiming("north", 0.0, 0.0, 30, density_level="LOW"),
            south=DirectionTiming("south", 0.0, 0.0, 30, density_level="LOW"),
            east=DirectionTiming("east", 0.0, 0.0, 30, density_level="LOW"),
            west=DirectionTiming("west", 0.0, 0.0, 30, density_level="LOW"),
            cycle_length=66,
            lost_time=8,
            total_pcu=0.0,
            y_total=0.0,
            formula_note="No vehicles detected — using default 30s N/S, 30s E/W.",
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
