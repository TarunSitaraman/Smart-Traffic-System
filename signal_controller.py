# =============================================================================
# signal_controller.py — Webster-based Adaptive Traffic Signal Controller
#
# Mathematical model:
#
#  1. VEHICLE CLASSIFICATION → Passenger Car Units (PCU)
#       Each detected class is converted to a PCU weight that reflects
#       how much road space / green time it demands:
#           Car / Van   = 1.0 PCU
#           Truck / Bus = 2.5 PCU  (occupy ~2.5× more space)
#           Motorcycle  = 0.5 PCU
#           Bicycle     = 0.3 PCU
#           Person      = 0.2 PCU  (pedestrian crossing demand)
#
#  2. SPATIAL LANE SPLIT
#       The camera frame is divided into two vertical halves:
#           Left  half (x_centre < 0.5)  → Main Road  (Direction A)
#           Right half (x_centre ≥ 0.5)  → Cross Road (Direction B)
#
#  3. PROXIMITY WEIGHTING
#       Vehicles lower in the frame (higher y_centre) are closer to the
#       stop-line and therefore have higher priority:
#           proximity_weight = 0.5 + 0.5 × y_centre   ∈ [0.5, 1.0]
#       Effective PCU = base_PCU × proximity_weight
#
#  4. DEMAND FLOW RATIO (critical volume method)
#       y_i = PCU_i / SATURATION_FLOW
#       Y   = y_A + y_B          (sum of critical flow ratios)
#
#  5. WEBSTER'S OPTIMAL CYCLE LENGTH
#       C_opt = (1.5 × L + 5) / (1 − Y)
#       where L = total lost time per cycle = LOST_TIME × num_phases
#       Clamped to [MIN_CYCLE, MAX_CYCLE].
#
#  6. EFFECTIVE GREEN ALLOCATION (proportional to flow ratio)
#       g_i = (y_i / Y) × (C_opt − L)
#       Clamped to [MIN_GREEN, MAX_GREEN].
#
#  7. FINAL PHASE BREAKDOWN (returned per direction)
#       Phase = green + yellow + all-red clearance
# =============================================================================

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# PCU (Passenger Car Unit) weights
# ---------------------------------------------------------------------------

PCU_WEIGHTS: Dict[str, float] = {
    "car":        1.0,
    "van":        1.5,
    "truck":      2.5,
    "bus":        2.5,
    "motorcycle": 0.5,
    "bicycle":    0.3,
    "person":     0.2,
}

# ---------------------------------------------------------------------------
# Timing constants (all in seconds)
# ---------------------------------------------------------------------------

SATURATION_FLOW   = 20.0   # max PCU/s that can pass in ideal conditions (normalised)
LOST_TIME_PER_PHASE = 4    # startup delay + clearance lost time per phase
YELLOW            = 3      # yellow/amber interval
ALL_RED           = 1      # all-red clearance between phases
MIN_GREEN         = 10     # absolute minimum green
MAX_GREEN         = 90     # absolute maximum green
MIN_CYCLE         = 40     # minimum cycle length
MAX_CYCLE         = 180    # maximum cycle length

# Congestion thresholds (PCU)
LEVEL_HIGH_THRESH   = 10.0
LEVEL_MEDIUM_THRESH = 4.0


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DirectionTiming:
    name:            str
    pcu:             float         # effective PCU count
    flow_ratio:      float         # y_i = pcu / saturation_flow
    green:           int           # suggested green seconds
    yellow:          int = YELLOW
    all_red:         int = ALL_RED
    density_level:   str = "LOW"   # LOW / MEDIUM / HIGH
    vehicle_count:   int = 0       # raw vehicle detections used
    breakdown:       Dict[str, float] = field(default_factory=dict)  # class → PCU contrib


@dataclass
class CycleResult:
    main_road:     DirectionTiming
    cross_road:    DirectionTiming
    cycle_length:  int
    lost_time:     int
    total_pcu:     float
    y_total:       float           # sum of critical flow ratios
    formula_note:  str             # human-readable explanation


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class TrafficSignalController:
    """
    Computes adaptive signal timings from a list of YOLOv8 detection dicts.

    Each detection dict must contain:
        class_name, x1, y1, x2, y2, frame_id
    """

    def compute(self, detections: List[Dict]) -> CycleResult:
        pcu_a, pcu_b, count_a, count_b, bdwn_a, bdwn_b = self._compute_pcu(detections)
        total_pcu = pcu_a + pcu_b

        if total_pcu < 0.1:
            return self._default_result()

        # --- Flow ratios ---
        y_a = pcu_a / SATURATION_FLOW
        y_b = pcu_b / SATURATION_FLOW
        Y   = y_a + y_b

        # --- Webster's optimal cycle ---
        num_phases = 2
        L = LOST_TIME_PER_PHASE * num_phases     # total lost time
        if Y >= 1.0:
            Y = 0.95      # saturated — cap to prevent division by zero
        C_opt = (1.5 * L + 5) / (1.0 - Y)
        C_opt = max(MIN_CYCLE, min(MAX_CYCLE, math.ceil(C_opt)))

        # --- Effective green (C - L, split proportionally) ---
        effective = C_opt - L
        g_a = (y_a / Y) * effective
        g_b = effective - g_a

        g_a = max(MIN_GREEN, min(MAX_GREEN, round(g_a)))
        g_b = max(MIN_GREEN, min(MAX_GREEN, round(g_b)))

        # Recalculate actual cycle with clamped greens
        cycle = g_a + YELLOW + ALL_RED + g_b + YELLOW + ALL_RED

        dir_a = DirectionTiming(
            name="Main Road",
            pcu=round(pcu_a, 2),
            flow_ratio=round(y_a, 4),
            green=g_a,
            density_level=_density_level(pcu_a),
            vehicle_count=count_a,
            breakdown=bdwn_a,
        )
        dir_b = DirectionTiming(
            name="Cross Road",
            pcu=round(pcu_b, 2),
            flow_ratio=round(y_b, 4),
            green=g_b,
            density_level=_density_level(pcu_b),
            vehicle_count=count_b,
            breakdown=bdwn_b,
        )

        note = (
            f"Webster C_opt = (1.5×{L}+5)/(1−{round(Y,3)}) = {C_opt}s  |  "
            f"Main {g_a}s ({round(y_a/Y*100,1)}%)  "
            f"Cross {g_b}s ({round(y_b/Y*100,1)}%)"
        )

        return CycleResult(
            main_road=dir_a,
            cross_road=dir_b,
            cycle_length=cycle,
            lost_time=L,
            total_pcu=round(total_pcu, 2),
            y_total=round(Y, 4),
            formula_note=note,
        )

    # ------------------------------------------------------------------

    def _compute_pcu(
        self, detections: List[Dict]
    ) -> Tuple[float, float, int, int, Dict, Dict]:
        pcu_a = pcu_b = 0.0
        count_a = count_b = 0
        bdwn_a: Dict[str, float] = {}
        bdwn_b: Dict[str, float] = {}

        seen: set = set()

        for d in detections:
            base_pcu = PCU_WEIGHTS.get(d.get("class_name", ""), 0.0)
            if base_pcu == 0.0:
                continue

            # Deduplicate: same class at roughly same position in same frame
            key = (
                d.get("frame_id", 0),
                d.get("class_name"),
                round(d.get("x1", 0), 2),
                round(d.get("y1", 0), 2),
            )
            if key in seen:
                continue
            seen.add(key)

            x1, y1 = d.get("x1", 0.0), d.get("y1", 0.0)
            x2, y2 = d.get("x2", 1.0), d.get("y2", 1.0)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Proximity weight: vehicles closer to stop-line (lower y = higher in frame
            # for typical traffic cam angle) get slightly more weight.
            # Using cy: higher cy = lower in frame = closer = more urgent.
            proximity = 0.5 + 0.5 * cy   # range [0.5, 1.0]
            eff_pcu = base_pcu * proximity

            cls = d.get("class_name", "unknown")
            if cx < 0.5:
                pcu_a += eff_pcu
                count_a += 1
                bdwn_a[cls] = round(bdwn_a.get(cls, 0.0) + eff_pcu, 2)
            else:
                pcu_b += eff_pcu
                count_b += 1
                bdwn_b[cls] = round(bdwn_b.get(cls, 0.0) + eff_pcu, 2)

        return pcu_a, pcu_b, count_a, count_b, bdwn_a, bdwn_b

    def _default_result(self) -> CycleResult:
        return CycleResult(
            main_road=DirectionTiming(
                name="Main Road", pcu=0.0, flow_ratio=0.0, green=30,
                density_level="LOW", vehicle_count=0,
            ),
            cross_road=DirectionTiming(
                name="Cross Road", pcu=0.0, flow_ratio=0.0, green=30,
                density_level="LOW", vehicle_count=0,
            ),
            cycle_length=66,
            lost_time=8,
            total_pcu=0.0,
            y_total=0.0,
            formula_note="No vehicles detected — using default 30s equal split.",
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
