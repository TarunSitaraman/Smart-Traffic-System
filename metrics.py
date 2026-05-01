# =============================================================================
# metrics.py — Traffic performance metrics and KPI calculation
#
# Computes throughput, delay, queue lengths, and signal efficiency from
# persisted detection and signal event data
# =============================================================================

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("metrics")


class MetricsEngine:
    """Computes traffic performance KPIs."""

    def __init__(self, db=None):
        """
        Args:
            db: Database module for querying historical data
        """
        self.db = db

    def compute_kpis(self, window_minutes: int = 60) -> Dict:
        """
        Compute current KPIs over a time window.

        Args:
            window_minutes: Duration of the window in minutes

        Returns:
            Dict with throughput, delay, queue metrics, signal efficiency, accidents
        """
        if not self.db:
            return self._empty_kpis()

        now_iso = datetime.now(timezone.utc).isoformat()
        then_iso = (
            datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
        ).isoformat()

        try:
            # Get lane metrics for the window
            lane_data = self.db.get_lane_metrics_in_range(then_iso, now_iso)

            # Get signal events
            signal_events = self.db.get_signal_events_in_range(then_iso, now_iso)

            # Get accidents
            accidents = self.db.get_accidents_in_range(then_iso, now_iso)

            # Compute metrics
            throughput_per_min = self._compute_throughput(lane_data)
            avg_delay = self._compute_delay(lane_data)
            queue_stats = self._compute_queue_stats(lane_data)
            signal_efficiency = self._compute_signal_efficiency(signal_events)

            return {
                "window_minutes": window_minutes,
                "timestamp": now_iso,
                "throughput_vehicles_per_min": round(throughput_per_min, 2),
                "average_delay_seconds": round(avg_delay, 2),
                "queue_average_vehicles": round(queue_stats["avg"], 1),
                "queue_max_vehicles": queue_stats["max"],
                "signal_efficiency_percent": round(signal_efficiency * 100, 1),
                "accidents_in_window": len(accidents),
                "lane_breakdown": self._lane_breakdown(lane_data),
            }

        except Exception as e:
            logger.error("Error computing KPIs: %s", e)
            return self._empty_kpis()

    def get_history_series(self, hours: int = 24, interval_min: int = 15) -> Dict:
        """
        Get hourly aggregated metrics for historical chart.

        Args:
            hours: Number of hours to look back
            interval_min: Aggregation interval in minutes

        Returns:
            Dict with timestamps and series data
        """
        if not self.db:
            return {"series": []}

        now = datetime.now(timezone.utc)
        then = now - timedelta(hours=hours)

        series = []
        current = then

        while current <= now:
            next_bucket = current + timedelta(minutes=interval_min)
            current_iso = current.isoformat()
            next_iso = next_bucket.isoformat()

            try:
                lane_data = self.db.get_lane_metrics_in_range(current_iso, next_iso)
                throughput = self._compute_throughput(lane_data)
                queue_stats = self._compute_queue_stats(lane_data)
                series.append(
                    {
                        "timestamp": current_iso,
                        "throughput": round(throughput, 2),
                        "queue_avg": round(queue_stats["avg"], 1),
                    }
                )
            except Exception:
                series.append(
                    {
                        "timestamp": current_iso,
                        "throughput": 0,
                        "queue_avg": 0,
                    }
                )

            current = next_bucket

        return {"hours": hours, "interval_minutes": interval_min, "series": series}

    @staticmethod
    def _compute_throughput(lane_data: list) -> float:
        """Compute vehicles per minute."""
        if not lane_data:
            return 0.0

        total_vehicles = sum(row.get("vehicle_count", 0) for row in lane_data)
        duration_minutes = 60  # assume 60 min window
        return total_vehicles / duration_minutes if duration_minutes > 0 else 0.0

    @staticmethod
    def _compute_delay(lane_data: list) -> float:
        """Estimate average vehicle delay in seconds."""
        if not lane_data:
            return 0.0

        # Average delay is (queue_length * cycle_time) / 2
        # Simplified: use queue counts as a proxy
        total_queue = sum(row.get("queue_length", 0) for row in lane_data)
        count = len(lane_data)

        if count == 0:
            return 0.0

        avg_queue = total_queue / count
        # Each vehicle in queue ~ 5 sec delay per position
        return avg_queue * 5.0

    @staticmethod
    def _compute_queue_stats(lane_data: list) -> Dict:
        """Get queue statistics."""
        if not lane_data:
            return {"avg": 0.0, "max": 0}

        queues = [row.get("queue_length", 0) for row in lane_data]
        return {
            "avg": np.mean(queues) if queues else 0.0,
            "max": max(queues) if queues else 0,
        }

    @staticmethod
    def _compute_signal_efficiency(signal_events: list) -> float:
        """
        Compute signal efficiency as ratio of green time to cycle time.

        Returns:
            Efficiency as a fraction (0-1)
        """
        if not signal_events:
            return 0.5  # default baseline

        total_green = 0
        total_time = 0

        for event in signal_events:
            state = event.get("state", "RED")
            duration = event.get("duration_sec", 0)
            if state == "GREEN":
                total_green += duration
            total_time += duration

        if total_time == 0:
            return 0.5

        return total_green / total_time

    @staticmethod
    def _lane_breakdown(lane_data: list) -> Dict:
        """Get per-lane metrics."""
        breakdown = {}

        for row in lane_data:
            lane = row.get("lane", "unknown")
            if lane not in breakdown:
                breakdown[lane] = {
                    "vehicle_count": 0,
                    "queue_length": 0.0,
                }
            breakdown[lane]["vehicle_count"] += row.get("vehicle_count", 0)
            breakdown[lane]["queue_length"] = row.get("queue_length", 0)

        return breakdown

    @staticmethod
    def _empty_kpis() -> Dict:
        """Return a zero-filled KPI dict."""
        return {
            "window_minutes": 60,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "throughput_vehicles_per_min": 0,
            "average_delay_seconds": 0,
            "queue_average_vehicles": 0,
            "queue_max_vehicles": 0,
            "signal_efficiency_percent": 50,
            "accidents_in_window": 0,
            "lane_breakdown": {},
        }
