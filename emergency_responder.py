# =============================================================================
# emergency_responder.py — Coordinates emergency response to accidents
#
# Persists accidents, generates alerts, and optionally triggers signal
# pre-emption for emergency vehicles
# =============================================================================

import logging
import threading
import time
from typing import Optional, Dict, Callable
import requests

logger = logging.getLogger("responder")


class EmergencyResponder:
    """Handles emergency response coordination."""

    def __init__(
        self,
        on_alert: Optional[Callable] = None,
        webhook_url: Optional[str] = None,
        webhook_timeout: float = 3.0,
        signal_preemption_duration: float = 30.0,
    ):
        """
        Args:
            on_alert: Callback(severity, message, event) called on accident
            webhook_url: Optional POST endpoint for accident notifications
            webhook_timeout: Request timeout in seconds
            signal_preemption_duration: How long to hold preemption (seconds)
        """
        self.on_alert = on_alert
        self.webhook_url = webhook_url
        self.webhook_timeout = webhook_timeout
        self.signal_preemption_duration = signal_preemption_duration

    def handle(self, event) -> Dict:
        """
        Handle an accident event.

        Args:
            event: AccidentEvent object

        Returns:
            Response dict with action taken
        """
        response = {
            "event_id": event.event_id,
            "status": "processed",
            "actions": [],
        }

        # Map accident type to severity
        severity = self._get_severity(event.accident_type, event.confidence)
        message = self._format_message(event, severity)

        logger.error("EMERGENCY: %s", message)

        # Trigger alert callback
        if self.on_alert:
            self.on_alert(severity=severity, message=message, event=event)
            response["actions"].append("alert_generated")

        # Post to webhook (non-blocking)
        if self.webhook_url:
            self._post_webhook_async(event, severity, message)
            response["actions"].append("webhook_queued")

        # Could trigger signal preemption here:
        # response["preemption"] = {
        #     "duration_sec": self.signal_preemption_duration,
        #     "priority_lane": self._get_priority_lane(event),
        # }
        response["actions"].append("signal_preemption_eligible")

        return response

    def _get_severity(self, accident_type: str, confidence: float) -> str:
        """Map accident type and confidence to severity level."""
        if accident_type == "collision":
            return "CRITICAL"
        elif accident_type == "sudden_stop":
            return "HIGH" if confidence > 0.85 else "MEDIUM"
        elif accident_type == "stationary":
            return "MEDIUM" if confidence > 0.8 else "LOW"
        elif accident_type == "abnormal_motion":
            return "HIGH" if confidence > 0.9 else "MEDIUM"
        else:
            return "MEDIUM"

    def _format_message(self, event, severity: str) -> str:
        """Format a human-readable alert message."""
        return (
            f"[{severity}] {event.accident_type.upper()} ACCIDENT in {event.lane} lane "
            f"(track {event.primary_track_id}, confidence {event.confidence:.0%})"
        )

    def _post_webhook_async(self, event, severity: str, message: str):
        """Post to webhook in a background thread (never blocks)."""
        def _post():
            try:
                payload = {
                    "event_id": event.event_id,
                    "timestamp": event.timestamp,
                    "severity": severity,
                    "accident_type": event.accident_type,
                    "lane": event.lane,
                    "track_id": event.primary_track_id,
                    "confidence": event.confidence,
                    "message": message,
                }
                requests.post(
                    self.webhook_url,
                    json=payload,
                    timeout=self.webhook_timeout,
                )
                logger.info("Webhook dispatched for event %d", event.event_id)
            except Exception as e:
                logger.warning("Webhook dispatch failed: %s", e)

        thread = threading.Thread(target=_post, daemon=True)
        thread.start()

    def _get_priority_lane(self, event) -> str:
        """Determine which lane should get signal preemption."""
        return event.lane
