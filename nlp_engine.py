# =============================================================================
# nlp_engine.py — Lightweight NLP for scene captioning + alert generation
#
# Designed for resource-constrained edge devices (Jetson Nano).
# Uses a rule/template-based approach (no large LLM required on-device).
# Swap SceneCaptioner._llm_caption() with a quantised model call if you have
# the RAM to run e.g. TinyLlama-1.1B-Chat via llama-cpp-python.
# =============================================================================

import logging
import random
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sentence templates
# ---------------------------------------------------------------------------

_CAPTION_INTRO = [
    "The scene shows",
    "The camera captures",
    "Currently visible:",
    "In the frame:",
    "The feed shows",
]

_SINGLE_OBJECT = [
    "a single {obj}",
    "one {obj}",
    "a {obj}",
]

_COUNT_PHRASES = {
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
}

_ALERT_TEMPLATES = {
    "CRITICAL": [
        "[CRITICAL] ALERT: {count} {obj} detected with {conf:.0%} confidence at {time}.",
        "[CRITICAL] {obj} identified — confidence {conf:.0%}. Immediate attention required.",
    ],
    "HIGH": [
        "[HIGH] HIGH: {count} {obj} detected ({conf:.0%} confidence) at {time}.",
        "[HIGH] ALERT: {obj} present in frame. Confidence: {conf:.0%}.",
    ],
    "MEDIUM": [
        "[NOTICE] MEDIUM: {count} {obj} spotted ({conf:.0%} confidence) — {time}.",
        "[NOTICE] {obj} detected. Confidence {conf:.0%}.",
    ],
    "LOW": [
        "[INFO] LOW: {obj} detected at {conf:.0%} confidence.",
    ],
}

_NO_DETECTION_CAPTIONS = [
    "The scene appears clear — no objects detected.",
    "No objects currently visible in the frame.",
    "Empty scene — nothing detected at this moment.",
]


# ---------------------------------------------------------------------------
# Scene Captioner
# ---------------------------------------------------------------------------

class SceneCaptioner:
    """
    Converts a list of detection dicts into a natural-language scene description.

    Each detection dict expected to have keys:
        class_name  (str)
        confidence  (float 0-1)
        x1, y1, x2, y2  (float, normalised 0-1 or pixel)

    Usage:
        captioner = SceneCaptioner()
        caption = captioner.caption(detections, frame_width=1280, frame_height=720)
    """

    def caption(
        self,
        detections: List[Dict[str, Any]],
        frame_width: int = 1280,
        frame_height: int = 720,
    ) -> str:
        if not detections:
            return random.choice(_NO_DETECTION_CAPTIONS)

        counts = Counter(d["class_name"] for d in detections)
        # Keep only the top N objects to avoid overly long captions
        top = counts.most_common(config.NLP_MAX_OBJECTS_IN_CAPTION)
        leftover = sum(counts.values()) - sum(c for _, c in top)

        object_phrases = []
        for obj, cnt in top:
            object_phrases.append(self._count_phrase(obj, cnt))

        # Spatial context for the most confident detection
        primary = max(detections, key=lambda d: d["confidence"])
        position = self._spatial_label(primary, frame_width, frame_height)

        parts = ", ".join(object_phrases)
        if leftover > 0:
            parts += f" and {leftover} more object{'s' if leftover > 1 else ''}"

        intro = random.choice(_CAPTION_INTRO)
        caption = f"{intro} {parts}."

        if position:
            primary_name = self._pluralise(primary["class_name"], 1)
            caption += f" The {primary_name} appears {position} of the frame."

        return caption

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _count_phrase(self, obj: str, count: int) -> str:
        if count == 1:
            article = "an" if obj[0].lower() in "aeiou" else "a"
            return f"{article} {obj}"
        word = _COUNT_PHRASES.get(count, str(count))
        return f"{word} {self._pluralise(obj, count)}"

    @staticmethod
    def _pluralise(word: str, count: int) -> str:
        if count == 1:
            return word
        # Simple English pluralisation rules
        if word.endswith(("s", "sh", "ch", "x", "z")):
            return word + "es"
        if word.endswith("y") and word[-2] not in "aeiou":
            return word[:-1] + "ies"
        if word.endswith("f"):
            return word[:-1] + "ves"
        if word.endswith("fe"):
            return word[:-2] + "ves"
        return word + "s"

    @staticmethod
    def _spatial_label(
        detection: Dict[str, Any],
        frame_width: int,
        frame_height: int,
    ) -> str:
        """
        Return a human-readable position string ('top-left', 'centre', etc.)
        Works with both pixel and normalised (0-1) coordinates.
        """
        cx = (detection["x1"] + detection["x2"]) / 2
        cy = (detection["y1"] + detection["y2"]) / 2

        # Normalise if pixel coords
        if cx > 1 or cy > 1:
            cx = cx / frame_width
            cy = cy / frame_height

        h_zone = "left" if cx < 0.35 else ("right" if cx > 0.65 else "centre")
        v_zone = "top"  if cy < 0.35 else ("bottom" if cy > 0.65 else "middle")

        if v_zone == "middle" and h_zone == "centre":
            return "in the centre"
        if v_zone == "middle":
            return f"on the {h_zone}"
        if h_zone == "centre":
            return f"at the {v_zone}"
        return f"at the {v_zone}-{h_zone}"


# ---------------------------------------------------------------------------
# Alert Generator
# ---------------------------------------------------------------------------

class AlertGenerator:
    """
    Evaluates detections against ALERT_CLASSES rules and produces
    natural-language alert messages with per-class cooldown.

    Usage:
        generator = AlertGenerator()
        alerts = generator.evaluate(detections, frame_id=42)
        # alerts = [{"class_name": ..., "severity": ..., "confidence": ...,
        #            "message": ..., "frame_id": ...}, ...]
    """

    def __init__(self) -> None:
        # Track last alert time per class to enforce cooldown
        self._last_alert: Dict[str, datetime] = defaultdict(
            lambda: datetime.min
        )

    def evaluate(
        self,
        detections: List[Dict[str, Any]],
        frame_id: int,
    ) -> List[Dict[str, Any]]:
        """
        Returns a (possibly empty) list of alert dicts for detections that
        match ALERT_CLASSES rules and are not on cooldown.
        """
        if not detections:
            return []

        alerts: List[Dict[str, Any]] = []
        now = datetime.utcnow()
        cooldown = timedelta(seconds=config.ALERT_COOLDOWN_SECONDS)

        # Group detections by class, keep the most confident per class
        best: Dict[str, Dict] = {}
        for d in detections:
            cls = d["class_name"]
            if cls in config.ALERT_CLASSES:
                if cls not in best or d["confidence"] > best[cls]["confidence"]:
                    best[cls] = d

        for cls, det in best.items():
            rule = config.ALERT_CLASSES[cls]
            if det["confidence"] < rule["min_confidence"]:
                continue
            if now - self._last_alert[cls] < cooldown:
                logger.debug("Alert for '%s' suppressed (cooldown)", cls)
                continue

            severity = rule["severity"]
            count = sum(1 for d in detections if d["class_name"] == cls)
            msg = self._generate_message(cls, severity, det["confidence"], count)

            self._last_alert[cls] = now
            alerts.append({
                "class_name": cls,
                "severity":   severity,
                "confidence": det["confidence"],
                "message":    msg,
                "frame_id":   frame_id,
            })
            if severity in ["CRITICAL", "HIGH"]:
                logger.warning("ALERT [%s] %s", severity, msg)

        return alerts

    @staticmethod
    def _generate_message(
        class_name: str,
        severity: str,
        confidence: float,
        count: int,
    ) -> str:
        templates = _ALERT_TEMPLATES.get(severity, _ALERT_TEMPLATES["LOW"])
        template = random.choice(templates)
        time_str = datetime.utcnow().strftime("%H:%M:%S UTC")
        return template.format(
            obj=class_name,
            conf=confidence,
            count=count,
            time=time_str,
        )

    def reset_cooldowns(self) -> None:
        """Force-reset all cooldowns (e.g. for testing)."""
        self._last_alert.clear()