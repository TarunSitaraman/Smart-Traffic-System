# =============================================================================
# video_source.py — Unified video source abstraction
#
# Supports:
#   • Integer / "0"        → local USB webcam
#   • Local file path      → .mp4 / .avi / .mov / .mkv etc. (loops when done)
#   • rtsp://...           → RTSP camera (GStreamer HW-accel on Jetson, else OpenCV)
#   • http(s)://...        → HTTP MJPEG stream (direct OpenCV)
#   • youtube:// or any    → YouTube URL resolved via yt-dlp
#
# Usage:
#   cap = open_source(url)          # returns cv2.VideoCapture
#   loop = should_loop(url)         # True for local video files
# =============================================================================

import logging
import os
from typing import Optional, Tuple

import cv2

import config

logger = logging.getLogger(__name__)

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".ts", ".flv"}


# ---------------------------------------------------------------------------
# Source type helpers
# ---------------------------------------------------------------------------

def _is_webcam(url: str) -> bool:
    return url.isdigit()


def _is_video_file(url: str) -> bool:
    _, ext = os.path.splitext(url.lower())
    return ext in _VIDEO_EXTENSIONS or (os.path.isfile(url) and not url.startswith(("rtsp://", "http://", "https://")))


def _is_youtube(url: str) -> bool:
    return "youtube.com/watch" in url or "youtu.be/" in url


def should_loop(url: str) -> bool:
    """Return True when the source is a local file that should restart on EOF."""
    return _is_video_file(url) and config.VIDEO_LOOP


# ---------------------------------------------------------------------------
# YouTube resolution via yt-dlp
# ---------------------------------------------------------------------------

def _resolve_youtube(url: str) -> Optional[str]:
    """
    Use yt-dlp Python API to extract a direct streamable URL from a YouTube link.
    Works with live streams and regular videos.
    Returns None if yt-dlp is unavailable or resolution fails.
    """
    try:
        import yt_dlp  # noqa: PLC0415

        ydl_opts = {
            "format": "bestvideo[ext=mp4][height<=720]/best[ext=mp4][height<=720]/best",
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "extractor_args": {"youtube": {"player_client": ["web"]}},
            "js_runtimes": ["nodejs"],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            stream_url = info.get("url") or (info.get("formats") or [{}])[-1].get("url")
            if stream_url:
                logger.info("yt-dlp resolved stream URL (truncated): %s…", stream_url[:80])
                return stream_url
            logger.warning("yt-dlp returned no URL for: %s", url)
    except ImportError:
        logger.warning("yt-dlp not installed — run: pip install yt-dlp")
    except Exception as exc:
        logger.warning("yt-dlp failed to resolve %s: %s", url, exc)
    return None


# ---------------------------------------------------------------------------
# GStreamer pipeline (Jetson Nano hardware-accelerated RTSP)
# ---------------------------------------------------------------------------

def _build_gstreamer_pipeline(rtsp_url: str, width: int, height: int, fps: int) -> str:
    return (
        f"rtspsrc location={rtsp_url} latency=100 ! "
        "rtph264depay ! h264parse ! "
        "nvv4l2decoder ! "
        "nvvidconv ! "
        f"video/x-raw,format=BGRx,width={width},height={height},framerate={fps}/1 ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink drop=1"
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def open_source(url: str) -> Tuple[cv2.VideoCapture, bool]:
    """
    Open a video source and return (VideoCapture, looping).

    looping=True means the caller should seek to frame 0 on EOF
    instead of treating it as a stream error.
    """
    if _is_webcam(url):
        return _open_webcam(url)

    if _is_youtube(url):
        return _open_youtube(url)

    if _is_video_file(url):
        return _open_file(url)

    # HTTP(S) MJPEG or RTSP
    return _open_network(url), False


# ---------------------------------------------------------------------------
# Source-specific openers
# ---------------------------------------------------------------------------

def _open_webcam(url: str) -> Tuple[cv2.VideoCapture, bool]:
    index = int(url)
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          config.CAMERA_FPS)
    logger.info("Opened webcam index %d", index)
    return cap, False


def _open_file(url: str) -> Tuple[cv2.VideoCapture, bool]:
    if not os.path.exists(url):
        raise RuntimeError(f"Video file not found: {url}")
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {url}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    logger.info(
        "Opened video file: %s  |  %d frames  |  %.1f fps  |  loop=%s",
        os.path.basename(url), total, fps, config.VIDEO_LOOP,
    )
    return cap, config.VIDEO_LOOP


def _open_youtube(url: str) -> Tuple[cv2.VideoCapture, bool]:
    stream_url = _resolve_youtube(url)
    if stream_url is None:
        raise RuntimeError(
            f"Could not resolve YouTube URL: {url}\n"
            "Install yt-dlp with: pip install yt-dlp"
        )
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open resolved YouTube stream from: {url}")
    logger.info("Opened YouTube stream: %s", url)
    return cap, False


def _open_network(url: str) -> cv2.VideoCapture:
    # Try GStreamer first for RTSP (Jetson hardware path)
    if url.startswith("rtsp://"):
        gst_pipe = _build_gstreamer_pipeline(
            url, config.CAMERA_WIDTH, config.CAMERA_HEIGHT, config.CAMERA_FPS
        )
        cap = cv2.VideoCapture(gst_pipe, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            logger.info("Opened RTSP via GStreamer hardware pipeline")
            return cap
        logger.warning("GStreamer pipeline failed — falling back to direct OpenCV")

    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open network source: {url}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          config.CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    logger.info("Opened network source via OpenCV: %s", url[:80])
    return cap
