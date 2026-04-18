#!/usr/bin/env python3
# =============================================================================
# demo_setup.py — Interactive helper for running the Smart Traffic System
#                 with publicly available camera footage.
#
# Usage:
#   python demo_setup.py            # guided setup
#   python demo_setup.py --test     # probe all public sources and report
#   python demo_setup.py --youtube  # resolve a YouTube live stream and print cmd
# =============================================================================

import argparse
import os
import subprocess
import sys
import textwrap
import time
import urllib.request

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

# ---------------------------------------------------------------------------
# Public sources catalogue
# ---------------------------------------------------------------------------

# Each entry: name, url, type, notes
# Types: mjpeg, rtsp_public, youtube_live, youtube_vod
PUBLIC_SOURCES = [
    # ── YouTube live streams (traffic / street cams) ──────────────────────
    # These are popular 24/7 live stream channels; URLs are stable.
    {
        "name": "Jackson Hole Town Square — Wyoming, USA (24/7 live)",
        "url":  "https://www.youtube.com/watch?v=1-iHoSGfu8s",
        "type": "youtube_live",
        "note": "Busy pedestrian square with vehicles and foot traffic",
    },
    {
        "name": "Times Square NYC — New York, USA (24/7 live)",
        "url":  "https://www.youtube.com/watch?v=_on1Nbq8lPk",
        "type": "youtube_live",
        "note": "Heavy pedestrian and vehicle density — excellent for demo",
    },
    {
        "name": "Las Vegas Strip — Nevada, USA (24/7 live)",
        "url":  "https://www.youtube.com/watch?v=NyDDyT1lDhA",
        "type": "youtube_live",
        "note": "Busy road with cars and pedestrians",
    },
    {
        "name": "Abbey Road Crossing — London, UK (24/7 live)",
        "url":  "https://www.youtube.com/watch?v=nKOvCJFI7A0",
        "type": "youtube_live",
        "note": "Famous pedestrian crossing — good for person detection",
    },
    # ── YouTube VODs (downloadable for offline demo) ──────────────────────
    {
        "name": "Highway Traffic Compilation (VOD)",
        "url":  "https://www.youtube.com/watch?v=MNn9qKG2UFI",
        "type": "youtube_vod",
        "note": "Mix of highway and urban traffic — works offline once downloaded",
    },
    {
        "name": "Busy Intersection Time-lapse (VOD)",
        "url":  "https://www.youtube.com/watch?v=PnGiHDpsSMA",
        "type": "youtube_vod",
        "note": "Multiple vehicle classes + pedestrians",
    },
    # ── Public MJPEG streams ──────────────────────────────────────────────
    # Note: availability varies; these are best-effort public feeds.
    {
        "name": "Poznan, Poland — City Traffic Cam",
        "url":  "http://193.0.199.2/mjpg/video.mjpg",
        "type": "mjpeg",
        "note": "European city traffic; HTTP MJPEG — no auth required",
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_banner() -> None:
    print("\n" + "=" * 65)
    print("  Smart Traffic System — Demo Source Setup")
    print("=" * 65)


def _check_yt_dlp() -> bool:
    try:
        r = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            print(f"  yt-dlp {r.stdout.strip()} found")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    print("  yt-dlp NOT found — YouTube sources unavailable")
    print("  Install with:  pip install yt-dlp")
    return False


def _probe_mjpeg(url: str, timeout: int = 5) -> bool:
    """Return True if the MJPEG URL responds within timeout."""
    try:
        req = urllib.request.urlopen(url, timeout=timeout)
        content_type = req.headers.get("Content-Type", "")
        req.close()
        return "multipart" in content_type or "jpeg" in content_type or "video" in content_type
    except Exception:
        return False


def _probe_youtube(url: str) -> bool:
    """Return True if yt-dlp can resolve the URL."""
    try:
        r = subprocess.run(
            ["yt-dlp", "--no-playlist", "-g", "--skip-download", url],
            capture_output=True, text=True, timeout=15,
        )
        return r.returncode == 0
    except Exception:
        return False


def _resolve_stream_url(url: str) -> str | None:
    try:
        r = subprocess.run(
            ["yt-dlp", "--no-playlist", "-f",
             "bestvideo[ext=mp4][height<=720]+bestaudio/best[ext=mp4][height<=720]/best",
             "-g", url],
            capture_output=True, text=True, timeout=20,
        )
        if r.returncode == 0:
            return r.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return None


def _download_video(url: str, out_path: str) -> bool:
    """Download a YouTube video to a local file using yt-dlp."""
    try:
        r = subprocess.run(
            [
                "yt-dlp",
                "--no-playlist",
                "-f", "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best",
                "--merge-output-format", "mp4",
                "-o", out_path,
                url,
            ],
            timeout=300,
        )
        return r.returncode == 0 and os.path.exists(out_path)
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


def _print_run_cmd(source_url: str, label: str = "") -> None:
    if label:
        print(f"\n  {label}")
    print(f"\n  Run the system with this source:")
    if sys.platform == "win32":
        print(f'\n    set RTSP_URL={source_url}& set DEVICE=cpu& python main.py')
    else:
        print(f'\n    RTSP_URL="{source_url}" DEVICE=cpu python main.py')


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_test_all() -> None:
    """Probe all public sources and report availability."""
    _print_banner()
    yt_ok = _check_yt_dlp()
    print()

    for i, src in enumerate(PUBLIC_SOURCES, 1):
        t = src["type"]
        name = src["name"]
        url = src["url"]
        print(f"  [{i}] {name}")
        print(f"      {url}")

        if t == "mjpeg":
            ok = _probe_mjpeg(url)
            status = "ONLINE" if ok else "OFFLINE/unreachable"
            print(f"      Status: {status}")
        elif t in ("youtube_live", "youtube_vod"):
            if not yt_ok:
                print("      Status: SKIP (yt-dlp not installed)")
            else:
                print("      Status: resolving…", end="", flush=True)
                ok = _probe_youtube(url)
                print(f"\r      Status: {'RESOLVABLE' if ok else 'UNRESOLVABLE'}")
        print()


def cmd_youtube_live(url: str | None = None) -> None:
    """Resolve a YouTube live stream and print the run command."""
    _print_banner()
    yt_ok = _check_yt_dlp()
    if not yt_ok:
        sys.exit(1)

    if url is None:
        # List only live streams
        live_sources = [s for s in PUBLIC_SOURCES if s["type"] == "youtube_live"]
        print("\n  Available YouTube live streams:\n")
        for i, s in enumerate(live_sources, 1):
            print(f"  [{i}] {s['name']}")
            print(f"      {s['url']}")
            print(f"      Note: {s['note']}\n")
        choice = input("  Enter number (or paste a custom YouTube URL): ").strip()
        try:
            url = live_sources[int(choice) - 1]["url"]
        except (ValueError, IndexError):
            url = choice  # treat as raw URL

    print(f"\n  Resolving: {url}")
    stream_url = _resolve_stream_url(url)
    if stream_url:
        print(f"  Resolved stream URL (first 100 chars):\n  {stream_url[:100]}…")
        _print_run_cmd(url)
    else:
        print("  Could not resolve stream URL. Try a different YouTube URL.")
        sys.exit(1)


def cmd_download_demo() -> None:
    """Download a VOD for offline demo use."""
    _print_banner()
    yt_ok = _check_yt_dlp()
    if not yt_ok:
        sys.exit(1)

    vod_sources = [s for s in PUBLIC_SOURCES if s["type"] == "youtube_vod"]
    print("\n  Available VODs for download:\n")
    for i, s in enumerate(vod_sources, 1):
        print(f"  [{i}] {s['name']}")
        print(f"      {s['url']}")
        print(f"      Note: {s['note']}\n")

    choice = input("  Enter number to download (or paste a YouTube URL): ").strip()
    try:
        src = vod_sources[int(choice) - 1]
        url = src["url"]
        suggested_name = src["name"].split("—")[0].strip().replace(" ", "_").lower()
    except (ValueError, IndexError):
        url = choice
        suggested_name = "demo_traffic"

    out_path = f"{suggested_name}.mp4"
    print(f"\n  Downloading to: {out_path}")
    ok = _download_video(url, out_path)
    if ok:
        size_mb = os.path.getsize(out_path) / 1_048_576
        print(f"  Downloaded {size_mb:.1f} MB → {out_path}")
        _print_run_cmd(out_path, label=f"Use the downloaded video as source:")
    else:
        print("  Download failed.")
        sys.exit(1)


def cmd_guided() -> None:
    """Interactive guided setup."""
    _print_banner()
    print(textwrap.dedent("""
    This tool helps you run the Smart Traffic System using publicly
    available camera footage — no Jetson Nano or private RTSP feed needed.

    Options:
      [1] Use a YouTube live traffic stream (requires yt-dlp)
      [2] Download a traffic video for offline looped playback (requires yt-dlp)
      [3] Use a public MJPEG stream directly
      [4] Use your webcam (index 0)
      [5] Test all public sources for availability
    """))

    choice = input("  Select option [1-5]: ").strip()

    if choice == "1":
        cmd_youtube_live()
    elif choice == "2":
        cmd_download_demo()
    elif choice == "3":
        mjpeg_sources = [s for s in PUBLIC_SOURCES if s["type"] == "mjpeg"]
        print("\n  Public MJPEG streams:\n")
        for i, s in enumerate(mjpeg_sources, 1):
            print(f"  [{i}] {s['name']}")
            print(f"      {s['url']}\n")
        c = input("  Enter number (or paste a custom MJPEG URL): ").strip()
        try:
            url = mjpeg_sources[int(c) - 1]["url"]
        except (ValueError, IndexError):
            url = c
        _print_run_cmd(url)
    elif choice == "4":
        _print_run_cmd("0", label="Webcam (index 0):")
    elif choice == "5":
        cmd_test_all()
    else:
        print("  Invalid choice.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smart Traffic System — demo source helper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
          python demo_setup.py                         # guided interactive setup
          python demo_setup.py --test                  # probe all public sources
          python demo_setup.py --youtube               # pick a YouTube live stream
          python demo_setup.py --youtube <URL>         # resolve specific YouTube URL
          python demo_setup.py --download              # download a VOD for offline use
        """),
    )
    parser.add_argument("--test",     action="store_true", help="Probe all public sources")
    parser.add_argument("--youtube",  nargs="?", const="", metavar="URL",
                        help="Resolve a YouTube stream (omit URL to pick from list)")
    parser.add_argument("--download", action="store_true", help="Download a VOD for offline use")
    args = parser.parse_args()

    if args.test:
        cmd_test_all()
    elif args.youtube is not None:
        cmd_youtube_live(args.youtube or None)
    elif args.download:
        cmd_download_demo()
    else:
        cmd_guided()


if __name__ == "__main__":
    main()
