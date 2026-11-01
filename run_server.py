"""
City Guardian — Official Localhost Runtime Launcher
Launches the Flask Environmental Intelligence Server, pre-warms YOLOv8 models,
and automatically opens the interactive dashboard in your default web browser.

Usage:
    python run_server.py
    python run_server.py --port 5000 --no-browser
"""
import os
import sys
import time
import threading
import webbrowser
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def open_browser(url: str, delay: float = 1.2):
    """Opens the user's default browser after a short delay for server startup."""
    time.sleep(delay)
    print(f"\n🚀 Opening City Guardian Dashboard in your browser: {url}")
    webbrowser.open(url)


def main():
    parser = argparse.ArgumentParser(description="City Guardian Localhost Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind (default: 5000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically launch web browser")
    args = parser.parse_args()

    local_url = f"http://localhost:{args.port}"

    print("=" * 80)
    print(" CITY GUARDIAN — LOCALHOST ENVIRONMENTAL MONITORING RUNTIME")
    print(" Geo-Aware AI Environmental Intelligence & Legal Compliance System")
    print("=" * 80)
    print(f" 🌐 Dashboard URL:        {local_url}")
    print(f" 🩺 Health Check:         {local_url}/api/health")
    print(f" 📷 Image Inference API:  {local_url}/api/analyze-image")
    print(f" 📑 Incident Reports:     {local_url}/api/reports")
    print(f" 🗺️ Spatial Map Feed:     {local_url}/api/map")
    print("-" * 80)
    print(" 🧠 Active Models:        YOLOv8 Nano (5 Production Domains Pre-Warmed)")
    print(" ⚡ Hardware SIMD:        AVX2 Accelerated Intel CPU Execution")
    print(" 🛑 To stop the server:   Press CTRL + C in this terminal window")
    print("=" * 80)
    print()

    # Pre-launch browser
    if not args.no_browser:
        threading.Thread(target=open_browser, args=(local_url,), daemon=True).start()

    from backend.app import app
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()

