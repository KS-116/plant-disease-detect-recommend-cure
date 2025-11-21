#!/usr/bin/env python3
"""Lightweight end-to-end test for the VGG Flask server.

Usage:
  python3 scripts/test_inference.py [--start-server] [--port PORT]

This script will optionally start `vgg_flask.py` as a subprocess, poll `/health`
until it's ready, then POST a generated sample image to `/predict` and print the
responses. If it started the server, it will terminate it when done.
"""
import argparse
import subprocess
import sys
import time
import tempfile
import os


def ensure_requests():
    try:
        import requests  # noqa: F401
    except Exception:
        print("'requests' package not found. Installing...", file=sys.stderr)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])


def wait_for_health(url, timeout=30.0, interval=0.5):
    import requests
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=2.0)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(interval)
    raise TimeoutError(f"Timed out waiting for {url} to respond")


def run_test(host, port, start_server=False):
    ensure_requests()
    import requests
    server_proc = None
    if start_server:
        # Start server as a subprocess
        server_cmd = [sys.executable, "vgg_flask.py", "--port", str(port)]
        print("Starting server:", " ".join(server_cmd))
        server_proc = subprocess.Popen(server_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        health_url = f"http://{host}:{port}/health"
        print("Waiting for health endpoint:", health_url)
        info = wait_for_health(health_url, timeout=60.0)
        print("Health returned:", info)

        # Create a small sample image (green square)
        try:
            from PIL import Image
            import io
        except Exception:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
            from PIL import Image
            import io

        img = Image.new("RGB", (224, 224), color=(30, 200, 30))
        tmpf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        img.save(tmpf, format="JPEG")
        tmpf.close()

        predict_url = f"http://{host}:{port}/predict"
        print("POSTing sample image to:", predict_url)
        with open(tmpf.name, "rb") as fh:
            files = {"image": (os.path.basename(tmpf.name), fh, "image/jpeg")}
            r = requests.post(predict_url, files=files, timeout=20.0)

        print("Predict status:", r.status_code)
        try:
            print(r.json())
        except Exception:
            print(r.text)

    finally:
        if server_proc:
            print("Stopping server process...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except Exception:
                server_proc.kill()
        # cleanup
        try:
            os.unlink(tmpf.name)
        except Exception:
            pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start-server", action="store_true", help="Start vgg_flask.py as a subprocess")
    p.add_argument("--host", default="127.0.0.1", help="Server host")
    p.add_argument("--port", default=5001, type=int, help="Server port")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run_test(args.host, args.port, start_server=args.start_server)
    except Exception as e:
        print("Test failed:", e, file=sys.stderr)
        sys.exit(2)
    print("Test completed")
