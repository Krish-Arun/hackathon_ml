"""
Serve the evaluator UI - runs on Laptop 2.

    python scripts/serve_ui.py --port 8080

Then open http://localhost:8080/?api=http://<laptop-1-ip>:8000

Served over HTTP rather than opened as a file:// page because browsers block
cross-origin fetches from file:// regardless of the server's CORS headers.
"""
import argparse, functools, http.server, socketserver
from pathlib import Path

# Normally the UI lives in the repo. But Laptop 2 may only have this script
# and index.html copied next to each other, so fall back to our own directory.
_HERE = Path(__file__).resolve().parent
_IN_REPO = _HERE.parents[0] / "src" / "pathsnap" / "ui"
UI_DIR = _IN_REPO if (_IN_REPO / "index.html").exists() else _HERE


def main(port):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(UI_DIR))
    with socketserver.TCPServer(("0.0.0.0", port), handler) as httpd:
        print(f"UI on http://0.0.0.0:{port}  (serving {UI_DIR})")
        print(f"open  http://localhost:{port}/?api=http://<laptop-1-ip>:8000")
        httpd.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    main(ap.parse_args().port)
