"""
Serve the evaluator UI - runs on Laptop 2.

    python scripts/serve_ui.py --port 8080

Then open http://localhost:8080/?api=http://<laptop-1-ip>:8000

Served over HTTP rather than opened as a file:// page because browsers block
cross-origin fetches from file:// regardless of the server's CORS headers.
"""
import argparse, functools, http.server, socketserver
from pathlib import Path

UI_DIR = Path(__file__).resolve().parents[1] / "src" / "pathsnap" / "ui"


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
