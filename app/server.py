"""Tiny dependency-free web UI for the demo.

Serves the question box, the event table and the clips from one origin, which sidesteps
the CORS problem the challenge playbook warns about when a page calls the backend
directly from the browser.
"""
from __future__ import annotations

import json
import mimetypes
import os
import posixpath
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fred import agent, events, pipeline, trace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = {"path": None}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter console during a demo
        pass

    def _send(self, code, body, content_type="application/json", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path):
        if not os.path.isfile(path):
            return self._send(404, {"error": "not found"})
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        size = os.path.getsize(path)
        start, end = 0, size - 1
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            raw = range_header[6:].split("-")
            start = int(raw[0]) if raw[0] else 0
            if len(raw) > 1 and raw[1]:
                end = min(int(raw[1]), size - 1)
        length = end - start + 1
        self.send_response(206 if range_header else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if range_header:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(path, "rb") as fh:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                chunk = fh.read(min(1 << 16, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(chunk)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            return self._send_file(os.path.join(ROOT, "app", "index.html"))
        if path == "/api/events":
            return self._send(200, events.query(limit=500, path=DB_PATH["path"]))
        if path == "/api/stats":
            return self._send(200, events.stats(path=DB_PATH["path"]))
        if path == "/api/ask":
            params = urllib.parse.parse_qs(parsed.query)
            question = (params.get("q") or [""])[0]
            if not question:
                return self._send(400, {"error": "missing q"})
            try:
                return self._send(200, agent.ask(question, db_path=DB_PATH["path"]))
            except Exception as exc:  # noqa: BLE001
                return self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
        if path == "/api/reel":
            params = urllib.parse.parse_qs(parsed.query)
            assist = (params.get("assist_type") or [None])[0]
            rows = events.query(
                where="assist_type = ?" if assist else "",
                params=(assist,) if assist else (), limit=25, path=DB_PATH["path"])
            if not rows:
                return self._send(404, {"error": "no events"})
            try:
                out = pipeline.highlights_reel(rows, out_path="data/out/highlights.mp4")
                return self._send(200, {"reel": "/media/" + os.path.relpath(out, ROOT)})
            except Exception as exc:  # noqa: BLE001
                return self._send(500, {"error": str(exc)})
        if path.startswith("/media/"):
            rel = urllib.parse.unquote(path[len("/media/"):])
            safe = posixpath.normpath(rel).lstrip("/")
            if safe.startswith(".."):
                return self._send(403, {"error": "forbidden"})
            return self._send_file(os.path.join(ROOT, safe))
        return self._send(404, {"error": "not found"})


def serve(host="127.0.0.1", port=8800, db_path=None):
    DB_PATH["path"] = db_path
    trace.init()
    os.chdir(ROOT)
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Fred Lasso UI: http://{host}:{port}")
    print("Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
