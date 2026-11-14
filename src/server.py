"""Local HTTP server for WebGPU + AudioWorklet Voice Changer UI and ONNX models."""

import argparse
import functools
import http.server
import os
import socket
import socketserver
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class VCRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler serving web assets and ONNX models with COOP/COEP headers."""

    MIME_MAP = {
        ".html": "text/html; charset=utf-8",
        ".js": "application/javascript; charset=utf-8",
        ".mjs": "application/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".onnx": "application/octet-stream",
        ".wasm": "application/wasm",
        ".json": "application/json; charset=utf-8",
        ".svg": "image/svg+xml",
        ".png": "image/png",
    }

    def __init__(self, *args, project_root: Path = PROJECT_ROOT, **kwargs) -> None:
        self.project_root = Path(project_root).resolve()
        super().__init__(*args, directory=str(self.project_root), **kwargs)

    def end_headers(self) -> None:
        # Required for SharedArrayBuffer / multi-threaded ORT WASM and WebGPU compatibility
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def guess_type(self, path: str) -> str:
        ext = Path(path).suffix.lower()
        return self.MIME_MAP.get(ext, "application/octet-stream")

    def translate_path(self, path: str) -> str:
        # Strip query/fragment
        clean_path = path.split("?", 1)[0].split("#", 1)[0]
        if clean_path in ("/", "/index.html"):
            rel = "web/index.html"
        elif clean_path.startswith("/web/") or clean_path.startswith("/models/"):
            rel = clean_path.lstrip("/")
        elif clean_path in ("/app.js", "/vc-audio-worklet.js", "/styles.css"):
            rel = f"web/{clean_path.lstrip('/')}"
        else:
            # Restrict access strictly to web/ or models/
            return str(self.project_root / "__not_found__")

        candidate = (self.project_root / rel).resolve()
        web_dir = (self.project_root / "web").resolve()
        models_dir = (self.project_root / "models").resolve()
        if not (candidate.is_relative_to(web_dir) or candidate.is_relative_to(models_dir)):
            return str(self.project_root / "__forbidden__")
        return str(candidate)

    def log_message(self, format: str, *args) -> None:
        sys.stderr.write(f"[VC-Server] {self.address_string()} - {format % args}\n")


def ensure_onnx_models(project_root: Path = PROJECT_ROOT) -> None:
    """Auto-export ONNX models if missing."""
    models_dir = project_root / "models"
    spk_onnx = models_dir / "speaker_encoder.onnx"
    core_onnx = models_dir / "streaming_vc_core.onnx"
    if not (spk_onnx.exists() and core_onnx.exists()):
        print("[VC-Server] ONNX models not found. Exporting automatically...")
        from src.export_onnx import export_models

        export_models(output_dir=models_dir)


def create_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    project_root: Path = PROJECT_ROOT,
) -> ThreadingHTTPServer:
    """Create configured ThreadingHTTPServer instance."""
    ensure_onnx_models(project_root=project_root)
    handler = functools.partial(VCRequestHandler, project_root=project_root)
    return ThreadingHTTPServer((host, port), handler)


def _get_local_ip() -> Optional[str]:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal Flow Matching VC Web Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Listen port (default: 8080)")
    args = parser.parse_args()

    server = create_server(host=args.host, port=args.port)
    actual_port = server.server_address[1]
    print(f"VC WebGPU Server running at:")
    print(f"  Local:   http://localhost:{actual_port}")
    local_ip = _get_local_ip()
    if local_ip:
        print(f"  Network: http://{local_ip}:{actual_port}")
        print("  (Note: Microphone API on iPhone over LAN requires HTTPS or localhost)")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
