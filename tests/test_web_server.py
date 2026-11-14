"""Tests for the WebGPU Voice Changer local HTTP server."""

import sys
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.server import create_server


class TestWebServer(unittest.TestCase):
    """Verify static assets, ONNX models, COOP/COEP headers, and path isolation."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = create_server(host="127.0.0.1", port=0, project_root=PROJECT_ROOT)
        cls.port = cls.server.server_address[1]
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=3.0)

    def _fetch(self, path: str, head_only: bool = False):
        encoded_path = urllib.parse.quote(path, safe="/.")
        req = urllib.request.Request(
            f"{self.base_url}{encoded_path}",
            method="HEAD" if head_only else "GET",
        )
        return urllib.request.urlopen(req, timeout=5.0)

    def test_routes_and_headers(self) -> None:
        cases = [
            ("/", "text/html"),
            ("/web/styles.css", "text/css"),
            ("/web/app.js", "application/javascript"),
            ("/web/vc-audio-worklet.js", "application/javascript"),
            ("/models/speaker_encoder.onnx", "application/octet-stream"),
            ("/models/streaming_vc_core.onnx", "application/octet-stream"),
        ]
        for path, expected_mime in cases:
            with self.subTest(path=path):
                with self._fetch(path, head_only=path.endswith(".onnx")) as resp:
                    self.assertEqual(resp.status, 200)
                    ctype = resp.headers.get("Content-Type", "")
                    self.assertIn(expected_mime, ctype)
                    self.assertEqual(
                        resp.headers.get("Cross-Origin-Opener-Policy"),
                        "same-origin",
                    )
                    self.assertEqual(
                        resp.headers.get("Cross-Origin-Embedder-Policy"),
                        "credentialless",
                    )
                    self.assertEqual(
                        resp.headers.get("Access-Control-Allow-Origin"),
                        "*",
                    )

    def test_disallowed_paths_return_404(self) -> None:
        for bad_path in ["/src/server.py", "/発見メモ.md", "/../README.md"]:
            with self.subTest(path=bad_path):
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    self._fetch(bad_path)
                self.assertEqual(ctx.exception.code, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
