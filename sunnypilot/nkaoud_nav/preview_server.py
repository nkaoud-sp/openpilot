"""
Tiny read-only HTTP server that serves the latest visual vehicle detector
preview (letterboxed RGB tensor) as a PNG so the user can sanity-check the
NV12->RGB conversion and letterbox on their phone.

Pairs with adjacent_vehicle_detector.py: the detector writes
/tmp/nkaoud_vvd_preview.png on every frame *only while* the request sentinel
/tmp/nkaoud_vvd_preview.request exists. This server creates the sentinel on
start() and removes it on stop(), so the per-frame write cost stays at zero
whenever the dialog is closed.

The public interface intentionally mirrors token_server.ParamWebServer
(start/stop/url/token_saved) so the existing NavParamQrDialog renders it
unchanged. token_saved is never set -- the dialog stays on QR view until the
user taps Cancel.
"""
from __future__ import annotations

import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.nkaoud_nav.adjacent_vehicle_detector import (
  PREVIEW_PNG_PATH, PREVIEW_REQUEST_PATH,
)
from openpilot.sunnypilot.nkaoud_nav.token_server import get_local_ip


DEFAULT_PORT = 8082


PAGE = b"""<!doctype html>
<html><head><meta charset="utf-8"><title>Visual Vehicle Detector Preview</title>
<style>
  body { background:#111; color:#ddd; font-family:sans-serif; margin:0; padding:24px; text-align:center; }
  h2 { margin:0 0 8px 0; }
  p { margin:0 0 16px 0; color:#888; font-size:14px; }
  img { max-width:95vw; max-height:75vh; background:#222; border:1px solid #333;
        image-rendering:pixelated; image-rendering:crisp-edges; }
  #stamp { margin-top:8px; color:#666; font-size:12px; font-family:monospace; }
</style></head>
<body>
  <h2>Visual Vehicle Detector Preview</h2>
  <p>Letterboxed 320x320 RGB tensor the model sees. Should look like a normal road scene.</p>
  <div><img id="p" src="/preview.png" alt="(no preview yet -- start the detector and drive offroad demo)"></div>
  <div id="stamp">--</div>
  <script>
    const img = document.getElementById('p');
    const stamp = document.getElementById('stamp');
    function tick() {
      const t = Date.now();
      img.src = '/preview.png?t=' + t;
      stamp.textContent = new Date().toLocaleTimeString();
    }
    img.addEventListener('error', () => { stamp.textContent = 'waiting for detector...'; });
    setInterval(tick, 1000);
  </script>
</body></html>
"""


class PreviewWebServer:
  """Read-only image server. Mirrors ParamWebServer's interface."""

  def __init__(self, port: int = DEFAULT_PORT) -> None:
    self.port = port
    # Kept name `token_saved` for NavParamQrDialog compatibility; never set here.
    self.token_saved = threading.Event()
    self._server: ThreadingHTTPServer | None = None
    self._thread: threading.Thread | None = None

  def start(self) -> None:
    if self._server is not None:
      return

    # Tell the detector to start dumping the preview PNG on each frame.
    try:
      with open(PREVIEW_REQUEST_PATH, "w") as f:
        f.write("")
    except OSError:
      cloudlog.exception("nkaoud_vvd preview: failed to touch request sentinel")

    class _Handler(BaseHTTPRequestHandler):
      def log_message(self, fmt, *args):
        pass

      def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

      def do_GET(self):
        if self.path in ("/", "/index.html"):
          self._send(PAGE, "text/html; charset=utf-8")
          return
        if self.path.split("?", 1)[0] == "/preview.png":
          try:
            with open(PREVIEW_PNG_PATH, "rb") as f:
              data = f.read()
          except FileNotFoundError:
            self.send_error(404, "preview not yet written")
            return
          except OSError as e:
            self.send_error(500, f"preview read failed: {e}")
            return
          self._send(data, "image/png")
          return
        self.send_error(404)

    self._server = ThreadingHTTPServer(("0.0.0.0", self.port), _Handler)
    self._thread = threading.Thread(target=self._server.serve_forever,
                                    name="nkaoud_vvd_preview_ws", daemon=True)
    self._thread.start()
    cloudlog.info(f"nkaoud_vvd preview web: started on 0.0.0.0:{self.port}")

  def stop(self) -> None:
    if self._server is not None:
      try:
        self._server.shutdown()
        self._server.server_close()
      except OSError:
        pass
      self._server = None
      self._thread = None
      cloudlog.info("nkaoud_vvd preview web: stopped")

    try:
      os.remove(PREVIEW_REQUEST_PATH)
    except FileNotFoundError:
      pass
    except OSError:
      cloudlog.exception("nkaoud_vvd preview: failed to remove request sentinel")

  @property
  def url(self) -> str:
    return f"http://{get_local_ip()}:{self.port}/"


def visual_vehicle_preview_server(port: int = DEFAULT_PORT) -> PreviewWebServer:
  return PreviewWebServer(port=port)
