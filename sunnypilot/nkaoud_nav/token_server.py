"""
On-demand HTTP form for setting the Mapbox token from a phone/laptop browser.

The settings UI starts a TokenWebServer when the user opens the QR dialog
and stops it when they save or cancel. No standalone process is registered;
the server only exists while the dialog is up.

Threat model: LAN-only, no auth. Same as sunnypilot's copyparty. Don't
expose to public Wi-Fi. The server only runs while the dialog is visible.
"""
from __future__ import annotations

import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog


DEFAULT_PORT = 8081

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>nkaoud_nav: Mapbox token</title>
<style>
  body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:0;padding:24px;}
  h1{font-size:20px;margin:0 0 16px;}
  .status{padding:12px;border-radius:6px;margin:0 0 16px;background:#222;}
  .status.set{border-left:4px solid #80d8a6;}
  .status.unset{border-left:4px solid #c92231;}
  textarea{width:100%;height:140px;background:#1a1a1a;color:#eee;border:1px solid #333;
           border-radius:6px;padding:10px;font-family:ui-monospace,monospace;font-size:14px;
           box-sizing:border-box;}
  button{margin-top:12px;padding:14px 28px;border:0;border-radius:6px;background:#0086e9;
         color:#fff;font-size:16px;cursor:pointer;}
  button:active{background:#006bb8;}
  .hint{color:#888;font-size:13px;margin-top:8px;}
  .done{padding:16px;border-radius:6px;background:#163a26;border-left:4px solid #80d8a6;
        font-size:16px;margin-top:24px;}
</style></head><body>
<h1>nkaoud_nav &mdash; Mapbox token</h1>
%BODY%
</body></html>
"""

FORM_BODY = """
<div class="status %CLASS%">%STATUS%</div>
<form method="POST" action="/">
  <textarea name="token" placeholder="pk.eyJhbGciOiJI..." autofocus></textarea>
  <div class="hint">Public Mapbox tokens start with <code>pk.eyJ</code>.
  Whitespace is trimmed. Empty submit clears the token.</div>
  <button type="submit">Save token</button>
</form>
"""

DONE_BODY = """
<div class="done">Token saved. You can close this tab. The setup dialog on
the comma will close automatically.</div>
"""


def get_local_ip() -> str:
  """Best-effort: returns the device's outbound LAN IP without sending a packet.

  Falls back to 127.0.0.1 if no network is up.
  """
  s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  try:
    s.connect(("8.8.8.8", 80))
    return s.getsockname()[0]
  except OSError:
    return "127.0.0.1"
  finally:
    s.close()


def _render_form(message_html: str = "") -> bytes:
  token = (Params().get("NkaoudNavMapboxToken") or "").strip()
  if token:
    tail = token[-4:] if len(token) >= 4 else token
    status_cls = "set"
    status_txt = f"Token currently set (ends in &hellip;{tail}, length {len(token)})."
  else:
    status_cls = "unset"
    status_txt = "No token set yet."
  body = (FORM_BODY
          .replace("%CLASS%", status_cls)
          .replace("%STATUS%", status_txt))
  if message_html:
    body = message_html + body
  return PAGE.replace("%BODY%", body).encode("utf-8")


def _render_done() -> bytes:
  return PAGE.replace("%BODY%", DONE_BODY).encode("utf-8")


class TokenWebServer:
  """ThreadingHTTPServer that surfaces a 'token saved' Event."""

  def __init__(self, port: int = DEFAULT_PORT) -> None:
    self.port = port
    self.token_saved = threading.Event()
    self._server: ThreadingHTTPServer | None = None
    self._thread: threading.Thread | None = None

  def start(self) -> None:
    if self._server is not None:
      return
    saved_event = self.token_saved

    class _Handler(BaseHTTPRequestHandler):
      def log_message(self, fmt, *args):  # silence default access log
        pass

      def _send_html(self, body: bytes, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

      def do_GET(self):
        if self.path not in ("/", "/index.html"):
          self.send_error(404)
          return
        self._send_html(_render_done() if saved_event.is_set() else _render_form())

      def do_POST(self):
        if self.path != "/":
          self.send_error(404)
          return
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length > 0 else b""
        fields = parse_qs(body.decode("utf-8", errors="replace"))
        token = (fields.get("token", [""])[0]).strip()
        params = Params()
        if token:
          params.put("NkaoudNavMapboxToken", token)
          cloudlog.info(f"nkaoud_navd token_server: token updated via web form (len {len(token)})")
        else:
          params.remove("NkaoudNavMapboxToken")
          cloudlog.info("nkaoud_navd token_server: token cleared via web form")
        saved_event.set()
        self._send_html(_render_done())

    self._server = ThreadingHTTPServer(("0.0.0.0", self.port), _Handler)
    self._thread = threading.Thread(target=self._server.serve_forever, name="nkaoud_token_ws", daemon=True)
    self._thread.start()
    cloudlog.info(f"nkaoud_navd token_server: started on 0.0.0.0:{self.port}")

  def stop(self) -> None:
    if self._server is None:
      return
    try:
      self._server.shutdown()
      self._server.server_close()
    except OSError:
      pass
    self._server = None
    self._thread = None
    cloudlog.info("nkaoud_navd token_server: stopped")

  @property
  def url(self) -> str:
    return f"http://{get_local_ip()}:{self.port}/"
