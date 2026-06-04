#!/usr/bin/env python3
"""
Tiny offroad HTTP form for setting the Mapbox token without typing it on
the on-screen keyboard. Serves a single page at
  http://<comma3x-ip>:8081/
with a textarea + submit button. POSTs land in the NkaoudNavMapboxToken
param.

No auth -- same threat model as sunnypilot's copyparty (LAN-only,
offroad-only). Don't run on public Wi-Fi.
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog


PORT = 8081

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
  .ok{color:#80d8a6;}
  .err{color:#ff8a8a;}
</style></head><body>
<h1>nkaoud_nav &mdash; Mapbox token</h1>
<div class="status %CLASS%">%STATUS%</div>
%MESSAGE%
<form method="POST" action="/">
  <textarea name="token" placeholder="pk.eyJhbGciOiJI..." autofocus></textarea>
  <div class="hint">Public Mapbox tokens start with <code>pk.eyJ</code>. Whitespace is trimmed.
  Submit replaces the stored token; leave blank and submit to clear it.</div>
  <button type="submit">Save token</button>
</form>
</body></html>
"""


def _render(message_html: str = "") -> bytes:
  token = (Params().get("NkaoudNavMapboxToken") or "").strip()
  if token:
    tail = token[-4:] if len(token) >= 4 else token
    status_cls = "set"
    status_txt = f"Token currently set (ends in &hellip;{tail}, length {len(token)})."
  else:
    status_cls = "unset"
    status_txt = "No token set yet."
  return (PAGE
          .replace("%CLASS%", status_cls)
          .replace("%STATUS%", status_txt)
          .replace("%MESSAGE%", message_html)
          .encode("utf-8"))


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
    self._send_html(_render())

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
      msg = '<p class="ok">Saved. navd will pick it up on the next tick.</p>'
    else:
      params.remove("NkaoudNavMapboxToken")
      cloudlog.info("nkaoud_navd token_server: token cleared via web form")
      msg = '<p class="err">Token cleared.</p>'
    self._send_html(_render(msg))


def main() -> None:
  server = ThreadingHTTPServer(("0.0.0.0", PORT), _Handler)
  cloudlog.info(f"nkaoud_navd token_server: listening on 0.0.0.0:{PORT}")
  try:
    server.serve_forever()
  finally:
    server.server_close()


if __name__ == "__main__":
  main()
