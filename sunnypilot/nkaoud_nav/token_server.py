"""
On-demand HTTP form for setting a nkaoud_nav param from a phone/laptop browser.

The settings UI instantiates a ParamWebServer with the (param key, page
title, label, example, optional test handler) it wants, starts the server
when the QR dialog opens, and stops it when the user saves or cancels.
No standalone process is registered; the server only exists while the
dialog is up.

Two preconfigured factories:
  - mapbox_token_server() -- writes NkaoudNavMapboxToken
  - share_endpoint_server() -- writes NkaoudNavShareEndpoint; the spec
    also wires in a test handler that hits Neon /sql and returns the
    most recent rows so the user can confirm credentials + table.

Threat model: LAN-only, no auth. Same as sunnypilot's copyparty. Don't
expose to public Wi-Fi. The server only runs while the dialog is visible.
"""
from __future__ import annotations

import html
import json
import socket
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import requests

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog


DEFAULT_PORT = 8081

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>nkaoud_nav: %TITLE%</title>
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
  .hint{color:#888;font-size:13px;margin-top:8px;line-height:1.45;}
  .hint code{background:#1a1a1a;padding:1px 4px;border-radius:3px;color:#ddd;}
  .example{margin-top:10px;padding:10px;background:#1a1a1a;border:1px solid #2a2a2a;
           border-radius:6px;font-family:ui-monospace,monospace;font-size:12px;color:#ddd;
           white-space:pre-wrap;word-break:break-all;}
  .example .lbl{display:block;color:#888;font-family:system-ui,sans-serif;font-size:12px;
                margin-bottom:4px;}
  .actions{display:flex;gap:10px;flex-wrap:wrap;}
  button.secondary{background:#2a2a2a;color:#eee;}
  button.secondary:active{background:#444;}
  .result{margin-top:14px;padding:12px;border-radius:6px;font-family:ui-monospace,monospace;
          font-size:12px;white-space:pre-wrap;word-break:break-all;}
  .result.busy{background:#222;color:#ddd;border-left:4px solid #888;}
  .result.ok{background:#0f2a1a;color:#bee9c8;border-left:4px solid #80d8a6;}
  .result.err{background:#2a0f0f;color:#ffb3b3;border-left:4px solid #c92231;}
  .done{padding:16px;border-radius:6px;background:#163a26;border-left:4px solid #80d8a6;
        font-size:16px;margin-top:24px;}
</style></head><body>
<h1>nkaoud_nav &mdash; %TITLE%</h1>
%BODY%
</body></html>
"""

FORM_BODY = """
<div class="status %CLASS%">%STATUS%</div>
<form id="f" method="POST" action="/">
  <textarea id="value" name="value" placeholder="%PLACEHOLDER%" autofocus></textarea>
  <div class="hint">%HINT%</div>
  %EXAMPLE_BLOCK%
  <div class="actions">
    <button type="submit">Save</button>
    %TEST_BUTTON%
  </div>
</form>
<div id="result" class="result" hidden></div>
<script>
(function(){
  const btn = document.getElementById('test-btn');
  if(!btn) return;
  const ta = document.getElementById('value');
  const out = document.getElementById('result');
  btn.addEventListener('click', async () => {
    const val = ta.value.trim();
    if(!val){ out.hidden = false; out.className = 'result err'; out.textContent = 'paste a connection string first'; return; }
    out.hidden = false; out.className = 'result busy'; out.textContent = 'testing...';
    try{
      const r = await fetch('/test', { method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'}, body: 'value=' + encodeURIComponent(val) });
      const j = await r.json();
      if(j.ok){
        out.className = 'result ok';
        const body = (j.rows && j.rows.length) ? JSON.stringify(j.rows, null, 2) : '(query returned no rows -- table is empty?)';
        out.textContent = (j.message || 'OK') + '\\n\\n' + body;
      } else {
        out.className = 'result err';
        out.textContent = 'FAILED: ' + (j.error || 'unknown error');
      }
    }catch(e){
      out.className = 'result err';
      out.textContent = 'request failed: ' + e;
    }
  });
})();
</script>
"""

TEST_BUTTON_HTML = '<button id="test-btn" type="button" class="secondary">Test connection</button>'

DONE_BODY = """
<div class="done">Saved. You can close this tab. The setup dialog on
the comma will close automatically.</div>
"""


# Test handler signature: takes the raw textarea value, returns a dict
# {"ok": bool, "message"?: str, "rows"?: list, "error"?: str}.
TestHandler = Callable[[str], dict]


@dataclass(frozen=True)
class ParamWebFormSpec:
  """Everything the form needs to render + persist a single param."""
  param_key: str
  title: str                    # browser tab title and h1, e.g. "Mapbox token"
  placeholder: str              # textarea placeholder
  hint_html: str                # the "Public Mapbox tokens..." line; HTML allowed
  example_label: str = ""       # optional caption above the example block
  example_value: str = ""       # optional example string; rendered HTML-escaped
  status_set_template: str = "Currently set (length {length}, ends in &hellip;{tail})."
  status_unset: str = "Not set yet."
  # Optional in-form connection test. When set, the page renders a
  # "Test connection" button next to Save; clicking it POSTs to /test
  # which calls this handler with the current textarea value and renders
  # the result inline. Save is unaffected.
  test_handler: TestHandler | None = field(default=None, compare=False, repr=False)


def get_local_ip() -> str:
  """Best-effort: returns the device's outbound LAN IP without sending a packet."""
  s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  try:
    s.connect(("8.8.8.8", 80))
    return s.getsockname()[0]
  except OSError:
    return "127.0.0.1"
  finally:
    s.close()


def _render_form(spec: ParamWebFormSpec, message_html: str = "") -> bytes:
  current = (Params().get(spec.param_key) or "").strip()
  if current:
    tail = current[-4:] if len(current) >= 4 else current
    status_cls = "set"
    status_txt = spec.status_set_template.format(length=len(current), tail=html.escape(tail))
  else:
    status_cls = "unset"
    status_txt = spec.status_unset

  test_button_html = TEST_BUTTON_HTML if spec.test_handler is not None else ""

  if spec.example_value:
    example_block = (
      f'<div class="example"><span class="lbl">{html.escape(spec.example_label or "Example")}</span>'
      f'{html.escape(spec.example_value)}</div>'
    )
  else:
    example_block = ""

  body = (FORM_BODY
          .replace("%CLASS%", status_cls)
          .replace("%STATUS%", status_txt)
          .replace("%PLACEHOLDER%", html.escape(spec.placeholder))
          .replace("%HINT%", spec.hint_html)
          .replace("%EXAMPLE_BLOCK%", example_block)
          .replace("%TEST_BUTTON%", test_button_html))
  if message_html:
    body = message_html + body
  return PAGE.replace("%TITLE%", html.escape(spec.title)).replace("%BODY%", body).encode("utf-8")


def _render_done(spec: ParamWebFormSpec) -> bytes:
  return PAGE.replace("%TITLE%", html.escape(spec.title)).replace("%BODY%", DONE_BODY).encode("utf-8")


class ParamWebServer:
  """ThreadingHTTPServer that surfaces a 'saved' Event after a POST."""

  def __init__(self, spec: ParamWebFormSpec, port: int = DEFAULT_PORT) -> None:
    self.spec = spec
    self.port = port
    self.token_saved = threading.Event()   # kept name for backward compatibility with dialog
    self._server: ThreadingHTTPServer | None = None
    self._thread: threading.Thread | None = None

  def start(self) -> None:
    if self._server is not None:
      return
    spec = self.spec
    saved_event = self.token_saved

    class _Handler(BaseHTTPRequestHandler):
      def log_message(self, fmt, *args):  # silence default access log
        pass

      def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

      def _send_html(self, body: bytes, code: int = 200) -> None:
        self._send(body, "text/html; charset=utf-8", code=code)

      def _send_json(self, payload: dict, code: int = 200) -> None:
        self._send(json.dumps(payload).encode("utf-8"), "application/json", code=code)

      def _read_form_value(self) -> str:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length > 0 else b""
        fields = parse_qs(body.decode("utf-8", errors="replace"))
        # Accept either "value" (current spec) or "token" (old form name) so
        # an old cached page from a phone doesn't break.
        return (fields.get("value", fields.get("token", [""]))[0]).strip()

      def do_GET(self):
        if self.path not in ("/", "/index.html"):
          self.send_error(404)
          return
        self._send_html(_render_done(spec) if saved_event.is_set() else _render_form(spec))

      def do_POST(self):
        if self.path == "/test":
          if spec.test_handler is None:
            self._send_json({"ok": False, "error": "no test handler for this param"}, code=400)
            return
          value = self._read_form_value()
          if not value:
            self._send_json({"ok": False, "error": "empty value"})
            return
          try:
            result = spec.test_handler(value)
          except Exception as e:   # noqa: BLE001 -- surface anything the handler raised
            result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
          ok = bool(result.get("ok"))
          cloudlog.info(f"nkaoud_navd web form: /test {spec.param_key} ok={ok}")
          self._send_json(result)
          return
        if self.path != "/":
          self.send_error(404)
          return
        value = self._read_form_value()
        params = Params()
        if value:
          params.put(spec.param_key, value)
          cloudlog.info(f"nkaoud_navd web form: {spec.param_key} updated (len {len(value)})")
        else:
          params.remove(spec.param_key)
          cloudlog.info(f"nkaoud_navd web form: {spec.param_key} cleared")
        saved_event.set()
        self._send_html(_render_done(spec))

    self._server = ThreadingHTTPServer(("0.0.0.0", self.port), _Handler)
    self._thread = threading.Thread(target=self._server.serve_forever,
                                    name=f"nkaoud_param_ws_{self.spec.param_key}", daemon=True)
    self._thread.start()
    cloudlog.info(f"nkaoud_navd web form: started on 0.0.0.0:{self.port} for {spec.param_key}")

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
    cloudlog.info(f"nkaoud_navd web form: stopped ({self.spec.param_key})")

  @property
  def url(self) -> str:
    return f"http://{get_local_ip()}:{self.port}/"


# ----- Preconfigured factories -----

MAPBOX_TOKEN_SPEC = ParamWebFormSpec(
  param_key="NkaoudNavMapboxToken",
  title="Mapbox token",
  placeholder="pk.eyJhbGciOiJI...",
  hint_html=(
    'Public Mapbox tokens start with <code>pk.eyJ</code>. '
    'Whitespace is trimmed. Empty submit clears the token.'
  ),
  status_set_template="Token currently set (length {length}, ends in &hellip;{tail}).",
  status_unset="No token set yet.",
)

# `SELECT *` so the test survives any table schema as long as the user
# has at least latitude / longitude columns. The reshape below uses the
# response's `fields` metadata to label each value with its column name.
NEON_TEST_QUERY = "SELECT * FROM destinations ORDER BY id DESC LIMIT 5"


def _neon_test(connection_string: str) -> dict:
  """Hit Neon /sql with the supplied connection string and pull the 5 most
  recent destinations. Used by the Share-endpoint form's Test button so the
  user can verify credentials + table layout without leaving the page."""
  parsed = urlparse(connection_string)
  scheme = parsed.scheme.lower()
  if scheme not in ("postgres", "postgresql"):
    return {"ok": False, "error": f"expected postgresql://... (got scheme {scheme!r})"}
  host = parsed.hostname
  if not host:
    return {"ok": False, "error": "connection string is missing a hostname"}
  url = f"https://{host}/sql"
  try:
    resp = requests.post(
      url,
      timeout=8,
      headers={
        "Neon-Connection-String": connection_string,
        "Neon-Raw-Text-Output": "true",
        "Neon-Array-Mode": "true",
      },
      json={"query": NEON_TEST_QUERY, "params": []},
    )
  except requests.RequestException as e:
    return {"ok": False, "error": f"network error: {e}"}
  if resp.status_code != 200:
    return {"ok": False, "error": f"neon http {resp.status_code}: {resp.text[:500]}"}
  try:
    body = resp.json()
  except ValueError as e:
    return {"ok": False, "error": f"neon response was not JSON: {e}"}
  rows = body.get("rows") or []
  fields = [f.get("name") for f in (body.get("fields") or []) if isinstance(f, dict)]
  # Reshape into list-of-dicts when we have field names so the user
  # sees columns labelled.
  pretty: list = []
  if fields and rows and isinstance(rows[0], list):
    for r in rows:
      pretty.append({fields[i]: r[i] for i in range(min(len(fields), len(r)))})
  else:
    pretty = rows
  return {
    "ok": True,
    "message": f"Neon responded OK with {len(rows)} row(s) from `destinations`.",
    "rows": pretty,
  }


SHARE_ENDPOINT_SPEC = ParamWebFormSpec(
  param_key="NkaoudNavShareEndpoint",
  title="Neon connection string",
  placeholder="postgresql://USER:PASS@HOST/DB?sslmode=require&channel_binding=require",
  hint_html=(
    'Paste your Neon database connection string, then use <code>Test '
    'connection</code> to fetch the latest rows from your '
    '<code>destinations</code> table before saving. When you tap '
    '<code>Share</code> on the comma, nkaoud_nav uses the most recent row. '
    'Whitespace trimmed; empty submit clears the connection string.'
  ),
  example_label="What to paste, and what the test does",
  example_value=(
    "Connection string (one line, from your Neon dashboard):\n"
    "  postgresql://neondb_owner:npg_XxXxXxXxX@ep-gentle-bonus-aqrtri2b\n"
    "    -pooler.c-8.us-east-1.aws.neon.tech/neondb\n"
    "    ?sslmode=require&channel_binding=require\n\n"
    "Tapping `Test connection` runs ONLY a SELECT (no CREATE, no INSERT):\n"
    "  SELECT * FROM destinations ORDER BY id DESC LIMIT 5\n\n"
    "Required columns in `destinations`:\n"
    "  latitude   DOUBLE PRECISION\n"
    "  longitude  DOUBLE PRECISION\n"
    "  id         (any sortable column used for `ORDER BY id DESC`)\n\n"
    "Optional label column -- first match wins, falls back to\n"
    '"Shared destination" if none exist:\n'
    "  place_name | name | label | title\n\n"
    "Extra columns are ignored. nkaoud_nav reads the most recent row\n"
    "every time you tap Share on the comma.\n\n"
    "Security: the connection string is stored in cleartext on the device.\n"
    "Rotate the Neon password if the device ever leaves your control."
  ),
  status_set_template="Connection string set (length {length}, ends in &hellip;{tail}).",
  status_unset="No connection string set yet.",
  test_handler=_neon_test,
)


def mapbox_token_server(port: int = DEFAULT_PORT) -> ParamWebServer:
  return ParamWebServer(MAPBOX_TOKEN_SPEC, port=port)


def share_endpoint_server(port: int = DEFAULT_PORT) -> ParamWebServer:
  return ParamWebServer(SHARE_ENDPOINT_SPEC, port=port)


# Back-compat alias so existing imports of TokenWebServer keep working.
TokenWebServer = mapbox_token_server
