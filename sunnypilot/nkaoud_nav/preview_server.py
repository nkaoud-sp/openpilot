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

import json
import os
import threading
import zipfile
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.nkaoud_nav.adjacent_vehicle_detector import (
  BUF_GEOMETRY_PATH, CAPTURE_MAX_BYTES, CAPTURE_MAX_FILES, MAX_DETECTOR_HZ, PREVIEW_DETECTOR_CROP_PATH,
  PREVIEW_FULL_FRAME_CROP_PATH, PREVIEW_MODEL_INPUT_PATH, PREVIEW_PNG_PATH,
  PREVIEW_PNG_PATH_FULL, PREVIEW_PNG_PATH_LIMITED, PREVIEW_RAW_U_PATH, PREVIEW_RAW_V_PATH,
  PREVIEW_RAW_Y_PATH, PREVIEW_REQUEST_PATH, TUNING_DEFAULTS, TUNING_KEYS,
  active_camera, capture_delete_all, capture_files, capture_hz, capture_set_request,
  capture_stats, frame_info, load_tuning, save_tuning,
)
from openpilot.sunnypilot.nkaoud_nav.token_server import get_local_ip


DEFAULT_PORT = 8082
STAGES_PORT = 8083
TUNING_PORT = 8084
CAPTURE_PORT = 8086


PAGE = b"""<!doctype html>
<html><head><meta charset="utf-8"><title>Visual Vehicle Detector Preview</title>
<style>
  body { background:#111; color:#ddd; font-family:sans-serif; margin:0; padding:16px; }
  h2 { margin:0 0 4px 0; text-align:center; }
  p.lead { margin:0 0 16px 0; color:#888; font-size:14px; text-align:center; }
  .row { display:flex; gap:16px; justify-content:center; flex-wrap:wrap; }
  .col { display:flex; flex-direction:column; align-items:center; }
  .label { margin-bottom:6px; font-family:monospace; font-size:14px; padding:4px 10px; border-radius:4px; }
  .label.good { background:#1d3a23; color:#9bdca8; }
  .label.bad  { background:#3a1d1d; color:#dc9b9b; }
  img { width:46vw; max-width:520px; min-width:280px; background:#222; border:1px solid #333;
        image-rendering:pixelated; image-rendering:crisp-edges; }
  @media (max-width:700px) { img { width:92vw; max-width:none; } }
  #stamp { margin-top:12px; color:#666; font-size:12px; font-family:monospace; text-align:center; }
</style></head>
<body>
  <h2>Visual Vehicle Detector Preview</h2>
  <p class="lead">Letterboxed 320x320 RGB tensor. Left = correct (full range), right = old buggy (limited range).</p>
  <div class="row">
    <div class="col">
      <div class="label good">BT.601 full range &mdash; production</div>
      <img id="pf" src="/preview_full.png" alt="(waiting...)">
    </div>
    <div class="col">
      <div class="label bad">BT.601 limited range &mdash; old (for comparison)</div>
      <img id="pl" src="/preview_limited.png" alt="(waiting...)">
    </div>
  </div>
  <h3 style="text-align:center; margin:24px 0 4px 0;">Raw planes (grayscale)</h3>
  <p class="lead">Y is luma. U bright = blue-ish, dark = yellow-ish. V bright = red-ish, dark = cyan-ish. Watch for shearing, doubled rows, or speckle in flat regions -- those are stride / alignment bugs.</p>
  <div class="row">
    <div class="col">
      <div class="label good">Y (luma, full-res)</div>
      <img id="ry" src="/raw_y.png" alt="(waiting...)">
    </div>
    <div class="col">
      <div class="label good">U (chroma, quarter-res)</div>
      <img id="ru" src="/raw_u.png" alt="(waiting...)">
    </div>
    <div class="col">
      <div class="label good">V (chroma, quarter-res)</div>
      <img id="rv" src="/raw_v.png" alt="(waiting...)">
    </div>
  </div>
  <div id="stamp">--</div>
  <pre id="geom" style="margin:16px auto; max-width:900px; background:#1a1a1a; color:#9ec; padding:12px;
       border:1px solid #333; font-size:13px; text-align:left; white-space:pre-wrap; word-break:break-all;">
loading buffer geometry...</pre>
  <script>
    const pf = document.getElementById('pf');
    const pl = document.getElementById('pl');
    const stamp = document.getElementById('stamp');
    const geom = document.getElementById('geom');
    const ry = document.getElementById('ry');
    const ru = document.getElementById('ru');
    const rv = document.getElementById('rv');
    function tick() {
      const t = Date.now();
      pf.src = '/preview_full.png?t=' + t;
      pl.src = '/preview_limited.png?t=' + t;
      ry.src = '/raw_y.png?t=' + t;
      ru.src = '/raw_u.png?t=' + t;
      rv.src = '/raw_v.png?t=' + t;
      stamp.textContent = new Date().toLocaleTimeString();
    }
    function refreshGeom() {
      fetch('/geometry.json?t=' + Date.now()).then(r => r.ok ? r.json() : null).then(j => {
        if (!j) { geom.textContent = 'geometry not yet written -- waiting for a frame...'; return; }
        const lines = [
          'NV12 buffer geometry (from VisionBuf):',
          '  width            = ' + j.width,
          '  height           = ' + j.height,
          '  stride           = ' + j.stride + '  (Y row pitch in bytes)',
          '  uv_offset        = ' + j.uv_offset + '  (UV plane start)',
          '  uv_height        = ' + j.uv_height + '  (UV plane rows)',
          '  uv_plane_size    = ' + j.uv_plane_size,
          '  data_len         = ' + j.data_len,
          '  y_plane (h*s)    = ' + (j.stride * j.height),
          '  y_plane (32-al)  = ' + (j.stride * ((Math.floor((j.height + 31) / 32)) * 32)),
          '  uv_offset == y_plane?         ' + (j.uv_offset_matches_y_plane ? 'YES' : 'no'),
          '  uv_offset == y_plane(32-al)?  ' + (j.uv_offset_matches_y_plane_aligned ? 'YES' : 'no'),
        ];
        geom.textContent = lines.join('\\n');
      }).catch(() => { geom.textContent = 'geometry fetch failed'; });
    }
    pf.addEventListener('error', () => { stamp.textContent = 'waiting for detector...'; });
    setInterval(tick, 1000);
    setInterval(refreshGeom, 2000);
    refreshGeom();
  </script>
</body></html>
"""


# Route -> on-disk PNG written by the detector. The detector only writes these
# while PREVIEW_REQUEST_PATH exists, which this server creates on start().
PREVIEW_ROUTES = {
  "/preview.png":          PREVIEW_PNG_PATH,
  "/preview_full.png":     PREVIEW_PNG_PATH_FULL,
  "/preview_limited.png":  PREVIEW_PNG_PATH_LIMITED,
  "/raw_y.png":            PREVIEW_RAW_Y_PATH,
  "/raw_u.png":            PREVIEW_RAW_U_PATH,
  "/raw_v.png":            PREVIEW_RAW_V_PATH,
}

# Pipeline-stage images: full frame -> crop (before YOLO) -> letterboxed model
# input fed to YOLO (with ROI + detection boxes drawn).
STAGES_ROUTES = {
  "/full_frame.png":       PREVIEW_FULL_FRAME_CROP_PATH,
  "/detector_crop.png":    PREVIEW_DETECTOR_CROP_PATH,
  "/model_input.png":      PREVIEW_MODEL_INPUT_PATH,
}


STAGES_PAGE = b"""<!doctype html>
<html><head><meta charset="utf-8"><title>Visual Vehicle Detector Stages</title>
<style>
  body { background:#111; color:#ddd; font-family:sans-serif; margin:0; padding:16px; }
  h2 { margin:0 0 4px 0; text-align:center; }
  p.lead { margin:0 0 16px 0; color:#888; font-size:14px; text-align:center; }
  .row { display:flex; gap:16px; justify-content:center; flex-wrap:wrap; }
  .col { display:flex; flex-direction:column; align-items:center; max-width:560px; }
  .label { margin-bottom:6px; font-family:monospace; font-size:14px; padding:4px 10px;
           border-radius:4px; background:#1d2a3a; color:#9bc3dc; }
  .desc { margin:4px 0 0 0; color:#777; font-size:12px; text-align:center; }
  img { width:46vw; max-width:540px; min-width:280px; background:#222; border:1px solid #333;
        image-rendering:pixelated; image-rendering:crisp-edges; }
  @media (max-width:700px) { img { width:92vw; max-width:none; } }
  #stamp { margin-top:12px; color:#666; font-size:12px; font-family:monospace; text-align:center; }
</style></head>
<body>
  <h2>Visual Vehicle Detector &mdash; Pipeline Stages</h2>
  <p class="lead">Each frame as it moves through the detector. ROI overlay: red = LEFT, blue = RIGHT. Green boxes = vehicle detections.</p>
  <div class="row">
    <div class="col">
      <div class="label">1. Full camera frame</div>
      <img id="s1" src="/full_frame.png" alt="(waiting...)">
      <p class="desc">Whole wide-road frame with the fixed 928x416 detector crop drawn (yellow).</p>
    </div>
    <div class="col">
      <div class="label">2. Detector crop &mdash; before YOLO</div>
      <img id="s2" src="/detector_crop.png" alt="(waiting...)">
      <p class="desc">The raw cropped region in crop coordinates, with the LEFT/RIGHT ROI split.</p>
    </div>
  </div>
  <div class="row" style="margin-top:16px;">
    <div class="col">
      <div class="label">3. Model input &mdash; fed to YOLO</div>
      <img id="s3" src="/model_input.png" alt="(waiting...)">
      <p class="desc">Exact letterboxed/resized tensor YOLO sees, with ROI and detection boxes mapped back.</p>
    </div>
  </div>
  <div id="stamp">--</div>
  <script>
    const ids = ['s1', 's2', 's3'];
    const imgs = ids.map(i => document.getElementById(i));
    const stamp = document.getElementById('stamp');
    const srcs = ['/full_frame.png', '/detector_crop.png', '/model_input.png'];
    function tick() {
      const t = Date.now();
      imgs.forEach((img, i) => { img.src = srcs[i] + '?t=' + t; });
      stamp.textContent = new Date().toLocaleTimeString();
    }
    imgs[0].addEventListener('error', () => { stamp.textContent = 'waiting for detector...'; });
    setInterval(tick, 1000);
    tick();
  </script>
</body></html>
"""


class PreviewWebServer:
  """Read-only image server. Mirrors ParamWebServer's interface.

  `page`/`routes` let the same server back either the color-conversion preview
  or the pipeline-stage view; both rely on the detector writing PNGs while the
  shared PREVIEW_REQUEST_PATH sentinel exists.
  """

  def __init__(self, port: int = DEFAULT_PORT, page: bytes = PAGE,
               routes: dict[str, str] | None = None,
               json_routes: dict[str, Callable[[], bytes]] | None = None,
               post_handler: Callable[[str, bytes], tuple[int, str, bytes]] | None = None) -> None:
    self.port = port
    self._page = page
    self._routes = PREVIEW_ROUTES if routes is None else routes
    self._json_routes = json_routes or {}
    self._post_handler = post_handler
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

    page = self._page
    routes = self._routes
    json_routes = self._json_routes
    post_handler = self._post_handler

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

      def do_POST(self):
        if post_handler is None:
          self.send_error(404)
          return
        route = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else b""
        try:
          code, ctype, resp = post_handler(route, body)
        except Exception as e:
          self.send_error(500, f"post failed: {e}")
          return
        self._send(resp, ctype, code)

      def do_GET(self):
        if self.path in ("/", "/index.html"):
          self._send(page, "text/html; charset=utf-8")
          return
        route = self.path.split("?", 1)[0]
        json_fn = json_routes.get(route)
        if json_fn is not None:
          try:
            self._send(json_fn(), "application/json")
          except Exception as e:
            self.send_error(500, f"json failed: {e}")
          return
        png_path = routes.get(route)
        if png_path is not None:
          try:
            with open(png_path, "rb") as f:
              data = f.read()
          except FileNotFoundError:
            self.send_error(404, "preview not yet written")
            return
          except OSError as e:
            self.send_error(500, f"preview read failed: {e}")
            return
          self._send(data, "image/png")
          return
        if route == "/geometry.json":
          try:
            with open(BUF_GEOMETRY_PATH, "rb") as f:
              data = f.read()
          except FileNotFoundError:
            self.send_error(404, "geometry not yet written")
            return
          except OSError as e:
            self.send_error(500, f"geometry read failed: {e}")
            return
          self._send(data, "application/json")
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


def visual_vehicle_stages_server(port: int = STAGES_PORT) -> PreviewWebServer:
  return PreviewWebServer(port=port, page=STAGES_PAGE, routes=STAGES_ROUTES)


CROP_PORT = 8085

TUNING_ROUTES = {
  "/processed.png":     PREVIEW_MODEL_INPUT_PATH,
  "/detector_crop.png": PREVIEW_DETECTOR_CROP_PATH,
}
CROP_ROUTES = {
  "/full_frame.png":    PREVIEW_FULL_FRAME_CROP_PATH,
  "/detector_crop.png": PREVIEW_DETECTOR_CROP_PATH,
}

# Which sliders appear on which portal. They all write the same tuning file, so
# each portal just shows its own subset.
TUNE_KEYS = ["right_x1", "right_y1", "right_x2", "right_y2",
             "min_box_w", "min_box_h", "min_bottom_y", "confidence"]
CROP_KEYS = ["crop_x", "crop_y", "crop_w", "crop_h", "hz", "blocked_threshold"]


def _tuning_json() -> bytes:
  # "_camera"/"_frame_*" let the portal show the active profile and auto-range
  # the crop sliders to the live stream; apply() ignores non-slider keys.
  info = frame_info()
  return json.dumps({**load_tuning(), "_camera": active_camera(),
                     "_frame_w": info.get("frame_w"), "_frame_h": info.get("frame_h")}).encode()


def _tuning_post(route: str, body: bytes) -> tuple[int, str, bytes]:
  if route == "/reset":
    return 200, "application/json", json.dumps(save_tuning(dict(TUNING_DEFAULTS))).encode()
  if route != "/tuning":
    return 404, "text/plain; charset=utf-8", b"not found"
  try:
    updates = json.loads(body or b"{}")
  except ValueError:
    return 400, "text/plain; charset=utf-8", b"bad json"
  if not isinstance(updates, dict):
    return 400, "text/plain; charset=utf-8", b"expected object"
  return 200, "application/json", json.dumps(save_tuning(updates)).encode()


# Slider config drives the controls; ranges are display-only (save_tuning still
# clamps to TUNING_KEYS).
_SLIDER_RANGES = {
  "right_x1":     (0.0, 1.0, 0.01),
  "right_x2":     (0.0, 1.0, 0.01),
  "right_y1":     (0.0, 1.0, 0.01),
  "right_y2":     (0.0, 1.0, 0.01),
  "min_box_w":    (0.0, 0.6, 0.01),
  "min_box_h":    (0.0, 0.8, 0.01),
  "min_bottom_y": (0.0, 1.0, 0.01),
  "confidence":   (0.05, 0.9, 0.01),
  "crop_x":       (0.0, 1928.0, 1.0),
  "crop_y":       (0.0, 1208.0, 1.0),
  "crop_w":       (64.0, 1928.0, 1.0),
  "crop_h":       (64.0, 1208.0, 1.0),
  "hz":           (1.0, float(MAX_DETECTOR_HZ), 1.0),
  "blocked_threshold": (0.05, 0.95, 0.01),
}
_SLIDER_LABELS = {
  "right_x1":     "Right ROI  left edge (x1)",
  "right_x2":     "Right ROI  right edge (x2)",
  "right_y1":     "Right ROI  top edge (y1)",
  "right_y2":     "Right ROI  bottom edge (y2)",
  "min_box_w":    "Min box width (far-lane reject)",
  "min_box_h":    "Min box height (far-lane reject)",
  "min_bottom_y": "Min bottom-y (horizon reject)",
  "confidence":   "Detection confidence",
  "crop_x":       "Crop box  X (px)",
  "crop_y":       "Crop box  Y (px)",
  "crop_w":       "Crop box  width (px)",
  "crop_h":       "Crop box  height (px)",
  "hz":           "Detector rate (Hz)",
  "blocked_threshold": "Blocked threshold (driver cam)",
}


def _sliders_json(keys: list[str]) -> str:
  return json.dumps([
    {"key": k, "min": _SLIDER_RANGES[k][0], "max": _SLIDER_RANGES[k][1],
     "step": _SLIDER_RANGES[k][2], "label": _SLIDER_LABELS[k]}
    for k in keys
  ])


_PAGE_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  body { background:#111; color:#ddd; font-family:sans-serif; margin:0; padding:16px; }
  h2 { margin:0 0 4px 0; text-align:center; }
  p.lead { margin:0 0 12px 0; color:#888; font-size:13px; text-align:center; }
  .imgs { display:flex; gap:12px; justify-content:center; flex-wrap:wrap; }
  .col { display:flex; flex-direction:column; align-items:center; }
  .cap { font-family:monospace; font-size:12px; color:#9bc3dc; margin-bottom:4px; }
  img { width:46vw; max-width:540px; min-width:280px; background:#222; border:1px solid #333;
        image-rendering:pixelated; image-rendering:crisp-edges; }
  @media (max-width:700px) { img { width:92vw; max-width:none; } }
  .controls { max-width:760px; margin:16px auto 0 auto; }
  .ctl { margin:10px 0; }
  .ctl .row { display:flex; justify-content:space-between; font-size:13px; margin-bottom:2px; }
  .ctl .val { font-family:monospace; color:#9bdca8; }
  input[type=range] { width:100%; height:34px; }
  .btns { text-align:center; margin-top:14px; }
  button { background:#243; color:#cfe; border:1px solid #365; border-radius:6px;
           padding:10px 18px; font-size:15px; }
  #stamp { margin-top:10px; color:#666; font-size:12px; font-family:monospace; text-align:center; }
</style></head>
<body>
  <h2>__TITLE__</h2>
  <p class="lead">__LEAD__</p>
  <p class="lead">Camera profile: <b id="cam" style="color:#9bdca8;">--</b> (set the camera under Vision Detection settings)</p>
  <div class="imgs">__IMAGES_HTML__</div>
  <div class="controls" id="controls"></div>
  <div class="btns"><button id="reset">Reset to defaults</button></div>
  <div id="stamp">--</div>
  <script>
    const SLIDERS = __SLIDERS__;
    const IMAGES = __IMAGES__;
    const DEFAULTS = __DEFAULTS__;
    const controls = document.getElementById('controls');
    const els = {};
    function fmt(s, v) { return parseFloat(v).toFixed(s.step < 1 ? 2 : 0); }
    SLIDERS.forEach(s => {
      const wrap = document.createElement('div'); wrap.className = 'ctl';
      const row = document.createElement('div'); row.className = 'row';
      const lab = document.createElement('span'); lab.textContent = s.label;
      const val = document.createElement('span'); val.className = 'val'; val.textContent = '--';
      row.appendChild(lab); row.appendChild(val);
      const inp = document.createElement('input');
      inp.type = 'range'; inp.min = s.min; inp.max = s.max; inp.step = s.step;
      wrap.appendChild(row); wrap.appendChild(inp); controls.appendChild(wrap);
      els[s.key] = { inp, val };
      let timer = null;
      inp.addEventListener('input', () => {
        val.textContent = fmt(s, inp.value);
        clearTimeout(timer);
        timer = setTimeout(() => post({ [s.key]: parseFloat(inp.value) }), 120);
      });
    });
    function apply(state) {
      if (state._camera) document.getElementById('cam').textContent = state._camera;
      // Auto-range the crop sliders to the live frame size of this camera.
      if (state._frame_w && els['crop_x']) { els['crop_x'].inp.max = state._frame_w; els['crop_w'].inp.max = state._frame_w; }
      if (state._frame_h && els['crop_y']) { els['crop_y'].inp.max = state._frame_h; els['crop_h'].inp.max = state._frame_h; }
      SLIDERS.forEach(s => {
        if (state[s.key] === undefined) return;
        els[s.key].inp.value = state[s.key];
        els[s.key].val.textContent = fmt(s, state[s.key]);
      });
    }
    function post(body) {
      fetch('/tuning', { method: 'POST', headers: {'Content-Type':'application/json'},
                         body: JSON.stringify(body) })
        .then(r => r.ok ? r.json() : null).then(j => { if (j) apply(j); }).catch(() => {});
    }
    document.getElementById('reset').addEventListener('click', () => post(DEFAULTS));
    let curCam = null;
    fetch('/tuning.json').then(r => r.json()).then(state => { curCam = state._camera; apply(state); }).catch(() => {});
    // Reload sliders only when the selected camera changes (avoids overwriting
    // a slider mid-drag while still tracking external camera switches).
    setInterval(() => {
      fetch('/tuning.json').then(r => r.json()).then(state => {
        if (state._camera && state._camera !== curCam) { curCam = state._camera; apply(state); }
      }).catch(() => {});
    }, 1500);
    const imgEls = IMAGES.map(im => document.getElementById(im.id));
    const stamp = document.getElementById('stamp');
    if (imgEls[0]) imgEls[0].addEventListener('error', () => { stamp.textContent = 'waiting for detector...'; });
    setInterval(() => {
      const t = Date.now();
      IMAGES.forEach((im, i) => { imgEls[i].src = im.route + '?t=' + t; });
      stamp.textContent = new Date().toLocaleTimeString();
    }, 700);
  </script>
</body></html>
"""


def _render_tuning_page(title: str, lead: str, images: list[dict[str, str]], keys: list[str]) -> bytes:
  imgs_html = "".join(
    f'<div class="col"><div class="cap">{im["cap"]}</div>'
    f'<img id="{im["id"]}" src="{im["route"]}" alt="(waiting...)"></div>'
    for im in images
  )
  defaults = {k: TUNING_DEFAULTS[k] for k in keys}
  return (_PAGE_TEMPLATE
          .replace("__TITLE__", title)
          .replace("__LEAD__", lead)
          .replace("__IMAGES_HTML__", imgs_html)
          .replace("__SLIDERS__", _sliders_json(keys))
          .replace("__IMAGES__", json.dumps([{"id": im["id"], "route": im["route"]} for im in images]))
          .replace("__DEFAULTS__", json.dumps(defaults))
          ).encode()


TUNING_PAGE = _render_tuning_page(
  "Visual Vehicle Detector &mdash; Live Tuning",
  "Green boxes trip the right-lane flag; gray are ignored. Cyan = right ROI band, yellow line = min bottom-y. Drag to tune; saved to the device instantly.",
  [{"id": "proc", "route": "/processed.png", "cap": "model input (fed to YOLO)"},
   {"id": "crop", "route": "/detector_crop.png", "cap": "detector crop"}],
  TUNE_KEYS,
)

CROP_PAGE = _render_tuning_page(
  "Visual Vehicle Detector &mdash; Crop &amp; Rate",
  "Drag the crop box (yellow rectangle) over the area you want YOLO to see, and set the detector rate. Saved to the device instantly.",
  [{"id": "full", "route": "/full_frame.png", "cap": "full frame + crop box"},
   {"id": "crop", "route": "/detector_crop.png", "cap": "resulting crop"}],
  CROP_KEYS,
)


def visual_vehicle_tuning_server(port: int = TUNING_PORT) -> PreviewWebServer:
  return PreviewWebServer(
    port=port, page=TUNING_PAGE, routes=TUNING_ROUTES,
    json_routes={"/tuning.json": _tuning_json},
    post_handler=_tuning_post,
  )


def visual_vehicle_crop_server(port: int = CROP_PORT) -> PreviewWebServer:
  return PreviewWebServer(
    port=port, page=CROP_PAGE, routes=CROP_ROUTES,
    json_routes={"/tuning.json": _tuning_json},
    post_handler=_tuning_post,
  )


# ---------------- Dataset capture portal ----------------

CAPTURE_PAGE = b"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Visual Vehicle Detector Capture</title>
<style>
  body { background:#111; color:#ddd; font-family:sans-serif; margin:0; padding:16px; text-align:center; }
  h2 { margin:0 0 4px 0; }
  p.lead { color:#888; font-size:13px; margin:0 0 18px 0; }
  .stat { font-family:monospace; font-size:15px; margin:6px 0; }
  .stat b { color:#9bdca8; }
  .rec { display:inline-block; padding:6px 14px; border-radius:6px; font-family:monospace; margin:8px 0; }
  .rec.on { background:#3a1d1d; color:#ff8a8a; }
  .rec.off { background:#222; color:#888; }
  .ctl { margin:14px auto; max-width:520px; }
  input[type=range] { width:100%; height:34px; }
  .btns { margin-top:16px; display:flex; gap:12px; justify-content:center; flex-wrap:wrap; }
  a.btn, button { background:#243; color:#cfe; border:1px solid #365; border-radius:6px;
                  padding:12px 20px; font-size:15px; text-decoration:none; display:inline-block; }
  button.danger { background:#3a1d1d; color:#ffb3b3; border-color:#633; }
  #msg { margin-top:12px; color:#888; font-size:13px; font-family:monospace; min-height:18px; }
</style></head>
<body>
  <h2>Visual Vehicle Detector &mdash; Image Capture</h2>
  <p class="lead">Saves the selected camera's crop for training while this page's QR dialog is open and the car is onroad. No live preview.</p>
  <div class="rec off" id="rec">--</div>
  <div class="stat">camera: <b id="cam">--</b></div>
  <div class="stat">images: <b id="count">--</b> &nbsp; size: <b id="size">--</b></div>
  <div class="stat" id="cap" style="color:#caa;"></div>
  <div class="ctl">
    <div style="display:flex; justify-content:space-between; font-size:13px;"><span>Capture rate</span><span class="stat" style="margin:0;"><b id="hzval">--</b> Hz</span></div>
    <input type="range" id="hz" min="0.2" max="5" step="0.2">
  </div>
  <div class="btns">
    <a class="btn" id="dl" href="/download.zip">Download ZIP</a>
    <button class="danger" id="del">Delete all</button>
  </div>
  <div id="msg"></div>
  <script>
    const recEl = document.getElementById('rec');
    const fmtMB = b => (b / (1024*1024)).toFixed(1) + ' MB';
    function refresh() {
      fetch('/status.json?t=' + Date.now()).then(r => r.json()).then(s => {
        document.getElementById('cam').textContent = s.camera || '--';
        document.getElementById('count').textContent = s.count;
        document.getElementById('size').textContent = fmtMB(s.bytes);
        recEl.textContent = s.onroad ? 'RECORDING (onroad)' : 'PAUSED (offroad)';
        recEl.className = 'rec ' + (s.onroad ? 'on' : 'off');
        const capped = s.count >= s.max_files || s.bytes >= s.max_bytes;
        document.getElementById('cap').textContent = capped
          ? ('STORAGE CAP REACHED -- capture stopped (max ' + s.max_files + ' files / ' + fmtMB(s.max_bytes) + ')') : '';
        const hz = document.getElementById('hz');
        if (document.activeElement !== hz) { hz.value = s.hz; document.getElementById('hzval').textContent = (+s.hz).toFixed(1); }
      }).catch(() => {});
    }
    const hz = document.getElementById('hz');
    let t = null;
    hz.addEventListener('input', () => {
      document.getElementById('hzval').textContent = (+hz.value).toFixed(1);
      clearTimeout(t);
      t = setTimeout(() => fetch('/hz', {method:'POST', body: hz.value}).catch(()=>{}), 150);
    });
    document.getElementById('del').addEventListener('click', () => {
      if (!confirm('Delete ALL captured images?')) return;
      fetch('/delete', {method:'POST'}).then(r => r.json()).then(j => {
        document.getElementById('msg').textContent = 'Deleted ' + j.removed + ' images.';
        refresh();
      }).catch(() => {});
    });
    setInterval(refresh, 1500);
    refresh();
  </script>
</body></html>
"""


class CaptureWebServer:
  """Dataset capture portal. Mirrors ParamWebServer's interface (start/stop/url/
  token_saved) so NavParamQrDialog renders it. While running it holds the
  capture sentinel, so the detector records the selected camera's crop (onroad
  only). Streams all images as a single zip and supports delete-all."""

  def __init__(self, port: int = CAPTURE_PORT) -> None:
    self.port = port
    self.token_saved = threading.Event()  # never set; dialog stays on QR view
    self._server: ThreadingHTTPServer | None = None
    self._thread: threading.Thread | None = None

  def _status(self) -> bytes:
    try:
      from openpilot.common.params import Params
      onroad = not Params().get_bool("IsOffroad")
    except Exception:
      onroad = False
    stats = capture_stats()
    return json.dumps({
      "camera": active_camera(),
      "count": stats["count"],
      "bytes": stats["bytes"],
      "hz": capture_hz(),
      "onroad": onroad,
      "max_files": CAPTURE_MAX_FILES,
      "max_bytes": CAPTURE_MAX_BYTES,
    }).encode()

  def start(self) -> None:
    if self._server is not None:
      return
    capture_set_request(True, capture_hz())
    server = self

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
        route = self.path.split("?", 1)[0]
        if route in ("/", "/index.html"):
          self._send(CAPTURE_PAGE, "text/html; charset=utf-8")
          return
        if route == "/status.json":
          self._send(server._status(), "application/json")
          return
        if route == "/download.zip":
          self._stream_zip()
          return
        self.send_error(404)

      def _stream_zip(self):
        files = capture_files()
        # Stream the zip straight to the socket (stdlib zipfile handles the
        # non-seekable stream via data descriptors), so we never hold the whole
        # archive in RAM. No Content-Length -> close the connection at the end.
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Disposition", 'attachment; filename="vvd_captures.zip"')
        self.send_header("Connection", "close")
        self.end_headers()
        try:
          with zipfile.ZipFile(self.wfile, "w", zipfile.ZIP_STORED) as zf:
            for path in files:
              try:
                zf.write(path, arcname=os.path.basename(path))
              except OSError:
                continue
        except Exception:
          cloudlog.exception("nkaoud_vvd capture: zip stream failed")

      def do_POST(self):
        route = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else b""
        if route == "/delete":
          removed = capture_delete_all()
          self._send(json.dumps({"removed": removed}).encode(), "application/json")
          return
        if route == "/hz":
          try:
            capture_set_request(True, float(body.decode().strip()))
          except (ValueError, AttributeError):
            self.send_error(400, "bad hz")
            return
          self._send(server._status(), "application/json")
          return
        self.send_error(404)

    self._server = ThreadingHTTPServer(("0.0.0.0", self.port), _Handler)
    self._thread = threading.Thread(target=self._server.serve_forever,
                                    name="nkaoud_vvd_capture_ws", daemon=True)
    self._thread.start()
    cloudlog.info(f"nkaoud_vvd capture web: started on 0.0.0.0:{self.port}")

  def stop(self) -> None:
    if self._server is not None:
      try:
        self._server.shutdown()
        self._server.server_close()
      except OSError:
        pass
      self._server = None
      self._thread = None
      cloudlog.info("nkaoud_vvd capture web: stopped")
    capture_set_request(False)

  @property
  def url(self) -> str:
    return f"http://{get_local_ip()}:{self.port}/"


def visual_vehicle_capture_server(port: int = CAPTURE_PORT) -> CaptureWebServer:
  return CaptureWebServer(port=port)
