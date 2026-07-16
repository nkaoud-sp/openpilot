"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Lane-line marking classifier: solid vs. broken (dashed).

Motivation
----------
Whether a lane change is *legal* is governed by the marking you would cross:
a broken (dashed) line is crossable, a solid line is not. This subsumes the
"emergency/shoulder lane" case (shoulders are bordered by a solid line) plus
every other no-cross boundary (solid centre lines, gore areas, HOV dividers).

The driving model (modelV2) already localises each lane line as a 3D polyline
(``laneLines[i].x/.y/.z``). That turns a hard vision problem ("where is the
marking?") into an easy 1D signal problem ("given I know where the line is, is
it continuous or does it have periodic gaps?"). So this classifier does NOT use
a neural net: it samples image luminance along the projected line and decides
solid vs. broken from the duty cycle and periodicity of the marking signal.

Key ideas
---------
* Sample uniformly in **real-world metres** along the line (not image pixels),
  so the dash period is perspective-invariant and shows up as a clean
  autocorrelation peak regardless of distance.
* Scan laterally across the line **in real-world metres** too. On the real
  road camera (tici fcam, ~2650 px focal length) a 13 cm marking is ~30 px wide
  at 10 m and ~5 px at 60 m, so any fixed-pixel scan is either swallowed by the
  paint near-field or misses it far-field. A constant-width world-space scan
  keeps the paint at a constant *fraction* of the scan at every distance:
  high percentile - median then measures paint-vs-road contrast everywhere,
  and the scan width doubles as tolerance for projection/calibration error.
* Marking presence is decided by contrast **and** signal-to-noise against the
  local road texture (robust MAD estimate), so thresholds hold up across
  day/night exposure and shadowed or washed-out frames.
* Everything below the classification thresholds or with too little contrast is
  reported as ``UNKNOWN`` so the caller can fail safe (treat as not-crossable).

The core functions are pure (numpy only, no cereal/openpilot deps) so they can
be unit-tested with synthetic frames. ``LaneLineClassifier`` wraps them with a
per-side debounce for use against a live ``modelV2``.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class LaneLineType(IntEnum):
  UNKNOWN = 0   # not enough signal to decide -> caller should fail safe
  BROKEN = 1    # dashed -> crossable
  SOLID = 2     # continuous -> not crossable
  DOUBLE = 3    # two parallel solids -> not crossable (best-effort)


CROSSABLE_TYPES = (LaneLineType.BROKEN,)


# --- sampling geometry (metres) ---------------------------------------------
SAMPLE_X_MIN = 4.0       # start of the trusted window ahead (m)
SAMPLE_X_MAX = 60.0      # end of the trusted window ahead (m); far field too few px
SAMPLE_STEP_M = 0.5      # uniform along-line spacing (m); << typical dash period
SCAN_HALF_M = 0.45       # lateral scan half-width in world metres; also the
                         # tolerance budget for projection/model/calib error
SCAN_STEPS = 31          # samples across the lateral scan (~3 cm world spacing)
MIN_INFRAME_FRAC = 0.7   # scan points that must land in-frame for a valid sample
MAX_REPAIR_GAP_M = 1.0   # fill tiny missing gaps; well below a real dashed gap

# --- classification tunables (TUNE ON REAL DATA) -----------------------------
# Contrast = selected bright-evidence(scan) - median(scan), in 8-bit luminance
# counts. P90 is the default; the live selector also supports P95, top-3 mean,
# and max. A sample counts as "marking present" when contrast clears BOTH an
# absolute floor and an SNR gate vs. the scan's own road texture (see
# classify_line), or - for dim/washed-out paint - a lower floor at twice the SNR.
MIN_CONTRAST = 18.0      # absolute contrast floor (8-bit counts)
MIN_SNR = 3.0            # contrast / robust-noise gate
MIN_VALID_FRAC = 0.35    # need this fraction of in-frame samples to decide at all
SOLID_DUTY = 0.80        # duty (present fraction) at/above this with few gaps -> SOLID
BROKEN_DUTY_LO = 0.10    # broken lines sit roughly in this duty band
BROKEN_DUTY_HI = 0.75
MIN_PERIOD_M = 3.0       # plausible dash period range (line+gap), metres
MAX_PERIOD_M = 30.0
MIN_AUTOCORR = 0.30      # normalised autocorrelation peak to trust periodicity
SOLID_CONTIG_FRAC = 0.30  # longest continuous present-run as a fraction of valid samples
SOLID_MAX_GAP_FRAC = 0.18 # largest absent run allowed for continuity-biased SOLID
SOLID_TOP2_FRAC = 0.50    # combined coverage of the two largest present-runs
                          # (even a 9m/3m long-dash line only scores ~0.33 here,
                          # so this cleanly separates fragmented-solid from broken)
# run-length fallback for dashes whose spacing is too irregular for autocorr
DASH_RUN_MIN_M = 0.5      # plausible painted dash length range (m)
DASH_RUN_MAX_M = 12.0
DASH_GAP_MIN_M = 1.5      # plausible gap length range (m); > repaired occlusions
DASH_GAP_MAX_M = 25.0
CONTRAST_SMOOTH_WINDOW = 3

# Lateral scan evidence methods. Keep these integer values stable because the
# Lane Visualizer tweaks menu stores the selected method in Params.
CONTRAST_METHOD_P90 = 0       # robust baseline: P90 - median
CONTRAST_METHOD_P95 = 1       # thinner/sparser markings: P95 - median
CONTRAST_METHOD_TOP3 = 2      # mean of three brightest samples - median
CONTRAST_METHOD_MAX = 3       # most sensitive, but most noise-prone
CONTRAST_METHOD_COUNT = 4

# --- debounce (mirrors lane_position.py hysteresis) --------------------------
COUNTER_ON = 3
COUNTER_OFF = 1
COUNTER_MAX = 6

# Lane-line indexing in modelV2.laneLines (left -> right)
LEFT_EGO_LINE = 1
RIGHT_EGO_LINE = 2
MIN_LINE_PROB = 0.3      # below this modelV2 doesn't really see a line


@dataclass
class LaneLineClassifierConfig:
  """Live-tunable knobs. Defaults mirror the module constants; the daemon
  overrides these from params so they can be tuned from the menu."""
  sample_x_max: float = SAMPLE_X_MAX
  min_contrast: float = MIN_CONTRAST
  min_snr: float = MIN_SNR
  min_valid_frac: float = MIN_VALID_FRAC
  solid_duty: float = SOLID_DUTY
  broken_duty_lo: float = BROKEN_DUTY_LO
  broken_duty_hi: float = BROKEN_DUTY_HI
  min_period_m: float = MIN_PERIOD_M
  max_period_m: float = MAX_PERIOD_M
  min_autocorr: float = MIN_AUTOCORR
  solid_contig_frac: float = SOLID_CONTIG_FRAC
  solid_max_gap_frac: float = SOLID_MAX_GAP_FRAC
  solid_top2_frac: float = SOLID_TOP2_FRAC
  scan_half_m: float = SCAN_HALF_M
  contrast_smooth_window: int = CONTRAST_SMOOTH_WINDOW
  max_repair_gap_m: float = MAX_REPAIR_GAP_M
  contrast_method: int = CONTRAST_METHOD_P90


DEFAULT_CONFIG = LaneLineClassifierConfig()


@dataclass
class LaneLineResult:
  line_type: LaneLineType = LaneLineType.UNKNOWN
  confidence: float = 0.0       # 0..1
  duty: float = 0.0             # fraction of samples with marking present
  period_m: float = 0.0         # estimated dash period (0 if none/solid)
  valid_frac: float = 0.0       # fraction of samples that projected in-frame
  n_samples: int = 0
  # raw signal kept for offline plotting / debugging
  contrast: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
  present: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.bool_))

  @property
  def crossable(self) -> bool:
    return self.line_type in CROSSABLE_TYPES


def _project(points_dev: np.ndarray, transform: np.ndarray) -> np.ndarray:
  """Project Nx3 device/calib-frame points to Nx2 image pixels (NaN if behind).

  ``transform`` is the 3x3 device->image matrix, i.e. the UI's ``calib_transform``
  ( intrinsics @ view_from_calib ) WITHOUT the display ``video_transform``.
  """
  proj = transform @ points_dev.T            # 3xN
  z = proj[2]
  with np.errstate(divide='ignore', invalid='ignore'):
    px = proj[0] / z
    py = proj[1] / z
  out = np.stack([px, py], axis=1)
  out[z <= 1e-6] = np.nan                    # points at/behind the camera
  return out


def _resample_line_uniform_x(line_x, line_y, line_z, camera_offset: float, sample_x_max: float = SAMPLE_X_MAX):
  """Interpolate the 3D line onto a uniform grid of forward distance x (m)."""
  x = np.asarray(line_x, dtype=np.float64)
  y = np.asarray(line_y, dtype=np.float64) + camera_offset
  z = np.asarray(line_z, dtype=np.float64)
  if x.size < 2:
    return None
  x_hi = min(sample_x_max, float(x[-1]))
  if x_hi <= SAMPLE_X_MIN + SAMPLE_STEP_M:
    return None
  xs = np.arange(SAMPLE_X_MIN, x_hi, SAMPLE_STEP_M, dtype=np.float64)
  ys = np.interp(xs, x, y)
  zs = np.interp(xs, x, z)
  return np.stack([xs, ys, zs], axis=1)      # Nx3


def _ground_perp(grid: np.ndarray) -> np.ndarray:
  """World-space unit normal of the line in the ground plane, per grid point."""
  tang = np.gradient(grid[:, :2], axis=0)
  tang /= np.maximum(np.hypot(tang[:, 0], tang[:, 1]), 1e-9)[:, None]
  return np.stack([-tang[:, 1], tang[:, 0]], axis=1)


def scan_geometry_uv(line_x, line_y, line_z, transform: np.ndarray, camera_offset: float = 0.0,
                     cfg: LaneLineClassifierConfig | None = None):
  """Project the scan geometry into image pixels for overlays/debug snapshots.

  Returns (centre_uv, [left_rail_uv, right_rail_uv]) - each an Nx2 array with
  NaN where a point falls behind the camera - or None if the line is too
  short to classify. centre_uv rows correspond 1:1 with the classifier's
  along-line samples (and thus with ``LaneLineResult.present``).
  """
  if cfg is None:
    cfg = DEFAULT_CONFIG
  grid = _resample_line_uniform_x(line_x, line_y, line_z, camera_offset, cfg.sample_x_max)
  if grid is None:
    return None
  perp = _ground_perp(grid)
  centre = _project(grid, transform)
  rails = []
  for side in (-1.0, 1.0):
    pts = grid.copy()
    pts[:, :2] += perp * (side * cfg.scan_half_m)
    rails.append(_project(pts, transform))
  return centre, rails


def _scan_contrast(frame_y: np.ndarray, grid: np.ndarray, transform: np.ndarray,
                   scan_half_m: float = SCAN_HALF_M,
                   contrast_method: int = CONTRAST_METHOD_P90) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Scan laterally across the line in world metres at each along-line sample.

  At each grid point a constant-world-width strip perpendicular to the line is
  projected into the image and sampled. Because the strip is fixed in metres,
  the marking occupies the same fraction of the scan at every distance, so
  The selected lateral evidence method compares the bright marking response
  against the scan median. P90 is the default; alternatives are exposed by
  the Lane Visualizer tweaks menu for on-road comparison.

  Returns (contrast, noise, valid) per along-line sample; ``noise`` is a robust
  (MAD-based) estimate of the road texture inside the scan.
  """
  h, w = frame_y.shape[:2]
  n = grid.shape[0]
  contrast = np.zeros(n, dtype=np.float32)
  noise = np.full(n, np.inf, dtype=np.float32)
  valid = np.zeros(n, dtype=np.bool_)
  if n < 2:
    return contrast, noise, valid

  perp = _ground_perp(grid)

  offsets = np.linspace(-scan_half_m, scan_half_m, SCAN_STEPS)
  pts = np.empty((n, SCAN_STEPS, 3), dtype=np.float64)
  pts[..., 0] = grid[:, None, 0] + perp[:, None, 0] * offsets
  pts[..., 1] = grid[:, None, 1] + perp[:, None, 1] * offsets
  pts[..., 2] = grid[:, None, 2]

  uv = _project(pts.reshape(-1, 3), transform).reshape(n, SCAN_STEPS, 2)
  ix = np.round(uv[..., 0])
  iy = np.round(uv[..., 1])
  inb = np.isfinite(ix) & np.isfinite(iy) & (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
  valid = inb.mean(axis=1) >= MIN_INFRAME_FRAC
  if not valid.any():
    return contrast, noise, valid

  vals = np.full((n, SCAN_STEPS), np.nan, dtype=np.float32)
  vals[inb] = frame_y[iy[inb].astype(np.int64), ix[inb].astype(np.int64)]
  v = vals[valid]
  med = np.nanmedian(v, axis=1)
  method = int(np.clip(contrast_method, 0, CONTRAST_METHOD_COUNT - 1))
  if method == CONTRAST_METHOD_P95:
    bright = np.nanpercentile(v, 95, axis=1)
  elif method == CONTRAST_METHOD_TOP3:
    # Three samples are about 10% of the 31-point scan. This keeps a compact
    # bright peak while averaging away one-pixel road-texture spikes.
    k = min(3, v.shape[1])
    sortable = np.where(np.isfinite(v), v, -np.inf)
    bright = np.mean(np.sort(sortable, axis=1)[:, -k:], axis=1)
  elif method == CONTRAST_METHOD_MAX:
    bright = np.nanmax(v, axis=1)
  else:
    bright = np.nanpercentile(v, 90, axis=1)
  mad = np.nanmedian(np.abs(v - med[:, None]), axis=1)
  contrast[valid] = np.maximum(bright - med, 0.0).astype(np.float32)
  noise[valid] = (1.4826 * mad + 0.5).astype(np.float32)   # +0.5: quantisation floor
  return contrast, noise, valid


def _smooth_valid_signal(values: np.ndarray, valid: np.ndarray, window: int) -> np.ndarray:
  """Blur the along-line signal without letting invalid samples drag it down."""
  window = max(1, int(window))
  if window <= 1 or values.size == 0:
    return values.astype(np.float32, copy=True)
  kernel = np.ones(window, dtype=np.float32)
  weight = np.convolve(valid.astype(np.float32), kernel, mode='same')
  numer = np.convolve(np.where(valid, values, 0.0).astype(np.float32), kernel, mode='same')
  out = np.zeros_like(values, dtype=np.float32)
  np.divide(numer, np.maximum(weight, 1e-6), out=out, where=weight > 0)
  return out


def _despeckle(present: np.ndarray, valid: np.ndarray) -> np.ndarray:
  """Drop isolated single-sample 'present' spikes (0.5 m << any real dash)."""
  out = present.astype(np.bool_, copy=True)
  n = out.size
  for i in range(n):
    if not out[i] or not valid[i]:
      continue
    left_off = i == 0 or not out[i - 1]
    right_off = i == n - 1 or not out[i + 1]
    if left_off and right_off:
      out[i] = False
  return out


def _repair_short_gaps(present: np.ndarray, valid: np.ndarray, max_gap_m: float) -> np.ndarray:
  """Fill tiny absent runs inside an otherwise-present marking signal.

  This helps solid lines survive brief dropouts from shadows or cracks while
  leaving true dashed gaps untouched because the repair budget is much shorter
  than a normal dash cycle.
  """
  out = present.astype(np.bool_, copy=True)
  if out.size == 0 or max_gap_m <= 0:
    return out

  max_gap = max(0, int(round(max_gap_m / SAMPLE_STEP_M)))
  if max_gap <= 0:
    return out

  i = 0
  n = out.size
  while i < n:
    if out[i] or not valid[i]:
      i += 1
      continue
    j = i
    while j < n and valid[j] and not out[j]:
      j += 1
    gap_len = j - i
    left_on = i > 0 and out[i - 1]
    right_on = j < n and out[j]
    if left_on and right_on and gap_len <= max_gap:
      out[i:j] = True
    i = j + 1 if j == i else j
  return out


def _run_lengths(mask: np.ndarray) -> list[int]:
  runs: list[int] = []
  run = 0
  for value in mask:
    if value:
      run += 1
    elif run:
      runs.append(run)
      run = 0
  if run:
    runs.append(run)
  return runs


def _autocorr_period(present: np.ndarray, valid: np.ndarray,
                     min_period_m: float = MIN_PERIOD_M,
                     max_period_m: float = MAX_PERIOD_M) -> tuple[float, float]:
  """Estimate dash period (m) and normalised autocorrelation strength.

  ``present`` is the uniform-in-metres binary marking signal (step SAMPLE_STEP_M).
  Invalid samples are filled with the valid mean so out-of-frame stretches do
  not masquerade as gaps; the estimate is overlap-corrected (unbiased) so long
  dash periods are not penalised, and lags are capped at half the window so at
  least two full periods support any peak. Returns (period_m, strength);
  period_m == 0 when no clear period is found.
  """
  x = present.astype(np.float64)
  if valid is not None and valid.any():
    x = np.where(valid, x, x[valid].mean())
  x = x - x.mean()
  if not np.any(x) or x.size < 8:
    return 0.0, 0.0
  n = x.size
  ac = np.correlate(x, x, mode='full')[n - 1:]
  if ac[0] <= 0:
    return 0.0, 0.0
  overlap = np.arange(n, 0, -1, dtype=np.float64)
  ac = (ac / ac[0]) * (n / overlap)          # unbiased normalisation
  lo = max(1, int(round(min_period_m / SAMPLE_STEP_M)))
  hi = min(n // 2, int(round(max_period_m / SAMPLE_STEP_M)))
  if hi <= lo:
    return 0.0, 0.0
  band = ac[lo:hi + 1]
  k = int(np.argmax(band)) + lo
  return k * SAMPLE_STEP_M, float(ac[k])


def classify_line(frame_y: np.ndarray, line_x, line_y, line_z,
                  transform: np.ndarray, camera_offset: float = 0.0,
                  cfg: LaneLineClassifierConfig | None = None) -> LaneLineResult:
  """Classify a single lane line as SOLID / BROKEN / UNKNOWN.

  frame_y     : HxW uint8/float luminance (Y) channel of the road camera.
  line_x/y/z  : the modelV2 lane-line polyline (device/calib frame, metres).
  transform   : 3x3 device->image matrix (UI ``calib_transform``).
  camera_offset: lateral offset added to y, matching model_renderer.
  cfg         : live-tunable thresholds; defaults to DEFAULT_CONFIG.
  """
  if cfg is None:
    cfg = DEFAULT_CONFIG
  grid = _resample_line_uniform_x(line_x, line_y, line_z, camera_offset, cfg.sample_x_max)
  if grid is None:
    return LaneLineResult()

  contrast, noise, valid = _scan_contrast(np.asarray(frame_y), grid, transform,
                                          cfg.scan_half_m, cfg.contrast_method)

  n = contrast.size
  valid_frac = float(valid.mean()) if n else 0.0
  # presence is decided on the raw per-sample signal: smoothing would dilate
  # dash edges (~0.5 m per side) and push long-dash patterns over the solid
  # duty threshold. The smoothed copy is only for plotting/debug output.
  res = LaneLineResult(valid_frac=valid_frac, n_samples=n,
                       contrast=_smooth_valid_signal(contrast, valid, cfg.contrast_smooth_window))
  if valid_frac < cfg.min_valid_frac:
    return res  # UNKNOWN

  # marking present when contrast clears the absolute floor AND stands out from
  # the local road texture; dim-but-clean paint (night, washout) passes on a
  # lower floor at double the SNR.
  snr = contrast / np.maximum(noise, 1e-6)
  strong = (contrast >= cfg.min_contrast) & (snr >= cfg.min_snr)
  weak = (contrast >= max(8.0, 0.4 * cfg.min_contrast)) & (snr >= 2.0 * cfg.min_snr)
  present = valid & (strong | weak)
  present = _despeckle(present, valid)
  present = _repair_short_gaps(present, valid, cfg.max_repair_gap_m)
  res.present = present
  vpresent = present[valid]
  duty = float(vpresent.mean()) if vpresent.size else 0.0
  res.duty = duty
  valid_n = int(vpresent.size)

  period_m, ac_strength = _autocorr_period(present, valid, cfg.min_period_m, cfg.max_period_m)
  present_runs = _run_lengths(vpresent)
  absent_runs = _run_lengths(~vpresent) if valid_n else []
  longest_present_frac = (max(present_runs) / valid_n) if present_runs and valid_n else 0.0
  longest_absent_frac = (max(absent_runs) / valid_n) if absent_runs and valid_n else 0.0
  top2_present_frac = (sum(sorted(present_runs, reverse=True)[:2]) / valid_n) if present_runs and valid_n else 0.0

  if duty < cfg.broken_duty_lo:
    # almost nothing there: worn line, wrong projection, or no marking
    res.line_type = LaneLineType.UNKNOWN
    res.confidence = 0.0
    return res

  if duty >= cfg.solid_duty:
    res.line_type = LaneLineType.SOLID
    res.confidence = float(np.clip((duty - cfg.solid_duty) / (1 - cfg.solid_duty) * 0.5 + 0.5, 0, 1))
    res.period_m = 0.0
    return res

  # a solid line with dirt/shadow dropouts can look quasi-periodic, so don't
  # veto continuity on autocorrelation alone; only genuinely dash-shaped gaps
  # (long enough that gap-repair can't have healed them, at broken-band duty)
  # disqualify a continuity-based SOLID call.
  median_gap_m = (float(np.median(absent_runs)) * SAMPLE_STEP_M) if absent_runs else 0.0
  gaps_dashlike = median_gap_m >= DASH_GAP_MIN_M and duty <= cfg.broken_duty_hi
  continuity_bias_solid = (
    valid_n >= 8 and
    duty >= max(cfg.broken_duty_hi - 0.10, 0.55) and
    longest_present_frac >= cfg.solid_contig_frac and
    top2_present_frac >= cfg.solid_top2_frac and
    longest_absent_frac <= cfg.solid_max_gap_frac and
    not gaps_dashlike
  )
  if continuity_bias_solid:
    res.line_type = LaneLineType.SOLID
    contig_score = 0.55 + 0.35 * longest_present_frac + 0.20 * duty - 0.15 * longest_absent_frac
    res.confidence = float(np.clip(contig_score, 0.0, 0.95))
    res.period_m = 0.0
    return res

  # mid-band duty: dashed if there's a plausible, strong period
  if cfg.broken_duty_lo <= duty <= cfg.broken_duty_hi and ac_strength >= cfg.min_autocorr and period_m > 0:
    res.line_type = LaneLineType.BROKEN
    res.period_m = period_m
    res.confidence = float(np.clip(ac_strength, 0, 1))
    return res

  # autocorr needs ~3 regular cycles in the window; real dashes are often too
  # irregular (worn paint, merges, variable gaps). Fall back to run-length
  # shape: several paint runs of dash-like length separated by gaps that are
  # clearly longer than anything gap-repair would have healed on a solid line.
  # cap at 0.65 duty: above that a fragmented solid is too easily mistaken
  # for dashes, and the continuity path above owns that band
  if cfg.broken_duty_lo <= duty <= min(0.65, cfg.broken_duty_hi) and len(present_runs) >= 2 and len(absent_runs) >= 2:
    dash_m = float(np.median(present_runs)) * SAMPLE_STEP_M
    gap_m = float(np.median(absent_runs)) * SAMPLE_STEP_M
    if DASH_RUN_MIN_M <= dash_m <= DASH_RUN_MAX_M and DASH_GAP_MIN_M <= gap_m <= DASH_GAP_MAX_M:
      res.line_type = LaneLineType.BROKEN
      res.period_m = dash_m + gap_m
      res.confidence = float(np.clip(0.35 + 0.05 * min(len(present_runs), 6), 0, 0.65))
      return res

  # gappy but neither periodic nor dash-shaped -> undecided, fail safe
  res.line_type = LaneLineType.UNKNOWN
  res.confidence = 0.0
  res.period_m = period_m
  return res


class _Debounce:
  """Asymmetric-threshold latch, same shape as lane_position.py."""
  def __init__(self):
    self.counter = 0
    self.state = False

  def update(self, vote: bool) -> bool:
    self.counter = min(COUNTER_MAX, self.counter + 1) if vote else max(0, self.counter - 1)
    if self.state:
      self.state = self.counter > COUNTER_OFF
    else:
      self.state = self.counter >= COUNTER_ON
    return self.state


class _TemporalLineFilter:
  """Short history stabilizer for the displayed per-line classification.

  It never turns UNKNOWN into crossable by itself; it only helps a confident
  known class survive a single noisy frame so the UI/readout does not flicker.
  """
  def __init__(self, hold_frames: int = 1):
    self._history: deque[LaneLineResult] = deque(maxlen=4)
    self._hold_frames = max(0, hold_frames)
    self._unknown_run = 0

  def update(self, result: LaneLineResult) -> LaneLineResult:
    last_known = next((r for r in reversed(self._history) if r.line_type != LaneLineType.UNKNOWN), None)
    out = result

    if result.line_type == LaneLineType.UNKNOWN:
      self._unknown_run += 1
      if (self._unknown_run <= self._hold_frames and last_known is not None and
          last_known.confidence >= 0.8 and result.valid_frac >= 0.5 * max(last_known.valid_frac, 1e-6)):
        out = LaneLineResult(
          line_type=last_known.line_type,
          confidence=last_known.confidence * 0.85,
          duty=last_known.duty,
          period_m=last_known.period_m,
          valid_frac=result.valid_frac,
          n_samples=result.n_samples,
          contrast=result.contrast,
          present=result.present,
        )
    else:
      self._unknown_run = 0
      if last_known is not None and last_known.line_type == result.line_type:
        out = LaneLineResult(
          line_type=result.line_type,
          confidence=float(np.clip(0.4 * last_known.confidence + 0.6 * result.confidence, 0, 1)),
          duty=float(0.4 * last_known.duty + 0.6 * result.duty),
          period_m=float(0.4 * last_known.period_m + 0.6 * result.period_m),
          valid_frac=result.valid_frac,
          n_samples=result.n_samples,
          contrast=result.contrast,
          present=result.present,
        )

    self._history.append(out)
    return out


@dataclass
class LaneChangeGate:
  left_crossable: bool = False
  right_crossable: bool = False
  left: LaneLineResult = field(default_factory=LaneLineResult)
  right: LaneLineResult = field(default_factory=LaneLineResult)


class LaneLineClassifier:
  """Stateful wrapper: classify the two ego lane lines and debounce crossability.

  ``crossable`` latches True only after COUNTER_ON consecutive BROKEN votes, and
  releases quickly, so a momentarily-dashed-looking solid line (occlusion, worn
  paint) will not open a lane change. UNKNOWN counts as a not-crossable vote.
  """
  def __init__(self):
    self._left = _Debounce()
    self._right = _Debounce()
    self._left_filter = _TemporalLineFilter()
    self._right_filter = _TemporalLineFilter()

  @staticmethod
  def _line_prob(modelV2, idx: int) -> float:
    probs = getattr(modelV2, 'laneLineProbs', None)
    if probs is None or len(probs) <= idx:
      return 1.0
    return float(probs[idx])

  def update(self, frame_y: np.ndarray, modelV2, transform: np.ndarray,
             camera_offset: float = 0.0, cfg: LaneLineClassifierConfig | None = None) -> LaneChangeGate:
    lines = list(modelV2.laneLines)
    gate = LaneChangeGate()
    if len(lines) > LEFT_EGO_LINE:
      ll = lines[LEFT_EGO_LINE]
      # when the model itself doesn't see a line, don't classify road texture
      res = (classify_line(frame_y, ll.x, ll.y, ll.z, transform, camera_offset, cfg)
             if self._line_prob(modelV2, LEFT_EGO_LINE) >= MIN_LINE_PROB else LaneLineResult())
      gate.left = self._left_filter.update(res)
    if len(lines) > RIGHT_EGO_LINE:
      rl = lines[RIGHT_EGO_LINE]
      res = (classify_line(frame_y, rl.x, rl.y, rl.z, transform, camera_offset, cfg)
             if self._line_prob(modelV2, RIGHT_EGO_LINE) >= MIN_LINE_PROB else LaneLineResult())
      gate.right = self._right_filter.update(res)

    gate.left_crossable = self._left.update(gate.left.crossable)
    gate.right_crossable = self._right.update(gate.right.crossable)
    return gate
