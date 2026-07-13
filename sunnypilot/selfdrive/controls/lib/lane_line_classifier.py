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
* Scan **perpendicular** to the projected line at each sample and take
  (peak - background) contrast. The perpendicular scan crosses the marking, so
  small lateral projection/calibration error still lands on the paint.
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
PERP_SCAN_PX = 7         # half-length of the perpendicular scan in image pixels
PERP_SCAN_STEPS = 15     # samples across the perpendicular scan
CENTER_SEARCH_PX = 4     # local recenter search around projected line (pixels)
CENTER_SEARCH_STEPS = 5  # odd count so the unshifted centre is always tested
CONTRAST_SMOOTH_WINDOW = 5
MAX_REPAIR_GAP_M = 1.0   # fill tiny missing gaps; well below a real dashed gap

# --- classification tunables (TUNE ON REAL DATA) -----------------------------
# Contrast = marking_peak - local_background, in 8-bit luminance counts.
MIN_CONTRAST = 18.0      # a sample counts as "marking present" above this
MIN_VALID_FRAC = 0.35    # need this fraction of in-frame samples to decide at all
SOLID_DUTY = 0.80        # duty (present fraction) at/above this with few gaps -> SOLID
BROKEN_DUTY_LO = 0.10    # broken lines sit roughly in this duty band
BROKEN_DUTY_HI = 0.75
MIN_PERIOD_M = 3.0       # plausible dash period range (line+gap), metres
MAX_PERIOD_M = 30.0
MIN_AUTOCORR = 0.30      # normalised autocorrelation peak to trust periodicity

# --- debounce (mirrors lane_position.py hysteresis) --------------------------
COUNTER_ON = 3
COUNTER_OFF = 1
COUNTER_MAX = 6

# Lane-line indexing in modelV2.laneLines (left -> right)
LEFT_EGO_LINE = 1
RIGHT_EGO_LINE = 2


@dataclass
class LaneLineClassifierConfig:
  """Live-tunable knobs. Defaults mirror the module constants; the daemon
  overrides these from params so they can be tuned from the menu."""
  sample_x_max: float = SAMPLE_X_MAX
  min_contrast: float = MIN_CONTRAST
  min_valid_frac: float = MIN_VALID_FRAC
  solid_duty: float = SOLID_DUTY
  broken_duty_lo: float = BROKEN_DUTY_LO
  broken_duty_hi: float = BROKEN_DUTY_HI
  min_period_m: float = MIN_PERIOD_M
  max_period_m: float = MAX_PERIOD_M
  min_autocorr: float = MIN_AUTOCORR
  center_search_px: float = CENTER_SEARCH_PX
  contrast_smooth_window: int = CONTRAST_SMOOTH_WINDOW
  max_repair_gap_m: float = MAX_REPAIR_GAP_M


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


def _perp_contrast(frame_y: np.ndarray, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Sample perpendicular scans across the projected line.

  Returns (contrast, valid) per along-line sample. ``contrast`` is
  peak_luminance - median_luminance across a short perpendicular scan; ``valid``
  marks samples whose scan stayed inside the frame.
  """
  h, w = frame_y.shape[:2]
  n = uv.shape[0]
  contrast = np.zeros(n, dtype=np.float32)
  valid = np.zeros(n, dtype=np.bool_)

  # image-space tangent from neighbouring projected points
  tang = np.gradient(uv, axis=0)
  norm = np.hypot(tang[:, 0], tang[:, 1])
  with np.errstate(divide='ignore', invalid='ignore'):
    tang_u = tang / norm[:, None]
  perp = np.stack([-tang_u[:, 1], tang_u[:, 0]], axis=1)   # rotate 90 deg

  offsets = np.linspace(-PERP_SCAN_PX, PERP_SCAN_PX, PERP_SCAN_STEPS)
  for i in range(n):
    if not np.all(np.isfinite(uv[i])) or not np.all(np.isfinite(perp[i])):
      continue
    sx = uv[i, 0] + perp[i, 0] * offsets
    sy = uv[i, 1] + perp[i, 1] * offsets
    ix = np.round(sx).astype(np.int64)
    iy = np.round(sy).astype(np.int64)
    inb = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    if inb.sum() < PERP_SCAN_STEPS // 2:
      continue
    vals = frame_y[iy[inb], ix[inb]].astype(np.float32)
    contrast[i] = float(vals.max() - np.median(vals))
    valid[i] = True
  return contrast, valid


def _perp_contrast_search(frame_y: np.ndarray, uv: np.ndarray,
                          center_search_px: float = CENTER_SEARCH_PX) -> tuple[np.ndarray, np.ndarray]:
  """Sample perpendicular scans with a small recenter search.

  The line projection from modelV2 can be a few pixels off because of calibration
  or model jitter. For each along-line sample, try a few nearby scan centres
  along the local normal and keep the strongest paint-like response.
  """
  h, w = frame_y.shape[:2]
  n = uv.shape[0]
  contrast = np.zeros(n, dtype=np.float32)
  valid = np.zeros(n, dtype=np.bool_)

  tang = np.gradient(uv, axis=0)
  norm = np.hypot(tang[:, 0], tang[:, 1])
  with np.errstate(divide='ignore', invalid='ignore'):
    tang_u = tang / norm[:, None]
  perp = np.stack([-tang_u[:, 1], tang_u[:, 0]], axis=1)

  offsets = np.linspace(-PERP_SCAN_PX, PERP_SCAN_PX, PERP_SCAN_STEPS)
  centre_offsets = np.linspace(-center_search_px, center_search_px, CENTER_SEARCH_STEPS, dtype=np.float32)
  for i in range(n):
    if not np.all(np.isfinite(uv[i])) or not np.all(np.isfinite(perp[i])):
      continue

    best_score = -np.inf
    best_valid = False
    for centre_shift in centre_offsets:
      centre = uv[i] + perp[i] * centre_shift
      sx = centre[0] + perp[i, 0] * offsets
      sy = centre[1] + perp[i, 1] * offsets
      ix = np.round(sx).astype(np.int64)
      iy = np.round(sy).astype(np.int64)
      inb = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
      if inb.sum() < PERP_SCAN_STEPS // 2:
        continue
      vals = frame_y[iy[inb], ix[inb]].astype(np.float32)
      score = float(np.percentile(vals, 90) - np.median(vals))
      if score > best_score:
        best_score = score
        best_valid = True

    if best_valid:
      contrast[i] = max(0.0, best_score)
      valid[i] = True
  return contrast, valid


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


def _autocorr_period(present: np.ndarray, min_period_m: float = MIN_PERIOD_M,
                     max_period_m: float = MAX_PERIOD_M) -> tuple[float, float]:
  """Estimate dash period (m) and normalised autocorrelation strength.

  ``present`` is the uniform-in-metres binary marking signal (step SAMPLE_STEP_M).
  Returns (period_m, strength). period_m == 0 when no clear period is found.
  """
  x = present.astype(np.float64)
  x = x - x.mean()
  if np.count_nonzero(x) == 0 or x.size < 8:
    return 0.0, 0.0
  ac = np.correlate(x, x, mode='full')[x.size - 1:]
  if ac[0] <= 0:
    return 0.0, 0.0
  ac = ac / ac[0]
  lo = max(1, int(round(min_period_m / SAMPLE_STEP_M)))
  hi = min(ac.size - 1, int(round(max_period_m / SAMPLE_STEP_M)))
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

  uv = _project(grid, transform)
  contrast, valid = _perp_contrast_search(np.asarray(frame_y), uv, cfg.center_search_px)
  contrast = _smooth_valid_signal(contrast, valid, cfg.contrast_smooth_window)

  n = contrast.size
  valid_frac = float(valid.mean()) if n else 0.0
  res = LaneLineResult(valid_frac=valid_frac, n_samples=n, contrast=contrast)
  if valid_frac < cfg.min_valid_frac:
    return res  # UNKNOWN

  present = valid & (contrast >= cfg.min_contrast)
  present = _repair_short_gaps(present, valid, cfg.max_repair_gap_m)
  res.present = present
  vpresent = present[valid]
  duty = float(vpresent.mean()) if vpresent.size else 0.0
  res.duty = duty

  period_m, ac_strength = _autocorr_period(present, cfg.min_period_m, cfg.max_period_m)

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

  # mid-band duty: dashed if there's a plausible, strong period
  if cfg.broken_duty_lo <= duty <= cfg.broken_duty_hi and ac_strength >= cfg.min_autocorr and period_m > 0:
    res.line_type = LaneLineType.BROKEN
    res.period_m = period_m
    res.confidence = float(np.clip(ac_strength, 0, 1))
    return res

  # duty says gappy but no clean period -> undecided, fail safe
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

  def update(self, frame_y: np.ndarray, modelV2, transform: np.ndarray,
             camera_offset: float = 0.0, cfg: LaneLineClassifierConfig | None = None) -> LaneChangeGate:
    lines = list(modelV2.laneLines)
    gate = LaneChangeGate()
    if len(lines) > LEFT_EGO_LINE:
      ll = lines[LEFT_EGO_LINE]
      gate.left = self._left_filter.update(classify_line(frame_y, ll.x, ll.y, ll.z, transform, camera_offset, cfg))
    if len(lines) > RIGHT_EGO_LINE:
      rl = lines[RIGHT_EGO_LINE]
      gate.right = self._right_filter.update(classify_line(frame_y, rl.x, rl.y, rl.z, transform, camera_offset, cfg))

    gate.left_crossable = self._left.update(gate.left.crossable)
    gate.right_crossable = self._right.update(gate.right.crossable)
    return gate
