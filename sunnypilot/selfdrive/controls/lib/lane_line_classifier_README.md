# Lane-line marking classifier (solid vs. broken)

Decides whether the lane line you would cross is **broken (crossable)** or
**solid (not crossable)** — the actual legal signal for a lane change, which
also covers the emergency/shoulder-lane case (shoulders are bordered by solid
lines).

No neural net: modelV2 already localises each lane line as a 3D polyline, so
this samples camera luminance along the projected line and classifies from the
marking's **duty cycle** and **periodicity**.

## How a sample is taken (why it's robust)

Both sampling axes are in **world metres**, never fixed pixels:

- *Along* the line: every 0.5 m of forward distance, 4–60 m ahead.
- *Across* the line: a ±0.45 m strip perpendicular to the line, projected into
  the image. On the real road camera (focal ~2650 px) a 13 cm marking is
  ~30 px wide at 10 m but only ~5 px at 60 m; a fixed-pixel scan is swallowed
  by the paint near-field and misses it far-field, which made results flip
  with centimetres of projection jitter. The world-space strip keeps the paint
  at a constant ~14% of the scan at every distance, so
  `P90(scan) - median(scan)` measures paint-vs-road contrast identically near
  and far — and the strip width doubles as tolerance for model/calibration
  error (up to ~0.3 m of lateral offset still lands on the paint).

A sample counts as "marking present" when contrast clears an absolute floor
(`MIN_CONTRAST`) **and** stands out from the scan's own road texture
(`MIN_SNR`, robust MAD estimate) — or, for dim-but-clean paint at night, a
lower floor at twice the SNR. This keeps one set of thresholds workable across
day/night exposure.

## How the presence signal is classified

1. duty ≥ `SOLID_DUTY` → **SOLID**
2. high duty with one/two dominant runs and no dash-shaped gaps → **SOLID**
   (continuity bias: fragmented paint, dirt, short occlusions)
3. mid duty + strong autocorrelation peak at a plausible dash period → **BROKEN**
4. mid duty + several dash-length paint runs separated by gap-length absences
   → **BROKEN** (run-length fallback for dashes too irregular for autocorr)
5. anything else → **UNKNOWN** (fails safe to not-crossable)

## Files
- `lane_line_classifier.py` — pure-numpy core.
  - `classify_line(frame_y, x, y, z, transform, camera_offset)` → `LaneLineResult`
    (`line_type`, `confidence`, `duty`, `period_m`, `crossable`, raw signal).
  - `LaneLineClassifier.update(frame_y, modelV2, transform, camera_offset)` →
    `LaneChangeGate` with debounced `left_crossable` / `right_crossable`
    (classifies `laneLines[1]` and `laneLines[2]`; skips lines the model
    itself barely sees, `laneLineProbs < 0.3`).
- `tests/test_lane_line_classifier.py` — synthetic tests at the real tici
  camera geometry (focal 2648 px) with realistic 13 cm paint — the geometry
  that broke the old fixed-pixel scan (no camera or route needed).
- `lane_line_classifier_replay.py` — run it on a real route.

## The daemon pairs frames by frameId

`lane_line_classifierd` keeps a small cache of recent camera Y-planes and
classifies each `modelV2` against the exact frame it was computed from
(`modelV2.frameId`), instead of pairing newest-frame with newest-model. A few
frames of mismatch is metres of forward travel — on curves that alone pulled
the far-field scan off the paint.

## Test now (no hardware)
```bash
python -m pytest sunnypilot/selfdrive/controls/lib/tests/test_lane_line_classifier.py -v
```

## Test on a real route
```bash
python -m sunnypilot.selfdrive.controls.lib.lane_line_classifier_replay \
    "<dongleId>|<route>" --stride 5 --limit 200
```
Prints per-frame `L[...] cross=0/1  R[...] cross=0/1`. Drive a stretch with a
known solid line on one side and a dashed line on the other and confirm the
labels match.

## The one thing to validate first
Everything rests on the projection landing the sampling strip on the paint —
the scan tolerates ~0.3 m of error, not more. If labels look random, the
transform/camera-offset is off, not the classifier — check `build_transform`
(in the replay harness) matches your camera and that `--camera-offset` matches
your `CameraOffset` param.

## Tunables (in `lane_line_classifier.py`)
Menu-exposed (via params): `MIN_CONTRAST`, `SOLID_DUTY`, `MIN_PERIOD_M`,
`MAX_PERIOD_M`, `MIN_AUTOCORR`, `LaneLineSampleMaxM`.
Config-only: `MIN_SNR`, `SCAN_HALF_M`, duty bands, continuity-bias and
run-length-fallback shape limits. `UNKNOWN` always fails safe to
not-crossable. Note that raising `MIN_AUTOCORR` no longer forces irregular
dashes to `UNKNOWN` — the run-length fallback can still call them broken (at
capped confidence ≤ 0.65).

## Wiring into lane-change gating (next step, not done here)
`lane_position.py` already blocks edge lanes with geometric votes. Add
`left_crossable` / `right_crossable` as a precondition on the requested side in
the lane-change decision (`lane_position.py` / `desire_helper.py`). Keep
`UNKNOWN → block` so degraded perception preserves today's behaviour.
