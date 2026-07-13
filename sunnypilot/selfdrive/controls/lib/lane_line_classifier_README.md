# Lane-line marking classifier (solid vs. broken)

Decides whether the lane line you would cross is **broken (crossable)** or
**solid (not crossable)** — the actual legal signal for a lane change, which
also covers the emergency/shoulder-lane case (shoulders are bordered by solid
lines).

No neural net: modelV2 already localises each lane line as a 3D polyline, so
this samples camera luminance along the projected line and classifies from the
marking's **duty cycle** and **periodicity**.

## Files
- `lane_line_classifier.py` — pure-numpy core.
  - `classify_line(frame_y, x, y, z, transform, camera_offset)` → `LaneLineResult`
    (`line_type`, `confidence`, `duty`, `period_m`, `crossable`, raw signal).
  - `LaneLineClassifier.update(frame_y, modelV2, transform, camera_offset)` →
    `LaneChangeGate` with debounced `left_crossable` / `right_crossable`
    (classifies `laneLines[1]` and `laneLines[2]`).
- `tests/test_lane_line_classifier.py` — synthetic solid/broken/blank tests
  (no camera or route needed).
- `lane_line_classifier_replay.py` — run it on a real route.

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
Everything rests on the projection landing the sampling strip on the paint. If
labels look random, the transform/camera-offset is off, not the classifier —
check `build_transform` (in the replay harness) matches your camera and that
`--camera-offset` matches your `CameraOffset` param.

## Tunables (in `lane_line_classifier.py`)
`MIN_CONTRAST`, `SOLID_DUTY`, `BROKEN_DUTY_LO/HI`, `MIN_PERIOD_M`,
`MAX_PERIOD_M`, `MIN_AUTOCORR`. Defaults are reasonable starting points; tune on
your footage. `UNKNOWN` always fails safe to not-crossable.

## Wiring into lane-change gating (next step, not done here)
`lane_position.py` already blocks edge lanes with geometric votes. Add
`left_crossable` / `right_crossable` as a precondition on the requested side in
the lane-change decision (`lane_position.py` / `desire_helper.py`). Keep
`UNKNOWN → block` so degraded perception preserves today's behaviour.
```
