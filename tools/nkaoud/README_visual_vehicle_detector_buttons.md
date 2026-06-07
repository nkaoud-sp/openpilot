# Visual Vehicle Detector: no-command-line workflow

This add-on lets the comma UI manage the visual detector model from Tweaks.

## UI flow

Open:

```text
Settings -> Tweaks -> Visual Vehicle Detector (test) -> Manage Visual Detector Settings
```

Then use:

```text
Download ONNX
Compile PKL
```

The manager status row shows whether the ONNX/PKL files exist, their sizes, and the latest error/status.

## Runtime files

The buttons manage:

```text
selfdrive/modeld/models/visual_vehicle_detector.onnx
selfdrive/modeld/models/visual_vehicle_detector_tinygrad.pkl
selfdrive/modeld/models/visual_vehicle_detector_tinygrad.json
```

## Notes

- The model manager only runs offroad.
- Keep `Allow ONNX Fallback` off on comma3x for normal testing.
- The detector daemon still only writes a UI/debug status file. It does not control steering, speed, navigation, or lane changes.
