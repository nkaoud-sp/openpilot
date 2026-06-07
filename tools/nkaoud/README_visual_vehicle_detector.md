# Visual Vehicle Detector test path

This feature is standalone UI/debug only. It does not control steering, speed, navigation, or lane changes.

## Runtime files

Preferred on comma3x:

```text
selfdrive/modeld/models/visual_vehicle_detector_tinygrad.pkl
selfdrive/modeld/models/visual_vehicle_detector_tinygrad.json
```

Optional debug fallback:

```text
selfdrive/modeld/models/visual_vehicle_detector.onnx
```

The daemon prefers the tinygrad `.pkl`. It only attempts ONNX Runtime if `VisualVehicleDetectorAllowOnnx` is enabled in Tweaks.

## Export ONNX on PC

```bash
pip install ultralytics
mkdir -p selfdrive/modeld/models
yolo export model=yolo11n.pt format=onnx imgsz=320 simplify=True opset=12
cp yolo11n.onnx selfdrive/modeld/models/visual_vehicle_detector.onnx
```

## Compile ONNX to tinygrad pkl on comma3x

```bash
cd /data/openpilot
python3 tools/nkaoud/compile_visual_vehicle_detector_tinygrad.py \
  --onnx selfdrive/modeld/models/visual_vehicle_detector.onnx \
  --out selfdrive/modeld/models/visual_vehicle_detector_tinygrad.pkl \
  --imgsz 320
```

Keep the generated `.json` beside the `.pkl`; the daemon uses it to recover input name/shape.

## Enable

Open Tweaks:

- Visual Vehicle Detector (test): ON
- Manage Visual Detector Settings
- Show Detector Readout: ON
- Allow ONNX Fallback: OFF on comma3x unless only testing process/UI behavior
- Log Detector Debug: OFF unless tuning
