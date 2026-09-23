"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Rolling timing for a drive, so a model that misses its frame budget is readable afterwards.

cameraOdometry is marked invalid on any dropped camera frame (fill_pose_msg), and locationd
turns that into inputsOK = False, so "locationd Temporary Error" is usually modeld running
long rather than anything wrong with localization. The execution time that decides it is
already measured per frame; this keeps enough of it to answer the question offroad.
"""

BUCKET_MS = 2.0
BUCKETS = 64


class RunStats:
  def __init__(self, model_freq: float):
    self.budget_ms = 1000.0 / model_freq
    self.runs = 0
    self.dropped = 0
    self.drop_events = 0
    self.over_budget = 0
    self.max_ms = 0.0
    self.histogram = [0] * BUCKETS

  def update(self, execution_s: float, dropped_frames: int) -> None:
    self.runs += 1
    self.dropped += dropped_frames
    if dropped_frames:
      self.drop_events += 1

    execution_ms = execution_s * 1000.0
    self.max_ms = max(self.max_ms, execution_ms)
    if execution_ms > self.budget_ms:
      self.over_budget += 1
    self.histogram[min(BUCKETS - 1, max(0, int(execution_ms / BUCKET_MS)))] += 1

  def percentile_ms(self, fraction: float) -> float:
    """Upper edge of the bucket the given fraction of runs falls within."""
    if not self.runs:
      return 0.0
    target, seen = fraction * self.runs, 0
    for index, count in enumerate(self.histogram):
      seen += count
      if seen >= target:
        return (index + 1) * BUCKET_MS
    return BUCKETS * BUCKET_MS

  def summary(self) -> str:
    if not self.runs:
      return "no model runs yet"
    return "\n".join([
      f"{self.runs} runs, budget {self.budget_ms:.0f}ms",
      f"  exec p50 {self.percentile_ms(0.5):.0f}ms, p95 {self.percentile_ms(0.95):.0f}ms, max {self.max_ms:.0f}ms",
      f"  over budget {self.over_budget} runs ({100.0 * self.over_budget / self.runs:.1f}%)",
      f"  dropped {self.dropped} camera frames in {self.drop_events} events",
      f"  {100.0 * self.drop_events / self.runs:.2f}% of runs invalidated cameraOdometry",
    ])
