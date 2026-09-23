"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Rolling timing for a drive, so a model that misses its frame budget is readable afterwards.

cameraOdometry is marked invalid on any dropped camera frame (fill_pose_msg), and locationd
turns that into inputsOK = False, so "locationd Temporary Error" is usually modeld running
long rather than anything wrong with localization.

Two timings, because they fail differently. `exec` is the model itself. `gap` is everything
between one model run finishing and the next starting - the vipc wait, the submaster update,
publishing - which should be whatever is left of the frame period and nothing more. A model
that is simply too slow shows up as exec near the budget; a stall somewhere else shows up as
a large gap while exec stays small, and only the second kind can drop frames in bulk.
"""

BUCKET_MS = 2.0
BUCKETS = 64
# frame ids have no history to compare against on the first runs, so drops there are meaningless
WARMUP_RUNS = 10


class _Histogram:
  def __init__(self):
    self.counts = [0] * BUCKETS
    self.max_ms = 0.0
    self.runs = 0

  def update(self, milliseconds: float) -> None:
    self.runs += 1
    self.max_ms = max(self.max_ms, milliseconds)
    self.counts[min(BUCKETS - 1, max(0, int(milliseconds / BUCKET_MS)))] += 1

  def percentile_ms(self, fraction: float) -> float:
    """Upper edge of the bucket the given fraction of runs falls within, never above the
    largest value actually seen - a percentile reading higher than the maximum is nonsense."""
    if not self.runs:
      return 0.0
    target, seen = fraction * self.runs, 0
    for index, count in enumerate(self.counts):
      seen += count
      if seen >= target:
        return min((index + 1) * BUCKET_MS, self.max_ms)
    return min(BUCKETS * BUCKET_MS, self.max_ms)

  def describe(self, label: str) -> str:
    percentiles = f"p50 {self.percentile_ms(0.5):.0f}ms, p95 {self.percentile_ms(0.95):.0f}ms"
    return f"  {label} {percentiles}, max {self.max_ms:.0f}ms"


class RunStats:
  def __init__(self, model_freq: float):
    self.budget_ms = 1000.0 / model_freq
    self.runs = 0
    self.dropped = 0
    self.drop_events = 0
    self.worst_drop = 0
    self.over_budget = 0
    self.execution = _Histogram()
    self.gap = _Histogram()

  def update(self, execution_s: float, gap_s: float, dropped_frames: int) -> None:
    self.runs += 1

    execution_ms = execution_s * 1000.0
    self.execution.update(execution_ms)
    self.gap.update(gap_s * 1000.0)
    if execution_ms > self.budget_ms:
      self.over_budget += 1

    if self.runs > WARMUP_RUNS and dropped_frames:
      self.dropped += dropped_frames
      self.drop_events += 1
      self.worst_drop = max(self.worst_drop, dropped_frames)

  def summary(self) -> str:
    if not self.runs:
      return "no model runs yet"
    return "\n".join([
      f"{self.runs} runs, budget {self.budget_ms:.0f}ms",
      self.execution.describe("exec"),
      self.gap.describe("gap "),
      f"  over budget {self.over_budget} runs ({100.0 * self.over_budget / self.runs:.1f}%)",
      f"  dropped {self.dropped} frames in {self.drop_events} events, worst {self.worst_drop}",
      f"  {100.0 * self.drop_events / self.runs:.2f}% of runs invalidated cameraOdometry",
    ])
