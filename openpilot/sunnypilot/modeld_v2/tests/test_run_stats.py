"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.modeld_v2.run_stats import BUCKET_MS, BUCKETS, WARMUP_RUNS, RunStats


class TestRunStats(OpenpilotTestCase):
  def test_empty(self):
    assert RunStats(20.0).summary() == "no model runs yet"

  def test_budget_comes_from_the_model_frequency(self):
    assert RunStats(20.0).budget_ms == 50.0
    assert RunStats(10.0).budget_ms == 100.0

  def test_counts_only_runs_over_budget(self):
    stats = RunStats(20.0)
    for execution_ms in (10.0, 49.9, 50.0, 50.1, 90.0):
      stats.update(execution_ms / 1000.0, 0.004, 0)
    assert stats.runs == 5
    assert stats.over_budget == 2
    assert stats.execution.max_ms == 90.0

  def test_drop_events_are_distinct_from_dropped_frames(self):
    stats = RunStats(20.0)
    for _ in range(WARMUP_RUNS):
      stats.update(0.01, 0.004, 0)
    stats.update(0.01, 0.004, 3)
    stats.update(0.01, 0.004, 1)
    assert stats.dropped == 4
    assert stats.drop_events == 2
    assert stats.worst_drop == 3

  def test_warmup_drops_are_ignored(self):
    # the first run has no previous frame id to compare against, so camerad's whole count
    # shows up as a drop and would otherwise dominate the totals
    stats = RunStats(20.0)
    stats.update(0.01, 0.004, 5000)
    assert stats.dropped == 0
    assert stats.drop_events == 0

  def test_gap_is_tracked_separately_from_execution(self):
    stats = RunStats(20.0)
    for _ in range(99):
      stats.update(0.046, 0.004, 0)
    stats.update(0.046, 5.0, 96)  # a stall outside the model, which is what drops frames in bulk
    assert stats.execution.max_ms == 46.0
    assert stats.gap.max_ms == 5000.0
    assert stats.over_budget == 0
    assert stats.dropped == 96

  def test_percentiles_never_exceed_the_observed_maximum(self):
    stats = RunStats(20.0)
    for _ in range(10):
      stats.update(0.046, 0.004, 0)
    # 46ms sits inside the 46-48ms bucket, but nothing ever took 48ms
    assert stats.execution.percentile_ms(0.5) == 46.0
    assert stats.execution.max_ms == 46.0

  def test_percentiles_track_the_distribution(self):
    stats = RunStats(20.0)
    for _ in range(95):
      stats.update(0.020, 0.004, 0)
    for _ in range(5):
      stats.update(0.080, 0.004, 0)
    assert stats.execution.percentile_ms(0.5) == 22.0
    assert stats.execution.percentile_ms(0.95) == 22.0
    assert stats.execution.percentile_ms(0.99) == 80.0  # clamped to the observed max

  def test_outliers_land_in_the_last_bucket_rather_than_out_of_range(self):
    stats = RunStats(20.0)
    stats.update(10.0, 0.004, 0)  # 10 seconds, far past the histogram
    assert stats.execution.counts[BUCKETS - 1] == 1
    assert stats.execution.percentile_ms(0.5) == BUCKETS * BUCKET_MS
    # the true maximum is kept exactly, unlike the bucketed percentiles
    assert stats.execution.max_ms == 10000.0

  def test_summary_reports_what_the_alert_depends_on(self):
    stats = RunStats(20.0)
    for _ in range(99):
      stats.update(0.020, 0.004, 0)
    stats.update(0.060, 0.004, 2)

    summary = stats.summary()
    assert "100 runs, budget 50ms" in summary
    assert "exec p50 22ms" in summary
    assert "gap  p50 4ms" in summary
    assert "over budget 1 runs (1.0%)" in summary
    assert "dropped 2 frames in 1 events, worst 2" in summary
    assert "1.00% of runs invalidated cameraOdometry" in summary
