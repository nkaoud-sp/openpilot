"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.modeld_v2.run_stats import BUCKET_MS, BUCKETS, RunStats


class TestRunStats(OpenpilotTestCase):
  def test_empty(self):
    assert RunStats(20.0).summary() == "no model runs yet"

  def test_budget_comes_from_the_model_frequency(self):
    assert RunStats(20.0).budget_ms == 50.0
    assert RunStats(10.0).budget_ms == 100.0

  def test_counts_only_runs_over_budget(self):
    stats = RunStats(20.0)
    for execution_ms in (10.0, 49.9, 50.0, 50.1, 90.0):
      stats.update(execution_ms / 1000.0, 0)
    assert stats.runs == 5
    assert stats.over_budget == 2
    assert stats.max_ms == 90.0

  def test_drop_events_are_distinct_from_dropped_frames(self):
    stats = RunStats(20.0)
    stats.update(0.01, 0)
    stats.update(0.01, 3)
    stats.update(0.01, 1)
    assert stats.dropped == 4
    assert stats.drop_events == 2

  def test_percentiles_track_the_distribution(self):
    stats = RunStats(20.0)
    for _ in range(95):
      stats.update(0.020, 0)
    for _ in range(5):
      stats.update(0.080, 0)
    assert stats.percentile_ms(0.5) == 22.0
    assert stats.percentile_ms(0.95) == 22.0
    assert stats.percentile_ms(0.99) == 82.0

  def test_outliers_land_in_the_last_bucket_rather_than_out_of_range(self):
    stats = RunStats(20.0)
    stats.update(10.0, 0)  # 10 seconds, far past the histogram
    assert stats.histogram[BUCKETS - 1] == 1
    assert stats.percentile_ms(0.5) == BUCKETS * BUCKET_MS
    # the true maximum is kept exactly, unlike the bucketed percentiles
    assert stats.max_ms == 10000.0

  def test_summary_reports_what_the_alert_depends_on(self):
    stats = RunStats(20.0)
    for _ in range(99):
      stats.update(0.020, 0)
    stats.update(0.060, 2)

    summary = stats.summary()
    assert "100 runs, budget 50ms" in summary
    assert "over budget 1 runs (1.0%)" in summary
    assert "dropped 2 camera frames in 1 events" in summary
    assert "1.00% of runs invalidated cameraOdometry" in summary
