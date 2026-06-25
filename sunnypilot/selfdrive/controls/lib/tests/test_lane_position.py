from types import SimpleNamespace

from openpilot.sunnypilot.selfdrive.controls.lib.lane_position import (
  LanePositionEstimator,
  EDGE_BLOCK_COUNTER_MAX,
  FILTER_MODE_NONE,
  FILTER_MODE_WIDTH,
  FILTER_MODE_SEPARATION,
  FILTER_MODE_BOTH_AND,
  FILTER_MODE_BOTH_OR,
)


def make_line(y: float, xs: tuple[float, ...] = (5.0, 15.0, 30.0)):
  return SimpleNamespace(
    x=list(xs),
    y=[y] * len(xs),
    z=[0.0] * len(xs),
  )


def make_model(dist_left: float, dist_right: float,
               lane_ys: tuple[float, float, float, float],
               lane_probs: tuple[float, float, float, float]):
  return SimpleNamespace(
    roadEdges=(make_line(-dist_left), make_line(dist_right)),
    roadEdgeStds=(0.6, 0.6),
    laneLines=tuple(make_line(y) for y in lane_ys),
    laneLineProbs=lane_probs,
  )


def _settle(est, model, mode, n=3):
  out = (0, 0, "unknown")
  for _ in range(n):
    out = est.update(model, filter_mode=mode)
  return out


def test_none_mode_matches_base_behaviour():
  est = LanePositionEstimator()
  model = make_model(
    dist_left=9.25,
    dist_right=5.55,
    lane_ys=(-5.4, -1.8, 1.8, 3.8),
    lane_probs=(0.1, 0.95, 0.95, 0.1),
  )
  current, total, conf = _settle(est, model, FILTER_MODE_NONE, 3)
  assert (current, total) == (3, 4)
  assert conf == "high"
  assert est.debug.blocked_right is False
  assert est.debug.blocked_left is False


def test_both_and_blocks_narrow_edge_lane_after_persistence():
  est = LanePositionEstimator()
  model = make_model(
    dist_left=9.25,
    dist_right=5.55,
    lane_ys=(-5.4, -1.8, 1.8, 3.8),
    lane_probs=(0.1, 0.95, 0.95, 0.1),
  )
  current, total, conf = _settle(est, model, FILTER_MODE_BOTH_AND, 3)
  assert (current, total, conf) == (3, 3, "high")
  assert est.debug.raw_current_lane == 3
  assert est.debug.raw_total_lanes == 4
  assert est.debug.blocked_right is True
  assert est.debug.width_vote_right is True
  assert est.debug.separation_vote_right is True


def test_both_and_does_not_block_when_outer_lane_looks_normal_width():
  est = LanePositionEstimator()
  model = make_model(
    dist_left=9.25,
    dist_right=5.55,
    lane_ys=(-5.4, -1.8, 1.8, 5.4),
    lane_probs=(0.1, 0.95, 0.95, 0.1),
  )
  for _ in range(6):
    current, total, conf = est.update(model, filter_mode=FILTER_MODE_BOTH_AND)
  assert (current, total, conf) == (3, 4, "high")
  assert est.debug.blocked_right is False
  assert est.debug.width_vote_right is False
  assert est.debug.separation_vote_right is True


def test_width_only_blocks_on_narrow_edge_alone():
  # Lane-line probs are uniform so separation vote is False; width alone should drive the block.
  est = LanePositionEstimator()
  model = make_model(
    dist_left=9.25,
    dist_right=5.55,
    lane_ys=(-5.4, -1.8, 1.8, 3.8),
    lane_probs=(0.95, 0.95, 0.95, 0.95),
  )
  current, total, _ = _settle(est, model, FILTER_MODE_WIDTH, 3)
  assert (current, total) == (3, 3)
  assert est.debug.blocked_right is True
  assert est.debug.width_vote_right is True
  assert est.debug.separation_vote_right is False


def test_separation_only_blocks_when_inner_strong_outer_weak():
  # Strong inner lines with weak outer lines on both sides → both edges demoted.
  est = LanePositionEstimator()
  model = make_model(
    dist_left=9.25,
    dist_right=5.55,
    lane_ys=(-5.4, -1.8, 1.8, 5.4),
    lane_probs=(0.1, 0.95, 0.95, 0.1),
  )
  current, total, _ = _settle(est, model, FILTER_MODE_SEPARATION, 3)
  assert (current, total) == (2, 2)
  assert est.debug.blocked_left is True
  assert est.debug.blocked_right is True
  assert est.debug.separation_vote_left is True
  assert est.debug.separation_vote_right is True
  assert est.debug.width_vote_right is False


def test_both_or_blocks_on_separation_alone():
  est = LanePositionEstimator()
  model = make_model(
    dist_left=9.25,
    dist_right=5.55,
    lane_ys=(-5.4, -1.8, 1.8, 5.4),
    lane_probs=(0.1, 0.95, 0.95, 0.1),
  )
  current, total, _ = _settle(est, model, FILTER_MODE_BOTH_OR, 3)
  assert (current, total) == (2, 2)
  assert est.debug.blocked_left is True
  assert est.debug.blocked_right is True


def test_block_does_not_chatter_when_vote_toggles_each_frame():
  # Drive the counter up to the ON threshold (3), then alternate True/False every frame.
  # With symmetric +1/-1 the counter oscillates 3<->2 around the threshold; the asymmetric
  # OFF latch must keep `blocked_right` stuck True until the counter actually falls to <=1.
  est = LanePositionEstimator()
  blocking_model = make_model(
    dist_left=9.25,
    dist_right=5.55,
    lane_ys=(-5.4, -1.8, 1.8, 3.8),
    lane_probs=(0.1, 0.95, 0.95, 0.1),
  )
  clean_model = make_model(
    dist_left=9.25,
    dist_right=5.55,
    lane_ys=(-5.4, -1.8, 1.8, 5.4),
    lane_probs=(0.95, 0.95, 0.95, 0.95),
  )

  # Engage the block.
  for _ in range(3):
    est.update(blocking_model, filter_mode=FILTER_MODE_WIDTH)
  assert est.debug.blocked_right is True

  # Now alternate blocking/clean frames. blocked_right must stay True the whole way.
  for i in range(8):
    est.update(blocking_model if i % 2 == 0 else clean_model, filter_mode=FILTER_MODE_WIDTH)
    assert est.debug.blocked_right is True, f"flipped off at frame {i}"


def test_block_releases_after_sustained_clean_run():
  est = LanePositionEstimator()
  blocking_model = make_model(
    dist_left=9.25,
    dist_right=5.55,
    lane_ys=(-5.4, -1.8, 1.8, 3.8),
    lane_probs=(0.1, 0.95, 0.95, 0.1),
  )
  clean_model = make_model(
    dist_left=9.25,
    dist_right=5.55,
    lane_ys=(-5.4, -1.8, 1.8, 5.4),
    lane_probs=(0.95, 0.95, 0.95, 0.95),
  )
  for _ in range(EDGE_BLOCK_COUNTER_MAX):
    est.update(blocking_model, filter_mode=FILTER_MODE_WIDTH)
  assert est.debug.blocked_right is True

  # Counter starts at MAX=6, needs to fall to <= OFF=1 -> 5 clean frames.
  for _ in range(5):
    est.update(clean_model, filter_mode=FILTER_MODE_WIDTH)
  assert est.debug.blocked_right is False


def test_width_only_ignores_separation_signal():
  # Strong separation, but edge lane is normal-width → no block under WIDTH mode.
  est = LanePositionEstimator()
  model = make_model(
    dist_left=9.25,
    dist_right=5.55,
    lane_ys=(-5.4, -1.8, 1.8, 5.4),
    lane_probs=(0.1, 0.95, 0.95, 0.1),
  )
  for _ in range(6):
    current, total, _ = est.update(model, filter_mode=FILTER_MODE_WIDTH)
  assert (current, total) == (3, 4)
  assert est.debug.blocked_right is False
