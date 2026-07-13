"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from openpilot.selfdrive.ui.sunnypilot.onroad.chevron_metrics import ChevronMetrics
from openpilot.selfdrive.ui.sunnypilot.onroad.follow_readout import FollowReadout
from openpilot.selfdrive.ui.sunnypilot.onroad.jerk_readout import JerkReadout
from openpilot.selfdrive.ui.sunnypilot.onroad.lane_position_indicator import LanePositionIndicator
from openpilot.selfdrive.ui.sunnypilot.onroad.lane_line_visualizer_readout import LaneLineVisualizerReadout
from openpilot.selfdrive.ui.sunnypilot.onroad.launch_readout import LaunchReadout
from openpilot.selfdrive.ui.sunnypilot.onroad.model_frame_drops_readout import ModelFrameDropsReadout
from openpilot.selfdrive.ui.sunnypilot.onroad.visual_vehicle_readout import VisualVehicleReadout
from openpilot.selfdrive.ui.sunnypilot.onroad.rainbow_path import RainbowPath


class ModelRendererSP:
  def __init__(self):
    self.rainbow_path = RainbowPath()
    self.chevron_metrics = ChevronMetrics()
    self.follow_readout = FollowReadout()
    self.jerk_readout = JerkReadout()
    self.lane_position_indicator = LanePositionIndicator()
    self.lane_line_visualizer_readout = LaneLineVisualizerReadout()
    self.launch_readout = LaunchReadout()
    self.model_frame_drops_readout = ModelFrameDropsReadout()
    self.visual_vehicle_readout = VisualVehicleReadout()
