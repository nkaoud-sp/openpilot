import math

import pytest

from openpilot.common.constants import CV
from openpilot.sunnypilot.nkaoud_nav.geometry import Coordinate
from openpilot.sunnypilot.nkaoud_nav.starpilot_navigation import (
  NAV_TURN_COMFORT_DECEL, NAV_TURN_DISTANCE_BUFFER, StarPilotNavigationProvider,
  StarPilotRoute, _nav_instruction_state, starpilot_turn_speed_target,
)


def _mapbox_response():
  # A small eastbound route. The banner belongs to the current (depart) step
  # and describes the upcoming left turn, which mirrors Mapbox / StarPilot
  # instruction semantics.
  return {
    "code": "Ok",
    "uuid": "starpilot-test-route",
    "routes": [{
      "distance": 333.0,
      "duration": 30.0,
      "geometry": {
        "coordinates": [
          [0.0000, 0.0000],
          [0.0010, 0.0000],
          [0.0020, 0.0000],
          [0.0030, 0.0000],
        ],
      },
      "legs": [{
        "annotation": {
          "maxspeed": [
            {"speed": 50, "unit": "km/h"},
            {"speed": 50, "unit": "km/h"},
            {"speed": 50, "unit": "km/h"},
          ],
        },
        "steps": [
          {
            "distance": 111.0,
            "duration": 10.0,
            "maneuver": {
              "type": "depart",
              "instruction": "Head east",
              "location": [0.0000, 0.0000],
            },
            "bannerInstructions": [{
              "distanceAlongGeometry": 80.0,
              "primary": {"text": "Turn left", "type": "turn", "modifier": "left"},
              "sub": {
                "components": [{
                  "type": "lane",
                  "active": True,
                  "directions": ["left"],
                  "active_direction": "left",
                }],
              },
            }],
          },
          {
            "distance": 111.0,
            "duration": 10.0,
            "maneuver": {
              "type": "turn",
              "modifier": "left",
              "instruction": "Turn left",
              "location": [0.0010, 0.0000],
            },
          },
          {
            "distance": 111.0,
            "duration": 10.0,
            "maneuver": {
              "type": "arrive",
              "instruction": "Your destination is on the right",
              "location": [0.0020, 0.0000],
            },
          },
        ],
      }],
    }],
  }


@pytest.fixture
def route():
  parsed = StarPilotRoute.from_mapbox_response(_mapbox_response())
  assert parsed is not None
  return parsed


def test_route_progress_instruction_and_reroute_thresholds(route):
  progress = route.get_progress(Coordinate(0.0, 0.0005))
  assert progress is not None
  assert progress.current_step_index == 0
  assert progress.next_step is not None
  assert progress.next_step.maneuver == "turn"

  instruction = route.build_instruction_payload(progress)
  assert instruction["maneuverType"] == "turn"
  assert instruction["maneuverModifier"] == "left"
  assert instruction["lanes"] == [{
    "directions": ["left"],
    "active": True,
    "activeDirection": "left",
  }]
  assert len(instruction["allManeuvers"]) == 3

  off_route_progress = route.get_progress(Coordinate(0.0007, 0.0005))
  assert off_route_progress is not None
  assert route.off_route_distance_exceeded(off_route_progress, 0.0)
  assert not route.off_route_distance_exceeded(off_route_progress, 40.0)
  assert route.route_bearing_misaligned(progress.closest_segment_index, 200.0, 3.0)
  assert not route.route_bearing_misaligned(progress.closest_segment_index, 200.0, 2.0)


def test_starpilot_instruction_state_keeps_full_lane_guidance(route):
  progress = route.get_progress(Coordinate(0.0, 0.0008))
  assert progress is not None

  route_state = _nav_instruction_state(route.build_instruction_payload(progress))
  assert route_state["valid"]
  assert route_state["maneuverType"] == "turn"
  assert route_state["nextManeuverType"] == "turn"
  assert route_state["nextManeuverModifier"] == "left"

  slight_instruction = {
    "maneuverType": "off ramp",
    "maneuverModifier": "slightRight",
    "maneuverDistance": 50.0,
    "lanes": [{
      "directions": ["straight", "slightRight"],
      "active": True,
      "activeDirection": "slightRight",
    }],
    "allManeuvers": [],
  }
  state = _nav_instruction_state(slight_instruction)
  assert state == {
    "valid": True,
    "maneuverModifier": "slightRight",
    "maneuverType": "off ramp",
    "maneuverDistance": 50.0,
    "nextManeuverType": "",
    "nextManeuverModifier": "",
    "nextManeuverDistance": 0.0,
    "laneCount": 1,
    "activeLaneDirection": "slightRight",
    "activeLaneIndex": 0,
    "activeLaneAtRoadEdge": True,
    "hasSharedSameSideLane": True,
    "sameSideLaneCount": 1,
  }


def test_starpilot_turn_cap_never_raises_cruise(route):
  progress = route.get_progress(Coordinate(0.0, 0.0008))
  assert progress is not None
  instruction = route.build_instruction_payload(progress)
  cap = starpilot_turn_speed_target(instruction, 30.0)
  turn_speed = 14.0 * CV.MPH_TO_MS
  expected = math.sqrt(turn_speed ** 2 + 2.0 * NAV_TURN_COMFORT_DECEL *
                       max(instruction["maneuverDistance"] - NAV_TURN_DISTANCE_BUFFER, 0.0))
  assert cap == pytest.approx(expected)
  assert cap < 30.0
  assert starpilot_turn_speed_target(instruction, 1.0) == 0.0

  uturn = {
    "maneuverType": "turn",
    "maneuverModifier": "uturn",
    "maneuverDistance": 8.0,
    "allManeuvers": [],
  }
  assert starpilot_turn_speed_target(uturn, 30.0, min_steer_speed=4.0) == pytest.approx(4.0)

  next_turn = {
    "maneuverType": "depart",
    "maneuverModifier": "none",
    "maneuverDistance": 90.0,
    "allManeuvers": [
      {"type": "depart", "modifier": "none", "distance": 90.0},
      {"type": "roundabout", "modifier": "right", "distance": 30.0},
    ],
  }
  assert 0.0 < starpilot_turn_speed_target(next_turn, 30.0) < 30.0


def test_provider_rejects_old_route_and_stops_after_arrival(route):
  provider = StarPilotNavigationProvider()
  destination = Coordinate(0.0, 0.0030)
  position = Coordinate(0.0, 0.0005)
  assert provider.set_destination(destination)
  request = provider.next_fetch_request(position, 90.0, "token", 0.0, False)
  assert request is not None
  assert provider.accept_fetch(request.key, route)

  active = provider.update(position, 90.0, 5.0, 30.0, 0.0, 0.0)
  assert active.valid
  assert active.active
  assert active.route is route

  off_route = Coordinate(0.0007, 0.0005)
  provider.update(off_route, 90.0, 1.0, 30.0, 1.0, 0.0)
  rerouting = provider.update(off_route, 90.0, 1.0, 30.0, 3.1, 0.0)
  assert rerouting.rerouting
  assert not rerouting.valid
  assert rerouting.instruction_state == {}
  assert rerouting.maneuver_target_speed == 0.0

  other_destination = Coordinate(0.0, 0.0040)
  assert provider.set_destination(other_destination)
  assert not provider.accept_fetch(request.key, route)

  assert provider.set_destination(destination)
  request = provider.next_fetch_request(position, 90.0, "token", 1.0, False)
  assert request is not None
  assert provider.accept_fetch(request.key, route)
  near_destination = Coordinate(0.0, 0.00275)
  provider.update(near_destination, 90.0, 1.0, 30.0, 2.0, 0.0)
  arrived = provider.update(near_destination, 90.0, 1.0, 30.0, 7.1, 0.0)
  assert arrived.arrived
  assert not arrived.active
  # The provider never owns a Params value. navd observes this state and
  # clears the target's existing NkaoudNavDestination parameter.
  assert provider.destination == destination
  assert provider.next_fetch_request(near_destination, 90.0, "token", 8.0, False) is None
