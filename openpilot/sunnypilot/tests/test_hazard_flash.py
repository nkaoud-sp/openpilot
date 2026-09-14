"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import pytest

from openpilot.sunnypilot.hazard_flash import (
  CMD_ADDR,
  CMD_BUS,
  DEFAULT_DURATION_S,
  FLASH_PERIOD_MS,
  HALF_CYCLE_MS,
  REQUEST_CONSECUTIVE_OFF,
  REQUEST_CONSECUTIVE_ON,
  REQUEST_FIRST_FRAME,
  SCRIPT_RECORD_LEN,
  SESSION_DEFAULT,
  SESSION_EXTENDED,
  ScriptFrame,
  build_hazard_flash_frames,
  build_hazard_flash_script,
  build_hazard_stop_frames,
  build_hazard_stop_script,
  encode_script,
  flashes_for_duration,
  script_duration_s,
)


def decode_script(script: bytes) -> list[tuple[int, int, int, bytes]]:
  """The inverse of encode_script, written the way pandad reads the records."""
  assert len(script) % SCRIPT_RECORD_LEN == 0
  records = []
  for i in range(0, len(script), SCRIPT_RECORD_LEN):
    rec = script[i:i + SCRIPT_RECORD_LEN]
    delay_ms = (rec[0] << 8) | rec[1]
    addr = (rec[2] << 8) | rec[3]
    records.append((delay_ms, addr, rec[4], rec[6:6 + rec[5]]))
  return records


class TestEncodeScript:
  def test_record_layout(self):
    script = encode_script([ScriptFrame(300, b"\x02\x10\x03", addr=0x7C0, bus=1)])
    assert script == bytes([0x01, 0x2C, 0x07, 0xC0, 0x01, 0x08]) + b"\x02\x10\x03\x00\x00\x00\x00\x00"

  def test_round_trip(self):
    frames = build_hazard_flash_frames()
    records = decode_script(encode_script(frames))
    assert records == [(f.delay_ms, f.addr, f.bus, f.data) for f in frames]

  def test_rejects_out_of_range_delay(self):
    with pytest.raises(ValueError):
      encode_script([ScriptFrame(0x10000, SESSION_DEFAULT)])

  def test_rejects_oversized_frame(self):
    with pytest.raises(ValueError):
      encode_script([ScriptFrame(0, b"\x00" * 9)])


class TestHazardFlashScript:
  def test_all_frames_addressed_to_the_lamp_ecu(self):
    for _delay_ms, addr, bus, data in decode_script(build_hazard_flash_script()):
      assert addr == CMD_ADDR
      assert bus == CMD_BUS
      assert len(data) == 8, "ISO 15765-4 frames are 8 bytes, and the ELM327 safety mode drops the rest"

  @pytest.mark.parametrize("flashes", [1, 3, 10])
  def test_one_on_off_pair_per_flash(self, flashes):
    data = [rec[3] for rec in decode_script(build_hazard_flash_script(flashes))]
    assert data.count(REQUEST_CONSECUTIVE_ON) == flashes
    assert data.count(REQUEST_CONSECUTIVE_OFF) == flashes
    assert data.count(REQUEST_FIRST_FRAME) == 2 * flashes

  def test_opens_and_closes_the_session(self):
    data = [rec[3] for rec in decode_script(build_hazard_flash_script())]
    # 0x2F is only accepted in the extended session, and the closing reset returns the lamps.
    assert data[0] == SESSION_DEFAULT
    assert data[1] == b"\x02\x10\x03\x00\x00\x00\x00\x00"
    assert data[-1] == SESSION_DEFAULT

  def test_lamps_always_end_up_off(self):
    """Whatever else happens, the last state written is off, so a run can't leave them latched."""
    for flashes in range(1, 5):
      data = [rec[3] for rec in decode_script(build_hazard_flash_script(flashes))]
      states = [d for d in data if d in (REQUEST_CONSECUTIVE_ON, REQUEST_CONSECUTIVE_OFF)]
      assert states[-1] == REQUEST_CONSECUTIVE_OFF
      assert states == [REQUEST_CONSECUTIVE_ON, REQUEST_CONSECUTIVE_OFF] * flashes

  def test_blink_rate_matches_the_capture(self):
    records = decode_script(build_hazard_flash_script())
    elapsed, starts = 0, []
    for delay_ms, _, _, data in records:
      elapsed += delay_ms
      if data == REQUEST_FIRST_FRAME:
        starts.append(elapsed)
    assert all(b - a == HALF_CYCLE_MS for a, b in zip(starts, starts[1:], strict=False))

  def test_consecutive_frame_follows_its_first_frame(self):
    """An ISO-TP consecutive frame sent too late is dropped: N_Cr is 1 s on a stock stack."""
    records = decode_script(build_hazard_flash_script())
    for (_, _, _, first), (delay_ms, _, _, consecutive) in zip(records, records[1:], strict=False):
      if first == REQUEST_FIRST_FRAME:
        assert consecutive in (REQUEST_CONSECUTIVE_ON, REQUEST_CONSECUTIVE_OFF)
        assert 0 < delay_ms < 1000

  def test_default_run_lasts_a_minute(self):
    # Within one flash of the minute: the sequence can only end on a dark half cycle.
    duration = script_duration_s(build_hazard_flash_frames())
    assert abs(duration - DEFAULT_DURATION_S) < FLASH_PERIOD_MS / 1000

  @pytest.mark.parametrize("seconds", [5, 30, 60, 300])
  def test_duration_scales(self, seconds):
    frames = build_hazard_flash_frames(flashes_for_duration(seconds))
    assert abs(script_duration_s(frames) - seconds) < FLASH_PERIOD_MS / 1000

  def test_never_builds_an_empty_run(self):
    """A press has to do something visible, however short the duration asked for."""
    assert flashes_for_duration(0) == 1

  def test_delays_fit_the_record(self):
    for delay_ms, _, _, _ in decode_script(build_hazard_flash_script()):
      assert 0 <= delay_ms <= 0xFFFF


class TestHazardStopScript:
  def test_puts_the_lamps_out_and_closes_the_session(self):
    data = [rec[3] for rec in decode_script(build_hazard_stop_script())]
    # The session is reopened first: a run that already ended has timed out of the extended one.
    assert data[0] == SESSION_EXTENDED
    assert REQUEST_CONSECUTIVE_ON not in data
    assert REQUEST_CONSECUTIVE_OFF in data
    assert data[-1] == SESSION_DEFAULT

  def test_short_enough_to_feel_immediate(self):
    assert script_duration_s(build_hazard_stop_frames()) < 0.5

  def test_frames_match_the_flash_script(self):
    """Both scripts talk to the same ECU the same way, so a stop can't be left behind by a change."""
    flash = {rec[3] for rec in decode_script(build_hazard_flash_script())}
    for _delay_ms, addr, bus, data in decode_script(build_hazard_stop_script()):
      assert (addr, bus) == (CMD_ADDR, CMD_BUS)
      assert data in flash
