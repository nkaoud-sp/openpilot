import unittest

from openpilot.sunnypilot.autolock_commands import (
  CMD_ADDR, LOCK_CMD, MIRR_FOLD_ALL, WINDOW_CLOSE_ALL, build_queue, frame_record,
)


class TestAutoLockCommands(unittest.TestCase):
  def test_frame_record(self):
    record = frame_record(LOCK_CMD)
    self.assertEqual(len(record), 12)
    self.assertEqual(record[:4], bytes([CMD_ADDR >> 8, CMD_ADDR & 0xFF, 0, 8]))
    self.assertEqual(record[4:], LOCK_CMD)

  def test_lock_only(self):
    self.assertEqual(build_queue(False, False), frame_record(LOCK_CMD))

  def test_optional_commands_precede_lock(self):
    queue = build_queue(True, True)
    records = [queue[index:index + 12] for index in range(0, len(queue), 12)]

    self.assertEqual(len(records), len(WINDOW_CLOSE_ALL) + len(MIRR_FOLD_ALL) + 1)
    self.assertEqual(records[0], frame_record(WINDOW_CLOSE_ALL[0]))
    self.assertEqual(records[len(WINDOW_CLOSE_ALL)], frame_record(MIRR_FOLD_ALL[0]))
    self.assertEqual(records[-1], frame_record(LOCK_CMD))

  def test_no_commands_when_lock_disabled(self):
    self.assertEqual(build_queue(False, False, lock=False), b"")
