"""Death-window ticker telemetry: lifecycle, closed-set format, no blocking.

Real threads and real small files only; no model, media, or network.
"""

from __future__ import annotations

import re
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np  # noqa: F401  (must precede the patch.dict block below)

from courselens_worker import asr
from courselens_worker.asr import (
    _ChunkTicker,
    _mem_available_kb,
    _mem_available_kb_from,
)


TICK_LINE_RE = re.compile(
    r"^stage=asr-tick elapsed=\d+ chunk=\d+ pcm_bytes=\d+ mem_avail_kb=-?\d+$"
)
TICK_END_RE = re.compile(r"^stage=asr-tick-end elapsed=\d+ chunks=\d+$")

MEMINFO_TEXT = (
    "MemTotal:       16384000 kB\n"
    "MemFree:        4200000 kB\n"
    "MemAvailable:   12345678 kB\n"
    "Buffers:         100000 kB\n"
    "Cached:         3100000 kB\n"
)


def _wait_for(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class MemAvailableTests(unittest.TestCase):
    def test_parses_mem_available_kilobytes(self):
        self.assertEqual(_mem_available_kb_from(MEMINFO_TEXT), 12345678)

    def test_returns_minus_one_when_field_missing(self):
        self.assertEqual(_mem_available_kb_from("MemTotal: 1 kB\n"), -1)

    def test_returns_minus_one_on_malformed_value(self):
        self.assertEqual(_mem_available_kb_from("MemAvailable: not-a-number kB\n"), -1)

    def test_missing_platform_file_returns_minus_one(self):
        missing = Path(tempfile.gettempdir()) / "courselens-no-such-meminfo"
        self.assertEqual(_mem_available_kb(missing), -1)


class ChunkTickerTests(unittest.TestCase):
    def _state(self, **overrides) -> dict:
        state = {"chunk": 0, "pcm": None, "done": 0}
        state.update(overrides)
        return state

    def test_emits_closed_set_tick_lines_with_state_and_pcm_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            pcm = Path(temporary) / "chunk-0000.f32le"
            pcm.write_bytes(b"\x00" * 1234)
            lines: list[str] = []
            with patch.object(asr, "_mem_available_kb", return_value=2048):
                ticker = _ChunkTicker(
                    self._state(chunk=3, pcm=pcm, done=1),
                    interval=0.05,
                    emit=lines.append,
                )
                ticker.start()
                self.assertTrue(_wait_for(lambda: len(lines) >= 2))
                ticker.stop()
        ticks = [line for line in lines if line.startswith("stage=asr-tick ")]
        self.assertTrue(ticks)
        for line in ticks:
            self.assertRegex(line, TICK_LINE_RE)
            self.assertIn("chunk=3", line)
            self.assertIn("pcm_bytes=1234", line)
            self.assertIn("mem_avail_kb=2048", line)
        self.assertRegex(lines[-1], TICK_END_RE)

    def test_stop_emits_termination_line_with_completed_chunks(self):
        lines: list[str] = []
        ticker = _ChunkTicker(self._state(done=4), interval=60.0, emit=lines.append)
        ticker.start()
        ticker.stop()
        self.assertEqual(len(lines), 1)
        self.assertRegex(lines[0], TICK_END_RE)
        self.assertIn("chunks=4", lines[0])
        self.assertFalse(ticker._thread.is_alive())

    def test_stop_before_start_is_safe(self):
        lines: list[str] = []
        ticker = _ChunkTicker(self._state(), interval=60.0, emit=lines.append)
        ticker.stop()
        self.assertEqual(len(lines), 1)
        self.assertRegex(lines[0], TICK_END_RE)

    def test_thread_is_daemon_and_stop_is_prompt(self):
        lines: list[str] = []
        ticker = _ChunkTicker(self._state(), interval=0.05, emit=lines.append)
        ticker.start()
        self.assertTrue(ticker._thread.daemon)
        started = time.monotonic()
        ticker.stop()
        self.assertLess(time.monotonic() - started, 2.0)

    def test_ticker_does_not_block_the_main_loop(self):
        lines: list[str] = []
        ticker = _ChunkTicker(self._state(chunk=1), interval=0.05, emit=lines.append)
        ticker.start()
        try:
            started = time.monotonic()
            for _ in range(20):
                time.sleep(0.005)
            self.assertLess(time.monotonic() - started, 1.0)
            self.assertGreaterEqual(len(lines), 1)
        finally:
            ticker.stop()
        self.assertTrue(_wait_for(lambda: not ticker._thread.is_alive()))


if __name__ == "__main__":
    unittest.main()
