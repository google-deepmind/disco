# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Unit tests for target response scraping tolerance and backfill handling."""

import asyncio
import random
from unittest import mock

from absl.testing import absltest

from disco.agent import scrape_tolerance


class IsThrottlingErrorTest(absltest.TestCase):

  def test_matches_throttler_text(self):
    exc = RuntimeError("Request throttled due to rate limiting.")
    self.assertTrue(scrape_tolerance.is_throttling_error(exc))

  def test_matches_conventional_quota_errors(self):
    for text in (
        "RESOURCE_EXHAUSTED: quota exceeded",
        "HTTP 429 Too Many Requests",
        "DEADLINE_EXCEEDED",
        "UNAVAILABLE: backend overloaded",
    ):
      with self.subTest(text=text):
        self.assertTrue(
            scrape_tolerance.is_throttling_error(RuntimeError(text))
        )

  def test_rejects_permanent_errors(self):
    for text in (
        "Unsupported shape_type 'not_a_shape'",
        "INVALID_ARGUMENT: image too large",
    ):
      with self.subTest(text=text):
        self.assertFalse(
            scrape_tolerance.is_throttling_error(RuntimeError(text))
        )


class GatherTolerantTest(absltest.TestCase):

  def test_one_failure_does_not_lose_the_batch(self):
    # A single failing task yields None in its slot without destroying
    # sibling results in the batch.
    async def ok(value):
      return value

    def make(i):
      if i == 7:

        async def boom():
          raise RuntimeError("Request throttled at the client")

        return boom
      return lambda i=i: ok(i)

    funcs = [make(i) for i in range(20)]
    results, errors = asyncio.run(
        scrape_tolerance.gather_tolerant(funcs, bandwidth=5)
    )

    self.assertLen(results, 20)
    self.assertLen(errors, 1)
    self.assertIsNone(results[7])
    self.assertEqual(
        [r for r in results if r is not None], [i for i in range(20) if i != 7]
    )

  def test_preserves_positional_alignment(self):
    # Alignment is load-bearing: slot i must answer probe i, so a failure may
    # never compact the list.
    async def slow(value):
      await asyncio.sleep(0.01 * (value % 3))
      return value

    async def boom():
      raise ValueError("nope")

    funcs = [(lambda i=i: slow(i)) if i % 4 else boom for i in range(12)]
    results, errors = asyncio.run(
        scrape_tolerance.gather_tolerant(funcs, bandwidth=4)
    )

    self.assertLen(results, 12)
    self.assertLen(errors, 3)
    for i, res in enumerate(results):
      if i % 4:
        self.assertEqual(res, i)
      else:
        self.assertIsNone(res)

  def test_handles_sync_callables(self):
    results, errors = asyncio.run(
        scrape_tolerance.gather_tolerant(
            [lambda: 1, lambda: 1 / 0, lambda: 3], bandwidth=2
        )
    )
    self.assertEqual(results, [1, None, 3])
    self.assertLen(errors, 1)

  def test_empty_input(self):
    results, errors = asyncio.run(scrape_tolerance.gather_tolerant([]))
    self.assertEmpty(results)
    self.assertEmpty(errors)

  def test_all_failures_return_all_none_rather_than_raising(self):
    async def boom():
      raise RuntimeError("throttled")

    results, errors = asyncio.run(
        scrape_tolerance.gather_tolerant([boom, boom], bandwidth=2)
    )
    self.assertEqual(results, [None, None])
    self.assertLen(errors, 2)

  def test_logs_a_single_digest_not_one_line_per_failure(self):
    async def boom():
      raise RuntimeError("Request throttled at the client")

    lines = []
    asyncio.run(
        scrape_tolerance.gather_tolerant(
            [boom] * 50, bandwidth=10, log=lines.append
        )
    )
    self.assertLen(lines, 1)
    self.assertIn("50/50", lines[0])
    self.assertIn("throttled", lines[0])


class SummarizeErrorsTest(absltest.TestCase):

  def test_groups_by_kind(self):
    errors = [RuntimeError("throttled")] * 3 + [ValueError("bad shape")]
    summary = scrape_tolerance.summarize_errors(errors)
    self.assertIn("3x throttled", summary)
    self.assertIn("1x ValueError", summary)

  def test_empty(self):
    self.assertEqual(scrape_tolerance.summarize_errors([]), "")


class BackoffDelaysTest(absltest.TestCase):

  def test_grows_and_is_capped(self):
    # Sized to outlast a sustained throttle, not a transient blip.
    self.assertGreaterEqual(scrape_tolerance.backoff_delays(1), 12.0)
    self.assertGreater(
        scrape_tolerance.backoff_delays(3, factor=3.0),
        scrape_tolerance.backoff_delays(1, factor=3.0),
    )
    for attempt in range(1, 10):
      self.assertLessEqual(
          scrape_tolerance.backoff_delays(attempt, max_delay=300.0), 360.0
      )


class RoundGuardTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.sleeps: list[float] = []
    self.logs: list[str] = []

    async def fake_sleep(delay, result=None):
      self.sleeps.append(delay)
      return result

    self.enter_context(mock.patch.object(asyncio, "sleep", fake_sleep))
    self.enter_context(mock.patch.object(random, "uniform", return_value=1.0))

  def _run(self, guard, kinds):
    async def go():
      for kind in kinds:
        await guard.after_round(kind)

    asyncio.run(go())

  def test_raises_on_the_cap_th_consecutive_target_skip(self):
    guard = scrape_tolerance.RoundGuard(3, self.logs.append)
    self._run(guard, ["target_side", "short"])
    self.assertEqual(guard.consecutive, 2)
    # 5s then 10s; no wait before raising.
    self.assertEqual(self.sleeps, [5.0, 10.0])
    with self.assertRaisesRegex(
        scrape_tolerance.TargetOutageError, "3 consecutive"
    ):
      self._run(guard, ["target_side"])
    self.assertEqual(self.sleeps, [5.0, 10.0])
    self.assertIn("--max_consecutive_skips=3", self.logs[-1])

  def test_backoff_is_capped_at_60s(self):
    guard = scrape_tolerance.RoundGuard(100, self.logs.append)
    self._run(guard, ["target_side"] * 6)
    self.assertEqual(self.sleeps, [5.0, 10.0, 20.0, 40.0, 60.0, 60.0])

  def test_success_resets_and_auditor_side_is_ignored(self):
    guard = scrape_tolerance.RoundGuard(3, self.logs.append)
    self._run(
        guard,
        ["target_side", "target_side", None, "short", "auditor_side", "short"],
    )
    self.assertEqual(guard.consecutive, 2)
    with self.assertRaises(scrape_tolerance.TargetOutageError):
      self._run(guard, ["target_side"])

  def test_zero_disables_cap_and_wait(self):
    guard = scrape_tolerance.RoundGuard(0, self.logs.append)
    self._run(guard, ["target_side"] * 50)
    self.assertEqual(self.sleeps, [])
    self.assertEqual(self.logs, [])


if __name__ == "__main__":
  absltest.main()
