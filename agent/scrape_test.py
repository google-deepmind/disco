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

"""Tests for agent/scrape.py and StandaloneAgent scraping.

`ScrapeWithBackfillTest` covers the shared loop directly.
`AgentScrapeCharacterizationTest` tests the text path
(`StandaloneAgent.scrape_responses`) and the symmetry path
(`_tool_generate_and_send_symmetry_probes`) — gather calls, bandwidth,
jitter, sleeps, log lines, and raise vs. FAILED.
"""

import asyncio
import functools
import inspect
import io
import json
import random
from typing import Any
from unittest import mock

from absl.testing import absltest
import numpy as np
import PIL.Image

from disco import standalone_agent
from disco.agent import scrape
from disco.agent import scrape_tolerance
from disco.models import content as content_api

_THROTTLED = "RESOURCE_EXHAUSTED: request throttled"


async def _sequential_to_threads(tasks, **unused_kwargs):
  """Deterministic stand-in for `content_api.to_threads_with_throttle`."""
  results = []
  for task in tasks:
    result = task()
    if inspect.isawaitable(result):
      result = await result
    results.append(result)
  return results


class _Harness(absltest.TestCase):
  """Patches concurrency, sleep and randomness; records gathers and sleeps."""

  def setUp(self):
    super().setUp()
    self.gather_calls: list[tuple[str | None, int | None, int]] = []
    self.sleeps: list[float] = []
    self.gather_failures = 0
    real_gather = scrape_tolerance.gather_tolerant

    async def recording_gather(funcs, *, desc=None, bandwidth=None, log=None):
      self.gather_calls.append((desc, bandwidth, len(funcs)))
      if self.gather_failures:
        self.gather_failures -= 1
        raise RuntimeError("boom")
      return await real_gather(funcs, desc=desc, bandwidth=bandwidth, log=log)

    async def recording_sleep(delay, result=None):
      self.sleeps.append(delay)
      return result

    self.enter_context(
        mock.patch.object(
            scrape_tolerance.content_api,
            "to_threads_with_throttle",
            side_effect=_sequential_to_threads,
        )
    )
    self.enter_context(
        mock.patch.object(scrape_tolerance, "gather_tolerant", recording_gather)
    )
    self.enter_context(mock.patch.object(asyncio, "sleep", recording_sleep))
    self.enter_context(mock.patch.object(random, "uniform", return_value=1.0))

  def backoff(self, attempt: int) -> float:
    return scrape_tolerance.backoff_delays(attempt)


class ScrapeWithBackfillTest(_Harness):

  def _run(self, script, n, **kwargs):
    """Runs scrape_with_backfill where request i's k-th call is script(i, k)."""
    calls: dict[int, int] = {}

    async def issue(i):
      calls[i] = calls.get(i, 0) + 1
      result = script(i, calls[i])
      if isinstance(result, BaseException):
        raise result
      return result

    logs: list[str] = []
    kwargs.setdefault("is_usable", lambda r: bool(r.strip()))
    kwargs.setdefault("bandwidth", 5)
    kwargs.setdefault("labels", scrape.TEXT_LABELS)
    outcome = asyncio.run(
        scrape.scrape_with_backfill(
            lambda i: functools.partial(issue, i), n, log=logs.append, **kwargs
        )
    )
    return outcome, calls, logs

  def test_backfill_refills_slots_in_place(self):
    def script(i, k):
      if i == 1 and k == 1:
        return "  "
      if i == 3 and k == 1:
        return ValueError("bad request")
      return f"r{i}"

    outcome, calls, _ = self._run(script, 4)
    self.assertEqual(outcome.slots, ["r0", "r1", "r2", "r3"])
    self.assertEqual(outcome.n_requested, 4)
    self.assertEqual(outcome.n_received, 4)
    self.assertEqual(outcome.n_missing_initially, 2)
    self.assertEqual(outcome.n_backfill_attempts, 1)
    self.assertEqual(outcome.n_backfilled, 2)
    self.assertEqual(outcome.missing_indices, [])
    self.assertEqual(calls, {0: 1, 1: 2, 2: 1, 3: 2})
    self.assertLen(outcome.errors, 1)

  def test_runs_at_most_max_backfill_waves(self):
    outcome, calls, _ = self._run(
        lambda i, k: ValueError("bad") if i == 0 else f"r{i}", 3
    )
    self.assertEqual(outcome.n_backfill_attempts, scrape.MAX_BACKFILL)
    self.assertEqual(outcome.missing_indices, [0])
    self.assertEqual(outcome.valid_indices, [1, 2])
    self.assertEqual(outcome.n_backfilled, 0)
    self.assertEqual(calls[0], 1 + scrape.MAX_BACKFILL)

  def test_sleeps_only_before_waves_that_follow_throttling(self):
    errors_by_call = {
        1: RuntimeError(_THROTTLED),
        2: ValueError("bad"),
        3: RuntimeError(_THROTTLED),
    }
    outcome, _, _ = self._run(lambda i, k: errors_by_call.get(k, "ok"), 1)
    self.assertEqual(outcome.slots, ["ok"])
    self.assertEqual(outcome.n_backfill_attempts, 3)
    # Before wave 1 (first pass throttled) and wave 3 (wave 2 throttled).
    self.assertEqual(self.sleeps, [self.backoff(1), self.backoff(3)])

  def test_chunked_first_pass_descs_and_bandwidth(self):
    self._run(lambda i, k: f"r{i}", 5, chunk_size=2, bandwidth=3)
    self.assertEqual(
        self.gather_calls,
        [
            ("Scraping chunk 1/3", 2, 2),
            ("Scraping chunk 2/3", 2, 2),
            ("Scraping chunk 3/3", 1, 1),
        ],
    )

  def test_unchunked_first_pass_is_one_batch(self):
    self._run(
        lambda i, k: f"r{i}", 12, bandwidth=10, labels=scrape.SYMMETRY_LABELS
    )
    self.assertEqual(self.gather_calls, [("Sending symmetry probes", 10, 12)])

  def test_chunk_retried_when_gather_raises(self):
    self.gather_failures = 2
    outcome, _, logs = self._run(lambda i, k: f"r{i}", 1, chunk_size=20)
    self.assertEqual(outcome.slots, ["r0"])
    self.assertEqual(self.sleeps, [2.0, 4.0])
    self.assertEqual(
        logs,
        [
            (
                "  [Scrape Retry] Chunk 1/1 failed (attempt 1/4), retrying in"
                " 2.0s: boom"
            ),
            (
                "  [Scrape Retry] Chunk 1/1 failed (attempt 2/4), retrying in"
                " 4.0s: boom"
            ),
        ],
    )

  def test_chunk_failure_calls_hook_then_reraises(self):
    self.gather_failures = 4
    hook = mock.Mock()
    with self.assertRaisesRegex(RuntimeError, "boom"):
      self._run(lambda i, k: "r", 1, chunk_size=20, on_chunk_failure=hook)
    hook.assert_called_once()
    self.assertEqual(self.sleeps, [2.0, 4.0, 8.0])

  def test_unchunked_gather_failure_is_not_retried(self):
    self.gather_failures = 1
    with self.assertRaisesRegex(RuntimeError, "boom"):
      self._run(lambda i, k: "r", 1, labels=scrape.SYMMETRY_LABELS)
    self.assertLen(self.gather_calls, 1)
    self.assertEqual(self.sleeps, [])

  def test_no_top_up_once_min_fraction_is_met(self):
    outcome, calls, _ = self._run(
        lambda i, k: ValueError("bad") if i == 0 else f"r{i}",
        3,
        min_fraction=0.5,
    )
    self.assertEqual(outcome.n_top_up_attempts, 0)
    self.assertEqual(outcome.n_backfill_attempts, scrape.MAX_BACKFILL)
    self.assertEqual(calls[0], 1 + scrape.MAX_BACKFILL)

  def test_top_up_runs_until_min_fraction_is_met(self):
    # Slot 0 recovers on its 5th call (the first top-up wave); slot 1 never
    # does, but 9/10 already meets 0.9.
    def script(i, k):
      if (i == 0 and k < 5) or i == 1:
        return ValueError("bad")
      return f"r{i}"

    outcome, calls, logs = self._run(script, 10, min_fraction=0.9)
    self.assertEqual(outcome.n_top_up_attempts, 1)
    self.assertEqual(outcome.n_backfill_attempts, scrape.MAX_BACKFILL + 1)
    self.assertEqual(outcome.missing_indices, [1])
    self.assertEqual(outcome.n_received, 9)
    self.assertEqual(calls[0], 5)
    self.assertEqual(calls[1], 5)
    # The regular waves were not throttled, so only the top-up wave waited.
    self.assertEqual(self.sleeps, [self.backoff(1)])
    self.assertEqual(
        self.gather_calls[-1], ("Scraping backfill top-up 1/5", 2, 2)
    )
    self.assertIn(
        (
            "  [Scrape Top-up] 8/10 response(s) received (< 0.9 required);"
            f" waiting {self.backoff(1):.0f}s before top-up attempt 1/5 for 2"
            " prompt(s)..."
        ),
        logs,
    )

  def test_top_up_stops_after_max_top_up_waves(self):
    outcome, calls, _ = self._run(
        lambda i, k: ValueError("bad") if i == 0 else f"r{i}",
        2,
        min_fraction=0.9,
    )
    self.assertEqual(outcome.n_top_up_attempts, scrape.MAX_TOP_UP)
    self.assertEqual(
        outcome.n_backfill_attempts, scrape.MAX_BACKFILL + scrape.MAX_TOP_UP
    )
    self.assertEqual(calls[0], 1 + scrape.MAX_BACKFILL + scrape.MAX_TOP_UP)
    self.assertEqual(
        self.sleeps,
        [self.backoff(k) for k in range(1, scrape.MAX_TOP_UP + 1)],
    )

  def test_no_top_up_when_nothing_arrived(self):
    # Every slot is still empty after the regular waves: an outage, so no
    # top-up wave runs and nothing waits.
    outcome, calls, logs = self._run(
        lambda i, k: ValueError("down"), 3, min_fraction=0.9
    )
    self.assertEqual(outcome.n_top_up_attempts, 0)
    self.assertEqual(outcome.n_backfill_attempts, scrape.MAX_BACKFILL)
    self.assertEqual(outcome.missing_indices, [0, 1, 2])
    self.assertEqual(calls, {i: 1 + scrape.MAX_BACKFILL for i in range(3)})
    self.assertEqual(self.sleeps, [])
    self.assertEqual(
        logs[-1],
        "  [Scrape Top-up] Skipped: 0/3 response(s) received after 3"
        " backfill attempt(s); treating the target as down rather than"
        " topping up.",
    )

  def test_top_up_still_runs_after_one_response_arrives(self):
    # A single usable response makes it a partial batch, not an outage.
    outcome, _, _ = self._run(
        lambda i, k: f"r{i}" if (i, k) == (0, 4) else ValueError("bad"),
        3,
        min_fraction=0.9,
    )
    self.assertEqual(outcome.n_received, 1)
    self.assertEqual(outcome.n_top_up_attempts, scrape.MAX_TOP_UP)

  def test_top_up_stops_before_waiting_past_the_deadline(self):
    # Clock reads: scrape start, then before each top-up wait.
    clock = iter([0.0, 0.0, 50.0, 60.0]).__next__
    outcome, _, logs = self._run(
        lambda i, k: ValueError("bad") if i == 0 else f"r{i}",
        2,
        min_fraction=0.9,
        top_up_deadline_s=100.0,
        clock=clock,
    )
    # 0 + 15 and 50 + 45 fit in 100s; 60 + 135 does not.
    self.assertEqual(outcome.n_top_up_attempts, 2)
    self.assertEqual(self.sleeps, [self.backoff(1), self.backoff(2)])
    self.assertEqual(
        logs[-1],
        (
            "  [Scrape Top-up] Stopping: 1/2 response(s) received (< 0.9"
            f" required), but waiting {self.backoff(3):.0f}s more would pass"
            " the 100s deadline (60s elapsed)."
        ),
    )

  def test_blank_slot_logging_follows_labels(self):
    _, _, text_logs = self._run(
        lambda i, k: "  " if k == 1 else "ok", 1, max_backfill=1
    )
    _, _, sym_logs = self._run(
        lambda i, k: "  " if k == 1 else "ok",
        1,
        max_backfill=1,
        labels=scrape.SYMMETRY_LABELS,
    )
    self.assertEqual(
        text_logs,
        [
            (
                "  [Scrape Warning] Response 1 was empty/blank. Will attempt to"
                " refill this slot."
            ),
            (
                "  [Scrape Replenish] Deficit of 1 response(s). Attempt 1/1 for"
                " 1 prompt(s)..."
            ),
        ],
    )
    self.assertEqual(
        sym_logs,
        [
            (
                "  [Scrape Replenish] Deficit of 1 symmetry response(s)."
                " Attempt 1/1 for 1 probe(s)..."
            ),
        ],
    )


def _text_response(text: str) -> content_api.Response:
  return content_api.Response(
      content=content_api.Content([content_api.ContentChunk(text)]),
      probe=mock.MagicMock(spec=content_api.Probe),
  )


def _image_response() -> content_api.Response:
  buf = io.BytesIO()
  PIL.Image.fromarray(np.full((20, 20, 3), 255, dtype=np.uint8)).save(
      buf, format="PNG"
  )
  return content_api.Response(
      content=content_api.Content(
          [content_api.ContentChunk(buf.getvalue(), mimetype="image/png")]
      ),
      probe=mock.MagicMock(spec=content_api.Probe),
  )


class AgentScrapeCharacterizationTest(_Harness):

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.object(standalone_agent.genai_client, "get_genai_client")
    )
    self.enter_context(
        mock.patch.object(standalone_agent, "_get_or_create_candidate")
    )
    self.logs: list[str] = []
    self.candidate = mock.MagicMock()

  def _agent(
      self, target_model: str, **kwargs: Any
  ) -> standalone_agent.StandaloneAgent:
    agent = standalone_agent.StandaloneAgent(
        agent_model="models/gemini-3.8-flash",
        target_model=target_model,
        output_dir=self.create_tempdir().full_path,
        log_fn=self.logs.append,
        scrape_bandwidth=3,
        **kwargs,
    )
    agent._candidate_cls = self.candidate
    return agent

  def _assert_outcome(
      self,
      agent,
      *,
      n_requested,
      n_received,
      n_missing,
      n_attempts,
      n_backfilled,
  ):
    outcome = agent.last_scrape_outcome
    self.assertIsNotNone(outcome)
    assert outcome is not None
    self.assertEqual(
        (
            outcome.n_requested,
            outcome.n_received,
            outcome.n_missing_initially,
            outcome.n_backfill_attempts,
            outcome.n_backfilled,
        ),
        (n_requested, n_received, n_missing, n_attempts, n_backfilled),
    )
    self.assertEqual(agent.scrape_history, [outcome])

  def _scrape_logs(self) -> list[str]:
    # `[Scrape Partial]` comes from gather_tolerant itself, which is unchanged.
    return [
        l for l in self.logs if "[Scrape" in l and "[Scrape Partial]" not in l
    ]

  def _respond_kwargs(self) -> list[dict[str, Any]]:
    return [c.kwargs for c in self.candidate.respond.call_args_list]

  def _script_text(self, script):
    calls: dict[str, int] = {}

    def respond(content, *args, **kwargs):
      del args, kwargs
      text = content.as_text(ignore_unsupported_types=True)
      calls[text] = calls.get(text, 0) + 1
      result = script(text, calls[text])
      if isinstance(result, BaseException):
        raise result
      return _text_response(result)

    self.candidate.respond.side_effect = respond

  def test_text_backfill_and_partial_failure(self):
    def script(text, k):
      if text == "p1" and k == 1:
        return "   "
      if text == "p2" and k == 1:
        return RuntimeError(_THROTTLED)
      if text == "p4":
        return ValueError("bad request")
      return f"ok:{text}"

    self._script_text(script)
    agent = self._agent("gemini-3.8-flash")
    result = asyncio.run(
        agent.scrape_responses([f"p{i}" for i in range(5)], chunk_size=2)
    )

    self.assertEqual(
        [r.content.as_text(ignore_unsupported_types=True) for r in result],
        ["ok:p0", "ok:p1", "ok:p2", "ok:p3"],
    )
    self.assertEqual(agent.last_scrape_status, "OK")
    self.assertEqual(agent.last_valid_response_count, 4)
    self.assertEqual(agent.last_response_indices, [0, 1, 2, 3])
    self.assertEqual(agent.last_prompt_count, 5)
    self._assert_outcome(
        agent,
        n_requested=5,
        n_received=4,
        n_missing=3,
        n_attempts=3,
        n_backfilled=2,
    )
    self.assertEqual(
        self.gather_calls,
        [
            ("Scraping chunk 1/3", 2, 2),
            ("Scraping chunk 2/3", 2, 2),
            ("Scraping chunk 3/3", 1, 1),
            ("Scraping backfill 1/3", 3, 3),
            ("Scraping backfill 2/3", 1, 1),
            ("Scraping backfill 3/3", 1, 1),
        ],
    )
    self.assertEqual(self.sleeps, [self.backoff(1)])
    self.assertEqual(self._respond_kwargs(), [{"jitter": 1.0}] * 10)
    self.assertEqual(
        self._scrape_logs(),
        [
            (
                "  [Scrape Warning] Response 2 was empty/blank. Will attempt to"
                " refill this slot."
            ),
            (
                "  [Scrape Replenish] Load shedding detected; waiting"
                f" {self.backoff(1):.0f}s before backfill attempt 1/3."
            ),
            (
                "  [Scrape Replenish] Deficit of 3 response(s). Attempt 1/3 for"
                " 3 prompt(s)..."
            ),
            (
                "  [Scrape Replenish] Deficit of 1 response(s). Attempt 2/3 for"
                " 1 prompt(s)..."
            ),
            (
                "  [Scrape Replenish] Deficit of 1 response(s). Attempt 3/3 for"
                " 1 prompt(s)..."
            ),
            (
                "  [Scrape Warning] 1 slot(s) still unfilled after 3 backfill"
                " attempt(s). Prompt indices: [4]"
            ),
        ],
    )

  def test_text_tops_up_to_min_response_fraction(self):
    # p0 recovers on its 6th call (the second top-up wave).
    self._script_text(
        lambda text, k: "  " if text == "p0" and k < 6 else f"ok:{text}"
    )
    agent = self._agent("gemini-3.8-flash", min_response_fraction=0.9)
    result = asyncio.run(agent.scrape_responses(["p0", "p1"]))
    self.assertLen(result, 2)
    self.assertEqual(agent.last_response_indices, [0, 1])
    outcome = agent.last_scrape_outcome
    assert outcome is not None
    self.assertEqual(outcome.n_top_up_attempts, 2)
    self.assertEqual(outcome.n_backfill_attempts, 5)
    self.assertEqual(self.sleeps, [self.backoff(1), self.backoff(2)])

  def test_text_without_min_response_fraction_does_not_top_up(self):
    self._script_text(lambda text, k: "  " if text == "p0" else f"ok:{text}")
    agent = self._agent("gemini-3.8-flash")
    asyncio.run(agent.scrape_responses(["p0", "p1"]))
    outcome = agent.last_scrape_outcome
    assert outcome is not None
    self.assertEqual(outcome.n_top_up_attempts, 0)
    self.assertEqual(self.sleeps, [])
    self.assertEqual(
        self._scrape_logs()[-1],
        "  [Scrape Warning] 1 slot(s) still unfilled after 3 backfill"
        " attempt(s). Prompt indices: [0]",
    )

  def test_text_all_empty_with_min_response_fraction_fails_without_top_up(
      self,
  ):
    self._script_text(lambda text, k: "  ")
    agent = self._agent("gemini-3.8-flash", min_response_fraction=0.9)
    with self.assertRaisesRegex(RuntimeError, "Failed to obtain valid"):
      asyncio.run(agent.scrape_responses(["a", "b"]))
    self.assertEqual(agent.last_scrape_status, "FAILED")
    self.assertEqual(self.sleeps, [])
    self.assertEqual(
        self.gather_calls,
        [("Scraping chunk 1/1", 2, 2)]
        + [(f"Scraping backfill {k}/3", 2, 2) for k in (1, 2, 3)],
    )

  def test_text_all_empty_raises(self):
    self._script_text(lambda text, k: "  ")
    agent = self._agent("gemini-3.8-flash")
    with self.assertRaisesRegex(RuntimeError, "Failed to obtain valid"):
      asyncio.run(agent.scrape_responses(["a", "b"]))
    self.assertEqual(agent.last_scrape_status, "FAILED")
    self.assertEqual(agent.last_valid_response_count, 0)
    self.assertEqual(agent.last_response_indices, [])
    self.assertEqual(agent.last_prompt_count, 2)
    self.assertEqual(
        self.gather_calls,
        [("Scraping chunk 1/1", 2, 2)]
        + [(f"Scraping backfill {k}/3", 2, 2) for k in (1, 2, 3)],
    )
    self.assertEqual(
        self._scrape_logs(),
        [
            (
                "  [Scrape Warning] Response 1 was empty/blank. Will attempt to"
                " refill this slot."
            ),
            (
                "  [Scrape Warning] Response 2 was empty/blank. Will attempt to"
                " refill this slot."
            ),
        ]
        + [
            f"  [Scrape Replenish] Deficit of 2 response(s). Attempt {k}/3 for"
            " 2 prompt(s)..."
            for k in (1, 2, 3)
        ]
        + ["  [Scrape Error] All scraped responses were empty or failed."],
    )

  def test_text_chunk_retry_then_success(self):
    self._script_text(lambda text, k: f"ok:{text}")
    self.gather_failures = 2
    agent = self._agent("gemini-3.8-flash")
    asyncio.run(agent.scrape_responses(["a"]))
    self.assertEqual(agent.last_scrape_status, "OK")
    self.assertEqual(self.gather_calls, [("Scraping chunk 1/1", 1, 1)] * 3)
    self.assertEqual(self.sleeps, [2.0, 4.0])
    self.assertEqual(
        self._scrape_logs(),
        [
            (
                "  [Scrape Retry] Chunk 1/1 failed (attempt 1/4), retrying in"
                " 2.0s: boom"
            ),
            (
                "  [Scrape Retry] Chunk 1/1 failed (attempt 2/4), retrying in"
                " 4.0s: boom"
            ),
        ],
    )

  def test_text_chunk_exhausts_retries_and_raises(self):
    self._script_text(lambda text, k: f"ok:{text}")
    self.gather_failures = 4
    agent = self._agent("gemini-3.8-flash")
    with self.assertRaisesRegex(RuntimeError, "boom"):
      asyncio.run(agent.scrape_responses(["a"]))
    self.assertEqual(agent.last_scrape_status, "FAILED")
    self.assertEqual(agent.last_valid_response_count, 0)
    self.assertEqual(agent.last_prompt_count, 1)
    self.assertEqual(self.sleeps, [2.0, 4.0, 8.0])
    self.assertEqual(
        self._scrape_logs()[-1],
        "  [Scrape Error] Chunk 1/1 failed after 4 attempts: boom",
    )

  def _script_symmetry(self, script):
    calls = {"n": 0}

    def respond(unused_content, *args, **kwargs):
      del args, kwargs
      k = calls["n"]
      calls["n"] += 1
      result = script(k)
      if isinstance(result, BaseException):
        raise result
      return result

    self.candidate.respond.side_effect = respond

  def test_symmetry_backfill_ignores_scrape_bandwidth_and_jitter(self):
    image = _image_response()
    no_image = _text_response("refused")

    def script(k):
      if k == 0:
        return no_image
      if k == 12:
        return RuntimeError(_THROTTLED)
      return image

    self._script_symmetry(script)
    agent = self._agent("gemini-3.1-flash-image")
    result = json.loads(
        asyncio.run(
            agent._tool_generate_and_send_symmetry_probes({
                "shape_type": "square",
                "n_probes": 4,
                "samples_per_probe": 3,
            })
        )
    )
    self.assertEqual(result["n_valid_outputs"], 12)
    self._assert_outcome(
        agent,
        n_requested=12,
        n_received=12,
        n_missing=1,
        n_attempts=2,
        n_backfilled=1,
    )
    self.assertEqual(agent.last_scrape_status, "OK")
    self.assertEqual(agent.last_prompt_count, 12)
    self.assertEqual(
        self.gather_calls,
        [
            ("Sending symmetry probes", 10, 12),
            ("Symmetry backfill 1/3", 1, 1),
            ("Symmetry backfill 2/3", 1, 1),
        ],
    )
    self.assertEqual(self.sleeps, [self.backoff(2)])
    self.assertEqual(self._respond_kwargs(), [{}] * 14)
    self.assertEqual(
        self._scrape_logs(),
        [
            (
                "  [Scrape Replenish] Deficit of 1 symmetry response(s)."
                " Attempt 1/3 for 1 probe(s)..."
            ),
            (
                "  [Scrape Replenish] Load shedding detected; waiting"
                f" {self.backoff(2):.0f}s before backfill attempt 2/3."
            ),
            (
                "  [Scrape Replenish] Deficit of 1 symmetry response(s)."
                " Attempt 2/3 for 1 probe(s)..."
            ),
            "  [Scrape Replenish] Recovered 1/1 response(s) over 2 attempt(s).",
        ],
    )

  def test_symmetry_all_empty_reports_failed_without_raising(self):
    self._script_symmetry(lambda k: _text_response("refused"))
    agent = self._agent("gemini-3.1-flash-image")
    asyncio.run(
        agent._tool_generate_and_send_symmetry_probes({
            "shape_type": "square",
            "n_probes": 1,
            "samples_per_probe": 1,
        })
    )
    self.assertEqual(agent.last_scrape_status, "FAILED")
    self.assertEqual(agent.last_valid_response_count, 0)
    self.assertEqual(agent.last_prompt_count, 1)
    self.assertEqual(
        self.gather_calls,
        [("Sending symmetry probes", 1, 1)]
        + [(f"Symmetry backfill {k}/3", 1, 1) for k in (1, 2, 3)],
    )
    self.assertEqual(self.sleeps, [])
    self.assertEqual(
        self._scrape_logs(),
        [
            "  [Scrape Replenish] Deficit of 1 symmetry response(s). Attempt"
            f" {k}/3 for 1 probe(s)..."
            for k in (1, 2, 3)
        ]
        + [
            "  [Scrape Replenish] Recovered 0/1 response(s) over 3 attempt(s).",
            (
                "  [Scrape Warning] 1 symmetry probe slot(s) still unfilled"
                " after 3 backfill attempt(s). They are recorded as unanswered,"
                " NOT as failed completions. Probe indices: [0]"
            ),
        ],
    )


if __name__ == "__main__":
  absltest.main()
