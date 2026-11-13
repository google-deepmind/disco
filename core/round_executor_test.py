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

"""Unit tests for the pure parts of `round_executor`.

The full round (call order, prompts, budget flow) is pinned end to end by the
smoke goldens in testing/disco_runner_smoke_test.py.
"""

import asyncio
import types
from typing import Any, cast
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from disco.core import checkpoint as checkpoint_lib
from disco.core import config as config_lib
from disco.core import records as records_lib
from disco.core import round_executor
from disco.core import task_executor
from disco.stats import plan as stats_plan


def _agent(**attrs: Any) -> Any:
  defaults = dict(
      last_scrape_status="OK",
      last_responses=[],
      all_responses=[],
      symmetry_output_images=[],
  )
  defaults.update(attrs)
  return types.SimpleNamespace(**defaults)


def _task(
    ex: round_executor.RoundExecutor, symmetry: bool
) -> task_executor.TaskExecutor:
  if symmetry:
    return task_executor.SymmetryTaskExecutor(ex.config, ex.log_fn)
  return task_executor.TextTaskExecutor(ex.config, ex.log_fn)


def _executor(budget: int = 10) -> round_executor.RoundExecutor:
  state = checkpoint_lib.RunState(remaining_budget=budget)
  state.hypothesis_groups["h1"] = records_lib.HypothesisRecord(text="H")
  config = config_lib.DiscoConfig(
      target_model="t",
      seed_instruction="s",
      budget=budget,
      alpha=0.1,
      kappa=0.5,
  )
  return round_executor.RoundExecutor(
      config, cast(Any, None), state, lambda *a, **k: None
  )


class ChargeBudgetTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          "text_all_responses",
          False,
          dict(all_responses=[1, 2, 3], last_responses=[1]),
          3,
      ),
      ("text_last_responses", False, dict(last_responses=[1, 2]), 2),
      (
          "sym_last_responses",
          True,
          dict(last_responses=[1, 2], symmetry_output_images=[1, 2, 3]),
          2,
      ),
      ("sym_images", True, dict(symmetry_output_images=[1, 2, 3]), 3),
  )
  def test_charges_counted_queries(self, symmetry, attrs, expected):
    ex = _executor()
    charged = ex.charge_budget(_agent(**attrs), "h1", _task(ex, symmetry), 7)
    self.assertEqual(charged, expected)
    self.assertEqual(ex.state.remaining_budget, 10 - expected)
    self.assertEqual(ex.state.hypothesis_groups["h1"].queries, expected)

  @parameterized.named_parameters(
      (
          "text_failed_status",
          False,
          dict(last_scrape_status="FAILED", last_responses=[1]),
      ),
      ("text_no_responses", False, dict(all_responses=[1])),
      (
          "sym_failed_status",
          True,
          dict(last_scrape_status="FAILED", last_responses=[1]),
      ),
      ("sym_nothing", True, dict(all_responses=[1])),
  )
  def test_scrape_failure_charges_nothing(self, symmetry, attrs):
    ex = _executor()
    self.assertIsNone(
        ex.charge_budget(_agent(**attrs), "h1", _task(ex, symmetry), 7)
    )
    self.assertEqual(ex.state.remaining_budget, 10)
    self.assertEqual(ex.state.hypothesis_groups["h1"].queries, 0)


class ShortRoundReasonTest(parameterized.TestCase):

  @parameterized.named_parameters(
      # 8 of 10 planned responses: 0.8 < 0.9.
      (
          "text_short",
          False,
          False,
          dict(all_responses=[1] * 8, last_responses=[1]),
          10,
          True,
      ),
      (
          "text_enough",
          False,
          False,
          dict(all_responses=[1] * 9, last_responses=[1]),
          10,
          False,
      ),
      # 4 complete pairs of the 5 planned.
      (
          "text_paired_short",
          False,
          True,
          dict(
              all_responses=[1] * 9,
              last_responses=[1],
              last_response_indices=list(range(9)),
          ),
          10,
          True,
      ),
      # A failed scrape is left to `charge_budget`.
      (
          "text_failed",
          False,
          False,
          dict(last_scrape_status="FAILED", last_responses=[1]),
          10,
          False,
      ),
      # Symmetry rounds: scored completions (missing images excluded) count.
      ("symmetry_short", True, False, dict(last_responses=[1] * 8), 10, True),
      (
          "symmetry_enough",
          True,
          False,
          dict(last_responses=[1] * 9),
          10,
          False,
      ),
  )
  def test_short_round_reason(
      self, symmetry, is_paired, attrs, q_count, expect_short
  ):
    ex = _executor()
    reason = ex.short_round_reason(
        _agent(**attrs), _task(ex, symmetry), q_count, is_paired
    )
    self.assertEqual(bool(reason), expect_short)
    if expect_short:
      self.assertStartsWith(reason, "INSUFFICIENT_RESPONSES: ")

  def test_symmetry_contrast_round_checks_each_arm(self):
    ex = _executor()
    task = _task(ex, True)
    agent = _agent(last_responses=[1] * 18)
    # 18/20 pooled would pass; arm B alone is 8/10.
    task.last_arms = [("A", 10, 10), ("B", 8, 10)]
    reason = ex.short_round_reason(agent, task, 20, False)
    self.assertStartsWith(reason, "INSUFFICIENT_RESPONSES: arm 'B' 8/10")
    task.last_arms = [("A", 9, 10), ("B", 9, 10)]
    self.assertEqual(ex.short_round_reason(agent, task, 20, False), "")

  def test_disabled_by_zero_fraction(self):
    ex = _executor()
    ex.config.min_response_fraction = 0.0
    self.assertEqual(
        ex.short_round_reason(
            _agent(all_responses=[1], last_responses=[1]),
            _task(ex, False),
            10,
            False,
        ),
        "",
    )


class ResolveIsPairedTest(parameterized.TestCase):

  @parameterized.parameters(
      (False, None, False),
      (True, None, True),
      (False, False, False),
      (False, True, True),
      (True, False, True),
  )
  def test_resolve(self, config_paired, plan_paired, expected):
    config = cast(Any, types.SimpleNamespace(is_paired=config_paired))
    if plan_paired is None:
      plan = None
    else:
      plan = cast(
          stats_plan.PreRegisteredTestPlan,
          types.SimpleNamespace(implies_paired=plan_paired),
      )
    self.assertEqual(round_executor.resolve_is_paired(config, plan), expected)


class PreregisterAfterResumeTest(absltest.TestCase):

  def test_restored_plan_is_reused_without_asking_the_agent(self):
    plan = stats_plan.PreRegisteredTestPlan(
        test_type="Binomial test", target_metric="m", null_baseline=0.3
    )
    ex = _executor()
    ex.state.hypothesis_groups["h1"].test_plan = plan
    payload = checkpoint_lib.to_checkpoint(
        ex.state, runner_title="t", budget=10
    )
    resumed = _executor()
    checkpoint_lib.restore_checkpoint(resumed.state, payload, budget_delta=0)

    sender = mock.MagicMock()
    sender.send_and_extract = mock.AsyncMock(return_value=None)
    got = asyncio.run(
        resumed.preregister(
            sender,
            cast(Any, None),
            "H",
            resumed.state.hypothesis_groups["h1"],
        )
    )
    self.assertEqual(got, plan)
    sender.send_and_extract.assert_not_awaited()


if __name__ == "__main__":
  absltest.main()
