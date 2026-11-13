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

"""Tests for core/checkpoint.

`_legacy_checkpoint` is a frozen copy of the payload DiscoRunner wrote
before the move (base `_save_checkpoint` plus `_get_extra_checkpoint_state`),
so the on-disk JSON can be checked byte for byte.
"""

import json
import os
from typing import Any
from unittest import mock

from absl.testing import absltest
import numpy as np

from disco.core import checkpoint
from disco.core import records as records_lib
from disco.stats import plan as stats_plan


def _legacy_checkpoint(
    state: checkpoint.RunState, runner_title: str, budget: int
) -> dict[str, Any]:
  serialized_groups = {}
  for h_id, record in state.hypothesis_groups.items():
    serialized_groups[h_id] = {
        "text": record.text,
        "e_values": [
            {
                "sub_hypothesis": ev.sub_hypothesis,
                "e_value": ev.e_value,
                "cumulative_queries": ev.cumulative_queries,
            }
            for ev in record.e_values
        ],
        "sub_hypotheses": list(record.sub_hypotheses),
        "failures": list(record.failures),
        "failure_reflections": list(record.failure_reflections),
        "executed_queries": list(record.executed_queries),
        "queries": record.queries,
    }
  payload = {
      "current_round": state.current_round,
      "remaining_budget": state.remaining_budget,
      "total_scrape_rounds": state.total_scrape_rounds,
      "hypothesis_groups": serialized_groups,
      "runner_title": runner_title,
      "budget": budget,
  }
  payload["extra"] = {
      "running_e_values": dict(state.running_e_values),
      "final_e_values": dict(state.final_e_values),
      "probe_counts": dict(state.probe_counts),
      "terminal_statuses": dict(state.terminal_statuses),
      "bin_trajectories": {
          k: list(v) for k, v in state.bin_trajectories.items()
      },
      "prev_bins": dict(state.prev_bins),
      "transition_scores": {
          k: list(v) for k, v in state.transition_scores.items()
      },
      "active_hypotheses": list(state.active_hypotheses),
      "k_total": state.k_total,
      # skip_counts was added last, after every pre-existing key.
      "skip_counts": dict(state.skip_counts),
  }
  return payload


def _populated_state() -> checkpoint.RunState:
  return checkpoint.RunState(
      remaining_budget=40,
      current_round=4,
      total_scrape_rounds=7,
      hypothesis_groups={
          "h_1": records_lib.HypothesisRecord(
              text="Answers are too short.",
              e_values=[
                  records_lib.EValueEntry("s1", 1.7, 10),
                  records_lib.EValueEntry("s2", 0.4, 20),
              ],
              sub_hypotheses=["s1", "s2"],
              failures=[{"prompt": "p", "response": "r"}],
              failure_reflections=["too terse"],
              executed_queries={"q1"},
              queries=20,
          ),
          "h_2": records_lib.HypothesisRecord(text="Uses lists."),
      },
      running_e_values={"h_1": 0.68, "h_2": 1.0},
      final_e_values={"h_1": 0.68},
      probe_counts={"h_1": 2, "h_2": 0},
      terminal_statuses={"h_1": "L"},
      bin_trajectories={"h_1": [1, 2, 1]},
      prev_bins={"h_1": 1, "h_2": 1},
      transition_scores={"h_1": [0.3, 0.1]},
      active_hypotheses={"h_2"},
      k_total=2,
  )


class SerializationTest(absltest.TestCase):

  def test_json_matches_legacy_byte_for_byte(self):
    for state in (checkpoint.RunState(remaining_budget=10), _populated_state()):
      want = json.dumps(_legacy_checkpoint(state, "DISCO", 60), indent=2)
      got = json.dumps(
          checkpoint.to_checkpoint(state, runner_title="DISCO", budget=60),
          indent=2,
      )
      self.assertEqual(got, want)

  def test_round_trip(self):
    state = _populated_state()
    payload = json.loads(
        json.dumps(checkpoint.to_checkpoint(state, runner_title="t", budget=60))
    )
    restored = checkpoint.RunState(remaining_budget=0)
    checkpoint.restore_checkpoint(restored, payload, budget_delta=5)
    state.remaining_budget += 5
    self.assertEqual(restored, state)

  def test_record_defaults_for_missing_keys(self):
    rec = checkpoint.deserialize_record(
        {"text": "t", "e_values": [{"sub_hypothesis": "s", "e_value": 2.0}]}
    )
    self.assertEqual(
        rec,
        records_lib.HypothesisRecord(
            text="t", e_values=[records_lib.EValueEntry("s", 2.0, 0)]
        ),
    )
    self.assertIsNone(rec.test_plan)

  def test_test_plan_round_trip(self):
    plan = stats_plan.PreRegisteredTestPlan(
        test_type="continuous_means_test",
        target_metric="word count",
        null_baseline=20.0,
        alternative="smaller",
        reasoning="short answers",
        is_paired=False,
    )
    state = _populated_state()
    state.hypothesis_groups["h_1"].test_plan = plan
    payload = json.loads(
        json.dumps(checkpoint.to_checkpoint(state, runner_title="t", budget=60))
    )
    self.assertEqual(list(payload["hypothesis_groups"]["h_1"])[-1], "test_plan")
    self.assertNotIn("test_plan", payload["hypothesis_groups"]["h_2"])

    restored = checkpoint.RunState(remaining_budget=0)
    checkpoint.restore_checkpoint(restored, payload, budget_delta=0)
    self.assertEqual(restored.hypothesis_groups["h_1"].test_plan, plan)
    self.assertIsNone(restored.hypothesis_groups["h_2"].test_plan)

  def test_empty_extra_keeps_existing_state(self):
    state = _populated_state()
    checkpoint.restore_extra(state, {})
    self.assertEqual(state, _populated_state())

  def test_unseeded_state_writes_no_rng_key(self):
    payload = checkpoint.to_checkpoint(
        _populated_state(), runner_title="t", budget=60
    )
    self.assertNotIn("pval_rng_state", payload)

  def test_rng_state_round_trip_continues_the_stream(self):
    rng = np.random.default_rng(7)
    rng.random(5)
    state = _populated_state()
    state.pval_rng_state = dict(rng.bit_generator.state)
    payload = json.loads(
        json.dumps(checkpoint.to_checkpoint(state, runner_title="t", budget=60))
    )
    self.assertEqual(list(payload)[-1], "pval_rng_state")

    restored = checkpoint.RunState(remaining_budget=0)
    checkpoint.restore_checkpoint(restored, payload, budget_delta=0)
    restored_state = restored.pval_rng_state
    assert restored_state is not None
    resumed = np.random.default_rng()
    resumed.bit_generator.state = restored_state
    np.testing.assert_array_equal(resumed.random(5), rng.random(5))

  def test_checkpoint_without_rng_key_restores_none(self):
    state = checkpoint.RunState(remaining_budget=0, pval_rng_state={"x": 1})
    payload = checkpoint.to_checkpoint(
        _populated_state(), runner_title="t", budget=60
    )
    checkpoint.restore_checkpoint(state, payload, budget_delta=0)
    self.assertIsNone(state.pval_rng_state)

  def test_skip_counts_round_trip(self):
    state = _populated_state()
    state.skip_counts["target_side"] = 3
    state.skip_counts["auditor_side"] = 1
    payload = json.loads(
        json.dumps(checkpoint.to_checkpoint(state, runner_title="t", budget=60))
    )
    self.assertEqual(
        payload["extra"]["skip_counts"],
        {"target_side": 3, "short": 0, "auditor_side": 1},
    )
    restored = checkpoint.RunState(remaining_budget=0)
    checkpoint.restore_checkpoint(restored, payload, budget_delta=0)
    self.assertEqual(restored.skip_counts, state.skip_counts)

  def test_checkpoint_without_skip_counts_restores_zeros(self):
    payload = checkpoint.to_checkpoint(
        _populated_state(), runner_title="t", budget=60
    )
    del payload["extra"]["skip_counts"]
    state = checkpoint.RunState(remaining_budget=0)
    state.skip_counts["short"] = 9
    checkpoint.restore_checkpoint(state, payload, budget_delta=0)
    self.assertEqual(
        state.skip_counts, {"target_side": 0, "short": 0, "auditor_side": 0}
    )


class CheckpointStoreTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.path = os.path.join(
        self.create_tempdir().full_path, "exp", checkpoint.CHECKPOINT_FILENAME
    )
    self.store = checkpoint.CheckpointStore(self.path)
    self.logs: list[str] = []

  def test_save_writes_file_and_no_temp(self):
    self.store.save(_populated_state(), runner_title="DISCO", budget=60)
    self.assertTrue(os.path.exists(self.path))
    # No temp file is left beside the checkpoint.
    self.assertEqual(
        os.listdir(os.path.dirname(self.path)), [os.path.basename(self.path)]
    )
    with open(self.path) as f:
      self.assertEqual(
          f.read(),
          json.dumps(
              _legacy_checkpoint(_populated_state(), "DISCO", 60), indent=2
          ),
      )

  def test_failed_save_keeps_previous_checkpoint_and_logs(self):
    self.store.save(_populated_state(), runner_title="DISCO", budget=60)
    with open(self.path) as f:
      before = f.read()
    with (
        mock.patch.object(
            checkpoint, "write_json_atomic", side_effect=OSError("disk full")
        ),
        mock.patch.object(checkpoint.logging, "warning") as warning,
    ):
      self.store.save(
          checkpoint.RunState(remaining_budget=1),
          runner_title="DISCO",
          budget=60,
      )
    with open(self.path) as f:
      self.assertEqual(f.read(), before)
    warning.assert_called_once()
    self.assertIn("disk full", str(warning.call_args))

  def test_missing_file(self):
    state = checkpoint.RunState(remaining_budget=60)
    self.assertFalse(self.store.load_into(state, budget=60))
    self.assertEqual(state, checkpoint.RunState(remaining_budget=60))

  def test_corrupt_file_starts_fresh(self):
    os.makedirs(os.path.dirname(self.path))
    with open(self.path, "w") as f:
      f.write("{not json")
    state = checkpoint.RunState(remaining_budget=60)
    self.assertFalse(
        self.store.load_into(state, budget=60, log_fn=self.logs.append)
    )
    self.assertLen(self.logs, 1)
    self.assertIn("failed to load checkpoint", self.logs[0])

  def test_smaller_budget_starts_fresh(self):
    self.store.save(_populated_state(), runner_title="t", budget=60)
    state = checkpoint.RunState(remaining_budget=50)
    self.assertFalse(
        self.store.load_into(state, budget=50, log_fn=self.logs.append)
    )
    self.assertEqual(state, checkpoint.RunState(remaining_budget=50))
    self.assertIn("smaller than checkpoint budget (60)", self.logs[0])

  def test_larger_budget_adds_delta(self):
    self.store.save(_populated_state(), runner_title="t", budget=60)
    state = checkpoint.RunState(remaining_budget=75)
    self.assertTrue(self.store.load_into(state, budget=75))
    want = _populated_state()
    want.remaining_budget = 40 + 15
    self.assertEqual(state, want)
    self.assertEqual(self.logs, [])


if __name__ == "__main__":
  absltest.main()
