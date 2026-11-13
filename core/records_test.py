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

"""Unit tests for EValueEntry and HypothesisRecord data structures and formatting."""

from absl.testing import absltest

from disco.core import records
from disco.stats import plan as stats_plan


class RecordsTest(absltest.TestCase):

  def test_e_value_entry(self):
    entry = records.EValueEntry("sub_1", 2.5, cumulative_queries=10)
    self.assertEqual(entry.sub_hypothesis, "sub_1")
    self.assertEqual(entry.e_value, 2.5)
    self.assertEqual(entry.cumulative_queries, 10)
    # Test unpacking
    s, e = entry
    self.assertEqual((s, e), ("sub_1", 2.5))
    # Test indexing
    self.assertEqual(entry[0], "sub_1")
    self.assertEqual(entry[1], 2.5)
    self.assertEqual(entry[2], 10)

  def test_hypothesis_record_failures_formatting(self):
    rec = records.HypothesisRecord(text="Test H")
    self.assertEqual(rec.format_failures(), "")

    rec.failures = [{
        "judgment": "FAILED",
        "prompt": "Say hello",
        "response": "Goodbye",
        "reasoning": "Wrong word",
    }]
    formatted = rec.format_failures()
    self.assertIn("Concrete Failure Patterns discovered:", formatted)
    self.assertIn("Say hello", formatted)
    self.assertIn("Goodbye", formatted)
    self.assertIn("Wrong word", formatted)

  def test_hypothesis_record_with_test_plan(self):
    plan = stats_plan.PreRegisteredTestPlan(
        test_type="Binomial test",
        target_metric="rate",
        null_baseline=0.5,
    )
    rec = records.HypothesisRecord(text="H with plan", test_plan=plan)
    self.assertEqual(rec.test_plan, plan)

  def test_disco_result(self):
    result = records.DiscoResult(
        all_records={"h1": records.HypothesisRecord(text="H1")},
        rejected_hypotheses={"h1"},
        final_e_values={"h1": 10.0},
        boosted_e_values={"h1": 20.0},
        boost_factors={"h1": 2.0},
        probe_counts={"h1": 5},
        terminal_statuses={"h1": "H"},
        total_hypotheses_k=1,
        rejection_threshold=5.0,
        nominal_alpha_used=0.05,
    )
    self.assertIn("h1", result.rejected_hypotheses)
    self.assertEqual(result.boosted_e_values["h1"], 20.0)

  def test_log_hypothesis_details(self):
    logs = []
    rec = records.HypothesisRecord(
        text="H1",
        queries=15,
        e_values=[records.EValueEntry("sub1", 3.0)],
    )
    records.log_hypothesis_details(
        "h1", rec, agg_e=3.0, log_fn=logs.append, unboosted_e=1.5
    )
    full_log = "\n".join(logs)
    self.assertIn("ID: h1 (Hypothesis: H1)", full_log)
    self.assertIn("Final Aggregated E-value: 3.0000", full_log)
    self.assertIn("Unboosted E-value: 1.5000", full_log)
    self.assertIn("Total Queries Used: 15", full_log)
    self.assertIn("- Sub-H: sub1 (E-val: 3.0000)", full_log)


if __name__ == "__main__":
  absltest.main()
