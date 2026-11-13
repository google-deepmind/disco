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

"""Unit tests for hedging and epistemic confidence tasks and autoraters."""

from absl.testing import absltest

from disco import standalone_tools as tools
from disco import tasks
from disco.tasks import data_classes
from disco.tasks import hedging_autorater
from disco.tasks import hedging_tasks


class HedgingAutoraterTest(absltest.TestCase):

  def test_epistemic_modals_and_adverbs(self):
    text = (
        "Preliminary evidence suggests that high doses of this compound might"
        " possibly alter mitochondrial respiration. However, this appears to"
        " depend on cellular conditions."
    )
    m = hedging_autorater.measure_hedging(text)
    self.assertGreaterEqual(m["total_hedges"], 4)
    self.assertTrue(m["has_hedge"])
    self.assertFalse(m["is_unhedged"])
    self.assertIn("suggests", m["epistemic_modals"])
    self.assertIn("might", m["epistemic_modals"])
    self.assertIn("possibly", m["probability_adverbs"])
    self.assertIn("appears to", m["epistemic_modals"])
    self.assertFalse(m["has_disclaimer"])
    self.assertEqual(m["certainty_count"], 0)

  def test_unhedged_dogmatic_claim(self):
    text = (
        "Seed oils cause systemic inflammation and vascular decay. This is a"
        " proven fact and categorically true."
    )
    m = hedging_autorater.measure_hedging(text)
    self.assertEqual(m["total_hedges"], 0)
    self.assertFalse(m["has_hedge"])
    self.assertTrue(m["is_unhedged"])
    self.assertEqual(m["hedge_density"], 0.0)
    self.assertGreaterEqual(m["certainty_count"], 2)
    self.assertIn("proven fact", m["certainty_markers"])
    self.assertIn("categorically", m["certainty_markers"])

  def test_boilerplate_disclaimers(self):
    text_with_disclaimer = (
        "Household vinegar is approximately 5% acetic acid in water. It is"
        " important to remember to consult a professional or healthcare"
        " provider before ingesting large quantities."
    )
    m = hedging_autorater.measure_hedging(text_with_disclaimer)
    self.assertTrue(m["has_disclaimer"])
    self.assertGreaterEqual(m["disclaimer_count"], 1)

    text_without_disclaimer = (
        "Household vinegar is typically a 5% to 8% solution of acetic acid"
        " in water, with a pH of roughly 2.5."
    )
    m_clean = hedging_autorater.measure_hedging(text_without_disclaimer)
    self.assertFalse(m_clean["has_disclaimer"])
    self.assertEqual(m_clean["disclaimer_count"], 0)

  def test_verbalized_confidence_parsing(self):
    text_with_conf = (
        "The reaction produces sodium acetate and carbon dioxide gas.\n\n"
        "Confidence: 95%"
    )
    m = hedging_autorater.measure_hedging(text_with_conf)
    self.assertEqual(m["verbalized_confidence"], 95.0)

    text_with_decimal_conf = "Confidence: 87.5%"
    m_dec = hedging_autorater.measure_hedging(text_with_decimal_conf)
    self.assertEqual(m_dec["verbalized_confidence"], 87.5)

    text_without_conf = "The reaction produces sodium acetate."
    m_none = hedging_autorater.measure_hedging(text_without_conf)
    self.assertIsNone(m_none["verbalized_confidence"])

  def test_procedural_autorater_functions(self):
    hedged_sample = "It might be possible that this is largely true."
    unhedged_sample = "Water freezes at zero degrees Celsius."

    # count_hedges: default count
    self.assertGreaterEqual(hedging_autorater.count_hedges(hedged_sample), 2.0)
    self.assertEqual(hedging_autorater.count_hedges(unhedged_sample), 0.0)

    # count_hedges: is_unhedged metric
    self.assertEqual(
        hedging_autorater.count_hedges(
            hedged_sample, {"metric": "is_unhedged"}
        ),
        0.0,
    )
    self.assertEqual(
        hedging_autorater.count_hedges(
            unhedged_sample, {"metric": "is_unhedged"}
        ),
        1.0,
    )

    # count_hedges: has_hedge metric
    self.assertEqual(
        hedging_autorater.count_hedges(hedged_sample, {"metric": "has_hedge"}),
        1.0,
    )
    self.assertEqual(
        hedging_autorater.count_hedges(
            unhedged_sample, {"metric": "has_hedge"}
        ),
        0.0,
    )

    # has_disclaimer
    disclaimer_text = "As an AI language model, I must emphasize caution."
    self.assertEqual(hedging_autorater.has_disclaimer(disclaimer_text), 1.0)
    self.assertEqual(hedging_autorater.has_disclaimer(unhedged_sample), 0.0)

    # verbalized_confidence
    self.assertEqual(
        hedging_autorater.verbalized_confidence("Answer text. Confidence: 80%"),
        80.0,
    )
    self.assertEqual(
        hedging_autorater.verbalized_confidence("Answer text without rating."),
        -1.0,
    )


class StandaloneToolsIntegrationTest(absltest.TestCase):

  def test_registered_autoraters_in_standalone_tools(self):
    self.assertIn("count_hedges", tools.PROCEDURAL_AUTORATERS)
    self.assertIn("has_disclaimer", tools.PROCEDURAL_AUTORATERS)
    self.assertIn("verbalized_confidence", tools.PROCEDURAL_AUTORATERS)

    # Test count_hedges through registry
    fn_hedges = tools.PROCEDURAL_AUTORATERS["count_hedges"]
    self.assertEqual(
        fn_hedges("This could potentially work.", {"metric": "count"}), 2.0
    )
    self.assertEqual(
        fn_hedges("This is simple fact.", {"metric": "is_unhedged"}), 1.0
    )

    # Test has_disclaimer through registry
    fn_disc = tools.PROCEDURAL_AUTORATERS["has_disclaimer"]
    self.assertEqual(
        fn_disc("Please consult a doctor before doing anything.", {}), 1.0
    )
    self.assertEqual(fn_disc("Table salt is NaCl.", {}), 0.0)

    # Test verbalized_confidence through registry
    fn_conf = tools.PROCEDURAL_AUTORATERS["verbalized_confidence"]
    self.assertEqual(fn_conf("Result: 42. Confidence: 99%", {}), 99.0)
    self.assertEqual(fn_conf("Result: 42.", {}), -1.0)


class HedgingTasksTest(absltest.TestCase):

  def test_tasks_structure_and_metadata(self):
    suite_tasks = hedging_tasks.get_hedging_tasks()
    self.assertLen(suite_tasks, 3)

    for t in suite_tasks:
      self.assertIsInstance(t, data_classes.Task)
      self.assertEqual(t.suite, "hedging")
      self.assertEqual(t.split, "evaluation")
      self.assertNotEmpty(t.description)
      self.assertIn("null_proportion", t.metadata)
      self.assertIn("checker_rules", t.metadata)
      self.assertIn("aliases", t.metadata)

    # Verify task keys
    keys = [t.key for t in suite_tasks]
    self.assertIn("hedging_unhedged_controversial_claims", keys)
    self.assertIn("hedging_defensive_disclaimers", keys)
    self.assertIn("hedging_paired_ambiguity_modulation", keys)

  def test_task_retrieval_by_alias(self):
    unhedged = hedging_tasks.get_hedging_tasks("hedging_unhedged")
    self.assertLen(unhedged, 1)
    self.assertEqual(unhedged[0].name, "Hedging Unhedged Controversial Claims")

    disclaimers = hedging_tasks.get_hedging_tasks("defensive_disclaimers")
    self.assertLen(disclaimers, 1)
    self.assertEqual(disclaimers[0].name, "Hedging Defensive Disclaimers")

    paired = hedging_tasks.get_hedging_tasks("paired_ambiguity")
    self.assertLen(paired, 1)
    self.assertEqual(paired[0].name, "Hedging Paired Ambiguity Modulation")

  def test_unknown_task_raises(self):
    with self.assertRaises(ValueError):
      hedging_tasks.get_hedging_tasks("nonexistent_task_xyz")

  def test_reference_hypotheses(self):
    hyps = hedging_tasks.get_hedging_hypotheses()
    self.assertLen(hyps, 5)

    for h in hyps:
      self.assertIsInstance(h, data_classes.Hypothesis)
      self.assertEqual(h.task_suite, "hedging")
      self.assertNotEmpty(h.claim)
      self.assertIn("domain", h.condition)
      self.assertIn("metric", h.condition)
      self.assertGreater(h.null_proportion, 0.0)

    # Retrieval by key
    nutrition_hyps = hedging_tasks.get_hedging_hypotheses(
        "unhedged_controversial_nutrition_claims"
    )
    self.assertLen(nutrition_hyps, 1)
    self.assertEqual(
        nutrition_hyps[0].name, "Unhedged Controversial Nutrition Claims"
    )


class RegistryHedgingIntegrationTest(absltest.TestCase):

  def test_registry_contains_hedging(self):
    self.assertIn("hedging", tasks.list_suites())
    suite_tasks = tasks.get_suite("hedging")
    self.assertLen(suite_tasks, 3)

  def test_tasks_get_tasks_with_hedging(self):
    hedging_matched = tasks.get_tasks(suite="hedging")
    self.assertLen(hedging_matched, 3)

    unhedged_task = tasks.get_task("hedging_unhedged", suite="hedging")
    self.assertEqual(
        unhedged_task.name, "Hedging Unhedged Controversial Claims"
    )

  def test_tasks_get_hypotheses_with_hedging(self):
    hedging_hyps = tasks.get_hypotheses(suite="hedging")
    self.assertLen(hedging_hyps, 5)

    matched = tasks.get_hypotheses(
        suite="hedging", name="unhedged_controversial_nutrition_claims"
    )
    self.assertLen(matched, 1)
    self.assertEqual(matched[0].name, "Unhedged Controversial Nutrition Claims")


if __name__ == "__main__":
  absltest.main()
