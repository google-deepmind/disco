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

"""Tests for visual symmetry prompt construction and condition inference.

The contrast tests verify that comparative claims of the form "A does worse than
B" resolve to two-arm contrast plans rather than collapsing to a single shape
family tested against the absolute 0.90 reference rate.
"""

from absl.testing import absltest
from absl.testing import parameterized

from disco.tasks.visual_symmetry import prompts


class InferShapeTypeTest(parameterized.TestCase):
  """Guards the single-condition inferrer against regression.

  `infer_contrast` was added alongside this function rather than replacing it,
  specifically so that single-condition absolute hypotheses resolve as expected.
  These cases pin that.
  """

  @parameterized.named_parameters(
      ("hard_family_named", "complex_chinese patterns fail", "complex_chinese"),
      (
          "hard_beats_easy_lookalike",
          "high-stroke chinese fails",
          "complex_chinese",
      ),
      ("easy_family_named", "chinese_char patterns succeed", "chinese_char"),
      ("greek", "greek letters are reflected reliably", "greek_letter"),
      ("fallback", "something entirely unrelated", "english_letter"),
  )
  def test_infer_shape_type(self, text, expected):
    self.assertEqual(prompts.infer_shape_type(text), expected)


class FindMentionsTest(absltest.TestCase):

  def test_orders_by_position(self):
    self.assertEqual(
        prompts.find_shape_mentions(
            "greek_letter is easier than complex_chinese"
        ),
        ["greek_letter", "complex_chinese"],
    )

  def test_hard_family_consumes_its_easy_lookalike(self):
    # "complex_chinese" contains "chinese", which is `chinese_char`'s keyword.
    # Reporting both would fabricate a contrast between a family and itself.
    self.assertEqual(
        prompts.find_shape_mentions("complex_chinese completions"),
        ["complex_chinese"],
    )

  def test_english_word_plus_is_not_a_shape(self):
    # `infer_shape_type` matches bare substrings, so "plus" in ordinary prose
    # selects the `plus` family. Harmless when picking one family of many;
    # here it would split the batch into two arms for no reason.
    self.assertEqual(
        prompts.find_shape_mentions("circles plus noisy backgrounds"),
        ["circle"],
    )

  def test_finds_background_colours_in_order(self):
    self.assertEqual(
        prompts.find_background_mentions(
            "accuracy on navy is lower than on white"
        ),
        ["navy", "white"],
    )


class InferContrastTest(absltest.TestCase):

  def test_shape_axis_contrast(self):
    plan = prompts.infer_contrast(
        "complex_chinese fails more often than chinese_char"
    )
    self.assertTrue(plan.is_contrastive)
    self.assertEqual(plan.axis, "shape")
    self.assertEqual(plan.condition_a.shape_type, "complex_chinese")
    self.assertIsNotNone(plan.condition_b)
    assert plan.condition_b is not None
    self.assertEqual(plan.condition_b.shape_type, "chinese_char")
    # Background is not what varies, so it must be pinned identically.
    self.assertEqual(plan.condition_a.bg_color, plan.condition_b.bg_color)

  def test_background_axis_contrast(self):
    plan = prompts.infer_contrast(
        "circle probes score worse on navy than on white"
    )
    self.assertTrue(plan.is_contrastive)
    self.assertEqual(plan.axis, "background")
    assert plan.condition_b is not None
    self.assertEqual(plan.condition_a.bg_color, "navy")
    self.assertEqual(plan.condition_b.bg_color, "white")
    # Shape is not what varies.
    self.assertEqual(plan.condition_a.shape_type, plan.condition_b.shape_type)

  def test_conjunction_is_not_a_contrast(self):
    # Two families named, but the claim is about both jointly, not about a
    # difference. Splitting it would halve the sample and test the wrong thing.
    plan = prompts.infer_contrast(
        "circle and square probes both succeed above 90 percent"
    )
    self.assertFalse(plan.is_contrastive)

  def test_comparison_against_reference_rate_is_not_a_contrast(self):
    # "than" is present, but only one condition is named: this is the ordinary
    # absolute claim the 0.90 null exists for.
    plan = prompts.infer_contrast(
        "complex_chinese succeeds less often than 90 percent of the time"
    )
    self.assertFalse(plan.is_contrastive)
    self.assertEqual(plan.condition_a.shape_type, "complex_chinese")

  def test_seed1_style_absolute_claim_is_unchanged(self):
    text = (
        "The model completes filigree patterns with accuracy below 90 percent"
    )
    plan = prompts.infer_contrast(text)
    self.assertFalse(plan.is_contrastive)
    self.assertEqual(
        plan.condition_a.shape_type, prompts.infer_shape_type(text)
    )

  def test_absolute_plan_keeps_a_single_named_background(self):
    plan = prompts.infer_contrast("maze probes on navy backgrounds fail")
    self.assertFalse(plan.is_contrastive)
    self.assertEqual(plan.condition_a.bg_color, "navy")

  def test_conditions_property(self):
    absolute = prompts.infer_contrast("maze probes fail")
    self.assertLen(absolute.conditions, 1)
    contrast = prompts.infer_contrast("maze fails more than circle")
    self.assertLen(contrast.conditions, 2)


class ProbeConditionTest(absltest.TestCase):

  def test_label_omits_default_background(self):
    self.assertEqual(prompts.ProbeCondition("maze").label, "maze")

  def test_label_includes_non_default_background(self):
    self.assertEqual(prompts.ProbeCondition("maze", "navy").label, "maze@navy")

  def test_probe_args(self):
    self.assertEqual(
        prompts.ProbeCondition("maze", "navy").probe_args(50),
        {"shape_type": "maze", "bg_color": "navy", "n_probes": 50},
    )


class ContrastivePromptTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.plan = prompts.infer_contrast(
        "complex_chinese fails more often than chinese_char"
    )

  def test_briefing_names_both_arms_and_their_sizes(self):
    text = prompts.contrastive_execution_briefing(
        "h", self.plan, n_a=50, n_b=50
    )
    self.assertIn("complex_chinese", text)
    self.assertIn("chinese_char", text)
    self.assertIn("50 probes", text)

  def test_briefing_rejects_absolute_plan(self):
    absolute = prompts.infer_contrast("maze probes fail")
    with self.assertRaises(ValueError):
      prompts.contrastive_execution_briefing("h", absolute, 50, 50)

  def test_stats_prompt_supplies_counts_and_forbids_the_absolute_null(self):
    text = prompts.contrastive_stats_prompt(
        "h", self.plan, n_success_a=20, n_a=50, n_success_b=45, n_b=50
    )
    self.assertIn("`n_successes_a`=20", text)
    self.assertIn("`n_obs_a`=50", text)
    self.assertIn("`n_successes_b`=45", text)
    self.assertIn("run_proportion_z_test_2_sample", text)
    self.assertIn("Do NOT use the 90 percent reference rate", text)

  def test_stats_prompt_rejects_absolute_plan(self):
    absolute = prompts.infer_contrast("maze probes fail")
    with self.assertRaises(ValueError):
      prompts.contrastive_stats_prompt("h", absolute, 1, 2, 1, 2)


if __name__ == "__main__":
  absltest.main()
