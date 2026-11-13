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

"""Tests for statistical test plan models, tool routing, and tie policies."""

from absl.testing import absltest
from absl.testing import parameterized
from disco.stats import plan


class PlanTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("mcnemar", "McNemar test", plan.StatsTool.MCNEMAR),
      (
          "paired_continuous",
          "Paired continuous test",
          plan.StatsTool.PAIRED_MEANS,
      ),
      ("paired_means", "Paired means t-test", plan.StatsTool.PAIRED_MEANS),
      ("paired_student", "Paired Student diff", plan.StatsTool.PAIRED_MEANS),
      (
          "two_sample_prop",
          "Two-sample proportion test",
          plan.StatsTool.PROPORTION_Z_TEST_2_SAMPLE,
      ),
      (
          "2_sample_z",
          "2-sample z-test",
          plan.StatsTool.PROPORTION_Z_TEST_2_SAMPLE,
      ),
      (
          "fisher",
          "Fisher exact test",
          plan.StatsTool.PROPORTION_Z_TEST_2_SAMPLE,
      ),
      (
          "continuous_means",
          "Continuous means test",
          plan.StatsTool.CONTINUOUS_MEANS,
      ),
      ("student_t", "Student t-test", plan.StatsTool.CONTINUOUS_MEANS),
      ("proportion_z", "Proportion z-test", plan.StatsTool.PROPORTION_Z_TEST),
      ("binomial", "Binomial test", plan.StatsTool.PROPORTION_Z_TEST),
      ("unrecognized", "Arbitrary non-statistical evaluation", None),
      ("empty", "", None),
  )
  def test_match_test_to_tool_priority(self, test_name, expected_tool):
    self.assertEqual(plan.match_test_to_tool(test_name), expected_tool)

  @parameterized.named_parameters(
      ("prop_z", "Proportion z-test", True, False),
      ("two_sample_prop", "Two-sample proportion test", True, False),
      ("mcnemar", "McNemar test", True, False),
      ("continuous_means", "Continuous means test", False, True),
      ("paired_means", "Paired continuous test", False, True),
      ("unknown", "Unknown test", False, False),
      ("none", None, False, False),
  )
  def test_is_discrete_and_continuous(
      self, test_name, expected_discrete, expected_continuous
  ):
    self.assertEqual(plan.is_discrete_test(test_name), expected_discrete)
    self.assertEqual(plan.is_continuous_test(test_name), expected_continuous)

  @parameterized.named_parameters(
      (
          "explicit_paired",
          plan.PreRegisteredTestPlan(
              test_type="Proportion test",
              target_metric="acc",
              is_paired=True,
          ),
          True,
          "pair",
      ),
      (
          "mcnemar_implied_paired",
          plan.PreRegisteredTestPlan(
              test_type="McNemar test",
              target_metric="acc",
              is_paired=False,
          ),
          True,
          "pair",
      ),
      (
          "paired_means_implied",
          plan.PreRegisteredTestPlan(
              test_type="Paired continuous test",
              target_metric="diff",
              is_paired=False,
          ),
          True,
          "pair",
      ),
      (
          "unpaired_prop",
          plan.PreRegisteredTestPlan(
              test_type="Proportion z-test",
              target_metric="acc",
              is_paired=False,
          ),
          False,
          "response",
      ),
      (
          "unpaired_continuous",
          plan.PreRegisteredTestPlan(
              test_type="Continuous means test",
              target_metric="score",
              is_paired=False,
          ),
          False,
          "response",
      ),
  )
  def test_implies_paired(self, test_plan, expected_paired, expected_unit):
    self.assertEqual(test_plan.implies_paired, expected_paired)
    self.assertEqual(test_plan.unit_of_analysis, expected_unit)

  @parameterized.named_parameters(
      (
          "not_paired",
          plan.PreRegisteredTestPlan(
              test_type="Proportion z-test",
              target_metric="acc",
              is_paired=False,
          ),
          plan.TiePolicy.NOT_APPLICABLE,
      ),
      (
          "paired_means_retain",
          plan.PreRegisteredTestPlan(
              test_type="Paired continuous test",
              target_metric="diff",
              is_paired=True,
          ),
          plan.TiePolicy.RETAIN,
      ),
      (
          "continuous_means_retain",
          plan.PreRegisteredTestPlan(
              test_type="Continuous means test",
              target_metric="diff",
              is_paired=True,
          ),
          plan.TiePolicy.RETAIN,
      ),
      (
          "mcnemar_drop_even_with_skewed_null",
          plan.PreRegisteredTestPlan(
              test_type="McNemar test",
              target_metric="acc",
              null_baseline=0.1,
              is_paired=True,
          ),
          plan.TiePolicy.DROP,
      ),
      (
          "paired_prop_symmetry_null_drop",
          plan.PreRegisteredTestPlan(
              test_type="Proportion z-test",
              target_metric="acc",
              null_baseline=0.5,
              is_paired=True,
          ),
          plan.TiePolicy.DROP,
      ),
      (
          "paired_prop_skewed_null_count_failure",
          plan.PreRegisteredTestPlan(
              test_type="Proportion z-test",
              target_metric="acc",
              null_baseline=0.2,
              is_paired=True,
          ),
          plan.TiePolicy.COUNT_AS_FAILURE,
      ),
      (
          "paired_2sample_skewed_null_count_failure",
          plan.PreRegisteredTestPlan(
              test_type="Two-sample proportion test",
              target_metric="acc",
              null_baseline=0.8,
              is_paired=True,
          ),
          plan.TiePolicy.COUNT_AS_FAILURE,
      ),
      (
          "unrecognized_paired_retain",
          plan.PreRegisteredTestPlan(
              test_type="Custom paired metric",
              target_metric="metric",
              is_paired=True,
          ),
          plan.TiePolicy.RETAIN,
      ),
  )
  def test_tie_policy_matrix(self, test_plan, expected_policy):
    self.assertEqual(test_plan.tie_policy, expected_policy)

  def test_guard_p_value_direction(self):
    logs = []
    # 1. Larger alternative, positive delta -> p unchanged
    res1 = plan.StatsResult(
        test_type="Binomial test",
        p_value=0.03,
        group_a_mean=0.7,
        alternative="larger",
        null_proportion=0.5,
    )
    p_adj1 = plan.guard_p_value_direction(res1, logs.append)
    self.assertEqual(p_adj1, 0.03)

    # 2. Larger alternative, negative delta, p < 0.5 -> p inverted to 1 - p
    res2 = plan.StatsResult(
        test_type="Binomial test",
        p_value=0.04,
        group_a_mean=0.3,
        alternative="larger",
        null_proportion=0.5,
    )
    p_adj2 = plan.guard_p_value_direction(res2, logs.append)
    self.assertAlmostEqual(p_adj2, 0.96)
    self.assertTrue(any("Direction Guard" in l for l in logs))

    # 3. Smaller alternative, positive delta, p < 0.5 -> p inverted
    res3 = plan.StatsResult(
        test_type="Binomial test",
        p_value=0.02,
        group_a_mean=0.8,
        alternative="smaller",
        null_proportion=0.5,
    )
    p_adj3 = plan.guard_p_value_direction(res3, logs.append)
    self.assertAlmostEqual(p_adj3, 0.98)

    # 4. Two-sided -> p unchanged
    res4 = plan.StatsResult(
        test_type="Binomial test",
        p_value=0.04,
        group_a_mean=0.3,
        alternative="two-sided",
        null_proportion=0.5,
    )
    self.assertEqual(plan.guard_p_value_direction(res4, logs.append), 0.04)

    # 5. Guard disabled -> p unchanged
    self.assertEqual(
        plan.guard_p_value_direction(
            res2, logs.append, use_direction_guard=False
        ),
        0.04,
    )

    # 6. Two-sample test with group_b_mean
    res6 = plan.StatsResult(
        test_type="Two-sample test",
        p_value=0.01,
        group_a_mean=0.4,
        group_b_mean=0.6,
        alternative="larger",
    )
    self.assertAlmostEqual(
        plan.guard_p_value_direction(res6, logs.append), 0.99
    )


if __name__ == "__main__":
  absltest.main()
