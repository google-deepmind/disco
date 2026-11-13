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

"""Unit tests for statistical utilities, confidence intervals, and p-value randomization."""

from absl.testing import absltest
import numpy as np
import pydantic

from disco import stats_utils
from disco.parsing import schemas
from disco.stats import plan as stats_plan


class StatsUtilsTest(absltest.TestCase):

  def test_extract_null_proportion_percent(self):
    seed = (
        "Investigating response length. Null hypothesis is that English recipes"
        " are shorter than French recipes 10% as the null proportion."
    )
    self.assertAlmostEqual(
        stats_utils.extract_null_proportion_from_seed(seed), 0.10
    )

  def test_extract_null_proportion_word_percent(self):
    seed = "Testing against 25 percent as the null baseline."
    self.assertAlmostEqual(
        stats_utils.extract_null_proportion_from_seed(seed), 0.25
    )

  def test_extract_null_proportion_decimal(self):
    seed = "Tested against 0.10 as the null hypothesis."
    self.assertAlmostEqual(
        stats_utils.extract_null_proportion_from_seed(seed), 0.10
    )

  def test_extract_null_proportion_default(self):
    seed = "Investigating refusal behavior across categories."
    self.assertAlmostEqual(
        stats_utils.extract_null_proportion_from_seed(seed, default=0.5), 0.5
    )

  def test_maybe_randomize_p_value_continuous_unchanged(self):
    rng = np.random.default_rng(42)
    res = stats_plan.StatsResult(
        test_type="two-sample t-test (continuous means)",
        p_value=0.03,
        group_a_obs=20,
        group_a_mean=100.0,
        group_b_obs=20,
        group_b_mean=80.0,
    )
    p_rand = stats_utils.maybe_randomize_p_value(0.03, res, rng)
    self.assertEqual(p_rand, 0.03)

  def test_maybe_randomize_p_value_binomial(self):
    rng = np.random.default_rng(42)
    res = stats_plan.StatsResult(
        test_type="1-sample binomial test",
        p_value=0.05,
        group_a_obs=20,
        group_a_mean=0.4,
        null_proportion=0.1,
    )
    p_rand = stats_utils.maybe_randomize_p_value(0.05, res, rng)
    self.assertGreater(p_rand, 0.0)
    self.assertLessEqual(p_rand, 1.0)
    # Randomized p-value should not equal raw p-value almost surely
    self.assertNotEqual(p_rand, 0.05)

  def test_maybe_randomize_p_value_fisher(self):
    rng = np.random.default_rng(42)
    res = stats_plan.StatsResult(
        test_type="two-sample fisher exact test",
        p_value=0.04,
        group_a_obs=15,
        group_a_mean=0.8,
        group_b_obs=15,
        group_b_mean=0.2,
    )
    p_rand = stats_utils.maybe_randomize_p_value(0.04, res, rng)
    self.assertGreater(p_rand, 0.0)
    self.assertLessEqual(p_rand, 1.0)

  def test_maybe_randomize_p_value_mcnemar(self):
    res = stats_plan.StatsResult(
        test_type="paired mcnemar test",
        p_value=0.02,
        group_a_obs=10,
        group_a_mean=0.8,
    )
    p_rand = stats_utils.maybe_randomize_p_value(
        0.02, res, np.random.default_rng(42)
    )
    self.assertGreater(p_rand, 0.0)
    self.assertLessEqual(p_rand, 1.0)
    self.assertNotEqual(p_rand, 0.02)

    # Different seeds must give different draws, which is the whole point of
    # Tocher randomization.
    draws = {
        stats_utils.maybe_randomize_p_value(
            0.02, res, np.random.default_rng(seed)
        )
        for seed in range(8)
    }
    self.assertGreater(len(draws), 1)

  def test_maybe_randomize_p_value_paired_rate_is_randomized(self):
    # A paired rate test is discrete and should be randomized.
    res = stats_plan.StatsResult(
        test_type="paired proportion test",
        p_value=0.05,
        group_a_obs=20,
        group_a_mean=0.7,
        null_proportion=0.5,
    )
    p_rand = stats_utils.maybe_randomize_p_value(
        0.05, res, np.random.default_rng(0)
    )
    self.assertNotEqual(p_rand, 0.05)

  def test_maybe_randomize_p_value_paired_means_is_not_randomized(self):
    # A paired continuous test already has a continuous p-value.
    res = stats_plan.StatsResult(
        test_type="Paired continuous test",
        p_value=0.05,
        group_a_obs=20,
        group_a_mean=1.5,
    )
    for seed in range(4):
      self.assertEqual(
          stats_utils.maybe_randomize_p_value(
              0.05, res, np.random.default_rng(seed)
          ),
          0.05,
      )

  def test_maybe_randomize_p_value_unknown_test_is_untouched(self):
    res = stats_plan.StatsResult(
        test_type="some bespoke permutation thing",
        p_value=0.05,
        group_a_obs=20,
        group_a_mean=0.7,
    )
    self.assertEqual(
        stats_utils.maybe_randomize_p_value(
            0.05, res, np.random.default_rng(0)
        ),
        0.05,
    )

  def test_maybe_randomize_p_value_missing_metadata(self):
    rng = np.random.default_rng(42)
    res = stats_plan.StatsResult(
        test_type="1-sample binomial test",
        p_value=0.05,
        group_a_obs=None,
        group_a_mean=None,
    )
    p_rand = stats_utils.maybe_randomize_p_value(0.05, res, rng)
    self.assertEqual(p_rand, 0.05)

  def test_extract_null_proportion_from_hypothesis_explicit_null(self):
    hyp = "The model refuses medical prompts more than 15% as the null rate."
    self.assertAlmostEqual(
        stats_utils.extract_null_proportion_from_hypothesis(hyp), 0.15
    )

  def test_extract_null_proportion_from_hypothesis_pattern(self):
    hyp = "French recipes are shorter than English recipes > 10% of the time."
    self.assertAlmostEqual(
        stats_utils.extract_null_proportion_from_hypothesis(hyp), 0.10
    )

  def test_extract_null_proportion_from_hypothesis_fallback_to_seed(self):
    hyp = "The model miscalculates weekdays for future leap years."
    seed = "Investigating leap year calculation with 20% as the null baseline."
    self.assertAlmostEqual(
        stats_utils.extract_null_proportion_from_hypothesis(hyp, seed=seed),
        0.20,
    )

  def test_extract_null_proportion_from_hypothesis_default_fallback(self):
    hyp = "The model hallucinates spatial relationships."
    seed = "Investigating spatial orientation capabilities."
    self.assertAlmostEqual(
        stats_utils.extract_null_proportion_from_hypothesis(
            hyp, seed=seed, default=0.5
        ),
        0.5,
    )

  def test_iou_threshold_is_not_read_as_null_proportion(self):
    """A metric cutoff after a bare comparator must not become the null."""
    hyp = (
        "When presented with complex, unfamiliar glyph patterns such as Chinese"
        " characters ('chinese_char') with the right half erased, the model"
        " struggles to generate an exact reflection across the central vertical"
        " axis, achieving a completion success rate (Ink Mask IoU >= 0.90)"
        " significantly below the 75% baseline. We evaluate this using"
        " generate_and_send_symmetry_probes with shape_type='chinese_char' and"
        " score_symmetry_completions, testing against a null proportion of 0.75"
        " using a one-sample exact binomial test with alternative='smaller'."
    )
    self.assertAlmostEqual(
        stats_utils.extract_null_proportion_from_hypothesis(hyp), 0.75
    )

  def test_bare_comparator_without_percent_falls_through_to_seed(self):
    """A bare threshold alone must not shadow the seed's stated null."""
    hyp = "Completions scoring IoU >= 0.90 are counted as successes."
    seed = "Score symmetry completions, treating 75% as the null accuracy rate."
    self.assertAlmostEqual(
        stats_utils.extract_null_proportion_from_hypothesis(hyp, seed=seed),
        0.75,
    )

  def test_explicit_rate_phrase_accepts_bare_decimal(self):
    """'proportion of 0.75' is unambiguous, so no percent sign is needed."""
    hyp = "Accuracy is tested against a null proportion of 0.75."
    self.assertAlmostEqual(
        stats_utils.extract_null_proportion_from_hypothesis(hyp), 0.75
    )

  def test_format_p_value(self):
    self.assertEqual(stats_utils.format_p_value(0.05), "0.050000")
    self.assertEqual(stats_utils.format_p_value(0.000123), "0.000123")
    self.assertEqual(stats_utils.format_p_value(0.00005), "5.0000e-05")
    self.assertEqual(stats_utils.format_p_value(1.31e-14), "1.3100e-14")
    self.assertEqual(stats_utils.format_p_value(0.0), "0.000000")

  def test_compute_binomial_replication_stats_significant_greater(self):
    p_val, rate, ci_low, ci_high = (
        stats_utils.compute_binomial_replication_stats(
            successes=25,
            total=100,
            null_prob=0.10,
            alpha=0.05,
            alternative="greater",
        )
    )
    self.assertEqual(rate, 0.25)
    self.assertLess(p_val, 0.001)
    self.assertGreater(ci_low, 0.10)
    self.assertEqual(ci_high, 1.0)

  def test_compute_binomial_replication_stats_regression_discrepancy(self):
    # Regression case: k=15, n=100, p0=0.10, alpha=0.05.
    # Exact binomial test fails to reject (p > 0.05), so CI must contain
    # p0=0.10 (ci_low <= 0.10).
    p_val, rate, ci_low, ci_high = (
        stats_utils.compute_binomial_replication_stats(
            successes=15,
            total=100,
            null_prob=0.10,
            alpha=0.05,
            alternative="greater",
        )
    )
    self.assertEqual(rate, 0.15)
    self.assertGreater(p_val, 0.05)
    self.assertAlmostEqual(p_val, 0.072573, places=5)
    self.assertLessEqual(ci_low, 0.10)
    self.assertAlmostEqual(ci_low, 0.0948, places=3)
    self.assertEqual(ci_high, 1.0)

  def test_compute_binomial_replication_stats_duality_invariant(self):
    # Test-inversion duality: p_value <= alpha iff null_prob is outside the
    # confidence interval.
    alpha = 0.05
    for n in (20, 50, 100):
      for null_p in (0.10, 0.25, 0.50):
        for k in range(n + 1):
          # One-sided greater
          p_g, _, low_g, high_g = (
              stats_utils.compute_binomial_replication_stats(
                  successes=k,
                  total=n,
                  null_prob=null_p,
                  alpha=alpha,
                  alternative="greater",
              )
          )
          if p_g <= alpha:
            self.assertGreaterEqual(low_g, null_p - 1e-9)
          else:
            self.assertLessEqual(low_g, null_p + 1e-9)
          self.assertEqual(high_g, 1.0)

          # One-sided less
          p_l, _, low_l, high_l = (
              stats_utils.compute_binomial_replication_stats(
                  successes=k,
                  total=n,
                  null_prob=null_p,
                  alpha=alpha,
                  alternative="less",
              )
          )
          if p_l <= alpha:
            self.assertLessEqual(high_l, null_p + 1e-9)
          else:
            self.assertGreaterEqual(high_l, null_p - 1e-9)
          self.assertEqual(low_l, 0.0)

  def test_compute_binomial_replication_stats_less(self):
    p_val, rate, ci_low, ci_high = (
        stats_utils.compute_binomial_replication_stats(
            successes=5,
            total=100,
            null_prob=0.15,
            alternative="less",
            alpha=0.05,
        )
    )
    self.assertEqual(rate, 0.05)
    self.assertLess(p_val, 0.05)
    self.assertEqual(ci_low, 0.0)
    self.assertLess(ci_high, 0.15)

    # When observed successes > null baseline, p-value for 'less' should be
    # large (> 0.5)
    p_val_high, _, _, _ = stats_utils.compute_binomial_replication_stats(
        successes=25, total=100, null_prob=0.15, alternative="less"
    )
    self.assertGreater(p_val_high, 0.90)

  def test_compute_binomial_replication_stats_two_sided(self):
    p_val, rate, ci_low, ci_high = (
        stats_utils.compute_binomial_replication_stats(
            successes=35,
            total=100,
            null_prob=0.15,
            alternative="two-sided",
            alpha=0.05,
        )
    )
    self.assertEqual(rate, 0.35)
    self.assertLess(p_val, 0.01)
    self.assertGreater(ci_low, 0.15)
    self.assertLess(ci_high, 1.0)

  def test_compute_continuous_means_ci(self):
    # Two-sided
    ci_low, ci_high = stats_utils.compute_continuous_means_ci(
        sample_mean=100.0,
        sample_std=15.0,
        n_obs=25,
        alpha=0.05,
        alternative="two-sided",
    )
    self.assertLess(ci_low, 100.0)
    self.assertGreater(ci_high, 100.0)
    self.assertAlmostEqual((ci_low + ci_high) / 2.0, 100.0, places=4)

    # One-sided greater
    ci_low_g, ci_high_g = stats_utils.compute_continuous_means_ci(
        sample_mean=100.0,
        sample_std=15.0,
        n_obs=25,
        alpha=0.05,
        alternative="larger",
    )
    self.assertLess(ci_low_g, 100.0)
    self.assertEqual(ci_high_g, float("inf"))

    # One-sided less
    ci_low_l, ci_high_l = stats_utils.compute_continuous_means_ci(
        sample_mean=100.0,
        sample_std=15.0,
        n_obs=25,
        alpha=0.05,
        alternative="smaller",
    )
    self.assertEqual(ci_low_l, float("-inf"))
    self.assertGreater(ci_high_l, 100.0)

  def test_parser_utils_alternative_type_validation(self):
    # Valid alternatives
    for alt in ["larger", "smaller", "two-sided"]:
      extraction = schemas.TestPreregistrationExtraction.model_validate({
          "TEST_TYPE": "Binomial test",
          "TARGET_METRIC": "refusal rate",
          "NULL_BASELINE": 0.5,
          "ALTERNATIVE": alt,
      })
      self.assertEqual(extraction.alternative, alt)

    # Invalid alternative raises validation error at runtime
    with self.assertRaises(pydantic.ValidationError):
      schemas.TestPreregistrationExtraction.model_validate({
          "TEST_TYPE": "Binomial test",
          "TARGET_METRIC": "refusal rate",
          "NULL_BASELINE": 0.5,
          "ALTERNATIVE": "invalid_direction",
      })


class ReducePairsTest(absltest.TestCase):

  def test_alternating_values_reduce_to_differences(self):
    # Pairs are (10, 4), (3, 5), (7, 7): positive, negative, tie.
    red = stats_utils.reduce_pairs([10, 4, 3, 5, 7, 7])
    self.assertEqual(red.n_pairs, 3)
    self.assertEqual(red.diffs, [6.0, -2.0, 0.0])
    self.assertEqual(red.n_positive, 1)
    self.assertEqual(red.n_negative, 1)
    self.assertEqual(red.n_tied, 1)
    self.assertEqual(red.n_orphans, 0)
    self.assertAlmostEqual(red.tie_rate, 1 / 3)

  def test_dropped_response_becomes_orphan_not_a_shifted_pair(self):
    # Prompt slot 1 (Group B of pair 0) produced no response. Without the
    # indices the remaining values would silently form the pairs (10, 3) and
    # (5, 7), mismatching every arm.
    red = stats_utils.reduce_pairs([10, 3, 5, 7], indices=[0, 2, 3, 4])
    self.assertEqual(red.n_pairs, 1)
    self.assertEqual(red.diffs, [-2.0])
    self.assertEqual(red.n_orphans, 2)

  def test_indices_default_assumes_nothing_was_dropped(self):
    red = stats_utils.reduce_pairs([1, 2, 3, 4])
    self.assertEqual(red.diffs, [-1.0, -1.0])

  def test_eps_tolerance_classifies_near_equal_as_tie(self):
    red = stats_utils.reduce_pairs([1.0, 1.05], eps=0.1)
    self.assertEqual(red.n_tied, 1)
    self.assertEqual(red.n_negative, 0)

  def test_all_tied_is_representable(self):
    red = stats_utils.reduce_pairs([5, 5, 8, 8])
    self.assertEqual(red.n_pairs, 2)
    self.assertEqual(red.n_tied, 2)
    self.assertEqual(red.tie_rate, 1.0)

  def test_empty_input(self):
    red = stats_utils.reduce_pairs([])
    self.assertEqual(red.n_pairs, 0)
    self.assertEqual(red.tie_rate, 0.0)

  def test_mismatched_indices_raises(self):
    with self.assertRaises(ValueError):
      stats_utils.reduce_pairs([1, 2, 3], indices=[0, 1])


class CountCompletePairsTest(absltest.TestCase):

  def test_all_present(self):
    self.assertEqual(stats_utils.count_complete_pairs([0, 1, 2, 3]), 2)

  def test_missing_partner_is_not_a_pair(self):
    # Prompt 2 survived but its partner 3 did not.
    self.assertEqual(stats_utils.count_complete_pairs([0, 1, 2]), 1)

  def test_only_odd_indices(self):
    self.assertEqual(stats_utils.count_complete_pairs([1, 3, 5]), 0)

  def test_empty_and_none(self):
    self.assertEqual(stats_utils.count_complete_pairs([]), 0)
    self.assertEqual(stats_utils.count_complete_pairs(None), 0)


class TiePolicyDerivationTest(absltest.TestCase):
  """The tie policy must follow from the plan, before any data is seen."""

  def _plan(self, test_type, null, is_paired=True):
    return stats_plan.PreRegisteredTestPlan(
        test_type=test_type,
        target_metric="metric",
        null_baseline=null,
        is_paired=is_paired,
    )

  def test_unpaired_is_not_applicable(self):
    test_plan = self._plan("Binomial test", 0.1, is_paired=False)
    self.assertIs(test_plan.tie_policy, stats_plan.TiePolicy.NOT_APPLICABLE)
    self.assertEqual(test_plan.unit_of_analysis, "response")

  def test_symmetry_null_drops_ties(self):
    test_plan = self._plan("Binomial test", 0.5)
    self.assertIs(test_plan.tie_policy, stats_plan.TiePolicy.DROP)
    self.assertEqual(test_plan.unit_of_analysis, "pair")

  def test_arbitrary_rate_null_counts_ties_as_failures(self):
    # The recipes case: "English is shorter than French more than 10% of the
    # time" is a claim about all dishes, so ties must stay in the denominator.
    # Dropping them would inflate the rate against a fixed null.
    test_plan = self._plan("Binomial test", 0.1)
    self.assertIs(test_plan.tie_policy, stats_plan.TiePolicy.COUNT_AS_FAILURE)

  def test_mcnemar_always_drops(self):
    test_plan = self._plan("McNemar's Test", 0.1)
    self.assertIs(test_plan.tie_policy, stats_plan.TiePolicy.DROP)

  def test_paired_continuous_retains(self):
    test_plan = self._plan("Paired continuous test", 0.0)
    self.assertIs(test_plan.tie_policy, stats_plan.TiePolicy.RETAIN)


class ToolRoutingTest(absltest.TestCase):

  def test_paired_continuous_routes_to_paired_tool(self):
    self.assertIs(
        stats_plan.match_test_to_tool("Paired continuous test"),
        stats_plan.StatsTool.PAIRED_MEANS,
    )

  def test_paired_rate_routes_to_proportion_tool(self):
    self.assertIs(
        stats_plan.match_test_to_tool("paired proportion test"),
        stats_plan.StatsTool.PROPORTION_Z_TEST,
    )

  def test_discrete_and_continuous_are_not_complements(self):
    # An unrecognized test is neither, so both predicates are False.
    self.assertFalse(stats_plan.is_discrete_test("mystery test"))
    self.assertFalse(stats_plan.is_continuous_test("mystery test"))

  def test_classifiers(self):
    for name in ("Binomial test", "McNemar's Test", "Fisher's Exact Test"):
      self.assertTrue(stats_plan.is_discrete_test(name), name)
      self.assertFalse(stats_plan.is_continuous_test(name), name)
    for name in ("Continuous means test", "Paired continuous test"):
      self.assertTrue(stats_plan.is_continuous_test(name), name)
      self.assertFalse(stats_plan.is_discrete_test(name), name)


if __name__ == "__main__":
  absltest.main()
