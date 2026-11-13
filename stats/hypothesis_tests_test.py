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

"""Unit tests for stats.hypothesis_tests."""

from absl.testing import absltest

from disco import stats_utils
from disco.stats import hypothesis_tests as ht_lib


class HypothesisTestsTest(absltest.TestCase):

  def test_get_t_test_p_value(self):
    self.assertEqual(ht_lib.get_t_test_p_value(2.0, "greater", df=0), 1.0)
    p_two = ht_lib.get_t_test_p_value(2.0, "two-sided", df=10)
    p_one = ht_lib.get_t_test_p_value(2.0, "greater", df=10)
    self.assertAlmostEqual(p_two, 2.0 * p_one)

  def test_compute_binomial_replication_stats(self):
    p_val, rate, ci_low, ci_high = ht_lib.compute_binomial_replication_stats(
        successes=8, total=10, null_prob=0.5, alpha=0.05, alternative="greater"
    )
    self.assertAlmostEqual(rate, 0.8)
    self.assertLess(p_val, 0.06)
    self.assertGreaterEqual(ci_low, 0.0)
    self.assertLessEqual(ci_high, 1.0)

  def test_compute_continuous_means_ci(self):
    low, high = ht_lib.compute_continuous_means_ci(
        sample_mean=10.0,
        sample_std=2.0,
        n_obs=25,
        alpha=0.05,
        alternative="two-sided",
    )
    self.assertLess(low, 10.0)
    self.assertGreater(high, 10.0)

    # One-sided greater has infinite upper bound
    low_g, high_g = ht_lib.compute_continuous_means_ci(
        sample_mean=10.0,
        sample_std=2.0,
        n_obs=25,
        alpha=0.05,
        alternative="greater",
    )
    self.assertEqual(high_g, float("inf"))
    self.assertLess(low_g, 10.0)

  def test_stats_utils_shim_parity(self):
    self.assertIs(stats_utils.get_t_test_p_value, ht_lib.get_t_test_p_value)
    self.assertIs(
        stats_utils.compute_binomial_replication_stats,
        ht_lib.compute_binomial_replication_stats,
    )
    self.assertIs(
        stats_utils.compute_continuous_means_ci,
        ht_lib.compute_continuous_means_ci,
    )


if __name__ == "__main__":
  absltest.main()
