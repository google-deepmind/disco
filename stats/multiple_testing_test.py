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

"""Unit tests for stats.multiple_testing."""

from absl.testing import absltest
import numpy as np
from scipy import stats

from disco.stats import calibrators
from disco.stats import multiple_testing


class MultipleTestingTest(absltest.TestCase):

  def test_e_bh_empty_or_zero_alpha(self):
    rejections, threshold = multiple_testing.e_bh_rejection({}, 0.05)
    self.assertEqual(rejections, set())
    self.assertEqual(threshold, float("inf"))

    rejections, threshold = multiple_testing.e_bh_rejection({"h1": 100.0}, 0.0)
    self.assertEqual(rejections, set())
    self.assertEqual(threshold, float("inf"))

  def test_e_bh_rejection(self):
    # K=2, alpha=0.05, 1/alpha=20.
    # For k=1: 1 * e[1] / 2 >= 20 => e[1] >= 40.
    # For k=2: 2 * e[2] / 2 >= 20 => e[2] >= 20.
    e_vals = {"h1": 50.0, "h2": 25.0}
    rejections, threshold = multiple_testing.e_bh_rejection(e_vals, alpha=0.05)
    self.assertEqual(rejections, {"h1", "h2"})
    self.assertAlmostEqual(threshold, 2 / (0.05 * 2))  # 20.0

  def test_compute_harmonic_number(self):
    self.assertEqual(multiple_testing.compute_harmonic_number(0), 1.0)
    self.assertEqual(multiple_testing.compute_harmonic_number(1), 1.0)
    self.assertAlmostEqual(multiple_testing.compute_harmonic_number(2), 1.5)
    self.assertAlmostEqual(
        multiple_testing.compute_harmonic_number(3), 1.0 + 0.5 + 1.0 / 3.0
    )

  def test_compute_bh_procedure(self):
    res_empty = multiple_testing.compute_bh_procedure(
        [], alpha=0.05, m_total=10
    )
    self.assertEqual(res_empty["k_star"], 0)
    self.assertEqual(res_empty["rejected_indices"], set())

    # p-values: [0.01, 0.04, 0.5], m_total=3, alpha=0.05
    # k=1: p_(1)=0.01 <= (1/3)*0.05 = 0.016667 (True)
    # k=2: p_(2)=0.04 <= (2/3)*0.05 = 0.033333 (False)
    # k=3: p_(3)=0.5 <= (3/3)*0.05 = 0.05 (False)
    # k*=1, threshold = (1/3)*0.05 = 0.016667, rejects index 0
    res = multiple_testing.compute_bh_procedure(
        [0.01, 0.04, 0.5], alpha=0.05, m_total=3
    )
    self.assertEqual(res["k_star"], 1)
    self.assertEqual(res["rejected_indices"], {0})
    self.assertAlmostEqual(res["rejection_threshold"], 0.05 / 3.0)
    self.assertEqual(res["ranks"], [1, 2, 3])

  def test_compute_by_procedure(self):
    res = multiple_testing.compute_by_procedure(
        [0.01, 0.04, 0.5], alpha=0.05, m_total=3
    )
    c_m = 1.0 + 0.5 + 1.0 / 3.0
    self.assertAlmostEqual(res["harmonic_c"], c_m)
    self.assertLessEqual(res["k_star"], 1)

  def test_compute_ebh_unboosted(self):
    res = multiple_testing.compute_ebh_unboosted(
        [0.001, 0.1], alpha=0.05, kappa=0.5, m_total=2
    )
    self.assertIn("rejected_indices", res)
    self.assertIn("equivalent_p_cutoff", res)
    self.assertLen(res["e_values"], 2)

  def test_compute_ebh_boosted(self):
    res = multiple_testing.compute_ebh_boosted(
        [0.001, 0.1], alpha=0.05, kappa=0.5, m_total=2, probe_counts=[1, 2]
    )
    self.assertIn("boost_factors", res)
    self.assertIn("boosted_e_values", res)
    self.assertGreaterEqual(res["boost_factors"][0], 1.0)
    self.assertGreaterEqual(res["boost_factors"][1], 1.0)

  def test_e_bh_properties(self):
    e_vals = {"h1": 50.0, "h2": 25.0, "h3": 10.0}
    rej, _ = multiple_testing.e_bh_rejection(e_vals, alpha=0.05)

    # Raising e-values rejects a superset
    e_vals_higher = {"h1": 60.0, "h2": 30.0, "h3": 15.0}
    rej_higher, _ = multiple_testing.e_bh_rejection(e_vals_higher, alpha=0.05)
    self.assertTrue(rej.issubset(rej_higher))

    # Raising k_total_override rejects a subset
    rej_k3, _ = multiple_testing.e_bh_rejection(
        e_vals, alpha=0.05, k_total_override=3
    )
    rej_k5, _ = multiple_testing.e_bh_rejection(
        e_vals, alpha=0.05, k_total_override=5
    )
    self.assertTrue(rej_k5.issubset(rej_k3))

    # Permutation invariance
    e_vals_perm = {"h3": 10.0, "h1": 50.0, "h2": 25.0}
    rej_perm, _ = multiple_testing.e_bh_rejection(e_vals_perm, alpha=0.05)
    self.assertEqual(rej, rej_perm)

  def test_bh_by_match_scipy(self):
    pvals = [0.001, 0.01, 0.03, 0.05, 0.2, 0.8]
    alpha = 0.05
    scipy_bh_adj = stats.false_discovery_control(pvals, method="bh")
    scipy_bh_rejections = set(np.where(scipy_bh_adj <= alpha)[0])
    our_bh = multiple_testing.compute_bh_procedure(
        pvals, alpha=alpha, m_total=len(pvals)
    )
    self.assertEqual(our_bh["rejected_indices"], scipy_bh_rejections)

    scipy_by_adj = stats.false_discovery_control(pvals, method="by")
    scipy_by_rejections = set(np.where(scipy_by_adj <= alpha)[0])
    our_by = multiple_testing.compute_by_procedure(
        pvals, alpha=alpha, m_total=len(pvals)
    )
    self.assertEqual(our_by["rejected_indices"], scipy_by_rejections)

  def test_ebh_unboosted_equals_e_bh_rejection(self):
    pvals = [0.001, 0.02, 0.04, 0.15]
    alpha = 0.05
    kappa = 0.5
    res = multiple_testing.compute_ebh_unboosted(
        pvals, alpha=alpha, kappa=kappa, m_total=len(pvals)
    )
    e_vals_dict = {
        f"h{i}": calibrators.p_to_e(p, kappa=kappa) for i, p in enumerate(pvals)
    }
    rej_set, _ = multiple_testing.e_bh_rejection(
        e_vals_dict, alpha=alpha, k_total_override=len(pvals)
    )
    expected_indices = {int(h[1:]) for h in rej_set}
    self.assertEqual(res["rejected_indices"], expected_indices)

  def test_boosted_rejections_superset_of_unboosted(self):
    pvals = [0.001, 0.01, 0.03, 0.05, 0.1, 0.2]
    unboosted = multiple_testing.compute_ebh_unboosted(
        pvals, alpha=0.05, kappa=0.5, m_total=len(pvals)
    )
    boosted = multiple_testing.compute_ebh_boosted(
        pvals,
        alpha=0.05,
        kappa=0.5,
        m_total=len(pvals),
        probe_counts=[1] * len(pvals),
    )
    self.assertTrue(
        unboosted["rejected_indices"].issubset(boosted["rejected_indices"])
    )


if __name__ == "__main__":
  absltest.main()
