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

"""Unit tests for stats.estimators."""

from absl.testing import absltest
import numpy as np
from scipy import stats

from disco.stats import estimators


class EstimatorsTest(absltest.TestCase):

  def test_bca_bootstrap_ci_empty(self):
    est = estimators.bca_bootstrap_ci([])
    self.assertFalse(est.ci_valid)
    self.assertTrue(np.isnan(est.mean))

  def test_bca_bootstrap_ci_single(self):
    est = estimators.bca_bootstrap_ci([5.0])
    self.assertEqual(est.mean, 5.0)
    self.assertFalse(est.ci_valid)

  def test_bca_bootstrap_ci_constant(self):
    est = estimators.bca_bootstrap_ci([3.0, 3.0, 3.0])
    self.assertEqual(est.mean, 3.0)
    self.assertEqual(est.ci_low, 3.0)
    self.assertEqual(est.ci_high, 3.0)
    self.assertTrue(est.ci_valid)
    self.assertEqual(est.ci_method, "point")

  def test_bca_bootstrap_ci_clamping(self):
    est = estimators.bca_bootstrap_ci(
        [0.1, 0.2, 0.3], val_min=0.0, val_max=1.0, resamples=500, seed=42
    )
    self.assertTrue(est.ci_valid)
    self.assertGreaterEqual(est.ci_low, 0.0)
    self.assertLessEqual(est.ci_high, 1.0)

  def test_storey_pi0_estimator(self):
    val = estimators.storey_pi0_estimator(
        w_sum=5.0, r_total=20, lam=0.5, plus_one=True
    )
    # (1 + 5) / (20 * 0.5) = 6 / 10 = 0.6
    self.assertIsNotNone(val)
    assert val is not None
    self.assertAlmostEqual(val, 0.6)
    self.assertGreaterEqual(val, 0.0)
    self.assertLessEqual(val, 1.0)

  def test_compute_storey_and_thresholded_fdr(self):
    records = [
        {"status": "OK", "p_value": 0.01},
        {"status": "OK", "p_value": 0.04},
        {"status": "OK", "p_value": 0.20},
        {"status": "OK", "p_value": 0.80},
        {"status": "SCRAPE_FAILED", "p_value": 1.0},
    ]
    # 4 valid records: 1 has p > 0.5 (0.80).
    # plus_one=True -> (1 + 1) / (4 * 0.5) = 1.0
    # plus_one=False -> 1 / (4 * 0.5) = 0.5
    storey_no_plus = estimators.compute_storey_fdr(
        records, lam=0.5, plus_one=False
    )
    self.assertIsNotNone(storey_no_plus)
    assert storey_no_plus is not None
    self.assertAlmostEqual(storey_no_plus, 0.5)
    storey_plus = estimators.compute_storey_fdr(records, lam=0.5, plus_one=True)
    self.assertIsNotNone(storey_plus)
    assert storey_plus is not None
    self.assertAlmostEqual(storey_plus, 1.0)
    # 2 of 4 valid records have p > 0.05 -> thresholded FDR = 0.5
    thresh = estimators.compute_thresholded_fdr(records, alpha=0.05)
    self.assertIsNotNone(thresh)
    assert thresh is not None
    self.assertAlmostEqual(thresh, 0.5)
    self.assertIsNone(estimators.compute_storey_fdr([]))
    self.assertIsNone(estimators.compute_thresholded_fdr([]))

  def test_bca_bootstrap_ci_matches_scipy(self):
    data = [1.0, 2.0, 3.5, 4.0, 5.0, 6.0, 7.5, 8.0, 9.0, 10.0]
    est = estimators.bca_bootstrap_ci(data, resamples=2000, seed=42)
    self.assertTrue(est.ci_valid)
    scipy_res = stats.bootstrap(
        (data,),
        np.mean,
        confidence_level=0.95,
        n_resamples=2000,
        method="BCa",
        random_state=42,
    )
    self.assertAlmostEqual(
        est.ci_low, float(scipy_res.confidence_interval.low), delta=0.5
    )
    self.assertAlmostEqual(
        est.ci_high, float(scipy_res.confidence_interval.high), delta=0.5
    )


if __name__ == "__main__":
  absltest.main()
