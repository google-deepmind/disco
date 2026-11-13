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

"""Unit tests for stats.calibrators."""

from absl.testing import absltest
import numpy as np
from scipy import integrate

from disco.stats import calibrators


class CalibratorsTest(absltest.TestCase):

  def test_p_to_e_values(self):
    self.assertAlmostEqual(
        calibrators.p_to_e(0.5, kappa=0.5), 0.5 * (0.5**-0.5)
    )
    self.assertAlmostEqual(
        calibrators.p_to_e(0.1, kappa=0.5), 0.5 * (0.1**-0.5)
    )
    self.assertAlmostEqual(calibrators.p_to_e(1.0, kappa=0.5), 0.5)

  def test_p_to_e_monotonicity(self):
    p_grid = np.linspace(0.01, 1.0, 50)
    e_vals = [calibrators.p_to_e(p, kappa=0.5) for p in p_grid]
    for i in range(len(e_vals) - 1):
      self.assertGreaterEqual(e_vals[i], e_vals[i + 1])

  def test_p_to_e_integral_le_one(self):
    for kappa in [0.1, 0.25, 0.5, 0.75, 0.9]:
      integral, _ = integrate.quad(
          lambda p, kappa=kappa: calibrators.p_to_e(p, kappa=kappa), 0.0, 1.0
      )
      self.assertLessEqual(integral, 1.0 + 1e-9)

  def test_p_to_e_domain_errors(self):
    with self.assertRaises(ValueError):
      calibrators.p_to_e(-0.1)
    with self.assertRaises(ValueError):
      calibrators.p_to_e(1.1)
    with self.assertRaises(ValueError):
      calibrators.p_to_e(0.5, kappa=0.0)
    with self.assertRaises(ValueError):
      calibrators.p_to_e(0.5, kappa=1.0)

  def test_get_threshold_from_alpha(self):
    self.assertAlmostEqual(calibrators.get_threshold_from_alpha(0.05), 20.0)
    self.assertAlmostEqual(calibrators.get_threshold_from_alpha(0.01), 100.0)

  def test_calibrated_evalue_survival(self):
    self.assertEqual(calibrators.calibrated_evalue_survival(0.0, 0.5, 1), 1.0)
    self.assertEqual(calibrators.calibrated_evalue_survival(-1.0, 0.5, 1), 1.0)
    s_scalar = calibrators.calibrated_evalue_survival(5.0, 0.5, 2)
    s_arr = calibrators.calibrated_evalue_survival(np.array([5.0]), 0.5, 2)
    self.assertIsInstance(s_scalar, float)
    self.assertIsInstance(s_arr, np.ndarray)
    self.assertAlmostEqual(s_scalar, float(s_arr[0]))
    self.assertGreaterEqual(s_scalar, 0.0)
    self.assertLessEqual(s_scalar, 1.0)

  def test_calibrated_evalue_survival_monotonicity_and_mc(self):
    x_grid = np.linspace(0.1, 10.0, 20)
    surv_vals = [
        calibrators.calibrated_evalue_survival(x, kappa=0.5, m=1)
        for x in x_grid
    ]
    for i in range(len(surv_vals) - 1):
      self.assertGreaterEqual(surv_vals[i], surv_vals[i + 1])

    rng = np.random.default_rng(42)
    p_samples = rng.uniform(0.0001, 1.0, size=50000)
    e_samples = np.array([calibrators.p_to_e(p, kappa=0.5) for p in p_samples])
    for x in [1.0, 2.0, 5.0]:
      mc_p = float(np.mean(e_samples >= x))
      exact_p = float(calibrators.calibrated_evalue_survival(x, kappa=0.5, m=1))
      self.assertAlmostEqual(mc_p, exact_p, delta=0.02)


if __name__ == "__main__":
  absltest.main()
