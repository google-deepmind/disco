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

"""Unit tests for stats.boosting."""

from absl.testing import absltest
import numpy as np

from disco.stats import boosting


class BoostingTest(absltest.TestCase):

  def test_staircase_t_below_one(self):
    self.assertEqual(boosting.staircase_t(0.5, 10), 0.0)
    self.assertEqual(boosting.staircase_t(0.0, 10), 0.0)
    self.assertEqual(boosting.staircase_t(-1.0, 10), 0.0)

  def test_staircase_t_scalar_and_array(self):
    val = boosting.staircase_t(2.5, 5)
    arr = boosting.staircase_t(np.array([2.5]), 5)
    self.assertIsInstance(val, float)
    self.assertIsInstance(arr, np.ndarray)
    self.assertAlmostEqual(val, float(arr[0]))
    self.assertLessEqual(val, 2.5)

  def test_staircase_t_monotonicity(self):
    x_grid = np.linspace(0.0, 10.0, 50)
    for k in [1, 5, 10]:
      t_vals = [boosting.staircase_t(x, k) for x in x_grid]
      for i in range(len(t_vals) - 1):
        self.assertLessEqual(t_vals[i], t_vals[i + 1])
        self.assertLessEqual(t_vals[i], x_grid[i])

  def test_compute_boost_factor(self):
    b = boosting.compute_boost_factor(alpha=0.05, kappa=0.5, m=3, k_total=10)
    self.assertGreaterEqual(b, 1.0)
    for m in [1, 2, 5]:
      for k in [1, 5, 10]:
        factor = boosting.compute_boost_factor(
            alpha=0.05, kappa=0.5, m=m, k_total=k
        )
        self.assertGreaterEqual(factor, 1.0)
    # Non-positive parameters return 1.0
    self.assertEqual(boosting.compute_boost_factor(0.05, 0.5, 0, 10), 1.0)
    self.assertEqual(boosting.compute_boost_factor(0.05, 0.5, 3, 0), 1.0)

  def test_boost_probe_count_clips_to_one_and_n_max(self):
    self.assertEqual(boosting.boost_probe_count(0, 3), 1)
    self.assertEqual(boosting.boost_probe_count(2, 3), 2)
    self.assertEqual(boosting.boost_probe_count(12, 3), 3)

  def test_generate_k_grid(self):
    grid = boosting.generate_k_grid(k_min=2, k_max=20, n_points=5)
    self.assertEqual(grid[0], 2)
    self.assertEqual(grid[-1], 20)
    self.assertEqual(grid, sorted(list(set(grid))))


if __name__ == "__main__":
  absltest.main()
