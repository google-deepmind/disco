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

"""Unit tests for stats.coarsened.calibration."""

from absl.testing import absltest
import numpy as np

from disco.stats.coarsened import calibration as calib_lib
from disco.stats.coarsened import null_tables as nt_lib


class CalibrationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.dummy_tables = nt_lib.NullCoarsenedTables(
        num_bins=1,
        bin_edges=np.array([0.5, 10.0]),
        E_A_table=np.array([1.5, 2.0]),
        E_A_max=2.0,
        E_L_table=np.array([0.2, 0.1]),
        P_H_table=np.array([0.05, 0.1]),
        P_surv_table=np.array([0.9, 0.8]),
        P_H_max=0.1,
        boost_factors_grid=np.ones((2, 2)),
        phi_grid=np.zeros((2, 2)),
        phi_max=0.0,
        analytical_ceiling=1.0,
        alpha=0.05,
        kappa=0.5,
        e_min=0.5,
        e_max=10.0,
        n_max=2,
        k_max=2,
        r_hat_transition=np.zeros((3, 3)),
        r_hat_transition_counts=np.zeros((3, 3)),
        r_hat_marginal=np.zeros(3),
    )

  def test_find_optimal_prehoc_alpha_bh_zero_target(self):
    alpha_opt, phi_val = calib_lib.find_optimal_prehoc_alpha_bh(
        alpha_star=0.05,
        kappa=0.5,
        k_bar=2,
        null_tables=self.dummy_tables,
    )
    self.assertEqual(alpha_opt, 0.0)
    self.assertEqual(phi_val, 0.0)

  def test_compute_disco_alpha_sweep_empty(self):
    entries = calib_lib.compute_disco_alpha_sweep(
        final_e_values={},
        terminal_statuses={},
        probe_counts={},
        null_tables=self.dummy_tables,
        sweep_alphas=[0.05, 0.10],
    )
    self.assertEqual(entries, [])

  def test_compute_disco_alpha_sweep(self):
    e_vals = {"h1": 25.0, "h2": 5.0}
    statuses = {"h1": "A", "h2": "L"}
    counts = {"h1": 1, "h2": 1}
    entries = calib_lib.compute_disco_alpha_sweep(
        final_e_values=e_vals,
        terminal_statuses=statuses,
        probe_counts=counts,
        null_tables=self.dummy_tables,
        sweep_alphas=[0.05, 0.10],
        use_boosting=False,
    )
    self.assertLen(entries, 2)
    self.assertEqual(entries[0].alpha_target, 0.05)
    self.assertEqual(entries[1].alpha_target, 0.10)

  def test_compute_disco_alpha_sweep_list_inputs_match_dict_inputs(self):
    # h1 has 2 probes, so its boost uses m=2. List counts used to be read as
    # n_g=1 for every hypothesis, and dict statuses with list counts paired
    # them by position in two different dicts.
    e_vals = {"h1": 25.0, "h2": 5.0, "h3": 2.0}
    statuses = {"h3": "L", "h1": "A", "h2": "A"}
    counts = {"h1": 2, "h2": 1, "h3": 1}
    kwargs = dict(
        final_e_values=e_vals,
        null_tables=self.dummy_tables,
        sweep_alphas=[0.05, 0.2, 0.5],
        use_boosting=True,
    )
    by_dict = calib_lib.compute_disco_alpha_sweep(
        terminal_statuses=statuses, probe_counts=counts, **kwargs
    )
    as_lists = calib_lib.compute_disco_alpha_sweep(
        terminal_statuses=[statuses[h] for h in e_vals],
        probe_counts=[counts[h] for h in e_vals],
        **kwargs,
    )
    mixed = calib_lib.compute_disco_alpha_sweep(
        terminal_statuses=statuses,
        probe_counts=[counts[h] for h in e_vals],
        **kwargs,
    )
    self.assertEqual(as_lists, by_dict)
    self.assertEqual(mixed, by_dict)

  def test_compute_disco_alpha_sweep_rejects_list_length_mismatch(self):
    with self.assertRaisesRegex(ValueError, "probe_counts has 1 entries"):
      calib_lib.compute_disco_alpha_sweep(
          final_e_values={"h1": 25.0, "h2": 5.0},
          terminal_statuses=["A", "A"],
          probe_counts=[2],
          null_tables=self.dummy_tables,
          sweep_alphas=[0.05],
      )


if __name__ == "__main__":
  absltest.main()
