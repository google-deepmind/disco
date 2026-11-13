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

"""Unit tests for stats.coarsened.null_tables."""

from absl.testing import absltest
import numpy as np

from disco.stats.coarsened import null_tables as nt_lib
from disco.stats.coarsened import status as status_lib


class NullTablesTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Build a tiny mock NullCoarsenedTables for testing lookup and expected
    # values.
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

  def test_get_expected_null_e(self):
    # Exit high gives 0.0
    self.assertEqual(
        nt_lib.get_expected_null_e(
            status_lib.TerminalStatus.high(), 1, self.dummy_tables
        ),
        0.0,
    )
    # Exit low gives E_L_table
    self.assertEqual(
        nt_lib.get_expected_null_e(
            status_lib.TerminalStatus.low(), 1, self.dummy_tables
        ),
        0.2,
    )
    # Active gives E_A_table
    self.assertEqual(
        nt_lib.get_expected_null_e(
            status_lib.TerminalStatus.active(), 2, self.dummy_tables
        ),
        2.0,
    )
    # Continuous active with exact_e returns exact_e
    self.assertEqual(
        nt_lib.get_expected_null_e(
            status_lib.TerminalStatus.active(),
            1,
            self.dummy_tables,
            exact_e=3.14,
        ),
        3.14,
    )

  def test_compute_null_coarsened_tables_enforce_exit_low(self):
    tables_true = nt_lib.compute_null_coarsened_tables(
        kappa=0.5,
        e_min=0.8,
        e_max=3.0,
        n_max=5,
        k_max=3,
        num_bins=2,
        alpha=0.05,
        n_sim=5_000,
        seed=123,
        enforce_exit_low=True,
    )
    tables_false = nt_lib.compute_null_coarsened_tables(
        kappa=0.5,
        e_min=0.8,
        e_max=3.0,
        n_max=5,
        k_max=3,
        num_bins=2,
        alpha=0.05,
        n_sim=5_000,
        seed=123,
        enforce_exit_low=False,
    )
    self.assertTrue(tables_true.enforce_exit_low)
    self.assertFalse(tables_false.enforce_exit_low)
    # Without low absorption, survival + high exit = 1 at every step, and P_H
    # is higher.
    np.testing.assert_allclose(
        tables_false.P_surv_table + tables_false.P_H_table, 1.0
    )
    self.assertGreater(tables_false.P_H_max, tables_true.P_H_max)
    # Transitions out of bin 0 (L) occur only when enforce_exit_low=False.
    self.assertEqual(tables_true.r_hat_transition_counts[0, :].sum(), 0)
    self.assertGreater(tables_false.r_hat_transition_counts[0, :].sum(), 0)

  def test_compute_null_coarsened_tables_binary_coarsening(self):
    tables = nt_lib.compute_null_coarsened_tables(
        kappa=0.5,
        e_min=0.0,
        e_max=5.0,
        n_max=4,
        k_max=3,
        num_bins=1,
        alpha=0.05,
        n_sim=2_000,
        seed=42,
    )
    np.testing.assert_allclose(tables.bin_edges, [0.0, 5.0])
    np.testing.assert_allclose(tables.P_surv_table + tables.P_H_table, 1.0)
    np.testing.assert_allclose(tables.E_L_table, 0.0)
    assert tables.bin_log_centers is not None
    self.assertTrue(np.all(np.isfinite(tables.bin_log_centers)))
    score_a_to_a = nt_lib.lookup_transition_score(tables, 1, 1)
    score_a_to_h = nt_lib.lookup_transition_score(tables, 1, 2)
    self.assertTrue(np.isfinite(score_a_to_a))
    self.assertTrue(np.isfinite(score_a_to_h))


if __name__ == "__main__":
  absltest.main()
