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

"""Unit tests for stats.coarsened.certificates."""

from absl.testing import absltest
import numpy as np

from disco.stats.coarsened import certificates as cert_lib
from disco.stats.coarsened import null_tables as nt_lib


class CertificatesTest(absltest.TestCase):

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

  def test_compute_delta_post(self):
    counts = [1, 2]
    # P_H_table[0] = 0.05, P_H_table[1] = 0.10; delta_post = K_obs * P_H(n_max)
    delta = cert_lib.compute_delta_post(
        counts, self.dummy_tables.P_H_table, n_max=2
    )
    self.assertAlmostEqual(delta, 0.20)

  def test_compute_epsilon_sum(self):
    statuses = ["A", "L"]
    counts = [1, 1]
    # E_A[0] = 1.5, E_L[0] = 0.2
    eps_sum = cert_lib.compute_epsilon_sum(statuses, counts, self.dummy_tables)
    self.assertAlmostEqual(eps_sum, 1.7)

  def test_compute_disco_posthoc_certificate(self):
    cert = cert_lib.compute_disco_posthoc_certificate(
        terminal_statuses={"h1": "A", "h2": "L"},
        probe_counts={"h1": 1, "h2": 1},
        null_tables=self.dummy_tables,
        alpha_bh=0.05,
    )
    self.assertAlmostEqual(cert.delta_post, 0.20)
    self.assertAlmostEqual(cert.confidence, 0.80)
    self.assertEqual(cert.k_obs, 2)

  def _cert(self, statuses, counts, **kwargs):
    return cert_lib.compute_disco_posthoc_certificate(
        statuses, counts, self.dummy_tables, 0.05, **kwargs
    )

  def test_list_counts_follow_status_dict_order(self):
    # h2 is listed first; the list counts are read in that key order.
    by_id = self._cert({"h2": "L", "h1": "A"}, {"h1": 1, "h2": 2})
    mixed = self._cert({"h2": "L", "h1": "A"}, [2, 1])
    self.assertEqual(mixed, by_id)
    # E_A(1) + E_L(2) = 1.5 + 0.1, and delta_post = 2 * P_H(2) = 0.20.
    self.assertAlmostEqual(by_id.epsilon_post, 0.05 * 0.5 * 1.6)
    self.assertAlmostEqual(by_id.delta_post, 0.20)

  def test_list_statuses_follow_final_e_values_order(self):
    final_e = {"h2": 1.0, "h1": 3.0}
    by_id = self._cert(
        {"h1": "A", "h2": "L"}, {"h1": 1, "h2": 2}, final_e_values=final_e
    )
    mixed = self._cert(["L", "A"], {"h1": 1, "h2": 2}, final_e_values=final_e)
    self.assertEqual(mixed, by_id)

  def test_dict_counts_without_ids_raise(self):
    with self.assertRaisesRegex(ValueError, "probe_counts is keyed"):
      self._cert(["A", "L"], {"h1": 1, "h2": 2})

  def test_list_length_mismatch_raises(self):
    with self.assertRaisesRegex(
        ValueError, "probe_counts has 1 entries but terminal_statuses has 2"
    ):
      self._cert({"h1": "A", "h2": "L"}, [1])
    with self.assertRaisesRegex(
        ValueError, "probe_counts has 3 entries but terminal_statuses has 2"
    ):
      self._cert(["A", "L"], [1, 1, 1])

  def test_align_to_ids(self):
    self.assertEqual(
        cert_lib.align_to_ids({"b": 2}, ["a", "b"], 0, "x"), [0, 2]
    )
    self.assertEqual(cert_lib.align_to_ids([5, 6], ["a", "b"], 0, "x"), [5, 6])

  def test_compute_prehoc_phi_max_uses_k_grid(self):
    tables = nt_lib.compute_null_coarsened_tables(
        kappa=0.5,
        e_min=0.5,
        e_max=10.0,
        n_max=2,
        k_max=10,
        num_bins=1,
        alpha=0.1,
        n_sim=500,
        seed=42,
    )
    val_grid = cert_lib.compute_prehoc_phi_max(0.1, 0.5, 10, tables)
    phi_at_k_bar = max(
        max(
            cert_lib.recompute_phi_at_alpha(
                0.1, 0.5, [1, 2], 10, tables, status="A"
            ).values()
        ),
        max(
            cert_lib.recompute_phi_at_alpha(
                0.1, 0.5, [1, 2], 10, tables, status="L"
            ).values()
        ),
    )
    self.assertGreaterEqual(val_grid, phi_at_k_bar)


if __name__ == "__main__":
  absltest.main()
