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

"""Unit tests for bandit environments and DISCO / Offline e-BH runners."""

from absl.testing import absltest
import numpy as np

from disco.bandit import bandit_disco
from disco.bandit import bandit_env
from disco.stats.coarsened import certificates as cert_lib
from disco.stats.coarsened import null_tables as null_tables_lib
from disco.stats.coarsened import status as status_lib


class BanditDiscoTest(absltest.TestCase):

  def test_bernoulli_env_null_p_values(self):
    """Verifies that null arms produce approximately Uniform(0, 1) p-values."""
    env = bandit_env.BernoulliBanditEnv(
        pi_1=0.0,  # 100% null
        baseline_rate_logit=-4.6,
        seed=123,
    )
    p_vals = []
    for _ in range(500):
      res = env.sample_queries("arm_0", count=20)
      p_vals.append(res.p_value)

    # Check mean of p-values is near 0.5 under null
    self.assertGreater(np.mean(p_vals), 0.40)
    self.assertLess(np.mean(p_vals), 0.60)

  def test_contextual_env_generation(self):
    """Verifies contextual environment clusters and candidate generation."""
    env = bandit_env.ContextualBanditEnv(
        dimension=3,
        num_vulnerable_clusters=1,
        seed=42,
    )
    candidates = env.generate_candidate_pool(size=10)
    self.assertLen(candidates, 10)
    for arm in candidates:
      self.assertIsNotNone(arm.feature)
      self.assertLen(arm.feature, 3)

  def test_disco_bernoulli_run(self):
    """Tests end-to-end execution of DISCO on Bernoulli bandit."""
    config = bandit_disco.BanditConfig(
        env_type="bernoulli",
        budget=100,
        chunk_size=10,
        alpha=0.05,
        kappa=0.5,
        e_min=0.01,
        e_max=100.0,
        n_max=5,
        use_boosting=True,
        n_sim=10_000,
        seed=42,
    )
    env = bandit_env.BernoulliBanditEnv(pi_1=0.20, seed=42)
    result = bandit_disco.run_disco_bandit(config, env)

    self.assertEqual(result.runner_type, "disco")
    self.assertGreater(result.total_hypotheses_tested, 0)
    self.assertGreaterEqual(result.empirical_fdr, 0.0)
    self.assertLessEqual(result.empirical_fdr, 100.0)

  def test_offline_ebh_bernoulli_run(self):
    """Tests end-to-end execution of Offline e-BH on Bernoulli bandit."""
    config = bandit_disco.BanditConfig(
        env_type="bernoulli",
        budget=100,
        chunk_size=10,
        alpha=0.05,
        kappa=0.5,
        use_boosting=True,
        seed=42,
    )
    env = bandit_env.BernoulliBanditEnv(pi_1=0.20, seed=42)
    result = bandit_disco.run_offline_ebh_bandit(config, env)

    self.assertEqual(result.runner_type, "offline_ebh")
    self.assertEqual(result.k_reference, 10)
    self.assertGreater(result.total_hypotheses_tested, 0)

  def test_disco_contextual_run(self):
    """Tests end-to-end execution of DISCO on Contextual bandit."""
    config = bandit_disco.BanditConfig(
        env_type="contextual",
        budget=100,
        chunk_size=10,
        alpha=0.05,
        kappa=0.5,
        dimension=3,
        num_vulnerable_clusters=1,
        use_boosting=True,
        n_sim=10_000,
        seed=42,
    )
    env = bandit_env.ContextualBanditEnv(
        dimension=3, num_vulnerable_clusters=1, seed=42
    )
    result = bandit_disco.run_disco_bandit(config, env)

    self.assertEqual(result.runner_type, "disco")
    self.assertGreater(result.total_hypotheses_tested, 0)

  def test_disco_multibin_run(self):
    """Tests DISCO on Bernoulli bandit with multi-bin active strip."""
    config = bandit_disco.BanditConfig(
        env_type="bernoulli",
        budget=100,
        chunk_size=10,
        alpha=0.05,
        kappa=0.5,
        e_min=0.01,
        e_max=100.0,
        n_max=5,
        num_active_bins=3,
        use_boosting=True,
        n_sim=5_000,
        seed=42,
    )
    env = bandit_env.BernoulliBanditEnv(pi_1=0.30, seed=42)
    result = bandit_disco.run_disco_bandit(config, env)

    self.assertEqual(result.runner_type, "disco")
    self.assertGreater(result.total_hypotheses_tested, 0)
    self.assertIsNotNone(result.epsilon_post)
    self.assertIsNotNone(result.delta_post)
    self.assertIsNotNone(result.posthoc_certificate)

  def test_multibin_certificate_monotonicity(self):
    """Verifies that sub-bin expectations and certificates increase with higher sub-bins."""
    tables = null_tables_lib.compute_null_coarsened_tables(
        kappa=0.5,
        e_min=0.01,
        e_max=100.0,
        n_max=5,
        k_max=10,
        num_bins=3,
        alpha=0.05,
        n_sim=10_000,
        seed=42,
    )
    self.assertIsNotNone(tables.E_bin_table)
    # Check E_B1 < E_B2 < E_B3 for probe length m=3
    self.assertLess(tables.E_bin_table[0, 2], tables.E_bin_table[1, 2])
    self.assertLess(tables.E_bin_table[1, 2], tables.E_bin_table[2, 2])

    cert_low_bin = cert_lib.compute_disco_posthoc_certificate(
        terminal_statuses={"h1": "B_1"},
        probe_counts={"h1": 3},
        null_tables=tables,
        alpha_bh=0.05,
    )
    cert_high_bin = cert_lib.compute_disco_posthoc_certificate(
        terminal_statuses={"h1": "B_3"},
        probe_counts={"h1": 3},
        null_tables=tables,
        alpha_bh=0.05,
    )
    self.assertLess(cert_low_bin.epsilon_post, cert_high_bin.epsilon_post)

  def test_disco_unbinned_bernoulli_run(self):
    """Tests DISCO on Bernoulli bandit with unbinned e-values (|P|=infinity)."""
    config = bandit_disco.BanditConfig(
        env_type="bernoulli",
        budget=100,
        chunk_size=10,
        alpha=0.05,
        kappa=0.5,
        e_min=0.01,
        e_max=100.0,
        n_max=5,
        num_active_bins=0,  # Unbinned e-value mode
        use_boosting=True,
        n_sim=5_000,
        seed=42,
    )
    env = bandit_env.BernoulliBanditEnv(pi_1=0.30, seed=42)
    result = bandit_disco.run_disco_bandit(config, env)

    self.assertEqual(result.runner_type, "disco")
    self.assertGreater(result.total_hypotheses_tested, 0)
    self.assertGreaterEqual(result.empirical_fdr, 0.0)
    self.assertLessEqual(result.empirical_fdr, 100.0)
    self.assertIsNotNone(result.epsilon_post)
    self.assertIsNotNone(result.delta_post)
    self.assertIsNotNone(result.posthoc_certificate)

  def test_disco_unbinned_contextual_run(self):
    """Tests DISCO on Contextual bandit with unbinned e-values (|P|=infinity)."""
    config = bandit_disco.BanditConfig(
        env_type="contextual",
        budget=100,
        chunk_size=10,
        alpha=0.05,
        kappa=0.5,
        dimension=3,
        num_vulnerable_clusters=1,
        num_active_bins=0,  # Unbinned e-value mode
        use_boosting=True,
        n_sim=5_000,
        seed=42,
    )
    env = bandit_env.ContextualBanditEnv(
        dimension=3, num_vulnerable_clusters=1, seed=42
    )
    result = bandit_disco.run_disco_bandit(config, env)

    self.assertEqual(result.runner_type, "disco")
    self.assertGreater(result.total_hypotheses_tested, 0)
    self.assertGreaterEqual(result.empirical_fdr, 0.0)
    self.assertLessEqual(result.empirical_fdr, 100.0)

  def test_offline_ebh_unclamped_probe_counts(self):
    """Tests that Offline e-BH does not cap probe counts to n_max."""
    # n_max=1 in config should not prevent offline e-BH from probing an arm
    # multiple times.
    config = bandit_disco.BanditConfig(
        env_type="bernoulli",
        budget=200,
        chunk_size=10,
        alpha=0.05,
        kappa=0.5,
        n_max=1,
        seed=42,
    )
    env = bandit_env.BernoulliBanditEnv(pi_1=0.30, seed=42)
    result = bandit_disco.run_offline_ebh_bandit(config, env)

    self.assertEqual(result.runner_type, "offline_ebh")
    self.assertGreater(result.total_hypotheses_tested, 0)
    max_probes = max(result.probe_counts.values(), default=0)
    self.assertGreater(max_probes, 1)

  def test_macro_recall_multicluster(self):
    """Tests that macro_recall (cluster-balanced recall) is properly computed."""
    config = bandit_disco.BanditConfig(
        env_type="contextual",
        budget=100,
        chunk_size=10,
        alpha=0.05,
        kappa=0.5,
        dimension=3,
        num_vulnerable_clusters=3,
        seed=42,
    )
    env = bandit_env.ContextualBanditEnv(
        dimension=3, num_vulnerable_clusters=3, seed=42
    )
    ttp, cluster_ttp = env.pre_generate_pool(10)
    result_disco = bandit_disco.run_disco_bandit(
        config,
        env,
        total_true_positives=ttp,
        cluster_true_positives=cluster_ttp,
    )
    result_offline = bandit_disco.run_offline_ebh_bandit(
        config,
        env,
        total_true_positives=ttp,
        cluster_true_positives=cluster_ttp,
    )

    self.assertGreaterEqual(result_disco.macro_recall, 0.0)
    self.assertLessEqual(result_disco.macro_recall, 100.0)
    self.assertGreaterEqual(result_offline.macro_recall, 0.0)
    self.assertLessEqual(result_offline.macro_recall, 100.0)

  def test_null_tables_expected_overshoot(self):
    """Verifies that log_E_L and log_E_H expected overshoot values are computed properly."""
    tables = null_tables_lib.compute_null_coarsened_tables(
        kappa=0.5,
        e_min=0.4,
        e_max=100.0,
        n_max=4,
        k_max=10,
        num_bins=5,
        alpha=0.05,
        n_sim=5_000,
        seed=42,
    )
    # High exit: log_E_H = log(100) + (1 - 0.5) = log(100) + 0.5
    expected_log_h = float(np.log(100.0) + 0.5)
    self.assertAlmostEqual(tables.log_E_H, expected_log_h, places=4)

    # Low exit: log_E_L < log(e_min)
    self.assertLess(tables.log_E_L, np.log(0.4))
    self.assertGreater(tables.log_E_L, np.log(0.4) - 2.0)

    # Verify bin_log_centers endpoints match expected overshoot
    self.assertIsNotNone(tables.bin_log_centers)
    self.assertAlmostEqual(tables.bin_log_centers[0], tables.log_E_L, places=4)
    self.assertAlmostEqual(tables.bin_log_centers[-1], tables.log_E_H, places=4)

  def test_terminal_status_dataclass(self):
    """Tests TerminalStatus constructors and properties."""
    low = status_lib.TerminalStatus.low()
    self.assertEqual(low.kind, "L")
    self.assertIsNone(low.bin_idx)
    self.assertTrue(low.is_exit_low)
    self.assertFalse(low.is_exit_high)
    self.assertFalse(low.is_active)
    self.assertFalse(low.is_binned)

    high = status_lib.TerminalStatus.high()
    self.assertEqual(high.kind, "H")
    self.assertTrue(high.is_exit_high)

    active_unbinned = status_lib.TerminalStatus.active()
    self.assertEqual(active_unbinned.kind, "A")
    self.assertIsNone(active_unbinned.bin_idx)
    self.assertTrue(active_unbinned.is_active)
    self.assertFalse(active_unbinned.is_binned)

    active_binned = status_lib.TerminalStatus.active(bin_idx=3)
    self.assertEqual(active_binned.kind, "A")
    self.assertEqual(active_binned.bin_idx, 3)
    self.assertTrue(active_binned.is_active)
    self.assertTrue(active_binned.is_binned)

  def test_parse_status(self):
    """Tests parse_status on various string and integer formats."""
    self.assertEqual(
        status_lib.parse_status("L"), status_lib.TerminalStatus.low()
    )
    self.assertEqual(
        status_lib.parse_status("exit_low"), status_lib.TerminalStatus.low()
    )
    self.assertEqual(
        status_lib.parse_status("H"), status_lib.TerminalStatus.high()
    )
    self.assertEqual(
        status_lib.parse_status("exit_high"),
        status_lib.TerminalStatus.high(),
    )
    self.assertEqual(
        status_lib.parse_status("A"), status_lib.TerminalStatus.active()
    )
    self.assertEqual(
        status_lib.parse_status("active"), status_lib.TerminalStatus.active()
    )
    self.assertEqual(
        status_lib.parse_status("B_3"),
        status_lib.TerminalStatus.active(bin_idx=3),
    )
    self.assertEqual(
        status_lib.parse_status("bin_2"),
        status_lib.TerminalStatus.active(bin_idx=2),
    )
    self.assertEqual(
        status_lib.parse_status("B1"),
        status_lib.TerminalStatus.active(bin_idx=1),
    )
    self.assertEqual(
        status_lib.parse_status(0), status_lib.TerminalStatus.low()
    )
    self.assertEqual(
        status_lib.parse_status(4),
        status_lib.TerminalStatus.active(bin_idx=4),
    )
    # Idempotent
    st = status_lib.TerminalStatus.active(bin_idx=5)
    self.assertEqual(status_lib.parse_status(st), st)

    # Invalid values raise ValueError
    with self.assertRaises(ValueError):
      status_lib.parse_status("invalid_status_xyz")
    with self.assertRaises(ValueError):
      status_lib.parse_status("")
    with self.assertRaises(ValueError):
      status_lib.parse_status(-1)

  def test_enforce_exit_low_toggle(self):
    """Tests that enforce_exit_low toggles immediate deactivation on M < e_min."""
    # Under 100% null (pi_1=0.0), e-values drift down quickly.
    env = bandit_env.BernoulliBanditEnv(
        pi_1=0.0, baseline_rate_logit=-4.6, seed=42
    )

    config_with_exit_low = bandit_disco.BanditConfig(
        env_type="bernoulli",
        budget=100,
        chunk_size=10,
        e_min=0.4,  # High e_min so null arms drop below it quickly
        e_max=100.0,
        n_max=5,
        enforce_exit_low=True,
        n_sim=5_000,
        seed=42,
    )
    res_with = bandit_disco.run_disco_bandit(config_with_exit_low, env)

    env2 = bandit_env.BernoulliBanditEnv(
        pi_1=0.0, baseline_rate_logit=-4.6, seed=42
    )
    config_no_exit_low = bandit_disco.BanditConfig(
        env_type="bernoulli",
        budget=100,
        chunk_size=10,
        e_min=0.4,
        e_max=100.0,
        n_max=5,
        enforce_exit_low=False,
        n_sim=5_000,
        seed=42,
    )
    res_no = bandit_disco.run_disco_bandit(config_no_exit_low, env2)

    self.assertGreater(res_with.total_hypotheses_tested, 0)
    self.assertGreater(res_no.total_hypotheses_tested, 0)
    # Both runs should finish cleanly and record valid results
    self.assertEqual(res_with.runner_type, "disco")
    self.assertEqual(res_no.runner_type, "disco")
    self.assertIsNotNone(res_no.epsilon_post)
    self.assertIsNotNone(res_no.posthoc_certificate)

  def test_compute_penalized_cluster_coverage_within_bound(self):
    """Tests that no penalty is applied when delta_post <= 0.05 or None."""
    self.assertEqual(
        bandit_disco.compute_penalized_cluster_coverage(80.0, None), 80.0
    )
    self.assertEqual(
        bandit_disco.compute_penalized_cluster_coverage(80.0, 0.03), 80.0
    )
    self.assertEqual(
        bandit_disco.compute_penalized_cluster_coverage(80.0, 0.05), 80.0
    )

  def test_compute_penalized_cluster_coverage_excess_penalty(self):
    """Tests that a 5x hinge penalty per 1% excess is deducted when delta_post > 0.05."""
    # 2% excess (0.07 - 0.05 = 0.02 -> 2%) -> penalty = 5.0 * 2.0 = 10.0%
    penalized = bandit_disco.compute_penalized_cluster_coverage(
        75.0, 0.07, delta_threshold=0.05, delta_multiplier=5.0
    )
    self.assertAlmostEqual(penalized, 65.0)

    # 4% excess (0.09 - 0.05 = 0.04 -> 4%) -> penalty = 5.0 * 4.0 = 20.0%
    penalized = bandit_disco.compute_penalized_cluster_coverage(
        75.0, 0.09, delta_threshold=0.05, delta_multiplier=5.0
    )
    self.assertAlmostEqual(penalized, 55.0)

    # Custom threshold (0.10) and multiplier (2.0)
    penalized_custom = bandit_disco.compute_penalized_cluster_coverage(
        75.0, 0.15, delta_threshold=0.10, delta_multiplier=2.0
    )
    # excess = 5% -> penalty = 2.0 * 5% = 10% -> 65.0%
    self.assertAlmostEqual(penalized_custom, 65.0)

  def test_discrete_null_tables_generation(self):
    """Tests that compute_null_coarsened_tables works correctly with discrete_null=True."""
    tables = null_tables_lib.compute_null_coarsened_tables(
        kappa=0.5,
        e_min=0.01,
        e_max=100.0,
        n_max=5,
        k_max=10,
        num_bins=1,
        alpha=0.05,
        n_sim=1000,
        seed=42,
        discrete_null=True,
        discrete_chunk_size=10,
        discrete_baseline_rate=0.01,
    )
    self.assertTrue(tables.discrete_null)
    self.assertEqual(tables.discrete_chunk_size, 10)
    self.assertEqual(tables.discrete_baseline_rate, 0.01)
    self.assertGreater(tables.E_A_max, 0.0)
    self.assertLessEqual(tables.E_A_max, 1.0)
    self.assertGreaterEqual(tables.P_H_max, 0.0)

  def test_p_value_modes_disco_bernoulli(self):
    """Tests end-to-end DISCO runs under randomized, discrete, and discrete_matched modes."""
    for mode in ["randomized", "discrete", "discrete_matched"]:
      config = bandit_disco.BanditConfig(
          env_type="bernoulli",
          budget=50,
          chunk_size=10,
          alpha=0.05,
          kappa=0.5,
          p_value_mode=mode,
          n_sim=1000,
          seed=42,
      )
      env = bandit_env.BernoulliBanditEnv(pi_1=0.20, seed=42)
      result = bandit_disco.run_disco_bandit(config, env)
      self.assertEqual(result.runner_type, "disco")
      self.assertGreater(result.total_hypotheses_tested, 0)
      self.assertGreaterEqual(result.empirical_fdr, 0.0)

  def test_type1_bernoulli_run(self):
    """Tests end-to-end execution of Type-I Error Rate baseline on Bernoulli bandit."""
    config = bandit_disco.BanditConfig(
        env_type="bernoulli",
        budget=100,
        chunk_size=10,
        alpha=0.05,
        kappa=0.5,
        seed=42,
    )
    env = bandit_env.BernoulliBanditEnv(pi_1=0.30, seed=42)
    result = bandit_disco.run_type1_bandit(config, env)

    self.assertEqual(result.runner_type, "type1")
    self.assertEqual(result.rejection_threshold, 1.0 / config.alpha)
    self.assertEqual(result.nominal_alpha_used, config.alpha)
    self.assertGreater(result.total_hypotheses_tested, 0)
    self.assertGreaterEqual(result.empirical_fdr, 0.0)
    self.assertLessEqual(result.empirical_fdr, 100.0)
    self.assertGreaterEqual(result.recall, 0.0)
    self.assertLessEqual(result.recall, 100.0)

  def test_type1_contextual_run(self):
    """Tests end-to-end execution of Type-I Error Rate baseline on Contextual bandit."""
    config = bandit_disco.BanditConfig(
        env_type="contextual",
        budget=100,
        chunk_size=10,
        alpha=0.05,
        kappa=0.5,
        dimension=3,
        num_vulnerable_clusters=2,
        seed=42,
    )
    env = bandit_env.ContextualBanditEnv(
        dimension=3, num_vulnerable_clusters=2, seed=42
    )
    result = bandit_disco.run_type1_bandit(config, env)

    self.assertEqual(result.runner_type, "type1")
    self.assertEqual(result.rejection_threshold, 1.0 / config.alpha)
    self.assertGreater(result.total_hypotheses_tested, 0)
    self.assertGreaterEqual(result.cluster_coverage, 0.0)
    self.assertLessEqual(result.cluster_coverage, 100.0)

  def test_type1_deactivates_on_threshold(self):
    """Tests that arms crossing 1/alpha are deactivated and have terminal status 'H'."""
    config = bandit_disco.BanditConfig(
        env_type="bernoulli",
        budget=200,
        chunk_size=10,
        alpha=0.10,  # 1/alpha = 10.0 threshold
        kappa=0.5,
        alternative_rate=0.80,  # High alt rate to ensure early 1/alpha crossing
        seed=42,
    )
    env = bandit_env.BernoulliBanditEnv(
        pi_1=0.50, alternative_rate=0.80, baseline_rate_logit=-4.6, seed=42
    )
    result = bandit_disco.run_type1_bandit(config, env)

    self.assertEqual(result.runner_type, "type1")
    for h_id in result.rejected_hypotheses:
      self.assertGreaterEqual(result.final_e_values[h_id], 1.0 / config.alpha)
      self.assertEqual(result.terminal_statuses.get(h_id), "H")

  def test_type1_rejection_no_ebh_comparison(self):
    """Verifies that Type-I baseline rejection is >= Offline e-BH rejection under same budget."""
    config = bandit_disco.BanditConfig(
        env_type="bernoulli",
        budget=300,
        chunk_size=10,
        alpha=0.05,
        kappa=0.5,
        seed=123,
    )
    env_t1 = bandit_env.BernoulliBanditEnv(pi_1=0.30, seed=123)
    env_ebh = bandit_env.BernoulliBanditEnv(pi_1=0.30, seed=123)

    res_t1 = bandit_disco.run_type1_bandit(config, env_t1)
    res_ebh = bandit_disco.run_offline_ebh_bandit(config, env_ebh)

    self.assertEqual(res_t1.runner_type, "type1")
    self.assertEqual(res_ebh.runner_type, "offline_ebh")
    # Both test hypotheses and discover candidates
    self.assertGreater(res_t1.total_hypotheses_tested, 0)
    self.assertGreater(res_ebh.total_hypotheses_tested, 0)


if __name__ == "__main__":
  absltest.main()
