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

"""Unit tests for DISCO bound value calculations and report generation."""

import json
from absl.testing import absltest
import numpy as np
from disco import compute_disco_bound_values


class ComputeDiscoBoundValuesTest(absltest.TestCase):

  def test_pareto_conditional_mean_harmonic(self):
    # When kappa = 0.5, the conditional mean of Pareto tail in [a, b]
    # is exactly the harmonic mean: 2 * a * b / (a + b).
    # For [100, 1000], 2 * 100 * 1000 / 1100 = 200000 / 1100 = 181.818...
    mean_val = compute_disco_bound_values.pareto_conditional_mean(
        100.0, 1000.0, kappa=0.5
    )
    self.assertAlmostEqual(mean_val, 181.8181818, places=4)

  def test_compute_base_walk_stats(self):
    kappa = 0.5
    e_max = 100.0
    n_max = 5
    n_sim = 10_000

    rng = np.random.default_rng(42)
    p = rng.uniform(0.0, 1.0, size=(n_sim, n_max))
    log_e = np.log(kappa) + (kappa - 1.0) * np.log(p)
    log_m_all = np.cumsum(log_e, axis=1)

    stats = compute_disco_bound_values.compute_base_walk_stats(
        log_m_all=log_m_all,
        kappa=kappa,
        e_max=e_max,
        n_max=n_max,
        budget_probes=50,
        k_bar=20,
    )

    # 1. E_H = e_max / kappa
    self.assertAlmostEqual(stats["E_H"], 200.0, places=4)

    # 2. P_H(m) is non-decreasing with m
    p_h = stats["P_H_table"]
    for i in range(len(p_h) - 1):
      self.assertLessEqual(p_h[i], p_h[i + 1])

    # 3. E_A <= 1.0 strictly (martingale property for survivors)
    self.assertLessEqual(stats["E_A"], 1.0)
    self.assertGreater(stats["E_A"], 0.0)

    # 4. Baseline exit-event bound K_bar * P_H
    self.assertAlmostEqual(
        stats["baseline_K_PH"], 20 * stats["P_H"], places=9
    )

  def test_multibin_tradeoff(self):
    e_max = 1000.0
    kappa = 0.5
    n_max = 10

    mean_j2, _, _ = compute_disco_bound_values.compute_multibin_stats(
        kappa, e_max, n_max, num_bins=2
    )
    mean_j3, _, _ = compute_disco_bound_values.compute_multibin_stats(
        kappa, e_max, n_max, num_bins=3
    )

    self.assertGreater(mean_j3, mean_j2)
    self.assertGreater(mean_j2, 1.0)
    self.assertAlmostEqual(mean_j3, 181.8181818, places=4)

  def test_multibin_with_emin(self):
    e_min = 1.0
    e_max = 1000.0
    kappa = 0.5
    n_max = 5

    mean_val, bin_edges, _ = compute_disco_bound_values.compute_multibin_stats(
        kappa, e_max, n_max, num_bins=2, e_min=e_min
    )
    # With num_bins=2 and e_min=1.0: bin 1 is [0, 1.0), bin 2 is [1.0, 1000.0)
    np.testing.assert_allclose(bin_edges, [0.0, 1.0, 1000.0])
    # Top sub-bin [1.0, 1000.0] harmonic mean = 2 * 1 * 1000 / 1001 =
    # 1.998001998...
    self.assertAlmostEqual(mean_val, 2000.0 / 1001.0, places=4)

  def test_generate_reports_all_modes(self):
    # Construct a sample record
    rec = {
        "kappa": 0.5,
        "e_max": 1000.0,
        "n_max": 3,
        "num_bins": 1,
        "bin_edges": np.array([0.0, 1000.0]),
        "E_H": 2000.0,
        "P_H_table": np.array([0.0, 0.0, 2e-6]),
        "P_surv_table": np.array([1.0, 1.0, 0.999998]),
        "P_H": 2e-6,
        "E_A_table": np.array([0.998, 0.995, 0.989]),
        "E_A_formula_table": np.array([1.0, 1.0, 0.996]),
        "E_A": 1.0,
        "E_bin_max": 1.0,
        "baseline_K_PH": 6e-5,
        "prehoc_baseline": 0.05006,
        "calibrated_alpha_baseline": 0.04994,
        "boost_factors": np.array([6.408, 4.654, 3.795]),
        "phi_analytical_ceiling": np.array([0.05, 0.05, 0.0500001]),
    }
    records = [rec]

    # Test summary, detailed, multibin, latex
    for mode in ("all", "summary", "detailed", "multibin", "latex"):
      out = compute_disco_bound_values.generate_text_reports(
          records,
          alpha=0.05,
          budget_raw=200,
          budget_probes=200,
          chunk_size=1,
          k_bar=30,
          mode=mode,
      )
      self.assertIsInstance(out, str)
      self.assertNotEmpty(out)

    # Test JSON mode
    json_str = compute_disco_bound_values.generate_text_reports(
        records,
        alpha=0.05,
        budget_raw=200,
        budget_probes=200,
        chunk_size=1,
        k_bar=30,
        mode="json",
    )
    parsed = json.loads(json_str)
    self.assertIn("metadata", parsed)
    self.assertIn("records", parsed)
    self.assertLen(parsed["records"], 1)
    self.assertEqual(parsed["records"][0]["kappa"], 0.5)

  def test_parse_tuples(self):
    # 1. Parenthesized tuples with 3, 4, 5 elements
    s1 = "(0.5, 100, 5), (0.5, 500, 10, 2), (0.1741, 1000, 15, 3, 1.0)"
    t1 = compute_disco_bound_values.parse_tuples(s1)
    self.assertLen(t1, 3)
    self.assertEqual(t1[0].kappa, 0.5)
    self.assertEqual(t1[0].e_max, 100.0)
    self.assertEqual(t1[0].n_max, 5)
    self.assertEqual(t1[0].num_bins, 1)
    self.assertFalse(t1[0].num_bins_explicit)

    self.assertEqual(t1[1].num_bins, 2)
    self.assertTrue(t1[1].num_bins_explicit)

    self.assertEqual(t1[2].num_bins, 3)
    self.assertEqual(t1[2].e_min, 1.0)
    self.assertTrue(t1[2].num_bins_explicit)

    # 2. Semicolon separated
    s2 = "0.5, 100, 5; 0.1741, 500, 10, 2"
    t2 = compute_disco_bound_values.parse_tuples(s2)
    self.assertLen(t2, 2)
    self.assertEqual(t2[0].kappa, 0.5)
    self.assertEqual(t2[1].kappa, 0.1741)
    self.assertEqual(t2[1].num_bins, 2)

    # 3. JSON array format
    s3 = "[[0.5, 100, 5], [0.5, 500, 10, 2]]"
    t3 = compute_disco_bound_values.parse_tuples(s3)
    self.assertLen(t3, 2)
    self.assertEqual(t3[0].e_max, 100.0)
    self.assertEqual(t3[1].num_bins, 2)

  def test_compute_bounds_for_tuples(self):
    tuples = [
        compute_disco_bound_values.ParamTuple(
            kappa=0.5, e_max=100.0, n_max=3, num_bins=1
        ),
        compute_disco_bound_values.ParamTuple(
            kappa=0.5, e_max=200.0, n_max=5, num_bins=2
        ),
    ]
    records = compute_disco_bound_values.compute_bounds_for_tuples(
        tuples, alpha=0.05, budget_raw=30, chunk_size=1, n_sim=1000, seed=42
    )
    self.assertLen(records, 2)
    self.assertEqual(records[0]["e_max"], 100.0)
    self.assertEqual(records[1]["e_max"], 200.0)
    self.assertIn("prehoc_baseline", records[0])

  def test_latex_table_format(self):
    rec = {
        "kappa": 0.5,
        "e_max": 100.0,
        "n_max": 5,
        "num_bins": 1,
        "num_bins_explicit": False,
        "bin_edges": np.array([0.0, 100.0]),
        "E_H": 200.0,
        "P_H_table": np.array([0.0, 0.0, 1e-4, 2e-4, 3e-4]),
        "P_surv_table": np.array([1.0, 1.0, 0.9999, 0.9998, 0.9997]),
        "P_H": 3e-4,
        "E_A_table": np.array([1.0, 1.0, 0.99, 0.98, 0.97]),
        "E_A_formula_table": np.array([1.0, 1.0, 0.99, 0.98, 0.97]),
        "E_A": 1.0,
        "E_bin_max": 1.0,
        "baseline_K_PH": 9e-3,
        "prehoc_baseline": 0.059,
        "calibrated_alpha_baseline": 0.041,
        "boost_factors": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        "phi_analytical_ceiling": np.array([0.05, 0.05, 0.05, 0.05, 0.05]),
    }
    records = [rec]
    out = compute_disco_bound_values.generate_text_reports(
        records,
        alpha=0.05,
        budget_raw=30,
        budget_probes=30,
        chunk_size=1,
        k_bar=30,
        mode="latex",
        is_tuple_mode=True,
    )

    # 1. Header checks
    self.assertIn(r"$|\mathcal{P}|$", out)
    self.assertIn(r"$\mathrm{Bound}$ & $\alpha^\star$", out)

    # 2. Columns removed checks
    self.assertNotIn(r"\mathcal{V}^*", out)
    self.assertNotIn(r"\mathcal{V}^\star", out)
    self.assertNotIn(r"\bar{K} P_H", out)

    # 3. Caption and label at bottom
    expected_caption = (
        "\\caption{\n"
        r"Typical values for \textsc{DISCO} pre-hoc FDR bound and"
        r" $\alpha^\star$ ($\alpha=0.05$, $B=30$, $\bar{K}=30$)}"
    )
    self.assertIn(expected_caption, out)
    self.assertIn(r"\label{tab:disco_bound_typical_values}", out)

    # Verify caption is AFTER \end{tabular}
    end_tab_pos = out.find(r"\end{tabular}")
    caption_pos = out.find(r"\caption{")
    self.assertGreater(caption_pos, end_tab_pos)


if __name__ == "__main__":
  absltest.main()
