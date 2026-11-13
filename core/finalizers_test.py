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

"""Characterization tests for core/finalizers.

Each finalizer is compared against a frozen copy of the finalization code that
lived in `DiscoRunner.run_loop` before it was extracted, across modes,
calibration modes, boosting, bin settings and edge cases. Results, logs and
`fdr_bounds.json` must match exactly.
"""

import copy
import dataclasses
import json
import os
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from disco.core import checkpoint as checkpoint_lib
from disco.core import config as config_lib
from disco.core import finalizers
from disco.core import records as records_lib
from disco.stats import boosting
from disco.stats import multiple_testing
from disco.stats.coarsened import calibration as calibration_lib
from disco.stats.coarsened import certificates as cert_lib
from disco.stats.coarsened import null_tables as null_tables_lib
from disco.stats.coarsened import status as status_lib


class _LegacyRunner:
  """Holds the attributes the legacy finalization code reads from `self`."""

  def __init__(
      self,
      config: config_lib.DiscoConfig,
      state: checkpoint_lib.RunState,
      null_tables: null_tables_lib.NullCoarsenedTables | None,
      effective_bins: int,
      bin_edges: np.ndarray,
      unbinned_e_values: bool,
  ):
    self.config = config
    self.null_tables = null_tables
    self.effective_bins = effective_bins
    self.bin_edges = bin_edges
    self.unbinned_e_values = unbinned_e_values
    self.hypothesis_groups = state.hypothesis_groups
    self.running_e_values = state.running_e_values
    self.final_e_values = state.final_e_values
    self.probe_counts = state.probe_counts
    self.terminal_statuses = state.terminal_statuses

  def legacy_finalize(self, log) -> records_lib.DiscoResult:
    """Frozen copy of the pre-extraction run_loop finalization."""
    config = self.config
    # === FINALIZATION ===
    if config.baseline_type1_mode:
      log("\n=== DISCO Type-I Error Baseline Finalization ===")
      for h_id, curr_e in list(self.running_e_values.items()):
        self.final_e_values[h_id] = curr_e
        self.terminal_statuses[h_id] = "A"

      threshold = 1.0 / config.alpha
      k_total = len(self.final_e_values)
      log(f"Total hypotheses tested (K_obs): {k_total}")
      log(f"Type-I rejection threshold (1/alpha): {threshold:.4f}")

      # No boosting for baseline.
      boosted_e_values = dict(self.final_e_values)
      boost_factors = {h_id: 1.0 for h_id in self.final_e_values}

      rejected_set = {
          h_id
          for h_id, e_val in self.final_e_values.items()
          if e_val >= threshold
      }
      log(f"  Rejected (E >= 1/alpha): {len(rejected_set)} hypotheses")

      # Final summary logging.
      log("\n=== Rejected Hypotheses (Type-I Error Baseline) ===")
      for h_id in rejected_set:
        h_data = self.hypothesis_groups[h_id]
        records_lib.log_hypothesis_details(
            h_id,
            h_data,
            self.final_e_values[h_id],
            log,
        )

      log("\n=== Non-rejected Hypotheses ===")
      for h_id in self.final_e_values:
        if h_id not in rejected_set:
          h_data = self.hypothesis_groups[h_id]
          records_lib.log_hypothesis_details(
              h_id,
              h_data,
              self.final_e_values.get(h_id, 0.0),
              log,
          )

      bounds_report_dict = {
          "nominal_calibrated_alpha": float(config.alpha),
          "target_alpha": float(config.alpha),
          "kappa": float(config.kappa),
          "baseline_mode": True,
          "baseline_type1_mode": True,
          "k_obs": int(k_total),
          "rejection_threshold": float(threshold),
      }

      # Run alpha sweep for Type-I baseline: threshold at 1/alpha for each
      # alpha.
      sweep_alphas = [
          float(x)
          for x in (
              np.linspace(
                  np.sqrt(config.alpha_sweep_min),
                  np.sqrt(config.alpha_sweep_max),
                  config.alpha_sweep_steps,
              )
              ** 2
          )
      ]
      if config.alpha not in sweep_alphas:
        sweep_alphas.append(config.alpha)
        sweep_alphas.sort()

      alpha_sweep_records = []
      for a in sweep_alphas:
        thresh_a = 1.0 / a if a > 0 else float("inf")
        rej = {
            h_id
            for h_id, e_val in self.final_e_values.items()
            if e_val >= thresh_a
        }
        alpha_sweep_records.append({
            "alpha_target": float(a),
            "nominal_alpha": float(a),
            "num_discoveries": int(len(rej)),
            "rejected_hypotheses": sorted(list(rej)),
            "rejection_threshold": float(thresh_a),
        })

      fdr_bounds_data = {
          "params": {
              "target_model": config.target_model,
              "seed": config.seed_instruction,
              "budget": config.budget,
              "agent_model": config.agent_model,
              "output_dir": config.output_dir,
              "name": config.experiment_name,
              "alpha": config.alpha,
              "kappa": config.kappa,
              "chunk_size": config.chunk_size,
              "baseline_mode": True,
              "baseline_type1_mode": True,
              "alpha_sweep_min": config.alpha_sweep_min,
              "alpha_sweep_max": config.alpha_sweep_max,
              "alpha_sweep_steps": config.alpha_sweep_steps,
          },
          "bounds_report": bounds_report_dict,
          "alpha_sweep": alpha_sweep_records,
          "hypotheses_data": {
              "probe_counts": {
                  str(k): int(v) for k, v in self.probe_counts.items()
              },
              "final_e_values": {
                  str(k): float(v) for k, v in self.final_e_values.items()
              },
              "boosted_e_values": {
                  str(k): float(v) for k, v in boosted_e_values.items()
              },
              "boost_factors": {
                  str(k): float(v) for k, v in boost_factors.items()
              },
              "terminal_statuses": {
                  str(k): str(v) for k, v in self.terminal_statuses.items()
              },
          },
      }

      output_dir = os.path.join(config.output_dir, config.experiment_name)
      os.makedirs(output_dir, exist_ok=True)
      bounds_path = os.path.join(output_dir, "fdr_bounds.json")
      with open(bounds_path, "w", encoding="utf-8") as f:
        json.dump(fdr_bounds_data, f, indent=2)
      log(f"Saved FDR bounds report to: {bounds_path}")

      return records_lib.DiscoResult(
          all_records=self.hypothesis_groups,
          rejected_hypotheses=rejected_set,
          final_e_values=self.final_e_values,
          boosted_e_values=boosted_e_values,
          boost_factors=boost_factors,
          probe_counts=self.probe_counts,
          terminal_statuses=self.terminal_statuses,
          total_hypotheses_k=k_total,
          rejection_threshold=threshold,
          nominal_alpha_used=config.alpha,
          bounds_report=bounds_report_dict,
          alpha_sweep_results=alpha_sweep_records,
      )

    if config.baseline_mode:
      log("\n=== DISCO Baseline Finalization ===")
      for h_id, curr_e in list(self.running_e_values.items()):
        self.final_e_values[h_id] = curr_e
        self.terminal_statuses[h_id] = "A"

      k_bar = max(1, config.budget // config.chunk_size)
      k_total = len(self.final_e_values)
      log(f"Total hypotheses tested (K_obs): {k_total}")
      log(f"Baseline K_bar for e-BH: {k_bar}")

      # No boosting for baseline.
      boosted_e_values = dict(self.final_e_values)
      boost_factors = {h_id: 1.0 for h_id in self.final_e_values}

      rejected_set, rejection_thresh = multiple_testing.e_bh_rejection(
          self.final_e_values, config.alpha, k_total_override=k_bar
      )
      log(f"  Rejection threshold: {rejection_thresh:.4f}")
      log(f"  Rejected: {len(rejected_set)} hypotheses (out of K_bar={k_bar})")

      # Final summary logging.
      log("\n=== Rejected Hypotheses (Baseline e-BH) ===")
      for h_id in rejected_set:
        h_data = self.hypothesis_groups[h_id]
        records_lib.log_hypothesis_details(
            h_id,
            h_data,
            self.final_e_values[h_id],
            log,
        )

      log("\n=== Non-rejected Hypotheses ===")
      for h_id in self.final_e_values:
        if h_id not in rejected_set:
          h_data = self.hypothesis_groups[h_id]
          records_lib.log_hypothesis_details(
              h_id,
              h_data,
              self.final_e_values.get(h_id, 0.0),
              log,
          )

      bounds_report_dict = {
          "nominal_calibrated_alpha": float(config.alpha),
          "target_alpha": float(config.alpha),
          "kappa": float(config.kappa),
          "baseline_mode": True,
          "k_bar": int(k_bar),
          "k_obs": int(k_total),
      }

      # Run alpha sweep for baseline.
      sweep_alphas = [
          float(x)
          for x in (
              np.linspace(
                  np.sqrt(config.alpha_sweep_min),
                  np.sqrt(config.alpha_sweep_max),
                  config.alpha_sweep_steps,
              )
              ** 2
          )
      ]
      if config.alpha not in sweep_alphas:
        sweep_alphas.append(config.alpha)
        sweep_alphas.sort()

      alpha_sweep_records = []
      for a in sweep_alphas:
        rej, thresh = multiple_testing.e_bh_rejection(
            self.final_e_values, a, k_total_override=k_bar
        )
        alpha_sweep_records.append({
            "alpha_target": float(a),
            "nominal_alpha": float(a),
            "num_discoveries": int(len(rej)),
            "rejected_hypotheses": sorted(list(rej)),
            "rejection_threshold": float(thresh),
        })

      fdr_bounds_data = {
          "params": {
              "target_model": config.target_model,
              "seed": config.seed_instruction,
              "budget": config.budget,
              "agent_model": config.agent_model,
              "output_dir": config.output_dir,
              "name": config.experiment_name,
              "alpha": config.alpha,
              "kappa": config.kappa,
              "chunk_size": config.chunk_size,
              "baseline_mode": True,
              "baseline_k_bar": k_bar,
              "alpha_sweep_min": config.alpha_sweep_min,
              "alpha_sweep_max": config.alpha_sweep_max,
              "alpha_sweep_steps": config.alpha_sweep_steps,
          },
          "bounds_report": bounds_report_dict,
          "alpha_sweep": alpha_sweep_records,
          "hypotheses_data": {
              "probe_counts": {
                  str(k): int(v) for k, v in self.probe_counts.items()
              },
              "final_e_values": {
                  str(k): float(v) for k, v in self.final_e_values.items()
              },
              "boosted_e_values": {
                  str(k): float(v) for k, v in boosted_e_values.items()
              },
              "boost_factors": {
                  str(k): float(v) for k, v in boost_factors.items()
              },
              "terminal_statuses": {
                  str(k): str(v) for k, v in self.terminal_statuses.items()
              },
          },
      }

      output_dir = os.path.join(config.output_dir, config.experiment_name)
      os.makedirs(output_dir, exist_ok=True)
      bounds_path = os.path.join(output_dir, "fdr_bounds.json")
      with open(bounds_path, "w", encoding="utf-8") as f:
        json.dump(fdr_bounds_data, f, indent=2)
      log(f"Saved FDR bounds report to: {bounds_path}")

      return records_lib.DiscoResult(
          all_records=self.hypothesis_groups,
          rejected_hypotheses=rejected_set,
          final_e_values=self.final_e_values,
          boosted_e_values=boosted_e_values,
          boost_factors=boost_factors,
          probe_counts=self.probe_counts,
          terminal_statuses=self.terminal_statuses,
          total_hypotheses_k=k_bar,
          rejection_threshold=rejection_thresh,
          nominal_alpha_used=config.alpha,
          bounds_report=bounds_report_dict,
          alpha_sweep_results=alpha_sweep_records,
      )

    log("\n=== DISCO Finalization ===")
    assert self.null_tables is not None
    null_tables = self.null_tables

    # Finalize survivors.
    for h_id, curr_e in list(self.running_e_values.items()):
      self.final_e_values[h_id] = curr_e
      if curr_e < config.e_min:
        self.terminal_statuses[h_id] = "L"
      elif curr_e >= config.e_max:
        self.terminal_statuses[h_id] = "H"
      elif self.effective_bins <= 1:
        self.terminal_statuses[h_id] = "A"
      else:
        b_idx = int(np.searchsorted(self.bin_edges, curr_e, side="right"))
        b_val = max(1, min(b_idx, self.effective_bins))
        self.terminal_statuses[h_id] = f"B_{b_val}"

    k_total = len(self.final_e_values)
    log(f"Total hypotheses K: {k_total}")

    if k_total == 0:
      log("\n[Warning] No hypotheses were tested. Returning empty result.")
      return records_lib.DiscoResult(
          all_records=self.hypothesis_groups,
          rejected_hypotheses=set(),
          final_e_values={},
          boosted_e_values={},
          boost_factors={},
          probe_counts=self.probe_counts,
          terminal_statuses=self.terminal_statuses,
          total_hypotheses_k=0,
          rejection_threshold=0.0,
          nominal_alpha_used=0.0,
      )

    # Delta_post = K_obs * P_H(n_max).
    delta_post = cert_lib.compute_delta_post(
        (self.probe_counts.get(h, 1) for h in self.final_e_values),
        null_tables.P_H_table,
        null_tables.n_max,
    )
    log(f"  delta_post: {delta_post:.6f}")

    # E_bar_obs for alpha calibration.
    statuses_list = [
        self.terminal_statuses.get(h, "A") for h in self.final_e_values
    ]
    counts_list = [self.probe_counts.get(h, 1) for h in self.final_e_values]
    k_bar = max(1, config.budget // config.chunk_size)

    if self.unbinned_e_values:
      e_mean_max = config.e_max
    elif self.effective_bins > 1:
      e_mean_max = null_tables.E_bin_max
    else:
      e_mean_max = null_tables.E_A_max

    # Alpha calibration.
    log(f"\n--- Alpha Calibration (mode={config.alpha_calibration_mode}) ---")

    e_sum = sum(
        null_tables_lib.get_expected_null_e(
            self.terminal_statuses.get(h, "A"),
            self.probe_counts.get(h, 1),
            null_tables,
            exact_e=(
                self.final_e_values.get(h, 1.0)
                if self.unbinned_e_values
                else None
            ),
        )
        for h in self.terminal_statuses
        if self.terminal_statuses.get(h) != "H"
    )
    e_bar_obs = e_sum / max(1, k_total)

    if not config.use_boosting:
      if config.alpha_calibration_mode == "nominal":
        nominal_alpha = config.alpha
      elif config.alpha_calibration_mode == "pre_hoc":
        prehoc_bad_event = float(k_bar * null_tables.P_H_max)
        nominal_alpha = float(
            np.clip(
                (config.alpha - prehoc_bad_event) / max(1e-10, e_mean_max),
                0.0,
                1.0,
            )
        )
      elif config.alpha_calibration_mode == "post_hoc":
        nominal_alpha = float(
            np.clip(
                (config.alpha - delta_post) / max(1e-10, e_bar_obs),
                0.0,
                1.0,
            )
        )
      elif config.alpha_calibration_mode == "good_event":
        nominal_alpha = float(
            np.clip(config.alpha / max(1e-10, e_bar_obs), 0.0, 1.0)
        )
      else:
        nominal_alpha = config.alpha
    else:
      if config.alpha_calibration_mode == "nominal":
        nominal_alpha = config.alpha
      elif config.alpha_calibration_mode == "pre_hoc":
        nominal_alpha, _ = calibration_lib.find_optimal_prehoc_alpha_bh(
            alpha_star=config.alpha,
            kappa=config.kappa,
            k_bar=k_bar,
            null_tables=null_tables,
        )
      elif config.alpha_calibration_mode in ("good_event", "post_hoc"):
        nominal_alpha, _ = calibration_lib.find_optimal_alpha_bh(
            alpha_star=config.alpha,
            kappa=config.kappa,
            statuses=statuses_list,
            counts=counts_list,
            k_obs=k_total,
            null_tables=null_tables,
            delta_post=delta_post,
            mode=config.alpha_calibration_mode,
            final_e_values=(
                self.final_e_values if self.unbinned_e_values else None
            ),
            terminal_statuses_dict=(
                self.terminal_statuses if self.unbinned_e_values else None
            ),
            probe_counts_dict=(
                self.probe_counts if self.unbinned_e_values else None
            ),
            n_max=config.n_max if self.unbinned_e_values else None,
        )
      else:
        nominal_alpha = config.alpha

    log(f"  Calibrated alpha_BH: {nominal_alpha:.6f}")

    # Pre-hoc FDR bounds.
    alpha_for_prehoc = (
        nominal_alpha
        if config.alpha_calibration_mode == "pre_hoc"
        else config.alpha
    )
    prehoc_bounds = cert_lib.compute_prehoc_fdr_bounds(
        null_tables,
        k_bar,
        alpha_for_prehoc,
    )
    log("\n--- Pre-Hoc Bounds (Stated Before Running) ---")
    log(f"  Pre-Hoc Unboosted:        {prehoc_bounds['unboosted']:.6f}")
    log(f"  Pre-Hoc Boosted:          {prehoc_bounds['boosted']:.6f}")
    log(
        f"  Pre-Hoc Ceiling:          {prehoc_bounds['analytical_ceiling']:.6f}"
    )

    # Boosting and e-BH.
    log("\n--- Boosting + e-BH ---")
    boosted_e_values = {}
    boost_factors = {}
    for h_id, e_val in self.final_e_values.items():
      m = min(max(1, self.probe_counts.get(h_id, 1)), config.n_max)
      if config.use_boosting and nominal_alpha > 0:
        b = boosting.compute_boost_factor(
            nominal_alpha, config.kappa, m, k_total, use_exact_t=True
        )
      else:
        b = 1.0
      boosted_e_values[h_id] = float(b * e_val)
      boost_factors[h_id] = float(b)

    rejected_set, rejection_thresh = multiple_testing.e_bh_rejection(
        boosted_e_values, nominal_alpha
    )
    log(f"  Rejection threshold: {rejection_thresh:.4f}")
    log(f"  Rejected: {len(rejected_set)} hypotheses")

    # Post-hoc DISCO certificate.
    log("\n--- Post-hoc DISCO Certificate ---")
    cert = cert_lib.compute_disco_posthoc_certificate(
        self.terminal_statuses,
        self.probe_counts,
        null_tables,
        nominal_alpha,
        final_e_values=(
            self.final_e_values if self.unbinned_e_values else None
        ),
        use_boosting=config.use_boosting,
        kappa=config.kappa,
        n_max_override=config.n_max,
    )
    log(f"  epsilon_post (unboosted): {cert.epsilon_post:.6f}")
    log(f"  epsilon_post (boosted):   {cert.epsilon_post_boost:.6f}")
    log(f"  delta_post:               {cert.delta_post:.6f}")
    log(f"  Post-hoc cert (unboosted): {cert.posthoc_unboosted:.6f}")
    log(f"  Post-hoc cert (boosted):   {cert.posthoc_boosted:.6f}")
    log(f"  Confidence (1 - delta):    {cert.confidence:.6f}")

    # Final summary.
    log("\n=== Rejected Hypotheses (DISCO FDR-controlled) ===")
    for h_id in rejected_set:
      h_data = self.hypothesis_groups[h_id]
      records_lib.log_hypothesis_details(
          h_id,
          h_data,
          boosted_e_values[h_id],
          log,
          unboosted_e=self.final_e_values.get(h_id),
      )

    log("\n=== Non-rejected Hypotheses ===")
    for h_id in self.final_e_values:
      if h_id not in rejected_set:
        h_data = self.hypothesis_groups[h_id]
        records_lib.log_hypothesis_details(
            h_id,
            h_data,
            boosted_e_values.get(h_id, 0.0),
            log,
            unboosted_e=self.final_e_values.get(h_id),
        )
    bounds_report_dict = {
        "nominal_calibrated_alpha": float(nominal_alpha),
        "target_alpha": float(config.alpha),
        "kappa": float(config.kappa),
        "prehoc_unboosted": float(prehoc_bounds["unboosted"]),
        "prehoc_boosted": float(prehoc_bounds["boosted"]),
        "prehoc_analytical_ceiling": float(prehoc_bounds["analytical_ceiling"]),
        "posthoc_epsilon": float(cert.epsilon_post),
        "posthoc_delta": float(cert.delta_post),
        "posthoc_marginal": float(cert.posthoc_unboosted),
        "posthoc_epsilon_boost": float(cert.epsilon_post_boost),
        "posthoc_marginal_boost": float(cert.posthoc_boosted),
        "confidence": float(cert.confidence),
    }

    # Run alpha sweep
    sweep_alphas = [
        float(x)
        for x in (
            np.linspace(
                np.sqrt(config.alpha_sweep_min),
                np.sqrt(config.alpha_sweep_max),
                config.alpha_sweep_steps,
            )
            ** 2
        )
    ]
    if config.alpha not in sweep_alphas:
      sweep_alphas.append(config.alpha)
      sweep_alphas.sort()

    sweep_entries = calibration_lib.compute_disco_alpha_sweep(
        final_e_values=self.final_e_values,
        terminal_statuses=self.terminal_statuses,
        probe_counts=self.probe_counts,
        null_tables=null_tables,
        sweep_alphas=sweep_alphas,
        alpha_calibration_mode=config.alpha_calibration_mode,
        use_boosting=config.use_boosting,
        kappa=config.kappa,
        continuous=self.unbinned_e_values,
        n_max=config.n_max,
    )

    alpha_sweep_records = []
    for entry in sweep_entries:
      alpha_sweep_records.append({
          "alpha_target": float(entry.alpha_target),
          "nominal_alpha": float(entry.nominal_alpha),
          "num_discoveries": int(entry.num_discoveries),
          "rejected_hypotheses": sorted(list(entry.rejected_hypotheses)),
          "rejection_threshold": float(entry.rejection_threshold),
          "epsilon_post": float(
              entry.epsilon_post_boost
              if config.use_boosting
              else entry.epsilon_post
          ),
          "marginal_posthoc": float(
              entry.posthoc_boosted
              if config.use_boosting
              else entry.posthoc_unboosted
          ),
          "delta_post": float(entry.delta_post),
          "confidence": float(entry.confidence),
      })

    fdr_bounds_data = {
        "params": {
            "target_model": config.target_model,
            "seed": config.seed_instruction,
            "budget": config.budget,
            "agent_model": config.agent_model,
            "output_dir": config.output_dir,
            "name": config.experiment_name,
            "alpha": config.alpha,
            "kappa": config.kappa,
            "e_min": config.e_min,
            "e_max": config.e_max,
            "use_boosting": config.use_boosting,
            "chunk_size": config.chunk_size,
            "alpha_calibration_mode": config.alpha_calibration_mode,
            "n_max": config.n_max,
            "num_active_bins": config.num_active_bins,
            "alpha_sweep_min": config.alpha_sweep_min,
            "alpha_sweep_max": config.alpha_sweep_max,
            "alpha_sweep_steps": config.alpha_sweep_steps,
        },
        "bounds_report": bounds_report_dict,
        "alpha_sweep": alpha_sweep_records,
        "hypotheses_data": {
            "probe_counts": {
                str(k): int(v) for k, v in self.probe_counts.items()
            },
            "final_e_values": {
                str(k): float(v) for k, v in self.final_e_values.items()
            },
            "boosted_e_values": {
                str(k): float(v) for k, v in boosted_e_values.items()
            },
            "boost_factors": {
                str(k): float(v) for k, v in boost_factors.items()
            },
            "terminal_statuses": {
                str(k): str(v) for k, v in self.terminal_statuses.items()
            },
        },
    }

    output_dir = os.path.join(config.output_dir, config.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    bounds_path = os.path.join(output_dir, "fdr_bounds.json")
    with open(bounds_path, "w", encoding="utf-8") as f:
      json.dump(fdr_bounds_data, f, indent=2)
    log(f"Saved FDR bounds report to: {bounds_path}")

    return records_lib.DiscoResult(
        all_records=self.hypothesis_groups,
        rejected_hypotheses=rejected_set,
        final_e_values=self.final_e_values,
        boosted_e_values=boosted_e_values,
        boost_factors=boost_factors,
        probe_counts=self.probe_counts,
        terminal_statuses=self.terminal_statuses,
        total_hypotheses_k=k_total,
        rejection_threshold=rejection_thresh,
        nominal_alpha_used=nominal_alpha,
        epsilon_post=cert.epsilon_post,
        delta_post=cert.delta_post,
        posthoc_certificate=cert.posthoc_unboosted,
        confidence=cert.confidence,
        epsilon_post_boost=cert.epsilon_post_boost,
        posthoc_certificate_boost=cert.posthoc_boosted,
        prehoc_bounds=prehoc_bounds,
        bounds_report=bounds_report_dict,
        alpha_sweep_results=alpha_sweep_records,
    )


_TABLE_CACHE: dict[tuple[Any, ...], null_tables_lib.NullCoarsenedTables] = {}


def _null_tables(
    config: config_lib.DiscoConfig, effective_bins: int
) -> null_tables_lib.NullCoarsenedTables:
  key = (
      config.kappa,
      config.e_min,
      config.e_max,
      config.n_max,
      config.budget,
      config.chunk_size,
      effective_bins,
      config.alpha,
  )
  if key not in _TABLE_CACHE:
    _TABLE_CACHE[key] = null_tables_lib.compute_null_coarsened_tables(
        kappa=config.kappa,
        e_min=config.e_min,
        e_max=config.e_max,
        n_max=config.n_max,
        k_max=max(1, config.budget // config.chunk_size),
        num_bins=effective_bins,
        alpha=config.alpha,
        n_sim=500,
        seed=42,
    )
  return _TABLE_CACHE[key]


def _make_state(
    seed: int, n_hyp: int, strong: bool = True
) -> checkpoint_lib.RunState:
  """Returns a synthetic end-of-run state with frozen and running hyps."""
  rng = np.random.default_rng(seed)
  state = checkpoint_lib.RunState(remaining_budget=0)
  frozen_statuses = ["H", "L", "A", "B_1", "B_2"]
  for i in range(n_hyp):
    h_id = f"H{i}"
    n = int(rng.integers(1, 6))
    e_vals = [float(np.exp(rng.normal(0.4, 1.5))) for _ in range(n)]
    record = records_lib.HypothesisRecord(text=f"hypothesis {i}")
    record.queries = 3 * n
    for j, e in enumerate(e_vals):
      record.e_values.append(
          records_lib.EValueEntry(
              sub_hypothesis=f"sub {i}.{j}", e_value=e, cumulative_queries=j
          )
      )
    state.hypothesis_groups[h_id] = record
    state.probe_counts[h_id] = n
    curr_e = float(np.prod(e_vals))
    if i % 3 == 0:
      state.final_e_values[h_id] = curr_e
      state.terminal_statuses[h_id] = frozen_statuses[i % len(frozen_statuses)]
    else:
      state.running_e_values[h_id] = curr_e
      if i % 2:
        state.terminal_statuses[h_id] = "A"
  if not strong:
    return state
  # One very strong hypothesis so e-BH rejects something.
  state.running_e_values["H_strong"] = 500.0
  state.hypothesis_groups["H_strong"] = records_lib.HypothesisRecord(
      text="strong"
  )
  state.probe_counts["H_strong"] = 4
  return state


def _normalize(value: Any) -> Any:
  if dataclasses.is_dataclass(value) and not isinstance(value, type):
    return {
        f.name: _normalize(getattr(value, f.name))
        for f in dataclasses.fields(value)
    }
  if isinstance(value, dict):
    return {k: _normalize(v) for k, v in value.items()}
  if isinstance(value, (set, frozenset)):
    return sorted(value)
  if isinstance(value, (list, tuple)):
    return [_normalize(v) for v in value]
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, np.generic):
    return value.item()
  return value


def _make_config(out_dir: str) -> config_lib.DiscoConfig:
  return config_lib.DiscoConfig(
      target_model="fake-target",
      seed_instruction="Produce short answers.",
      agent_model="fake-agent",
      budget=30,
      chunk_size=3,
      alpha=0.1,
      kappa=0.5,
      e_min=0.05,
      e_max=20.0,
      n_max=5,
      alpha_sweep_steps=4,
      experiment_name="exp",
      output_dir=out_dir,
  )


class FinalizersTest(parameterized.TestCase):

  def _run_both(
      self,
      config_kwargs: dict[str, Any],
      n_hyp: int = 8,
      strong: bool = True,
  ) -> tuple[records_lib.DiscoResult, str | None]:
    out_dir = self.create_tempdir().full_path
    config = _make_config(out_dir)
    for k, v in config_kwargs.items():
      setattr(config, k, v)
    unbinned_e_values = config.num_active_bins == 0
    effective_bins = 1 if unbinned_e_values else config.num_active_bins
    is_baseline = config.baseline_mode or config.baseline_type1_mode
    tables = None if is_baseline else _null_tables(config, effective_bins)
    if is_baseline:
      bin_edges = np.array([])
    else:
      bin_edges = status_lib.compute_strip_bins(
          config.e_min, config.e_max, effective_bins
      )
    state = _make_state(seed=7, n_hyp=n_hyp, strong=strong)
    bounds_path = os.path.join(out_dir, "exp", "fdr_bounds.json")

    legacy_state = copy.deepcopy(state)
    legacy_log: list[str] = []
    legacy = _LegacyRunner(
        config,
        legacy_state,
        tables,
        effective_bins,
        bin_edges,
        unbinned_e_values,
    ).legacy_finalize(legacy_log.append)
    if os.path.exists(bounds_path):
      with open(bounds_path, "r", encoding="utf-8") as f:
        legacy_json = f.read()
      os.remove(bounds_path)
    else:
      legacy_json = None

    new_state = copy.deepcopy(state)
    new_log: list[str] = []
    new = finalizers.make_finalizer(
        config,
        null_tables=tables,
        effective_bins=effective_bins,
        bin_edges=bin_edges,
        unbinned_e_values=unbinned_e_values,
    ).finalize(new_state, new_log.append)
    if os.path.exists(bounds_path):
      with open(bounds_path, "r", encoding="utf-8") as f:
        new_json = f.read()
    else:
      new_json = None

    self.assertEqual(_normalize(new), _normalize(legacy))
    self.assertEqual(new_log, legacy_log)
    self.assertEqual(new_json, legacy_json)
    self.assertEqual(_normalize(new_state), _normalize(legacy_state))
    return new, new_json

  def test_type1(self):
    result, bounds = self._run_both({"baseline_type1_mode": True})
    self.assertIsNotNone(bounds)
    self.assertIn("H_strong", result.rejected_hypotheses)

  @parameterized.parameters(3, 10)
  def test_baseline_ebh(self, chunk_size):
    # K_bar is budget // chunk_size (budget is 30).
    k_bar = 30 // chunk_size
    result, bounds = self._run_both(
        {"baseline_mode": True, "chunk_size": chunk_size}
    )
    assert bounds is not None
    self.assertEqual(json.loads(bounds)["params"]["baseline_k_bar"], k_bar)
    self.assertEqual(result.total_hypotheses_k, k_bar)

  @parameterized.product(
      mode=("nominal", "pre_hoc", "post_hoc", "good_event", "unknown"),
      use_boosting=(True, False),
      num_active_bins=(0, 1, 3),
  )
  def test_disco(self, mode, use_boosting, num_active_bins):
    result, bounds = self._run_both({
        "alpha_calibration_mode": mode,
        "use_boosting": use_boosting,
        "num_active_bins": num_active_bins,
    })
    self.assertIsNotNone(bounds)
    self.assertNotEmpty(result.alpha_sweep_results)

  @parameterized.product(
      mode=("nominal", "pre_hoc", "post_hoc", "good_event"),
      use_boosting=(True, False),
      num_active_bins=(0, 1, 3),
  )
  def test_sweep_at_run_alpha_matches_headline(
      self, mode, use_boosting, num_active_bins
  ):
    result, _ = self._run_both({
        "alpha_calibration_mode": mode,
        "use_boosting": use_boosting,
        "num_active_bins": num_active_bins,
    })
    (entry,) = [
        e
        for e in result.alpha_sweep_results
        if abs(e["alpha_target"] - 0.1) < 1e-12
    ]
    self.assertAlmostEqual(
        entry["nominal_alpha"], result.nominal_alpha_used, places=9
    )
    self.assertEqual(
        entry["rejected_hypotheses"], sorted(result.rejected_hypotheses)
    )
    self.assertAlmostEqual(
        entry["epsilon_post"],
        result.epsilon_post_boost if use_boosting else result.epsilon_post,
        places=9,
    )
    self.assertAlmostEqual(entry["delta_post"], result.delta_post, places=9)

  def test_disco_no_hypotheses(self):
    result, bounds = self._run_both({}, n_hyp=0, strong=False)
    self.assertIsNone(bounds)
    self.assertEqual(result.total_hypotheses_k, 0)


if __name__ == "__main__":
  absltest.main()
