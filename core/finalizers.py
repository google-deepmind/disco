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

"""End-of-run finalization strategies for the DISCO discovery loop.

After the budget is spent, each mode turns the run state into a rejection set,
bounds report, alpha sweep and `fdr_bounds.json`:

  * `Type1Finalizer` (baseline_type1_mode): reject E >= 1/alpha, no e-BH.
  * `BaselineEBHFinalizer` (baseline_mode): offline e-BH with K = K_bar.
  * `DiscoFinalizer` (default): alpha calibration, boosting, e-BH and post-hoc
    DISCO certificates.

Finalizers mutate `state.final_e_values` / `state.terminal_statuses` for the
hypotheses still active at the end of the run, exactly as the runner did.
"""

from collections.abc import Callable
from typing import Protocol

import numpy as np

from disco.core import checkpoint as checkpoint_lib
from disco.core import config as config_lib
from disco.core import fdr_outputs
from disco.core import records as records_lib
from disco.stats import boosting
from disco.stats import multiple_testing
from disco.stats.coarsened import calibration
from disco.stats.coarsened import certificates
from disco.stats.coarsened import null_tables as null_tables_lib

LogFn = Callable[[str], None]


class Finalizer(Protocol):
  """Turns the end-of-run state into the run result."""

  def finalize(
      self, state: checkpoint_lib.RunState, log_fn: LogFn
  ) -> records_lib.DiscoResult:
    ...


class Type1Finalizer:
  """Type-I baseline: rejects every hypothesis with E >= 1/alpha."""

  def __init__(self, config: config_lib.DiscoConfig):
    self._config = config

  def finalize(
      self, state: checkpoint_lib.RunState, log_fn: LogFn
  ) -> records_lib.DiscoResult:
    """Finalizes Type-I baseline run by rejecting hypotheses with E >= 1/alpha."""
    config = self._config
    log = log_fn
    log("\n=== DISCO Type-I Error Baseline Finalization ===")
    for h_id, curr_e in list(state.running_e_values.items()):
      state.final_e_values[h_id] = curr_e
      state.terminal_statuses[h_id] = "A"

    threshold = 1.0 / config.alpha
    k_total = len(state.final_e_values)
    log(f"Total hypotheses tested (K_obs): {k_total}")
    log(f"Type-I rejection threshold (1/alpha): {threshold:.4f}")

    # No boosting for baseline.
    boosted_e_values = dict(state.final_e_values)
    boost_factors = {h_id: 1.0 for h_id in state.final_e_values}

    rejected_set = {
        h_id
        for h_id, e_val in state.final_e_values.items()
        if e_val >= threshold
    }
    log(f"  Rejected (E >= 1/alpha): {len(rejected_set)} hypotheses")

    fdr_outputs.log_rejections(
        state.hypothesis_groups,
        rejected_set,
        state.final_e_values,
        rejected_header="\n=== Rejected Hypotheses (Type-I Error Baseline) ===",
        rejected_e=lambda h: state.final_e_values[h],
        non_rejected_e=lambda h: state.final_e_values.get(h, 0.0),
        log_fn=log,
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

    # Alpha sweep: threshold at 1/alpha for each alpha.
    alpha_sweep_records = []
    for a in fdr_outputs.sweep_alphas(config):
      thresh_a = 1.0 / a if a > 0 else float("inf")
      rej = {
          h_id
          for h_id, e_val in state.final_e_values.items()
          if e_val >= thresh_a
      }
      alpha_sweep_records.append({
          "alpha_target": float(a),
          "nominal_alpha": float(a),
          "num_discoveries": int(len(rej)),
          "rejected_hypotheses": sorted(list(rej)),
          "rejection_threshold": float(thresh_a),
      })

    fdr_outputs.write_fdr_bounds(
        config,
        params=fdr_outputs.fdr_params(
            config,
            {
                "chunk_size": config.chunk_size,
                "baseline_mode": True,
                "baseline_type1_mode": True,
            },
        ),
        bounds_report=bounds_report_dict,
        alpha_sweep=alpha_sweep_records,
        hyp_data=fdr_outputs.hypotheses_data(
            state.probe_counts,
            state.final_e_values,
            boosted_e_values,
            boost_factors,
            state.terminal_statuses,
        ),
        log_fn=log,
    )

    return records_lib.DiscoResult(
        all_records=state.hypothesis_groups,
        rejected_hypotheses=rejected_set,
        final_e_values=state.final_e_values,
        boosted_e_values=boosted_e_values,
        boost_factors=boost_factors,
        probe_counts=state.probe_counts,
        terminal_statuses=state.terminal_statuses,
        total_hypotheses_k=k_total,
        rejection_threshold=threshold,
        nominal_alpha_used=config.alpha,
        bounds_report=bounds_report_dict,
        alpha_sweep_results=alpha_sweep_records,
    )


class BaselineEBHFinalizer:
  """Unrestricted-search baseline: offline e-BH with K = K_bar."""

  def __init__(self, config: config_lib.DiscoConfig):
    self._config = config

  def finalize(
      self, state: checkpoint_lib.RunState, log_fn: LogFn
  ) -> records_lib.DiscoResult:
    """Finalizes unrestricted baseline run using offline e-BH with K = K_bar."""
    config = self._config
    log = log_fn
    log("\n=== DISCO Baseline Finalization ===")
    for h_id, curr_e in list(state.running_e_values.items()):
      state.final_e_values[h_id] = curr_e
      state.terminal_statuses[h_id] = "A"

    k_bar = max(1, config.budget // config.chunk_size)
    k_total = len(state.final_e_values)
    log(f"Total hypotheses tested (K_obs): {k_total}")
    log(f"Baseline K_bar for e-BH: {k_bar}")

    # No boosting for baseline.
    boosted_e_values = dict(state.final_e_values)
    boost_factors = {h_id: 1.0 for h_id in state.final_e_values}

    rejected_set, rejection_thresh = multiple_testing.e_bh_rejection(
        state.final_e_values, config.alpha, k_total_override=k_bar
    )
    log(f"  Rejection threshold: {rejection_thresh:.4f}")
    log(f"  Rejected: {len(rejected_set)} hypotheses (out of K_bar={k_bar})")

    fdr_outputs.log_rejections(
        state.hypothesis_groups,
        rejected_set,
        state.final_e_values,
        rejected_header="\n=== Rejected Hypotheses (Baseline e-BH) ===",
        rejected_e=lambda h: state.final_e_values[h],
        non_rejected_e=lambda h: state.final_e_values.get(h, 0.0),
        log_fn=log,
    )

    bounds_report_dict = {
        "nominal_calibrated_alpha": float(config.alpha),
        "target_alpha": float(config.alpha),
        "kappa": float(config.kappa),
        "baseline_mode": True,
        "k_bar": int(k_bar),
        "k_obs": int(k_total),
    }

    alpha_sweep_records = []
    for a in fdr_outputs.sweep_alphas(config):
      rej, thresh = multiple_testing.e_bh_rejection(
          state.final_e_values, a, k_total_override=k_bar
      )
      alpha_sweep_records.append({
          "alpha_target": float(a),
          "nominal_alpha": float(a),
          "num_discoveries": int(len(rej)),
          "rejected_hypotheses": sorted(list(rej)),
          "rejection_threshold": float(thresh),
      })

    fdr_outputs.write_fdr_bounds(
        config,
        params=fdr_outputs.fdr_params(
            config,
            {
                "chunk_size": config.chunk_size,
                "baseline_mode": True,
                "baseline_k_bar": k_bar,
            },
        ),
        bounds_report=bounds_report_dict,
        alpha_sweep=alpha_sweep_records,
        hyp_data=fdr_outputs.hypotheses_data(
            state.probe_counts,
            state.final_e_values,
            boosted_e_values,
            boost_factors,
            state.terminal_statuses,
        ),
        log_fn=log,
    )

    # The result reports K = K_bar (not K_obs), unlike the other modes.
    return records_lib.DiscoResult(
        all_records=state.hypothesis_groups,
        rejected_hypotheses=rejected_set,
        final_e_values=state.final_e_values,
        boosted_e_values=boosted_e_values,
        boost_factors=boost_factors,
        probe_counts=state.probe_counts,
        terminal_statuses=state.terminal_statuses,
        total_hypotheses_k=k_bar,
        rejection_threshold=rejection_thresh,
        nominal_alpha_used=config.alpha,
        bounds_report=bounds_report_dict,
        alpha_sweep_results=alpha_sweep_records,
    )


class DiscoFinalizer:
  """DISCO: alpha calibration, boosting, e-BH and DISCO certificates."""

  def __init__(
      self,
      config: config_lib.DiscoConfig,
      *,
      null_tables: null_tables_lib.NullCoarsenedTables,
      effective_bins: int,
      bin_edges: np.ndarray,
      unbinned_e_values: bool,
  ):
    self._config = config
    self._null_tables = null_tables
    self._effective_bins = effective_bins
    self._bin_edges = bin_edges
    self._unbinned_e_values = unbinned_e_values

  def _finalize_survivors(self, state: checkpoint_lib.RunState) -> None:
    """Assigns final e-values and statuses to still-active hypotheses."""
    config = self._config
    for h_id, curr_e in list(state.running_e_values.items()):
      state.final_e_values[h_id] = curr_e
      if curr_e < config.e_min:
        state.terminal_statuses[h_id] = "L"
      elif curr_e >= config.e_max:
        state.terminal_statuses[h_id] = "H"
      elif self._effective_bins <= 1:
        state.terminal_statuses[h_id] = "A"
      else:
        b_idx = int(np.searchsorted(self._bin_edges, curr_e, side="right"))
        b_val = max(1, min(b_idx, self._effective_bins))
        state.terminal_statuses[h_id] = f"B_{b_val}"

  def _calibrate_alpha(
      self,
      state: checkpoint_lib.RunState,
      *,
      k_total: int,
      delta_post: float,
  ) -> float:
    """Returns the calibrated alpha_BH for the configured mode."""
    config = self._config
    null_tables = self._null_tables
    unbinned_e_values = self._unbinned_e_values
    k_bar = max(1, config.budget // config.chunk_size)

    statuses_list = [
        state.terminal_statuses.get(h, "A") for h in state.final_e_values
    ]
    counts_list = [state.probe_counts.get(h, 1) for h in state.final_e_values]

    if unbinned_e_values:
      e_mean_max = config.e_max
    elif self._effective_bins > 1:
      e_mean_max = null_tables.E_bin_max
    else:
      e_mean_max = null_tables.E_A_max

    e_sum = sum(
        null_tables_lib.get_expected_null_e(
            state.terminal_statuses.get(h, "A"),
            state.probe_counts.get(h, 1),
            null_tables,
            exact_e=(
                state.final_e_values.get(h, 1.0) if unbinned_e_values else None
            ),
        )
        for h in state.terminal_statuses
        if state.terminal_statuses.get(h) != "H"
    )
    e_bar_obs = e_sum / max(1, k_total)

    if not config.use_boosting:
      if config.alpha_calibration_mode == "nominal":
        return config.alpha
      if config.alpha_calibration_mode == "pre_hoc":
        prehoc_bad_event = float(k_bar * null_tables.P_H_max)
        return float(
            np.clip(
                (config.alpha - prehoc_bad_event) / max(1e-10, e_mean_max),
                0.0,
                1.0,
            )
        )
      if config.alpha_calibration_mode == "post_hoc":
        return float(
            np.clip(
                (config.alpha - delta_post) / max(1e-10, e_bar_obs),
                0.0,
                1.0,
            )
        )
      if config.alpha_calibration_mode == "good_event":
        return float(np.clip(config.alpha / max(1e-10, e_bar_obs), 0.0, 1.0))
      return config.alpha

    if config.alpha_calibration_mode == "nominal":
      return config.alpha
    if config.alpha_calibration_mode == "pre_hoc":
      nominal_alpha, _ = calibration.find_optimal_prehoc_alpha_bh(
          alpha_star=config.alpha,
          kappa=config.kappa,
          k_bar=k_bar,
          null_tables=null_tables,
      )
      return nominal_alpha
    if config.alpha_calibration_mode in ("good_event", "post_hoc"):
      nominal_alpha, _ = calibration.find_optimal_alpha_bh(
          alpha_star=config.alpha,
          kappa=config.kappa,
          statuses=statuses_list,
          counts=counts_list,
          k_obs=k_total,
          null_tables=null_tables,
          delta_post=delta_post,
          mode=config.alpha_calibration_mode,
          final_e_values=(
              state.final_e_values if unbinned_e_values else None
          ),
          terminal_statuses_dict=(
              state.terminal_statuses if unbinned_e_values else None
          ),
          probe_counts_dict=(
              state.probe_counts if unbinned_e_values else None
          ),
          n_max=config.n_max if unbinned_e_values else None,
      )
      return nominal_alpha
    return config.alpha

  def finalize(
      self, state: checkpoint_lib.RunState, log_fn: LogFn
  ) -> records_lib.DiscoResult:
    """Finalizes DISCO run with alpha calibration, boosting, and e-BH."""
    config = self._config
    log = log_fn
    null_tables = self._null_tables
    unbinned_e_values = self._unbinned_e_values

    log("\n=== DISCO Finalization ===")
    self._finalize_survivors(state)

    k_total = len(state.final_e_values)
    log(f"Total hypotheses K: {k_total}")

    if k_total == 0:
      log("\n[Warning] No hypotheses were tested. Returning empty result.")
      return records_lib.DiscoResult(
          all_records=state.hypothesis_groups,
          rejected_hypotheses=set(),
          final_e_values={},
          boosted_e_values={},
          boost_factors={},
          probe_counts=state.probe_counts,
          terminal_statuses=state.terminal_statuses,
          total_hypotheses_k=0,
          rejection_threshold=0.0,
          nominal_alpha_used=0.0,
      )

    # Delta_post = K_obs * P_H(n_max).
    delta_post = certificates.compute_delta_post(
        (state.probe_counts.get(h, 1) for h in state.final_e_values),
        null_tables.P_H_table,
        null_tables.n_max,
    )
    log(f"  delta_post: {delta_post:.6f}")

    log(f"\n--- Alpha Calibration (mode={config.alpha_calibration_mode}) ---")
    nominal_alpha = self._calibrate_alpha(
        state, k_total=k_total, delta_post=delta_post
    )
    log(f"  Calibrated alpha_BH: {nominal_alpha:.6f}")

    # Pre-hoc FDR bounds.
    alpha_for_prehoc = (
        nominal_alpha
        if config.alpha_calibration_mode == "pre_hoc"
        else config.alpha
    )
    prehoc_bounds = certificates.compute_prehoc_fdr_bounds(
        null_tables,
        max(1, config.budget // config.chunk_size),
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
    for h_id, e_val in state.final_e_values.items():
      m = boosting.boost_probe_count(
          state.probe_counts.get(h_id, 1), config.n_max
      )
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
    cert = certificates.compute_disco_posthoc_certificate(
        state.terminal_statuses,
        state.probe_counts,
        null_tables,
        nominal_alpha,
        final_e_values=(state.final_e_values if unbinned_e_values else None),
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

    fdr_outputs.log_rejections(
        state.hypothesis_groups,
        rejected_set,
        state.final_e_values,
        rejected_header="\n=== Rejected Hypotheses (DISCO FDR-controlled) ===",
        rejected_e=lambda h: boosted_e_values[h],
        non_rejected_e=lambda h: boosted_e_values.get(h, 0.0),
        unboosted_e=state.final_e_values.get,
        log_fn=log,
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

    sweep_entries = calibration.compute_disco_alpha_sweep(
        final_e_values=state.final_e_values,
        terminal_statuses=state.terminal_statuses,
        probe_counts=state.probe_counts,
        null_tables=null_tables,
        sweep_alphas=fdr_outputs.sweep_alphas(config),
        alpha_calibration_mode=config.alpha_calibration_mode,
        use_boosting=config.use_boosting,
        kappa=config.kappa,
        continuous=unbinned_e_values,
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

    fdr_outputs.write_fdr_bounds(
        config,
        params=fdr_outputs.fdr_params(
            config,
            {
                "e_min": config.e_min,
                "e_max": config.e_max,
                "use_boosting": config.use_boosting,
                "chunk_size": config.chunk_size,
                "alpha_calibration_mode": config.alpha_calibration_mode,
                "n_max": config.n_max,
                "num_active_bins": config.num_active_bins,
            },
        ),
        bounds_report=bounds_report_dict,
        alpha_sweep=alpha_sweep_records,
        hyp_data=fdr_outputs.hypotheses_data(
            state.probe_counts,
            state.final_e_values,
            boosted_e_values,
            boost_factors,
            state.terminal_statuses,
        ),
        log_fn=log,
    )

    return records_lib.DiscoResult(
        all_records=state.hypothesis_groups,
        rejected_hypotheses=rejected_set,
        final_e_values=state.final_e_values,
        boosted_e_values=boosted_e_values,
        boost_factors=boost_factors,
        probe_counts=state.probe_counts,
        terminal_statuses=state.terminal_statuses,
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


def make_finalizer(
    config: config_lib.DiscoConfig,
    *,
    null_tables: null_tables_lib.NullCoarsenedTables | None,
    effective_bins: int,
    bin_edges: np.ndarray,
    unbinned_e_values: bool,
) -> Finalizer:
  """Returns the finalizer for the configured mode."""
  if config.baseline_type1_mode:
    return Type1Finalizer(config)
  if config.baseline_mode:
    return BaselineEBHFinalizer(config)
  assert null_tables is not None
  return DiscoFinalizer(
      config,
      null_tables=null_tables,
      effective_bins=effective_bins,
      bin_edges=bin_edges,
      unbinned_e_values=unbinned_e_values,
  )
