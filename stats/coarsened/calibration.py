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

"""Alpha calibration and post-run alpha sweeping for DISCO."""

import dataclasses

import numpy as np

from disco.stats import boosting
from disco.stats import multiple_testing
from disco.stats.coarsened import certificates as cert_lib
from disco.stats.coarsened import null_tables as null_tables_lib
from disco.stats.coarsened import status as status_lib


def find_optimal_prehoc_alpha_bh(
    alpha_star: float,
    kappa: float,
    k_bar: int,
    null_tables: null_tables_lib.NullCoarsenedTables,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> tuple[float, float]:
  """Binary searches for nominal alpha in pre_hoc boosted mode.

  Finds largest alpha_bh in [0, 1] such that phi_max(alpha_bh) <= alpha_star -
  K_bar * P_H(n_max).

  Args:
    alpha_star: Target FDR level alpha*.
    kappa: Calibrator parameter.
    k_bar: Upper bound on number of hypotheses K_bar.
    null_tables: Precomputed NullCoarsenedTables.
    tol: Binary search tolerance.
    max_iter: Maximum bisection iterations.

  Returns:
    Tuple of (optimal_alpha_bh, phi_max_at_solution).
  """
  target = max(0.0, alpha_star - k_bar * null_tables.P_H_max)
  if target <= 0.0:
    return 0.0, 0.0

  # Check alpha = 1.0
  phi_at_1 = cert_lib.compute_prehoc_phi_max(1.0, kappa, k_bar, null_tables)
  if phi_at_1 <= target:
    return 1.0, phi_at_1

  alpha_lo, alpha_hi = 0.0, 1.0
  for _ in range(max_iter):
    alpha_mid = (alpha_lo + alpha_hi) / 2.0
    phi_mid = cert_lib.compute_prehoc_phi_max(
        alpha_mid, kappa, k_bar, null_tables
    )
    if phi_mid <= target:
      alpha_lo = alpha_mid
    else:
      alpha_hi = alpha_mid
    if (alpha_hi - alpha_lo) < tol:
      break

  phi_final = cert_lib.compute_prehoc_phi_max(
      alpha_lo, kappa, k_bar, null_tables
  )
  return alpha_lo, phi_final


def find_optimal_alpha_bh(
    alpha_star: float,
    kappa: float,
    statuses: list[str],
    counts: list[int],
    k_obs: int,
    null_tables: null_tables_lib.NullCoarsenedTables,
    delta_post: float = 0.0,
    mode: str = "good_event",
    tol: float = 1e-6,
    max_iter: int = 50,
    *,
    final_e_values: dict[str, float] | None = None,
    terminal_statuses_dict: dict[str, str] | None = None,
    probe_counts_dict: dict[str, int] | None = None,
    n_max: int | None = None,
) -> tuple[float, float]:
  """Binary searches for the largest alpha_bh such that FDR certificate <= target.

  For unbinned e-values mode, pass final_e_values (along with
  terminal_statuses_dict, probe_counts_dict, and n_max) to use exact
  observed e-values in the phi_bar computation instead of table-based
  conditional expectations.

  Args:
    alpha_star: Target FDR level alpha*.
    kappa: Calibrator parameter.
    statuses: Terminal statuses for all tested hypotheses.
    counts: Probe counts for all tested hypotheses.
    k_obs: Total number of tested hypotheses K.
    null_tables: Precomputed NullCoarsenedTables.
    delta_post: Bad-event probability delta_post = K_obs * P_H(n_max).
    mode: 'good_event', 'post_hoc', or 'nominal'.
    tol: Binary search tolerance on alpha.
    max_iter: Maximum bisection iterations.
    final_e_values: For unbinned e-values mode — exact observed e-values per
      hypothesis.
    terminal_statuses_dict: For unbinned e-values mode — hypothesis_id -> status
      dict.
    probe_counts_dict: For unbinned e-values mode — hypothesis_id -> probe count
      dict.
    n_max: For unbinned e-values mode — maximum probes per hypothesis.

  Returns:
    Tuple of (optimal_alpha_bh, phi_bar_at_solution).
  """
  effective_n_max = n_max if n_max is not None else null_tables.n_max

  def _phi_bar(alpha_bh: float) -> float:
    if (
        final_e_values is not None
        and terminal_statuses_dict is not None
        and probe_counts_dict is not None
    ):
      return cert_lib.compute_continuous_phi_bar(
          alpha_bh,
          kappa,
          terminal_statuses_dict,
          probe_counts_dict,
          final_e_values,
          null_tables,
          effective_n_max,
      )
    return cert_lib.compute_phi_bar(
        alpha_bh, kappa, statuses, counts, k_obs, null_tables
    )

  if mode == "good_event":
    target = alpha_star
  elif mode == "post_hoc":
    target = max(0.0, alpha_star - delta_post)
  elif mode == "nominal":
    return alpha_star, _phi_bar(alpha_star)
  else:
    return alpha_star, 0.0

  if target <= 0.0:
    return 0.0, 0.0

  # Check if alpha_hi = 1.0 is already feasible
  phi_at_1 = _phi_bar(1.0)
  if phi_at_1 <= target:
    return 1.0, phi_at_1

  # Binary search on [0, 1]
  alpha_lo, alpha_hi = 0.0, 1.0
  for _ in range(max_iter):
    alpha_mid = (alpha_lo + alpha_hi) / 2.0
    phi_mid = _phi_bar(alpha_mid)
    if phi_mid <= target:
      alpha_lo = alpha_mid
    else:
      alpha_hi = alpha_mid
    if (alpha_hi - alpha_lo) < tol:
      break

  phi_final = _phi_bar(alpha_lo)
  return alpha_lo, phi_final


@dataclasses.dataclass
class DiscoAlphaSweepEntry:
  """Result for a single alpha level in a post-run alpha sweep."""

  alpha_target: float
  nominal_alpha: float
  num_discoveries: int
  rejected_hypotheses: set[str]
  rejection_threshold: float
  posthoc_unboosted: float
  posthoc_boosted: float
  epsilon_post: float
  epsilon_post_boost: float
  delta_post: float
  confidence: float


def compute_disco_alpha_sweep(
    final_e_values: dict[str, float],
    terminal_statuses: dict[str, str] | list[str],
    probe_counts: dict[str, int] | list[int],
    null_tables: null_tables_lib.NullCoarsenedTables,
    sweep_alphas: list[float],
    alpha_calibration_mode: str = "post_hoc",
    use_boosting: bool = True,
    kappa: float = 0.5,
    *,
    continuous: bool = False,
    n_max: int | None = None,
) -> list[DiscoAlphaSweepEntry]:
  """Computes post-run alpha sweep across multiple target alpha levels.

  Each entry is computed exactly as the run's headline would be at that alpha,
  so the entry at the run's own alpha reproduces the headline rejections.

  Args:
    final_e_values: Dict mapping hypothesis_id -> unboosted e-value M_tau^g.
    terminal_statuses: Terminal statuses ('A', 'L', 'H', 'B_j') as dict or list.
    probe_counts: Probe counts n_g as dict or list.
    null_tables: Precomputed NullCoarsenedTables.
    sweep_alphas: List of target alpha* values to evaluate.
    alpha_calibration_mode: 'nominal', 'pre_hoc', 'post_hoc', or 'good_event'.
    use_boosting: Whether to apply per-hypothesis e-value boosting.
    kappa: Calibrator parameter.
    continuous: Unbinned e-values (`num_active_bins=0`). Active hypotheses then
      contribute their exact observed e-value to E_bar_obs, the alpha
      calibration and phi_bar instead of the table expectation E_A(n_g), and the
      unboosted pre-hoc bound is `null_tables.e_max`.
    n_max: Probe cap for the boost clip. Defaults to `null_tables.n_max`.

  Returns:
    List of DiscoAlphaSweepEntry for each alpha in sweep_alphas.
  """
  h_ids = list(final_e_values.keys())
  k_obs = len(h_ids)
  if k_obs == 0:
    return []
  statuses = [
      str(s)
      for s in cert_lib.align_to_ids(
          terminal_statuses, h_ids, "A", "terminal_statuses"
      )
  ]
  counts = [
      int(c)
      for c in cert_lib.align_to_ids(probe_counts, h_ids, 1, "probe_counts")
  ]
  effective_n_max = n_max if n_max is not None else null_tables.n_max
  statuses_by_id = dict(zip(h_ids, statuses))
  counts_by_id = dict(zip(h_ids, counts))

  # Fixed bad-event risk delta_post = K_obs * P_H(n_max)
  delta_post = cert_lib.compute_delta_post(
      counts, null_tables.P_H_table, null_tables.n_max
  )

  confidence = 1.0 - delta_post

  # Fixed average survivor expectation E_bar_obs = (1/K) * sum E_{D_g}(n_g)
  epsilon_sum = 0.0
  for h_id, status, n_g in zip(h_ids, statuses, counts):
    idx = min(max(1, n_g), null_tables.n_max) - 1
    st = status_lib.parse_status(status)
    if st.is_exit_low:
      epsilon_sum += null_tables.E_L_table[idx]
    elif (
        st.is_active
        and st.bin_idx is not None
        and null_tables.E_bin_table is not None
    ):
      b_idx = min(max(1, st.bin_idx), null_tables.num_bins) - 1
      epsilon_sum += null_tables.E_bin_table[b_idx, idx]
    elif st.is_active and not st.is_binned:
      epsilon_sum += (
          final_e_values[h_id] if continuous else null_tables.E_A_table[idx]
      )

  e_bar_obs = epsilon_sum / float(k_obs)

  results: list[DiscoAlphaSweepEntry] = []

  prehoc_bad_event = float(null_tables.k_max * null_tables.P_H_max)
  for a_star in sweep_alphas:
    # Calibrate nominal alpha
    if not use_boosting:
      if alpha_calibration_mode == "nominal":
        nominal_alpha = a_star
      elif alpha_calibration_mode == "pre_hoc":
        if continuous:
          e_max_bound = null_tables.e_max
        elif null_tables.num_bins > 1:
          e_max_bound = null_tables.E_bin_max
        else:
          e_max_bound = null_tables.E_A_max
        nominal_alpha = float(
            np.clip(
                (a_star - prehoc_bad_event) / max(1e-10, e_max_bound),
                0.0,
                1.0,
            )
        )
      elif alpha_calibration_mode == "post_hoc":
        nominal_alpha = float(
            np.clip((a_star - delta_post) / max(1e-10, e_bar_obs), 0.0, 1.0)
        )
      elif alpha_calibration_mode == "good_event":
        nominal_alpha = float(np.clip(a_star / max(1e-10, e_bar_obs), 0.0, 1.0))
      else:
        nominal_alpha = a_star
    else:
      if alpha_calibration_mode == "nominal":
        nominal_alpha = a_star
      elif alpha_calibration_mode == "pre_hoc":
        nominal_alpha, _ = find_optimal_prehoc_alpha_bh(
            alpha_star=a_star,
            kappa=kappa,
            k_bar=null_tables.k_max,
            null_tables=null_tables,
        )
      elif alpha_calibration_mode in ("good_event", "post_hoc"):
        nominal_alpha, _ = find_optimal_alpha_bh(
            alpha_star=a_star,
            kappa=kappa,
            statuses=statuses,
            counts=counts,
            k_obs=k_obs,
            null_tables=null_tables,
            delta_post=delta_post,
            mode=alpha_calibration_mode,
            final_e_values=final_e_values if continuous else None,
            terminal_statuses_dict=statuses_by_id if continuous else None,
            probe_counts_dict=counts_by_id if continuous else None,
            n_max=effective_n_max if continuous else None,
        )
      else:
        nominal_alpha = a_star

    # Compute boosted e-values for this alpha
    if use_boosting and nominal_alpha > 0:
      boosted_e = {}
      for h_id, n_g in zip(h_ids, counts):
        m = boosting.boost_probe_count(n_g, effective_n_max)
        b = boosting.compute_boost_factor(
            nominal_alpha, kappa, m, k_obs, use_exact_t=True
        )
        boosted_e[h_id] = float(b * final_e_values[h_id])
    else:
      boosted_e = dict(final_e_values)

    # e-BH rejection
    rejected_set, thresh = multiple_testing.e_bh_rejection(
        boosted_e, nominal_alpha
    )

    # Certificates
    eps_unboosted = nominal_alpha * e_bar_obs
    if use_boosting and continuous:
      eps_boosted = cert_lib.compute_continuous_phi_bar(
          nominal_alpha,
          kappa,
          statuses_by_id,
          counts_by_id,
          final_e_values,
          null_tables,
          effective_n_max,
      )
    elif use_boosting:
      eps_boosted = cert_lib.compute_phi_bar(
          nominal_alpha, kappa, statuses, counts, k_obs, null_tables
      )
    else:
      eps_boosted = eps_unboosted
    post_unboosted = eps_unboosted + delta_post
    post_boosted = eps_boosted + delta_post

    results.append(
        DiscoAlphaSweepEntry(
            alpha_target=a_star,
            nominal_alpha=nominal_alpha,
            num_discoveries=len(rejected_set),
            rejected_hypotheses=rejected_set,
            rejection_threshold=thresh,
            posthoc_unboosted=post_unboosted,
            posthoc_boosted=post_boosted,
            epsilon_post=eps_unboosted,
            epsilon_post_boost=eps_boosted,
            delta_post=delta_post,
            confidence=confidence,
        )
    )

  return results
