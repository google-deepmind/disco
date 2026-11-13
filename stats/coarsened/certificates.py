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

"""Post-hoc and pre-hoc FDR certificates and error bounds for DISCO."""

from collections.abc import Collection, Iterable, Mapping, Sequence
import dataclasses
import typing

import numpy as np

from disco.stats import boosting
from disco.stats.coarsened import null_tables as null_tables_lib
from disco.stats.coarsened import status as status_lib

_V = typing.TypeVar("_V")


def align_to_ids(
    values: Mapping[str, _V] | Sequence[_V],
    h_ids: Sequence[str],
    default: _V,
    name: str,
    reference: str = "final_e_values",
) -> list[_V]:
  """Returns `values` ordered like `h_ids`.

  A dict is looked up by id (missing ids get `default`). A list must already
  be in `h_ids` order, so its length has to match.

  Args:
    values: Per-hypothesis values, keyed by id or positional.
    h_ids: Hypothesis ids, in the order of `reference`.
    default: Value for ids a dict does not contain.
    name: Argument name for the error message.
    reference: Name of the argument `h_ids` came from, for the error message.

  Raises:
    ValueError: If a list's length differs from `len(h_ids)`.
  """
  if isinstance(values, Mapping):
    return [values.get(k, default) for k in h_ids]
  if len(values) != len(h_ids):
    raise ValueError(
        f"{name} has {len(values)} entries but {reference} has"
        f" {len(h_ids)}; a list must give one value per hypothesis, in"
        f" {reference} order."
    )
  return list(values)


@dataclasses.dataclass
class DiscoPostHocCertificate:
  """Post-hoc FDR certificate for a DISCO run."""

  epsilon_post: float  # alpha * (1/K) * sum E_{D_g}(n_g) over D_g in {A, L}
  epsilon_post_boost: float  # (1/K) * sum phi(A, n_g, K_obs) over D_g = A
  delta_post: float  # K_obs * P_H(n_max)
  posthoc_unboosted: float  # epsilon_post + delta_post
  posthoc_boosted: float  # epsilon_post_boost + delta_post
  confidence: float  # 1 - delta_post
  k_obs: int


def compute_delta_post(
    counts: Iterable[int],
    P_H_table: np.ndarray,  # pylint: disable=invalid-name  # Paper notation P_H.
    n_max: int,
) -> float:
  """Computes delta_post = K_obs * P_H(n_max)."""
  n_hyps = sum(1 for _ in counts)
  if n_hyps == 0 or n_max <= 0 or len(P_H_table) == 0:
    return 0.0
  idx = min(max(1, n_max), len(P_H_table)) - 1
  return float(n_hyps) * float(P_H_table[idx])


def compute_epsilon_sum(
    statuses: Iterable[str | status_lib.TerminalStatus],
    counts: Iterable[int],
    null_tables: null_tables_lib.NullCoarsenedTables,
    *,
    exact_e_values: Sequence[float | None] | None = None,
) -> float:
  """Computes sum over g where D_g in {L, B_1, ..., B_J, A} of E_{D_g}(n_g)."""
  epsilon_sum = 0.0
  for i, (status, n_g) in enumerate(zip(statuses, counts)):
    st = status_lib.parse_status(status)
    exact_val = exact_e_values[i] if exact_e_values is not None else None
    epsilon_sum += null_tables_lib.get_expected_null_e(
        st, n_g, null_tables, exact_e=exact_val
    )
  return epsilon_sum


def recompute_phi_at_alpha(
    alpha_bh: float,
    kappa: float,
    m_values: Collection[int],
    k_obs: int,
    null_tables: null_tables_lib.NullCoarsenedTables,
    status: str | int | status_lib.TerminalStatus = "A",
) -> dict[int, float]:
  """Recomputes phi values for a specific nominal alpha_BH and terminal status.

  Args:
    alpha_bh: Nominal alpha level for e-BH.
    kappa: Calibrator parameter.
    m_values: Iterable of probe counts m.
    k_obs: Total observed hypotheses K.
    null_tables: Precomputed NullCoarsenedTables.
    status: TerminalStatus, 'A' for survivors, 'L' for low exiters, 'B_j' for
      sub-bins.

  Returns:
    Dict mapping m -> phi(status, m, k; alpha_bh).
  """
  phi_dict = {}
  if alpha_bh <= 0.0 or k_obs <= 0:
    return {m: 0.0 for m in m_values}

  st = status_lib.parse_status(status)

  for m in set(m_values):
    m_eff = boosting.boost_probe_count(m, null_tables.n_max)
    m_idx = m_eff - 1
    b = boosting.compute_boost_factor(
        alpha_bh, kappa, m_eff, k_obs, use_exact_t=True
    )
    if st.is_active and not st.is_binned:
      if (
          null_tables.mc_m_free is not None
          and null_tables.mc_survived_cum is not None
      ):
        surv_mask = null_tables.mc_survived_cum[:, m_idx].astype(bool)
        if null_tables.enforce_exit_low:
          active_a_mask = surv_mask
        else:
          active_a_mask = surv_mask & (
              null_tables.mc_m_free[:, m_idx] >= null_tables.e_min
          )
        if np.any(active_a_mask):
          surv_m = null_tables.mc_m_free[active_a_mask, m_idx]
          phi_dict[m] = float(
              np.mean(boosting.staircase_t(alpha_bh * b * surv_m, k_obs))
          )
        else:
          phi_dict[m] = 0.0
      else:
        p_surv = null_tables.P_surv_table[m_idx]
        phi_dict[m] = alpha_bh / max(p_surv, 1e-10)
    elif st.is_active and st.bin_idx is not None:
      b_idx = min(max(1, st.bin_idx), null_tables.num_bins) - 1
      if (
          null_tables.mc_m_free is not None
          and null_tables.mc_survived_cum is not None
      ):
        surv_mask = null_tables.mc_survived_cum[:, m_idx].astype(bool)
        m_free_m = null_tables.mc_m_free[:, m_idx]
        b_low = null_tables.bin_edges[b_idx]
        b_high = null_tables.bin_edges[b_idx + 1]
        bin_mask = surv_mask & (m_free_m >= b_low) & (m_free_m < b_high)
        if np.any(bin_mask):
          bin_m = m_free_m[bin_mask]
          phi_dict[m] = float(
              np.mean(boosting.staircase_t(alpha_bh * b * bin_m, k_obs))
          )
        else:
          b_mid = np.sqrt(b_low * b_high)
          phi_dict[m] = float(boosting.staircase_t(alpha_bh * b * b_mid, k_obs))
      else:
        if null_tables.E_bin_table is not None:
          phi_dict[m] = float(
              boosting.staircase_t(
                  alpha_bh * b * null_tables.E_bin_table[b_idx, m_idx], k_obs
              )
          )
        else:
          phi_dict[m] = alpha_bh / max(null_tables.P_surv_table[m_idx], 1e-10)
    elif st.is_exit_low:
      if (
          null_tables.enforce_exit_low
          and null_tables.mc_m_free is not None
          and null_tables.mc_first_exit_idx is not None
          and null_tables.mc_exited_low is not None
      ):
        low_mask = null_tables.mc_exited_low & (
            null_tables.mc_first_exit_idx == m_idx
        )
        if np.any(low_mask):
          low_m = null_tables.mc_m_free[low_mask, m_idx]
          phi_dict[m] = float(
              np.mean(boosting.staircase_t(alpha_bh * b * low_m, k_obs))
          )
        else:
          phi_dict[m] = 0.0
      elif (
          not null_tables.enforce_exit_low
          and null_tables.mc_m_free is not None
          and null_tables.mc_survived_cum is not None
      ):
        surv_mask = null_tables.mc_survived_cum[:, m_idx].astype(bool)
        low_mask = surv_mask & (
            null_tables.mc_m_free[:, m_idx] < null_tables.e_min
        )
        if np.any(low_mask):
          low_m = null_tables.mc_m_free[low_mask, m_idx]
          phi_dict[m] = float(
              np.mean(boosting.staircase_t(alpha_bh * b * low_m, k_obs))
          )
        else:
          phi_dict[m] = 0.0
      else:
        if alpha_bh * b * null_tables.e_min >= 1.0:
          phi_dict[m] = float(
              boosting.staircase_t(alpha_bh * b * null_tables.e_min, k_obs)
          )
        else:
          phi_dict[m] = 0.0
    elif st.is_exit_high:
      phi_dict[m] = 0.0

  return phi_dict


def compute_phi_bar(
    alpha_bh: float,
    kappa: float,
    statuses: Iterable[str | int | status_lib.TerminalStatus],
    counts: Iterable[int],
    k_obs: int,
    null_tables: null_tables_lib.NullCoarsenedTables,
) -> float:
  """Computes phi_bar = (1/K) * sum phi(status_g, n_g, K; alpha_bh).

  Args:
    alpha_bh: Nominal alpha level for e-BH.
    kappa: Calibrator parameter.
    statuses: Iterable of terminal statuses for all tested hypotheses.
    counts: Iterable of probe counts for all tested hypotheses.
    k_obs: Total observed hypotheses K.
    null_tables: Precomputed NullCoarsenedTables.

  Returns:
    Average phi across all tested hypotheses.
  """
  if k_obs <= 0 or alpha_bh <= 0.0:
    return 0.0

  status_groups: dict[status_lib.TerminalStatus, list[int]] = {}
  for s, c in zip(statuses, counts):
    st = status_lib.parse_status(s)
    if st.is_exit_high:
      continue
    status_groups.setdefault(st, []).append(c)

  phi_sum = 0.0
  for st, m_list in status_groups.items():
    phi_dict = recompute_phi_at_alpha(
        alpha_bh, kappa, m_list, k_obs, null_tables, status=st
    )
    for m in m_list:
      phi_sum += phi_dict.get(m, 0.0)

  return phi_sum / float(k_obs)


def compute_continuous_phi_bar(
    alpha_bh: float,
    kappa: float,
    terminal_statuses: dict[str, str],
    probe_counts: dict[str, int],
    final_e_values: dict[str, float],
    null_tables: null_tables_lib.NullCoarsenedTables,
    n_max: int,
) -> float:
  """Computes phi_bar using exact observed e-values for unbinned e-values mode.

  For hypotheses with active/survived status, uses the exact observed
  e-value from final_e_values instead of the table-based conditional
  expectation E_A(m). Low-exit hypotheses still use E_L_table.

  Args:
    alpha_bh: Nominal alpha level for e-BH.
    kappa: Calibrator parameter.
    terminal_statuses: Mapping hypothesis_id -> terminal status string.
    probe_counts: Mapping hypothesis_id -> probe count n_g.
    final_e_values: Mapping hypothesis_id -> exact observed e-value.
    null_tables: Precomputed NullCoarsenedTables.
    n_max: Maximum probes per hypothesis (for boost factor computation).

  Returns:
    Average phi across all non-high-exit tested hypotheses.
  """
  k_total = len(terminal_statuses)
  if alpha_bh <= 0.0 or k_total <= 0:
    return 0.0
  phi_sum = 0.0
  for h_id, st_str in terminal_statuses.items():
    st = status_lib.parse_status(st_str)
    if st.is_exit_high:
      continue
    n_g = probe_counts.get(h_id, 1)
    m_eff = boosting.boost_probe_count(n_g, n_max)
    b = boosting.compute_boost_factor(
        alpha_bh, kappa, m_eff, k_total, use_exact_t=True
    )
    if st.is_exit_low:
      idx = min(max(1, n_g), null_tables.n_max) - 1
      e_val = float(null_tables.E_L_table[idx])
    else:
      e_val = final_e_values.get(h_id, 1.0)
    phi_sum += float(boosting.staircase_t(alpha_bh * b * e_val, k_total))
  return phi_sum / float(k_total)


def compute_prehoc_phi_max(
    alpha_bh: float,
    kappa: float,
    k_bar: int,
    null_tables: null_tables_lib.NullCoarsenedTables,
) -> float:
  """Computes max_{m <= n_max, k in k_grid(1, k_bar)} max_D phi(D, m, k; alpha)."""
  if alpha_bh <= 0.0 or k_bar <= 0:
    return 0.0

  m_values = list(range(1, null_tables.n_max + 1))
  k_grid = boosting.generate_k_grid(1, k_bar)
  max_val = 0.0
  for k in k_grid:
    if null_tables.num_bins > 1:
      for b_idx in range(1, null_tables.num_bins + 1):
        phi_b = recompute_phi_at_alpha(
            alpha_bh, kappa, m_values, k, null_tables, status=f"B_{b_idx}"
        )
        if phi_b:
          max_val = max(max_val, max(phi_b.values()))
    else:
      phi_a = recompute_phi_at_alpha(
          alpha_bh, kappa, m_values, k, null_tables, status="A"
      )
      if phi_a:
        max_val = max(max_val, max(phi_a.values()))
    phi_l = recompute_phi_at_alpha(
        alpha_bh, kappa, m_values, k, null_tables, status="L"
    )
    if phi_l:
      max_val = max(max_val, max(phi_l.values()))
  return max_val


def compute_prehoc_fdr_bounds(
    null_tables: null_tables_lib.NullCoarsenedTables,
    k_bar: int,
    alpha_star: float,
) -> dict[str, float]:
  """Computes pre-hoc FDR bounds from null coarsened tables.

  Args:
    null_tables: Precomputed NullCoarsenedTables.
    k_bar: Upper bound on number of hypotheses K_bar.
    alpha_star: Target FDR level alpha*.

  Returns:
    Dict with keys: baseline_Kbar_PH, unboosted, boosted, analytical_ceiling.
  """
  baseline_kbar_ph = float(k_bar * null_tables.P_H_max)
  e_mean_max = (
      null_tables.E_bin_max if null_tables.num_bins > 1 else null_tables.E_A_max
  )
  unboosted = alpha_star * e_mean_max + baseline_kbar_ph

  if abs(alpha_star - null_tables.alpha) > 1e-12:
    phi_max = compute_prehoc_phi_max(
        alpha_star, null_tables.kappa, k_bar, null_tables
    )
    with np.errstate(divide="ignore", invalid="ignore"):
      ceiling_arr = np.where(
          null_tables.P_surv_table > 0,
          alpha_star / null_tables.P_surv_table,
          float("inf"),
      )
    analytical_ceiling = float(np.max(ceiling_arr)) + baseline_kbar_ph
  else:
    phi_max = null_tables.phi_max
    analytical_ceiling = null_tables.analytical_ceiling + baseline_kbar_ph

  boosted = phi_max + baseline_kbar_ph

  return {
      "baseline_Kbar_PH": baseline_kbar_ph,
      "unboosted": unboosted,
      "boosted": boosted,
      "analytical_ceiling": analytical_ceiling,
  }


def compute_disco_posthoc_certificate(
    terminal_statuses: dict[str, str] | list[str],
    probe_counts: dict[str, int] | list[int],
    null_tables: null_tables_lib.NullCoarsenedTables,
    alpha_bh: float,
    *,
    final_e_values: dict[str, float] | None = None,
    use_boosting: bool = True,
    kappa: float | None = None,
    n_max_override: int | None = None,
) -> DiscoPostHocCertificate:
  """Computes a post-hoc FDR certificate for a DISCO run.

  For unbinned e-values mode, pass final_e_values to use exact observed
  e-values for active/survived hypotheses instead of the table-based
  E_A(m) expectations.

  Per-hypothesis inputs are lined up by hypothesis id. The ids come from
  `terminal_statuses` when it is a dict, else from `final_e_values`. A dict
  input is looked up by id and a list must already be in that id order (same
  length). With no ids at all (both lists), the lists pair up by position and
  must have the same length; a dict `probe_counts` is then rejected.

  Args:
    terminal_statuses: Terminal statuses for each hypothesis ('A', 'L', 'H',
      'B_j'), as either a dict (mapping hypothesis_id -> status) or a list.
    probe_counts: Probe counts n_g for each hypothesis, as either a dict
      (mapping hypothesis_id -> count) or a list.
    null_tables: Precomputed NullCoarsenedTables.
    alpha_bh: The alpha level used for e-BH rejection.
    final_e_values: For unbinned e-values mode — exact observed e-values per
      hypothesis. When provided, active/unbinned hypotheses use these values
      instead of E_A_table lookups.
    use_boosting: Whether boosting is enabled (used for unbinned e-values
      certificate computation).
    kappa: Calibrator parameter (used for unbinned e-values boost factors).
      Defaults to null_tables.kappa if not provided.
    n_max_override: Override for n_max (used for unbinned e-values boost
      factors). Defaults to null_tables.n_max if not provided.

  Returns:
    DiscoPostHocCertificate with post-hoc FDR bounds.

  Raises:
    ValueError: If a list input's length does not match the hypotheses, or if
      `probe_counts` is a dict while no hypothesis ids are available.
  """
  del use_boosting
  # Hypothesis ids fix the order every per-hypothesis input is read in. A dict
  # input is looked up by id; a list must already be in that order.
  h_ids: list[str] | None = None
  reference = ""
  if isinstance(terminal_statuses, Mapping):
    h_ids, reference = list(terminal_statuses.keys()), "terminal_statuses"
  elif final_e_values is not None:
    h_ids, reference = list(final_e_values.keys()), "final_e_values"
  if h_ids is not None:
    statuses: list[str] = [
        str(s)
        for s in align_to_ids(
            terminal_statuses, h_ids, "A", "terminal_statuses", reference
        )
    ]
    counts: list[int] = [
        int(c)
        for c in align_to_ids(probe_counts, h_ids, 1, "probe_counts", reference)
    ]
  else:
    # Both positional, or positional statuses with keyed counts. Keyed counts
    # have no order to line up with, so they are rejected.
    if isinstance(probe_counts, Mapping):
      raise ValueError(
          "probe_counts is keyed by hypothesis id but terminal_statuses is a"
          " list; pass terminal_statuses (or final_e_values) keyed by id too."
      )
    statuses = [str(s) for s in terminal_statuses]
    counts = [
        int(c)
        for c in align_to_ids(
            probe_counts,
            [str(i) for i in range(len(statuses))],
            1,
            "probe_counts",
            "terminal_statuses",
        )
    ]

  k_obs = len(statuses)
  if k_obs == 0:
    return DiscoPostHocCertificate(
        epsilon_post=0.0,
        epsilon_post_boost=0.0,
        delta_post=0.0,
        posthoc_unboosted=0.0,
        posthoc_boosted=0.0,
        confidence=1.0,
        k_obs=0,
    )

  effective_kappa = kappa if kappa is not None else null_tables.kappa
  effective_n_max = (
      n_max_override if n_max_override is not None else null_tables.n_max
  )

  # delta_post = K_obs * P_H(n_max)
  delta_post = compute_delta_post(
      counts, null_tables.P_H_table, null_tables.n_max
  )

  # epsilon_sum = sum over g where D_g in {L, B_1, ..., B_J, A} of E_{D_g}(n_g)
  if final_e_values is not None and h_ids is not None:
    exact_vals: list[float | None] | None = [
        final_e_values.get(h_ids[i], 1.0) for i in range(k_obs)
    ]
  elif final_e_values is not None:
    exact_vals = [None for _ in range(k_obs)]
  else:
    exact_vals = None
  epsilon_sum = compute_epsilon_sum(
      statuses, counts, null_tables, exact_e_values=exact_vals
  )

  epsilon_post = alpha_bh * (1.0 / k_obs) * epsilon_sum

  # epsilon_post_boost = (1/k_obs) * sum phi(D_g, n_g, K; alpha_bh)
  if final_e_values is not None and h_ids is not None:
    epsilon_post_boost = compute_continuous_phi_bar(
        alpha_bh,
        effective_kappa,
        dict(zip(h_ids, statuses)),
        dict(zip(h_ids, counts)),
        final_e_values,
        null_tables,
        effective_n_max,
    )
  else:
    epsilon_post_boost = compute_phi_bar(
        alpha_bh, effective_kappa, statuses, counts, k_obs, null_tables
    )

  posthoc_unboosted = epsilon_post + delta_post
  posthoc_boosted = epsilon_post_boost + delta_post
  confidence = 1.0 - delta_post

  return DiscoPostHocCertificate(
      epsilon_post=epsilon_post,
      epsilon_post_boost=epsilon_post_boost,
      delta_post=delta_post,
      posthoc_unboosted=posthoc_unboosted,
      posthoc_boosted=posthoc_boosted,
      confidence=confidence,
      k_obs=k_obs,
  )
