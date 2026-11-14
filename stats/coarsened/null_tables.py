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

"""Lookup tables and simulation for null random walk with coarsened e-processes."""

import dataclasses
import time

import numpy as np
from scipy import stats
import tqdm

from disco.stats import boosting
from disco.stats.coarsened import status as status_lib


# Field names match the paper's mathematical notation (E_A, E_L, P_H).
# pylint: disable=invalid-name
@dataclasses.dataclass
class NullCoarsenedTables:
  """Lookup tables for null random walk with coarsened e-processes."""

  num_bins: int  # active sub-bins in [e_min, e_max) (>= 1)
  bin_edges: np.ndarray  # shape (num_bins+1,) from e_min to e_max
  E_A_table: (
      np.ndarray
  )  # shape (n_max,) — E_A(m) = E[M^free_m | survives m steps in strip]
  E_A_max: float  # max_m E_A(m)
  E_L_table: (
      np.ndarray
  )  # shape (n_max,) — E_L(m) = E[M^free_T | exits low by step m]
  P_H_table: np.ndarray  # shape (n_max,) — P(exit high within m probes)
  P_surv_table: np.ndarray  # shape (n_max,) — P(T > m), survival probability
  P_H_max: float  # P_H(n_max)
  boost_factors_grid: np.ndarray  # shape (n_max, k_max) — b(m,k) boost factors
  phi_grid: np.ndarray  # shape (n_max, k_max) — phi(A,m,k)
  phi_max: float  # max_{m,k} max(phi(A,m,k), phi(L,m,k))
  analytical_ceiling: float  # max_m alpha/P(T>m)
  alpha: float  # the alpha used to compute boost factors
  kappa: float  # calibrator parameter
  e_min: float
  e_max: float
  n_max: int
  k_max: int
  r_hat_transition: (
      np.ndarray
  )  # shape (num_bins+2, num_bins+2) — E[log(E_m) | b_prev, b_curr]
  r_hat_transition_counts: (
      np.ndarray
  )  # shape (num_bins+2, num_bins+2) — sample counts per cell
  r_hat_marginal: np.ndarray  # shape (num_bins+2,) — E[log(E_m) | b_curr]
  min_transition_samples: int = 50
  bin_log_centers: np.ndarray | None = (
      None  # shape (num_bins+2,) — log centers of bins
  )
  phi_L_grid: np.ndarray | None = None  # shape (n_max, k_max) — phi(L,m,k)
  E_bin_table: np.ndarray | None = (
      None  # shape (num_bins, n_max) — E[M^free_m | survives, M_m in B_j]
  )
  E_bin_max: float = 1.0  # max_{j,m} E_bin_table[j,m]
  phi_bin_grid: np.ndarray | None = (
      None  # shape (num_bins, n_max, k_max) — phi(B_j, m, k)
  )
  # Raw MC paths stored for recompute_phi_at_alpha (computing phi at different
  # alpha_bh values post-hoc). Memory cost: ~80 MB at n_sim=200k, n_max=50.
  # Set to None to save memory if post-hoc alpha recomputation is not needed.
  mc_m_free: np.ndarray | None = None
  mc_survived_cum: np.ndarray | None = None
  mc_first_exit_idx: np.ndarray | None = None
  mc_exited_low: np.ndarray | None = None
  log_E_L: float = 0.0
  log_E_H: float = 0.0
  discrete_null: bool = False
  discrete_chunk_size: int | None = None
  discrete_baseline_rate: float | None = None
  enforce_exit_low: bool = True
# pylint: enable=invalid-name


def get_expected_null_e(
    status: str | status_lib.TerminalStatus,
    probe_count: int,
    null_tables: NullCoarsenedTables,
    *,
    exact_e: float | None = None,
) -> float:
  """Returns expected null e-value E_{D_g}(n_g) for a hypothesis.

  For table-based coarsened modes (binary/ternary or multi-bin), uses
  precomputed E_A, E_L, and E_bin tables. For unbinned e-values, pass exact_e
  to use the exact observed e-value for active/survived hypotheses.

  Args:
    status: Terminal status ('A', 'L', 'H', 'B_j').
    probe_count: Number of probes n_g.
    null_tables: Precomputed NullCoarsenedTables.
    exact_e: For unbinned e-values — if provided, used as E_{A}(n_g) instead of
      the table lookup. Ignored for L/H/B statuses.

  Returns:
    Expected null e-value for this hypothesis.
  """
  idx = min(max(1, probe_count), null_tables.n_max) - 1
  st = status_lib.parse_status(status)
  if st.is_exit_high:
    return 0.0
  if st.is_exit_low:
    return float(null_tables.E_L_table[idx])
  if exact_e is not None and st.is_active and not st.is_binned:
    return exact_e
  if (
      st.is_active
      and st.bin_idx is not None
      and null_tables.E_bin_table is not None
  ):
    b_idx = min(max(1, st.bin_idx), null_tables.num_bins) - 1
    return float(null_tables.E_bin_table[b_idx, idx])
  return float(null_tables.E_A_table[idx])


def lookup_transition_score(
    tables: NullCoarsenedTables,
    prev_bin: int,
    curr_bin: int,
    prior_base: float = 10.0,
    prior_scale: float = 10.0,
) -> float:
  """Returns bin-transition score with adaptive Bayesian shrinkage towards midpoint.

  Args:
    tables: Precomputed NullCoarsenedTables instance.
    prev_bin: Source bin index (0=L, 1..num_bins=active, num_bins+1=H).
    curr_bin: Destination bin index.
    prior_base: Base pseudo-count N_0 for Bayesian shrinkage.
    prior_scale: Resolution scale parameter.

  Returns:
    Estimated expected log e-value increment for this transition.
  """
  n_count = float(tables.r_hat_transition_counts[prev_bin, curr_bin])

  if tables.bin_log_centers is not None:
    midpoint_inc = float(
        tables.bin_log_centers[curr_bin] - tables.bin_log_centers[prev_bin]
    )
  else:
    num_bins = tables.r_hat_transition.shape[0] - 2
    if num_bins > 0:
      centers = np.zeros(num_bins + 2)
      centers[num_bins + 1] = np.log(tables.e_max)
      if tables.e_min > 0.0:
        edges = np.geomspace(tables.e_min, tables.e_max, num_bins + 1)
        centers[0] = np.log(tables.e_min)
        centers[1 : num_bins + 1] = 0.5 * (
            np.log(edges[:-1]) + np.log(edges[1:])
        )
      midpoint_inc = float(centers[curr_bin] - centers[prev_bin])
    else:
      midpoint_inc = float(tables.r_hat_marginal[curr_bin])

  # Adaptive pseudo-count scaling with active bin resolution:
  # At few active bins (coarse), empirical null data dominates.
  # At many active bins (fine), midpoint precision increases, so midpoint prior
  # dominates.
  num_bins = tables.num_bins
  n0 = prior_base * (1.0 + num_bins / prior_scale)

  emp_val = (
      float(tables.r_hat_transition[prev_bin, curr_bin]) if n_count > 0 else 0.0
  )
  return float((n_count * emp_val + n0 * midpoint_inc) / (n_count + n0))


def compute_null_coarsened_tables(
    kappa: float,
    e_min: float,
    e_max: float,
    n_max: int,
    k_max: int,
    num_bins: int = 1,
    alpha: float = 0.05,
    n_sim: int = 200_000,
    seed: int = 42,
    discrete_null: bool = False,
    discrete_chunk_size: int = 10,
    discrete_baseline_rate: float = 0.01,
    enforce_exit_low: bool = True,
) -> NullCoarsenedTables:
  """Computes null coarsened lookup tables via Monte Carlo simulation.

  Generates free null random walks and computes survival, exit, and boost
  factor tables needed for DISCO FDR bounds.

  Args:
    kappa: Calibrator parameter in (0, 1).
    e_min: Lower strip boundary.
    e_max: Upper strip boundary.
    n_max: Maximum number of probes (walk length).
    k_max: Maximum number of hypotheses for boost factor grid.
    num_bins: Number of log-spaced active sub-bins in [e_min, e_max) (>= 1).
    alpha: Significance level for boost factor computation.
    n_sim: Number of Monte Carlo trajectories.
    seed: RNG seed for reproducibility.
    discrete_null: If True, simulate null steps using discrete Binomial failure
      counts and non-randomized one-sided p-values instead of continuous U(0,1).
    discrete_chunk_size: Number of queries per chunk for discrete null (N).
    discrete_baseline_rate: Background null baseline failure rate (p_0).
    enforce_exit_low: If True, walks absorb at the lower boundary e_min. If
      False, only the upper boundary e_max is absorbing during the walk.

  Returns:
    NullCoarsenedTables with all precomputed lookup tables.
  """
  t_start = time.time()

  rng = np.random.default_rng(seed)
  bin_edges = status_lib.compute_strip_bins(e_min, e_max, num_bins)

  # --- Step 1: Generate free null random walk paths ---
  if discrete_null:
    failures = rng.binomial(
        discrete_chunk_size, discrete_baseline_rate, size=(n_sim, n_max)
    )
    p_strictly_greater = 1.0 - stats.binom.cdf(
        failures, discrete_chunk_size, discrete_baseline_rate
    )
    p_equal = stats.binom.pmf(
        failures, discrete_chunk_size, discrete_baseline_rate
    )
    p_vals = np.clip(p_strictly_greater + p_equal, 1e-20, 1.0)
  else:
    # E_i = kappa * p_i^(kappa-1) where p_i ~ U(0,1)
    p_vals = rng.uniform(size=(n_sim, n_max))

  e_vals = kappa * (p_vals ** (kappa - 1.0))

  # Running product M^free_m = prod_{i=1}^m E_i
  # Work in log-space for strip checks to avoid float64 overflow at extreme
  # parameters (small kappa, large n_max). Only exponentiate for values
  # needed in expectations (in-strip and low-exit, which are bounded).
  log_e_vals = np.log(e_vals)
  log_m_free = np.cumsum(log_e_vals, axis=1)  # shape (n_sim, n_max)
  log_e_min = float(np.log(e_min)) if e_min > 0.0 else -float("inf")
  log_e_max = np.log(e_max)

  # --- Step 2-3: Track exit status (log-space comparisons) ---
  if enforce_exit_low:
    in_strip = (log_m_free >= log_e_min) & (log_m_free < log_e_max)
  else:
    in_strip = log_m_free < log_e_max

  # Cumulative survival: walk stays in strip for all steps 1..m
  survived_cum = np.cumprod(in_strip, axis=1)  # shape (n_sim, n_max)

  # First exit step (0-indexed): first step where in_strip is False
  exited = ~in_strip
  ever_exits = np.any(exited, axis=1)  # (n_sim,)
  first_exit_idx = np.argmax(exited, axis=1)  # 0-indexed, (n_sim,)
  # For walks that never exit, set sentinel to n_max
  first_exit_idx = np.where(ever_exits, first_exit_idx, n_max)

  # Exit direction: compare in log-space to avoid overflow
  first_exit_log_e = np.zeros(n_sim)
  exit_mask = ever_exits
  first_exit_log_e[exit_mask] = log_m_free[
      np.where(exit_mask)[0], first_exit_idx[exit_mask]
  ]
  exited_low = exit_mask & (first_exit_log_e < log_e_min)
  exited_high = exit_mask & (first_exit_log_e >= log_e_max)

  # Exponentiate with clipping for expectation computations.
  # High-exit paths (which can overflow) only contribute to P_H counts,
  # never to E_A/E_L/E_bin expectations, so clipping is safe.
  m_free = np.exp(np.clip(log_m_free, -700, 700))

  # --- Step 4: Compute tables ---
  p_surv_table = np.zeros(n_max)
  p_h_table = np.zeros(n_max)
  e_a_table = np.zeros(n_max)
  e_l_table = np.zeros(n_max)
  e_bin_table = np.zeros((num_bins, n_max))

  for m_idx in range(n_max):
    # Survivors at step m: walk stayed in strip for all steps 1..m
    surv_mask = survived_cum[:, m_idx].astype(bool)
    p_surv_table[m_idx] = np.mean(surv_mask)

    # Exit high by step m: exited high AND first exit step <= m
    exit_high_by_m = exited_high & (first_exit_idx <= m_idx)
    p_h_table[m_idx] = np.mean(exit_high_by_m)

    # E_A(m) = E[M^free_m | survives m steps and in active strip [e_min, e_max)]
    active_a_mask = (
        surv_mask
        if enforce_exit_low
        else (surv_mask & (log_m_free[:, m_idx] >= log_e_min))
    )
    if np.any(active_a_mask):
      e_a_table[m_idx] = np.mean(m_free[active_a_mask, m_idx])
    else:
      e_a_table[m_idx] = 0.0

    # E_bin_table[j, m_idx] = E[M^free_m | survives m steps and M_m in B_{j+1}]
    for b_idx in range(num_bins):
      b_low = bin_edges[b_idx]
      b_high = bin_edges[b_idx + 1]
      bin_mask = (
          surv_mask & (m_free[:, m_idx] >= b_low) & (m_free[:, m_idx] < b_high)
      )
      if np.any(bin_mask):
        e_bin_table[b_idx, m_idx] = np.mean(m_free[bin_mask, m_idx])
      else:
        e_bin_table[b_idx, m_idx] = np.sqrt(b_low * b_high)

    # E_L(m) = E[M^free_m | M^free_1..M^free_{m-1} in strip, M^free_m < e_min]
    # Conditioned on exiting low at step m (or landing < e_min at step m when
    # enforce_exit_low=False).
    exit_low_at_m = (
        exited_low & (first_exit_idx == m_idx)
        if enforce_exit_low
        else (surv_mask & (log_m_free[:, m_idx] < log_e_min))
    )
    if np.any(exit_low_at_m):
      e_l_table[m_idx] = np.mean(m_free[exit_low_at_m, m_idx])
    else:
      e_l_table[m_idx] = 0.0

  # --- Step 4b: Bin-transition score tables for spatial GP ---
  num_total_bins = num_bins + 2  # 0=L, 1..num_bins=active, num_bins+1=H

  # Prepend M_0 = 1.0 for "previous state at step 0"
  m_free_with_init = np.concatenate(
      [np.ones((n_sim, 1)), m_free], axis=1
  )  # shape (n_sim, n_max+1)

  # Map all positions to bin indices
  bin_idx_all = np.zeros((n_sim, n_max + 1), dtype=int)
  for col in range(n_max + 1):
    bin_idx_all[:, col] = status_lib.map_to_bin_idx(
        m_free_with_init[:, col], e_min, e_max, bin_edges, num_bins
    )

  # Log e-value increments per step: log(E_m) = log(M_m) - log(M_{m-1})
  log_increments = np.diff(
      np.log(np.maximum(m_free_with_init, 1e-300)), axis=1
  )  # shape (n_sim, n_max)

  r_hat_sums = np.zeros((num_total_bins, num_total_bins))
  r_hat_counts = np.zeros((num_total_bins, num_total_bins), dtype=np.int64)

  for step in range(n_max):
    b_prev = bin_idx_all[:, step]
    b_curr = bin_idx_all[:, step + 1]
    log_inc = log_increments[:, step]

    # Only count paths still active at step m-1
    if step == 0:
      valid = np.ones(n_sim, dtype=bool)
    elif enforce_exit_low:
      valid = (b_prev >= 1) & (b_prev <= num_bins)
    else:
      valid = survived_cum[:, step - 1].astype(bool)

    if np.any(valid):
      bp_v = b_prev[valid]
      bc_v = b_curr[valid]
      li_v = log_inc[valid]
      flat_idx = bp_v * num_total_bins + bc_v
      np.add.at(r_hat_sums.reshape(-1), flat_idx, li_v)
      counts = np.bincount(flat_idx, minlength=num_total_bins * num_total_bins)
      r_hat_counts.reshape(-1)[:] += counts

  # Normalize: mean log increment per transition
  r_hat_transition = np.zeros((num_total_bins, num_total_bins))
  nonzero = r_hat_counts > 0
  r_hat_transition[nonzero] = r_hat_sums[nonzero] / r_hat_counts[nonzero]

  # Marginal fallback: E[log(E_m) | M_m in b_curr] (pool across source bins)
  r_hat_marginal = np.zeros(num_total_bins)
  for bc in range(num_total_bins):
    col_total = r_hat_counts[:, bc].sum()
    if col_total > 0:
      r_hat_marginal[bc] = r_hat_sums[:, bc].sum() / col_total

  # Expected log e-value for exit states (representative coarsened scores)
  # High exit: memoryless property gives
  # E[log M | M >= e_max] = log(e_max) + (1 - kappa)
  log_e_h = float(np.log(e_max) + (1.0 - kappa))
  # Low exit: compute from MC paths that exited low (or visited < e_min)
  if enforce_exit_low and np.any(exited_low):
    low_exit_indices = first_exit_idx[exited_low]
    log_e_l = float(
        np.mean(
            np.log(
                np.maximum(
                    m_free[np.where(exited_low)[0], low_exit_indices], 1e-300
                )
            )
        )
    )
  elif not enforce_exit_low and np.any(
      survived_cum.astype(bool) & (log_m_free < log_e_min)
  ):
    low_visit_mask = survived_cum.astype(bool) & (log_m_free < log_e_min)
    log_e_l = float(np.mean(np.log(np.maximum(m_free[low_visit_mask], 1e-300))))
  else:
    log_e_l = float(np.log(e_min)) if e_min > 0.0 else 0.0

  # Compute geometric log centers for all bins (0=L, 1..num_bins=active,
  # num_bins+1=H)
  bin_log_centers = np.zeros(num_total_bins)
  bin_log_centers[0] = log_e_l
  bin_log_centers[num_bins + 1] = log_e_h
  if num_bins > 0:
    if e_min > 0.0:
      bin_log_centers[1 : num_bins + 1] = 0.5 * (
          np.log(bin_edges[:-1]) + np.log(bin_edges[1:])
      )
    else:
      bin_log_centers[1 : num_bins + 1] = 0.0

  # --- Step 5: Compute boost_factors_grid, phi_grid, and phi_L_grid ---
  boost_factors_grid = np.ones((n_max, k_max))
  phi_grid = np.zeros((n_max, k_max))
  phi_l_grid = np.zeros((n_max, k_max))
  phi_bin_grid = np.zeros((num_bins, n_max, k_max))

  total_pairs = n_max * k_max
  pbar = tqdm.tqdm(
      total=total_pairs,
      desc="[disco] Computing coarsened boost factors & phi",
  )

  for m_idx in range(n_max):
    m = m_idx + 1
    surv_mask = survived_cum[:, m_idx].astype(bool)
    active_a_mask = (
        surv_mask
        if enforce_exit_low
        else (surv_mask & (log_m_free[:, m_idx] >= log_e_min))
    )
    low_exit_mask = (
        exited_low & (first_exit_idx == m_idx)
        if enforce_exit_low
        else (surv_mask & (log_m_free[:, m_idx] < log_e_min))
    )
    m_free_m = m_free[:, m_idx]  # M^free_m for all paths

    # Pre-mask for sub-bins at step m
    bin_masks = [
        surv_mask & (m_free_m >= bin_edges[b]) & (m_free_m < bin_edges[b + 1])
        for b in range(num_bins)
    ]

    for k_idx in range(k_max):
      k = k_idx + 1

      b = boosting.compute_boost_factor(alpha, kappa, m, k, use_exact_t=True)
      boost_factors_grid[m_idx, k_idx] = b

      # phi_grid: E[T_k(alpha * b * M^free_m) | survives m steps in A]
      if np.any(active_a_mask):
        surv_m_free = m_free_m[active_a_mask]
        phi_vals = boosting.staircase_t(alpha * b * surv_m_free, k)
        phi_grid[m_idx, k_idx] = np.mean(phi_vals)
      else:
        phi_grid[m_idx, k_idx] = 0.0

      # phi_L_grid: E[T_k(alpha * b * M^free_m) | exits low at step m]
      if np.any(low_exit_mask):
        low_m_free = m_free_m[low_exit_mask]
        phi_l_vals = boosting.staircase_t(alpha * b * low_m_free, k)
        phi_l_grid[m_idx, k_idx] = np.mean(phi_l_vals)
      else:
        phi_l_grid[m_idx, k_idx] = 0.0

      # phi_bin_grid: per-sub-bin expectation
      for b_idx in range(num_bins):
        if np.any(bin_masks[b_idx]):
          b_m_free = m_free_m[bin_masks[b_idx]]
          phi_bin_grid[b_idx, m_idx, k_idx] = np.mean(
              boosting.staircase_t(alpha * b * b_m_free, k)
          )
        else:
          b_mid = np.sqrt(bin_edges[b_idx] * bin_edges[b_idx + 1])
          phi_bin_grid[b_idx, m_idx, k_idx] = float(
              boosting.staircase_t(alpha * b * b_mid, k)
          )

      pbar.update(1)

  pbar.close()

  # --- Steps 7-11: Derived quantities ---
  phi_max = float(max(np.max(phi_grid), np.max(phi_l_grid)))
  e_a_max = float(np.max(e_a_table))
  e_bin_max = float(np.max(e_bin_table))
  p_h_max = float(p_h_table[n_max - 1])

  # analytical_ceiling = max_m alpha / P(T > m)
  # Avoid division by zero for P_surv = 0
  with np.errstate(divide="ignore", invalid="ignore"):
    ceiling_arr = np.where(p_surv_table > 0, alpha / p_surv_table, float("inf"))
  analytical_ceiling = float(np.max(ceiling_arr))

  # --- Verify assumption ---
  max_boost = float(np.max(boost_factors_grid))
  if alpha * max_boost * e_min >= 1.0:
    print(
        "[disco] WARNING: alpha * max(b) * e_min ="
        f" {alpha * max_boost * e_min:.4f} >= 1."
        " Staircase assumption may be violated.",
        flush=True,
    )

  t_elapsed = time.time() - t_start
  print(
      f"[disco] compute_null_coarsened_tables complete in {t_elapsed:.1f}s"
      f" (n_max={n_max}, k_max={k_max}, n_sim={n_sim}).",
      flush=True,
  )

  return NullCoarsenedTables(
      num_bins=num_bins,
      bin_edges=bin_edges,
      E_A_table=e_a_table,
      E_A_max=e_a_max,
      E_L_table=e_l_table,
      P_H_table=p_h_table,
      P_surv_table=p_surv_table,
      P_H_max=p_h_max,
      boost_factors_grid=boost_factors_grid,
      phi_grid=phi_grid,
      phi_max=phi_max,
      analytical_ceiling=analytical_ceiling,
      alpha=alpha,
      kappa=kappa,
      e_min=e_min,
      e_max=e_max,
      n_max=n_max,
      k_max=k_max,
      r_hat_transition=r_hat_transition,
      r_hat_transition_counts=r_hat_counts,
      r_hat_marginal=r_hat_marginal,
      min_transition_samples=50,
      bin_log_centers=bin_log_centers,
      phi_L_grid=phi_l_grid,
      E_bin_table=e_bin_table,
      E_bin_max=e_bin_max,
      phi_bin_grid=phi_bin_grid,
      mc_m_free=m_free,
      mc_survived_cum=survived_cum,
      mc_first_exit_idx=first_exit_idx,
      mc_exited_low=exited_low,
      log_E_L=log_e_l,
      log_E_H=log_e_h,
      discrete_null=discrete_null,
      discrete_chunk_size=discrete_chunk_size if discrete_null else None,
      discrete_baseline_rate=discrete_baseline_rate if discrete_null else None,
      enforce_exit_low=enforce_exit_low,
  )
