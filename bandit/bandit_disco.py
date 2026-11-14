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

"""Core DISCO and baseline bandit runners for Bernoulli and Contextual Bandits.

Implements:
1. run_disco_bandit: DISCO sequential testing with coarsened e-processes, strip
   deactivation [e_min, e_max), probe cap n_max, active-slot UCB,
   transition-score expected log E estimation, and exact post-hoc
   calibrated/boosted e-BH on the observed K_obs.
2. run_offline_ebh_bandit: Unconstrained multi-armed bandit exploration followed
   by offline e-BH at level alpha with pre-hoc fixed K_bar.
3. run_type1_bandit: Unconstrained exploration with direct 1/alpha
   thresholding (no multiple-testing correction).
"""

import dataclasses

import numpy as np

from disco.bandit import bandit_env
from disco.stats import boosting
from disco.stats import calibrators
from disco.stats import multiple_testing
from disco.stats.coarsened import calibration as calibration_lib
from disco.stats.coarsened import certificates as cert_lib
from disco.stats.coarsened import null_tables as null_tables_lib
from disco.stats.coarsened import status as status_lib

ArmInfo = bandit_env.ArmInfo
BanditEnv = bandit_env.BanditEnv

# Candidate pool sizes drawn per round from the environment.
_CONTEXTUAL_CANDIDATE_POOL_SIZE = 5
_BERNOULLI_CANDIDATE_POOL_SIZE = 1


@dataclasses.dataclass
class BanditConfig:
  """Configuration for a bandit DISCO or Offline e-BH run."""

  env_type: str = "bernoulli"  # "bernoulli" or "contextual"
  budget: int = 500
  chunk_size: int = 10
  alpha: float = 0.05
  kappa: float = 0.5
  e_min: float = 0.01
  e_max: float = 100.0
  enforce_exit_low: bool = True
  n_max: int = 10
  k_bar: int | None = None
  # 0 = unbinned/exact, 1 = single active bin [e_min, e_max), >1 = multi-bin
  num_active_bins: int = 1
  alpha_calibration_mode: str = (
      "post_hoc"  # "nominal", "pre_hoc", "post_hoc", "good_event"
  )
  use_boosting: bool = True
  n_sim: int = 100_000
  gp_beta: float = 2.0  # UCB exploration factor
  ridge_lambda: float = 1.0  # Regularization parameter
  p_value_mode: str = (
      "randomized"  # "randomized", "discrete", "discrete_matched"
  )
  dimension: int = 3
  num_vulnerable_clusters: int = 1
  cluster_weight_min: float = 7.0
  cluster_weight_max: float = 10.0
  cluster_var_min: float = 0.05
  cluster_var_max: float = 0.15
  pi_1: float = 0.10
  baseline_rate_logit: float = -4.6
  alternative_rate: float = 0.20
  delta_threshold: float = 0.05
  delta_multiplier: float = 5.0
  seed: int = 42


@dataclasses.dataclass
class BanditResult:
  """Structured output metrics from a bandit simulation run."""

  runner_type: str
  env_type: str
  total_hypotheses_tested: int
  k_reference: int
  rejected_hypotheses: set[str]
  true_positives: int
  false_positives: int
  empirical_fdr: float
  recall: float
  macro_recall: float  # Cluster-balanced recall across vulnerable clusters
  cluster_coverage: (
      float  # Percentage of unique vulnerable clusters with >= 1 discovery
  )
  rejection_threshold: float
  nominal_alpha_used: float
  avg_probes_null: float
  avg_probes_alt: float
  final_e_values: dict[str, float]
  boosted_e_values: dict[str, float]
  probe_counts: dict[str, int]
  terminal_statuses: dict[str, str]
  is_alternative_map: dict[str, bool] = dataclasses.field(default_factory=dict)
  total_true_positives: int = (
      0  # Total alternatives in the full pool (recall denom)
  )
  cluster_true_positives: dict[int, int] = dataclasses.field(
      default_factory=dict
  )
  epsilon_post: float | None = None
  delta_post: float | None = None
  posthoc_certificate: float | None = None
  penalized_cluster_coverage: float = 0.0


def compute_penalized_cluster_coverage(
    cluster_coverage: float,
    delta_post: float | None,
    delta_threshold: float = 0.05,
    delta_multiplier: float = 5.0,
) -> float:
  """Computes penalized cluster coverage with a hinge penalty on post-hoc delta.

  Applies a hinge penalty when delta_post exceeds delta_threshold:
    penalty = delta_multiplier * max(0.0, (delta_post - delta_threshold) *
    100.0)
    penalized_coverage = cluster_coverage - penalty

  Args:
    cluster_coverage: Percentage of unique vulnerable clusters discovered (0 to
      100).
    delta_post: Post-hoc bad-event probability sum (0 to 1), or None.
    delta_threshold: Maximum tolerated delta_post before penalty kicks in
      (default 0.05).
    delta_multiplier: Penalty deducted from coverage (%) per 1% (0.01)
      delta_post excess (default 5.0).

  Returns:
    Penalized cluster coverage percentage.
  """
  d_val = delta_post if delta_post is not None else 0.0
  delta_excess_pct = max(0.0, (d_val - delta_threshold) * 100.0)
  penalty = delta_multiplier * delta_excess_pct
  return cluster_coverage - penalty


# =============================================================================
# Shared config-derived helpers
# =============================================================================


def sigmoid(logit: float) -> float:
  """Logistic function, written exactly as the simulators always have."""
  return 1.0 / (1.0 + np.exp(-logit))


def resolve_k_bar(config: BanditConfig) -> int:
  """Pre-hoc hypothesis-count bound K_bar (defaults to budget // chunk)."""
  if config.k_bar is not None:
    return config.k_bar
  return max(1, config.budget // config.chunk_size)


def build_null_tables(
    config: BanditConfig, k_bar: int
) -> null_tables_lib.NullCoarsenedTables:
  """Precomputes the null coarsened tables implied by a config (seeded, deterministic)."""
  unbinned_e_values = config.num_active_bins == 0
  effective_bins = 1 if unbinned_e_values else config.num_active_bins
  return null_tables_lib.compute_null_coarsened_tables(
      kappa=config.kappa,
      e_min=config.e_min,
      e_max=config.e_max,
      n_max=config.n_max,
      k_max=k_bar,
      num_bins=effective_bins,
      alpha=config.alpha,
      n_sim=config.n_sim,
      seed=config.seed,
      discrete_null=config.p_value_mode == "discrete_matched",
      discrete_chunk_size=config.chunk_size,
      discrete_baseline_rate=sigmoid(config.baseline_rate_logit),
      enforce_exit_low=config.enforce_exit_low,
  )


def _randomized_p(config: BanditConfig) -> bool:
  return config.p_value_mode == "randomized"


# =============================================================================
# Candidate Selectors (Contextual Bandit)
# =============================================================================


class _ContextualSelectorBase:
  """Shared Ridge-regression acquisition logic for contextual arm selection.

  Fits a probe-count-weighted closed-form Ridge model on (feature, score)
  pairs and scores arms by UCB. Subclasses define how the per-arm regression
  target is accumulated.
  """

  def __init__(
      self,
      dimension: int,
      beta: float,
      ridge_lambda: float,
      seed: int,
  ):
    self.dimension = dimension
    self.beta = beta
    self.ridge_lambda = ridge_lambda
    self.rng = np.random.default_rng(seed)

    self.arm_features: dict[str, np.ndarray] = {}
    self.arm_scores: dict[str, float] = {}
    self.arm_probes: dict[str, int] = {}

  def _ridge_target(self, arm_id: str, n_probes: int) -> float:
    """Coefficient multiplying ``feature`` in the Ridge right-hand side."""
    raise NotImplementedError

  def _fit_ridge(self) -> tuple[np.ndarray, np.ndarray]:
    """Fits the probe-weighted Ridge posterior; returns (w_hat, A^{-1})."""
    dim = self.dimension
    a_mat = self.ridge_lambda * np.eye(dim)
    b_vec = np.zeros(dim)
    for h, feat in self.arm_features.items():
      n_h = self.arm_probes.get(h, 1)
      a_mat += n_h * np.outer(feat, feat)
      b_vec += self._ridge_target(h, n_h) * feat
    w_hat = np.linalg.solve(a_mat, b_vec)
    a_inv = np.linalg.inv(a_mat)
    return w_hat, a_inv

  def _acquisition_argmax(
      self,
      all_options: list[ArmInfo],
      w_hat: np.ndarray,
      a_inv: np.ndarray,
  ) -> ArmInfo:
    """Returns the first arm attaining the maximal UCB acquisition score."""
    best_arm = all_options[0]
    best_score = -1e9
    for arm in all_options:
      if arm.feature is None:
        continue
      x_cand = arm.feature
      mu = float(x_cand @ w_hat)
      sigma = float(np.sqrt(np.maximum(0.0, x_cand @ a_inv @ x_cand)))
      score = mu + self.beta * sigma
      if score > best_score:
        best_score = score
        best_arm = arm
    return best_arm


class DISCOContextualCandidateSelector(_ContextualSelectorBase):
  """Selects arms for DISCO using coarsened expected log E transition scores.

  Respects e-value coarsening by regressing on cumulative expected log e-value
  increments looked up from null tables via
  ``null_tables_lib.lookup_transition_score`` (binned mode, finite |P|), or on
  exact clamped log e-values (unbinned mode, |P| = infinity,
  ``num_active_bins = 0``).
  """

  def __init__(
      self,
      dimension: int,
      null_tables: null_tables_lib.NullCoarsenedTables,
      beta: float = 2.0,
      ridge_lambda: float = 1.0,
      seed: int = 42,
  ):
    super().__init__(
        dimension=dimension,
        beta=beta,
        ridge_lambda=ridge_lambda,
        seed=seed,
    )
    self.null_tables = null_tables

  def update_transition(
      self, arm_id: str, feature: np.ndarray, prev_bin: int, curr_bin: int
  ) -> None:
    """Updates the cumulative expected log e-value score (binned mode)."""
    score_inc = null_tables_lib.lookup_transition_score(
        self.null_tables, prev_bin, curr_bin
    )
    self.arm_features[arm_id] = feature
    self.arm_scores[arm_id] = self.arm_scores.get(arm_id, 0.0) + score_inc
    self.arm_probes[arm_id] = self.arm_probes.get(arm_id, 0) + 1

  def update_exact(
      self, arm_id: str, feature: np.ndarray, log_e_score: float
  ) -> None:
    """Updates the cumulative log e-value score directly (unbinned mode)."""
    self.arm_features[arm_id] = feature
    self.arm_scores[arm_id] = log_e_score
    self.arm_probes[arm_id] = self.arm_probes.get(arm_id, 0) + 1

  def _ridge_target(self, arm_id: str, n_probes: int) -> float:
    return self.arm_scores[arm_id]

  def select_arm(
      self,
      active_arms: list[ArmInfo],
      candidate_pool: list[ArmInfo],
  ) -> ArmInfo:
    """Selects the arm (active or candidate) maximizing the acquisition score."""
    all_options = active_arms + candidate_pool
    if not all_options:
      raise ValueError("No arms to select from.")

    # If no history, pick a candidate uniformly at random.
    if not self.arm_features:
      pool = candidate_pool if candidate_pool else all_options
      idx = int(self.rng.integers(0, len(pool)))
      return pool[idx]

    w_hat, a_inv = self._fit_ridge()
    return self._acquisition_argmax(all_options, w_hat, a_inv)


class OfflineContextualCandidateSelector(_ContextualSelectorBase):
  """Unconstrained LinUCB selector for the offline baselines.

  Fits Ridge regression on continuous (clamped) log e-values.
  """

  def __init__(
      self,
      dimension: int,
      beta: float = 2.0,
      ridge_lambda: float = 1.0,
      seed: int = 42,
  ):
    super().__init__(
        dimension=dimension,
        beta=beta,
        ridge_lambda=ridge_lambda,
        seed=seed,
    )

  def update_observations(
      self, arm_id: str, feature: np.ndarray, log_e: float
  ) -> None:
    """Updates the (clamped) log e-value regression target for this arm."""
    self.arm_features[arm_id] = feature
    self.arm_scores[arm_id] = float(np.clip(log_e, -100.0, 100.0))
    self.arm_probes[arm_id] = self.arm_probes.get(arm_id, 0) + 1

  def _ridge_target(self, arm_id: str, n_probes: int) -> float:
    return self.arm_scores.get(arm_id, 0.0)

  def select_next_arm(
      self, discovered_arms: list[ArmInfo], candidate_pool: list[ArmInfo]
  ) -> ArmInfo:
    """Selects the arm (discovered or candidate) maximizing the acquisition score."""
    # If no history, pick a candidate uniformly at random.
    if not self.arm_features:
      idx = int(self.rng.integers(0, len(candidate_pool)))
      return candidate_pool[idx]

    w_hat, a_inv = self._fit_ridge()
    all_options = discovered_arms + candidate_pool
    return self._acquisition_argmax(all_options, w_hat, a_inv)


# =============================================================================
# Active Set Choosers (Selecting Which Active Slot to Probe)
# =============================================================================


def choose_active_slot(
    active_set: dict[str, ArmInfo],
    bin_trajectories: dict[str, list[int]],
    probe_counts: dict[str, int],
    beta: float,
    total_active_probes: int,
    null_tables: null_tables_lib.NullCoarsenedTables,
    init_bin: int,
    rng: np.random.Generator,
    running_e_values: dict[str, float] | None = None,
    unbinned_e_values: bool = False,
) -> str:
  """Chooses which active slot to probe next in DISCO using transition scores or exact e-values.

  Each slot is scored by its cumulative coarsened expected log e-value plus a
  UCB exploration bonus; ties are broken uniformly at random.

  Args:
    active_set: Currently active arms keyed by arm ID.
    bin_trajectories: Sequence of bin indices visited by each arm.
    probe_counts: Number of probes performed on each arm so far.
    beta: UCB exploration bonus coefficient.
    total_active_probes: Total number of probes allocated across active slots.
    null_tables: Precomputed null coarsened tables for transition score lookup.
    init_bin: Initial bin index corresponding to e = 1.0.
    rng: Random number generator used to break ties uniformly.
    running_e_values: Current accumulated e-value per arm (for unbinned mode).
    unbinned_e_values: Whether unbinned e-value mode is enabled.

  Returns:
    The arm ID of the chosen active slot.
  """
  h_ids = list(active_set.keys())
  if len(h_ids) == 1:
    return h_ids[0]

  scores = {}
  for h_id in h_ids:
    n_probes = probe_counts.get(h_id, 0)

    if unbinned_e_values and running_e_values is not None:
      cum_score = float(np.log(max(1e-20, running_e_values.get(h_id, 1.0))))
    else:
      # Compute cumulative coarsened expected log e-value score from bin
      # transitions
      cum_score = 0.0
      prev = init_bin
      for curr in bin_trajectories.get(h_id, []):
        cum_score += null_tables_lib.lookup_transition_score(
            null_tables, prev, curr
        )
        prev = curr

    ucb_bonus = beta * np.sqrt(
        (2.0 * np.log(max(2, total_active_probes))) / max(1, n_probes)
    )
    scores[h_id] = cum_score + float(ucb_bonus)

  max_score = max(scores.values())
  candidates = [h for h in h_ids if scores[h] == max_score]
  return str(rng.choice(candidates))


# =============================================================================
# Shared metrics & result assembly
# =============================================================================


@dataclasses.dataclass(frozen=True)
class _RunMetrics:
  """Discovery metrics shared by every runner."""

  true_positives: int
  false_positives: int
  empirical_fdr: float  # percent
  recall: float  # percent
  macro_recall: float  # percent
  cluster_coverage: float  # percent
  avg_probes_null: float
  avg_probes_alt: float


def _compute_run_metrics(
    config: BanditConfig,
    rejected_set: set[str],
    arms: dict[str, ArmInfo],
    final_e_values: dict[str, float],
    probe_counts: dict[str, int],
    total_true_positives: int,
    cluster_true_positives: dict[int, int] | None,
) -> _RunMetrics:
  """Computes TP/FP, FDR, recall, cluster metrics, and probe averages."""
  true_pos = 0
  false_pos = 0
  for h_id in rejected_set:
    arm = arms[h_id]
    if arm.is_alternative:
      true_pos += 1
    else:
      false_pos += 1

  total_rej = len(rejected_set)
  emp_fdr = (false_pos / max(1, total_rej)) if total_rej > 0 else 0.0

  recall = (
      (true_pos / max(1, total_true_positives)) * 100.0
      if total_true_positives > 0
      else 0.0
  )

  # Compute cluster-balanced (macro) recall across vulnerable clusters
  macro_recall = 0.0
  cluster_coverage = 0.0
  num_vuln_c = (
      config.num_vulnerable_clusters if config.env_type == "contextual" else 1
  )
  if num_vuln_c > 0 and cluster_true_positives:
    cluster_tps_found: dict[int, int] = {}
    for h_id in rejected_set:
      arm = arms[h_id]
      if arm.is_alternative and arm.cluster_id is not None:
        cluster_tps_found[arm.cluster_id] = (
            cluster_tps_found.get(arm.cluster_id, 0) + 1
        )
    rec_sum = 0.0
    covered_clusters = 0
    for c_idx in range(num_vuln_c):
      c_total = cluster_true_positives.get(c_idx, 0)
      if c_total > 0:
        rec_sum += cluster_tps_found.get(c_idx, 0) / c_total
        if cluster_tps_found.get(c_idx, 0) > 0:
          covered_clusters += 1
    macro_recall = (rec_sum / num_vuln_c) * 100.0
    cluster_coverage = (covered_clusters / num_vuln_c) * 100.0
  elif total_true_positives > 0:
    macro_recall = recall
    cluster_coverage = 100.0 if true_pos > 0 else 0.0

  null_probes = [
      probe_counts[h] for h in final_e_values if not arms[h].is_alternative
  ]
  alt_probes = [
      probe_counts[h] for h in final_e_values if arms[h].is_alternative
  ]

  avg_p_null = float(np.mean(null_probes)) if null_probes else 0.0
  avg_p_alt = float(np.mean(alt_probes)) if alt_probes else 0.0

  return _RunMetrics(
      true_positives=true_pos,
      false_positives=false_pos,
      empirical_fdr=emp_fdr * 100.0,
      recall=recall,
      macro_recall=macro_recall,
      cluster_coverage=cluster_coverage,
      avg_probes_null=avg_p_null,
      avg_probes_alt=avg_p_alt,
  )


def _build_result(
    *,
    runner_type: str,
    config: BanditConfig,
    metrics: _RunMetrics,
    arms: dict[str, ArmInfo],
    k_reference: int,
    rejected_set: set[str],
    rejection_threshold: float,
    nominal_alpha_used: float,
    final_e_values: dict[str, float],
    boosted_e_values: dict[str, float],
    probe_counts: dict[str, int],
    terminal_statuses: dict[str, str],
    total_true_positives: int,
    cluster_true_positives: dict[int, int] | None,
    epsilon_post: float | None = None,
    delta_post: float | None = None,
    posthoc_certificate: float | None = None,
) -> BanditResult:
  """Assembles a BanditResult; applies the delta hinge penalty to coverage."""
  penalized_coverage = compute_penalized_cluster_coverage(
      cluster_coverage=metrics.cluster_coverage,
      delta_post=delta_post,
      delta_threshold=config.delta_threshold,
      delta_multiplier=config.delta_multiplier,
  )
  return BanditResult(
      runner_type=runner_type,
      env_type=config.env_type,
      total_hypotheses_tested=len(final_e_values),
      k_reference=k_reference,
      rejected_hypotheses=rejected_set,
      true_positives=metrics.true_positives,
      false_positives=metrics.false_positives,
      empirical_fdr=metrics.empirical_fdr,
      recall=metrics.recall,
      macro_recall=metrics.macro_recall,
      cluster_coverage=metrics.cluster_coverage,
      rejection_threshold=rejection_threshold,
      nominal_alpha_used=nominal_alpha_used,
      avg_probes_null=metrics.avg_probes_null,
      avg_probes_alt=metrics.avg_probes_alt,
      final_e_values=final_e_values,
      boosted_e_values=boosted_e_values,
      probe_counts=probe_counts,
      terminal_statuses=terminal_statuses,
      is_alternative_map={h: arms[h].is_alternative for h in final_e_values},
      total_true_positives=total_true_positives,
      cluster_true_positives=cluster_true_positives or {},
      epsilon_post=epsilon_post,
      delta_post=delta_post,
      posthoc_certificate=posthoc_certificate,
      penalized_cluster_coverage=penalized_coverage,
  )


# =============================================================================
# DISCO Runner
# =============================================================================


def _calibrate_nominal_alpha(
    config: BanditConfig,
    null_tables: null_tables_lib.NullCoarsenedTables,
    *,
    k_bar: int,
    k_total: int,
    delta_post: float,
    e_bar_obs: float,
    e_mean_max: float,
    statuses_list: list[str],
    counts_list: list[int],
    unbinned_e_values: bool,
    final_e_values: dict[str, float],
    terminal_statuses: dict[str, str],
    probe_counts: dict[str, int],
) -> float:
  """Selects the nominal e-BH alpha per ``alpha_calibration_mode``."""
  mode = config.alpha_calibration_mode
  if not config.use_boosting:
    if mode == "nominal":
      return config.alpha
    if mode == "pre_hoc":
      prehoc_bad_event = float(k_bar * null_tables.P_H_max)
      return float(
          np.clip(
              (config.alpha - prehoc_bad_event) / max(1e-10, e_mean_max),
              0.0,
              1.0,
          )
      )
    if mode == "post_hoc":
      return float(
          np.clip((config.alpha - delta_post) / max(1e-10, e_bar_obs), 0.0, 1.0)
      )
    if mode == "good_event":
      return float(np.clip(config.alpha / max(1e-10, e_bar_obs), 0.0, 1.0))
    return config.alpha

  if mode == "nominal":
    return config.alpha
  if mode == "pre_hoc":
    nominal_alpha, _ = calibration_lib.find_optimal_prehoc_alpha_bh(
        alpha_star=config.alpha,
        kappa=config.kappa,
        k_bar=k_bar,
        null_tables=null_tables,
    )
    return nominal_alpha
  if mode in ("good_event", "post_hoc"):
    nominal_alpha, _ = calibration_lib.find_optimal_alpha_bh(
        alpha_star=config.alpha,
        kappa=config.kappa,
        statuses=statuses_list,
        counts=counts_list,
        k_obs=k_total,
        null_tables=null_tables,
        delta_post=delta_post,
        mode=mode,
        final_e_values=(final_e_values if unbinned_e_values else None),
        terminal_statuses_dict=(
            terminal_statuses if unbinned_e_values else None
        ),
        probe_counts_dict=(probe_counts if unbinned_e_values else None),
        n_max=config.n_max if unbinned_e_values else None,
    )
    return nominal_alpha
  return config.alpha


@dataclasses.dataclass
class _DiscoLoopResult:
  """Internal result from the DISCO probing loop (inputs to post-hoc inference)."""

  all_tested_arms: dict[str, ArmInfo]
  final_e_values: dict[str, float]
  probe_counts: dict[str, int]
  terminal_statuses: dict[str, str]


def _run_disco_probing_loop(
    config: BanditConfig,
    env: BanditEnv,
    null_tables: null_tables_lib.NullCoarsenedTables,
    bin_edges: np.ndarray,
) -> _DiscoLoopResult:
  """Runs the DISCO probing loop until the budget is spent."""
  rng = np.random.default_rng(config.seed)
  remaining_budget = config.budget
  chunk_sz = config.chunk_size
  unbinned_e_values = config.num_active_bins == 0
  effective_bins = 1 if unbinned_e_values else config.num_active_bins

  contextual_selector = DISCOContextualCandidateSelector(
      dimension=config.dimension,
      null_tables=null_tables,
      beta=config.gp_beta,
      ridge_lambda=config.ridge_lambda,
      seed=config.seed,
  )

  # Tracking state
  active_arms: dict[str, ArmInfo] = {}
  running_e_values: dict[str, float] = {}
  final_e_values: dict[str, float] = {}
  probe_counts: dict[str, int] = {}
  bin_trajectories: dict[str, list[int]] = {}
  terminal_statuses: dict[str, str] = {}
  prev_bins: dict[str, int] = {}
  all_tested_arms: dict[str, ArmInfo] = {}

  init_bin = int(np.searchsorted(bin_edges, 1.0, side="right"))
  init_bin = max(1, min(init_bin, effective_bins))

  def _active_status(bin_val: int | None) -> str:
    return "A" if config.num_active_bins <= 1 else f"B_{bin_val}"

  total_active_probes = 0

  while remaining_budget > 0:
    exclude_ids = set(all_tested_arms.keys())

    # Step 1: Select arm (fresh candidate or active slot)
    if config.env_type == "contextual":
      cand_pool = env.generate_candidate_pool(
          _CONTEXTUAL_CANDIDATE_POOL_SIZE, exclude_ids=exclude_ids
      )
      all_options = list(active_arms.values()) + cand_pool
      if not all_options:
        break
      selected_arm = contextual_selector.select_arm(
          list(active_arms.values()), cand_pool
      )
      g_id = selected_arm.arm_id
    else:
      cand_pool = env.generate_candidate_pool(
          _BERNOULLI_CANDIDATE_POOL_SIZE, exclude_ids=exclude_ids
      )
      selection_pool = dict(active_arms)
      if cand_pool:
        selection_pool[cand_pool[0].arm_id] = cand_pool[0]
      if not selection_pool:
        break

      g_id = choose_active_slot(
          active_set=selection_pool,
          bin_trajectories=bin_trajectories,
          probe_counts=probe_counts,
          beta=config.gp_beta,
          total_active_probes=total_active_probes,
          null_tables=null_tables,
          init_bin=init_bin,
          rng=rng,
          running_e_values=running_e_values,
          unbinned_e_values=unbinned_e_values,
      )
      selected_arm = selection_pool[g_id]

    # Initialize state if this is a newly selected candidate
    if g_id not in all_tested_arms:
      all_tested_arms[g_id] = selected_arm
      active_arms[g_id] = selected_arm
      running_e_values[g_id] = 1.0
      probe_counts[g_id] = 0
      bin_trajectories[g_id] = []
      prev_bins[g_id] = init_bin

    g_arm = all_tested_arms[g_id]

    # Step 2: Query arm
    q_count = min(chunk_sz, remaining_budget)
    query_res = env.sample_queries(
        g_id, q_count, randomized_p=_randomized_p(config)
    )
    remaining_budget -= q_count
    probe_counts[g_id] = probe_counts.get(g_id, 0) + 1
    total_active_probes += 1

    # Step 3: Update e-value privately
    local_e = calibrators.p_to_e(query_res.p_value, config.kappa)
    running_e_values[g_id] *= local_e
    curr_e = running_e_values[g_id]

    # Step 4/5: Bin mapping & coarsened selector update
    if not unbinned_e_values:
      bin_val = int(
          status_lib.map_to_bin_idx(
              curr_e,
              config.e_min,
              config.e_max,
              bin_edges,
              config.num_active_bins,
          )
      )
      bin_trajectories[g_id].append(bin_val)

      g_prev = prev_bins.get(g_id, init_bin)
      if g_arm.feature is not None:
        contextual_selector.update_transition(
            g_id, g_arm.feature, g_prev, bin_val
        )
      prev_bins[g_id] = bin_val
    else:
      # Unbinned: exact log e-value, clamped to the expected overshoot on exit.
      if g_arm.feature is not None:
        if config.enforce_exit_low and curr_e < config.e_min:
          score = null_tables.log_E_L
        elif curr_e >= config.e_max:
          score = null_tables.log_E_H
        else:
          score = float(np.log(curr_e))
        contextual_selector.update_exact(g_id, g_arm.feature, score)
      bin_val = None

    # Step 6: Deactivation check (L, H, or timeout cap n_max)
    deactivate = False
    status = _active_status(bin_val)
    if curr_e >= config.e_max:
      deactivate = True
      status = "H"
    elif config.enforce_exit_low and curr_e < config.e_min:
      deactivate = True
      status = "L"
    elif probe_counts[g_id] >= config.n_max:
      deactivate = True
      if curr_e < config.e_min:
        status = "L"
      else:
        status = _active_status(bin_val)

    if deactivate:
      final_e_values[g_id] = curr_e
      terminal_statuses[g_id] = status
      active_arms.pop(g_id, None)
      running_e_values.pop(g_id, None)
      prev_bins.pop(g_id, None)

  # Finalize any un-deactivated active arms that received at least 1 probe.
  # (Arms with e >= e_max are always deactivated above, so only L/A/B_j occur.)
  for h_id, curr_e in running_e_values.items():
    if probe_counts.get(h_id, 0) > 0:
      final_e_values[h_id] = curr_e
      terminal_statuses[h_id] = str(
          status_lib.classify_terminal_status(
              curr_e,
              config.e_min,
              config.e_max,
              config.num_active_bins,
              bin_edges,
          )
      )

  return _DiscoLoopResult(
      all_tested_arms=all_tested_arms,
      final_e_values=final_e_values,
      probe_counts=probe_counts,
      terminal_statuses=terminal_statuses,
  )


def run_disco_bandit(
    config: BanditConfig,
    env: BanditEnv,
    null_tables: null_tables_lib.NullCoarsenedTables | None = None,
    total_true_positives: int = 0,
    cluster_true_positives: dict[int, int] | None = None,
) -> BanditResult:
  """Executes the DISCO bandit algorithm.

  Probes with coarsened e-processes until the budget is spent, then calibrates
  the nominal e-BH level from the observed terminal statuses, boosts, rejects,
  and computes the post-hoc FDR certificate.

  Args:
    config: Run configuration.
    env: Bandit environment to probe.
    null_tables: Precomputed null coarsened tables; built from ``config`` if
      None.
    total_true_positives: Number of alternatives in the full pool (recall
      denominator); 0 disables recall.
    cluster_true_positives: Per-cluster alternative counts for macro recall and
      cluster coverage.

  Returns:
    The BanditResult for this run (``runner_type="disco"``).
  """
  k_bar = resolve_k_bar(config)
  unbinned_e_values = config.num_active_bins == 0
  effective_bins = 1 if unbinned_e_values else config.num_active_bins

  if null_tables is None:
    null_tables = build_null_tables(config, k_bar)
  bin_edges = status_lib.compute_strip_bins(
      config.e_min, config.e_max, effective_bins
  )

  loop = _run_disco_probing_loop(config, env, null_tables, bin_edges)
  all_tested_arms = loop.all_tested_arms
  final_e_values = loop.final_e_values
  probe_counts = loop.probe_counts
  terminal_statuses = loop.terminal_statuses

  k_total = len(final_e_values)

  # Alpha calibration inputs
  delta_post = cert_lib.compute_delta_post(
      (probe_counts.get(h, 1) for h in final_e_values),
      null_tables.P_H_table,
      null_tables.n_max,
  )

  e_sum = sum(
      null_tables_lib.get_expected_null_e(
          terminal_statuses.get(h, "A"),
          probe_counts.get(h, 1),
          null_tables,
          exact_e=(final_e_values.get(h, 1.0) if unbinned_e_values else None),
      )
      for h in terminal_statuses
      if terminal_statuses.get(h) != "H"
  )
  e_bar_obs = e_sum / max(1, k_total)

  statuses_list = [terminal_statuses.get(h, "A") for h in final_e_values]
  counts_list = [probe_counts.get(h, 1) for h in final_e_values]

  if unbinned_e_values:
    e_mean_max = config.e_max
  elif config.num_active_bins > 1:
    e_mean_max = null_tables.E_bin_max
  else:
    e_mean_max = null_tables.E_A_max

  nominal_alpha = _calibrate_nominal_alpha(
      config,
      null_tables,
      k_bar=k_bar,
      k_total=k_total,
      delta_post=delta_post,
      e_bar_obs=e_bar_obs,
      e_mean_max=e_mean_max,
      statuses_list=statuses_list,
      counts_list=counts_list,
      unbinned_e_values=unbinned_e_values,
      final_e_values=final_e_values,
      terminal_statuses=terminal_statuses,
      probe_counts=probe_counts,
  )

  # Apply boosting & e-BH on observed K_total
  boosted_e_values = {}
  for h_id, e_val in final_e_values.items():
    m = min(max(1, probe_counts.get(h_id, 1)), config.n_max)
    if config.use_boosting and nominal_alpha > 0:
      b = boosting.compute_boost_factor(
          nominal_alpha, config.kappa, m, k_total, use_exact_t=True
      )
    else:
      b = 1.0
    boosted_e_values[h_id] = float(b * e_val)

  rejected_set, rejection_thresh = multiple_testing.e_bh_rejection(
      boosted_e_values, nominal_alpha
  )

  metrics = _compute_run_metrics(
      config,
      rejected_set,
      all_tested_arms,
      final_e_values,
      probe_counts,
      total_true_positives,
      cluster_true_positives,
  )

  cert = cert_lib.compute_disco_posthoc_certificate(
      terminal_statuses,
      probe_counts,
      null_tables,
      nominal_alpha,
      final_e_values=(final_e_values if unbinned_e_values else None),
      use_boosting=config.use_boosting,
      kappa=config.kappa,
      n_max_override=config.n_max,
  )
  cert_epsilon = (
      cert.epsilon_post_boost if config.use_boosting else cert.epsilon_post
  )
  cert_posthoc = (
      cert.posthoc_boosted if config.use_boosting else cert.posthoc_unboosted
  )

  return _build_result(
      runner_type="disco",
      config=config,
      metrics=metrics,
      arms=all_tested_arms,
      k_reference=k_total,
      rejected_set=rejected_set,
      rejection_threshold=rejection_thresh,
      nominal_alpha_used=nominal_alpha,
      final_e_values=final_e_values,
      boosted_e_values=boosted_e_values,
      probe_counts=probe_counts,
      terminal_statuses=terminal_statuses,
      total_true_positives=total_true_positives,
      cluster_true_positives=cluster_true_positives,
      epsilon_post=cert_epsilon,
      delta_post=cert.delta_post,
      posthoc_certificate=cert_posthoc,
  )


# =============================================================================
# Offline Bandit Exploration Loop & Baseline Runners (e-BH and Type-I)
# =============================================================================


@dataclasses.dataclass
class _OfflineLoopResult:
  """Internal result from the shared unconstrained exploration loop."""

  discovered_arms: dict[str, ArmInfo]
  accumulated_e_values: dict[str, float]
  probe_counts: dict[str, int]
  failures_dict: dict[str, int]
  terminal_statuses: dict[str, str]


def _offline_exit_status(
    config: BanditConfig,
    curr_e: float,
    deactivate_on_threshold: float | None,
) -> str | None:
  """Returns "H"/"L" if ``curr_e`` triggers an immediate exit, else None.

  Checks, in order: the Type-I 1/alpha threshold and ``e_min`` (when
  ``enforce_exit_low``).

  Args:
    config: Run configuration (reads ``enforce_exit_low``, ``e_min``).
    curr_e: Current accumulated e-value of the arm.
    deactivate_on_threshold: Type-I baseline threshold (1/alpha), or None for
      Offline e-BH.
  """
  if deactivate_on_threshold is not None and curr_e >= deactivate_on_threshold:
    return "H"
  if config.enforce_exit_low and curr_e < config.e_min:
    return "L"
  return None


def _run_offline_bandit_loop(
    config: BanditConfig,
    env: BanditEnv,
    deactivate_on_threshold: float | None = None,
) -> _OfflineLoopResult:
  """Executes unconstrained MAB / Contextual UCB exploration with optional threshold deactivation."""
  remaining_budget = config.budget
  chunk_sz = config.chunk_size

  discovered_arms: dict[str, ArmInfo] = {}
  accumulated_e_values: dict[str, float] = {}
  probe_counts: dict[str, int] = {}
  failures_dict: dict[str, int] = {}
  terminal_statuses: dict[str, str] = {}
  deactivated_arms: set[str] = set()

  contextual_selector = OfflineContextualCandidateSelector(
      dimension=config.dimension,
      beta=config.gp_beta,
      ridge_lambda=config.ridge_lambda,
      seed=config.seed,
  )

  total_rounds = 0

  while remaining_budget > 0:
    q_count = min(chunk_sz, remaining_budget)
    total_rounds += 1

    # Candidate selection via UCB
    exclude_ids = set(discovered_arms.keys())
    cand_pool = env.generate_candidate_pool(
        _CONTEXTUAL_CANDIDATE_POOL_SIZE, exclude_ids=exclude_ids
    )

    available_discovered = []
    for arm in discovered_arms.values():
      if arm.arm_id in deactivated_arms:
        continue
      curr_e = accumulated_e_values.get(arm.arm_id, 1.0)
      if config.enforce_exit_low and curr_e < config.e_min:
        continue
      available_discovered.append(arm)

    if not available_discovered and not cand_pool:
      break

    if config.env_type == "contextual":
      selected_arm = contextual_selector.select_next_arm(
          available_discovered, cand_pool
      )
    else:
      # Standard Multi-Armed Bandit UCB
      p0 = sigmoid(config.baseline_rate_logit)
      cand_ucb = p0 + config.gp_beta * np.sqrt(
          (2.0 * np.log(max(2, total_rounds))) / 1.0
      )
      best_arm = (
          cand_pool[0]
          if cand_pool
          else (available_discovered[0] if available_discovered else None)
      )
      best_ucb = cand_ucb if cand_pool else -1e9

      for arm in available_discovered:
        n = probe_counts[arm.arm_id]
        mu = float(failures_dict[arm.arm_id]) / max(1, n * chunk_sz)
        ucb = mu + config.gp_beta * np.sqrt(
            (2.0 * np.log(max(2, total_rounds))) / max(1, n)
        )
        if ucb > best_ucb:
          best_ucb = ucb
          best_arm = arm

      if best_arm is None:
        break
      selected_arm = best_arm

    arm_id = selected_arm.arm_id
    if arm_id not in discovered_arms:
      discovered_arms[arm_id] = selected_arm
      accumulated_e_values[arm_id] = 1.0
      probe_counts[arm_id] = 0
      failures_dict[arm_id] = 0

    # Query Arm
    query_res = env.sample_queries(
        arm_id, q_count, randomized_p=_randomized_p(config)
    )
    remaining_budget -= q_count
    probe_counts[arm_id] += 1
    failures_dict[arm_id] += query_res.failures

    # Accumulate e-value
    local_e = calibrators.p_to_e(query_res.p_value, config.kappa)
    accumulated_e_values[arm_id] *= local_e
    curr_e_val = accumulated_e_values[arm_id]

    if selected_arm.feature is not None:
      contextual_selector.update_observations(
          arm_id,
          selected_arm.feature,
          log_e=float(np.log(max(1e-20, min(1e100, curr_e_val)))),
      )

    # Deactivation check (1/alpha threshold or e_min)
    exit_status = _offline_exit_status(
        config, curr_e_val, deactivate_on_threshold
    )
    if exit_status is not None:
      deactivated_arms.add(arm_id)
      terminal_statuses[arm_id] = exit_status

  # Finalize terminal statuses for any un-deactivated arms
  for h_id in discovered_arms:
    if h_id not in terminal_statuses:
      curr_e = accumulated_e_values.get(h_id, 1.0)
      terminal_statuses[h_id] = (
          _offline_exit_status(config, curr_e, deactivate_on_threshold) or "A"
      )

  return _OfflineLoopResult(
      discovered_arms=discovered_arms,
      accumulated_e_values=accumulated_e_values,
      probe_counts=probe_counts,
      failures_dict=failures_dict,
      terminal_statuses=terminal_statuses,
  )


def run_offline_ebh_bandit(
    config: BanditConfig,
    env: BanditEnv,
    total_true_positives: int = 0,
    cluster_true_positives: dict[int, int] | None = None,
) -> BanditResult:
  """Executes the Offline e-BH baseline with unconstrained UCB exploration."""
  loop_res = _run_offline_bandit_loop(config, env, deactivate_on_threshold=None)
  discovered_arms = loop_res.discovered_arms
  final_e_values = dict(loop_res.accumulated_e_values)
  probe_counts = loop_res.probe_counts
  k_bar = resolve_k_bar(config)

  # Rejection via e-BH at level alpha with fixed K = K_bar.
  # Boosting is not statistically valid for Offline e-BH without e-value
  # coarsening (DISCO), so unboosted e-values are used (b = 1.0).
  boosted_e_values = {
      h_id: float(e_val) for h_id, e_val in final_e_values.items()
  }
  rejected_set, rejection_thresh = multiple_testing.e_bh_rejection(
      boosted_e_values, config.alpha, k_total_override=k_bar
  )

  metrics = _compute_run_metrics(
      config,
      rejected_set,
      discovered_arms,
      final_e_values,
      probe_counts,
      total_true_positives,
      cluster_true_positives,
  )
  return _build_result(
      runner_type="offline_ebh",
      config=config,
      metrics=metrics,
      arms=discovered_arms,
      k_reference=k_bar,
      rejected_set=rejected_set,
      rejection_threshold=rejection_thresh,
      nominal_alpha_used=config.alpha,
      final_e_values=final_e_values,
      boosted_e_values=boosted_e_values,
      probe_counts=probe_counts,
      terminal_statuses=loop_res.terminal_statuses,
      total_true_positives=total_true_positives,
      cluster_true_positives=cluster_true_positives,
  )


def run_type1_bandit(
    config: BanditConfig,
    env: BanditEnv,
    total_true_positives: int = 0,
    cluster_true_positives: dict[int, int] | None = None,
) -> BanditResult:
  """Executes the Type-I Error Rate baseline (threshold 1/alpha, no e-BH).

  Deactivates arms as soon as they cross 1/alpha during unconstrained
  exploration, and rejects all arms with accumulated e-value >= 1/alpha without
  multiple testing correction.

  Args:
    config: Run configuration.
    env: Bandit environment to probe.
    total_true_positives: Number of alternatives in the full pool (recall
      denominator); 0 disables recall.
    cluster_true_positives: Per-cluster alternative counts for macro recall and
      cluster coverage.

  Returns:
    The BanditResult for this run (``runner_type="type1"``).
  """
  threshold = 1.0 / config.alpha
  loop_res = _run_offline_bandit_loop(
      config, env, deactivate_on_threshold=threshold
  )
  discovered_arms = loop_res.discovered_arms
  final_e_values = dict(loop_res.accumulated_e_values)
  probe_counts = loop_res.probe_counts

  # Rejection: directly threshold at 1/alpha without e-BH
  boosted_e_values = {
      h_id: float(e_val) for h_id, e_val in final_e_values.items()
  }
  rejected_set = {
      h_id for h_id, e_val in final_e_values.items() if e_val >= threshold
  }

  metrics = _compute_run_metrics(
      config,
      rejected_set,
      discovered_arms,
      final_e_values,
      probe_counts,
      total_true_positives,
      cluster_true_positives,
  )
  return _build_result(
      runner_type="type1",
      config=config,
      metrics=metrics,
      arms=discovered_arms,
      k_reference=len(final_e_values),
      rejected_set=rejected_set,
      rejection_threshold=threshold,
      nominal_alpha_used=config.alpha,
      final_e_values=final_e_values,
      boosted_e_values=boosted_e_values,
      probe_counts=probe_counts,
      terminal_statuses=loop_res.terminal_statuses,
      total_true_positives=total_true_positives,
      cluster_true_positives=cluster_true_positives,
  )
