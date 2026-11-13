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

"""Per-mode behavior of the DISCO discovery loop: DISCO, baseline and Type-I.

`runners.DiscoRunner.run_loop` runs one loop for three modes:

  * DISCO (the default): DISCO with coarsened e-processes. Null tables are
    precomputed. After each e-value update the hypothesis moves through the
    strip bins (or gets an unbinned score) and deactivates once it leaves
    [e_min, e_max) or reaches `n_max` probes.
  * Baseline (`baseline_mode`): unrestricted exploration, with offline e-BH at
    finalization. A hypothesis only deactivates by exiting low (with
    `baseline_enforce_exit_low`) or, with `baseline_exit_high`, by freezing
    once its running e-value reaches K_bar / alpha -- the largest threshold
    the fixed-K_bar e-BH can require, so its rejection is already certain.
  * Type-I baseline (`baseline_type1_mode`): a hypothesis freezes as soon as
    its running e-value crosses 1/alpha (or exits low, as above). No e-BH.

The modes differ in how the run is described in the log, how selections are
tagged, what the agent may see, and what happens to a hypothesis after its
e-value is updated. Those differences live here; the loop is shared.
"""

import abc
from typing import Callable

import numpy as np

from disco import stats_utils
from disco.core import checkpoint as checkpoint_lib
from disco.core import config as config_lib
from disco.stats.coarsened import null_tables as null_tables_lib
from disco.stats.coarsened import status as status_lib

LogFn = Callable[..., None]


class ModePolicy(abc.ABC):
  """What differs between the DISCO, baseline and Type-I discovery loops.

  Attributes:
    config: Run config.
    unbinned_e_values: True when `num_active_bins == 0` (exact log e-value
      scores).
    effective_bins: Number of strip bins (1 for unbinned e-values).
    null_tables: Precomputed coarsened null tables; None outside DISCO mode.
    bin_edges: Strip bin edges; empty outside DISCO mode.
    init_bin: Bin a newly proposed hypothesis starts in.
  """

  # Log tags for probing an existing hypothesis and for a new proposal.
  selection_tag: str = ""
  spawn_tag: str = ""
  # Baselines show the agent everything (failures, running e-values); DISCO
  # restricts it to coarsened status.
  is_baseline: bool = True

  def __init__(self, config: config_lib.DiscoConfig):
    self.config = config
    self.unbinned_e_values = config.num_active_bins == 0
    self.effective_bins = (
        1 if self.unbinned_e_values else config.num_active_bins
    )
    self.null_tables: null_tables_lib.NullCoarsenedTables | None = None
    self.bin_edges: np.ndarray = np.array([])
    self.init_bin: int = 1

  @abc.abstractmethod
  def log_config(self, log_fn: LogFn) -> None:
    """Logs this mode's parameters at the start of the run."""

  @abc.abstractmethod
  def banner(self) -> str:
    """The line logged when the loop starts."""

  @abc.abstractmethod
  def after_e_value(
      self,
      state: checkpoint_lib.RunState,
      g_id: str,
      curr_e: float,
      p_value: float,
      e_val: float,
      log_fn: LogFn,
  ) -> None:
    """Updates `g_id`'s status now that its running e-value is `curr_e`.

    Args:
      state: Run state; the hypothesis's status fields are updated in place.
      g_id: Hypothesis just probed.
      curr_e: Its running e-value after this round.
      p_value: This round's p-value.
      e_val: This round's e-value.
      log_fn: Logging function.
    """

  def _log_e_value(
      self, p_value: float, e_val: float, curr_e: float, log_fn: LogFn
  ) -> None:
    log_fn(
        f"\n  p-value: {p_value:.6f} | e-val: {e_val:.4f} |"
        f" Running E: {curr_e:.4f}"
    )

  def _exit_low(
      self,
      state: checkpoint_lib.RunState,
      g_id: str,
      curr_e: float,
      log_fn: LogFn,
  ) -> None:
    """Deactivates `g_id` as L (baseline modes, `baseline_enforce_exit_low`)."""
    state.final_e_values[g_id] = curr_e
    state.terminal_statuses[g_id] = "L"
    state.active_hypotheses.discard(g_id)
    state.running_e_values.pop(g_id, None)
    log_fn(
        f"  [Baseline Exit-Low] {g_id} deactivated"
        f" (E={curr_e:.4f}, n={state.probe_counts[g_id]})"
    )

  def _log_baseline_config(self, mode: str, log_fn: LogFn) -> None:
    config = self.config
    log_fn(mode)
    log_fn(f"Alpha: {config.alpha}, kappa: {config.kappa}")
    log_fn(f"P-value mode: {config.p_value_mode}")
    log_fn("Failure visibility: ALL (baseline)")
    log_fn(f"Baseline enforce_exit_low: {config.baseline_enforce_exit_low}")


class Type1BaselinePolicy(ModePolicy):
  """`baseline_type1_mode`: freeze at 1/alpha, no e-BH."""

  selection_tag = "Baseline Type-I Selection"
  spawn_tag = "Baseline Type-I Spawn"

  def log_config(self, log_fn: LogFn) -> None:
    """Logs Type-I baseline parameters at the start of the run."""
    threshold = 1.0 / self.config.alpha
    self._log_baseline_config(
        "Mode: BASELINE (Type-I Error Rate, threshold 1/alpha="
        f"{threshold:.2f}, no e-BH)",
        log_fn,
    )

  def banner(self) -> str:
    """Returns the banner string for Type-I baseline mode."""
    threshold = 1.0 / self.config.alpha
    return (
        "\n--- DISCO Baseline: Type-I Error Rate Mode"
        f" (threshold={threshold:.2f}, no e-BH) ---"
    )

  def after_e_value(
      self,
      state: checkpoint_lib.RunState,
      g_id: str,
      curr_e: float,
      p_value: float,
      e_val: float,
      log_fn: LogFn,
  ) -> None:
    """Freezes `g_id` once its running e-value crosses 1/alpha."""
    config = self.config
    self._log_e_value(p_value, e_val, curr_e, log_fn)
    type1_threshold = 1.0 / config.alpha
    if curr_e >= type1_threshold:
      state.final_e_values[g_id] = curr_e
      state.terminal_statuses[g_id] = "H"
      state.active_hypotheses.discard(g_id)
      state.running_e_values.pop(g_id, None)
      log_fn(
          f"  [Type-I Freeze] {g_id} crossed 1/alpha={type1_threshold:.2f}"
          f" (E={curr_e:.4f}, n={state.probe_counts[g_id]})"
      )
    elif config.baseline_enforce_exit_low and curr_e < config.e_min:
      self._exit_low(state, g_id, curr_e, log_fn)


class BaselinePolicy(ModePolicy):
  """`baseline_mode`: unrestricted exploration, offline e-BH at the end."""

  selection_tag = "Baseline Selection"
  spawn_tag = "Baseline Spawn"

  def k_bar(self) -> int:
    """The fixed family size K_bar of the offline e-BH."""
    config = self.config
    return max(1, config.budget // config.chunk_size)

  def exit_high_threshold(self) -> float | None:
    """Running e-value at which a hypothesis freezes, or None if disabled.

    `K_bar / alpha` is the threshold e-BH applies when it rejects exactly one
    hypothesis, i.e. the largest it can ever require: a hypothesis at or above
    it is rejected whatever the others do, so probing it further cannot change
    the outcome.

    Returns:
      The `K_bar / alpha` freeze threshold if `baseline_exit_high` is enabled,
      else None.
    """
    if not self.config.baseline_exit_high:
      return None
    return self.k_bar() / self.config.alpha

  def log_config(self, log_fn: LogFn) -> None:
    """Logs unrestricted baseline parameters at the start of the run."""
    self._log_baseline_config(
        "Mode: BASELINE (unrestricted exploration, offline e-BH with"
        f" K_bar={self.k_bar()})",
        log_fn,
    )
    threshold = self.exit_high_threshold()
    if threshold is None:
      log_fn("Baseline exit_high: False")
    else:
      log_fn(
          f"Baseline exit_high: True (freeze at K_bar/alpha={threshold:.2f})"
      )

  def banner(self) -> str:
    """Returns the banner string for unrestricted baseline mode."""
    return "\n--- DISCO Baseline: Unrestricted Exploration Mode ---"

  def after_e_value(
      self,
      state: checkpoint_lib.RunState,
      g_id: str,
      curr_e: float,
      p_value: float,
      e_val: float,
      log_fn: LogFn,
  ) -> None:
    """Updates `g_id` status under unrestricted exploration."""
    config = self.config
    self._log_e_value(p_value, e_val, curr_e, log_fn)
    threshold = self.exit_high_threshold()
    if threshold is not None and curr_e >= threshold:
      state.final_e_values[g_id] = curr_e
      state.terminal_statuses[g_id] = "H"
      state.active_hypotheses.discard(g_id)
      state.running_e_values.pop(g_id, None)
      log_fn(
          f"  [Baseline Exit-High] {g_id} crossed K_bar/alpha={threshold:.2f}"
          f" (E={curr_e:.4f}, n={state.probe_counts[g_id]})"
      )
    # Optionally enforce exit-low even in baseline mode.
    elif config.baseline_enforce_exit_low and curr_e < config.e_min:
      self._exit_low(state, g_id, curr_e, log_fn)


class DiscoPolicy(ModePolicy):
  """DISCO with coarsened e-processes: strip bins, null tables, deactivation."""

  selection_tag = "DISCO Selection"
  spawn_tag = "DISCO Spawn"
  is_baseline = False

  def __init__(self, config: config_lib.DiscoConfig):
    super().__init__(config)
    if config.show_all_failures and config.num_active_bins > 0:
      raise ValueError(
          "show_all_failures=True is not permitted when num_active_bins > 0"
          " in DISCO mode; set num_active_bins=0 or"
          " show_all_failures=False."
      )
    k_bar = max(1, config.budget // config.chunk_size)
    use_discrete_tables = config.p_value_mode == "discrete_matched"
    if (
        self.config.task_null_proportion is not None
        and self.config.task_null_proportion > 0.0
    ):
      null_prop = float(self.config.task_null_proportion)
    else:
      null_prop = stats_utils.extract_null_proportion_from_seed(
          config.seed_instruction, default=0.5
      )
    # Precompute null tables.
    self.null_tables = null_tables_lib.compute_null_coarsened_tables(
        kappa=config.kappa,
        e_min=config.e_min,
        e_max=config.e_max,
        n_max=config.n_max,
        k_max=k_bar,
        num_bins=self.effective_bins,
        alpha=config.alpha,
        n_sim=config.n_sim,
        seed=42,
        discrete_null=use_discrete_tables,
        discrete_chunk_size=config.chunk_size,
        discrete_baseline_rate=null_prop,
        enforce_exit_low=config.enforce_exit_low,
    )
    self.bin_edges = status_lib.compute_strip_bins(
        config.e_min, config.e_max, self.effective_bins
    )
    self.init_bin = int(
        np.clip(
            np.searchsorted(self.bin_edges, 1.0, side="right"),
            1,
            self.effective_bins,
        )
    )

  def log_config(self, log_fn: LogFn) -> None:
    """Logs DISCO parameters at the start of the run."""
    config = self.config
    log_fn(f"DISCO Parameters: alpha={config.alpha}, kappa={config.kappa}")
    log_fn(
        f"Strip: e_min={config.e_min}, e_max={config.e_max},"
        f" n_max={config.n_max}"
    )
    log_fn(
        f"Bins: num_active_bins={config.num_active_bins},"
        f" effective_bins={self.effective_bins},"
        f" unbinned_e_values={self.unbinned_e_values}"
    )
    log_fn(
        f"Alpha calibration: {config.alpha_calibration_mode},"
        f" boosting={config.use_boosting}"
    )
    log_fn(f"Null tables precomputed (n_sim={config.n_sim})")
    log_fn(f"P-value mode: {config.p_value_mode}")
    if config.show_all_failures:
      # Showing failures everywhere is coarsening-compliant only with unbinned
      # e-values (num_active_bins=0), where the agent already sees exact
      # e-values.
      coarsening_note = (
          "coarsening-compliant: unbinned e-values"
          if self.unbinned_e_values
          else "coarsening broken: binned e-values"
      )
      log_fn(
          "Failure visibility: ALL (selector, cross-hyp, intra-hyp;"
          f" {coarsening_note})"
      )
    elif config.show_intra_hypothesis_failures:
      log_fn("Failure visibility: INTRA-HYPOTHESIS ONLY (coarsening-compliant)")
    else:
      log_fn("Failure visibility: NONE (coarsened e-values)")

  def banner(self) -> str:
    """Returns the banner string for DISCO mode."""
    return (
        "\n--- DISCO: Null tables precomputed"
        f" (n_max={self.config.n_max}, bins={self.effective_bins}) ---"
    )

  def after_e_value(
      self,
      state: checkpoint_lib.RunState,
      g_id: str,
      curr_e: float,
      p_value: float,
      e_val: float,
      log_fn: LogFn,
  ) -> None:
    """Updates strip bin, transition score, and deactivation status for `g_id`."""
    config = self.config
    log_fn(
        f"\n  p-value: {p_value:.6f} | e-val: {e_val:.4f} |"
        f" Running E (private): {curr_e:.4f}"
    )

    # Bin mapping and transition score.
    assert self.null_tables is not None
    null_tables = self.null_tables
    if self.unbinned_e_values:
      # Unbinned e-values: use exact log e-value as transition score,
      # clamped to expected overshoot on exit.
      if config.enforce_exit_low and curr_e < config.e_min:
        t_score = null_tables.log_E_L
      elif curr_e >= config.e_max:
        t_score = null_tables.log_E_H
      else:
        t_score = float(np.log(curr_e))
      state.transition_scores[g_id].append(t_score)
      bin_val = None
      log_fn(f"  Unbinned e-value score: {t_score:+.3f}")
    else:
      if curr_e < config.e_min:
        bin_val = 0  # L
      elif curr_e >= config.e_max:
        bin_val = self.effective_bins + 1  # H
      else:
        b_idx = int(np.searchsorted(self.bin_edges, curr_e, side="right"))
        bin_val = max(1, min(b_idx, self.effective_bins))

      state.bin_trajectories[g_id].append(bin_val)
      g_prev = state.prev_bins.get(g_id, self.init_bin)
      t_score = null_tables_lib.lookup_transition_score(
          null_tables, g_prev, bin_val
      )
      state.transition_scores[g_id].append(t_score)
      state.prev_bins[g_id] = bin_val
      log_fn(f"  Bin: {g_prev} -> {bin_val} (transition score: {t_score:+.3f})")

    # Deactivation check.
    deactivate = False
    if self.unbinned_e_values or bin_val is None:
      status = "A"
    else:
      status = f"B_{bin_val}" if self.effective_bins > 1 else "A"
    if curr_e >= config.e_max:
      deactivate = True
      status = "H"
    elif config.enforce_exit_low and curr_e < config.e_min:
      deactivate = True
      status = "L"
    elif state.probe_counts[g_id] >= config.n_max:
      deactivate = True
      if curr_e < config.e_min:
        status = "L"
      elif curr_e >= config.e_max:
        status = "H"
      else:
        status = "A" if self.effective_bins <= 1 else f"B_{bin_val}"

    if deactivate:
      state.final_e_values[g_id] = curr_e
      state.terminal_statuses[g_id] = status
      state.active_hypotheses.discard(g_id)
      state.running_e_values.pop(g_id, None)
      state.prev_bins.pop(g_id, None)
      log_fn(
          f"  [Deactivation] {g_id} -> {status}"
          f" (E={curr_e:.4f}, n={state.probe_counts[g_id]})"
      )
    else:
      state.terminal_statuses[g_id] = status


def make_policy(config: config_lib.DiscoConfig) -> ModePolicy:
  """The policy for `config`'s mode (Type-I takes precedence over baseline)."""
  if config.baseline_type1_mode:
    return Type1BaselinePolicy(config)
  if config.baseline_mode:
    return BaselinePolicy(config)
  return DiscoPolicy(config)
