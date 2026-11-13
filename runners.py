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

"""Runner classes and execution loops for DISCO."""

import dataclasses
import datetime
import json
import os
from typing import Any, Callable, Generic, TypeVar, cast, overload

import numpy as np

from disco import agent_factory
from disco import stats_utils
from disco.agent import scrape_tolerance
from disco.agent import senders as agent_senders
from disco.core import checkpoint as checkpoint_lib
from disco.core import config as config_lib
from disco.core import fdr_outputs
from disco.core import finalizers
from disco.core import history
from disco.core import hypothesis_selector
from disco.core import mode_policy
from disco.core import prompts_protocol
from disco.core import records as records_lib
from disco.core import round_executor
from disco.core import task_executor
from disco.parsing import parsers as parsers_lib
from disco.stats import calibrators
from disco.stats import plan as stats_plan
from disco.stats.coarsened import null_tables as null_tables_lib

# Sentinel p-value to indicate statistical parsing failure.
STATS_PARSING_FAILURE_P_VALUE = stats_utils.STATS_PARSING_FAILURE_P_VALUE

_extract_null_proportion_from_seed = (
    stats_utils.extract_null_proportion_from_seed
)
_guard_p_value_direction = stats_plan.guard_p_value_direction
_maybe_randomize_p_value = stats_utils.maybe_randomize_p_value
_build_failure_summary = history.build_failure_summary
_summarize_failure_block = history.summarize_failure_block
_get_disco_history_summary = history.disco_history_summary


def _guard_null(
    res: stats_plan.StatsResult,
    test_plan: stats_plan.PreRegisteredTestPlan | None,
    proportion_null: float,
) -> float:
  """Returns the null the direction guard falls back to for a 1-sample test.

  The guard only uses it when `res` reports no null of its own. For a
  continuous test the null is on the metric's scale (e.g. 20 words), so the
  pre-registered null is used; otherwise the task's proportion null (the
  seed's rate, default 0.5 for text tasks; 0.9 for visual symmetry).

  Args:
    res: The parsed stats result.
    test_plan: The hypothesis's pre-registered plan, if any.
    proportion_null: The task's proportion null (`TaskExecutor.null_rate()`).
  """
  if (
      test_plan is not None
      and test_plan.null_baseline is not None
      and stats_plan.is_continuous_test(res.test_type)
  ):
    return test_plan.null_baseline
  return proportion_null


_T = TypeVar("_T")


class _StateAttr(Generic[_T]):
  """Forwards a runner attribute to a field of the runner's `RunState`."""

  def __init__(self, field: str):
    self._field = field

  @overload
  def __get__(self, obj: None, objtype: Any = None) -> "_StateAttr[_T]":
    ...

  @overload
  def __get__(self, obj: "_DiscoRunnerBase", objtype: Any = None) -> _T:
    ...

  def __get__(self, obj, objtype=None):
    if obj is None:
      return self
    return getattr(obj.state, self._field)

  def __set__(self, obj: "_DiscoRunnerBase", value: _T) -> None:
    setattr(obj.state, self._field, value)


class _DiscoRunnerBase:
  """Base class for Disco runners: run state, checkpointing and setup."""

  _runner_title: str = "Disco"

  # Loop state lives in `self.state` (persisted by the checkpoint); these
  # attributes forward to it so existing call sites keep working.
  hypothesis_groups = _StateAttr[dict[str, records_lib.HypothesisRecord]](
      "hypothesis_groups"
  )
  current_round = _StateAttr[int]("current_round")
  remaining_budget = _StateAttr[int]("remaining_budget")
  _total_scrape_rounds = _StateAttr[int]("total_scrape_rounds")
  running_e_values = _StateAttr[dict[str, float]]("running_e_values")
  final_e_values = _StateAttr[dict[str, float]]("final_e_values")
  probe_counts = _StateAttr[dict[str, int]]("probe_counts")
  terminal_statuses = _StateAttr[dict[str, str]]("terminal_statuses")
  bin_trajectories = _StateAttr[dict[str, list[int]]]("bin_trajectories")
  prev_bins = _StateAttr[dict[str, int]]("prev_bins")
  transition_scores = _StateAttr[dict[str, list[float]]]("transition_scores")
  active_hypotheses = _StateAttr[set[str]]("active_hypotheses")
  k_total = _StateAttr[int]("k_total")

  def __init__(
      self, config: config_lib.DiscoConfig, parsers: parsers_lib.Parsers
  ):
    self.config = config
    self.parsers = parsers
    # total_scrape_rounds counts _save_responses calls across all fresh agents.
    # It is passed as initial_scrape_round to each new StandaloneAgent so
    # folder numbering never resets between rounds.
    self.state = checkpoint_lib.RunState(remaining_budget=self.config.budget)
    self.threshold: float = calibrators.get_threshold_from_alpha(config.alpha)
    self.is_running: bool = False
    # RNG for p-value randomization (Tocher PIT) when p_value_mode="randomized".
    # Seeded runs save its state in the checkpoint (see `_save_checkpoint`).
    self._pval_rng = np.random.default_rng(config.random_seed)

  def _log_extra_config(self, log_fn: Callable[..., None]) -> None:
    """Hook for subclasses to log runner-specific configuration."""
    del log_fn  # Unused in base class.

  def _get_intro_prompt(
      self, prompts_mod: prompts_protocol.PromptsModule
  ) -> str:
    """Returns the introductory prompt for the runner's agent."""
    raise NotImplementedError("Subclasses must implement _get_intro_prompt.")

  # ---------------------------------------------------------------------------
  # Checkpoint / restart support
  # ---------------------------------------------------------------------------

  def _get_checkpoint_path(self) -> str:
    """Returns the path for the checkpoint file."""
    output_dir = os.path.join(
        self.config.output_dir, self.config.experiment_name
    )
    return os.path.join(output_dir, checkpoint_lib.CHECKPOINT_FILENAME)

  @property
  def _checkpoints(self) -> checkpoint_lib.CheckpointStore:
    return checkpoint_lib.CheckpointStore(self._get_checkpoint_path())

  def _save_checkpoint(self, log_fn: Callable[..., None] | None = None) -> None:
    """Saves the run state to checkpoint.json after each round."""
    if self.config.random_seed is not None:
      self.state.pval_rng_state = dict(self._pval_rng.bit_generator.state)
    self._checkpoints.save(
        self.state, runner_title=self._runner_title, budget=self.config.budget
    )
    if log_fn:
      log_fn(
          f"[CHECKPOINT] Saved round {self.current_round}"
          f" ({self.remaining_budget} queries remaining,"
          f" {len(self.hypothesis_groups)} hypotheses)"
      )

  def _load_checkpoint(self, log_fn: Callable[..., None] | None = None) -> bool:
    """Restores the run state from checkpoint.json if one is compatible.

    Args:
      log_fn: Optional logging function.

    Returns:
      True if a valid checkpoint was loaded, False otherwise.
    """
    if not self._checkpoints.load_into(
        self.state, budget=self.config.budget, log_fn=log_fn
    ):
      return False
    # A checkpoint from an unseeded run has no state; the RNG then keeps its
    # fresh seed.
    if (
        self.config.random_seed is not None
        and self.state.pval_rng_state is not None
    ):
      self._pval_rng.bit_generator.state = self.state.pval_rng_state
    if log_fn:
      log_fn(
          f"[CHECKPOINT] Restored from round {self.current_round}"
          f" ({self.remaining_budget} queries remaining,"
          f" {len(self.hypothesis_groups)} hypotheses)"
      )
    return True

  async def _setup_run(
      self,
  ) -> tuple[Callable[..., None], agent_senders.DiscoSession, str]:
    """Common setup: directories, logging, agent session, and intro."""
    config = self.config

    # === LOGGING SETUP ===
    output_dir = os.path.join(config.output_dir, config.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "disco.log")

    flags_path = os.path.join(output_dir, "flags.json")
    with open(flags_path, "w", encoding="utf-8") as f:
      json.dump(dataclasses.asdict(config), f, indent=2)

    log = fdr_outputs.make_log_function(log_file, config.log_to_console)

    # === INITIALIZATION AND AGENT SETUP ===
    # Append, never truncate: a run can be restarted after a crash or resumed
    # from a checkpoint, and the earlier attempt's log is the primary debugging
    # artifact. Each attempt is delimited by the banner below.
    log(
        f"\n\n=== {self._runner_title} Log Start"
        f" ({datetime.datetime.now().isoformat(timespec='seconds')}) ==="
    )
    log(f"Starting Disco with Target Model: {config.target_model}")
    log(f"Seed Instruction: {config.seed_instruction}")
    log(f"Budget: {config.budget} iterations")
    if config.procedural_autorater:
      log(f"Procedural Autorater: {config.procedural_autorater}")
    self._log_extra_config(log)

    session = agent_factory.build_agent_and_sender(config, log)
    sender = session.sender
    prompts_mod = session.prompts_mod

    log("\n--- Initializing Agent ---")
    intro_prompt = self._get_intro_prompt(prompts_mod)
    await sender.send_step(intro_prompt)

    return log, session, log_file

  def _round_executor(
      self, log_fn: Callable[..., None]
  ) -> round_executor.RoundExecutor:
    """Returns a round executor over this runner's state."""
    return round_executor.RoundExecutor(
        self.config, self.parsers, self.state, log_fn
    )


class DiscoRunner(_DiscoRunnerBase):
  """DISCO runner with proof-grounded FDR control via coarsened e-processes.

  Enforces coarsened e-value feedback: selector decisions (which hypothesis
  to probe, whether to propose new) only see coarsened bin status D_g,
  probe count n_g, transition scores from null tables, hypothesis text,
  and query metadata — never exact e-values, p-values, or failure examples.

  Hypotheses are proposed and probed organically by the agent. Deactivation
  happens when e-values exit the strip [e_min, e_max) or reach the probe cap
  n_max.

  Finalization: alpha calibration (3 modes), boosting, e-BH rejection,
  and post-hoc DISCO certificates.
  """

  _runner_title: str = "DISCO"

  def __init__(
      self, config: config_lib.DiscoConfig, parsers: parsers_lib.Parsers
  ):
    super().__init__(config, parsers)
    # Everything that differs between DISCO, baseline and Type-I runs: null
    # tables and bins, log text, and the post-e-value status update.
    self.policy = mode_policy.make_policy(config)

    # Per-hypothesis state lives in self.state: running/final e-values are
    # private (never shown to the agent in DISCO mode); probe counts, terminal
    # statuses, bins and transition scores are coarsening-safe.

  # Read by finalization and by tests; owned by the mode policy.
  @property
  def unbinned_e_values(self) -> bool:
    return self.policy.unbinned_e_values

  @property
  def effective_bins(self) -> int:
    return self.policy.effective_bins

  @property
  def null_tables(self) -> null_tables_lib.NullCoarsenedTables | None:
    return self.policy.null_tables

  @property
  def bin_edges(self) -> np.ndarray:
    return self.policy.bin_edges

  @property
  def init_bin(self) -> int:
    return self.policy.init_bin

  def _log_extra_config(self, log_fn: Callable[..., None]) -> None:
    self.policy.log_config(log_fn)

  def _get_intro_prompt(
      self, prompts_mod: prompts_protocol.PromptsModule
  ) -> str:
    fdr_prompts = cast(prompts_protocol.FDRPromptsModule, prompts_mod)
    is_baseline = self.policy.is_baseline
    prompt = fdr_prompts.get_disco_fresh_agent_intro_prompt(
        self.config.target_model,
        self.config.seed_instruction,
        chunk_size=self.config.chunk_size,
        e_min=self.config.e_min,
        e_max=self.config.e_max,
        baseline_mode=is_baseline,
        show_all_failures=self.config.show_all_failures,
    )
    if self.config.procedural_autorater:
      prompt += prompts_mod.get_procedural_autorater_addendum(
          self.config.procedural_autorater,
          allow_continuous_means=self.config.allow_continuous_means_test,
      )
    return prompt

  async def run_loop(self) -> records_lib.DiscoResult:
    """Runs the FDR-controlled DISCO loop."""
    config = self.config
    log, session, _ = await self._setup_run()
    sender = session.sender
    prompts_mod = cast(prompts_protocol.FDRPromptsModule, session.prompts_mod)

    log(self.policy.banner())

    # === CHECKPOINT RESTORE ===
    if self._load_checkpoint(log_fn=log):
      log(
          f"[CHECKPOINT] Resuming DISCO from round {self.current_round}"
          f" with {self.remaining_budget} queries remaining"
      )

    rounds = self._round_executor(log)
    # The direction guard's fallback null for proportion tests.
    proportion_null = task_executor.resolve_task_executor(
        config, log
    ).null_rate()
    selector = hypothesis_selector.HypothesisSelector(
        config, self.state, self.parsers, self.policy
    )
    round_guard = scrape_tolerance.RoundGuard(config.max_consecutive_skips, log)
    self.is_running = True
    try:
      while self.remaining_budget > 0:
        log(f"\n=== Round {self.current_round} ===")
        log(f"  Budget remaining: {self.remaining_budget}")
        log(f"  Active hypotheses: {len(self.active_hypotheses)}")
        log(f"  Total hypotheses (K): {self.k_total}")

        failure_summary = await selector.failure_summary(log)

        # Selection: probe existing or propose new.
        g_id, g_text, is_new = await selector.select(
            sender, prompts_mod, log, failure_summary=failure_summary
        )
        log(f"  Probing hypothesis: {g_id} ({'NEW' if is_new else 'existing'})")

        # Steps 2b-5: Probe with fresh agent (DISCO or baseline).
        result = await rounds.run(
            g_id,
            g_text,
            self.hypothesis_groups[g_id],
            cross_hyp_summary=failure_summary,
        )
        res = result.stats

        # E-value update.
        if result.status != "ok" or res is None:
          # A scrape failure charged nothing for the failed step; a stats
          # failure was charged but records no e-value.
          kind = result.skip_kind or "auditor_side"
          self.state.skip_counts[kind] = self.state.skip_counts.get(kind, 0) + 1
          log(
              "\n[Skip] No valid stats result or stats parsing failure."
              f" Skipping round ({kind}; {self.state.skip_counts[kind]} so"
              " far)."
          )
          self.current_round += 1
          self._save_checkpoint(log_fn=log)
          # After the checkpoint, so a TargetOutageError resumes from here.
          await round_guard.after_round(kind)
          continue
        await round_guard.after_round(None)

        p_value = _guard_p_value_direction(
            res,
            log,
            null_value=_guard_null(
                res,
                self.hypothesis_groups[g_id].test_plan,
                proportion_null,
            ),
            use_direction_guard=config.use_direction_guard,
        )
        if config.p_value_mode == "randomized":
          p_value = _maybe_randomize_p_value(p_value, res, self._pval_rng)

        e_val = calibrators.p_to_e(p_value, config.kappa)
        self.running_e_values[g_id] *= e_val
        cumulative_q = self.config.budget - self.remaining_budget
        entry = records_lib.EValueEntry(
            sub_hypothesis=g_text,
            e_value=e_val,
            cumulative_queries=cumulative_q,
        )
        self.hypothesis_groups[g_id].e_values.append(entry)
        self.probe_counts[g_id] = self.probe_counts.get(g_id, 0) + 1

        curr_e = self.running_e_values[g_id]
        self.policy.after_e_value(self.state, g_id, curr_e, p_value, e_val, log)

        self.current_round += 1
        self._save_checkpoint(log_fn=log)

    finally:
      self.is_running = False

    # === FINALIZATION ===
    finalizer = finalizers.make_finalizer(
        config,
        null_tables=self.null_tables,
        effective_bins=self.effective_bins,
        bin_edges=self.bin_edges,
        unbinned_e_values=self.unbinned_e_values,
    )
    result = finalizer.finalize(self.state, log)
    result.skip_counts = dict(self.state.skip_counts)
    n_skipped = sum(result.skip_counts.values())
    log(
        ("\n[Data Quality] WARNING: " if n_skipped else "\n[Data Quality] ")
        + f"{n_skipped} discovery round(s) skipped without an e-value ("
        + ", ".join(f"{k}={v}" for k, v in result.skip_counts.items())
        + ")"
    )
    return result
