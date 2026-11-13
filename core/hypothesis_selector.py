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

"""Chooses the hypothesis each DISCO discovery round probes.

Each round the agent either probes an active hypothesis or proposes a new one
(`get_disco_select_or_propose_prompt`). What it sees depends on the mode:
under DISCO only coarsened status, under the baselines (or
`show_all_failures`) also the cross-hypothesis failure summary. Showing the
failure summary is coarsening-compliant only with `num_active_bins=0` (unbinned
e-values, where the agent already sees the exact e-values); with binned
e-values it leaks information. A new proposal can be screened by the hypothesis
checker before it is accepted.
"""

from typing import Callable, cast

from disco import hypothesis_checker
from disco.agent import senders as agent_senders
from disco.core import checkpoint as checkpoint_lib
from disco.core import config as config_lib
from disco.core import history
from disco.core import mode_policy
from disco.core import prompts_protocol
from disco.core import records as records_lib
from disco.parsing import parsers as parsers_lib
from disco.parsing import schemas

LogFn = Callable[..., None]


class HypothesisSelector:
  """Asks the agent which hypothesis to probe and records new proposals."""

  def __init__(
      self,
      config: config_lib.DiscoConfig,
      state: checkpoint_lib.RunState,
      parsers: parsers_lib.Parsers,
      policy: mode_policy.ModePolicy,
  ):
    self.config = config
    self.state = state
    self.parsers = parsers
    self.policy = policy

  @property
  def shows_failures(self) -> bool:
    """Whether the agent sees the cross-hypothesis failure summary."""
    return self.config.show_all_failures or self.policy.is_baseline

  async def failure_summary(self, log_fn: LogFn) -> str:
    """The summarized cross-hypothesis failures, or "" if not shown or none."""
    if not self.shows_failures:
      return ""
    raw_summary = history.build_failure_summary(self.state.hypothesis_groups)
    if not raw_summary:
      return ""
    return await history.summarize_failure_block(
        raw_summary,
        self.config.seed_instruction,
        self.config.agent_model,
        log_fn,
    )

  async def select(
      self,
      sender: agent_senders.AgentSender,
      prompts_mod: prompts_protocol.FDRPromptsModule,
      log_fn: LogFn,
      failure_summary: str = "",
  ) -> tuple[str, str, bool]:
    """Asks the agent to probe an active hypothesis or propose a new one.

    Args:
      sender: The run's main agent sender (its history is cleared first).
      prompts_mod: Prompts module.
      log_fn: Logging function.
      failure_summary: This round's summary; computed here if empty.

    Returns:
      (h_id, h_text, is_new) where is_new indicates a newly proposed hypothesis.
    """
    config = self.config
    state = self.state
    disco_summary = history.disco_history_summary(
        state.hypothesis_groups,
        state.active_hypotheses,
        state.terminal_statuses,
        state.probe_counts,
        state.transition_scores,
        config.e_min,
        config.e_max,
        show_failures=self.shows_failures,
        baseline_mode=config.baseline_mode,
        baseline_type1_mode=config.baseline_type1_mode,
        running_e_values=state.running_e_values,
        final_e_values=state.final_e_values,
    )

    # Proposal order, not set order: set order depends on PYTHONHASHSEED.
    active_dict = {
        h_id: record.text
        for h_id, record in state.hypothesis_groups.items()
        if h_id in state.active_hypotheses
    }

    if not failure_summary:
      failure_summary = await self.failure_summary(log_fn)

    sender.clear_history()
    prompt = prompts_mod.get_disco_select_or_propose_prompt(
        disco_history_summary=disco_summary,
        active_hypotheses=active_dict,
        target_model=config.target_model,
        seed_instruction=config.seed_instruction,
        failure_summary=failure_summary,
        e_min=config.e_min,
        e_max=config.e_max,
        baseline_mode=self.policy.is_baseline,
    )
    cur_prompt = prompt
    for attempt in range(config.hypothesis_checker_max_retries + 1):
      result = cast(
          schemas.HypothesisResult,
          await sender.send_and_extract(
              cur_prompt,
              parsers_lib.parse_h,
              log_fn=log_fn,
              parsers=self.parsers,
          ),
      )
      h_id, h_text = result.h_id, result.h_text

      if h_id in state.active_hypotheses:
        # Probing an existing active hypothesis.
        log_fn(f"\n[{self.policy.selection_tag}] Probing existing: {h_id}")
        return h_id, state.hypothesis_groups[h_id].text, False

      if not config.enable_hypothesis_checker:
        break

      check_result = await hypothesis_checker.check_hypothesis(
          seed_instruction=config.seed_instruction,
          hypothesis_id=h_id,
          hypothesis_text=h_text,
          existing_hypotheses={
              hid: rec.text for hid, rec in state.hypothesis_groups.items()
          },
          checker_model=config.agent_model,
          rules=config.hypothesis_checker_rules,
      )
      if check_result.compliant:
        log_fn(f"\n[Hypothesis Checker] PASSED: {h_id}")
        break

      log_fn(
          "\n[Hypothesis Checker] REJECTED (attempt"
          f" {attempt + 1}/{config.hypothesis_checker_max_retries + 1}): {h_id}"
      )
      for v in check_result.violations:
        log_fn(f"  - {v}")
      log_fn(f"  Reasoning: {check_result.reasoning}")

      if attempt < config.hypothesis_checker_max_retries:
        log_fn(
            "\n[Hypothesis Checker] Re-prompting agent with rejection"
            " feedback..."
        )
        cur_prompt = hypothesis_checker.format_rejection_prompt(
            check_result=check_result,
            rejected_hypothesis_id=h_id,
            rejected_hypothesis_text=h_text,
        )
      else:
        log_fn(
            "\n[Hypothesis Checker] WARNING: Max retries"
            f" ({config.hypothesis_checker_max_retries}) exceeded for {h_id}."
            " Accepting hypothesis."
        )

    # New hypothesis proposed.
    if h_id in state.hypothesis_groups:
      h_id = f"{h_id}_{state.k_total}"

    state.hypothesis_groups[h_id] = records_lib.HypothesisRecord(text=h_text)
    state.active_hypotheses.add(h_id)
    state.running_e_values[h_id] = 1.0
    state.probe_counts[h_id] = 0
    state.bin_trajectories[h_id] = []
    state.prev_bins[h_id] = self.policy.init_bin
    state.transition_scores[h_id] = []
    state.k_total += 1

    log_fn(
        f"\n[{self.policy.spawn_tag}] New hypothesis: {h_id}"
        f" (K={state.k_total})"
    )
    return h_id, h_text, True
