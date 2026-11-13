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

"""Task-specific Step 4 execution for a DISCO round.

A `TaskExecutor` owns what differs between task families inside a round: the
intro prompt, whether the round pre-registers a test and generates text
queries, how Step 4 drives the target, and how executed queries are counted.

`resolve_task_executor` picks the executor from `config.task_suite` (the
`task_suite` flag). When it is empty, a seed instruction containing "symmetry"
still selects the symmetry executor (the shared rule in `task_detection`).
"""

from collections.abc import Callable
from typing import Any, ClassVar, Protocol, cast

from disco import stats_utils
from disco.agent import senders as agent_senders
from disco.core import checkpoint as checkpoint_lib
from disco.core import config as config_lib
from disco.core import prompts_protocol
from disco.core import task_detection
from disco.prompting import round_prompts
from disco.tasks.visual_symmetry import hypotheses as visual_symmetry_hypotheses
from disco.tasks.visual_symmetry import prompts as visual_symmetry_prompts


class RoundAgent(Protocol):
  """The parts of `StandaloneAgent` that query counting reads."""

  last_scrape_status: str
  all_responses: list[Any]

  @property
  def last_responses(self) -> list[Any]:
    ...

  @property
  def symmetry_output_images(self) -> list[Any]:
    ...


class TaskExecutor(Protocol):
  """Task-specific parts of a round.

  Attributes:
    name: Registry name of the task family.
    exclude_analyze: Whether the task forces the agent's analyze tool off.
    uses_preregistration: Whether Step 2b pre-registers a test plan.
    uses_text_querygen: Whether Step 3 generates text queries; otherwise the
      round runs `min(chunk_size, remaining_budget)` probes.
    last_arms: (label, scored completions, probes requested) per arm of the last
      Step 4 if it ran two independent arms (a contrastive symmetry round), else
      None. The round's `--min_response_fraction` check then applies per arm.
  """

  name: ClassVar[str]
  last_arms: list[tuple[str, int, int]] | None
  exclude_analyze: ClassVar[bool]
  uses_preregistration: ClassVar[bool]
  uses_text_querygen: ClassVar[bool]

  def intro_prompt(
      self, prompts_mod: prompts_protocol.PromptsModule, *, is_baseline: bool
  ) -> str:
    """Returns the first message sent to the round's agent."""
    ...

  async def execute_round(
      self,
      sender: agent_senders.AgentSender,
      prompts_mod: prompts_protocol.PromptsModule,
      q_count: int,
      h_text: str,
      state: checkpoint_lib.RunState,
      injected_queries: list[str] | None = None,
  ) -> None:
    """Runs Step 4: sends the probes and scores the responses."""
    ...

  def count_queries(self, agent: RoundAgent, q_count: int) -> int | None:
    """Returns the queries to charge, or None if the scrape failed."""
    ...

  def null_rate(self) -> float:
    """Returns the reference rate of the task's proportion tests."""
    ...


class TextTaskExecutor:
  """Text tasks: the agent scrapes the target and judges or measures."""

  name: ClassVar[str] = "text"
  exclude_analyze: ClassVar[bool] = False
  uses_preregistration: ClassVar[bool] = True
  uses_text_querygen: ClassVar[bool] = True

  def __init__(
      self, config: config_lib.DiscoConfig, log_fn: Callable[..., None]
  ):
    self.config = config
    self.log_fn = log_fn
    self.last_arms: list[tuple[str, int, int]] | None = None

  def null_rate(self) -> float:
    """Returns the configured task null rate, else the seed's stated null rate, else 0.5."""
    if (
        self.config.task_null_proportion is not None
        and self.config.task_null_proportion > 0.0
    ):
      return float(self.config.task_null_proportion)
    return stats_utils.extract_null_proportion_from_seed(
        self.config.seed_instruction, default=0.5
    )

  def intro_prompt(
      self, prompts_mod: prompts_protocol.PromptsModule, *, is_baseline: bool
  ) -> str:
    """Returns the introductory prompt for text discovery rounds."""
    config = self.config
    fdr_prompts = cast(prompts_protocol.FDRPromptsModule, prompts_mod)
    intro_prompt = fdr_prompts.get_disco_fresh_agent_intro_prompt(
        config.target_model,
        config.seed_instruction,
        chunk_size=config.chunk_size,
        e_min=config.e_min,
        e_max=config.e_max,
        baseline_mode=is_baseline,
        show_all_failures=config.show_all_failures,
    )
    if config.procedural_autorater:
      intro_prompt += prompts_mod.get_procedural_autorater_addendum(
          config.procedural_autorater,
          allow_continuous_means=config.allow_continuous_means_test,
      )
    return intro_prompt

  async def execute_round(
      self,
      sender: agent_senders.AgentSender,
      prompts_mod: prompts_protocol.PromptsModule,
      q_count: int,
      h_text: str,
      state: checkpoint_lib.RunState,
      injected_queries: list[str] | None = None,
  ) -> None:
    """Asks the agent to scrape the target and judge or measure responses."""
    del h_text  # The agent already has it in context.
    # Failure summaries reach the agent in the round context (Step 2), gated
    # by --show_all_failures / --show_intra_hypothesis_failures; Step 4 does
    # not repeat them.
    del state
    log_fn = self.log_fn
    query_preamble = round_prompts.injected_queries_preamble(
        injected_queries or []
    )

    if self.config.procedural_autorater:
      log_fn("\nStep 4: Scrape and Measure execution")
      prompt = prompts_mod.get_procedural_scrape_and_measure_prompt(
          q_count, self.config.procedural_autorater
      )
    else:
      log_fn("\nStep 4: Scrape and Judge execution")
      prompt = prompts_mod.get_scrape_and_judge_execution_prompt(q_count)
    prompt = query_preamble + prompt
    result = await sender.send_step(prompt)
    log_fn(f"\n  [Execution Summary] {result[:2000]}")

  def count_queries(self, agent: RoundAgent, q_count: int) -> int | None:
    """Counts all responses of the agent, else the last scrape's."""
    del q_count  # Text rounds count what was actually scraped.
    if agent.last_scrape_status == "FAILED" or not agent.last_responses:
      return None
    return (
        len(agent.all_responses)
        if getattr(agent, "all_responses", None)
        else len(agent.last_responses)
    )


class SymmetryTaskExecutor:
  """Visual reflection symmetry: image probes with algorithmic scoring."""

  name: ClassVar[str] = "visual_symmetry"
  exclude_analyze: ClassVar[bool] = True
  uses_preregistration: ClassVar[bool] = False
  uses_text_querygen: ClassVar[bool] = False

  def __init__(
      self, config: config_lib.DiscoConfig, log_fn: Callable[..., None]
  ):
    self.config = config
    self.log_fn = log_fn
    self.last_arms: list[tuple[str, int, int]] | None = None

  def null_rate(self) -> float:
    """Returns the reference accuracy the symmetry proportion test uses.

    Resolution order matches the verifier's (`precision_estimation.
    _default_null_rate`): a rate stated in the seed wins, otherwise the task
    definition's `null_proportion`. The prompt previously hard-coded "90%",
    which silently became a lie whenever the seed or the task metadata changed
    -- and the agent tests against the rate it is TOLD, so a stale number here
    produces hypotheses evaluated against a different null than the pipeline
    scores them with.
    """
    task_default = float(
        visual_symmetry_hypotheses.VISUAL_SYMMETRY_TASK.metadata.get(
            "null_proportion", 0.9
        )
    )
    return stats_utils.extract_null_proportion_from_seed(
        self.config.seed_instruction, default=task_default
    )

  def intro_prompt(
      self, prompts_mod: prompts_protocol.PromptsModule, *, is_baseline: bool
  ) -> str:
    """Returns the introductory prompt for visual symmetry completion testing.

    Delegates to `visual_symmetry_prompts` so that discovery and verification
    describe the same task. This text used to be a hand-maintained copy that
    advertised only 9 of the 22 shape families and described scoring as a raw
    ink-mask IoU against ground truth, which is not the metric in use.

    Args:
      prompts_mod: Unused; the symmetry prompts are task-specific.
      is_baseline: Unused; the intro is the same in every mode.
    """
    del prompts_mod, is_baseline
    return visual_symmetry_prompts.task_specification(
        target_model=self.config.target_model,
        q_count=self.config.chunk_size,
        null_rate=self.null_rate(),
        verifying=False,
    )

  async def execute_round(
      self,
      sender: agent_senders.AgentSender,
      prompts_mod: prompts_protocol.PromptsModule,
      q_count: int,
      h_text: str,
      state: checkpoint_lib.RunState,
      injected_queries: list[str] | None = None,
  ) -> None:
    """Executes visual symmetry probes and algorithmic scoring."""
    del prompts_mod, state, injected_queries  # Not used by symmetry rounds.
    log_fn = self.log_fn
    self.last_arms = None
    log_fn("\nStep 4: Visual Symmetry Probe Generation & Algorithmic Scoring")

    # Reset any previous symmetry output images / judgments so this step's
    # outputs are clean.
    agent: Any = cast(Any, getattr(sender, "agent", None))
    if agent:
      agent.reset_symmetry_state()

    contrast = visual_symmetry_prompts.infer_contrast(h_text)
    if agent and contrast.is_contrastive and contrast.condition_b is not None:
      n_a = (q_count + 1) // 2
      n_b = q_count - n_a
      success_criterion = visual_symmetry_prompts.infer_success_criterion(
          h_text
      )
      log_fn(
          f"\n  [Discovery Contrastive] Axis '{contrast.axis}':"
          f" A={contrast.condition_a.label} (n={n_a}) vs"
          f" B={contrast.condition_b.label} (n={n_b}),"
          f" criterion='{success_criterion}'"
      )
      combined_responses: list[Any] = []
      arm_counts: list[tuple[int, int]] = []
      self.last_arms = []
      for cond, n_probes in (
          (contrast.condition_a, n_a),
          (contrast.condition_b, n_b),
      ):
        agent.reset_symmetry_state(per_probe=True)
        await agent.run_symmetry_probes(cond.probe_args(n_probes))
        await agent.score_symmetry({"success_criterion": success_criterion})
        combined_responses.extend(getattr(agent, "last_responses", []) or [])
        vals = list(getattr(agent, "last_scalar_values", []) or [])
        arm_counts.append((int(sum(vals)), len(vals)))
        self.last_arms.append((cond.label, len(vals), n_probes))

      agent.set_combined_responses(combined_responses)
      succ_a, obs_a = arm_counts[0]
      succ_b, obs_b = arm_counts[1]
      summary_msg = visual_symmetry_prompts.contrastive_stats_prompt(
          h_text=h_text,
          plan=contrast,
          n_success_a=succ_a,
          n_a=obs_a,
          n_success_b=succ_b,
          n_b=obs_b,
      )
      await sender.send_step(summary_msg, enable_tools=False)
      return

    prompt = visual_symmetry_prompts.probe_execution_instructions(
        q_count=q_count,
        h_text=h_text,
        verifying=False,
    )

    result = await sender.send_step(prompt)
    log_fn(f"\n  [Symmetry Execution Summary] {result[:2000]}")

    # Determine expected shape_type from hypothesis text. Shared with the
    # verifier so that a hypothesis is probed with the same family in both
    # phases; the previous local copy knew only 9 of the 22 families and
    # mapped every hard-family claim onto an easy look-alike or onto the
    # `english_letter` default.
    combined_text = h_text.lower()
    expected_shape = visual_symmetry_prompts.infer_shape_type(h_text)

    acceptable_shapes = {c.shape_type for c in contrast.conditions}
    acceptable_shapes.add(expected_shape)

    # Safety guard: If the agent did not invoke symmetry tools or used
    # mismatched shape_type, execute/correct.
    if agent:
      last_shape = agent.last_symmetry_shape_type
      needs_execution = (
          not agent.symmetry_output_images or not agent.last_judgments
      )
      needs_correction = (
          bool(agent.symmetry_output_images)
          and last_shape not in acceptable_shapes
          and (
              expected_shape in ("chinese_char", "greek_letter")
              # A hard-family claim probed with its easy look-alike is the
              # worst case: 'complex_chinese' answered with 'chinese_char'
              # probes tests the EASY family, so a real failure verifies as a
              # success and the hypothesis is recorded as a false negative.
              or expected_shape in visual_symmetry_prompts.HARD_SHAPE_TYPES
              or last_shape in visual_symmetry_prompts.HARD_SHAPE_TYPES
              or "chinese" in combined_text
              or "greek" in combined_text
          )
      )

      if needs_execution or needs_correction:
        action_name = "Correction" if needs_correction else "Fallback"
        log_fn(
            f"\n  [Symmetry {action_name}]"
            f" {'Agent probed ' + last_shape + ' but hypothesis requires ' + expected_shape if needs_correction else 'Agent did not execute symmetry tools'};"
            f" executing probes: shape_type={expected_shape},"
            f" n_probes={q_count}"
        )
        await agent.run_symmetry_probes({
            "shape_type": expected_shape,
            "n_probes": q_count,
        })
        await agent.score_symmetry({})

  def count_queries(self, agent: RoundAgent, q_count: int) -> int | None:
    """Counts last responses, else output images, else all responses, else q_count."""
    if agent.last_scrape_status == "FAILED" or (
        not agent.last_responses and not agent.symmetry_output_images
    ):
      return None
    if getattr(agent, "last_responses", None):
      return len(agent.last_responses)
    if getattr(agent, "symmetry_output_images", None):
      return len(agent.symmetry_output_images)
    if getattr(agent, "all_responses", None):
      return len(agent.all_responses)
    return q_count


_TaskFactory = Callable[
    [config_lib.DiscoConfig, Callable[..., None]], TaskExecutor
]

# Task families with their own executor, keyed by `task_suite`. Every other
# suite runs as a text task.
_REGISTRY: dict[str, _TaskFactory] = {
    SymmetryTaskExecutor.name: SymmetryTaskExecutor,
}


def resolve_task_executor(
    config: config_lib.DiscoConfig, log_fn: Callable[..., None]
) -> TaskExecutor:
  """Returns the executor for the run's task family.

  Args:
    config: Run config.
    log_fn: Run logger, used by the executor.

  Returns:
    The registered executor for `config.task_suite`; else, when `task_suite`
    is empty and the seed instruction mentions "symmetry", the symmetry
    executor (see `task_detection`); else the text executor.
  """
  factory = _REGISTRY.get(config.task_suite)
  if factory is not None:
    return factory(config, log_fn)
  if task_detection.is_symmetry_task(config):
    return SymmetryTaskExecutor(config, log_fn)
  return TextTaskExecutor(config, log_fn)
