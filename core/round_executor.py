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

"""Runs one DISCO discovery round (Steps 2b-5) with a fresh agent.

A round, for one hypothesis:

  intro -> DISCO context -> Step 2b pre-registration -> Step 3 isolated query
  generation -> Step 4 execution -> budget charge -> recording
  -> Step 4b failure reflection -> Step 5 stats

Hypotheses are operationalized directly (there is no sub-hypothesis step), and
Step 3 always runs on a fresh, context-free generator rather than the round's
agent.

`RoundExecutor.run` returns a `RoundResult`. The caller (the runner loop)
switches on `RoundResult.status`; the e-value update, deactivation and
checkpointing stay in the loop.

The LLM call order and every prompt are pinned by the smoke goldens in
testing/testdata/disco_runner_smoke/, so this module must not reorder steps.
"""

from collections.abc import Callable, Iterable
import dataclasses
import json
import os
import random
import re
from typing import Literal

from disco import backbone_client as backbone_lib
from disco import standalone_agent
from disco import stats_utils
from disco import taxonomy_utils
from disco.agent import factory as agent_factory_lib
from disco.agent import senders as agent_senders
from disco.core import checkpoint as checkpoint_lib
from disco.core import config as config_lib
from disco.core import history
from disco.core import prompts_protocol
from disco.core import records as records_lib
from disco.core import salient
from disco.core import task_executor
from disco.parsing import parsers as parsers_lib
from disco.stats import plan as stats_plan
from disco.verification import scoring

# Sentinel p-value to indicate statistical parsing failure.
STATS_PARSING_FAILURE_P_VALUE = stats_utils.STATS_PARSING_FAILURE_P_VALUE

RoundStatus = Literal["ok", "scrape_failed", "short", "stats_failed"]


def resolve_is_paired(
    config: config_lib.DiscoConfig,
    plan: stats_plan.PreRegisteredTestPlan | None,
) -> bool:
  """Resolves whether a round collects paired A/B data.

  The config flags force pairing on for the whole run; the plan supplies the
  per-hypothesis choice when they do not. Resolved from an explicitly passed
  plan rather than read back off the parser, so a failed parse cannot leave a
  previous hypothesis's answer in place.

  Args:
    config: Run config.
    plan: The pre-registered plan, or None if extraction failed or skipped.

  Returns:
    True if the round collects paired data, False otherwise.
  """
  return bool(config.is_paired or (plan is not None and plan.implies_paired))


@dataclasses.dataclass(frozen=True)
class QueryPlan:
  """The outcome of Step 3 for a text task.

  Attributes:
    q_count: Number of queries the round will run.
    queries: The pre-generated queries Step 4 injects into the round's agent.
      None for tasks without text queries (visual symmetry).
  """

  q_count: int
  queries: list[str] | None


@dataclasses.dataclass
class RoundResult:
  """The outcome of one round.

  Attributes:
    h_id: Hypothesis that was probed.
    status: "ok"; "scrape_failed" (no responses; the last execution step was not
      charged and the round must be skipped); "short" (fewer responses than
      `--min_response_fraction` of the planned queries; not charged, nothing
      recorded, skipped); or "stats_failed" (budget was charged but Step 5
      returned the parsing-failure sentinel).
    stats: Step 5 result; None on "scrape_failed" and "short".
    queries_spent: Queries charged to the budget during the round.
    test_plan: The pre-registered plan used by the round, if any.
    skip_kind: For a non-"ok" round, why it was skipped (one of
      `scoring.SKIP_KINDS`): "target_side" when the target's scrape failed,
      "short" when it came back short, "auditor_side" when the auditor never
      produced responses or its stats reply did not parse. None for an "ok"
      round.
  """

  h_id: str
  status: RoundStatus
  stats: stats_plan.StatsResult | None
  queries_spent: int
  test_plan: stats_plan.PreRegisteredTestPlan | None
  skip_kind: str | None = None


class RoundExecutor:
  """Runs DISCO rounds against a shared `RunState`.

  The executor reads and updates `state.remaining_budget`,
  `state.hypothesis_groups`, `state.total_scrape_rounds` and reads
  `state.terminal_statuses`. It holds no other state, so creating one per
  round or per call is cheap.
  """

  def __init__(
      self,
      config: config_lib.DiscoConfig,
      parsers: parsers_lib.Parsers,
      state: checkpoint_lib.RunState,
      log_fn: Callable[..., None],
      task: task_executor.TaskExecutor | None = None,
  ):
    """Initializes the executor.

    Args:
      config: Run config.
      parsers: Parsers for the agent's structured answers.
      state: The run state shared with the loop.
      log_fn: Run logger.
      task: Task executor for Step 4; resolved from the config on each round if
        None.
    """
    self.config = config
    self.parsers = parsers
    self.state = state
    self.log_fn = log_fn
    self.task = task

  # ---------------------------------------------------------------------------
  # Step 2b: pre-registration
  # ---------------------------------------------------------------------------

  async def preregister(
      self,
      sender: agent_senders.AgentSender,
      prompts_mod: prompts_protocol.PromptsModule,
      h_text: str,
      h_record: records_lib.HypothesisRecord,
  ) -> stats_plan.PreRegisteredTestPlan | None:
    """Elicits the statistical test plan before any data is generated.

    Runs once per hypothesis: the plan is cached on `h_record` and reused for
    every subsequent round. That ordering is the point -- the agent commits to
    a test type, null baseline, direction and pairing while it still has no
    observations, so it cannot tune any of them to the data it later sees.

    It is also what lets an agent opt into a paired design during discovery.
    Before this step existed the only per-hypothesis pairing signal was a
    post-hoc guess by the distiller LLM, so `paired=False` was effectively
    "never paired" rather than "agent's choice".

    Args:
      sender: The agent sender for this round.
      prompts_mod: The active prompts module.
      h_text: Hypothesis text.
      h_record: Record for this hypothesis, used to cache the plan.

    Returns:
      The pre-registered plan, or None if extraction failed.
    """
    if h_record.test_plan is not None:
      return h_record.test_plan

    log_fn = self.log_fn
    log_fn("\nStep 2b: Statistical test pre-registration")
    prompt = prompts_mod.get_test_preregistration_prompt(h_text)
    plan: stats_plan.PreRegisteredTestPlan | None = (
        await sender.send_and_extract(
            prompt,
            parsers_lib.parse_and_validate_preregistration,
            max_retries=2,
            fallback_value=None,
            log_fn=log_fn,
            retry_prompt_fn=prompts_mod.get_default_preregistration_retry_prompt,
            parsers=self.parsers,
            enable_tools=False,
        )
    )
    if plan is None:
      # Not fatal: the round proceeds on the config-level pairing setting, the
      # same behaviour as before pre-registration existed in discovery.
      log_fn(
          "\n  [Pre-registration] Extraction failed; proceeding without a"
          " pre-registered plan."
      )
      return None

    h_record.test_plan = plan
    log_fn(
        f"\n  [Pre-registered plan] test_type='{plan.test_type}',"
        f" metric='{plan.target_metric}', null={plan.null_baseline},"
        f" alternative='{plan.alternative}', paired={plan.implies_paired}"
    )
    return plan

  def resolve_is_paired(
      self, plan: stats_plan.PreRegisteredTestPlan | None
  ) -> bool:
    """See the module-level `resolve_is_paired`."""
    return resolve_is_paired(self.config, plan)

  # ---------------------------------------------------------------------------
  # Step 3: query generation
  # ---------------------------------------------------------------------------

  def create_isolated_sender(
      self,
      prompts_mod: prompts_protocol.PromptsModule,
      output_dir: str,
  ) -> agent_senders.BaseSender:
    """Creates a fresh, isolated agent and sender for query generation."""
    return agent_factory_lib.build_isolated_sender(
        self.config,
        self.log_fn,
        prompts_mod,
        output_dir,
        self.state.total_scrape_rounds,
    )

  async def generate_queries_isolated(
      self,
      h_text: str,
      q_count: int,
      excluded_queries: Iterable[str] | None = None,
      is_paired: bool | None = None,
  ) -> list[str]:
    """Generates queries using a fresh, context-free subagent.

    Sends a single self-contained prompt with just the seed, H and chunk_size,
    and parses the numbered query list from the response.

    Args:
      h_text: Hypothesis text.
      q_count: Exact number of queries to generate.
      excluded_queries: Optional container of previously tested queries to
        exclude.
      is_paired: Whether to generate A/B pairs, normally resolved by the caller
        from the config flags and the pre-registered plan. Defaults to the
        config-level setting so an omitted argument cannot silently turn a
        paired run into an unpaired one.

    Returns:
      List of query strings.
    """
    log_fn = self.log_fn
    if is_paired is None:
      is_paired = self.resolve_is_paired(None)
    mode_str = " (Paired)" if is_paired else ""
    log_fn(
        f"\nStep 3 (Isolated): Query generation via clean subagent{mode_str}"
    )

    prompts_mod = self.config.get_prompts_module()
    prompt = prompts_mod.get_isolated_query_generation_prompt(
        seed_instruction=self.config.seed_instruction,
        hypothesis=h_text,
        num_queries=q_count,
        excluded_queries=list(excluded_queries) if excluded_queries else None,
        procedural_autorater=self.config.procedural_autorater,
        paired=is_paired,
    )

    backbone = backbone_lib.get_backbone_client(self.config.agent_model)
    try:
      result = await backbone.generate(
          model=self.config.agent_model,
          prompt=prompt,
          temperature=self.config.isolated_query_gen_temperature,
      )
    except backbone_lib.LLM_CALL_ERRORS as e:
      log_fn(f"\n  [Isolated Query Gen] LLM call failed: {e}")
      result = ""
    log_fn(f"\n  [Isolated Query Gen] Response preview: {result[:500]}")

    def _parse_isolated_text(text: str, paired: bool) -> list[str]:
      parsed: list[str] = []
      if paired:
        json_str = text.strip()
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
          json_str = match.group(1).strip()
        elif "[" in text and "]" in text:
          start = text.find("[")
          end = text.rfind("]") + 1
          json_str = text[start:end]

        try:
          data = json.loads(json_str)
          if isinstance(data, dict):
            for val in data.values():
              if isinstance(val, list):
                data = val
                break
          if isinstance(data, list):
            for item in data:
              if isinstance(item, dict):
                a = (
                    item.get("group_a")
                    or item.get("groupA")
                    or item.get("a")
                    or item.get("prompt_a")
                )
                b = (
                    item.get("group_b")
                    or item.get("groupB")
                    or item.get("b")
                    or item.get("prompt_b")
                )
                if a and b:
                  clean_a = re.sub(
                      r"^\[(?:A|B|Group A|Group B)\]\s*",
                      "",
                      str(a).strip(),
                      flags=re.IGNORECASE,
                  ).strip()
                  clean_b = re.sub(
                      r"^\[(?:A|B|Group A|Group B)\]\s*",
                      "",
                      str(b).strip(),
                      flags=re.IGNORECASE,
                  ).strip()
                  parsed.extend([clean_a, clean_b])
        except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
          pass

        if not parsed:
          for line in text.split("\n"):
            line = line.strip()
            match = re.search(r"^\d+[\.\\)]\s+(.+)$", line)
            if match:
              clean_q = re.sub(
                  r"^\[(?:A|B|Group A|Group B)\]\s*",
                  "",
                  match.group(1).strip(),
                  flags=re.IGNORECASE,
              ).strip()
              parsed.append(clean_q)
      else:
        for line in text.split("\n"):
          match = re.search(r"^\d+[\.\\)]\s+(.+)$", line.strip())
          if match:
            parsed.append(match.group(1).strip())
        if not parsed:
          parsed = [l.strip() for l in text.split("\n") if l.strip()]

      return parsed

    target_total = 2 * (q_count // 2) if is_paired else q_count
    queries: list[str] = _parse_isolated_text(result, is_paired)

    # Backfill replenishment if LLM returned fewer queries than requested
    retries = 0
    max_retries = 3
    while len(queries) < target_total and retries < max_retries:
      retries += 1
      needed = target_total - len(queries)
      if is_paired and needed % 2 != 0:
        needed += 1
      log_fn(
          f"  [Isolated Replenish] Deficit of {needed} queries"
          f" ({len(queries)}/{target_total}). Retrying"
          f" {retries}/{max_retries}..."
      )
      combined_excluded = set(excluded_queries or []) | set(queries)
      if combined_excluded:
        sampled_exclusions = random.sample(
            list(combined_excluded),
            min(200, len(combined_excluded)),
        )
      else:
        sampled_exclusions = None
      retry_prompt = prompts_mod.get_isolated_query_generation_prompt(
          seed_instruction=self.config.seed_instruction,
          hypothesis=h_text,
          num_queries=needed,
          excluded_queries=sampled_exclusions,
          procedural_autorater=self.config.procedural_autorater,
          paired=is_paired,
      )
      try:
        retry_text = await backbone.generate(
            model=self.config.agent_model,
            prompt=retry_prompt,
            temperature=self.config.isolated_query_gen_temperature,
        )
      except backbone_lib.LLM_CALL_ERRORS as e:
        log_fn(f"\n  [Isolated Query Gen Replenish] LLM call failed: {e}")
        retry_text = ""
      retry_parsed = _parse_isolated_text(retry_text, is_paired)
      if retry_parsed:
        queries.extend(retry_parsed)

    if is_paired:
      if len(queries) > target_total:
        queries = queries[:target_total]
      elif len(queries) % 2 != 0:
        queries = queries[:-1]
      log_fn(
          f"\n  [Isolated Query Gen (Paired)] Parsed {len(queries)//2} pairs"
          f" ({len(queries)} queries)"
      )
    else:
      if len(queries) > target_total:
        queries = queries[:target_total]
      log_fn(f"\n  [Isolated Query Gen] Parsed {len(queries)} queries")

    return queries

  async def plan_queries(
      self,
      prompts_mod: prompts_protocol.PromptsModule,
      h_text: str,
      h_record: records_lib.HypothesisRecord,
      *,
      is_paired: bool,
      output_dir: str,
  ) -> QueryPlan:
    """Step 3 for text tasks: isolated query generation.

    Queries always come from a fresh, context-free generator (standard or
    taxonomy) and are injected into Step 4; the round's agent never writes
    its own queries, so the failure history and status it has seen cannot
    bias them. A taxonomy run that yields no queries falls back to the
    standard isolated generator.

    Args:
      prompts_mod: Prompts module.
      h_text: Hypothesis text.
      h_record: The hypothesis record (for already-executed queries).
      is_paired: Whether queries must come in (A, B) pairs.
      output_dir: Experiment output dir (for the isolated taxonomy sender).

    Returns:
      The query plan.
    """
    remaining_budget = self.state.remaining_budget
    if self.config.query_generation_mode == "taxonomy":
      isolated_sender = self.create_isolated_sender(prompts_mod, output_dir)
      q_count, taxonomy_queries = (
          await taxonomy_utils.run_taxonomy_query_generation(
              self.config,
              isolated_sender,
              prompts_mod,
              remaining_budget,
              h_text,
              self.log_fn,
              paired=is_paired,
          )
      )
      if q_count != 0:
        return QueryPlan(q_count, taxonomy_queries)
    queries = await self.generate_queries_isolated(
        h_text=h_text,
        q_count=min(self.config.chunk_size, remaining_budget),
        excluded_queries=h_record.executed_queries,
        is_paired=is_paired,
    )
    return QueryPlan(len(queries), queries)

  # ---------------------------------------------------------------------------
  # Step 4 accounting
  # ---------------------------------------------------------------------------

  def charge_budget(
      self,
      agent: task_executor.RoundAgent,
      h_id: str,
      task: task_executor.TaskExecutor,
      q_count: int,
  ) -> int | None:
    """Charges the executed queries to the budget and the hypothesis.

    Args:
      agent: The round's agent, after execution.
      h_id: Hypothesis being probed.
      task: The executor whose Step 4 just ran; it counts the queries.
      q_count: Planned query count.

    Returns:
      The number of queries charged, or None (and nothing is charged) if the
      scrape failed or returned no responses; the round must then be skipped.
    """
    actual_queries = task.count_queries(agent, q_count)
    if actual_queries is None:
      self.log_fn(
          "\n[Scrape Error] Target model scraping failed or returned no valid"
          " responses. Skipping round without query deduction or hypothesis"
          " penalty."
      )
      return None
    self.state.remaining_budget -= actual_queries
    self.state.hypothesis_groups[h_id].queries += actual_queries
    return actual_queries

  def short_round_reason(
      self,
      agent: task_executor.RoundAgent,
      task: task_executor.TaskExecutor,
      q_count: int,
      is_paired: bool,
  ) -> str:
    """Why the round's Step 4 came back too short to use, or "".

    Args:
      agent: The round's agent, after execution.
      task: The executor whose Step 4 just ran.
      q_count: Planned query count.
      is_paired: Whether the round collected (A, B) pairs; complete pairs are
        then counted against `q_count // 2`.

    Returns:
      Non-empty reason string if the round is too short, or "" if usable.
    """
    n_received = task.count_queries(agent, q_count)
    if n_received is None:
      # A failed scrape; `charge_budget` skips the round as before.
      return ""
    if task.last_arms:
      # A two-arm symmetry round: each arm must reach the fraction of its own
      # allocation, as both halves of a pair must survive in a paired round.
      return scoring.short_arms_reason(
          task.last_arms, self.config.min_response_fraction
      )
    if is_paired:
      n_pairs = stats_utils.count_complete_pairs(
          getattr(agent, "last_response_indices", None)
      )
    else:
      n_pairs = 0
    return scoring.short_batch_reason(
        is_paired=is_paired,
        q_count=q_count,
        n_responses=n_received,
        n_pairs=n_pairs,
        min_fraction=self.config.min_response_fraction,
    )

  # ---------------------------------------------------------------------------
  # Recording, Step 4b reflection, Step 5 stats
  # ---------------------------------------------------------------------------

  async def record(
      self,
      agent: standalone_agent.StandaloneAgent,
      h_id: str,
      h_text: str,
      h_record: records_lib.HypothesisRecord,
  ) -> None:
    """Records the round's queries and salient examples."""
    hypothesis_groups = self.state.hypothesis_groups

    # Record executed queries
    recorded_responses = (
        agent.all_responses
        if getattr(agent, "all_responses", None)
        else agent.last_responses
    )
    if recorded_responses:
      for resp in recorded_responses:
        if resp and resp.probe:
          p_text = resp.probe.content.as_text(
              ignore_unsupported_types=True
          ).strip()
          if p_text:
            h_record.executed_queries.add(p_text)

    # Record salient examples using LLM selection
    candidate_judgments = (
        agent.all_judgments
        if getattr(agent, "all_judgments", None)
        else agent.last_judgments
    )
    salient_examples = await salient.select_salient_examples(
        candidate_judgments=candidate_judgments,
        seed_instruction=self.config.seed_instruction,
        hypothesis_groups=hypothesis_groups,
        current_h_id=h_id,
        current_h_text=h_text,
        model=self.config.agent_model,
        log_fn=self.log_fn,
        terminal_statuses=self.state.terminal_statuses,
    )
    for ex in salient_examples:
      if ex not in h_record.failures:
        h_record.failures.append(ex)
    if len(h_record.failures) > 3:
      h_record.failures = h_record.failures[-3:]

  async def reflect(
      self,
      sender: agent_senders.AgentSender,
      prompts_mod: prompts_protocol.PromptsModule,
  ) -> str:
    """Asks the agent to reflect on recent execution results and extract salient examples."""
    log_fn = self.log_fn
    step_name = "Salient Example Extraction"
    log_fn(f"\nStep 4b: {step_name}")
    prompt = prompts_mod.get_recent_failures_reflection_prompt()
    result = await sender.send_step(prompt)
    log_fn(f"\n  [Failure Reflection Summary]\n{result}")
    return result

  async def stats(
      self,
      sender: agent_senders.AgentSender,
      prompts_mod: prompts_protocol.PromptsModule,
      plan: stats_plan.PreRegisteredTestPlan | None = None,
  ) -> stats_plan.StatsResult:
    """Asks the agent to run a statistical significance test.

    Args:
      sender: The agent sender for this round.
      prompts_mod: The active prompts module.
      plan: The pre-registered plan, if one was obtained. Supplies the prompt
        with the committed test type, null and direction, and -- for a paired
        plan -- the instruction to report pairs rather than responses and the
        derived tie policy.

    Returns:
      The parsed stats result.
    """
    self.log_fn("\nStep 5: Statistical significance testing")
    res = await sender.send_and_extract(
        prompts_mod.get_stats_significance_prompt(plan=plan),
        parsers_lib.parse_and_validate_stats,
        fallback_value=stats_plan.StatsResult(
            test_type="Unknown", p_value=STATS_PARSING_FAILURE_P_VALUE
        ),
        log_fn=self.log_fn,
        parsers=self.parsers,
    )
    return res

  # ---------------------------------------------------------------------------
  # The round
  # ---------------------------------------------------------------------------

  async def run(
      self,
      h_id: str,
      h_text: str,
      h_record: records_lib.HypothesisRecord,
      cross_hyp_summary: str = "",
  ) -> RoundResult:
    """Runs Steps 2b-5 for one hypothesis with a fresh agent.

    1. The context prompt shows coarsened status in DISCO mode and full details
       in baseline mode.
    2. Step 4b failure reflection still runs (within-round), but its output
       is only stored on the record; it is NOT returned to the selector.

    Args:
      h_id: Hypothesis to probe.
      h_text: Hypothesis text.
      h_record: The hypothesis record.
      cross_hyp_summary: Cross-hypothesis failure summary for the context prompt
        (empty under DISCO).

    Returns:
      The round result.
    """
    config = self.config
    log_fn = self.log_fn
    is_baseline = config.baseline_mode or config.baseline_type1_mode
    if config.baseline_type1_mode:
      mode_str = "baseline Type-I mode"
    elif config.baseline_mode:
      mode_str = "baseline mode"
    else:
      mode_str = "DISCO mode"
    log_fn(
        f"\n[Fresh Agent] Initializing for Hypothesis ID: {h_id} ({mode_str})"
    )

    output_dir = os.path.join(config.output_dir, config.experiment_name)
    prompts_mod = config.get_prompts_module()
    agent, sender = agent_factory_lib.build_agent_and_sender(
        config,
        log_fn,
        prompts_mod,
        output_dir=output_dir,
        initial_scrape_round=self.state.total_scrape_rounds,
    )

    task = self.task or task_executor.resolve_task_executor(config, log_fn)
    intro_prompt = task.intro_prompt(prompts_mod, is_baseline=is_baseline)
    await sender.send_step(intro_prompt, enable_tools=False)

    # Coarsened-e-value context (no e-values, no failures).
    context = history.build_disco_round_context(
        config, self.state, h_text, h_record, h_id, cross_hyp_summary
    )
    await sender.send_step(context, enable_tools=False)

    # Step 2b: Statistical test pre-registration (see `preregister` for why
    # this has to happen before any data is generated).
    test_plan: stats_plan.PreRegisteredTestPlan | None = None
    if task.uses_preregistration:
      test_plan = await self.preregister(sender, prompts_mod, h_text, h_record)
    is_paired = self.resolve_is_paired(test_plan)

    def skipped() -> RoundResult:
      # A scrape the agent ran and the target failed is target-side; no
      # responses without a failed scrape means the agent never scraped.
      target_failed = getattr(agent, "last_scrape_status", "OK") == "FAILED"
      # Whatever the failed scrape saved stays in its folders, so the next
      # agent must start numbering after them (as on the short path).
      self.state.total_scrape_rounds = agent.scrape_round
      return RoundResult(
          h_id=h_id,
          status="scrape_failed",
          stats=None,
          queries_spent=0,
          test_plan=test_plan,
          skip_kind="target_side" if target_failed else "auditor_side",
      )

    # Steps 3-4: Query generation and execution.
    if task.uses_text_querygen:
      plan = await self.plan_queries(
          prompts_mod,
          h_text,
          h_record,
          is_paired=is_paired,
          output_dir=output_dir,
      )
    else:
      plan = QueryPlan(
          min(config.chunk_size, self.state.remaining_budget), None
      )
    await task.execute_round(
        sender,
        prompts_mod,
        plan.q_count,
        h_text,
        self.state,
        injected_queries=plan.queries,
    )
    short = self.short_round_reason(agent, task, plan.q_count, is_paired)
    if short:
      # Like a failed scrape: nothing is charged or recorded, so a short
      # batch cannot steer later rounds. The responses were saved, so the
      # next agent must not reuse their folders.
      log_fn(
          f"\n[Short Round] {short}. Skipping round without query"
          " deduction or hypothesis penalty."
      )
      self.state.total_scrape_rounds = agent.scrape_round
      return RoundResult(
          h_id=h_id,
          status="short",
          stats=None,
          queries_spent=0,
          test_plan=test_plan,
          skip_kind="short",
      )
    spent = self.charge_budget(agent, h_id, task, plan.q_count)
    if spent is None:
      return skipped()

    await self.record(agent, h_id, h_text, h_record)

    # Step 4b: Failure reflection
    reflection_summary = await self.reflect(sender, prompts_mod)
    if reflection_summary:
      h_record.failure_reflections.append(reflection_summary)

    # Step 5: Stats.
    res = await self.stats(sender, prompts_mod, plan=test_plan)

    # Update the scrape round offset for the next agent.
    self.state.total_scrape_rounds = agent.scrape_round

    stats_failed = res.p_value == STATS_PARSING_FAILURE_P_VALUE
    return RoundResult(
        h_id=h_id,
        status="stats_failed" if stats_failed else "ok",
        stats=res,
        queries_spent=spent,
        test_plan=test_plan,
        skip_kind="auditor_side" if stats_failed else None,
    )
