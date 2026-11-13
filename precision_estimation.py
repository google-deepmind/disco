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

"""Precision estimation for verified hypotheses."""

import asyncio
import dataclasses
import json
import os
import random
import re
import typing
from typing import Any, Callable, Mapping, Sequence, cast
import zlib

import numpy as np

from disco import backbone_client as backbone_lib
from disco import stats_utils
from disco import taxonomy_utils
from disco import verifier_config
from disco.agent import factory as agent_factory_lib
from disco.agent import senders as agent_senders
from disco.core import config as config_lib
from disco.core import records as records_lib
from disco.core import task_detection
from disco.parsing import parsers as parsers_lib
from disco.stats import plan as stats_plan
from disco.tasks.visual_symmetry import prompts as visual_symmetry_prompts
from disco.verification import scoring


@dataclasses.dataclass
class PairedDiagnostics:
  """Pair-level bookkeeping for a paired verification round.

  Recorded whether or not the pairing was used successfully, so that a run can
  be audited after the fact: a high `tie_rate` means the metric cannot resolve
  the two sides of a pair, and a non-zero `n_orphans` means responses were lost
  in a way that broke their pairing.
  """

  n_pairs: int = 0
  n_tied: int = 0
  n_positive: int = 0
  n_negative: int = 0
  n_orphans: int = 0
  tie_policy: str = "not_applicable"
  unit_of_analysis: str = "response"

  @property
  def tie_rate(self) -> float:
    return self.n_tied / self.n_pairs if self.n_pairs else 0.0


# Records whose test produced no usable measurement carry a non-OK `status`
# (SCRAPE_FAILED, PLAN_INCONSISTENT). How they enter precision/FDR is decided by
# `verification/scoring.score`; see its module docstring.


@dataclasses.dataclass
class HypothesisPrecisionRecord:
  """Per-hypothesis outcome of an out-of-sample precision verification check."""

  hypothesis_text: str
  verified: bool
  final_e_product: float = 1.0
  n_subhyps_used: int = 0
  e_values: list[tuple[str, float]] = dataclasses.field(default_factory=list)
  hypothesis_id: str = ""
  status: str = "OK"
  block_reasons: list[str] = dataclasses.field(default_factory=list)
  p_value: float = 1.0
  sample_size: int = 0
  observed_successes: int = 0
  observed_rate: float = 0.0
  null_rate: float = 0.5
  ci_lower: float = 0.0
  ci_upper: float = 0.0
  paired_diagnostics: PairedDiagnostics | None = None
  # Per-arm counts for a contrastive (two-sample) hypothesis, in (A, B) order.
  # Present only on the contrastive path. Kept on the record rather than in the
  # log alone so that a reported comparison can be re-derived after the fact:
  # `observed_rate` holds arm A's rate, which on its own is indistinguishable
  # from an absolute measurement.
  contrast_diagnostics: list[dict[str, Any]] | None = None
  # Target requests issued and usable responses received across every scrape
  # of this check (retries and backfill included). Diagnostic only.
  n_requested: int = 0
  n_received: int = 0
  # Why a non-OK record failed (block reasons or the plan inconsistency).
  failure_reason: str = ""
  # The single test plan this check pre-registered (or reused), kept so that a
  # retry can run the same test rather than eliciting a new one after seeing
  # the outcome. None when the agent chose no plan (symmetry tasks or failed
  # extraction).
  test_plan: stats_plan.PreRegisteredTestPlan | None = None
  # Queries this check was allotted (response count for the batch). Lets a
  # retry keep the same sample size (see `RetryBudget`).
  allocated_queries: int = 0


@dataclasses.dataclass(frozen=True)
class RetryBudget:
  """Keeps each re-checked hypothesis at the query count of its first check.

  `--retry_failed` re-checks only the failed subset of the original
  hypotheses. Splitting the total verification budget over that subset would
  give every retried hypothesis more queries than it first had, so how much a
  hypothesis is tested would depend on how many others happened to fail.

  Attributes:
    allocated_queries: Queries each hypothesis was first allotted, by id (the
      record's `allocated_queries`; 0 or missing when not recorded).
    n_original: Number of hypotheses the original check split the total budget
      across. Only used for records written before allocations were saved, which
      get the original per-hypothesis share instead.
  """

  allocated_queries: Mapping[str, int]
  n_original: int

  def stored(self, h_id: str) -> int | None:
    """The recorded allocation for `h_id`, or None if there is none."""
    q = self.allocated_queries.get(h_id, 0)
    return q if q > 0 else None


def _annotate_coverage(
    record: HypothesisPrecisionRecord, agents: list[Any]
) -> HypothesisPrecisionRecord:
  """Fills `n_requested` / `n_received` / `failure_reason` on `record`.

  Args:
    record: A freshly built record (counts still zero).
    agents: Every agent that scraped for this check.

  Returns:
    `record`, updated in place.
  """
  for agent in agents:
    for outcome in getattr(agent, "scrape_history", None) or ():
      record.n_requested += outcome.n_requested
      record.n_received += outcome.n_received
  if record.status != "OK" and not record.failure_reason:
    record.failure_reason = "; ".join(record.block_reasons) or record.status
  return record


@dataclasses.dataclass
class PrecisionResult:
  """Aggregated precision summary and per-hypothesis verification records."""

  records: list[HypothesisPrecisionRecord]
  # n_verified / n_evaluated with failed records dropped; None when nothing was
  # evaluated. Writers rescore under --failed_records with `scoring.score`.
  precision: float | None
  n_verified: int
  n_total: int
  n_evaluated: int = 0
  n_statistical_failures: int = 0
  n_execution_errors: int = 0
  n_verification_failures: int = 0

  def __post_init__(self):
    if not self.n_evaluated and self.records:
      self.n_evaluated = sum(1 for r in self.records if r.status == "OK")
    elif not self.n_evaluated and not self.records and self.n_total:
      self.n_evaluated = self.n_total
    if not self.n_statistical_failures and self.records:
      self.n_statistical_failures = sum(
          1 for r in self.records if r.status == "OK" and not r.verified
      )
    if not self.n_execution_errors and self.records:
      self.n_execution_errors = sum(1 for r in self.records if r.status != "OK")
    if not self.n_verification_failures:
      self.n_verification_failures = self.n_execution_errors


_SCIPY_ALT_MAP: dict[str, typing.Literal["greater", "less", "two-sided"]] = {
    "larger": "greater",
    "greater": "greater",
    "smaller": "less",
    "less": "less",
    "two-sided": "two-sided",
}


def _resolve_alternative(
    test_plan: stats_plan.PreRegisteredTestPlan | None,
    res: stats_plan.StatsResult | None = None,
) -> typing.Literal["greater", "less", "two-sided"]:
  """Resolves the Scipy alternative direction from test plan or stats result."""
  if test_plan is not None:
    return _SCIPY_ALT_MAP.get(test_plan.alternative, "greater")
  if res is not None and res.alternative:
    return _SCIPY_ALT_MAP.get(res.alternative, "greater")
  return "greater"


# ── Visual symmetry (multimodal) verification ─────────────────────────────────
#
# The default precision path probes the target with text via
# `generate_single_turn_responses`. That is meaningless for the visual symmetry
# task, whose probes are images and whose success criterion is a continuous ink
# mask IoU. The helpers below mirror the equivalent path in
# `core/task_executor.py` so that verification probes the target the same way
# discovery did. Without them the verifier silently reports a precision
# computed from text responses.

# Shape families the verifier may probe (derived from
# `visual_symmetry_prompts`).
SYMMETRY_SHAPE_TYPES: list[str] = visual_symmetry_prompts.SUPPORTED_SHAPE_TYPES

# Fallback null baseline success rate for visual symmetry completion, used only
# when the registry task carries no `metadata["null_proportion"]` (e.g. a
# hypotheses file verified standalone with no resolvable task). The task is
# defined against a 90% reference accuracy, so a symmetry hypothesis that states
# no rate of its own must be tested against 0.90, not the generic 0.5 coin-flip
# null. Prefer setting `null_proportion` in the task metadata over editing this.
# MUST agree with `VISUAL_SYMMETRY_TASK.metadata["null_proportion"]`.
SYMMETRY_NULL_RATE: float = 0.90

# Generic null for tasks with no task-specific reference rate.
DEFAULT_NULL_RATE: float = 0.5


def _is_symmetry_task(config: verifier_config.VerificationConfig) -> bool:
  """Checks whether this verification run targets visual reflection symmetry."""
  return task_detection.is_symmetry_task(config)


def _default_null_rate(config: verifier_config.VerificationConfig) -> float:
  """Returns the null baseline to assume when nothing else specifies one.

  Resolution order:
    1. `config.task_null_proportion`, set from the registry task's
       `metadata["null_proportion"]`.
    2. `SYMMETRY_NULL_RATE` if this is a visual symmetry run.
    3. `DEFAULT_NULL_RATE`.

  This is only the fallback. An explicit `single_batch_null_prob`, a rate
  stated in the hypothesis text, and a rate stated in the seed instruction all
  take precedence over it.

  Args:
    config: Verification config, used to detect the task modality.

  Returns:
    The task-appropriate null proportion.
  """
  task_rate = getattr(config, "task_null_proportion", 0.0) or 0.0
  if 0.0 < task_rate < 1.0:
    return task_rate
  return SYMMETRY_NULL_RATE if _is_symmetry_task(config) else DEFAULT_NULL_RATE


def _get_symmetry_intro_prompt(
    config: verifier_config.VerificationConfig,
    q_count: int,
) -> str:
  """Returns the orientation prompt for visual symmetry verification."""
  return visual_symmetry_prompts.task_specification(
      target_model=config.target_model,
      q_count=q_count,
      null_rate=_default_null_rate(config),
      verifying=True,
  )


def _infer_symmetry_shape_type(h_text: str) -> str:
  """Infers the probe shape_type a hypothesis is about."""
  return visual_symmetry_prompts.infer_shape_type(h_text)


def _get_symmetry_execution_prompt(q_count: int, h_text: str) -> str:
  """Returns the Step 4 prompt instructing the agent to run image symmetry probes."""
  return visual_symmetry_prompts.probe_execution_instructions(
      q_count=q_count,
      h_text=h_text,
      verifying=True,
  )


@dataclasses.dataclass
class ConditionOutcome:
  """Scored results for one arm of a symmetry probe batch."""

  label: str
  shape_type: str
  bg_color: str
  n_scored: int
  n_success: int

  @property
  def rate(self) -> float:
    return self.n_success / self.n_scored if self.n_scored else 0.0

  def to_dict(self) -> dict[str, Any]:
    """Serializes the arm's scored results to a JSON-compatible dict."""
    return {
        "label": self.label,
        "shape_type": self.shape_type,
        "bg_color": self.bg_color,
        "n_scored": self.n_scored,
        "n_success": self.n_success,
        "rate": self.rate,
    }


def split_contrast_budget(q_count: int) -> tuple[int, int]:
  """Splits a probe budget evenly between two arms.

  Balanced allocation maximises the power of a two-proportion test at fixed
  total cost, and an odd budget gives the extra probe to arm A, which is the
  condition the hypothesis is actually about.

  Args:
    q_count: Total probes available for this hypothesis.

  Returns:
    `(n_a, n_b)`, summing to `q_count`.
  """
  n_a = (q_count + 1) // 2
  return n_a, q_count - n_a


def _n_usable_responses(
    agent: Any, responses: Sequence[Any] | None, is_symmetry: bool
) -> int:
  """Usable responses for the `--min_response_fraction` check.

  Text batches count the responses the scrape kept. A symmetry batch counts
  its scored completions (`_last_responses` of the last probe run), which
  exclude probes whose image never arrived after backfill: those are missing
  data, as an unanswered text prompt is.

  Args:
    agent: The batch's agent.
    responses: The text batch's recorded responses.
    is_symmetry: Whether the batch is a visual symmetry batch.

  Returns:
    Number of usable responses in the batch.
  """
  if is_symmetry:
    return len(getattr(agent, "last_responses", None) or [])
  return len(responses) if responses else 0


async def _run_symmetry_contrast_execution(
    agent: Any,
    plan: visual_symmetry_prompts.ContrastPlan,
    q_count: int,
    success_criterion: str,
    log_fn: Callable[..., None],
    label: str = "Precision",
) -> list[ConditionOutcome]:
  """Probes both arms of a contrastive hypothesis and scores them separately.

  Runs the tools directly rather than prompting the agent to. Three reasons,
  all of which have bitten:

    * The single-arm execution prompt instructs the agent to pick one
      `shape_type` and to "hold `bg_color` FIXED across the batch", so a
      contrast cannot be expressed through it at all.
    * `_run_symmetry_execution`'s correction fallback compares the agent's last
      probed family against one inferred family. On a legitimate two-arm run
      the last family is whichever arm ran second, so the check can fire and
      replace real two-arm data with a single-family batch.
    * `generate_and_send_symmetry_probes` OVERWRITES the stored completions, so
      arm A must be scored before arm B is generated or it is silently lost.

  Args:
    agent: The live `StandaloneAgent`.
    plan: A contrastive plan; `plan.condition_b` must not be None.
    q_count: Total probes to spend across both arms.
    success_criterion: Scoring criterion, applied identically to both arms.
    log_fn: Logging callable.
    label: Prefix used in log lines to identify the calling path.

  Returns:
    One `ConditionOutcome` per arm, in (A, B) order. An arm whose probes all
    failed comes back with `n_scored == 0`; callers must treat that as a scrape
    failure rather than as a measured zero.

  Raises:
    ValueError: If `plan` is not contrastive.
  """
  if plan.condition_b is None:
    raise ValueError("_run_symmetry_contrast_execution requires two arms.")

  n_a, n_b = split_contrast_budget(q_count)
  log_fn(
      f"\n[{label}] Contrastive symmetry execution on axis '{plan.axis}':"
      f" A={plan.condition_a.label} (n={n_a}) vs"
      f" B={plan.condition_b.label} (n={n_b}), criterion='{success_criterion}'"
  )

  outcomes: list[ConditionOutcome] = []
  for condition, n_probes in (
      (plan.condition_a, n_a),
      (plan.condition_b, n_b),
  ):
    # Clear every per-round accumulator before the arm runs. Without this a
    # failed generation would leave the previous arm's images in place and the
    # scorer would happily re-score them, reporting the two arms as identical.
    agent.reset_symmetry_state(per_probe=True)

    await agent.run_symmetry_probes(condition.probe_args(n_probes))
    await agent.score_symmetry({"success_criterion": success_criterion})

    # `last_scalar_values` holds one 0/1 entry per probe that actually returned
    # a completion. Unanswered probes are absent rather than zero, so this is
    # the count of measurements, not of requests.
    values = list(getattr(agent, "last_scalar_values", []) or [])
    outcome = ConditionOutcome(
        label=condition.label,
        shape_type=condition.shape_type,
        bg_color=condition.bg_color,
        n_scored=len(values),
        n_success=int(sum(values)),
    )
    outcomes.append(outcome)
    log_fn(
        f"  [{label}] Arm '{outcome.label}': {outcome.n_success}/"
        f"{outcome.n_scored} successes ({outcome.rate:.1%}); requested"
        f" {n_probes}"
    )

  return outcomes


async def _run_symmetry_execution(
    sender: Any,
    agent: Any,
    q_count: int,
    h_text: str,
    log_fn: Callable[..., None],
    label: str = "Precision",
) -> None:
  """Executes visual symmetry probes and algorithmic scoring for one round.

  Mirrors `core/task_executor.py`, including the deterministic fallback: if the
  agent declines to call the symmetry tools, or probes a shape that does not
  match the hypothesis under test, the correct probes are executed directly.
  Verification cannot fall back to a text scrape here, so an unresponsive agent
  must not be allowed to silently produce an empty round.

  Args:
    sender: Agent sender used to drive the tool-calling step.
    agent: The underlying StandaloneAgent, used for state reset and fallback.
    q_count: Number of image probes to execute.
    h_text: Hypothesis text under verification.
    log_fn: Logging callable.
    label: Prefix used in log lines to identify the calling path.
  """
  log_fn(f"\n[{label}] Visual Symmetry Probe Generation & Algorithmic Scoring")

  # Reset prior symmetry outputs so this round's scoring cannot pick up images
  # or judgments left behind by an earlier round.
  if agent:
    agent.reset_symmetry_state()

  prompt = _get_symmetry_execution_prompt(q_count, h_text)
  result = await sender.send_step(prompt)
  log_fn(f"\n  [{label}] [Symmetry Execution Summary] {result[:2000]}")

  if not agent:
    return

  expected_shape = _infer_symmetry_shape_type(h_text)
  last_shape = agent.last_symmetry_shape_type
  combined_text = h_text.lower()

  # See the equivalent comment in `core/task_executor.py`: a
  # contrastive hypothesis names two families and the agent is right to probe
  # both, but only the last one is recorded. Comparing against a single
  # inferred family turned a correct two-arm run into a "correction".
  contrast = visual_symmetry_prompts.infer_contrast(h_text)
  acceptable_shapes = {c.shape_type for c in contrast.conditions}
  acceptable_shapes.add(expected_shape)

  needs_execution = not agent.symmetry_output_images or not agent.last_judgments
  needs_correction = (
      bool(agent.symmetry_output_images)
      and last_shape not in acceptable_shapes
      and (
          expected_shape in ("chinese_char", "greek_letter")
          or "chinese" in combined_text
          or "greek" in combined_text
      )
  )

  if needs_execution or needs_correction:
    action_name = "Correction" if needs_correction else "Fallback"
    reason = (
        f"Agent probed {last_shape} but hypothesis requires {expected_shape}"
        if needs_correction
        else "Agent did not execute symmetry tools"
    )
    log_fn(
        f"\n  [{label}] [Symmetry {action_name}] {reason};"
        f" executing probes: shape_type={expected_shape}, n_probes={q_count}"
    )
    await agent.run_symmetry_probes({
        "shape_type": expected_shape,
        "n_probes": q_count,
    })
    await agent.score_symmetry({})


def _create_isolated_sender(
    config: verifier_config.VerificationConfig,
    log_fn: Callable[..., None],
    prompts_mod: Any,
    output_dir: str,
    scrape_round_offset: int,
) -> agent_senders.BaseSender:
  """Creates a fresh, isolated agent and sender for query generation."""
  return agent_factory_lib.build_isolated_sender(
      config,
      log_fn,
      prompts_mod,
      output_dir,
      scrape_round_offset,
  )


async def _generate_queries_isolated(
    config: verifier_config.VerificationConfig,
    prompts_mod: Any,
    h_text: str,
    q_count: int,
    executed_queries: set[str],
    log_fn: Callable[..., None],
    paired: bool | None = None,
) -> list[str]:
  """Generates queries using a fresh, context-free subagent."""
  if paired is None:
    is_paired = config.is_paired
  else:
    is_paired = paired
  if executed_queries:
    excluded_queries = random.sample(
        list(executed_queries),
        min(200, len(executed_queries)),
    )
  else:
    excluded_queries = None
  iso_prompt = prompts_mod.get_isolated_query_generation_prompt(
      seed_instruction=config.seed_instruction,
      hypothesis=h_text,
      num_queries=q_count,
      excluded_queries=excluded_queries,
      procedural_autorater=config.procedural_autorater,
      paired=is_paired,
  )
  backbone = backbone_lib.get_backbone_client(config.agent_model)
  iso_text = ""
  max_retries = 3
  base_delay = 2.0
  max_delay = 15.0
  for attempt in range(max_retries):
    try:
      iso_text = await backbone.generate(
          model=config.agent_model,
          prompt=iso_prompt,
          temperature=getattr(config, "isolated_query_gen_temperature", 1.0),
      )
      break
    except backbone_lib.LLM_CALL_ERRORS as e:
      if attempt < max_retries - 1:
        exp_delay = base_delay * (2**attempt)
        jitter = random.uniform(0.8, 1.2)
        delay = min(exp_delay * jitter, max_delay)
        log_fn(
            f"  [Query Gen Retry] Attempt {attempt + 1}/{max_retries}"
            f" failed: {e}. Retrying in {delay:.1f}s..."
        )
        await asyncio.sleep(delay)
      else:
        log_fn(f"  [Query Gen Error] All {max_retries} attempts failed: {e}")
        iso_text = ""

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
  queries: list[str] = _parse_isolated_text(iso_text, is_paired)

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
        f" ({len(queries)}/{target_total}). Retrying {retries}/{max_retries}..."
    )
    combined_excluded = set(executed_queries) | set(queries)
    if combined_excluded:
      excluded_retry = random.sample(
          list(combined_excluded),
          min(200, len(combined_excluded)),
      )
    else:
      excluded_retry = None
    iso_prompt_retry = prompts_mod.get_isolated_query_generation_prompt(
        seed_instruction=config.seed_instruction,
        hypothesis=h_text,
        num_queries=needed,
        excluded_queries=excluded_retry,
        procedural_autorater=config.procedural_autorater,
        paired=is_paired,
    )
    retry_text = ""
    for attempt in range(max_retries):
      try:
        retry_text = await backbone.generate(
            model=config.agent_model,
            prompt=iso_prompt_retry,
            temperature=getattr(config, "isolated_query_gen_temperature", 1.0),
        )
        break
      except backbone_lib.LLM_CALL_ERRORS as e:
        if attempt < max_retries - 1:
          exp_delay = base_delay * (2**attempt)
          jitter = random.uniform(0.8, 1.2)
          delay = min(exp_delay * jitter, max_delay)
          log_fn(
              "  [Query Gen Retry] Replenish attempt"
              f" {attempt + 1}/{max_retries} failed: {e}. Retrying in"
              f" {delay:.1f}s..."
          )
          await asyncio.sleep(delay)
        else:
          log_fn(
              f"  [Query Gen Error] All {max_retries} replenish attempts"
              f" failed: {e}"
          )
          retry_text = ""
    retry_parsed = _parse_isolated_text(retry_text, is_paired)
    if retry_parsed:
      queries.extend(retry_parsed)

  if is_paired:
    if len(queries) > target_total:
      queries = queries[:target_total]
    elif len(queries) % 2 != 0:
      queries = queries[:-1]
  else:
    if len(queries) > target_total:
      queries = queries[:target_total]

  return queries


def _hypothesis_rng(seed: int | None, h_id: str) -> np.random.Generator:
  """Returns the p-value RNG of one hypothesis.

  With a seed, each hypothesis gets its own stream derived from the seed and
  its id, so the stream does not depend on which hypotheses run concurrently.

  Args:
    seed: The run's `random_seed`, or None to seed from OS entropy.
    h_id: Id of the hypothesis.
  """
  if seed is None:
    return np.random.default_rng()
  return np.random.default_rng(
      np.random.SeedSequence([seed, zlib.crc32(h_id.encode("utf-8"))])
  )


@dataclasses.dataclass
class _SingleBatchSetup:
  """A hypothesis that has pre-registered but not yet spent any queries.

  Exists so budget allocation can sit at a barrier between the two halves of a
  single-batch check. Allocation needs every hypothesis's pairing decision, and
  that decision is only known after pre-registration, but pre-registration is a
  turn in the agent's own conversation -- so the live agent and sender have to
  survive the wait rather than being rebuilt afterwards.
  """

  h_id: str
  h_text: str
  agent: Any
  sender: Any
  parsers: Any
  prompts_mod: Any
  output_dir: str
  log_fn: Callable[..., None]
  test_plan: stats_plan.PreRegisteredTestPlan | None
  is_paired: bool
  base_q_count: int
  null_rate: float
  alpha: float
  block_reasons: list[str]
  # Set for visual symmetry runs. `contrast.is_contrastive` decides whether the
  # hypothesis is measured against the task's reference rate or against a
  # second probe condition.
  contrast: visual_symmetry_prompts.ContrastPlan | None = None
  # Criterion the probes are scored under. Only consulted on the contrastive
  # path, where the verifier drives the scoring tool instead of the agent.
  success_criterion: str = "both"

  @property
  def is_two_sample(self) -> bool:
    """True when the hypothesis compares two probe conditions to each other."""
    return self.contrast is not None and self.contrast.is_contrastive

  @property
  def needs_double_budget(self) -> bool:
    """True when one observation costs two target queries, or two arms do.

    Paired and two-sample hypotheses are distinct designs -- a pair is one
    observation measured twice, two arms are two independent samples -- but
    they have the same budget consequence: at single weight each would end up
    with half the observations per condition of an absolute claim.
    """
    return self.is_paired or self.is_two_sample

  @property
  def budget_weight(self) -> int:
    """Share of a total budget this hypothesis should receive.

    A paired hypothesis spends two queries per observation, so at equal weight
    it would get half the effective sample size of an unpaired one. Weight 2
    buys it the same number of observations, at twice the query cost. A
    two-sample hypothesis splits its allocation across two arms, so weight 2
    likewise buys each arm the sample size an absolute claim gets.
    """
    return 2 if self.needs_double_budget else 1

  @property
  def standalone_q_count(self) -> int:
    """Query count when no total budget is set: the flat per-hypothesis size."""
    return (
        self.base_q_count * 2 if self.needs_double_budget else self.base_q_count
    )


def allocate_verification_budget(
    weights: Sequence[int], total_budget: int
) -> list[int]:
  """Splits a query budget across hypotheses in proportion to their weights.

  Uses the largest-remainder method so the parts sum to exactly `total_budget`
  rather than drifting below it through repeated flooring. Remainders are
  broken by hypothesis order, which keeps the split reproducible across runs.

  Every hypothesis is guaranteed at least one query, since a hypothesis with a
  zero allocation would be reported as an execution error rather than a
  measurement and would quietly bias the precision denominator.

  Args:
    weights: Per-hypothesis weight, in hypothesis order.
    total_budget: Total queries to distribute.

  Returns:
    Per-hypothesis query counts, in the same order as `weights`.
  """
  if not weights:
    return []
  total_weight = sum(weights)
  if total_weight <= 0:
    raise ValueError(f"Budget weights must be positive, got {weights!r}.")

  exact = [total_budget * w / total_weight for w in weights]
  allocation = [max(1, int(x)) for x in exact]

  # Hand out whatever flooring left over, largest fractional part first.
  remainder = total_budget - sum(allocation)
  if remainder > 0:
    order = sorted(range(len(weights)), key=lambda i: (-(exact[i] % 1.0), i))
    for k in range(remainder):
      allocation[order[k % len(order)]] += 1

  return allocation


async def _setup_and_preregister_single_batch(
    h_id: str,
    h_text: str,
    config: verifier_config.VerificationConfig,
    log_fn: Callable[..., None],
    stored_plan: stats_plan.PreRegisteredTestPlan | None = None,
) -> _SingleBatchSetup:
  """Builds the agent for one hypothesis and elicits its test plan.

  Stops short of generating any query, so a caller holding several of
  these can see how many hypotheses went paired before deciding how to
  divide a shared budget between them.

  Args:
    h_id: Hypothesis identifier.
    h_text: Hypothesis text.
    config: Verification config.
    log_fn: Logging function, already tagged with the hypothesis id.
    stored_plan: A plan pre-registered by an earlier check of this hypothesis.
      When given, it is used as is and the agent is not asked again. Ignored for
      symmetry tasks, which have no agent-chosen test.

  Returns:
    The live agent state plus the pre-registered plan.
  """
  log_fn(
      f"\n[Single-Batch Precision Check] Starting for Hypothesis [{h_id}]:"
      f" {h_text}"
  )

  block_reasons: list[str] = []
  parsers = config.build_parsers()
  prompts_mod = config.get_prompts_module()

  is_paired = config.is_paired
  base_q_count = getattr(config, "single_batch_num_queries", 100)

  # Resolve the probe condition(s) before anything is sized or sent. A
  # contrastive hypothesis ("A fails more than B") is two half-batches plus a
  # two-sample test, not one batch against the reference rate, and that changes
  # both the budget and the prompts.
  contrast: visual_symmetry_prompts.ContrastPlan | None = None
  success_criterion = "both"
  if _is_symmetry_task(config):
    contrast = visual_symmetry_prompts.infer_contrast(h_text)
    success_criterion = visual_symmetry_prompts.infer_success_criterion(h_text)
  is_two_sample = contrast is not None and contrast.is_contrastive

  # Provisional only: it sizes the intro prompt, which is sent before the
  # agent has pre-registered and therefore before pairing is finally known.
  q_count = base_q_count * 2 if (is_paired or is_two_sample) else base_q_count
  alpha = getattr(config, "single_batch_alpha", 0.05)

  config_null = getattr(config, "single_batch_null_prob", None)
  if config_null is not None and config_null > 0.0:
    null_rate = config_null
  else:
    null_rate = stats_utils.extract_null_proportion_from_hypothesis(
        h_text, config.seed_instruction, default=_default_null_rate(config)
    )

  log_fn(
      f"[Single-Batch] Provisional queries: {q_count} (paired={is_paired}),"
      f" Null rate: {null_rate:.4f}, Alpha: {alpha}"
  )

  safe_h_id = re.sub(r"[^\w-]", "_", h_id)[:40]
  output_dir = os.path.join(
      config.output_dir,
      f"{config.experiment_name}_precision",
      safe_h_id,
  )
  is_symmetry = _is_symmetry_task(config)
  # Symmetry completions are scored algorithmically by IoU, so the factory
  # blocks the LLM judge tool to keep it out of the success counts.
  agent, sender = agent_factory_lib.build_agent_and_sender(
      config,
      log_fn,
      prompts_mod,
      output_dir=output_dir,
      initial_scrape_round=0,
  )

  # Orient agent: intro prompt
  if is_symmetry:
    intro_prompt = _get_symmetry_intro_prompt(config, q_count)
  else:
    intro_prompt = prompts_mod.get_fresh_agent_intro_prompt(
        config.target_model,
        config.seed_instruction,
        threshold=1.0 / alpha,
        chunk_size=q_count,
    )
    if config.procedural_autorater:
      intro_prompt += prompts_mod.get_procedural_autorater_addendum(
          config.procedural_autorater,
          allow_continuous_means=config.allow_continuous_means_test,
      )
  await sender.send_step(intro_prompt, enable_tools=False)

  # Context prompt
  context_prompt = prompts_mod.get_precision_round_context_prompt(
      h_text,
      prior_e_values=[],
      failures=None,
      failure_reflections=None,
  )
  await sender.send_step(context_prompt, enable_tools=False)

  # Step 2b: Statistical test pre-registration.
  #
  # Symmetry tasks have no agent-chosen test: completions are scored
  # algorithmically by IoU against a fixed threshold, and probes are
  # independent (never paired). Eliciting a plan here would spend budget to
  # obtain a metric that is ignored, and a plan claiming `implies_paired`
  # would silently halve the effective sample size. `core/round_executor.py`
  # skips pre-registration for the same reason.
  test_plan: stats_plan.PreRegisteredTestPlan | None = None
  if is_symmetry:
    is_paired = False
    log_fn(
        "\n[Single-Batch] Step 2b skipped: visual symmetry uses a fixed"
        " algorithmic (IoU) test on unpaired probes."
    )
  else:
    if stored_plan is not None:
      log_fn(
          "\n[Single-Batch] Step 2b skipped: reusing the plan pre-registered"
          " by the original check."
      )
      test_plan = stored_plan
    else:
      log_fn("\n[Single-Batch] Step 2b: Statistical test pre-registration")
      prereg_prompt = prompts_mod.get_test_preregistration_prompt(h_text)
      test_plan = await sender.send_and_extract(
          prereg_prompt,
          parsers_lib.parse_and_validate_preregistration,
          max_retries=2,
          fallback_value=None,
          log_fn=log_fn,
          retry_prompt_fn=prompts_mod.get_default_preregistration_retry_prompt,
          parsers=parsers,
          enable_tools=False,
      )
    if test_plan:
      log_fn(
          "  [Single-Batch] Pre-registered plan:"
          f" test_type='{test_plan.test_type}',"
          f" metric='{test_plan.target_metric}',"
          f" null={test_plan.null_baseline},"
          f" alternative='{test_plan.alternative}'"
      )
      plan_paired = test_plan.implies_paired
      if plan_paired and not is_paired:
        is_paired = True
        log_fn(
            "  [Single-Batch] Pre-registered plan requires a paired test;"
            " this hypothesis will be collected paired."
        )
      if (
          config_null is None
          and test_plan.null_baseline is not None
          and test_plan.null_baseline > 0.0
      ):
        null_rate = test_plan.null_baseline
    else:
      log_fn(
          "  [Single-Batch] Pre-registration extraction failed; proceeding"
          " without pre-registered plan."
      )

  return _SingleBatchSetup(
      h_id=h_id,
      h_text=h_text,
      agent=agent,
      sender=sender,
      parsers=parsers,
      prompts_mod=prompts_mod,
      output_dir=output_dir,
      log_fn=log_fn,
      test_plan=test_plan,
      is_paired=is_paired,
      base_q_count=base_q_count,
      null_rate=null_rate,
      alpha=alpha,
      block_reasons=block_reasons,
      contrast=contrast,
      success_criterion=success_criterion,
  )


async def _execute_single_batch(
    setup: _SingleBatchSetup,
    q_count: int,
    config: verifier_config.VerificationConfig,
) -> HypothesisPrecisionRecord:
  """Spends `q_count` queries on an already pre-registered hypothesis.

  Args:
    setup: Live agent state from `_setup_and_preregister_single_batch`.
    q_count: Queries to spend. For a paired hypothesis this is the response
      count, i.e. twice the number of observations.
    config: Verification config.

  Returns:
    The precision record for this hypothesis, with its request/receipt totals
    and failure reason filled in.
  """
  record = await _execute_single_batch_unannotated(setup, q_count, config)
  record.test_plan = setup.test_plan
  record.allocated_queries = q_count
  return _annotate_coverage(record, [setup.agent])


async def _execute_single_batch_unannotated(
    setup: _SingleBatchSetup,
    q_count: int,
    config: verifier_config.VerificationConfig,
) -> HypothesisPrecisionRecord:
  """Implements `_execute_single_batch` (records are not yet annotated)."""
  h_id = setup.h_id
  h_text = setup.h_text
  log_fn = setup.log_fn
  agent = setup.agent
  sender = setup.sender
  parsers = setup.parsers
  prompts_mod = setup.prompts_mod
  output_dir = setup.output_dir
  test_plan = setup.test_plan
  is_paired = setup.is_paired
  null_rate = setup.null_rate
  alpha = setup.alpha
  block_reasons = setup.block_reasons

  is_symmetry = _is_symmetry_task(config)

  # Query Generation. Queries always come from a fresh, context-free
  # generator (standard or taxonomy); the batch's agent never writes its own.
  injected_queries: list[str] | None = None
  if is_symmetry:
    # Symmetry probes are half-erased images synthesized by the probe tool,
    # not text prompts, so there is nothing to generate here. Skipping also
    # keeps `q_count` exactly equal to the allocated budget: the text paths
    # below overwrite it with the number of queries the generator happened to
    # return, which would silently resize a pre-registered batch.
    log_fn(
        f"\n[Single-Batch] Query generation skipped: {q_count} visual symmetry"
        " probes are generated by the probe tool."
    )
  else:
    log_fn(
        f"\n[Single-Batch] Query generation ({q_count} queries,"
        f" paired={is_paired})"
    )
    if getattr(config, "query_generation_mode", "standard") == "taxonomy":
      isolated_sender = _create_isolated_sender(
          config, log_fn, prompts_mod, output_dir, 0
      )
      q_count, injected_queries = (
          await taxonomy_utils.run_taxonomy_query_generation(
              cast(config_lib.DiscoConfig, config),
              isolated_sender,
              prompts_mod,
              q_count,
              h_text,
              log_fn,
              target_num_queries=q_count,
              paired=is_paired,
          )
      )
      if q_count == 0 or not injected_queries:
        injected_queries = await _generate_queries_isolated(
            config=config,
            prompts_mod=prompts_mod,
            h_text=h_text,
            q_count=q_count,
            executed_queries=set(),
            log_fn=log_fn,
            paired=is_paired,
        )
        q_count = len(injected_queries)
    else:
      injected_queries = await _generate_queries_isolated(
          config=config,
          prompts_mod=prompts_mod,
          h_text=h_text,
          q_count=q_count,
          executed_queries=set(),
          log_fn=log_fn,
          paired=is_paired,
      )
      q_count = len(injected_queries)

  # Execution
  query_preamble = ""
  if injected_queries:
    formatted = "\n".join(f"{i+1}. {q}" for i, q in enumerate(injected_queries))
    query_preamble = (
        f"The following {len(injected_queries)} queries have been"
        " pre-generated for you. Use EXACTLY these queries:\n\n"
        f"{formatted}\n\n"
    )

  contrast_outcomes: list[ConditionOutcome] | None = None
  if is_symmetry and setup.is_two_sample:
    assert setup.contrast is not None
    n_a, n_b = split_contrast_budget(q_count)
    await sender.send_step(
        visual_symmetry_prompts.contrastive_execution_briefing(
            h_text=h_text, plan=setup.contrast, n_a=n_a, n_b=n_b
        ),
        enable_tools=False,
    )
    contrast_outcomes = await _run_symmetry_contrast_execution(
        agent=agent,
        plan=setup.contrast,
        q_count=q_count,
        success_criterion=setup.success_criterion,
        log_fn=log_fn,
        label="Single-Batch",
    )
    # Both arms are load-bearing. An empty arm is missing data, not a measured
    # zero, and scoring the comparison anyway would report a maximal effect
    # size wherever the target happened to be throttled.
    empty = [o.label for o in contrast_outcomes if o.n_scored == 0]
    if empty:
      log_fn(
          "\n[Single-Batch] [Symmetry Error] Contrastive arms returned no"
          f" usable completions: {empty}"
      )
      reason = (
          agent.last_block_reason
          or f"SCRAPE_FAILED: empty contrast arm(s) {empty}"
      )
      if reason not in block_reasons:
        block_reasons.append(reason)
      return HypothesisPrecisionRecord(
          hypothesis_id=h_id,
          hypothesis_text=h_text,
          verified=False,
          status="SCRAPE_FAILED",
          block_reasons=block_reasons,
          null_rate=null_rate,
      )
    # As for a paired text batch, where both halves of a pair must survive:
    # each arm must reach --min_response_fraction of its own allocation.
    short = scoring.short_arms_reason(
        [
            (outcome.label, outcome.n_scored, n_requested)
            for outcome, n_requested in zip(contrast_outcomes, (n_a, n_b))
        ],
        config.min_response_fraction,
    )
    if short:
      log_fn(f"\n[Single-Batch] [Insufficient Responses] {short}")
      block_reasons.append(short)
      return HypothesisPrecisionRecord(
          hypothesis_id=h_id,
          hypothesis_text=h_text,
          verified=False,
          status=scoring.INSUFFICIENT_RESPONSES_STATUS,
          block_reasons=block_reasons,
          null_rate=null_rate,
          failure_reason=short,
      )
  elif is_symmetry:
    # Visual symmetry probes are images, not text prompts; the agent drives
    # `generate_and_send_symmetry_probes` + `score_symmetry_completions`,
    # which populate the same `_last_responses` / `last_judgments` state the
    # text path produces, so everything downstream is unchanged.
    await _run_symmetry_execution(
        sender=sender,
        agent=agent,
        q_count=q_count,
        h_text=h_text,
        log_fn=log_fn,
        label="Single-Batch",
    )
    if not (agent.last_scrape_status == "OK" and agent.last_responses):
      log_fn(
          "\n[Single-Batch] [Symmetry Error] Symmetry probing returned no"
          " usable completions."
      )
      reason = (
          agent.last_block_reason
          or "SCRAPE_FAILED: No valid symmetry completions returned"
      )
      if reason not in block_reasons:
        block_reasons.append(reason)
      return HypothesisPrecisionRecord(
          hypothesis_id=h_id,
          hypothesis_text=h_text,
          verified=False,
          status="SCRAPE_FAILED",
          block_reasons=block_reasons,
          null_rate=null_rate,
      )
  else:
    if config.procedural_autorater:
      log_fn("\n[Single-Batch] Scrape and Measure execution")
      base_scrape_prompt = prompts_mod.get_procedural_scrape_and_measure_prompt(
          q_count, config.procedural_autorater
      )
    else:
      log_fn("\n[Single-Batch] Scrape and Judge execution")
      base_scrape_prompt = prompts_mod.get_scrape_and_judge_execution_prompt(
          q_count
      )
    base_scrape_prompt = query_preamble + base_scrape_prompt

    max_scrape_attempts = 3
    base_delay = 3.0
    for scrape_attempt in range(max_scrape_attempts):
      if scrape_attempt > 0:
        delay = (base_delay * (2 ** (scrape_attempt - 1))) * random.uniform(
            0.8, 1.5
        )
        log_fn(
            "\n[Single-Batch] [Scrape Retry] Attempt"
            f" {scrape_attempt + 1}/{max_scrape_attempts}. Waiting {delay:.2f}s"
            " before re-prompting..."
        )
        await asyncio.sleep(delay)
        retry_prompt = (
            "CRITICAL ERROR: You did not execute the"
            " 'generate_single_turn_responses' tool. You MUST call"
            f" 'generate_single_turn_responses' with the {q_count} prompts"
            " provided below.\n\n"
            + base_scrape_prompt
        )
      else:
        retry_prompt = base_scrape_prompt

      result = await sender.send_step(retry_prompt)
      log_fn(
          "\n  [Single-Batch] [Execution Summary (Attempt"
          f" {scrape_attempt + 1})] {result[:2000]}"
      )

      if agent.last_scrape_status == "OK" and agent.last_responses:
        break
    else:
      log_fn(
          "\n[Single-Batch] [Scrape Error] Target scraping failed after"
          " retries."
      )
      if (
          agent.last_block_reason
          and agent.last_block_reason not in block_reasons
      ):
        block_reasons.append(agent.last_block_reason)
      return HypothesisPrecisionRecord(
          hypothesis_id=h_id,
          hypothesis_text=h_text,
          verified=False,
          status="SCRAPE_FAILED",
          block_reasons=block_reasons,
          null_rate=null_rate,
      )

  if agent.last_block_reason and agent.last_block_reason not in block_reasons:
    block_reasons.append(agent.last_block_reason)

  # Statistical significance testing
  log_fn("\n[Single-Batch] Statistical significance testing")
  res: stats_plan.StatsResult | None = None
  p_value = 1.0
  resp_list = (
      agent.all_responses
      if getattr(agent, "all_responses", None)
      else agent.last_responses
  )
  n_obs = len(resp_list) if resp_list else q_count
  n_succ = 0
  obs_rate = 0.0
  ci_lower = 0.0
  ci_upper = 1.0

  # Paired bookkeeping. Under pairing the unit of analysis is the pair, so the
  # only defensible sample size is the number of pairs whose two arms both
  # survived. Everything below cross-checks against this rather than trusting
  # a response-level count, which would double the effective sample size.
  response_indices = getattr(agent, "last_response_indices", None)
  n_pairs_avail = (
      stats_utils.count_complete_pairs(response_indices) if is_paired else 0
  )

  # When pairing is forced by config the data really is paired, even if the
  # agent's plan said otherwise, so tie policy must be derived as if paired.
  effective_plan = test_plan
  if test_plan is not None and is_paired and not test_plan.is_paired:
    effective_plan = dataclasses.replace(test_plan, is_paired=True)
    log_fn(
        "  [Single-Batch] WARNING: queries were generated paired but the"
        " pre-registered plan declared IS_PAIRED=False (test"
        f" '{test_plan.test_type}'). Treating the analysis as paired."
    )
  tie_policy = (
      effective_plan.tie_policy
      if effective_plan is not None
      else stats_plan.TiePolicy.NOT_APPLICABLE
  )

  diagnostics: PairedDiagnostics | None = None
  if is_paired:
    diagnostics = PairedDiagnostics(
        n_pairs=n_pairs_avail,
        tie_policy=tie_policy.value,
        unit_of_analysis="pair",
    )
    n_responses = len(response_indices) if response_indices else 0
    log_fn(
        f"[Single-Batch] Paired: {n_pairs_avail} complete pairs from"
        f" {n_responses} responses; tie policy '{tie_policy.value}'"
    )

  # A batch that came back well short of what was pre-registered is not
  # scored: whatever made the target drop responses may not be independent of
  # the responses themselves. A symmetry batch counts its scored completions
  # (probes whose image never arrived are excluded); a contrastive symmetry
  # batch was checked arm by arm above.
  if contrast_outcomes is None:
    short = scoring.short_batch_reason(
        is_paired=is_paired and not is_symmetry,
        q_count=q_count,
        n_responses=_n_usable_responses(agent, resp_list, is_symmetry),
        n_pairs=n_pairs_avail,
        min_fraction=config.min_response_fraction,
    )
    if short:
      log_fn(f"\n[Single-Batch] [Insufficient Responses] {short}")
      return HypothesisPrecisionRecord(
          hypothesis_id=h_id,
          hypothesis_text=h_text,
          verified=False,
          status=scoring.INSUFFICIENT_RESPONSES_STATUS,
          block_reasons=block_reasons,
          null_rate=null_rate,
          paired_diagnostics=diagnostics,
          failure_reason=short,
      )

  # Set when the executed test contradicts the pre-registered plan. Such a
  # result is uninterpretable, so it fails the hypothesis rather than being
  # scored -- reporting a p-value nobody can attach a meaning to is worse than
  # reporting nothing.
  consistency_error: str | None = None

  # Two-sample hypotheses take a different prompt, and the counts are supplied
  # rather than elicited: they were measured algorithmically, arm by arm, and
  # asking the agent to recover them from the transcript is a chance to get
  # them wrong that buys nothing.
  if contrast_outcomes is not None:
    assert setup.contrast is not None
    outcome_a, outcome_b = contrast_outcomes
    n_obs = outcome_a.n_scored + outcome_b.n_scored
    obs_rate = outcome_a.rate
    n_succ = outcome_a.n_success
    stats_prompt = visual_symmetry_prompts.contrastive_stats_prompt(
        h_text=h_text,
        plan=setup.contrast,
        n_success_a=outcome_a.n_success,
        n_a=outcome_a.n_scored,
        n_success_b=outcome_b.n_success,
        n_b=outcome_b.n_scored,
    )
  else:
    stats_prompt = prompts_mod.get_stats_significance_prompt(
        plan=effective_plan
    )

  res = await sender.send_and_extract(
      stats_prompt,
      parsers_lib.parse_and_validate_stats,
      fallback_value=stats_plan.StatsResult(test_type="Unknown", p_value=-1.0),
      log_fn=log_fn,
      parsers=parsers,
  )
  if res is not None and res.p_value != -1.0:
    if contrast_outcomes is not None:
      # A two-sample test has no null PROPORTION -- its null is "the two arms
      # are equal". Comparing whatever the tool reported against the task's
      # 0.90 reference rate would flag every correct contrastive test as
      # inconsistent. What must be checked instead is that the agent ran the
      # test on the arm counts it was handed, rather than on numbers it
      # reconstructed from the transcript.
      outcome_a, outcome_b = contrast_outcomes
      if res.group_b_obs is None and res.group_b_mean is None:
        # No second group anywhere in the result means the agent ran a
        # one-sample test despite being handed two arms. That p-value is a
        # comparison against the absolute reference rate, which is exactly the
        # answer this whole path exists to stop reporting.
        consistency_error = (
            f"hypothesis is contrastive ({outcome_a.label} vs"
            f" {outcome_b.label}) but the executed test"
            f" '{res.test_type}' reported only one group"
        )
      elif (
          res.group_a_obs is not None and res.group_a_obs != outcome_a.n_scored
      ):
        consistency_error = (
            f"two-sample analysis reported {res.group_a_obs} observations in"
            f" arm A but {outcome_a.n_scored} were scored"
        )
      elif (
          res.group_b_obs is not None and res.group_b_obs != outcome_b.n_scored
      ):
        consistency_error = (
            f"two-sample analysis reported {res.group_b_obs} observations in"
            f" arm B but {outcome_b.n_scored} were scored"
        )
    elif (
        res.null_proportion is not None
        and abs(res.null_proportion - null_rate) > 1e-9
    ):
      # The null the agent tested against must be the null we score against,
      # otherwise the p-value answers a different question than the one this
      # hypothesis was registered to ask.
      consistency_error = (
          f"executed test used null {res.null_proportion} but the verifier"
          f" scores against {null_rate}"
      )

    # In paired mode the reported observation count must be the pair count.
    # A count equal to the response total is the classic failure: it doubles
    # the effective sample size and pools the two arms together. Under
    # TiePolicy.DROP, tied pairs are dropped from the analysis, so the
    # reported count is the number of discordant pairs (<= n_pairs_avail).
    if is_paired and n_pairs_avail > 0 and res.group_a_obs is not None:
      if tie_policy is stats_plan.TiePolicy.DROP:
        if res.group_a_obs > n_pairs_avail:
          consistency_error = (
              f"paired analysis reported {res.group_a_obs} observations but"
              f" only {n_pairs_avail} complete pairs were collected"
          )
        elif res.group_a_obs <= 0:
          consistency_error = (
              f"paired analysis reported {res.group_a_obs} observations;"
              " at least one discordant pair is required to evaluate"
              " significance"
          )
        elif diagnostics is not None:
          diagnostics.n_tied = n_pairs_avail - res.group_a_obs
      elif res.group_a_obs != n_pairs_avail:
        consistency_error = (
            f"paired analysis reported {res.group_a_obs} observations but only"
            f" {n_pairs_avail} complete pairs were collected"
        )

    # `null_rate` is a rate and ignores a plan null of 0.0 (see
    # `_setup_and_preregister_single_batch`). A continuous test's null is on
    # the metric's scale, so the guard uses the plan null for those.
    if (
        test_plan is not None
        and test_plan.null_baseline is not None
        and stats_plan.is_continuous_test(res.test_type)
    ):
      guard_null = test_plan.null_baseline
    else:
      guard_null = null_rate
    p_value = stats_plan.guard_p_value_direction(
        res,
        log_fn,
        null_value=guard_null,
        use_direction_guard=getattr(config, "use_direction_guard", True),
    )
    # Decide continuous-vs-rate from the tool the test routes to.
    is_continuous = stats_plan.is_continuous_test(res.test_type)
    if is_continuous:
      obs_rate = res.group_a_mean if res.group_a_mean is not None else 0.0
      if res.group_a_obs is not None and res.group_a_obs > 0:
        n_obs = res.group_a_obs
      elif resp_list:
        n_obs = len(resp_list)
      else:
        n_obs = q_count
      n_succ = 0
      alt = _resolve_alternative(test_plan, res)
      ci_lower, ci_upper = stats_utils.compute_continuous_means_ci(
          sample_mean=obs_rate,
          sample_std=getattr(res, "sample_std", 0.0) or 0.0,
          n_obs=n_obs,
          alpha=alpha,
          alternative=alt,
      )
    elif contrast_outcomes is not None:
      # Two-sample contrastive test: `p_value` is already the direction-guarded
      # two-sample Fisher exact p-value from `res`, and `n_obs` is the combined
      # two-arm sample size (`n_a + n_b`, e.g. 200). Running the one-sample
      # `compute_binomial_replication_stats` branch below would overwrite BOTH:
      # it replaces `n_obs` with `group_a_obs` (100) and replaces the two-sample
      # p-value with a one-sample binomial p-value of arm A against the 0.90
      # absolute null. Only compute the arm-A confidence interval here.
      outcome_a, outcome_b = contrast_outcomes
      n_obs = outcome_a.n_scored + outcome_b.n_scored
      n_succ = outcome_a.n_success
      obs_rate = outcome_a.rate
      if outcome_a.n_scored > 0:
        alt = _resolve_alternative(test_plan, res)
        _, _, ci_lower, ci_upper = (
            stats_utils.compute_binomial_replication_stats(
                outcome_a.n_success,
                outcome_a.n_scored,
                null_prob=null_rate,
                alternative=alt,
                alpha=alpha,
            )
        )
    elif res.group_b_obs is not None and res.group_b_obs > 0:
      # Generic two-sample proportion test: preserve the two-sample p-value and
      # total sample size across both groups rather than overwriting with a
      # one-sample binomial test against `null_rate`.
      n_a_obs = (
          res.group_a_obs if (res.group_a_obs and res.group_a_obs > 0) else 0
      )
      n_obs = n_a_obs + res.group_b_obs
      if res.group_a_mean is not None and n_a_obs > 0:
        obs_rate = res.group_a_mean
        n_succ = round(obs_rate * n_a_obs)
        alt = _resolve_alternative(test_plan, res)
        _, _, ci_lower, ci_upper = (
            stats_utils.compute_binomial_replication_stats(
                n_succ,
                n_a_obs,
                null_prob=null_rate,
                alternative=alt,
                alpha=alpha,
            )
        )
    else:
      if res.group_a_obs is not None and res.group_a_obs > 0:
        n_obs = res.group_a_obs
        if res.group_a_mean is not None:
          obs_rate = res.group_a_mean
          n_succ = round(obs_rate * n_obs)
          alt = _resolve_alternative(test_plan, res)
          p_value, obs_rate, ci_lower, ci_upper = (
              stats_utils.compute_binomial_replication_stats(
                  n_succ,
                  n_obs,
                  null_prob=null_rate,
                  alternative=alt,
                  alpha=alpha,
              )
          )
    if getattr(config, "p_value_mode", "randomized") == "randomized":
      rng = _hypothesis_rng(
          getattr(config, "random_seed", None), h_id or h_text
      )
      p_value = stats_utils.maybe_randomize_p_value(p_value, res, rng)

  if res is None or res.p_value == -1.0:
    log_fn(
        "\n[Single-Batch] Stats parsing failed, falling back to direct count if"
        " available."
    )
    alt = _resolve_alternative(test_plan, res)
    is_cont_plan = stats_plan.is_continuous_test(
        test_plan.test_type if test_plan else None
    )

    if contrast_outcomes is not None:
      consistency_error = (
          "contrastive stats extraction failed; refusing to fall back to a"
          " one-sample binomial test against the absolute reference rate"
      )
    elif is_paired:
      # A paired batch is 2N responses covering N items. Counting those as 2N
      # independent draws -- which is what the unpaired branches below do --
      # both doubles the sample size and pools the two arms into one rate, so
      # reduce to pairs first and refuse to proceed when that is impossible.
      reduction = (
          stats_utils.reduce_pairs(agent.last_scalar_values, response_indices)
          if agent.last_scalar_values
          else None
      )
      if reduction is None:
        consistency_error = (
            "paired fallback has no per-response scalar measurements, so"
            " pairs cannot be reconstructed"
        )
      elif reduction.n_orphans:
        consistency_error = (
            f"paired fallback found {reduction.n_orphans} responses whose"
            " partner arm is missing, so pairs cannot be reconstructed"
        )
      else:
        diagnostics = PairedDiagnostics(
            n_pairs=reduction.n_pairs,
            n_tied=reduction.n_tied,
            n_positive=reduction.n_positive,
            n_negative=reduction.n_negative,
            n_orphans=reduction.n_orphans,
            tie_policy=tie_policy.value,
            unit_of_analysis="pair",
        )
        if reduction.tie_rate > 0.2:
          log_fn(
              "  [Single-Batch] WARNING: tie rate"
              f" {reduction.tie_rate:.2f} exceeds 0.20 -- the metric may not"
              " resolve the two sides of a pair."
          )

        if tie_policy is stats_plan.TiePolicy.RETAIN:
          # Paired continuous: test the mean of the per-pair differences.
          # Ties are kept, since a zero difference is a real measurement.
          n_obs = reduction.n_pairs
          obs_rate = float(np.mean(reduction.diffs)) if reduction.diffs else 0.0
          sample_std = (
              float(np.std(reduction.diffs, ddof=1)) if n_obs > 1 else 0.0
          )
          if sample_std > 0:
            t_stat = (obs_rate - null_rate) / (sample_std / np.sqrt(n_obs))
            p_value = stats_utils.get_t_test_p_value(t_stat, alt, df=n_obs - 1)
          else:
            p_value = 1.0 if obs_rate == null_rate else 0.0
          ci_lower, ci_upper = stats_utils.compute_continuous_means_ci(
              sample_mean=obs_rate,
              sample_std=sample_std,
              n_obs=n_obs,
              alpha=alpha,
              alternative=alt,
          )
          log_fn(
              f"  [Single-Batch] Paired continuous fallback over {n_obs}"
              f" pairs (mean diff {obs_rate:.4f}, {reduction.n_tied} ties)"
          )
        else:
          # The pairs are reconstructable, but nothing in the pre-registration
          # records which side of a pair winning counts as a success. Guessing
          # is anti-conservative half the time, and a spurious rejection is
          # exactly what this framework exists to measure, so fail instead.
          consistency_error = (
              "paired rate fallback cannot determine which side of a pair"
              " counts as a success; the pre-registration does not record a"
              " pair-level success direction"
          )
    elif agent.last_scalar_values and is_cont_plan:
      n_obs = len(agent.last_scalar_values)
      obs_rate = float(np.mean(agent.last_scalar_values))
      sample_std = (
          float(np.std(agent.last_scalar_values, ddof=1)) if n_obs > 1 else 0.0
      )
      if sample_std > 0:
        df = n_obs - 1
        t_stat = (obs_rate - null_rate) / (sample_std / np.sqrt(n_obs))
        alt_str = _resolve_alternative(test_plan, res)
        p_value = stats_utils.get_t_test_p_value(t_stat, alt_str, df=df)
      else:
        p_value = 1.0 if obs_rate == null_rate else 0.0
      ci_lower, ci_upper = stats_utils.compute_continuous_means_ci(
          sample_mean=obs_rate,
          sample_std=sample_std,
          n_obs=n_obs,
          alpha=alpha,
          alternative=alt,
      )
    elif agent.last_scalar_values:
      thresh = 0.0
      n_obs = len(agent.last_scalar_values)
      n_succ = sum(1 for v in agent.last_scalar_values if v > thresh)
      obs_rate = n_succ / n_obs if n_obs > 0 else 0.0
      p_value, obs_rate, ci_lower, ci_upper = (
          stats_utils.compute_binomial_replication_stats(
              n_succ, n_obs, null_prob=null_rate, alternative=alt, alpha=alpha
          )
      )
    elif agent.last_judgments:
      judgments = [j for j in agent.last_judgments if isinstance(j, dict)]
      n_obs = len(judgments)
      n_succ = sum(
          1
          for j in judgments
          if j.get("is_failure", False)
          or j.get("exhibits_behavior", False)
          or "fail" in str(j.get("judgment", "")).lower()
          or "refus" in str(j.get("judgment", "")).lower()
      )
      obs_rate = n_succ / n_obs if n_obs > 0 else 0.0
      p_value, obs_rate, ci_lower, ci_upper = (
          stats_utils.compute_binomial_replication_stats(
              n_succ, n_obs, null_prob=null_rate, alternative=alt, alpha=alpha
          )
      )

  if consistency_error:
    # Scoring this would mean reporting a p-value whose meaning we cannot
    # state. It gets its own status so downstream analysis can exclude these
    # from the denominator rather than silently treating them as
    # non-discoveries, which would bias a Type-I error measurement downward.
    log_fn(
        f"\n[Single-Batch] PLAN INCONSISTENCY [{h_id}]: {consistency_error}."
        " Failing this hypothesis instead of reporting an uninterpretable"
        " p-value."
    )
    p_value = 1.0
    verified = False
  else:
    verified = p_value <= alpha

  log_fn(
      f"\n[Single-Batch Results] [{h_id}] Verified: {verified} | P-value:"
      f" {stats_utils.format_p_value(p_value)} | Obs Rate: {obs_rate:.4f}"
      f" ({n_succ}/{n_obs}) | Null: {null_rate:.4f} | 95% CI: [{ci_lower:.4f},"
      f" {ci_upper:.4f}]"
  )
  if diagnostics is not None:
    log_fn(
        f"[Single-Batch Paired] [{h_id}] pairs={diagnostics.n_pairs}"
        f" tied={diagnostics.n_tied} orphans={diagnostics.n_orphans}"
        f" tie_policy={diagnostics.tie_policy}"
    )

  if consistency_error:
    status = "PLAN_INCONSISTENT"
  elif not verified and block_reasons:
    status = "SCRAPE_FAILED"
  else:
    status = "OK"
  return HypothesisPrecisionRecord(
      hypothesis_id=h_id,
      hypothesis_text=h_text,
      verified=verified,
      final_e_product=1.0,
      n_subhyps_used=1,
      e_values=[(h_text, 1.0)],
      status=status,
      block_reasons=block_reasons,
      p_value=p_value,
      sample_size=n_obs,
      observed_successes=n_succ,
      observed_rate=obs_rate,
      null_rate=null_rate,
      ci_lower=ci_lower,
      ci_upper=ci_upper,
      paired_diagnostics=diagnostics,
      contrast_diagnostics=(
          [o.to_dict() for o in contrast_outcomes]
          if contrast_outcomes is not None
          else None
      ),
      failure_reason=consistency_error or "",
  )


async def _check_one_hypothesis_single_batch(
    h_id: str,
    h_text: str,
    config: verifier_config.VerificationConfig,
    log_fn: Callable[..., None],
    stored_plan: stats_plan.PreRegisteredTestPlan | None = None,
    q_count: int | None = None,
) -> HypothesisPrecisionRecord:
  """Runs single-batch p-value precision check for a single hypothesis.

  Args:
    h_id: Hypothesis identifier.
    h_text: Hypothesis text.
    config: Verification config.
    log_fn: Logging function.
    stored_plan: See `_setup_and_preregister_single_batch`.
    q_count: Queries to spend; None means the flat per-hypothesis size.

  Returns:
    The precision record.
  """
  setup = await _setup_and_preregister_single_batch(
      h_id, h_text, config, log_fn, stored_plan=stored_plan
  )
  return await _execute_single_batch(
      setup, setup.standalone_q_count if q_count is None else q_count, config
  )


async def run_single_batch_precision_check(
    verified_hypotheses: list[tuple[str, records_lib.HypothesisRecord]],
    config: verifier_config.VerificationConfig,
    log_fn: Callable[..., None],
    stored_plans: Mapping[str, stats_plan.PreRegisteredTestPlan] | None = None,
    retry_budget: RetryBudget | None = None,
) -> PrecisionResult:
  """Runs single-batch p-value precision check for all verified hypotheses in parallel.

  Args:
    verified_hypotheses: (id, record) pairs to check.
    config: Verification config.
    log_fn: Logging function.
    stored_plans: Plans pre-registered by an earlier check, keyed by hypothesis
      id. A hypothesis with a stored plan reuses it instead of pre-registering
      again.
    retry_budget: Set when re-checking a subset of an earlier check; each
      hypothesis then gets its original query count rather than a share of the
      total budget split over the subset.

  Returns:
    The aggregated precision result.
  """
  stored_plans = stored_plans or {}
  log_fn(
      "\n=== Starting Single-Batch Precision Check for"
      f" {len(verified_hypotheses)} hypotheses ==="
  )

  total_v_budget = getattr(config, "total_verification_budget", 0) or 0
  if total_v_budget > 0 and verified_hypotheses:
    n_hyps = (
        retry_budget.n_original
        if retry_budget is not None
        else len(verified_hypotheses)
    )
    per_hyp_queries = max(1, total_v_budget // n_hyps)
    log_fn(
        f"[Single-Batch] Total verification budget: {total_v_budget} split"
        f" across {n_hyps} hypotheses -> {per_hyp_queries} base queries per"
        " hypothesis"
    )
    if dataclasses.is_dataclass(config):
      config = dataclasses.replace(
          typing.cast(Any, config), single_batch_num_queries=per_hyp_queries
      )
    else:
      config.single_batch_num_queries = per_hyp_queries

  def _make_tagged_log(
      h_id: str, base_log_fn: Callable[..., None]
  ) -> Callable[..., None]:
    tag = f"[{h_id[:25]}]"

    def tagged(msg: str, *args: Any, **kwargs: Any) -> None:
      if "\n" in msg:
        tagged_msg = msg.replace("\n", f"\n{tag} ")
      else:
        tagged_msg = f"{tag} {msg}"
      base_log_fn(tagged_msg, *args, **kwargs)

    return tagged

  max_concurrency = getattr(config, "max_concurrent_hypotheses", 5) or 5
  sem = asyncio.Semaphore(max_concurrency)
  total_v_budget = getattr(config, "total_verification_budget", 0) or 0

  async def _sem_setup(
      h_id: str, h_record: records_lib.HypothesisRecord
  ) -> _SingleBatchSetup:
    async with sem:
      return await _setup_and_preregister_single_batch(
          h_id,
          h_record.text,
          config,
          _make_tagged_log(h_id, log_fn),
          stored_plan=stored_plans.get(h_id),
      )

  async def _sem_execute(
      setup: _SingleBatchSetup, q_count: int
  ) -> HypothesisPrecisionRecord:
    async with sem:
      return await _execute_single_batch(setup, q_count, config)

  async def _sem_check_single_batch(
      h_id: str, h_record: records_lib.HypothesisRecord
  ) -> HypothesisPrecisionRecord:
    async with sem:
      return await _check_one_hypothesis_single_batch(
          h_id,
          h_record.text,
          config,
          _make_tagged_log(h_id, log_fn),
          stored_plan=stored_plans.get(h_id),
          q_count=(
              retry_budget.stored(h_id) if retry_budget is not None else None
          ),
      )

  if total_v_budget > 0 and verified_hypotheses:
    # Phase 1: every hypothesis pre-registers before any query is spent.
    # Allocation depends on which hypotheses went paired, and that is not
    # known until they have each committed to a test plan.
    setups = await asyncio.gather(
        *[_sem_setup(h_id, h_record) for h_id, h_record in verified_hypotheses]
    )

    # Phase 2: divide the budget by weight. A paired hypothesis costs two
    # queries per observation, so weighting it 2 gives every hypothesis the
    # same effective sample size rather than halving the paired ones.
    n_paired = sum(1 for s in setups if s.is_paired)
    if retry_budget is not None:
      # A retry keeps each hypothesis at its original allocation. Records from
      # before allocations were saved get the original per-hypothesis share.
      base = max(1, total_v_budget // max(retry_budget.n_original, 1))
      allocations = [
          retry_budget.stored(s.h_id) or base * s.budget_weight for s in setups
      ]
      log_fn(
          f"[Single-Batch] Retry: {len(setups)} of {retry_budget.n_original}"
          " hypotheses keep their original query counts (unrecorded ones get"
          f" the original share, {base} per unit weight)."
      )
    else:
      weights = [s.budget_weight for s in setups]
      allocations = allocate_verification_budget(weights, total_v_budget)
      log_fn(
          f"[Single-Batch] Total verification budget: {total_v_budget} across"
          f" {len(setups)} hypotheses ({n_paired} paired, weight 2;"
          f" {len(setups) - n_paired} unpaired, weight 1)"
      )
    for setup, q_count in zip(setups, allocations):
      n_obs = q_count // 2 if setup.is_paired else q_count
      log_fn(
          f"  [Budget] {setup.h_id}: {q_count} queries"
          f" ({n_obs} observations, paired={setup.is_paired})"
      )
    spent = sum(allocations)
    if retry_budget is None and spent != total_v_budget:
      # Only reachable via the per-hypothesis floor of 1 query, which can
      # push a tiny budget over. Worth saying out loud rather than silently
      # overrunning what the caller asked for.
      log_fn(
          f"[Single-Batch] WARNING: allocated {spent} queries against a"
          f" budget of {total_v_budget}; the per-hypothesis minimum of 1"
          " query cannot be met within the budget."
      )

    # Phase 3: spend it. No further doubling -- the allocation is already the
    # response count, not the observation count.
    records = await asyncio.gather(*[
        _sem_execute(setup, q_count)
        for setup, q_count in zip(setups, allocations)
    ])
  else:
    records = await asyncio.gather(*[
        _sem_check_single_batch(h_id, h_record)
        for h_id, h_record in verified_hypotheses
    ])

  n_total = len(records)
  score = scoring.score(records)
  n_verified = score.n_verified
  n_evaluated = score.n_scored
  n_statistical_failures = sum(
      1 for r in records if r.status == "OK" and not r.verified
  )
  n_execution_errors = sum(1 for r in records if r.status != "OK")
  precision = score.precision

  log_fn("\n=== Single-Batch Precision Check Complete ===")
  log_fn(f"  Verified: {n_verified}/{n_evaluated} evaluated (Total: {n_total})")
  if n_statistical_failures > 0:
    log_fn(f"  Statistical Failures: {n_statistical_failures}/{n_evaluated}")
  if n_execution_errors > 0:
    log_fn(f"  Execution Errors: {n_execution_errors}/{n_total}")
  log_fn(f"  Precision: {scoring.format_precision(precision)}")
  log_fn(f"  Empirical FDR (1 - Precision): {score.empirical_fdr:.4f}")

  return PrecisionResult(
      records=records,
      precision=precision,
      n_verified=n_verified,
      n_total=n_total,
      n_evaluated=n_evaluated,
      n_statistical_failures=n_statistical_failures,
      n_execution_errors=n_execution_errors,
      n_verification_failures=n_execution_errors,
  )


async def run_precision_check(
    verified_hypotheses: list[tuple[str, records_lib.HypothesisRecord]],
    config: verifier_config.VerificationConfig,
    log_fn: Callable[..., None] = print,
    stored_plans: Mapping[str, stats_plan.PreRegisteredTestPlan] | None = None,
    retry_budget: RetryBudget | None = None,
) -> PrecisionResult:
  """Runs single-batch precision check for all verified hypotheses in parallel.

  Args:
    verified_hypotheses: (id, record) pairs to check.
    config: Verification config.
    log_fn: Logging function.
    stored_plans: Plans pre-registered by an earlier check, keyed by hypothesis
      id, reused instead of pre-registering again.
    retry_budget: Set when re-checking a subset of an earlier check; see
      `RetryBudget`.

  Returns:
    The aggregated precision result.
  """
  return await run_single_batch_precision_check(
      verified_hypotheses=verified_hypotheses,
      config=config,
      log_fn=log_fn,
      stored_plans=stored_plans or {},
      retry_budget=retry_budget,
  )
