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

"""Looped baselines: "Investigator Agent" and "Repeated-Sample Probing".

Both run rounds of probes over the task's fixed category taxonomy, keep a
research log of every probe and response, and have the investigator synthesize
hypotheses from that log.

  * Looped investigator (Sheshadri et al., 2026): has the investigator
    propose each round's probes from the research log so far.
  * Looped multi-sample (Sheshadri et al., 2026): never adaptive, but samples
    each probe k >= 2 times to measure stochasticity (pass@k).
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from disco.baselines import common
from disco.baselines.adapters import base as adapters_base
from disco.baselines.methods import base
from disco.verification import scoring

# Proposes a round's probes: (round, research log so far, number of probes so
# far that returned no usable target data) -> probe configs.
_ProposeFn = Callable[[int, list[str], int], Awaitable[list[dict[str, Any]]]]


def _log_entry(obs: common.GenericProbeObservation) -> str:
  return (
      f"[Round {obs.round_idx}] {obs.prompt_summary} => {obs.response_summary}"
  )


async def _run_rounds(
    adapter: adapters_base.TaskProbeAdapter,
    client: Any,
    target_model: str,
    num_rounds: int,
    probes_per_round: int,
    checkpointer: common.Checkpointer,
    checkpoint_dir: str,
    round_tag: str,
    probe_kwargs: dict[str, Any],
    propose: _ProposeFn | None,
    min_response_fraction: float,
    semaphore: asyncio.Semaphore | None,
    log: common.LogFn,
) -> tuple[list[str], list[common.GenericProbeObservation]]:
  """Runs (or resumes) the rounds of a looped baseline.

  Args:
    adapter: Task adapter.
    client: google.genai client.
    target_model: Model under audit.
    num_rounds: Rounds to run.
    probes_per_round: Probes per round.
    checkpointer: Saves progress after every round.
    checkpoint_dir: Directory of `checkpointer`, for logging.
    round_tag: Appended to the "Round r/n" log line.
    probe_kwargs: Extra keyword arguments of `adapter.execute_probe`.
    propose: Proposes each round's probes, if set. Otherwise rounds take
      consecutive slices of the fixed categories.
    min_response_fraction: `--min_response_fraction`, for
      `base.run_probe_with_resample`.
    semaphore: Bounds concurrent model calls, if set.
    log: Log function.

  Returns:
    The research log, of usable observations only, and every observation,
    usable or not.
  """
  research_log: list[str] = []
  all_observations: list[common.GenericProbeObservation] = []
  # The probe of each observation, so that a resumed run can re-run the
  # unusable ones. None for observations from older checkpoints.
  all_configs: list[dict[str, Any] | None] = []
  categories = adapter.get_fixed_probe_categories()

  def _save(completed_round: int) -> None:
    checkpointer.save(
        completed_round=completed_round,
        research_log=research_log,
        observations=[o.to_dict() for o in all_observations],
        probe_configs=all_configs,
    )

  start_round = 1
  ckpt = checkpointer.load()
  if ckpt:
    start_round = ckpt.get("completed_round", 0) + 1
    research_log = ckpt.get("research_log", [])
    all_observations = [
        common.obs_from_dict(d) for d in ckpt.get("observations", [])
    ]
    all_configs = list(ckpt.get("probe_configs") or [])[: len(all_observations)]
    all_configs += [None] * (len(all_observations) - len(all_configs))
    log(
        f"[CHECKPOINT] Restored {len(all_observations)} observations up to"
        f" round {start_round - 1} from {checkpoint_dir}."
    )
    if any(not o.usable for o in all_observations):
      all_observations = await base.rerun_unusable(
          adapter,
          client,
          target_model,
          all_observations,
          all_configs,
          min_fraction=min_response_fraction,
          semaphore=semaphore,
          log=log,
          **probe_kwargs,
      )
      # The log holds the usable observations, in order.
      research_log = [_log_entry(o) for o in base.usable(all_observations)]
      _save(start_round - 1)

  for r in range(start_round, num_rounds + 1):
    log(f"Round {r}/{num_rounds}{round_tag}...")
    if propose is not None:
      n_unusable = len(all_observations) - len(base.usable(all_observations))
      round_probes = await propose(r, research_log, n_unusable)
      if not round_probes:
        offset = ((r - 1) * probes_per_round) % len(categories)
        round_probes = categories[offset : offset + probes_per_round]
        if not round_probes:
          round_probes = categories[:probes_per_round]
      round_probes = round_probes[:probes_per_round]
    else:
      round_probes = categories[
          (r - 1) * probes_per_round : r * probes_per_round
      ]
      if not round_probes:
        round_probes = categories[:probes_per_round]

    round_configs = []
    base_offset = len(all_observations)
    for p_idx, p_cfg in enumerate(round_probes, 1):
      cfg = dict(p_cfg)
      cfg["seed"] = r * 100 + p_idx
      round_configs.append((base_offset + p_idx, cfg))

    round_obs = list(
        await asyncio.gather(*[
            base.run_probe_with_resample(
                adapter,
                client,
                target_model,
                cfg,
                round_idx=r,
                probe_idx=p_idx,
                min_fraction=min_response_fraction,
                semaphore=semaphore,
                log=log,
                **probe_kwargs,
            )
            for p_idx, cfg in round_configs
        ])
    )
    for obs, (_, cfg) in zip(round_obs, round_configs):
      all_observations.append(obs)
      all_configs.append(cfg)
      # Unusable observations stay out of the investigator's context.
      if obs.usable:
        research_log.append(_log_entry(obs))
    _save(r)

  return research_log, all_observations


def _observations_text(
    research_log: list[str],
    all_observations: list[common.GenericProbeObservation],
) -> str:
  """The research log, plus a note on probes left out of it."""
  text = "\n".join(research_log)
  note = base.unusable_note(
      len(all_observations) - len(base.usable(all_observations)),
      len(all_observations),
  )
  return f"{text}\n{note}" if note else text


async def run_looped_investigator_baseline(
    adapter: adapters_base.TaskProbeAdapter,
    client: Any,
    target_model: str,
    investigator_model: str,
    num_rounds: int = 3,
    probes_per_round: int = 4,
    max_predictions: int = 10,
    checkpoint_dir: str = "",
    semaphore: asyncio.Semaphore | None = None,
    log: common.LogFn = common.default_log,
    min_response_fraction: float = scoring.DEFAULT_MIN_RESPONSE_FRACTION,
) -> common.GenericBaselineResult:
  """4. Looped Investigator -> "Investigator Agent" (Sheshadri et al., 2026).

  The investigator proposes each round's probes from the research log so far
  (falling back to the fixed categories when its proposal cannot be parsed).
  Only usable observations (see `base.run_probe_with_resample`) enter the
  research log and are charged to `total_target_queries`.

  Args:
    adapter: Task adapter.
    client: google.genai client.
    target_model: Model under audit.
    investigator_model: Model that proposes probes and synthesizes predictions.
    num_rounds: Number of probing rounds to execute.
    probes_per_round: Number of probes per round.
    max_predictions: Maximum number of predictions to keep.
    checkpoint_dir: Directory for saving/restoring intermediate checkpoints.
    semaphore: Bounds concurrent model calls, if set.
    log: Log function.
    min_response_fraction: Minimum fraction of usable samples per probe.

  Returns:
    The `GenericBaselineResult` for the looped investigator run.
  """
  log(
      f"\n--- [Baseline 4/5: Looped Investigator] ({adapter.name},"
      f" rounds={num_rounds}, max_predictions={max_predictions}) ---"
  )

  async def _propose(
      r: int, research_log: list[str], n_unusable: int
  ) -> list[dict[str, Any]]:
    if r == 1:
      inv_prompt = (
          adapter.get_looped_investigator_system_prompt()
          + f"\nPropose {probes_per_round} initial probe configurations to"
          " begin investigating. Return your proposal wrapped in ```json ["
          " { ... } ] ```."
      )
    else:
      recent_log = "\n".join(research_log[-probes_per_round * 2 :])
      note = base.unusable_note(n_unusable, n_unusable + len(research_log))
      if note:
        recent_log += f"\n{note}"
      inv_prompt = (
          adapter.get_looped_investigator_system_prompt()
          + f"\nResearch log so far:\n{recent_log}\n\nBased on these findings,"
          f" propose {probes_per_round} new follow-up probe configurations to"
          " test your hypotheses. Return your proposal wrapped in ```json ["
          " { ... } ] ```."
      )

    raw_text = await common.ask_investigator(
        client=client,
        investigator_model=investigator_model,
        prompt=inv_prompt,
        temperature=0.7,
        semaphore=semaphore,
        log=log,
        error_message=f"Investigator probe generation error (round {r})",
    )
    return common.parse_probe_json_list(raw_text)

  research_log, all_observations = await _run_rounds(
      adapter=adapter,
      client=client,
      target_model=target_model,
      num_rounds=num_rounds,
      probes_per_round=probes_per_round,
      checkpointer=common.Checkpointer(
          checkpoint_dir, "LoopedInvestigatorBaseline", log
      ),
      checkpoint_dir=checkpoint_dir,
      round_tag="",
      # The adapters' defaults: one sample, full (not binary) scoring.
      probe_kwargs={},
      propose=_propose,
      min_response_fraction=min_response_fraction,
      semaphore=semaphore,
      log=log,
  )
  cov = base.coverage(all_observations)
  base.check_coverage(
      "LoopedInvestigatorBaseline", cov, min_response_fraction, log
  )

  pred_text, predictions = await base.synthesize_predictions(
      adapter=adapter,
      client=client,
      investigator_model=investigator_model,
      observations_text=_observations_text(research_log, all_observations),
      baseline_context_header="Here is the cumulative multi-turn research log:",
      max_predictions=max_predictions,
      semaphore=semaphore,
      log=log,
  )

  return common.GenericBaselineResult(
      method_name="LoopedInvestigatorBaseline",
      origin="Investigator Agent (Sheshadri et al., 2026)",
      predictions=predictions,
      structured_hypotheses=common.build_structured_hypotheses_from_preds(
          predictions, adapter.suite, adapter.task_null_proportion
      ),
      reasoning=pred_text,
      observations=all_observations,
      total_target_queries=len(base.usable(all_observations)),
      metadata={"coverage": cov},
  )


async def run_looped_multisample_baseline(
    adapter: adapters_base.TaskProbeAdapter,
    client: Any,
    target_model: str,
    investigator_model: str,
    num_rounds: int = 2,
    probes_per_round: int = 3,
    k_samples: int = 3,
    max_predictions: int = 10,
    checkpoint_dir: str = "",
    semaphore: asyncio.Semaphore | None = None,
    log: common.LogFn = common.default_log,
    min_response_fraction: float = scoring.DEFAULT_MIN_RESPONSE_FRACTION,
) -> common.GenericBaselineResult:
  """5.

  Looped Multi-Sample -> "Repeated-Sample Probing" (Sheshadri et al., 2026).

  A probe with fewer than `min_response_fraction` of its `k_samples` valid is
  resampled and, if still short, unusable (see `base.run_probe_with_resample`).
  Only usable observations enter the research log and are charged
  (`k_samples` queries each) to `total_target_queries`.

  Args:
    adapter: Task adapter.
    client: google.genai client.
    target_model: Model under audit.
    investigator_model: Model that synthesizes predictions from the log.
    num_rounds: Number of probing rounds to execute.
    probes_per_round: Number of probes per round.
    k_samples: Number of samples drawn per probe.
    max_predictions: Maximum number of predictions to keep.
    checkpoint_dir: Directory for saving/restoring intermediate checkpoints.
    semaphore: Bounds concurrent model calls, if set.
    log: Log function.
    min_response_fraction: Minimum fraction of valid samples per probe.

  Returns:
    The `GenericBaselineResult` for the looped multi-sample run.
  """
  log(
      f"\n--- [Baseline 5/5: Looped Multi-Sample] ({adapter.name},"
      f" k={k_samples}, max_predictions={max_predictions}) ---"
  )

  research_log, all_observations = await _run_rounds(
      adapter=adapter,
      client=client,
      target_model=target_model,
      num_rounds=num_rounds,
      probes_per_round=probes_per_round,
      checkpointer=common.Checkpointer(
          checkpoint_dir, "LoopedMultiSampleBaseline", log
      ),
      checkpoint_dir=checkpoint_dir,
      round_tag=f" (k={k_samples})",
      probe_kwargs={"k_samples": k_samples},
      propose=None,
      min_response_fraction=min_response_fraction,
      semaphore=semaphore,
      log=log,
  )
  cov = base.coverage(all_observations)
  base.check_coverage(
      "LoopedMultiSampleBaseline", cov, min_response_fraction, log
  )

  pred_text, predictions = await base.synthesize_predictions(
      adapter=adapter,
      client=client,
      investigator_model=investigator_model,
      observations_text=_observations_text(research_log, all_observations),
      baseline_context_header=(
          f"Here is the multi-sample (pass@{k_samples}) research log:"
      ),
      max_predictions=max_predictions,
      semaphore=semaphore,
      log=log,
  )

  return common.GenericBaselineResult(
      method_name="LoopedMultiSampleBaseline",
      origin="Repeated-Sample Probing (Sheshadri et al., 2026)",
      predictions=predictions,
      structured_hypotheses=common.build_structured_hypotheses_from_preds(
          predictions, adapter.suite, adapter.task_null_proportion
      ),
      reasoning=pred_text,
      observations=all_observations,
      total_target_queries=len(base.usable(all_observations)) * k_samples,
      metadata={"coverage": cov},
  )


class LoopedInvestigator:
  """`base.DiscoveryMethod` for `run_looped_investigator_baseline`."""

  name = "looped_investigator"

  def plan(self, budget: int, opts: base.PlanOptions) -> dict[str, Any]:
    """Splits `budget` into rounds that scale with it (at least 3)."""
    del opts  # Unused.
    num_rounds = max(3, budget // 30)
    return {
        "num_rounds": num_rounds,
        "probes_per_round": max(2, budget // num_rounds),
    }

  async def run(
      self,
      adapter: adapters_base.TaskProbeAdapter,
      ctx: base.RunContext,
      **plan: Any,
  ) -> common.GenericBaselineResult:
    """Runs the looped investigator baseline."""
    return await run_looped_investigator_baseline(
        adapter=adapter,
        client=ctx.client,
        target_model=ctx.target_model,
        investigator_model=ctx.investigator_model,
        max_predictions=ctx.max_predictions,
        checkpoint_dir=ctx.checkpoint_dir,
        semaphore=ctx.semaphore,
        log=ctx.log,
        min_response_fraction=ctx.min_response_fraction,
        **plan,
    )


class LoopedMultiSample:
  """`base.DiscoveryMethod` for `run_looped_multisample_baseline`."""

  name = "looped_multisample"

  def plan(self, budget: int, opts: base.PlanOptions) -> dict[str, Any]:
    """Splits `budget` into rounds, probes and samples.

    Rounds and probes per round grow slowly with the budget (capped at 5 and
    8); the rest of the budget goes into samples per probe (at least 3).

    Args:
      budget: Target query budget.
      opts: Unused.

    Returns:
      Dictionary of `num_rounds`, `probes_per_round`, and `k_samples`.
    """
    del opts  # Unused.
    ms_rounds = max(2, min(5, budget // 100))
    ms_probes = max(3, min(8, budget // 50))
    total_probe_slots = ms_rounds * ms_probes
    ms_k = max(3, budget // total_probe_slots)
    return {
        "num_rounds": ms_rounds,
        "probes_per_round": ms_probes,
        "k_samples": ms_k,
    }

  async def run(
      self,
      adapter: adapters_base.TaskProbeAdapter,
      ctx: base.RunContext,
      **plan: Any,
  ) -> common.GenericBaselineResult:
    """Runs the looped multi-sample baseline."""
    return await run_looped_multisample_baseline(
        adapter=adapter,
        client=ctx.client,
        target_model=ctx.target_model,
        investigator_model=ctx.investigator_model,
        max_predictions=ctx.max_predictions,
        checkpoint_dir=ctx.checkpoint_dir,
        semaphore=ctx.semaphore,
        log=ctx.log,
        min_response_fraction=ctx.min_response_fraction,
        **plan,
    )
