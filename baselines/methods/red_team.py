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

"""Red team probing -> "Adaptive Red Teaming" (Perez et al., 2022).

The investigator brainstorms adversarial/edge-case probes in batches,
conditioning each batch on the previous batch's responses, then synthesizes
hypotheses once at the end.
"""

import asyncio
from typing import Any

from disco.baselines import common
from disco.baselines.adapters import base as adapters_base
from disco.baselines.methods import base
from disco.verification import scoring


async def run_red_team_probing_baseline(
    adapter: adapters_base.TaskProbeAdapter,
    client: Any,
    target_model: str,
    investigator_model: str,
    budget: int = 15,
    max_predictions: int = 10,
    batch_size: int = 15,
    checkpoint_dir: str = "",
    semaphore: asyncio.Semaphore | None = None,
    log: common.LogFn = common.default_log,
    min_response_fraction: float = scoring.DEFAULT_MIN_RESPONSE_FRACTION,
) -> common.GenericBaselineResult:
  """3. Red Team Probing -> "Adaptive Red Teaming" (Perez et al. 2022).

  Probes are generated in batches of at most `batch_size`, each conditioned on
  the previous batch's observations (falling back to the fixed categories when
  a proposal cannot be parsed). Only usable observations (see
  `base.run_probe_with_resample`) are shown to the investigator and charged to
  `total_target_queries`.

  Args:
    adapter: Task adapter.
    client: google.genai client.
    target_model: Model under audit.
    investigator_model: Model that proposes probes and synthesizes predictions.
    budget: Total number of probes to execute.
    max_predictions: Maximum number of predictions to keep.
    batch_size: Maximum number of probes generated per batch.
    checkpoint_dir: Directory for saving/restoring intermediate checkpoints.
    semaphore: Bounds concurrent model calls, if set.
    log: Log function.
    min_response_fraction: Minimum fraction of valid samples per probe.

  Returns:
    The `GenericBaselineResult` for the adaptive red-team probing run.
  """
  log(
      f"\n--- [Baseline 3/5: Red Team Probing] ({adapter.name},"
      f" budget={budget}, batch_size={batch_size},"
      f" max_predictions={max_predictions}) ---"
  )

  async def _probe(
      p_cfg: dict[str, Any], round_idx: int, probe_idx: int
  ) -> common.GenericProbeObservation:
    return await base.run_probe_with_resample(
        adapter,
        client,
        target_model,
        p_cfg,
        round_idx=round_idx,
        probe_idx=probe_idx,
        min_fraction=min_response_fraction,
        semaphore=semaphore,
        log=log,
    )

  checkpointer = common.Checkpointer(
      checkpoint_dir, "RedTeamProbingBaseline", log
  )
  observations: list[common.GenericProbeObservation] = []
  # The probe of each observation, so that a resumed run can re-run the
  # unusable ones. None for observations from older checkpoints.
  probe_configs: list[dict[str, Any] | None] = []
  ckpt = checkpointer.load()
  if ckpt and "observations" in ckpt:
    observations = [common.obs_from_dict(d) for d in ckpt["observations"]]
    probe_configs = list(ckpt.get("probe_configs") or [])[: len(observations)]
    probe_configs += [None] * (len(observations) - len(probe_configs))
    log(
        f"[CHECKPOINT] Restored {len(observations)} red-team probes from"
        f" {checkpoint_dir}."
    )
    if any(not o.usable for o in observations):
      observations = await base.rerun_unusable(
          adapter,
          client,
          target_model,
          observations,
          probe_configs,
          min_fraction=min_response_fraction,
          semaphore=semaphore,
          log=log,
      )
      checkpointer.save(
          observations=[o.to_dict() for o in observations],
          probe_configs=probe_configs,
      )

  # Run in iterative batches with feedback, saving a checkpoint after every
  # batch so preempted runs resume from the last completed batch.
  all_target_probes: list[dict[str, Any]] = [
      dict(c) if c is not None else {} for c in probe_configs
  ]
  round_num = max((o.round_idx for o in observations), default=0) + 1
  while len(all_target_probes) < budget:
    needed = min(batch_size, budget - len(all_target_probes))
    if round_num == 1:
      red_team_prompt = (
          adapter.get_red_team_system_prompt()
          + f"\nGenerate {needed} diverse adversarial probe configurations"
          " as a JSON list."
      )
    else:
      # Unusable observations stay out of the investigator's context.
      usable_so_far = base.usable(observations)
      recent_obs = usable_so_far[-needed:]
      obs_text = "\n".join(
          [f"- {o.prompt_summary} => {o.response_summary}" for o in recent_obs]
      )
      note = base.unusable_note(
          len(observations) - len(usable_so_far),
          len(observations),
      )
      if note:
        obs_text += f"\n{note}"
      red_team_prompt = (
          adapter.get_red_team_system_prompt()
          + f"\nPrevious probe observations:\n{obs_text}\n\nBased on these"
          f" findings, generate {needed} diverse adversarial follow-up"
          " probe configurations as a JSON list."
      )

    raw_text = await common.ask_investigator(
        client=client,
        investigator_model=investigator_model,
        prompt=red_team_prompt,
        temperature=0.8,
        semaphore=semaphore,
        log=log,
        error_message=(
            f"Investigator red team generation error (round {round_num})"
        ),
    )

    batch_probes = common.parse_probe_json_list(raw_text)
    if not batch_probes:
      # Fallback to category pool
      cats = adapter.get_fixed_probe_categories()
      offset = len(all_target_probes) % len(cats)
      batch_probes = cats[offset : offset + needed]
      if not batch_probes:
        batch_probes = cats[:needed]

    batch_probes = batch_probes[:needed]
    log(
        f"Executing batch {round_num}: {len(batch_probes)} red team probes"
        f" ({len(all_target_probes) + len(batch_probes)}/{budget} total)..."
    )
    base_idx = len(all_target_probes)
    obs_batch = list(
        await asyncio.gather(*[
            _probe(p_cfg, round_num, base_idx + idx)
            for idx, p_cfg in enumerate(batch_probes, 1)
        ])
    )
    all_target_probes.extend(batch_probes)
    observations.extend(obs_batch)
    round_num += 1
    checkpointer.save(
        observations=[o.to_dict() for o in observations],
        probe_configs=list(all_target_probes),
    )

  cov = base.coverage(observations)
  base.check_coverage("RedTeamProbingBaseline", cov, min_response_fraction, log)
  usable_observations = base.usable(observations)
  observations_text = "\n".join([
      f"- {o.prompt_summary} => {o.response_summary}"
      for o in usable_observations
  ])
  note = base.unusable_note(
      len(observations) - len(usable_observations), len(observations)
  )
  if note:
    observations_text += f"\n{note}"

  pred_text, predictions = await base.synthesize_predictions(
      adapter=adapter,
      client=client,
      investigator_model=investigator_model,
      observations_text=observations_text,
      baseline_context_header=(
          "Here are observations from an adversarial red-team audit:"
      ),
      max_predictions=max_predictions,
      semaphore=semaphore,
      log=log,
  )

  return common.GenericBaselineResult(
      method_name="RedTeamProbingBaseline",
      origin="Adaptive Red Teaming (Perez et al., 2022)",
      predictions=predictions,
      structured_hypotheses=common.build_structured_hypotheses_from_preds(
          predictions, adapter.suite, adapter.task_null_proportion
      ),
      reasoning=pred_text,
      observations=observations,
      total_target_queries=len(usable_observations),
      metadata={"coverage": cov},
  )


class RedTeam:
  """`base.DiscoveryMethod` for `run_red_team_probing_baseline`."""

  name = "red_team"

  def plan(self, budget: int, opts: base.PlanOptions) -> dict[str, Any]:
    """Returns execution parameters for adaptive red teaming."""
    return {
        "budget": budget,
        "batch_size": opts.red_team_batch_size,
    }

  async def run(
      self,
      adapter: adapters_base.TaskProbeAdapter,
      ctx: base.RunContext,
      **plan: Any,
  ) -> common.GenericBaselineResult:
    """Runs the adaptive red-team probing baseline."""
    return await run_red_team_probing_baseline(
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
