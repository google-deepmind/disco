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

"""Simple probing -> "Uniform Probing" (Sheshadri et al., 2026).

Spends the full budget non-adaptively, spread evenly across the task's fixed
category taxonomy. Hypotheses are emitted programmatically, one per category,
from observed pass rates.
"""

import asyncio
import json
import random
from typing import Any

from disco.baselines import common
from disco.baselines.adapters import base as adapters_base
from disco.baselines.methods import base
from disco.verification import scoring


async def run_simple_probing_baseline(
    adapter: adapters_base.TaskProbeAdapter,
    client: Any,
    target_model: str,
    budget: int = 18,
    checkpoint_dir: str = "",
    seed: int = 42,
    semaphore: asyncio.Semaphore | None = None,
    log: common.LogFn = common.default_log,
    min_response_fraction: float = scoring.DEFAULT_MIN_RESPONSE_FRACTION,
) -> common.GenericBaselineResult:
  """2. Simple Probing Baseline -> "Uniform Probing" (Sheshadri et al., 2026).

  Rates, claims and `total_target_queries` cover usable observations only
  (see `base.run_probe_with_resample`); a category with none gets no rate.

  Args:
    adapter: Task adapter.
    client: google.genai client.
    target_model: Model under audit.
    budget: Total number of probes to execute.
    checkpoint_dir: Directory for saving/restoring intermediate checkpoints.
    seed: Random seed for stratified category/dish shuffling.
    semaphore: Bounds concurrent model calls, if set.
    log: Log function.
    min_response_fraction: Minimum fraction of valid samples per probe.

  Returns:
    The `GenericBaselineResult` for the uniform probing run.
  """
  log(
      f"\n--- [Baseline 2/5: Simple Probing] ({adapter.name},"
      f" budget={budget}, seed={seed}) ---"
  )
  categories = adapter.get_fixed_probe_categories()
  rng = random.Random(seed)

  # Group probe configs by category to enable stratified sampling across
  # categories.
  by_cat: dict[str, list[dict[str, Any]]] = {}
  for cat_cfg in categories:
    cat_name = cat_cfg.get("shape_type") or cat_cfg.get("category") or "general"
    by_cat.setdefault(cat_name, []).append(cat_cfg)

  # Shuffle dishes within each category deterministically using the replica seed
  for cat_name in by_cat:
    shuffled = list(by_cat[cat_name])
    rng.shuffle(shuffled)
    by_cat[cat_name] = shuffled

  # Sort category names for deterministic base order, then shuffle category
  # order.
  cat_names = sorted(by_cat.keys())
  rng.shuffle(cat_names)

  probe_configs: list[tuple[int, str, dict[str, Any]]] = []
  cat_ptrs: dict[str, int] = {c: 0 for c in cat_names}
  probe_idx = 0

  while len(probe_configs) < budget:
    for cat_name in cat_names:
      if len(probe_configs) >= budget:
        break
      dishes = by_cat[cat_name]
      ptr = cat_ptrs[cat_name]
      cat_cfg = dishes[ptr % len(dishes)]
      cat_ptrs[cat_name] += 1
      probe_idx += 1
      cfg = dict(cat_cfg)
      cfg["seed"] = seed * 1000 + probe_idx
      probe_configs.append((probe_idx, cat_name, cfg))

  total_probes = len(probe_configs)

  # Check for intermediate checkpoint
  checkpointer = common.Checkpointer(
      checkpoint_dir, "SimpleProbingBaseline", log
  )
  observations_map: dict[int, common.GenericProbeObservation] = {}
  ckpt = checkpointer.load()
  if ckpt:
    saved_raw = ckpt.get("observations", [])
    n_unusable_restored = 0
    for d in saved_raw:
      o = common.obs_from_dict(d)
      # A probe that returned no usable target data is run again, not reused.
      if not o.usable:
        n_unusable_restored += 1
        continue
      observations_map[o.probe_idx] = o
    if observations_map:
      log(
          f"[CHECKPOINT] Restored {len(observations_map)} completed probes"
          f" from {checkpoint_dir}."
      )
    if n_unusable_restored:
      log(
          f"[CHECKPOINT] Re-running {n_unusable_restored} restored probe(s)"
          " that returned no usable target data."
      )

  remaining_configs = [
      (p_idx, c_name, p_cfg)
      for p_idx, c_name, p_cfg in probe_configs
      if p_idx not in observations_map
  ]

  completed_count = len(observations_map)
  lock = asyncio.Lock()

  async def _run_probe(
      p_idx: int, c_name: str, p_cfg: dict[str, Any]
  ) -> common.GenericProbeObservation:
    del c_name
    nonlocal completed_count
    obs = await base.run_probe_with_resample(
        adapter,
        client,
        target_model,
        p_cfg,
        round_idx=1,
        probe_idx=p_idx,
        min_fraction=min_response_fraction,
        semaphore=semaphore,
        log=log,
    )
    async with lock:
      observations_map[p_idx] = obs
      completed_count += 1
      if (
          completed_count % max(1, total_probes // 5) == 0
          or completed_count == total_probes
      ):
        log(
            f"  [SimpleProbing] Progress: {completed_count}/{total_probes}"
            " probes completed..."
        )
        checkpointer.save(
            budget=budget,
            completed=completed_count,
            observations=[o.to_dict() for o in observations_map.values()],
        )
    return obs

  if remaining_configs:
    log(
        f"Launching {len(remaining_configs)} probes in parallel"
        f" ({len(observations_map)}/{total_probes} cached)..."
    )
    await asyncio.gather(*[
        _run_probe(p_idx, c_name, p_cfg)
        for p_idx, c_name, p_cfg in remaining_configs
    ])

  observations = [
      observations_map[p_idx]
      for p_idx, _, _ in probe_configs
      if p_idx in observations_map
  ]

  checkpointer.save(
      budget=budget,
      completed=len(observations),
      observations=[o.to_dict() for o in observations],
  )

  cov = base.coverage(observations)
  base.check_coverage("SimpleProbingBaseline", cov, min_response_fraction, log)
  usable_observations = base.usable(observations)

  # Rates over usable observations only.
  cat_counts: dict[str, int] = {}
  cat_passed: dict[str, int] = {}
  for obs in usable_observations:
    cat_counts[obs.category] = cat_counts.get(obs.category, 0) + 1
    if obs.passed:
      cat_passed[obs.category] = cat_passed.get(obs.category, 0) + 1

  rates = {
      cat: cat_passed.get(cat, 0) / count
      for cat, count in cat_counts.items()
      if count > 0
  }
  for cat in dict.fromkeys(o.category for o in observations):
    if cat not in rates:
      log(
          f"  [SimpleProbing] Category {cat!r} has no usable observations:"
          " no rate and no claim."
      )
  predictions = adapter.format_empirical_predictions(rates)

  return common.GenericBaselineResult(
      method_name="SimpleProbingBaseline",
      origin="Uniform Probing (Sheshadri et al., 2026)",
      predictions=predictions,
      structured_hypotheses=common.build_structured_hypotheses_from_preds(
          predictions, adapter.suite, adapter.task_null_proportion
      ),
      reasoning=json.dumps(rates, indent=2),
      observations=observations,
      total_target_queries=len(usable_observations),
      empirical_rates=rates,
      metadata={"coverage": cov},
  )


class SimpleProbing:
  """`base.DiscoveryMethod` for `run_simple_probing_baseline`."""

  name = "simple_probing"

  def plan(self, budget: int, opts: base.PlanOptions) -> dict[str, Any]:
    """Returns execution parameters for uniform probing."""
    return {"budget": budget, "seed": opts.seed}

  async def run(
      self,
      adapter: adapters_base.TaskProbeAdapter,
      ctx: base.RunContext,
      **plan: Any,
  ) -> common.GenericBaselineResult:
    """Runs the uniform probing baseline."""
    return await run_simple_probing_baseline(
        adapter=adapter,
        client=ctx.client,
        target_model=ctx.target_model,
        checkpoint_dir=ctx.checkpoint_dir,
        semaphore=ctx.semaphore,
        log=ctx.log,
        min_response_fraction=ctx.min_response_fraction,
        **plan,
    )
