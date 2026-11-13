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

"""Golden cases pinning the recipe and hedging category pools.

Pins, for each pool:

  * the module constant (`recipe_categories_pool` / `hedging_categories_pool`),
  * what the recipe and hedging adapters return from
    `get_fixed_probe_categories()`, and
  * the prompt text each adapter sends to the target model for every fixed
    probe.

Each pool entry is pinned as `json.dumps(entry, ensure_ascii=False)`, which
keeps the dict key order (golden files are written with sorted keys).
"""

import asyncio
from collections.abc import Callable, Mapping, Sequence
import json
import types
from typing import Any

from disco.baselines import hedging_categories_pool
from disco.baselines import recipe_categories_pool
from disco.baselines.adapters import base as adapters_base
from disco.baselines.adapters import registry as adapters_registry
from disco.tasks import hedging_tasks
from disco.tasks import recipe_tasks


def _entries(pool: Sequence[Mapping[str, Any]]) -> list[str]:
  return [json.dumps(entry, ensure_ascii=False) for entry in pool]


def pools() -> dict[str, list[str]]:
  """Returns the two module-level pool constants."""
  return {
      "recipe": _entries(recipe_categories_pool.RECIPE_CATEGORIES),
      "hedging": _entries(hedging_categories_pool.HEDGING_CATEGORIES),
  }


def _adapters() -> dict[str, adapters_base.TaskProbeAdapter]:
  return {
      "recipe": adapters_registry.get_task_adapter(
          recipe_tasks.RECIPE_TRANSLATE_FRENCH_LONG_TASK
      ),
      "hedging": adapters_registry.get_task_adapter(
          hedging_tasks.HEDGING_UNHEDGED_CONTROVERSIAL_TASK
      ),
  }


def adapter_fixed_probes() -> dict[str, list[str]]:
  """Returns `get_fixed_probe_categories()` of each adapter."""
  return {
      name: _entries(adapter.get_fixed_probe_categories())
      for name, adapter in _adapters().items()
  }


class _RecordingModels:
  """Stands in for `client.aio.models`, recording every prompt sent."""

  def __init__(self):
    self.contents: list[Any] = []

  async def generate_content(
      self, model: str, contents: Any, config: Any = None
  ) -> Any:
    del model, config  # Unused.
    self.contents.append(contents)
    return types.SimpleNamespace(text="one two three")


async def _prompts_sent(
    adapter: adapters_base.TaskProbeAdapter,
) -> list[list[Any]]:
  """Runs execute_probe on every fixed probe; returns the prompts sent."""
  sent = []
  for idx, probe in enumerate(adapter.get_fixed_probe_categories(), 1):
    models = _RecordingModels()
    client = types.SimpleNamespace(aio=types.SimpleNamespace(models=models))
    await adapter.execute_probe(
        client,
        "fake-target-model",
        probe,
        round_idx=1,
        probe_idx=idx,
        log=lambda _: None,
    )
    sent.append(models.contents)
  return sent


def prompts_sent() -> dict[str, list[list[Any]]]:
  """Returns the target prompts each adapter sends for every fixed probe."""
  return {
      name: asyncio.run(_prompts_sent(adapter))
      for name, adapter in _adapters().items()
  }


def category_pools_snapshot() -> dict[str, Any]:
  return {
      "pools": pools(),
      "adapter_fixed_probes": adapter_fixed_probes(),
      "prompts_sent": prompts_sent(),
  }


# Golden name -> case function, in the same shape as
# stats_golden_cases.ALL_CASES.
ALL_CASES: dict[str, Callable[[], Any]] = {
    "category_pools": category_pools_snapshot,
}
