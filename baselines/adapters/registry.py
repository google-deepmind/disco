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

"""Registry of the discovery baselines' task adapters."""

from disco.baselines.adapters import base
from disco.baselines.adapters import hedging
from disco.baselines.adapters import recipe
from disco.baselines.adapters import t2i
from disco.baselines.adapters import visual_symmetry
from disco.tasks import data_classes

# Adapter classes, keyed by `TaskSpec.adapter`.
ADAPTERS: dict[str, type[base.TaskProbeAdapter]] = {
    "t2i_bias": t2i.T2IBiasTaskAdapter,
    "visual_symmetry": visual_symmetry.VisualSymmetryTaskAdapter,
    "recipe": recipe.RecipeTaskAdapter,
    "hedging": hedging.HedgingTaskAdapter,
}

# The `get_task_adapter` options each adapter takes, on top of the task and
# seed instruction that every adapter takes.
_ADAPTER_OPTIONS: dict[str, tuple[str, ...]] = {
    "t2i_bias": ("judge_model",),
}


def get_task_adapter(
    task: data_classes.Task,
    seed_instruction: str = "",
    judge_model: str = "",
) -> base.TaskProbeAdapter:
  """Factory selecting the appropriate TaskProbeAdapter for a Task.

  Dispatches on `task.adapter` (set in tasks/data/<suite>.yaml): one of
  'hedging', 'recipe', 't2i_bias', 'visual_symmetry'.

  Args:
    task: Task to probe.
    seed_instruction: Passed to the adapter.
    judge_model: Passed to the T2I adapter.

  Returns:
    An initialized `TaskProbeAdapter` instance for `task`.

  Raises:
    ValueError: If `task.adapter` is missing or not a known adapter name.
  """
  if not task.adapter:
    raise ValueError(
        f"Task {task.name!r} (suite {task.suite!r}) sets no adapter."
    )
  adapter = task.adapter

  adapter_cls = ADAPTERS.get(adapter)
  if adapter_cls is None:
    raise ValueError(f"Task {task.name!r} has unknown adapter {adapter!r}")
  options = {"judge_model": judge_model}
  return adapter_cls(
      task,
      seed_instruction=seed_instruction,
      **{name: options[name] for name in _ADAPTER_OPTIONS.get(adapter, ())},
  )
