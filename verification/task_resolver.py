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

"""Resolves task settings for verification from run metadata and the registry.

Moved verbatim from standalone_precision_verifier (which keeps thin delegates)
so other tools can share one implementation. Precedence everywhere is:
explicit CLI value > the JSON `config` block / top-level fields > the task
registry (matched by `task_name` / `task_suite`) > a fixed default.

Registry matches go through `registry.resolve` (a suite gives its default
task, otherwise an exact task key or alias). Names it cannot resolve fall back
to the legacy lookup (split words such as 'iteration', 'all', the first of
several same-named tasks, substrings of task keys), which logs a warning.
"""

from typing import Any

from absl import logging

from disco.tasks import registry

DEFAULT_SEED_INSTRUCTION = "Verify the provided hypotheses"


def _legacy_tasks(norm_key: str, by_split: bool) -> list[registry.TaskSpec]:
  """The pre-`resolve` lookup: `norm_key` as a suite, then as a split / name."""
  task_candidates = []
  try:
    task_candidates = registry.get_tasks(suite=norm_key, split="all")
  except (ValueError, KeyError):
    pass
  if not task_candidates and by_split:
    try:
      task_candidates = registry.get_tasks(suite="all", split=norm_key)
    except (ValueError, KeyError):
      pass
  return task_candidates


def _registry_task(
    key: str, suites_only: bool = False
) -> registry.TaskSpec | None:
  """Returns the registry task `key` names, or None.

  Args:
    key: A suite name, task name, key or alias.
    suites_only: Only match `key` as a suite name.
  """
  norm_key = registry.normalize_key(key)
  # "all" names every suite in the legacy lookup, so it bypasses `resolve`.
  if norm_key != "all" and (not suites_only or norm_key in registry.suites()):
    try:
      return registry.resolve(norm_key)
    except (KeyError, ValueError):
      pass
  task_candidates = _legacy_tasks(norm_key, by_split=not suites_only)
  if not task_candidates:
    return None
  logging.warning(
      "Task %r matched %r only through the legacy split/substring lookup (of"
      " %s); use a suite name, task key or alias.",
      key,
      task_candidates[0].key,
      [t.key for t in task_candidates],
  )
  return task_candidates[0]


def _registry_metadata(data: dict[str, Any], field: str) -> Any:
  """Looks up `field` in the metadata of the task named by `data`.

  Tries `task_name` / `task_suite` at the top level and then inside `config`;
  each key is matched first as a suite, then as a task (see `_registry_task`).
  The first truthy value wins.

  Args:
    data: Parsed hypotheses/evaluation JSON dict.
    field: Task metadata key to read.

  Returns:
    The first truthy metadata value, or None.
  """
  eval_config = data.get("config", {})
  task_keys = [
      data.get("task_name"),
      data.get("task_suite"),
      eval_config.get("task_name") if isinstance(eval_config, dict) else None,
      eval_config.get("task_suite") if isinstance(eval_config, dict) else None,
  ]
  for key in task_keys:
    if key and isinstance(key, str):
      task = _registry_task(key)
      if task is not None:
        val = task.metadata.get(field, "")
        if val:
          return val
  return None


def resolve_task_autorater(
    data: dict[str, Any] | list[Any] | None,
    cli_flag_val: str,
) -> str:
  """Resolves the procedural autorater with CLI flag > JSON config > task registry."""
  if cli_flag_val:
    return cli_flag_val

  if not data or not isinstance(data, dict):
    return ""

  eval_config = data.get("config", {})
  if isinstance(eval_config, dict):
    autorater = eval_config.get("procedural_autorater") or eval_config.get(
        "disco_procedural_autorater"
    )
    if autorater:
      return str(autorater)

  if "procedural_autorater" in data and data["procedural_autorater"]:
    return str(data["procedural_autorater"])

  auto = _registry_metadata(data, "procedural_autorater")
  if auto is not None:
    return str(auto)
  return ""


def resolve_seed_instruction(
    data: dict[str, Any] | list[Any] | None,
    cli_flag_val: str,
) -> str:
  """Resolves the seed_instruction with CLI flag > JSON config > task registry."""
  if cli_flag_val:
    return cli_flag_val

  if data and isinstance(data, dict):
    eval_config = data.get("config", {})
    if isinstance(eval_config, dict):
      seed = (
          eval_config.get("seed_instruction")
          or eval_config.get("disco_seed")
          or eval_config.get("seed")
      )
      if seed:
        return str(seed)
    if "seed_instruction" in data and data["seed_instruction"]:
      return str(data["seed_instruction"])
    if "disco_seed" in data and data["disco_seed"]:
      return str(data["disco_seed"])

    seed = _registry_metadata(data, "seed_instruction")
    if seed is not None:
      return str(seed)

  return DEFAULT_SEED_INSTRUCTION


def resolve_task_suite(
    data: dict[str, Any] | list[Any] | None,
    cli_flag_val: str,
) -> str:
  """Resolves the task suite with CLI flag > JSON top level > JSON config."""
  if cli_flag_val:
    return cli_flag_val

  if data and isinstance(data, dict):
    if data.get("task_suite"):
      return str(data["task_suite"])
    eval_config = data.get("config", {})
    if isinstance(eval_config, dict) and eval_config.get("task_suite"):
      return str(eval_config["task_suite"])

  return ""


def resolve_task_null_proportion(
    data: dict[str, Any] | list[Any] | None,
    task_suite: str,
) -> float:
  """Resolves the task's reference/null success rate from JSON > task registry.

  This is only a fallback for hypotheses that state no rate of their own; an
  explicit `single_batch_null_prob` still takes precedence downstream.

  Args:
    data: Parsed hypotheses/evaluation JSON, if available.
    task_suite: Already-resolved task suite, used to look up the registry.

  Returns:
    The task null proportion, or 0.0 if none could be resolved.
  """
  if data and isinstance(data, dict):
    eval_config = data.get("config", {})
    for src in (data, eval_config if isinstance(eval_config, dict) else {}):
      raw = src.get("null_proportion")
      if raw is not None:
        try:
          val = float(raw)
        except (TypeError, ValueError):
          continue
        if 0.0 < val < 1.0:
          return val

  if task_suite:
    task = _registry_task(task_suite, suites_only=True)
    if task is not None:
      try:
        val = float(task.metadata.get("null_proportion", 0.0) or 0.0)
      except (TypeError, ValueError):
        return 0.0
      if 0.0 < val < 1.0:
        return val

  return 0.0
