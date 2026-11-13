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

"""Task and hypothesis registry for DISCO benchmark and evaluation tasks.

Provides a unified interface to register, discover, and retrieve:
1. Benchmark Tasks across suites (e.g., Recipes, Hedging, T2I Biases, Visual
Symmetry).
2. Hypotheses across suites (e.g., Visual Symmetry candidate capability claims).

The built-in task suites are loaded from `tasks/data/<suite>.yaml` (see
`loader`). Lookups by name:

*   `get(name, suite=None)`: exact key or alias, `KeyError` (with suggestions)
    otherwise.
*   `resolve(name_or_suite)`: a suite name gives its `default_task`, anything
    else goes through `get`.
*   `list_specs(suite=None, split=None)`, `suites()`, `register(spec)`.

The legacy functions (`get_tasks`, `get_task`, `get_suite`, `list_tasks`,
`register_suite`, `get_hypotheses`, ...) keep their exact behavior, including
the substring-of-key fallback, which logs a warning when it is what matched.
"""

from collections.abc import Iterable, Sequence
import difflib
from typing import Optional

from absl import logging

from disco.tasks import data_classes
from disco.tasks import hedging_tasks
from disco.tasks import loader
from disco.tasks import recipe_tasks
from disco.tasks import spec
from disco.tasks import t2i_cat_bias_tasks
from disco.tasks import t2i_handedness_bias_tasks
from disco.tasks.visual_symmetry import hypotheses as visual_symmetry_hypotheses

TaskSpec = spec.TaskSpec

_TASK_REGISTRY: dict[str, list[data_classes.Task]] = {}
_HYPOTHESIS_REGISTRY: dict[str, list[data_classes.Hypothesis]] = {}
# Suite key -> key of the task `resolve(suite)` returns. Suites without an
# entry (or whose entry no longer names a task) resolve to their first task.
_DEFAULT_TASKS: dict[str, str] = {}


normalize_key = spec.normalize_key
_normalize_key = normalize_key


def _warn_substring(lookup: str, query: str, keys: Sequence[str]) -> None:
  logging.warning(
      "%s: %r matched %s only by substring of the key; use an exact key or"
      " alias.",
      lookup,
      query,
      list(keys),
  )


# ── Task Registry ─────────────────────────────────────────────────────────────


def register_suite(suite_name: str, tasks: Sequence[data_classes.Task]) -> None:
  """Registers a list of tasks under a suite name, replacing any old ones."""
  suite_key = _normalize_key(suite_name)
  _TASK_REGISTRY[suite_key] = list(tasks)
  _DEFAULT_TASKS.pop(suite_key, None)


def list_suites() -> list[str]:
  """Returns a sorted list of registered task suite names."""
  return sorted(_TASK_REGISTRY.keys())


def get_suite(suite_name: str) -> list[data_classes.Task]:
  """Returns all tasks in a registered suite."""
  suite_key = _normalize_key(suite_name)
  if suite_key not in _TASK_REGISTRY:
    raise ValueError(
        f"Unknown task suite: {suite_name!r}. Registered suites:"
        f" {list_suites()}"
    )
  return list(_TASK_REGISTRY[suite_key])


def get_tasks(
    suite: str = "all",
    split: str = "all",
) -> list[data_classes.Task]:
  """Returns tasks filtered by suite and/or split.

  Args:
    suite: Suite name (e.g. 'recipes', 'visual_symmetry') or 'all' for all
      suites.
    split: Split name (e.g. 'iteration', 'evaluation', 'all') or specific task
      name / key.

  Returns:
    List of matching Task objects.
  """
  suite_key = _normalize_key(suite)
  split_key = _normalize_key(split)

  if suite_key == "all":
    all_tasks: list[data_classes.Task] = []
    for s_tasks in _TASK_REGISTRY.values():
      all_tasks.extend(s_tasks)
    candidate_tasks = all_tasks
  else:
    if suite_key not in _TASK_REGISTRY:
      raise ValueError(
          f"Unknown task suite: {suite!r}. Registered suites: {list_suites()}"
      )
    candidate_tasks = list(_TASK_REGISTRY[suite_key])

  if split_key == "all":
    return candidate_tasks
  if split_key in ("iteration", "iter"):
    return [t for t in candidate_tasks if t.split == "iteration"]
  if split_key in ("evaluation", "eval"):
    return [t for t in candidate_tasks if t.split == "evaluation"]

  # Check exact name/key match or metadata alias match
  exact_matches = [
      t
      for t in candidate_tasks
      if t.key == split_key or split_key in t.alias_keys
  ]
  if exact_matches:
    return exact_matches

  # Substring match fallback.
  substr_matches = [t for t in candidate_tasks if split_key in t.key]
  if substr_matches:
    _warn_substring(
        f"get_tasks(suite={suite!r})", split, [t.key for t in substr_matches]
    )
    return substr_matches

  # Also check if split matches task's custom split attribute
  split_matches = [t for t in candidate_tasks if t.split.lower() == split_key]
  if split_matches:
    return split_matches

  raise ValueError(
      f"No tasks found for suite={suite!r} and split={split!r}. "
      f"Available suites: {list_suites()}"
  )


def get_task(name: str, suite: Optional[str] = None) -> data_classes.Task:
  """Retrieves a single task by name.

  Prefer `get`, which does not fall back to substring matches.

  Args:
    name: Task name or key (e.g., 'recipe_translate_french_long').
    suite: Optional suite to scope the search to.

  Returns:
    The matching Task object.
  """
  suite_name = suite if suite else "all"
  matching = get_tasks(suite=suite_name, split=name)
  if not matching:
    raise ValueError(f"Task not found: {name!r} in suite={suite_name!r}")
  return matching[0]


def get_task_descriptions(
    suite: str = "all",
    split: str = "all",
) -> list[str]:
  """Returns failure mode / behavior description strings for matching tasks."""
  return [t.description for t in get_tasks(suite=suite, split=split)]


def list_tasks(suite: Optional[str] = None) -> list[str]:
  """Returns normalized keys of all available tasks."""
  suite_name = suite if suite else "all"
  return [t.key for t in get_tasks(suite=suite_name, split="all")]


# ── Typed lookups ─────────────────────────────────────────────────────────────


def _suggest(key: str, names: Iterable[str]) -> str:
  close = difflib.get_close_matches(key, list(names), n=3)
  return f" Did you mean {close}?" if close else ""


def suites() -> list[str]:
  """Returns the sorted names of the registered task suites."""
  return list_suites()


def list_specs(
    suite: Optional[str] = None, split: Optional[str] = None
) -> list[TaskSpec]:
  """Returns registered tasks, in registration order.

  Args:
    suite: If set, only this suite's tasks.
    split: If set, only tasks whose `split` equals it exactly (no 'iter' /
      'eval' shorthands).

  Raises:
    KeyError: If `suite` is not registered.
  """
  if suite is None:
    specs: list[TaskSpec] = []
    for tasks in _TASK_REGISTRY.values():
      specs.extend(tasks)
  else:
    suite_key = _normalize_key(suite)
    if suite_key not in _TASK_REGISTRY:
      raise KeyError(
          f"Unknown task suite {suite!r}."
          f"{_suggest(suite_key, _TASK_REGISTRY)} Registered suites:"
          f" {list_suites()}"
      )
    specs = list(_TASK_REGISTRY[suite_key])
  if split is not None:
    specs = [t for t in specs if t.split == split]
  return specs


def get(name: str, suite: Optional[str] = None) -> TaskSpec:
  """Returns the task whose key or alias is `name` (normalized).

  Unlike `get_task`, there is no substring fallback.

  Args:
    name: Task name, key or alias.
    suite: If set, only look in this suite.

  Raises:
    KeyError: If no task matches; the message suggests close names.
    ValueError: If tasks in different suites match.
  """
  key = _normalize_key(name)
  candidates = list_specs(suite)
  matches = [t for t in candidates if t.key == key or key in t.alias_keys]
  if len(matches) == 1:
    return matches[0]
  if matches:
    raise ValueError(
        f"Task {name!r} is ambiguous: it names"
        f" {[f'{t.suite}/{t.key}' for t in matches]}; pass suite=."
    )
  names: list[str] = []
  for t in candidates:
    names.extend((t.key, *t.alias_keys))
  scope = f" in suite {suite!r}" if suite is not None else ""
  raise KeyError(f"Unknown task {name!r}{scope}.{_suggest(key, names)}")


def resolve(name_or_suite: str) -> TaskSpec:
  """Returns a suite's default task, or the task named `name_or_suite`.

  Args:
    name_or_suite: A registered suite name (gives the suite's `default_task`,
      which for the built-in suites is its first task), or a task name, key or
      alias (see `get`).

  Raises:
    KeyError: If it names neither a suite nor a task, or the suite is empty.
  """
  suite_key = _normalize_key(name_or_suite)
  if suite_key not in _TASK_REGISTRY:
    return get(name_or_suite)
  tasks = _TASK_REGISTRY[suite_key]
  if not tasks:
    raise KeyError(f"Task suite {name_or_suite!r} has no tasks.")
  default = _DEFAULT_TASKS.get(suite_key)
  for task in tasks:
    if task.key == default:
      return task
  return tasks[0]


def register(task: TaskSpec) -> None:
  """Adds one task to its suite, creating the suite if needed.

  Args:
    task: Task to add. Its `suite` must be set.

  Raises:
    ValueError: If `task.suite` is empty, or its key or an alias is already
      taken in that suite.
  """
  if not task.suite:
    raise ValueError(f"Task {task.name!r} has no suite.")
  suite_tasks = _TASK_REGISTRY.setdefault(_normalize_key(task.suite), [])
  taken: set[str] = set()
  for t in suite_tasks:
    taken.update((t.key, *t.alias_keys))
  clashes = sorted(taken.intersection((task.key, *task.alias_keys)))
  if clashes:
    raise ValueError(
        f"Task {task.name!r}: {clashes} already taken in suite {task.suite!r}."
    )
  suite_tasks.append(task)


# ── Hypothesis Registry ───────────────────────────────────────────────────────


def register_hypotheses(
    suite_name: str,
    hypotheses: Sequence[data_classes.Hypothesis],
) -> None:
  """Registers candidate hypotheses under a suite name."""
  suite_key = _normalize_key(suite_name)
  _HYPOTHESIS_REGISTRY[suite_key] = list(hypotheses)


def get_hypotheses(
    suite: str = "all",
    name: str = "all",
) -> list[data_classes.Hypothesis]:
  """Returns hypotheses filtered by suite and/or name.

  Args:
    suite: Suite name (e.g. 'visual_symmetry') or 'all'.
    name: Hypothesis name / key or 'all'.

  Returns:
    List of matching Hypothesis objects.
  """
  suite_key = _normalize_key(suite)
  name_key = _normalize_key(name)

  if suite_key == "all":
    all_hyps: list[data_classes.Hypothesis] = []
    for s_hyps in _HYPOTHESIS_REGISTRY.values():
      all_hyps.extend(s_hyps)
    candidate_hyps = all_hyps
  else:
    if suite_key not in _HYPOTHESIS_REGISTRY:
      raise ValueError(
          f"Unknown hypothesis suite: {suite!r}. Available:"
          f" {list(_HYPOTHESIS_REGISTRY.keys())}"
      )
    candidate_hyps = list(_HYPOTHESIS_REGISTRY[suite_key])

  if name_key == "all":
    return candidate_hyps

  exact_matches = [h for h in candidate_hyps if h.key == name_key]
  if exact_matches:
    return exact_matches

  substr_matches = [h for h in candidate_hyps if name_key in h.key]
  if substr_matches:
    _warn_substring(
        f"get_hypotheses(suite={suite!r})",
        name,
        [h.key for h in substr_matches],
    )
    return substr_matches

  raise ValueError(
      f"No hypotheses found for suite={suite!r} and name={name!r}."
  )


def list_hypotheses(suite: Optional[str] = None) -> list[str]:
  """Returns normalized keys of all available hypotheses."""
  suite_name = suite if suite else "all"
  return [h.key for h in get_hypotheses(suite=suite_name, name="all")]


# Register default task suites, from tasks/data/<suite>.yaml. These are the
# same TaskSpec objects as the task modules' constants (the loader caches).
for _suite_name in loader.BUILTIN_SUITES:
  _suite_data = loader.load_suite(_suite_name)
  register_suite(_suite_name, _suite_data.tasks)
  _DEFAULT_TASKS[_suite_name] = _suite_data.default_task

# Register default hypotheses
register_hypotheses("recipes", recipe_tasks.RECIPE_HYPOTHESES)
register_hypotheses("hedging", hedging_tasks.HEDGING_HYPOTHESES)
register_hypotheses(
    "visual_symmetry", visual_symmetry_hypotheses.VISUAL_SYMMETRY_HYPOTHESES
)


def _convert_t2i_hypothesis(h) -> data_classes.Hypothesis:
  """Converts a T2I custom hypothesis dataclass to a standard Hypothesis."""
  return data_classes.Hypothesis(
      name=h.name,
      claim=h.claim,
      task_suite=h.task_name,
      null_proportion=h.null_proportion,
      condition={
          "test_type": h.test_type,
          "alternative": h.alternative,
      },
      metadata=h.metadata,
  )


register_hypotheses(
    "t2i_cat_bias",
    [
        _convert_t2i_hypothesis(h)
        for h in t2i_cat_bias_tasks.T2I_CAT_BIAS_HYPOTHESES
    ],
)
register_hypotheses(
    "t2i_handedness_bias",
    [
        _convert_t2i_hypothesis(h)
        for h in t2i_handedness_bias_tasks.T2I_HANDEDNESS_BIAS_HYPOTHESES
    ],
)
