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

"""Golden cases pinning the public task constants and the legacy task lookups.

Pins, per task module, every public module-level constant derived from the task
data (seeds, checker rules, task lists, key -> task maps), plus the results (or
error messages) of the per-module accessors and the legacy registry lookups for
a fixed set of queries.

Maps whose iteration order is observable (they are listed in error messages)
are pinned as ordered `[key, task_key]` pairs, because golden files are written
with sorted keys.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from disco.tasks import hedging_tasks
from disco.tasks import recipe_tasks
from disco.tasks import registry
from disco.tasks import t2i_cat_bias_tasks
from disco.tasks import t2i_handedness_bias_tasks
from disco.tasks.visual_symmetry import hypotheses as visual_symmetry_hypotheses

# Queries run through every task accessor: split words, every key and alias
# (added below), substrings that only the substring fallback resolves, and
# misses.
_EXTRA_TASK_QUERIES: tuple[str, ...] = (
    "all",
    "iteration",
    "iter",
    "evaluation",
    "eval",
    "test",
    "paired_ambiguity",
    "unhedged",
    "hedging",
    "recipe",
    "recipe_translate",
    "french",
    "cat",
    "orange_cat",
    "hand",
    "bias",
    "symmetry",
    "visual_symmetry_completion",
    "nonexistent_task_xyz",
)

_HYPOTHESIS_QUERIES: tuple[str, ...] = (
    "all",
    "symmetry_english_letters_1to1",
    "Symmetry English Letters 1to1",
    "symmetry",
    "chinese",
    "unhedged_controversial_nutrition_claims",
    "nutrition",
    "defensive_disclaimers",
    "french_pastries_shorter_in_english",
    "french",
    "orange_cat_baseline_frequency",
    "orange_cat",
    "left_hand_baseline_frequency",
    "left_hand",
    "nonexistent_hypothesis_xyz",
)


def _task_dicts(tasks: Sequence[Any]) -> list[dict[str, Any]]:
  return [t.to_dict() for t in tasks]


def _key_pairs(by_key: Mapping[str, Any]) -> list[list[str]]:
  return [[k, v.key] for k, v in by_key.items()]


def _keys_or_error(fn: Callable[[], Any]) -> Any:
  """Returns the keys `fn` returns, or `<ExceptionType>: <message>`.

  Strings (descriptions, keys) are returned as is; tasks and hypotheses are
  reported by key.

  Args:
    fn: Lookup to run.
  """
  try:
    result = fn()
  except (KeyError, ValueError) as e:
    return f"{type(e).__name__}: {e}"
  if isinstance(result, list):
    return [r if isinstance(r, str) else r.key for r in result]
  return result.key


def _all_task_queries() -> list[str]:
  """Returns every task key and alias, followed by the extra queries."""
  queries: list[str] = []
  for suite in registry.list_suites():
    for task in registry.get_suite(suite):
      queries.append(task.key)
      queries.extend(task.metadata.get("aliases", []))
  for query in _EXTRA_TASK_QUERIES:
    if query not in queries:
      queries.append(query)
  return queries


def module_constants() -> dict[str, Any]:
  """Returns the public task constants of each task module."""
  return {
      "hedging_tasks": {
          "HEDGING_UNHEDGED_CONTROVERSIAL_SEED": (
              hedging_tasks.HEDGING_UNHEDGED_CONTROVERSIAL_SEED
          ),
          "HEDGING_DEFENSIVE_DISCLAIMERS_SEED": (
              hedging_tasks.HEDGING_DEFENSIVE_DISCLAIMERS_SEED
          ),
          "HEDGING_PAIRED_AMBIGUITY_SEED": (
              hedging_tasks.HEDGING_PAIRED_AMBIGUITY_SEED
          ),
          "HEDGING_UNHEDGED_CONTROVERSIAL_CHECKER_RULES": (
              hedging_tasks.HEDGING_UNHEDGED_CONTROVERSIAL_CHECKER_RULES
          ),
          "HEDGING_DEFENSIVE_DISCLAIMERS_CHECKER_RULES": (
              hedging_tasks.HEDGING_DEFENSIVE_DISCLAIMERS_CHECKER_RULES
          ),
          "HEDGING_PAIRED_AMBIGUITY_CHECKER_RULES": (
              hedging_tasks.HEDGING_PAIRED_AMBIGUITY_CHECKER_RULES
          ),
          "HEDGING_UNHEDGED_CONTROVERSIAL_TASK": (
              hedging_tasks.HEDGING_UNHEDGED_CONTROVERSIAL_TASK.to_dict()
          ),
          "HEDGING_DEFENSIVE_DISCLAIMERS_TASK": (
              hedging_tasks.HEDGING_DEFENSIVE_DISCLAIMERS_TASK.to_dict()
          ),
          "HEDGING_PAIRED_AMBIGUITY_TASK": (
              hedging_tasks.HEDGING_PAIRED_AMBIGUITY_TASK.to_dict()
          ),
          "HEDGING_TASKS": _task_dicts(hedging_tasks.HEDGING_TASKS),
          "TASKS_BY_KEY": _key_pairs(hedging_tasks.TASKS_BY_KEY),
      },
      "recipe_tasks": {
          "RECIPE_TRANSLATE_FRENCH_LONG_SEED": (
              recipe_tasks.RECIPE_TRANSLATE_FRENCH_LONG_SEED
          ),
          "RECIPE_TRANSLATE_ENGLISH_LONG_SEED": (
              recipe_tasks.RECIPE_TRANSLATE_ENGLISH_LONG_SEED
          ),
          "RECIPE_TRANSLATE_FRENCH_LONG_CHECKER_RULES": (
              recipe_tasks.RECIPE_TRANSLATE_FRENCH_LONG_CHECKER_RULES
          ),
          "RECIPE_TRANSLATE_ENGLISH_LONG_CHECKER_RULES": (
              recipe_tasks.RECIPE_TRANSLATE_ENGLISH_LONG_CHECKER_RULES
          ),
          "RECIPE_TRANSLATE_FRENCH_LONG_TASK": (
              recipe_tasks.RECIPE_TRANSLATE_FRENCH_LONG_TASK.to_dict()
          ),
          "RECIPE_TRANSLATE_ENGLISH_LONG_TASK": (
              recipe_tasks.RECIPE_TRANSLATE_ENGLISH_LONG_TASK.to_dict()
          ),
          "RECIPE_TASKS": _task_dicts(recipe_tasks.RECIPE_TASKS),
          "TASKS_BY_KEY": _key_pairs(recipe_tasks.TASKS_BY_KEY),
      },
      "t2i_cat_bias_tasks": {
          "ORANGE_CAT_BIAS_SEED": t2i_cat_bias_tasks.ORANGE_CAT_BIAS_SEED,
          "T2I_CAT_BIAS_TASKS": _task_dicts(
              t2i_cat_bias_tasks.T2I_CAT_BIAS_TASKS
          ),
          "TASKS_BY_NAME": _key_pairs(t2i_cat_bias_tasks.TASKS_BY_NAME),
      },
      "t2i_handedness_bias_tasks": {
          "LEFT_HAND_BIAS_SEED": t2i_handedness_bias_tasks.LEFT_HAND_BIAS_SEED,
          "T2I_HANDEDNESS_BIAS_TASKS": _task_dicts(
              t2i_handedness_bias_tasks.T2I_HANDEDNESS_BIAS_TASKS
          ),
          "TASKS_BY_NAME": _key_pairs(t2i_handedness_bias_tasks.TASKS_BY_NAME),
      },
      "visual_symmetry_hypotheses": {
          "DEFAULT_SEED_INSTRUCTION": (
              visual_symmetry_hypotheses.DEFAULT_SEED_INSTRUCTION
          ),
          "VISUAL_SYMMETRY_TASK": (
              visual_symmetry_hypotheses.VISUAL_SYMMETRY_TASK.to_dict()
          ),
      },
  }


def accessor_lookups() -> dict[str, Any]:
  """Returns per-module accessor results for every task/hypothesis query."""
  task_accessors: dict[str, Callable[[str], Any]] = {
      "get_hedging_tasks": hedging_tasks.get_hedging_tasks,
      "get_recipe_tasks": recipe_tasks.get_recipe_tasks,
      "get_t2i_cat_bias_tasks": t2i_cat_bias_tasks.get_t2i_cat_bias_tasks,
      "get_t2i_handedness_bias_tasks": (
          t2i_handedness_bias_tasks.get_t2i_handedness_bias_tasks
      ),
  }
  hypothesis_accessors: dict[str, Callable[[str], Any]] = {
      "get_hedging_hypotheses": hedging_tasks.get_hedging_hypotheses,
      "get_recipe_hypotheses": recipe_tasks.get_recipe_hypotheses,
      "get_t2i_cat_bias_hypotheses": (
          t2i_cat_bias_tasks.get_t2i_cat_bias_hypotheses
      ),
      "get_t2i_handedness_bias_hypotheses": (
          t2i_handedness_bias_tasks.get_t2i_handedness_bias_hypotheses
      ),
      "get_visual_symmetry_hypotheses": (
          visual_symmetry_hypotheses.get_visual_symmetry_hypotheses
      ),
  }
  result: dict[str, Any] = {}
  for name, fn in task_accessors.items():
    for query in _all_task_queries():
      result[f"{name}({query})"] = _keys_or_error(lambda f=fn, q=query: f(q))
  for name, fn in hypothesis_accessors.items():
    for query in _HYPOTHESIS_QUERIES:
      result[f"{name}({query})"] = _keys_or_error(lambda f=fn, q=query: f(q))
  return result


def registry_lookups() -> dict[str, Any]:
  """Returns legacy registry lookup results for every task/hypothesis query."""
  result: dict[str, Any] = {
      "list_suites()": registry.list_suites(),
      "list_tasks()": registry.list_tasks(),
  }
  suites = ["all", *registry.list_suites(), "unknown_suite"]
  for suite in suites:
    for query in _all_task_queries():
      result[f"get_tasks({suite},{query})"] = _keys_or_error(
          lambda s=suite, q=query: registry.get_tasks(suite=s, split=q)
      )
    result[f"get_task_descriptions({suite})"] = _keys_or_error(
        lambda s=suite: registry.get_task_descriptions(suite=s)
    )
    result[f"get_suite({suite})"] = _keys_or_error(
        lambda s=suite: registry.get_suite(s)
    )
    result[f"list_tasks({suite})"] = _keys_or_error(
        lambda s=suite: registry.list_tasks(s)
    )
  for query in _all_task_queries():
    result[f"get_task({query})"] = _keys_or_error(
        lambda q=query: registry.get_task(q)
    )
  hypothesis_suites = [
      "all",
      "recipes",
      "hedging",
      "visual_symmetry",
      "t2i_cat_bias",
      "t2i_handedness_bias",
  ]
  for suite in hypothesis_suites:
    for query in _HYPOTHESIS_QUERIES:
      result[f"get_hypotheses({suite},{query})"] = _keys_or_error(
          lambda s=suite, q=query: registry.get_hypotheses(suite=s, name=q)
      )
    result[f"list_hypotheses({suite})"] = _keys_or_error(
        lambda s=suite: registry.list_hypotheses(s)
    )
  return result


def task_constants_snapshot() -> dict[str, Any]:
  """Returns the task constants and lookup results pinned by the golden."""
  return {
      "accessor_lookups": accessor_lookups(),
      "module_constants": module_constants(),
      "registry_lookups": registry_lookups(),
  }


# Golden name -> case function, in the same shape as
# stats_golden_cases.ALL_CASES.
ALL_CASES: dict[str, Callable[[], Any]] = {
    "task_constants": task_constants_snapshot,
}
