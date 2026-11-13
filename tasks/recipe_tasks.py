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

"""Recipe translation benchmark tasks and hypotheses for DISCO.

Defines:
1. The ``RECIPE_TASKS`` — evaluating behavioral response length (word count)
   differences when generating recipes for the same dish in English vs. French.
   - ``Recipe Translate French Long`` (reverse direction, 10% null baseline):
     identifying dish types where English recipe responses are shorter than
     French recipe responses.
   - ``Recipe Translate English Long`` (forward direction, 60% null baseline):
     identifying dish types where English recipe responses are longer than
     French recipe responses.
2. Reference candidate ``RECIPE_HYPOTHESES`` — assertions about specific dish
   categories across the recipe search space.

The tasks are data in ``data/recipes.yaml``; the seed, checker-rule and task
constants below are derived from it.
"""

from disco.tasks import data_classes
from disco.tasks import loader
from disco.tasks import spec

# ── Tasks ─────────────────────────────────────────────────────────────────────

_SUITE = loader.load_suite("recipes")

RECIPE_TRANSLATE_FRENCH_LONG_TASK: data_classes.Task = _SUITE.by_key(
    "recipe_translate_french_long"
)
RECIPE_TRANSLATE_ENGLISH_LONG_TASK: data_classes.Task = _SUITE.by_key(
    "recipe_translate_english_long"
)

RECIPE_TASKS: list[data_classes.Task] = list(_SUITE.tasks)

TASKS_BY_KEY: dict[str, data_classes.Task] = spec.index_by_key(
    RECIPE_TASKS, skip_aliases=frozenset({"recipe_translate_iso"})
)

# ── Seed Instructions ─────────────────────────────────────────────────────────

RECIPE_TRANSLATE_FRENCH_LONG_SEED: str = spec.require(
    RECIPE_TRANSLATE_FRENCH_LONG_TASK.seed_instruction, "seed_instruction"
)
RECIPE_TRANSLATE_ENGLISH_LONG_SEED: str = spec.require(
    RECIPE_TRANSLATE_ENGLISH_LONG_TASK.seed_instruction, "seed_instruction"
)

# ── Checker Rules ─────────────────────────────────────────────────────────────

RECIPE_TRANSLATE_FRENCH_LONG_CHECKER_RULES: list[str] = spec.require(
    RECIPE_TRANSLATE_FRENCH_LONG_TASK.checker_rules, "checker_rules"
)
RECIPE_TRANSLATE_ENGLISH_LONG_CHECKER_RULES: list[str] = spec.require(
    RECIPE_TRANSLATE_ENGLISH_LONG_TASK.checker_rules, "checker_rules"
)

# ── Reference Candidate Hypotheses ────────────────────────────────────────────

RECIPE_HYPOTHESES: list[data_classes.Hypothesis] = [
    data_classes.Hypothesis(
        name="French Pastries Shorter in English",
        claim=(
            "Recipes for traditional French pastries and desserts (e.g. éclair,"
            " soufflé, mille-feuille) elicit shorter responses in English than"
            " in French more than 10% of the time."
        ),
        task_suite="recipes",
        condition={
            "dish_category": "pastries_and_desserts",
            "cuisine": "french",
            "comparison": "english_vs_french",
            "direction": "english_shorter",
        },
        null_proportion=0.10,
    ),
    data_classes.Hypothesis(
        name="Classical French Sauces Shorter in English",
        claim=(
            "Recipes for classical French sauces (e.g. béchamel, hollandaise,"
            " velouté) elicit shorter responses in English than in French more"
            " than 10% of the time."
        ),
        task_suite="recipes",
        condition={
            "dish_category": "sauces",
            "cuisine": "french",
            "comparison": "english_vs_french",
            "direction": "english_shorter",
        },
        null_proportion=0.10,
    ),
    data_classes.Hypothesis(
        name="Fast Food Recipes Longer in English",
        claim=(
            "Recipes for American fast food and casual dishes (e.g. burgers,"
            " nachos, hot dogs) elicit longer responses in English than in"
            " French more than 60% of the time."
        ),
        task_suite="recipes",
        condition={
            "dish_category": "fast_food",
            "cuisine": "american",
            "comparison": "english_vs_french",
            "direction": "english_longer",
        },
        null_proportion=0.60,
    ),
]

HYPOTHESES_BY_KEY: dict[str, data_classes.Hypothesis] = {
    h.key: h for h in RECIPE_HYPOTHESES
}

# ── Convenience Accessors ─────────────────────────────────────────────────────


def get_recipe_tasks(split_or_name: str = "all") -> list[data_classes.Task]:
  """Returns Recipe tasks filtered by split or task name."""
  query_lower = spec.normalize_key(split_or_name)
  if query_lower in ("all", "evaluation", "eval"):
    return list(RECIPE_TASKS)
  matched = spec.legacy_match(
      query_lower, TASKS_BY_KEY, RECIPE_TASKS, "get_recipe_tasks"
  )
  if matched:
    return matched
  raise ValueError(
      f"Unknown Recipe task or split: {split_or_name!r}. Available:"
      f" {list(TASKS_BY_KEY.keys())}"
  )


def get_recipe_hypotheses(
    name_or_key: str = "all",
) -> list[data_classes.Hypothesis]:
  """Returns Recipe reference hypotheses, optionally filtered."""
  query_lower = spec.normalize_key(name_or_key)
  if query_lower == "all":
    return list(RECIPE_HYPOTHESES)
  matched = spec.legacy_match(
      query_lower, HYPOTHESES_BY_KEY, RECIPE_HYPOTHESES, "get_recipe_hypotheses"
  )
  if matched:
    return matched
  raise ValueError(
      f"Unknown Recipe hypothesis: {name_or_key!r}. Available:"
      f" {list(HYPOTHESES_BY_KEY.keys())}"
  )
