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

"""DISCO benchmark tasks and hypotheses package.

Provides task suites and hypotheses for evaluating behavior discovery
techniques (DISCO and baselines).
"""

from disco.tasks import data_classes
from disco.tasks import hedging_tasks
from disco.tasks import recipe_tasks
from disco.tasks import registry
from disco.tasks import spec
from disco.tasks import t2i_cat_bias_tasks
from disco.tasks import t2i_handedness_bias_tasks
from disco.tasks.visual_symmetry import hypotheses as visual_symmetry_hypotheses

TaskSpec = spec.TaskSpec
Task = data_classes.Task
Hypothesis = data_classes.Hypothesis

RECIPE_TASKS = recipe_tasks.RECIPE_TASKS
RECIPE_HYPOTHESES = recipe_tasks.RECIPE_HYPOTHESES
RECIPE_TRANSLATE_FRENCH_LONG_TASK = (
    recipe_tasks.RECIPE_TRANSLATE_FRENCH_LONG_TASK
)
RECIPE_TRANSLATE_ENGLISH_LONG_TASK = (
    recipe_tasks.RECIPE_TRANSLATE_ENGLISH_LONG_TASK
)

HEDGING_TASKS = hedging_tasks.HEDGING_TASKS
HEDGING_HYPOTHESES = hedging_tasks.HEDGING_HYPOTHESES
HEDGING_UNHEDGED_CONTROVERSIAL_TASK = (
    hedging_tasks.HEDGING_UNHEDGED_CONTROVERSIAL_TASK
)
HEDGING_DEFENSIVE_DISCLAIMERS_TASK = (
    hedging_tasks.HEDGING_DEFENSIVE_DISCLAIMERS_TASK
)
HEDGING_PAIRED_AMBIGUITY_TASK = hedging_tasks.HEDGING_PAIRED_AMBIGUITY_TASK

VISUAL_SYMMETRY_TASK = visual_symmetry_hypotheses.VISUAL_SYMMETRY_TASK
VISUAL_SYMMETRY_HYPOTHESES = (
    visual_symmetry_hypotheses.VISUAL_SYMMETRY_HYPOTHESES
)

get_recipe_tasks = recipe_tasks.get_recipe_tasks
get_recipe_hypotheses = recipe_tasks.get_recipe_hypotheses

get_hedging_tasks = hedging_tasks.get_hedging_tasks
get_hedging_hypotheses = hedging_tasks.get_hedging_hypotheses

get_visual_symmetry_hypotheses = (
    visual_symmetry_hypotheses.get_visual_symmetry_hypotheses
)

T2I_CAT_BIAS_TASKS = t2i_cat_bias_tasks.T2I_CAT_BIAS_TASKS
T2ICatBiasHypothesis = t2i_cat_bias_tasks.T2ICatBiasHypothesis
T2I_CAT_BIAS_HYPOTHESES = t2i_cat_bias_tasks.T2I_CAT_BIAS_HYPOTHESES
ORANGE_CAT_BIAS_SEED = t2i_cat_bias_tasks.ORANGE_CAT_BIAS_SEED
get_t2i_cat_bias_tasks = t2i_cat_bias_tasks.get_t2i_cat_bias_tasks
get_t2i_cat_bias_hypotheses = t2i_cat_bias_tasks.get_t2i_cat_bias_hypotheses

T2I_HANDEDNESS_BIAS_TASKS = t2i_handedness_bias_tasks.T2I_HANDEDNESS_BIAS_TASKS
T2IHandednessBiasHypothesis = (
    t2i_handedness_bias_tasks.T2IHandednessBiasHypothesis
)
T2I_HANDEDNESS_BIAS_HYPOTHESES = (
    t2i_handedness_bias_tasks.T2I_HANDEDNESS_BIAS_HYPOTHESES
)
LEFT_HAND_BIAS_SEED = t2i_handedness_bias_tasks.LEFT_HAND_BIAS_SEED
get_t2i_handedness_bias_tasks = (
    t2i_handedness_bias_tasks.get_t2i_handedness_bias_tasks
)
get_t2i_handedness_bias_hypotheses = (
    t2i_handedness_bias_tasks.get_t2i_handedness_bias_hypotheses
)

# Task registry functions
get_task = registry.get_task
get_tasks = registry.get_tasks
get_task_descriptions = registry.get_task_descriptions
get_suite = registry.get_suite
list_suites = registry.list_suites
list_tasks = registry.list_tasks
register_suite = registry.register_suite

# Hypothesis registry functions
get_hypotheses = registry.get_hypotheses
list_hypotheses = registry.list_hypotheses
register_hypotheses = registry.register_hypotheses

__all__ = [
    "TaskSpec",
    "Task",
    "Hypothesis",
    "RECIPE_TASKS",
    "RECIPE_HYPOTHESES",
    "RECIPE_TRANSLATE_FRENCH_LONG_TASK",
    "RECIPE_TRANSLATE_ENGLISH_LONG_TASK",
    "HEDGING_TASKS",
    "HEDGING_HYPOTHESES",
    "HEDGING_UNHEDGED_CONTROVERSIAL_TASK",
    "HEDGING_DEFENSIVE_DISCLAIMERS_TASK",
    "HEDGING_PAIRED_AMBIGUITY_TASK",
    "VISUAL_SYMMETRY_TASK",
    "VISUAL_SYMMETRY_HYPOTHESES",
    "T2I_CAT_BIAS_TASKS",
    "T2ICatBiasHypothesis",
    "T2I_CAT_BIAS_HYPOTHESES",
    "ORANGE_CAT_BIAS_SEED",
    "get_recipe_tasks",
    "get_recipe_hypotheses",
    "get_hedging_tasks",
    "get_hedging_hypotheses",
    "get_visual_symmetry_hypotheses",
    "get_t2i_cat_bias_tasks",
    "get_t2i_cat_bias_hypotheses",
    "T2I_HANDEDNESS_BIAS_TASKS",
    "T2IHandednessBiasHypothesis",
    "T2I_HANDEDNESS_BIAS_HYPOTHESES",
    "LEFT_HAND_BIAS_SEED",
    "get_t2i_handedness_bias_tasks",
    "get_t2i_handedness_bias_hypotheses",
    "get_task",
    "get_tasks",
    "get_task_descriptions",
    "get_suite",
    "list_suites",
    "list_tasks",
    "register_suite",
    "get_hypotheses",
    "list_hypotheses",
    "register_hypotheses",
]
