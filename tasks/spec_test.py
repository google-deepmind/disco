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

"""Tests for TaskSpec, the YAML task loader and the typed registry lookups."""

import dataclasses

from absl.testing import absltest

from disco.tasks import data_classes
from disco.tasks import hedging_tasks
from disco.tasks import loader
from disco.tasks import recipe_tasks
from disco.tasks import registry
from disco.tasks import spec
from disco.tasks import t2i_cat_bias_tasks
from disco.tasks.visual_symmetry import hypotheses as visual_symmetry_hypotheses

_LEGACY_METADATA = {
    "seed_instruction": "Seed.",
    "checker_rules": ["Rule 1.", "Rule 2."],
    "procedural_autorater": "count_hedges",
    "autorater_params": {},
    "null_proportion": 0.1,
    "domain": "testing",
    "modalities": ["text"],
    "aliases": ["legacy_alias"],
}


class TaskSpecTest(absltest.TestCase):

  def test_task_is_task_spec(self):
    self.assertIs(data_classes.Task, spec.TaskSpec)

  def test_legacy_metadata_round_trips_exactly(self):
    task = data_classes.Task(
        name="Legacy Task",
        description="Description.",
        split="evaluation",
        suite="custom",
        metadata=_LEGACY_METADATA,
    )
    self.assertEqual(task.metadata, _LEGACY_METADATA)
    self.assertEqual(list(task.metadata), list(_LEGACY_METADATA))
    self.assertEqual(task.seed_instruction, "Seed.")
    self.assertEqual(task.autorater_params, {})
    self.assertEqual(task.aliases, ["legacy_alias"])
    self.assertEqual(task.extra, {"domain": "testing"})
    self.assertIsNone(task.target_model)
    self.assertEqual(
        task.to_dict(),
        {
            "name": "Legacy Task",
            "description": "Description.",
            "split": "evaluation",
            "suite": "custom",
            "metadata": _LEGACY_METADATA,
        },
    )

  def test_legacy_positional_construction(self):
    task = data_classes.Task("A B-c", "Description.", "eval", "s", {"x": 1})
    self.assertEqual(task.key, "a_b_c")
    self.assertEqual(task.metadata, {"x": 1})

  def test_empty_metadata_is_omitted_from_to_dict(self):
    task = data_classes.Task(name="Bare", description="Description.")
    self.assertEqual(task.metadata, {})
    self.assertEqual(
        task.to_dict(),
        {"name": "Bare", "description": "Description.", "split": "all"},
    )

  def test_typed_construction_derives_metadata(self):
    task = spec.TaskSpec(
        name="Typed",
        description="Description.",
        suite="custom",
        null_proportion=0.5,
        seed_instruction="Seed.",
        extra={"domain": "d"},
    )
    self.assertEqual(
        task.metadata,
        {"seed_instruction": "Seed.", "null_proportion": 0.5, "domain": "d"},
    )

  def test_spec_only_fields_stay_out_of_metadata(self):
    task = spec.TaskSpec(
        name="Typed",
        description="Description.",
        recommended_seed="Seed.",
        adapter="recipe",
    )
    self.assertEqual(task.metadata, {})

  def test_metadata_is_a_copy(self):
    task = data_classes.Task(name="T", description="D", metadata={"x": 1})
    task.metadata["x"] = 2
    self.assertEqual(task.metadata, {"x": 1})

  def test_replace_with_metadata(self):
    task = hedging_tasks.HEDGING_UNHEDGED_CONTROVERSIAL_TASK
    seedless = dataclasses.replace(
        task, metadata={**task.metadata, "seed_instruction": ""}
    )
    self.assertEqual(seedless.seed_instruction, "")
    self.assertEqual(
        seedless.metadata, {**task.metadata, "seed_instruction": ""}
    )
    dropped = dataclasses.replace(task, metadata={"domain": "d"})
    self.assertIsNone(dropped.seed_instruction)
    self.assertEqual(dropped.metadata, {"domain": "d"})

  def test_replace_typed_field(self):
    task = recipe_tasks.RECIPE_TRANSLATE_FRENCH_LONG_TASK
    changed = dataclasses.replace(task, null_proportion=0.25)
    self.assertEqual(changed.metadata["null_proportion"], 0.25)
    self.assertEqual(list(changed.metadata), list(task.metadata))

  def test_inconsistent_fields_raise(self):
    with self.assertRaisesRegex(ValueError, "extra repeats typed fields"):
      spec.TaskSpec(name="T", description="D", extra={"aliases": []})
    with self.assertRaisesRegex(ValueError, "metadata_keys"):
      spec.TaskSpec(
          name="T", description="D", seed_instruction="S", metadata_keys=()
      )
    with self.assertRaisesRegex(ValueError, "metadata_keys"):
      spec.TaskSpec(name="T", description="D", metadata_keys=("missing",))


class LoaderTest(absltest.TestCase):

  def test_loaded_metadata_keeps_the_literal_key_order(self):
    self.assertEqual(
        list(hedging_tasks.HEDGING_UNHEDGED_CONTROVERSIAL_TASK.metadata),
        [
            "seed_instruction",
            "checker_rules",
            "procedural_autorater",
            "autorater_params",
            "null_proportion",
            "target_model",
            "domain",
            "modalities",
            "aliases",
        ],
    )

  def test_t2i_seed_is_recommended_seed_not_seed_instruction(self):
    task = t2i_cat_bias_tasks.T2I_CAT_BIAS_TASKS[0]
    self.assertEqual(
        task.recommended_seed, t2i_cat_bias_tasks.ORANGE_CAT_BIAS_SEED
    )
    self.assertIsNone(task.seed_instruction)
    self.assertNotIn("seed_instruction", task.metadata)
    self.assertNotIn("recommended_seed", task.metadata)
    # The population baseline is the task's null_proportion (0.2 orange cats),
    # which the verifier and the baseline adapter read.
    self.assertEqual(task.null_proportion, 0.2)
    self.assertEqual(task.metadata["null_proportion"], 0.2)

  def test_module_constants_are_views_of_the_specs(self):
    task = hedging_tasks.HEDGING_UNHEDGED_CONTROVERSIAL_TASK
    self.assertIs(
        hedging_tasks.HEDGING_UNHEDGED_CONTROVERSIAL_SEED, task.seed_instruction
    )
    self.assertIs(
        hedging_tasks.HEDGING_UNHEDGED_CONTROVERSIAL_CHECKER_RULES,
        task.checker_rules,
    )
    for i, t in enumerate(registry.get_suite("hedging")):
      self.assertIs(t, hedging_tasks.HEDGING_TASKS[i])

  def test_visual_symmetry_seed_mentions_symmetry(self):
    self.assertIn(
        "symmetry", visual_symmetry_hypotheses.DEFAULT_SEED_INSTRUCTION
    )

  def test_parse_suite_validates(self):
    good = (
        "suite: s\ndefault_task: a\ntasks:\n"
        "  - name: A\n    description: d\n    domain: x\n"
    )
    data = loader.parse_suite(good)
    self.assertEqual(data.default_task, "a")
    self.assertEqual(data.tasks[0].metadata, {"domain": "x"})
    self.assertEqual(data.tasks[0].suite, "s")
    with self.assertRaisesRegex(ValueError, "default_task"):
      loader.parse_suite(good.replace("default_task: a", "default_task: b"))
    with self.assertRaisesRegex(ValueError, "duplicate task keys"):
      loader.parse_suite(good + "  - name: a\n    description: d\n")
    with self.assertRaisesRegex(ValueError, "suite from the file"):
      loader.parse_suite(good + "    suite: other\n")
    with self.assertRaisesRegex(ValueError, "exactly the keys"):
      loader.parse_suite(good + "extra_key: 1\n")


class RegistryTest(absltest.TestCase):

  def test_get_by_key_and_alias(self):
    paired = hedging_tasks.HEDGING_PAIRED_AMBIGUITY_TASK
    self.assertIs(registry.get("hedging_paired_ambiguity_modulation"), paired)
    self.assertIs(registry.get("Hedging Paired Ambiguity Modulation"), paired)
    self.assertIs(registry.get("hedging-paired"), paired)
    self.assertIs(registry.get("hedging_modulation", suite="hedging"), paired)

  def test_get_has_no_substring_fallback(self):
    with self.assertRaises(KeyError):
      registry.get("paired_ambiguity")

  def test_get_miss_suggests_close_names(self):
    with self.assertRaisesRegex(KeyError, "hedging_unhedged"):
      registry.get("hedging_unhedge")
    with self.assertRaisesRegex(KeyError, "in suite 'recipes'"):
      registry.get("orange_cat_bias", suite="recipes")
    with self.assertRaisesRegex(KeyError, "hedging"):
      registry.get("orange_cat_bias", suite="hedgin")

  def test_resolve_suite_gives_default_task(self):
    for suite in loader.BUILTIN_SUITES:
      with self.subTest(suite=suite):
        self.assertIs(registry.resolve(suite), registry.get_suite(suite)[0])
    self.assertIs(
        registry.resolve("visual_symmetry"),
        visual_symmetry_hypotheses.VISUAL_SYMMETRY_TASK,
    )
    self.assertEqual(registry.resolve("orange_cat_bias").key, "orange_cat_bias")
    with self.assertRaises(KeyError):
      registry.resolve("no_such_task_or_suite")

  def test_list_specs_filters(self):
    self.assertLen(registry.list_specs("recipes", "evaluation"), 2)
    self.assertEqual(
        registry.list_specs(split="evaluation"),
        registry.get_tasks(suite="all", split="evaluation"),
    )
    self.assertEqual(registry.list_specs(), registry.get_tasks())
    self.assertEqual(registry.suites(), registry.list_suites())
    with self.assertRaises(KeyError):
      registry.list_specs("no_such_suite")

  def test_register(self):
    task = spec.TaskSpec(
        name="Spec Test Task",
        description="D",
        suite="spec_test_suite",
        aliases=["spec_test_alias"],
    )
    registry.register(task)
    self.assertIs(registry.get("spec_test_alias"), task)
    self.assertIs(registry.resolve("spec_test_suite"), task)
    with self.assertRaisesRegex(ValueError, "already taken"):
      registry.register(dataclasses.replace(task, aliases=None))
    with self.assertRaisesRegex(ValueError, "no suite"):
      registry.register(spec.TaskSpec(name="X", description="D"))

  def test_substring_fallback_still_works_and_warns(self):
    paired = hedging_tasks.HEDGING_PAIRED_AMBIGUITY_TASK
    with self.assertLogs(logger="absl", level="WARNING") as logs:
      self.assertEqual(
          hedging_tasks.get_hedging_tasks("paired_ambiguity"), [paired]
      )
      self.assertEqual(
          registry.get_tasks(suite="hedging", split="paired_ambiguity"),
          [paired],
      )
      self.assertLen(
          registry.get_hypotheses(suite="visual_symmetry", name="chinese"), 1
      )
    self.assertLen(logs.records, 3)
    for record in logs.records:
      self.assertIn("substring", record.getMessage())

  def test_exact_lookups_do_not_warn(self):
    with self.assertNoLogs(logger="absl", level="WARNING"):
      registry.get_tasks(suite="hedging", split="hedging_paired")
      registry.get_task("orange_cat_bias")
      hedging_tasks.get_hedging_tasks("hedging_paired")

  def test_every_registered_task_names_its_adapter(self):
    expected = {
        "recipes": "recipe",
        "hedging": "hedging",
        "visual_symmetry": "visual_symmetry",
        "t2i_cat_bias": "t2i_bias",
        "t2i_handedness_bias": "t2i_bias",
    }
    for suite in registry.list_suites():
      if suite not in expected:
        continue
      for task in registry.list_specs(suite):
        with self.subTest(task=task.key):
          self.assertEqual(task.adapter, expected[suite])
          self.assertNotIn("adapter", task.metadata)


if __name__ == "__main__":
  absltest.main()
