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

"""Unit tests for the DISCO tasks package."""

from absl.testing import absltest
from disco import tasks
from disco.tasks import data_classes


class TasksTest(absltest.TestCase):

  def test_task_model(self):
    task = data_classes.Task(
        name="Test Task",
        description="A test failure mode description.",
        split="test",
        suite="custom_suite",
        metadata={"difficulty": "hard"},
    )
    self.assertEqual(task.name, "Test Task")
    self.assertEqual(task.key, "test_task")
    self.assertEqual(task.split, "test")
    self.assertEqual(task.suite, "custom_suite")
    d = task.to_dict()
    self.assertEqual(d["name"], "Test Task")
    self.assertEqual(d["description"], "A test failure mode description.")
    self.assertEqual(d["split"], "test")
    self.assertEqual(d["suite"], "custom_suite")
    self.assertEqual(d["metadata"], {"difficulty": "hard"})

  def test_hypothesis_model(self):
    hyp = data_classes.Hypothesis(
        name="Test Hypothesis",
        claim="The model succeeds under condition X >= 75% of the time.",
        task_suite="visual_symmetry",
        condition={"shape": "square", "size": 10},
        null_proportion=0.75,
    )
    self.assertEqual(hyp.name, "Test Hypothesis")
    self.assertEqual(hyp.key, "test_hypothesis")
    self.assertEqual(hyp.task_suite, "visual_symmetry")
    self.assertEqual(hyp.null_proportion, 0.75)
    d = hyp.to_dict()
    self.assertEqual(d["name"], "Test Hypothesis")
    self.assertEqual(
        d["claim"], "The model succeeds under condition X >= 75% of the time."
    )
    self.assertEqual(d["task_suite"], "visual_symmetry")
    self.assertEqual(d["condition"], {"shape": "square", "size": 10})

  def test_registry_default_suites(self):
    suites = tasks.list_suites()
    self.assertIn("recipes", suites)
    self.assertIn("hedging", suites)
    self.assertIn("visual_symmetry", suites)
    self.assertIn("t2i_cat_bias", suites)
    self.assertIn("t2i_handedness_bias", suites)

  def test_registry_get_tasks(self):
    hedging_tasks = tasks.get_tasks(suite="hedging")
    self.assertLen(hedging_tasks, 3)

    # visual_symmetry suite has 1 overarching Task
    vs_tasks = tasks.get_tasks(suite="visual_symmetry")
    self.assertLen(vs_tasks, 1)
    self.assertEqual(vs_tasks[0].name, "Visual Symmetry Completion")

    all_tasks = tasks.get_tasks(suite="all", split="all")
    self.assertGreaterEqual(len(all_tasks), 10)

  def test_no_task_seed_instruction_contains_a_literal_percent_sign(self):
    """Task seed instructions spell out 'percent' instead of using '%'.

    `stats_utils.extract_null_proportion_from_seed` parses the null proportion
    via its `(%|percent)` alternation, and task seeds consistently spell out
     the word "percent" (e.g., "90 percent") to avoid shell/template expansion
    ambiguities.
    """
    offenders = []
    for task in tasks.get_tasks(suite="all", split="all"):
      seed = task.metadata.get("seed_instruction", "")
      if isinstance(seed, str) and "%" in seed:
        offenders.append(f"{task.suite}/{task.name}")
    self.assertEmpty(
        offenders,
        "These task seeds contain a literal '%'. Write the word 'percent'"
        f" instead: {offenders}",
    )

  def test_registry_get_task(self):
    task = tasks.get_task("recipe_translate_french_long", suite="recipes")
    self.assertEqual(task.name, "Recipe Translate French Long")

    # Across all suites
    task2 = tasks.get_task("recipe_translate_french_long")
    self.assertEqual(task2.name, "Recipe Translate French Long")

  def test_registry_get_task_descriptions(self):
    descs = tasks.get_task_descriptions(suite="recipes")
    self.assertLen(descs, 2)

  def test_registry_list_tasks(self):
    task_keys = tasks.list_tasks(suite="recipes")
    self.assertLen(task_keys, 2)
    self.assertIn("recipe_translate_french_long", task_keys)
    self.assertIn("recipe_translate_english_long", task_keys)

  def test_hypotheses_registry(self):
    vs_hyps = tasks.get_hypotheses("visual_symmetry")
    self.assertLen(vs_hyps, 2)

    single_hyp = tasks.get_hypotheses(
        "visual_symmetry", name="symmetry_english_letters_1to1"
    )
    self.assertLen(single_hyp, 1)
    self.assertEqual(single_hyp[0].name, "Symmetry English Letters 1to1")

    hyp_keys = tasks.list_hypotheses("visual_symmetry")
    self.assertLen(hyp_keys, 2)
    self.assertIn("symmetry_english_letters_1to1", hyp_keys)

  def test_recipe_tasks_registry(self):
    self.assertIn("recipes", tasks.list_suites())
    recipe_tasks_list = tasks.get_tasks(suite="recipes")
    self.assertLen(recipe_tasks_list, 2)

    french_long = tasks.get_task(
        "recipe_translate_french_long", suite="recipes"
    )
    self.assertEqual(french_long.name, "Recipe Translate French Long")
    self.assertEqual(french_long.metadata["null_proportion"], 0.10)
    self.assertEqual(
        french_long.metadata["procedural_autorater"], "count_words"
    )
    self.assertIn("seed_instruction", french_long.metadata)
    self.assertLen(french_long.metadata["checker_rules"], 6)

    # Test alias
    french_long_alias = tasks.get_task(
        "recipe_translate_reverse", suite="recipes"
    )
    self.assertEqual(french_long_alias.name, "Recipe Translate French Long")

    english_long = tasks.get_task(
        "recipe_translate_english_long", suite="recipes"
    )
    self.assertEqual(english_long.name, "Recipe Translate English Long")
    self.assertEqual(english_long.metadata["null_proportion"], 0.60)

  def test_recipe_hypotheses_registry(self):
    hyps = tasks.get_hypotheses(suite="recipes")
    self.assertLen(hyps, 3)

    single_hyp = tasks.get_hypotheses(
        "recipes", name="french_pastries_shorter_in_english"
    )
    self.assertLen(single_hyp, 1)
    self.assertEqual(single_hyp[0].name, "French Pastries Shorter in English")
    self.assertEqual(single_hyp[0].null_proportion, 0.10)

  def test_register_new_custom_suite(self):
    custom_tasks = [
        data_classes.Task(
            name="Format Constraint",
            description="Responses always follow format X.",
            split="eval",
            suite="custom_suite",
        ),
        data_classes.Task(
            name="Metric Units",
            description="Always use metric units.",
            split="eval",
            suite="custom_suite",
        ),
    ]
    tasks.register_suite("custom_suite", custom_tasks)

    self.assertIn("custom_suite", tasks.list_suites())
    suite_tasks = tasks.get_tasks(suite="custom_suite", split="all")
    self.assertLen(suite_tasks, 2)
    self.assertEqual(suite_tasks[0].name, "Format Constraint")

    retrieved = tasks.get_task("format_constraint", suite="custom_suite")
    self.assertEqual(retrieved.name, "Format Constraint")

  def test_t2i_cat_bias_suite_registered(self):
    suites = tasks.list_suites()
    self.assertIn("t2i_cat_bias", suites)

  def test_t2i_cat_bias_tasks_count(self):
    t2i_tasks = tasks.get_tasks(suite="t2i_cat_bias", split="all")
    self.assertLen(t2i_tasks, 2)
    self.assertLen(tasks.get_t2i_cat_bias_tasks(), 2)

  def test_t2i_cat_bias_orange_cat_task(self):
    task = tasks.get_task("orange_cat_bias", suite="t2i_cat_bias")
    self.assertEqual(task.name, "Orange Cat Bias")
    self.assertEqual(task.suite, "t2i_cat_bias")
    self.assertEqual(task.split, "evaluation")
    self.assertEqual(task.metadata["modality"], "text_to_image")
    self.assertEqual(task.metadata["category"], "color_bias")
    self.assertIn(
        "gemini-3.1-flash-image", task.metadata["recommended_target_models"]
    )

  def test_t2i_cat_bias_cat_breed_task(self):
    task = tasks.get_task("cat_breed_bias", suite="t2i_cat_bias")
    self.assertEqual(task.name, "Cat Breed Bias")
    self.assertEqual(task.metadata["category"], "breed_bias")

  def test_t2i_cat_bias_hypotheses(self):
    all_hyps = tasks.get_t2i_cat_bias_hypotheses("all")
    self.assertLen(all_hyps, 3)

    baseline_h = tasks.get_t2i_cat_bias_hypotheses(
        "orange_cat_baseline_frequency"
    )
    self.assertLen(baseline_h, 1)
    self.assertEqual(baseline_h[0].name, "Orange Cat Baseline Frequency")
    self.assertEqual(baseline_h[0].null_proportion, 0.20)
    self.assertEqual(baseline_h[0].alternative, "larger")
    self.assertEqual(baseline_h[0].test_type, "proportion_z_test")

    d = baseline_h[0].to_dict()
    self.assertEqual(d["name"], "Orange Cat Baseline Frequency")
    self.assertEqual(d["null_proportion"], 0.20)

  def test_t2i_cat_bias_seed_string(self):
    self.assertIn("orange or ginger cats", tasks.ORANGE_CAT_BIAS_SEED)
    self.assertIn("20 percent as the null", tasks.ORANGE_CAT_BIAS_SEED)
    self.assertIn("analyze_responses", tasks.ORANGE_CAT_BIAS_SEED)

  def test_t2i_cat_bias_task_in_all_suites(self):
    all_tasks = tasks.get_tasks(suite="all", split="all")
    t2i_keys = [t.key for t in all_tasks if t.suite == "t2i_cat_bias"]
    self.assertIn("orange_cat_bias", t2i_keys)
    self.assertIn("cat_breed_bias", t2i_keys)

  def test_t2i_handedness_bias_suite_registered(self):
    suites = tasks.list_suites()
    self.assertIn("t2i_handedness_bias", suites)

  def test_t2i_handedness_bias_tasks_count(self):
    t2i_tasks = tasks.get_tasks(suite="t2i_handedness_bias", split="all")
    self.assertLen(t2i_tasks, 2)
    self.assertLen(tasks.get_t2i_handedness_bias_tasks(), 2)

  def test_t2i_handedness_bias_left_hand_task(self):
    task = tasks.get_task("left_hand_bias", suite="t2i_handedness_bias")
    self.assertEqual(task.name, "Left Hand Bias")
    self.assertEqual(task.suite, "t2i_handedness_bias")
    self.assertEqual(task.split, "evaluation")
    self.assertEqual(task.metadata["modality"], "text_to_image")
    self.assertEqual(task.metadata["category"], "handedness_bias")
    self.assertIn(
        "gemini-3.1-flash-image", task.metadata["recommended_target_models"]
    )

  def test_t2i_handedness_bias_activity_task(self):
    task = tasks.get_task(
        "handedness_activity_bias", suite="t2i_handedness_bias"
    )
    self.assertEqual(task.name, "Handedness Activity Bias")
    self.assertEqual(task.metadata["category"], "activity_handedness_bias")

  def test_t2i_handedness_bias_hypotheses(self):
    all_hyps = tasks.get_t2i_handedness_bias_hypotheses("all")
    self.assertLen(all_hyps, 3)

    baseline_h = tasks.get_t2i_handedness_bias_hypotheses(
        "left_hand_baseline_frequency"
    )
    self.assertLen(baseline_h, 1)
    self.assertEqual(baseline_h[0].name, "Left Hand Baseline Frequency")
    self.assertEqual(baseline_h[0].null_proportion, 0.10)
    self.assertEqual(baseline_h[0].alternative, "smaller")
    self.assertEqual(baseline_h[0].test_type, "proportion_z_test")

    d = baseline_h[0].to_dict()
    self.assertEqual(d["name"], "Left Hand Baseline Frequency")
    self.assertEqual(d["null_proportion"], 0.10)

  def test_t2i_handedness_bias_seed_string(self):
    self.assertIn("right-handed people", tasks.LEFT_HAND_BIAS_SEED)
    self.assertIn("10 percent as the null", tasks.LEFT_HAND_BIAS_SEED)
    self.assertIn("analyze_responses", tasks.LEFT_HAND_BIAS_SEED)

  def test_t2i_handedness_bias_task_in_all_suites(self):
    all_tasks = tasks.get_tasks(suite="all", split="all")
    handedness_keys = [
        t.key for t in all_tasks if t.suite == "t2i_handedness_bias"
    ]
    self.assertIn("left_hand_bias", handedness_keys)
    self.assertIn("handedness_activity_bias", handedness_keys)


if __name__ == "__main__":
  absltest.main()
