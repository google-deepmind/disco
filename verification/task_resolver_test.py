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

"""Unit tests for resolving task autoraters, seeds, suites, and null rates."""

from absl.testing import absltest

from disco.verification import task_resolver


class ResolveTaskAutoraterTest(absltest.TestCase):

  def test_precedence(self):
    self.assertEqual(
        task_resolver.resolve_task_autorater(
            {"config": {"procedural_autorater": "count_hedges"}}, "count_words"
        ),
        "count_words",
    )
    self.assertEqual(
        task_resolver.resolve_task_autorater(
            {"config": {"disco_procedural_autorater": "count_hedges"}}, ""
        ),
        "count_hedges",
    )
    self.assertEqual(
        task_resolver.resolve_task_autorater(
            {"procedural_autorater": "count_hedges"}, ""
        ),
        "count_hedges",
    )

  def test_registry_lookup(self):
    for data in (
        {"task_name": "recipe_translate_french_long"},
        {"config": {"task_name": "Recipe Translate French Long"}},
        {"config": {"task_suite": "recipes"}},
    ):
      with self.subTest(data=data):
        self.assertEqual(
            task_resolver.resolve_task_autorater(data, ""), "count_words"
        )

  def test_unresolvable(self):
    self.assertEqual(task_resolver.resolve_task_autorater(None, ""), "")
    self.assertEqual(task_resolver.resolve_task_autorater(["x"], ""), "")
    self.assertEqual(
        task_resolver.resolve_task_autorater({"task_name": "no_such"}, ""), ""
    )


class ResolveSeedInstructionTest(absltest.TestCase):

  def test_precedence(self):
    self.assertEqual(
        task_resolver.resolve_seed_instruction(
            {"config": {"seed_instruction": "cfg"}}, "cli"
        ),
        "cli",
    )
    self.assertEqual(
        task_resolver.resolve_seed_instruction(
            {"config": {"seed_instruction": "cfg"}, "seed_instruction": "top"},
            "",
        ),
        "cfg",
    )
    self.assertEqual(
        task_resolver.resolve_seed_instruction({"config": {"seed": "s"}}, ""),
        "s",
    )
    self.assertEqual(
        task_resolver.resolve_seed_instruction({"disco_seed": "top"}, ""),
        "top",
    )

  def test_registry_and_default(self):
    self.assertIn(
        "recipe",
        task_resolver.resolve_seed_instruction(
            {"config": {"task_name": "recipe_translate_french_long"}}, ""
        ).lower(),
    )
    self.assertEqual(
        task_resolver.resolve_seed_instruction({"task_name": "no_such"}, ""),
        task_resolver.DEFAULT_SEED_INSTRUCTION,
    )
    self.assertEqual(
        task_resolver.resolve_seed_instruction(None, ""),
        "Verify the provided hypotheses",
    )


class ResolveTaskSuiteAndNullTest(absltest.TestCase):

  def test_task_suite(self):
    self.assertEqual(
        task_resolver.resolve_task_suite({"task_suite": "a"}, "cli"), "cli"
    )
    self.assertEqual(
        task_resolver.resolve_task_suite(
            {"task_suite": "a", "config": {"task_suite": "b"}}, ""
        ),
        "a",
    )
    self.assertEqual(
        task_resolver.resolve_task_suite({"config": {"task_suite": "b"}}, ""),
        "b",
    )
    self.assertEqual(task_resolver.resolve_task_suite(None, ""), "")

  def test_null_proportion(self):
    self.assertAlmostEqual(
        task_resolver.resolve_task_null_proportion(None, "visual_symmetry"),
        0.90,
    )
    self.assertAlmostEqual(
        task_resolver.resolve_task_null_proportion(
            {"null_proportion": 0.6}, "visual_symmetry"
        ),
        0.6,
    )
    self.assertAlmostEqual(
        task_resolver.resolve_task_null_proportion(
            {"null_proportion": "bad", "config": {"null_proportion": 0.8}}, ""
        ),
        0.8,
    )
    for bad in ({"null_proportion": 1.5}, {"null_proportion": 0.0}):
      self.assertEqual(
          task_resolver.resolve_task_null_proportion(bad, "no_such_suite"), 0.0
      )


class RegistryLookupTest(absltest.TestCase):

  def test_suite_task_key_and_alias_resolve_without_warning(self):
    with self.assertNoLogs(logger="absl", level="WARNING"):
      for key in (
          "hedging",
          "hedging_paired",
          "Hedging Paired Ambiguity Modulation",
      ):
        self.assertEqual(
            task_resolver.resolve_task_autorater({"task_name": key}, ""),
            "count_hedges",
        )
      self.assertEqual(
          task_resolver.resolve_task_null_proportion(None, "recipes"), 0.1
      )

  def test_legacy_lookup_still_resolves_and_warns(self):
    # "paired_ambiguity" is only a substring of a hedging task key.
    with self.assertLogs(logger="absl", level="WARNING") as logs:
      self.assertEqual(
          task_resolver.resolve_task_autorater(
              {"task_name": "paired_ambiguity"}, ""
          ),
          "count_hedges",
      )
    self.assertTrue(
        any("legacy" in r.getMessage() for r in logs.records), logs.output
    )

  def test_null_proportion_only_matches_suites(self):
    self.assertEqual(
        task_resolver.resolve_task_null_proportion(None, "hedging_paired"), 0.0
    )


if __name__ == "__main__":
  absltest.main()
