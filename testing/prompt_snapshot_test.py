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

r"""Snapshot tests for prompt builders and the task registry.

Regenerate after an intentional change with (add `--only=<golden>` to limit
to e.g. `prompts_prompts` or `task_registry`):

  python -m disco.testing.regenerate_goldens --output_dir=testing/testdata
"""

from absl.testing import absltest
from absl.testing import parameterized

from disco.testing import golden_utils
from disco.testing import inline_prompt_cases
from disco.testing import prompt_snapshot_cases


class PromptSnapshotTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (name, name, module)
      for name, module in prompt_snapshot_cases.PROMPT_MODULES.items()
  )
  def test_prompts_match_snapshot(self, name, module):
    expected = golden_utils.load_golden(f"prompts_{name}")
    actual = prompt_snapshot_cases.render_prompt_module(
        module, prompt_snapshot_cases.PROMPT_GROUPS[name]
    )
    self.assertSameElements(actual.keys(), expected.keys())
    for case in sorted(expected):
      with self.subTest(case=case):
        self.assertMultiLineEqual(actual[case], expected[case])

  @parameterized.named_parameters(
      (name, name, render)
      for name, render in inline_prompt_cases.INLINE_GROUPS.items()
  )
  def test_inline_prompts_match_snapshot(self, name, render):
    expected = golden_utils.load_golden(f"prompts_{name}")
    actual = render()
    self.assertSameElements(actual.keys(), expected.keys())
    for case in sorted(expected):
      with self.subTest(case=case):
        self.assertMultiLineEqual(actual[case], expected[case])

  def test_task_registry_matches_snapshot(self):
    expected = golden_utils.load_golden("task_registry")
    actual = golden_utils.to_jsonable(
        prompt_snapshot_cases.task_registry_snapshot()
    )
    mismatches = golden_utils.find_mismatches(actual, expected)
    self.assertEmpty(mismatches, "\n".join(mismatches[:50]))

  def test_every_builder_has_a_default_case(self):
    for name, module in prompt_snapshot_cases.PROMPT_MODULES.items():
      cases = prompt_snapshot_cases.render_prompt_module(
          module, prompt_snapshot_cases.PROMPT_GROUPS[name]
      )
      builders = {name.split("[", 1)[0] for name in cases}
      self.assertSameElements(builders, [n for n in cases if "[" not in n])


if __name__ == "__main__":
  absltest.main()
