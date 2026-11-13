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

"""Unit tests for symmetry task detection from task_suite and seed_instruction."""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized

from disco.core import task_detection


@dataclasses.dataclass
class _Cfg:
  seed_instruction: str | None = ""
  task_suite: str | None = ""


class IsSymmetryTaskTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("suite_symmetry", "Count the words.", "visual_symmetry", True),
      ("suite_and_seed", "Visual Symmetry", "visual_symmetry", True),
      # task_suite decides when set, whatever the seed says.
      ("other_suite_wins_over_seed", "Visual Symmetry", "hedging", False),
      ("seed_fallback", "Test reflection SYMMETRY.", "", True),
      ("neither", "Count the words.", "", False),
      ("none_fields", None, None, False),
  )
  def test_rule(self, seed, suite, expected):
    cfg = _Cfg(seed_instruction=seed, task_suite=suite)
    self.assertEqual(task_detection.is_symmetry_task(cfg), expected)

  def test_missing_attributes(self):
    self.assertFalse(task_detection.is_symmetry_task(object()))

  def test_seed_fallback_warns(self):
    with self.assertLogs(level="WARNING") as logs:
      task_detection.is_symmetry_task(_Cfg(seed_instruction="symmetry"))
    self.assertIn("inferred from the seed", logs.output[0])


if __name__ == "__main__":
  absltest.main()
