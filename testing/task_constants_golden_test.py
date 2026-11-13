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

r"""Golden test for the public task constants and the legacy task lookups.

Regenerate after an intentional change with:

  python -m disco.testing.regenerate_goldens \
    --output_dir=testing/testdata \
    --only=task_constants
"""

from absl.testing import absltest

from disco.testing import golden_utils
from disco.testing import task_constant_cases


class TaskConstantsGoldenTest(absltest.TestCase):

  def test_matches_golden(self):
    expected = golden_utils.load_golden("task_constants")
    actual = golden_utils.to_jsonable(
        task_constant_cases.task_constants_snapshot()
    )
    mismatches = golden_utils.find_mismatches(actual, expected, rtol=0.0)
    self.assertEmpty(mismatches, "\n".join(mismatches[:50]))


if __name__ == "__main__":
  absltest.main()
