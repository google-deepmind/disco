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

r"""Golden test for the discovery-baseline methods and their budget plans.

Regenerate after an intentional change with:

  python -m disco.testing.regenerate_method_goldens \
    --golden_dir=testing/testdata
"""

from absl.testing import absltest

from disco.testing import golden_utils
from disco.testing import method_cases


class MethodsGoldenTest(absltest.TestCase):

  def _check(self, name: str) -> None:
    expected = golden_utils.load_golden(name)
    actual = golden_utils.to_jsonable(method_cases.ALL_CASES[name]())
    mismatches = golden_utils.find_mismatches(actual, expected, rtol=0.0)
    self.assertEmpty(mismatches, "\n".join(mismatches[:50]))

  def test_method_runs_match_golden(self):
    self._check("method_runs")

  def test_budget_plans_match_golden(self):
    self._check("budget_plans")


if __name__ == "__main__":
  absltest.main()
