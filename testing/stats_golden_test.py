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

"""Golden regression tests for the statistics and FDR-control modules.

If a test fails after an intentional change, regenerate with:

  python -m disco.testing.regenerate_goldens --output_dir=testing/testdata

and explain the change in the commit/PR description.
"""

from absl.testing import absltest
from absl.testing import parameterized

from disco.testing import golden_utils
from disco.testing import stats_golden_cases


class StatsGoldenTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (name, name) for name in stats_golden_cases.ALL_CASES
  )
  def test_matches_golden(self, name):
    actual = golden_utils.to_jsonable(stats_golden_cases.ALL_CASES[name]())
    expected = golden_utils.load_golden(name)
    mismatches = golden_utils.find_mismatches(actual, expected)
    self.assertEmpty(
        mismatches,
        msg=f"{len(mismatches)} mismatches vs golden {name!r}:\n"
        + "\n".join(mismatches[:50]),
    )


class GoldenUtilsTest(absltest.TestCase):

  def test_find_mismatches_tolerates_float_noise(self):
    self.assertEmpty(golden_utils.find_mismatches(1.0 + 1e-12, 1.0))

  def test_find_mismatches_reports_differences(self):
    mismatches = golden_utils.find_mismatches(
        {"a": [1, 2.5], "b": True, "c": "x"},
        {"a": [1, 2.0], "b": False, "d": "x"},
    )
    self.assertLen(mismatches, 4)

  def test_find_mismatches_handles_nan_and_inf(self):
    self.assertEmpty(
        golden_utils.find_mismatches(
            [float("nan"), float("inf")], [float("nan"), float("inf")]
        )
    )
    self.assertNotEmpty(golden_utils.find_mismatches(float("inf"), 1.0))

  def test_find_mismatches_bool_is_not_int(self):
    self.assertNotEmpty(golden_utils.find_mismatches(True, 1))

  def test_to_jsonable_sorts_sets(self):
    self.assertEqual(
        golden_utils.to_jsonable({"s": {"b", "a"}}), {"s": ["a", "b"]}
    )

  def test_to_jsonable_encodes_non_finite_floats(self):
    self.assertEqual(
        golden_utils.to_jsonable(
            [float("inf"), -float("inf"), float("nan"), 1.5, True]
        ),
        ["Infinity", "-Infinity", "NaN", 1.5, True],
    )


if __name__ == "__main__":
  absltest.main()
