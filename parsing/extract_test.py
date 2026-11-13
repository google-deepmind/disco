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

"""Unit tests for text and JSON extraction helpers."""

from absl.testing import absltest
from absl.testing import parameterized
from disco.parsing import extract


class ExtractTest(parameterized.TestCase):

  def test_json_array_extraction(self):
    res = extract.parse_selected_example_indices(
        response_text="[1, 3, 5]", total_candidates=10, num_to_select=3
    )
    self.assertEqual(res, [0, 2, 4])

  def test_json_array_with_duplicates_and_out_of_bounds(self):
    res = extract.parse_selected_example_indices(
        response_text="[2, 2, 0, 15, 4]", total_candidates=10, num_to_select=3
    )
    # 2 -> 1, duplicate 2 skipped, 0 is <=0 skipped, 15 is >10 skipped, 4 -> 3
    # then fallback pads with first available (0)
    self.assertEqual(res, [1, 3, 0])

  def test_regex_fallback(self):
    res = extract.parse_selected_example_indices(
        response_text="Selected candidate indices are 2, 4 and 1.",
        total_candidates=5,
        num_to_select=3,
    )
    self.assertEqual(res, [1, 3, 0])

  def test_empty_string_fallback(self):
    res = extract.parse_selected_example_indices(
        response_text="", total_candidates=5, num_to_select=3
    )
    self.assertEqual(res, [0, 1, 2])

  def test_total_candidates_smaller_than_num_to_select(self):
    res = extract.parse_selected_example_indices(
        response_text="[1]", total_candidates=2, num_to_select=5
    )
    self.assertEqual(res, [0, 1])


if __name__ == "__main__":
  absltest.main()
