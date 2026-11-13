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

"""Unit tests for structured output Pydantic schemas."""

from absl.testing import absltest
from disco.parsing import schemas


class SchemasTest(absltest.TestCase):

  def test_hypothesis_extraction(self):
    item = schemas.HypothesisExtraction.model_validate(
        {"HYPOTHESIS_ID": "H_1", "HYPOTHESIS": "Some claim"}
    )
    self.assertFalse(item.insufficient_information)
    self.assertEqual(item.hypothesis_id, "H_1")
    self.assertEqual(item.hypothesis_text, "Some claim")

    insufficient = schemas.HypothesisExtraction.model_validate(
        {"INSUFFICIENT_INFORMATION": True}
    )
    self.assertTrue(insufficient.insufficient_information)

  def test_stats_extraction(self):
    item = schemas.StatsExtraction.model_validate({
        "TEST_TYPE": "run_proportion_z_test",
        "P-VALUE": 0.01,
        "GROUP_A_MEAN": 0.8,
        "GROUP_A_OBS": 50,
        "NULL_PROPORTION": 0.5,
        "ALTERNATIVE": "larger",
    })
    self.assertEqual(item.test_type, "run_proportion_z_test")
    self.assertEqual(item.p_value, 0.01)
    self.assertEqual(item.group_a_mean, 0.8)
    self.assertEqual(item.group_a_obs, 50)
    self.assertEqual(item.null_proportion, 0.5)
    self.assertEqual(item.alternative, "larger")

  def test_test_preregistration_extraction(self):
    item = schemas.TestPreregistrationExtraction.model_validate({
        "TEST_TYPE": "McNemar test",
        "TARGET_METRIC": "success_rate",
        "NULL_BASELINE": 0.5,
        "ALTERNATIVE": "larger",
        "IS_PAIRED": True,
    })
    self.assertEqual(item.test_type, "McNemar test")
    self.assertEqual(item.target_metric, "success_rate")
    self.assertEqual(item.null_baseline, 0.5)
    self.assertEqual(item.alternative, "larger")
    self.assertTrue(item.is_paired)

  def test_hypothesis_result(self):
    result = schemas.HypothesisResult(h_id="H_1", h_text="Claim text")
    self.assertEqual(result.h_id, "H_1")
    self.assertEqual(result.h_text, "Claim text")


if __name__ == "__main__":
  absltest.main()
