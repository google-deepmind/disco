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

"""Unit tests for hypothesis, query, and statistics response parsers."""

from absl.testing import absltest
from absl.testing import parameterized
from disco.parsing import parsers
from disco.parsing import schemas


class ParsersTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("simple", '{"a": 1}', '{"a": 1}'),
      ("markdown", '```json\n{"a": 1}\n```', '{"a": 1}'),
      (
          "surrounding_text",
          'Leading text {"key": "val"} trailing',
          '{"key": "val"}',
      ),
      (
          "multiple_objects",
          'First {"first": 1} then {"second": 2}',
          '{"first": 1}',
      ),
      ("no_json", "plain string without braces", "plain string without braces"),
  )
  def test_clean_json(self, raw_input, expected):
    self.assertEqual(parsers._clean_json(raw_input), expected)

  def test_hypothesis_parser_validate(self):
    p = parsers.HypothesisParser(model="test-model")
    res = p._validate(
        schemas.HypothesisExtraction(
            hypothesis_id="H_1", hypothesis_text="A testable claim"
        ),
        original_resp="",
    )
    self.assertEqual(res.h_id, "H_1")
    self.assertEqual(res.h_text, "A testable claim")

    with self.assertRaises(ValueError):
      p._validate(
          schemas.HypothesisExtraction(
              hypothesis_id=None, hypothesis_text="Claim"
          ),
          original_resp="",
      )

  def test_stats_parser_validate(self):
    p = parsers.StatsParser(model="test-model")
    parsed = schemas.StatsExtraction(
        test_type="run_proportion_z_test",
        p_value=0.04,
        group_a_mean=0.7,
        group_a_obs=20,
    )
    res = p._validate(
        parsed,
        original_resp=(
            "Null proportion: 0.3 Alternative: larger Sample std: 1.5"
        ),
    )
    self.assertEqual(res.test_type, "run_proportion_z_test")
    self.assertEqual(res.p_value, 0.04)
    self.assertEqual(res.group_a_mean, 0.7)
    self.assertEqual(res.null_proportion, 0.3)
    self.assertEqual(res.alternative, "larger")
    self.assertEqual(res.sample_std, 1.5)

    with self.assertRaises(ValueError):
      p._validate(
          schemas.StatsExtraction(test_type="test", p_value=-0.1),
          original_resp="",
      )
    with self.assertRaises(ValueError):
      p._validate(
          schemas.StatsExtraction(test_type="test", p_value=1.5),
          original_resp="",
      )
    with self.assertRaises(ValueError):
      p._validate(
          schemas.StatsExtraction(test_type="test", p_value=None),
          original_resp="",
      )

  def test_test_preregistration_parser_validate(self):
    p = parsers.TestPreregistrationParser(model="test-model")

    # Binary test default null is 0.5
    res_bin = p._validate(
        schemas.TestPreregistrationExtraction(
            test_type="Proportion z-test",
            target_metric="acc",
        ),
        original_resp="Is-Paired: yes",
    )
    self.assertEqual(res_bin.null_baseline, 0.5)
    self.assertTrue(res_bin.is_paired)

    # Continuous test default null is 0.0
    res_cont = p._validate(
        schemas.TestPreregistrationExtraction(
            test_type="Continuous means test",
            target_metric="diff",
        ),
        original_resp="",
    )
    self.assertEqual(res_cont.null_baseline, 0.0)

    # Missing test_type or target_metric raises ValueError
    with self.assertRaises(ValueError):
      p._validate(
          schemas.TestPreregistrationExtraction(
              test_type="", target_metric="acc"
          ),
          original_resp="",
      )
    with self.assertRaises(ValueError):
      p._validate(
          schemas.TestPreregistrationExtraction(
              test_type="z-test", target_metric=""
          ),
          original_resp="",
      )

  def test_parsers_from_model(self):
    bundle = parsers.Parsers.from_model("test-model", validate_stats_test=True)
    self.assertIsInstance(bundle.hypothesis, parsers.HypothesisParser)
    self.assertIsInstance(bundle.stats, parsers.StatsParser)
    self.assertIsInstance(
        bundle.test_preregistration, parsers.TestPreregistrationParser
    )
    self.assertTrue(bundle.stats._validate_stats_test)


if __name__ == "__main__":
  absltest.main()
