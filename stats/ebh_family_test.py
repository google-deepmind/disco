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

"""Unit tests for e-BH family size resolution and failed-slot warnings."""

import json
import os

from absl.testing import absltest

from disco.stats import ebh_family


class EbhFamilyTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    ebh_family.read_max_predictions.cache_clear()
    self.run_dir = self.create_tempdir().full_path
    with open(os.path.join(self.run_dir, "flags.json"), "w") as f:
      json.dump({"max_predictions": 10}, f)
    self.input_path = os.path.join(
        self.run_dir, "verify", "verification_results.json"
    )

  def test_seven_records_one_failed_cap_ten(self):
    # 7 records, 1 failed, cap 10: K = 10 / 9 / observed.
    self.assertEqual(
        ebh_family.resolve(self.input_path, "max_predictions", n_failed=1),
        (10, "flags_json"),
    )
    self.assertEqual(
        ebh_family.resolve(
            self.input_path, "max_predictions_minus_failed", n_failed=1
        ),
        (9, "flags_json_minus_failed"),
    )
    self.assertEqual(
        ebh_family.resolve(self.input_path, "observed", n_failed=1),
        (None, "observed"),
    )

  def test_override_wins_over_flags_json(self):
    self.assertEqual(
        ebh_family.resolve(self.input_path, "max_predictions", 25), (25, "flag")
    )

  def test_missing_cap_falls_back_to_observed(self):
    other = self.create_tempdir().full_path
    self.assertEqual(
        ebh_family.resolve(os.path.join(other, "evaluation_results.json")),
        (None, "observed_fallback"),
    )

  def test_failed_slots_warning_names_each_slot(self):
    failed = [
        {"label": "H3", "status": "SCRAPE_FAILED", "failure_reason": "outage"},
        {"label": "H5", "sample_size": 0},
    ]
    msg = ebh_family.failed_slots_warning(
        "run_a", failed, "max_predictions", 10
    )
    self.assertIn("run_a: 2 failed record(s) counted in K=10 at e=0", msg)
    self.assertIn("label='H3' status=SCRAPE_FAILED reason='outage'", msg)
    self.assertIn("label='H5' status=unset reason='-'", msg)
    self.assertIn(
        "excluded from the e-BH family",
        ebh_family.failed_slots_warning("r", failed, "observed", None),
    )
    bh = ebh_family.failed_slots_warning("r", failed, "", None)
    self.assertIn("2 failed record(s) excluded from the family: label=", bh)
    self.assertNotIn("--ebh_family_size", bh)

  def test_unknown_mode_raises(self):
    with self.assertRaises(ValueError):
      ebh_family.resolve(self.input_path, "bogus")


if __name__ == "__main__":
  absltest.main()
