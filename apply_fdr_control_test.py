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

"""Unit tests for apply_fdr_control: target resolution and output layout."""

import json
import os

from absl.testing import absltest

from disco import apply_fdr_control

_RECORDS = [
    {"label": "h1", "hypothesis_text": "A", "status": "OK", "p_value": 0.001},
    {"label": "h2", "hypothesis_text": "B", "status": "OK", "p_value": 0.04},
    {"label": "h3", "hypothesis_text": "C", "status": "OK", "p_value": 0.6},
    {
        "label": "h4",
        "hypothesis_text": "D",
        "status": "SCRAPE_FAILED",
        "p_value": 1.0,
    },
]


def _write_json(path: str, data: object) -> str:
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f)
  return path


class ResolveSingleTargetTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.run_dir = self.create_tempdir().full_path

  def _verification_file(self, subdir: str = "") -> str:
    return _write_json(
        os.path.join(self.run_dir, subdir, "verification_results.json"),
        {"verification_mode": "single_batch_p_value", "records": _RECORDS},
    )

  def _evaluation_file(self) -> str:
    return _write_json(
        os.path.join(self.run_dir, "evaluation_results.json"),
        {
            "method_name": "disco",
            "precision_check": {"records": _RECORDS},
        },
    )

  def test_requires_a_path(self):
    with self.assertRaisesRegex(ValueError, "--input_path or --run_dir"):
      apply_fdr_control.resolve_single_target()

  def test_input_path_file_writes_beside_the_input(self):
    v_path = self._verification_file("verif")
    targets = apply_fdr_control.resolve_single_target(input_path=v_path)
    self.assertLen(targets, 1)
    self.assertEqual(targets[0].source_type, "verification")
    self.assertLen(_RECORDS, targets[0].n_hypotheses)
    self.assertEqual(
        targets[0].output_dir,
        os.path.join(self.run_dir, "verif", apply_fdr_control.OUTPUT_SUBDIR),
    )

  def test_source_verification_prefers_direct_file(self):
    self._verification_file()
    self._evaluation_file()
    targets = apply_fdr_control.resolve_single_target(
        run_dir=self.run_dir, source="verification"
    )
    self.assertLen(targets, 1)
    self.assertEqual(targets[0].source_type, "verification")
    self.assertEqual(targets[0].source_subdir, "")

  def test_source_evaluation_reads_evaluation_results(self):
    self._verification_file()
    self._evaluation_file()
    targets = apply_fdr_control.resolve_single_target(
        run_dir=self.run_dir, source="evaluation"
    )
    self.assertLen(targets, 1)
    self.assertEqual(targets[0].source_type, "evaluation")
    self.assertEqual(targets[0].method_name, "disco")
    self.assertEqual(
        targets[0].output_dir,
        os.path.join(self.run_dir, apply_fdr_control.OUTPUT_SUBDIR),
    )

  def test_source_evaluation_without_file_is_empty(self):
    self._verification_file()
    self.assertEqual(
        apply_fdr_control.resolve_single_target(
            run_dir=self.run_dir, source="evaluation"
        ),
        [],
    )

  def test_auto_uses_single_verification_subdir(self):
    self._verification_file("verif_a")
    targets = apply_fdr_control.resolve_single_target(run_dir=self.run_dir)
    self.assertLen(targets, 1)
    self.assertEqual(targets[0].source_subdir, "verif_a")

  def test_auto_falls_back_to_evaluation_results(self):
    self._evaluation_file()
    targets = apply_fdr_control.resolve_single_target(run_dir=self.run_dir)
    self.assertLen(targets, 1)
    self.assertEqual(targets[0].source_type, "evaluation")

  def test_multiple_verification_subdirs_point_at_input_path(self):
    self._verification_file("verif_a")
    self._verification_file("verif_b")
    for source in ("auto", "verification"):
      with self.subTest(source=source):
        with self.assertRaisesRegex(
            ValueError,
            r"Multiple \(2\) verification subdirs .*verif_a, verif_b.*"
            r" --input_path=.*/<subdir>/verification_results\.json",
        ):
          apply_fdr_control.resolve_single_target(
              run_dir=self.run_dir, source=source
          )

  def test_multiple_verification_subdirs_lose_to_evaluation_in_auto(self):
    self._verification_file("verif_a")
    self._verification_file("verif_b")
    self._evaluation_file()
    targets = apply_fdr_control.resolve_single_target(run_dir=self.run_dir)
    self.assertLen(targets, 1)
    self.assertEqual(targets[0].source_type, "evaluation")


class ProcessTargetTest(absltest.TestCase):

  def test_writes_outputs_to_ebh_fdr(self):
    run_dir = self.create_tempdir().full_path
    _write_json(
        os.path.join(run_dir, "verification_results.json"),
        {"verification_mode": "single_batch_p_value", "records": _RECORDS},
    )
    _write_json(os.path.join(run_dir, "flags.json"), {"max_predictions": 10})
    (target,) = apply_fdr_control.resolve_single_target(run_dir=run_dir)
    self.assertEqual(target.output_dir, os.path.join(run_dir, "ebh_fdr"))

    row = apply_fdr_control.process_target(target, alpha=0.05)
    self.assertIsNotNone(row)
    assert row is not None
    self.assertEqual(row["n_total"], 4)
    self.assertEqual(row["n_evaluated"], 3)
    self.assertEqual(row["n_unadjusted"], 2)
    self.assertEqual(row["family_size_mode"], "evaluated")
    self.assertEqual(row["family_size_m"], 3)
    self.assertEqual(row["ebh_family_size_m"], 10)
    self.assertEqual(row["ebh_family_size_source"], "flags_json")

    for name in (
        "verification_results_fdr.json",
        "fdr_summary.json",
        "hypotheses_ranked.csv",
    ):
      self.assertTrue(os.path.exists(os.path.join(target.output_dir, name)))
    with open(
        os.path.join(target.output_dir, "fdr_summary.json"), encoding="utf-8"
    ) as f:
      summary = json.load(f)
    self.assertEqual(summary["config"]["family_size_mode"], "evaluated")
    self.assertEqual(summary["rollup_row"], row)

  def test_recomputes_existing_outputs(self):
    run_dir = self.create_tempdir().full_path
    _write_json(
        os.path.join(run_dir, "verification_results.json"),
        {"records": _RECORDS},
    )
    (target,) = apply_fdr_control.resolve_single_target(run_dir=run_dir)
    first = apply_fdr_control.process_target(target, alpha=0.05)
    second = apply_fdr_control.process_target(target, alpha=0.01)
    assert first is not None and second is not None
    self.assertEqual(first["alpha"], 0.05)
    self.assertEqual(second["alpha"], 0.01)
    self.assertLess(second["n_unadjusted"], first["n_unadjusted"])


if __name__ == "__main__":
  absltest.main()
