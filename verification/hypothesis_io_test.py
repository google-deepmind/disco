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

"""Tests for verification/hypothesis_io.

The `_legacy_*` functions below are frozen, file-I/O-free copies of the
loaders hypothesis_io replaced (standalone_precision_verifier.load_hypotheses
and tasks/visual_symmetry/verify_visual_symmetry._load_hypotheses_from_file).
Every case is checked against them so the shared implementation stays
behavior-identical.
"""

import json
import os
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized

from disco.verification import hypothesis_io

_METADATA_KEYS = {
    "task_suite",
    "task_name",
    "config",
    "experiment_name",
    "target_model",
    "investigator_model",
    "methods",
    "baseline_results",
}


def _legacy_methods(data: Any) -> list[tuple[str, str]]:
  """Frozen copy of the multi-method block shared by the loaders."""
  methods_list = data.get("methods") or data.get("baseline_results")
  extracted = []
  if isinstance(methods_list, list) and methods_list:
    for m_entry in methods_list:
      if not isinstance(m_entry, dict):
        continue
      m_name = m_entry.get("method_name") or m_entry.get("method") or "baseline"
      preds = (
          m_entry.get("predictions")
          or m_entry.get("hypotheses")
          or m_entry.get("rejected_hypotheses")
          or []
      )
      for i, pred in enumerate(preds):
        if isinstance(pred, str) and pred.strip():
          extracted.append((f"{m_name}_h_{i}", pred.strip()))
        elif isinstance(pred, dict):
          txt = (
              pred.get("text")
              or pred.get("hypothesis")
              or pred.get("claim")
              or pred.get("content")
              or ""
          )
          if str(txt).strip():
            lbl = pred.get("label") or pred.get("id") or f"{m_name}_h_{i}"
            extracted.append((str(lbl), str(txt).strip()))
  return extracted


def _legacy_labeled(
    data: Any, strict: bool, path: str
) -> list[tuple[str, str]]:
  """Frozen copy of standalone_precision_verifier.load_hypotheses (strict) and verify_visual_symmetry._load_hypotheses_from_file (non-strict)."""
  if isinstance(data, dict):
    extracted = _legacy_methods(data)
    if extracted:
      return [(lbl, txt) for lbl, txt in extracted if txt]
    for key in [
        "hypotheses",
        "rejected_hypotheses",
        "verified_hypotheses",
        "predictions",
    ]:
      if key in data:
        data = data[key]
        break

  result: list[tuple[str, str]] = []
  if isinstance(data, dict):
    for label, val in data.items():
      if str(label).lower() in _METADATA_KEYS:
        continue
      if isinstance(val, str) and val.strip():
        result.append((str(label), val.strip()))
      elif isinstance(val, dict):
        text = (
            val.get("text")
            or val.get("hypothesis")
            or val.get("claim")
            or val.get("content")
            or ""
        )
        if str(text).strip():
          result.append((str(label), str(text).strip()))
  elif isinstance(data, list):
    for i, item in enumerate(data):
      if isinstance(item, str) and item.strip():
        result.append((f"h_{i}", item.strip()))
      elif isinstance(item, dict):
        label = item.get("label") or item.get("id") or f"h_{i}"
        text = (
            item.get("text")
            or item.get("hypothesis")
            or item.get("claim")
            or item.get("content")
            or ""
        )
        if str(text).strip():
          result.append((str(label), str(text).strip()))
  elif strict:
    raise ValueError(
        f"Unexpected JSON format in {path}: expected dict or list, got"
        f" {type(data).__name__}"
    )
  return [(lbl, txt) for lbl, txt in result if txt]


# (name, payload). Synthetic strings only.
_CASES: list[tuple[str, Any]] = [
    ("dict_of_strings", {"h_a": " A ", "h_b": "B", "h_empty": "  "}),
    (
        "dict_of_dicts",
        {
            "h_a": {"text": "A"},
            "h_b": {"hypothesis": "B"},
            "h_c": {"claim": "C"},
            "h_d": {"content": "D"},
            "h_e": {"text": "", "claim": "E"},
            "h_none": {"other": "x"},
            "h_ws": {"text": "   "},
            "h_num": {"text": 7},
        },
    ),
    (
        "dict_with_metadata",
        {
            "task_suite": "recipes",
            "Config": {"text": "skipped (case-insensitive)"},
            "target_model": "m",
            "h_1": "kept",
        },
    ),
    ("list_of_strings", ["A", " ", "B "]),
    (
        "list_of_dicts",
        [
            {"label": "x", "text": "A"},
            {"id": "y", "hypothesis": "B"},
            {"claim": "C"},
            {"label": "z", "text": " "},
            "D",
            5,
        ],
    ),
    ("hypotheses_key", {"hypotheses": {"h_1": "A"}, "task_name": "t"}),
    ("rejected_key", {"rejected_hypotheses": ["A", "B"]}),
    ("verified_key", {"verified_hypotheses": ["A"]}),
    ("predictions_key", {"predictions": [{"text": "P"}]}),
    (
        "hypotheses_and_rejected",
        {"hypotheses": ["H1", "H2"], "rejected_hypotheses": ["R1"]},
    ),
    (
        "rejected_and_predictions",
        {"rejected_hypotheses": ["R1"], "predictions": ["P1", "P2"]},
    ),
    (
        "verified_and_hypotheses",
        {"verified_hypotheses": ["V1"], "hypotheses": ["H1"]},
    ),
    ("empty_container_wins", {"hypotheses": [], "rejected_hypotheses": ["R"]}),
    (
        "methods",
        {
            "methods": [
                {"method_name": "M1", "predictions": ["A", {"text": "B"}]},
                {"method": "M2", "hypotheses": [{"id": "i", "claim": "C"}]},
                {"rejected_hypotheses": ["D"]},
                "not a dict",
                {"method_name": "M3", "predictions": [], "hypotheses": ["E"]},
            ]
        },
    ),
    (
        "baseline_results",
        {"baseline_results": [{"method_name": "B", "predictions": ["X"]}]},
    ),
    (
        "methods_empty_falls_back",
        {
            "methods": [{"method_name": "M", "predictions": []}],
            "rejected_hypotheses": ["R"],
        },
    ),
    (
        "methods_falsy_uses_baseline_results",
        {"methods": [], "baseline_results": [{"predictions": ["Y"]}]},
    ),
    (
        "methods_non_list",
        {"methods": {"predictions": ["Z"]}, "hypotheses": ["H"]},
    ),
    ("only_methods_metadata", {"methods": [{"predictions": []}]}),
    ("no_container_key", {"foo": "F", "bar": {"text": "G"}}),
    ("empty_dict", {}),
    ("empty_list", []),
]

_SCALAR_CASES: list[tuple[str, Any]] = [
    ("top_level_int", 5),
    ("top_level_str", "a hypothesis"),
    ("top_level_none", None),
    ("container_scalar", {"hypotheses": 3}),
    ("container_none", {"rejected_hypotheses": None}),
]


class ExtractHypothesesTest(parameterized.TestCase):

  @parameterized.named_parameters(*_CASES)
  def test_matches_legacy_loaders(self, data):
    self.assertEqual(
        hypothesis_io.extract_hypotheses(data, strict_type=True, source="p"),
        _legacy_labeled(data, strict=True, path="p"),
    )
    labeled = _legacy_labeled(data, strict=False, path="")
    self.assertEqual(
        hypothesis_io.extract_hypotheses(data, strict_type=False), labeled
    )

  @parameterized.named_parameters(*_SCALAR_CASES)
  def test_scalar_payload(self, data):
    with self.assertRaisesRegex(
        ValueError, r"Unexpected JSON format in p: expected dict or list"
    ):
      hypothesis_io.extract_hypotheses(data, strict_type=True, source="p")
    with self.assertRaises(ValueError):
      _legacy_labeled(data, strict=True, path="p")
    self.assertEqual(
        hypothesis_io.extract_hypotheses(data, strict_type=False), []
    )

  def test_key_order_when_both_keys_present(self):
    data = {"hypotheses": ["H"], "rejected_hypotheses": ["R"]}
    self.assertEqual(hypothesis_io.extract_hypotheses(data), [("h_0", "H")])

  def test_method_labels(self):
    data = dict(_CASES)["methods"]
    self.assertEqual(
        hypothesis_io.extract_hypotheses(data),
        [
            ("M1_h_0", "A"),
            ("M1_h_1", "B"),
            ("i", "C"),
            ("baseline_h_0", "D"),
            ("M3_h_0", "E"),
        ],
    )

  def test_method_entries(self):
    self.assertIsNone(hypothesis_io.method_entries([1]))
    self.assertIsNone(hypothesis_io.method_entries({"methods": []}))
    self.assertIsNone(hypothesis_io.method_entries({"methods": {"a": 1}}))
    self.assertEqual(
        hypothesis_io.method_entries({"baseline_results": [1]}), [1]
    )

  def test_first_truthy(self):
    self.assertEqual(
        hypothesis_io.first_truthy({"a": [], "b": ["x"]}, ("a", "b")), ["x"]
    )
    self.assertIsNone(hypothesis_io.first_truthy({"a": 0}, ("a", "b")))
    self.assertEqual(hypothesis_io.first_truthy({}, ("a",), []), [])


class LoadHypothesesTest(absltest.TestCase):

  def _write(self, data: Any) -> str:
    path = self.create_tempfile(content=json.dumps(data)).full_path
    return path

  def test_load_hypotheses(self):
    path = self._write({"rejected_hypotheses": ["A", {"text": "B"}]})
    self.assertEqual(
        hypothesis_io.load_hypotheses(path), [("h_0", "A"), ("h_1", "B")]
    )

  def test_load_hypotheses_and_data_returns_raw(self):
    raw = {"hypotheses": {"x": "X"}, "config": {"seed": "s"}}
    hyps, data = hypothesis_io.load_hypotheses_and_data(self._write(raw))
    self.assertEqual(hyps, [("x", "X")])
    self.assertEqual(data, raw)

  def test_strict_type_error_names_path(self):
    path = self._write(42)
    with self.assertRaisesRegex(ValueError, f"in {path}: .* got int"):
      hypothesis_io.load_hypotheses(path)
    self.assertEqual(hypothesis_io.load_hypotheses(path, strict_type=False), [])

  def test_missing_file(self):
    missing = os.path.join(self.create_tempdir().full_path, "nope.json")
    with self.assertRaisesRegex(FileNotFoundError, "Hypotheses file not found"):
      hypothesis_io.load_hypotheses(missing)


if __name__ == "__main__":
  absltest.main()
