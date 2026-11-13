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

"""Single loader for hypothesis lists stored in DISCO / baseline JSON files.

Several tools share the same "find the hypotheses in this JSON blob" logic
(standalone_precision_verifier.load_hypotheses and
tasks/visual_symmetry/verify_visual_symmetry._load_hypotheses_from_file). They
agree on everything except what happens when the selected payload is neither a
dict nor a list (the verifier raises; the others return an empty list), which
is the `strict_type` parameter. They all use one container-key precedence,
`KEY_ORDER`.

Supported layouts (checked in this order):
  1. A multi-method container: `{"methods": [...]}` or
     `{"baseline_results": [...]}` whose entries carry `predictions`,
     `hypotheses`, or `rejected_hypotheses` (first truthy wins). Labels are
     `{method_name}_h_{i}` unless an entry dict has `label` / `id`.
  2. A single container key (first key *present* in `key_order` wins).
  3. The payload itself: a dict mapping label -> text / dict (metadata keys
     skipped), or a list of strings / dicts (labels `h_{i}` unless given).
"""

from collections.abc import Callable, Mapping, Sequence
import json
import os
from typing import Any

# Single-container precedence shared by every caller.
KEY_ORDER: tuple[str, ...] = (
    "hypotheses",
    "rejected_hypotheses",
    "verified_hypotheses",
    "predictions",
)
# Keys holding a list of per-method entries inside a hypotheses file.
METHOD_CONTAINER_KEYS: tuple[str, ...] = ("methods", "baseline_results")
# Per-method keys holding that method's hypotheses (first truthy wins).
METHOD_PREDICTION_KEYS: tuple[str, ...] = (
    "predictions",
    "hypotheses",
    "rejected_hypotheses",
)
# Keys holding hypothesis text inside an entry dict (first truthy wins).
TEXT_KEYS: tuple[str, ...] = ("text", "hypothesis", "claim", "content")
# Dict keys that are never hypotheses when a dict maps label -> hypothesis.
METADATA_KEYS: frozenset[str] = frozenset({
    "task_suite",
    "task_name",
    "config",
    "experiment_name",
    "target_model",
    "investigator_model",
    "methods",
    "baseline_results",
})


def first_truthy(
    data: Mapping[str, Any], keys: Sequence[str], default: Any = None
) -> Any:
  """Returns `data.get(k0) or data.get(k1) or ... or default`."""
  for key in keys:
    val = data.get(key)
    if val:
      return val
  return default


def _entry_text(entry: Mapping[str, Any]) -> str:
  """Returns the stripped hypothesis text of an entry dict (may be empty)."""
  return str(first_truthy(entry, TEXT_KEYS, "")).strip()


def method_entries(
    data: Any, containers: Sequence[str] = METHOD_CONTAINER_KEYS
) -> list[Any] | None:
  """Returns the non-empty per-method entry list of `data`, else None."""
  if not isinstance(data, dict):
    return None
  methods_list = first_truthy(data, containers)
  if isinstance(methods_list, list) and methods_list:
    return methods_list
  return None


def _extract_from_methods(methods_list: list[Any]) -> list[tuple[str, str]]:
  """Extracts (label, text) pairs from a list of per-method entry dicts."""
  extracted: list[tuple[str, str]] = []
  for m_entry in methods_list:
    if not isinstance(m_entry, dict):
      continue
    m_name = m_entry.get("method_name") or m_entry.get("method") or "baseline"
    preds = first_truthy(m_entry, METHOD_PREDICTION_KEYS, [])
    for i, pred in enumerate(preds):
      if isinstance(pred, str) and pred.strip():
        extracted.append((f"{m_name}_h_{i}", pred.strip()))
      elif isinstance(pred, dict):
        txt = _entry_text(pred)
        if txt:
          lbl = pred.get("label") or pred.get("id") or f"{m_name}_h_{i}"
          extracted.append((str(lbl), txt))
  return extracted


def extract_hypotheses(
    data: Any,
    *,
    key_order: Sequence[str] = KEY_ORDER,
    strict_type: bool = False,
    source: str = "",
) -> list[tuple[str, str]]:
  """Extracts (label, text) pairs from parsed hypotheses JSON.

  Args:
    data: Parsed JSON payload.
    key_order: Precedence of the single-container keys; the first key present in
      a dict payload selects the sub-payload.
    strict_type: If True, raise when the selected payload is neither a dict nor
      a list; otherwise return [].
    source: Path used in the error message when `strict_type` is set.

  Returns:
    List of (label, hypothesis_text) pairs with non-empty stripped text.

  Raises:
    ValueError: If `strict_type` and the payload is not a dict or list.
  """
  if isinstance(data, dict):
    methods_list = method_entries(data, METHOD_CONTAINER_KEYS)
    if methods_list is not None:
      extracted = _extract_from_methods(methods_list)
      if extracted:
        return [(lbl, txt) for lbl, txt in extracted if txt]

    for key in key_order:
      if key in data:
        data = data[key]
        break

  result: list[tuple[str, str]] = []
  if isinstance(data, dict):
    for label, val in data.items():
      if str(label).lower() in METADATA_KEYS:
        continue
      if isinstance(val, str) and val.strip():
        result.append((str(label), val.strip()))
      elif isinstance(val, dict):
        text = _entry_text(val)
        if text:
          result.append((str(label), text))
  elif isinstance(data, list):
    for i, item in enumerate(data):
      if isinstance(item, str) and item.strip():
        result.append((f"h_{i}", item.strip()))
      elif isinstance(item, dict):
        text = _entry_text(item)
        if text:
          label = item.get("label") or item.get("id") or f"h_{i}"
          result.append((str(label), text))
  elif strict_type:
    raise ValueError(
        f"Unexpected JSON format in {source}: expected dict or list, got"
        f" {type(data).__name__}"
    )

  return [(lbl, txt) for lbl, txt in result if txt]


def resolve_file_path(file_path: str) -> str:
  """Resolves a file path that may be relative to the package root or cwd."""
  if not file_path or os.path.exists(file_path):
    return file_path

  pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  for base in (pkg_root, os.getcwd()):
    candidate = os.path.join(base, file_path)
    if os.path.exists(candidate):
      return candidate

  return file_path


def read_hypotheses_json(path: str) -> Any:
  """Reads a hypotheses JSON file, resolving relative paths.

  Args:
    path: File path (absolute or relative).

  Returns:
    The parsed JSON payload.

  Raises:
    FileNotFoundError: If the resolved path does not exist.
  """
  resolved_path = resolve_file_path(path)
  if not os.path.exists(resolved_path):
    raise FileNotFoundError(
        f"Hypotheses file not found: {path} (resolved as: {resolved_path})"
    )
  with open(resolved_path, "r", encoding="utf-8") as f:
    return json.load(f)


def load_hypotheses_and_data(
    path: str,
    *,
    key_order: Sequence[str] = KEY_ORDER,
    strict_type: bool = True,
    reader: Callable[[str], Any] = read_hypotheses_json,
) -> tuple[list[tuple[str, str]], Any]:
  """Loads a hypotheses file; returns (label, text) pairs and the raw JSON."""
  data = reader(path)
  return (
      extract_hypotheses(
          data, key_order=key_order, strict_type=strict_type, source=path
      ),
      data,
  )


def load_hypotheses(
    path: str,
    *,
    key_order: Sequence[str] = KEY_ORDER,
    strict_type: bool = True,
) -> list[tuple[str, str]]:
  """Loads (label, text) pairs from a hypotheses JSON file."""
  hyps, _ = load_hypotheses_and_data(
      path, key_order=key_order, strict_type=strict_type
  )
  return hyps
