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

r"""Helpers for golden-value regression tests.

Golden tests pin the numerical output of deterministic code (statistics, table
aggregation) so that refactors can be checked for bit-level behavior
preservation. Each golden file is a JSON document produced by a case function
in this package; the test recomputes the cases and compares them to the
checked-in file.

To regenerate goldens after an intentional behavior change, run:

  python -m disco.testing.regenerate_goldens --output_dir=testing/testdata
"""

from collections.abc import Mapping, Sequence
import dataclasses
import json
import math
import os
import pathlib
from typing import Any

import numpy as np

_TESTDATA_DIR = pathlib.Path(__file__).parent / "testdata"


def _encode_float(value: float) -> float | str:
  """Returns value, or a string label if it is not finite (strict JSON)."""
  if math.isnan(value):
    return "NaN"
  if math.isinf(value):
    return "Infinity" if value > 0 else "-Infinity"
  return value


def to_jsonable(value: Any) -> Any:
  """Converts nested numpy/dataclass/set values into JSON-serializable form.

  Sets are sorted so that the serialized form is deterministic. Dataclasses
  are converted field by field, and pydantic models through `model_dump()`.
  Numpy arrays become nested lists. Non-finite floats become the strings
  "NaN", "Infinity" and "-Infinity" so the output is strict JSON.

  Args:
    value: The value to convert.

  Returns:
    A structure made only of dicts, lists, strings, numbers, bools and None.
  """
  if dataclasses.is_dataclass(value) and not isinstance(value, type):
    return {
        field.name: to_jsonable(getattr(value, field.name))
        for field in dataclasses.fields(value)
    }
  if not isinstance(value, type) and callable(
      getattr(value, "model_dump", None)
  ):
    return to_jsonable(value.model_dump())
  if isinstance(value, np.ndarray):
    return to_jsonable(value.tolist())
  if isinstance(value, (np.bool_, bool)):
    return bool(value)
  if isinstance(value, (np.floating, float)):
    return _encode_float(float(value))
  if isinstance(value, np.integer):
    return int(value)
  if isinstance(value, Mapping):
    return {str(k): to_jsonable(v) for k, v in value.items()}
  if isinstance(value, (set, frozenset)):
    return sorted(to_jsonable(v) for v in value)
  if isinstance(value, (list, tuple)):
    return [to_jsonable(v) for v in value]
  return value


def golden_path(name: str) -> str:
  """Returns the path of the checked-in golden file `name`.json."""
  return str(_TESTDATA_DIR / f"{name}.json")


def load_golden(name: str) -> Any:
  """Loads the checked-in golden file `name`.json."""
  with open(golden_path(name)) as f:
    return json.load(f)


def write_golden(output_dir: str, name: str, value: Any) -> str:
  """Writes `value` as the golden file `name`.json under `output_dir`."""
  path = os.path.join(output_dir, f"{name}.json")
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w") as f:
    json.dump(to_jsonable(value), f, indent=1, sort_keys=True, allow_nan=False)
    f.write("\n")
  return path


def _floats_match(actual: float, expected: float, rtol: float) -> bool:
  if math.isnan(expected):
    return math.isnan(actual)
  if math.isinf(expected):
    return actual == expected
  return math.isclose(actual, expected, rel_tol=rtol, abs_tol=rtol * 1e-3)


def find_mismatches(
    actual: Any, expected: Any, rtol: float = 1e-9, path: str = "$"
) -> list[str]:
  """Recursively compares two JSON-like structures.

  Floats are compared with relative tolerance `rtol`; every other type must
  match exactly.

  Args:
    actual: Freshly computed structure (already passed through to_jsonable).
    expected: Structure loaded from the golden file.
    rtol: Relative tolerance for float comparison.
    path: JSONPath-like location used in mismatch messages.

  Returns:
    Human-readable descriptions of every mismatch (empty if they match).
  """
  if isinstance(expected, Mapping):
    if not isinstance(actual, Mapping):
      return [f"{path}: expected object, got {type(actual).__name__}"]
    mismatches = []
    for key in sorted(set(expected) | set(actual)):
      if key not in actual:
        mismatches.append(f"{path}.{key}: missing from actual")
      elif key not in expected:
        mismatches.append(f"{path}.{key}: unexpected key in actual")
      else:
        mismatches.extend(
            find_mismatches(actual[key], expected[key], rtol, f"{path}.{key}")
        )
    return mismatches
  if isinstance(expected, Sequence) and not isinstance(expected, str):
    if not isinstance(actual, Sequence) or isinstance(actual, str):
      return [f"{path}: expected list, got {type(actual).__name__}"]
    if len(actual) != len(expected):
      return [f"{path}: length {len(actual)} != expected {len(expected)}"]
    mismatches = []
    for i, (a, e) in enumerate(zip(actual, expected)):
      mismatches.extend(find_mismatches(a, e, rtol, f"{path}[{i}]"))
    return mismatches
  if isinstance(expected, bool) or isinstance(actual, bool):
    if actual is expected:
      return []
    return [f"{path}: {actual!r} != expected {expected!r}"]
  numbers = (int, float)
  if (
      isinstance(actual, numbers)
      and isinstance(expected, numbers)
      and (isinstance(actual, float) or isinstance(expected, float))
  ):
    if _floats_match(float(actual), float(expected), rtol):
      return []
    return [f"{path}: {actual!r} != expected {expected!r} (rtol={rtol})"]
  if actual != expected:
    return [f"{path}: {actual!r} != expected {expected!r}"]
  return []
