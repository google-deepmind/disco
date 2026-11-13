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

r"""Regenerates the golden testdata used by ``bandit_golden_test``.

Only run this when a behavior change is *intended*. For behavior-preserving
refactors the goldens must stay untouched and the test must keep passing.

Usage:
  python -m disco.bandit.generate_bandit_goldens \
    --golden_dir=bandit/testdata
"""

from collections.abc import Sequence
import json
import os
import tempfile
from typing import Any

from absl import app
from absl import flags

from disco.bandit import golden_cases

_GOLDEN_DIR = flags.DEFINE_string(
    "golden_dir",
    None,
    "Directory to write golden JSON files into.",
    required=True,
)
_ONLY = flags.DEFINE_enum(
    "only",
    "all",
    ["all", "result", "e2e"],
    "Which golden set(s) to regenerate.",
)


def _write(path: str, payload: dict[str, Any]) -> None:
  with open(path, "w") as f:
    json.dump(payload, f, indent=1, sort_keys=True)
    f.write("\n")
  print(f"Wrote {path} ({len(payload)} cases)")


def main(argv: Sequence[str]) -> None:
  """Regenerates result and end-to-end bandit golden JSON files."""
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  out_dir = _GOLDEN_DIR.value
  if not os.path.exists(out_dir):
    os.makedirs(out_dir, exist_ok=True)

  if _ONLY.value in ("all", "result"):
    result_goldens = {}
    for name, case in golden_cases.RESULT_CASES.items():
      result_goldens[name] = golden_cases.run_result_case(case)
      print(f"  result case done: {name}")
    _write(
        os.path.join(out_dir, golden_cases.RESULT_GOLDENS_FILENAME),
        result_goldens,
    )

  if _ONLY.value in ("all", "e2e"):
    e2e_goldens = {}
    for name, overrides in golden_cases.E2E_CASES.items():
      with tempfile.TemporaryDirectory() as tmp:
        e2e_goldens[name] = golden_cases.run_e2e_case(overrides, tmp)
      print(f"  e2e case done: {name}")
    _write(
        os.path.join(out_dir, golden_cases.E2E_GOLDENS_FILENAME), e2e_goldens
    )


if __name__ == "__main__":
  app.run(main)
