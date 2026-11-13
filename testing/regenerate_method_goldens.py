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

r"""Regenerates the discovery-method goldens (see method_cases).

Kept out of regenerate_goldens because importing eval_discovery_baselines
defines flags (e.g. --output_dir) that clash with flags of that binary. Only
run this after an intentional behavior change.

Usage:
  python -m disco.testing.regenerate_method_goldens \
    --golden_dir=testing/testdata
"""

from collections.abc import Sequence

from absl import app
from absl import flags

from disco.testing import golden_utils
from disco.testing import method_cases

_GOLDEN_DIR = flags.DEFINE_string(
    "golden_dir",
    None,
    "Directory to write golden files to (the testing/testdata dir).",
    required=True,
)
_ONLY = flags.DEFINE_list("only", [], "If set, only regenerate these goldens.")


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  for name, case_fn in method_cases.ALL_CASES.items():
    if _ONLY.value and name not in _ONLY.value:
      continue
    path = golden_utils.write_golden(_GOLDEN_DIR.value, name, case_fn())
    print(f"Wrote {path}")


if __name__ == "__main__":
  app.run(main)
