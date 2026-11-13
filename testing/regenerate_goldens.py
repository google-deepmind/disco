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

r"""Regenerates the golden files used by the golden regression tests.

Only run this after an intentional behavior change, and describe the change
in the commit/PR that updates the goldens. JSON goldens go to `<output_dir>/`,
and rendered tables go to `<output_dir>/table_goldens/<fixture>/<variant>.txt`.

Usage:
  python -m disco.testing.regenerate_goldens --output_dir=testing/testdata
"""

from collections.abc import Sequence

from absl import app
from absl import flags

from disco.testing import adapter_cases
from disco.testing import category_pool_cases
from disco.testing import config_flag_cases
from disco.testing import disco_runner_smoke_cases
from disco.testing import golden_utils
from disco.testing import inline_prompt_cases
from disco.testing import prompt_snapshot_cases
from disco.testing import stats_golden_cases
from disco.testing import task_constant_cases

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    None,
    "Directory to write golden files to (the testing/testdata dir).",
    required=True,
)
_ONLY = flags.DEFINE_list(
    "only",
    [],
    "If set, only regenerate these goldens.",
)


def _selected(name: str) -> bool:
  return not _ONLY.value or name in _ONLY.value


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  all_cases = {
      **stats_golden_cases.ALL_CASES,
      **prompt_snapshot_cases.ALL_CASES,
      **inline_prompt_cases.ALL_CASES,
      **config_flag_cases.ALL_CASES,
      **disco_runner_smoke_cases.ALL_CASES,
      **task_constant_cases.ALL_CASES,
      **category_pool_cases.ALL_CASES,
      **adapter_cases.ALL_CASES,
  }
  for name, case_fn in all_cases.items():
    if not _selected(name):
      continue
    path = golden_utils.write_golden(_OUTPUT_DIR.value, name, case_fn())
    print(f"Wrote {path}")


if __name__ == "__main__":
  app.run(main)
