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

"""Pytest configuration for DISCO's absltest-based test suite.

1. Marks `absl.flags.FLAGS` as parsed so `absltest.TestCase.create_tempdir()`
   and flag-default reads work without `absltest.main()`.
2. Runs test modules that import top-level CLI entry points in isolated
   subprocesses so their overlapping `absl.flags` definitions (such as
   `--alpha`, `--budget`, and `--target_model`) do not collide in a single
   `FLAGS` registry.
"""

import importlib.util
import pathlib
import subprocess
import sys
from typing import Any

from absl import flags
import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent

# Test modules that import top-level CLI binaries defining overlapping flags on
# the global `absl.flags.FLAGS` singleton.
_ISOLATED_CLI_TESTS = frozenset({
    "apply_fdr_control_test.py",
    "compute_disco_bound_values_test.py",
    "standalone_precision_verifier_test.py",
    "bandit/bandit_golden_test.py",
    "bandit/tune_bandit_disco_test.py",
    "baselines/eval_discovery_baselines_test.py",
    "baselines/generic_discovery_baselines_test.py",
    "core/run_config_test.py",
    "tasks/visual_symmetry/verify_visual_symmetry_test.py",
    "testing/disco_runner_smoke_test.py",
})


def pytest_configure(config: Any) -> None:
  """Marks absl flags as parsed when running under pytest."""
  del config
  flags.FLAGS.mark_as_parsed()


class _SubprocessAbslTestItem(pytest.Item):
  """Executes an absltest module in a clean Python subprocess."""

  def __init__(
      self, *, rel_path: str, module_path: pathlib.Path, **kwargs: Any
  ):
    super().__init__(name=rel_path, **kwargs)
    self._module_path = module_path

  def runtest(self) -> None:
    if (
        self.name == "bandit/tune_bandit_disco_test.py"
        and importlib.util.find_spec("vizier") is None
    ):
      pytest.skip("google-vizier is not installed (install with `.[bandit]`).")
    proc = subprocess.run(
        [sys.executable, str(self._module_path)],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
      output = (proc.stdout + "\n" + proc.stderr).strip()
      raise AssertionError(
          f"{self._module_path.relative_to(_REPO_ROOT)} exited with code"
          f" {proc.returncode}:\n{output}"
      )

  def repr_failure(self, excinfo: Any, style: Any = None) -> str:
    del style
    return str(excinfo.value)

  def reportinfo(self) -> tuple[pathlib.Path, int, str]:
    return self._module_path, 0, f"isolated absltest: {self.name}"


class _SubprocessAbslTestFile(pytest.File):
  """Collector wrapping a CLI test file as an isolated subprocess item."""

  def collect(self) -> list[_SubprocessAbslTestItem]:
    rel_path = self.path.resolve().relative_to(_REPO_ROOT).as_posix()
    return [
        _SubprocessAbslTestItem.from_parent(
            self,
            rel_path=rel_path,
            module_path=self.path.resolve(),
        )
    ]


def pytest_pycollect_makemodule(
    module_path: pathlib.Path, parent: Any
) -> Any:
  """Routes CLI-flag test modules to isolated subprocesses."""
  try:
    rel_path = module_path.resolve().relative_to(_REPO_ROOT).as_posix()
  except ValueError:
    return None
  if rel_path in _ISOLATED_CLI_TESTS:
    return _SubprocessAbslTestFile.from_parent(parent, path=module_path)
  return None
