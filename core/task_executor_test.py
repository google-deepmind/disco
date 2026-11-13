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

"""Tests for task executor selection and flags."""

from absl.testing import absltest
from absl.testing import parameterized

from disco.core import config as config_lib
from disco.core import task_executor


def _config(seed: str, suite: str = "") -> config_lib.DiscoConfig:
  return config_lib.DiscoConfig(
      target_model="t",
      seed_instruction=seed,
      budget=9,
      alpha=0.1,
      kappa=0.5,
      task_suite=suite,
  )


def _noop_log(*args, **kwargs) -> None:
  del args, kwargs


class ResolveTaskExecutorTest(parameterized.TestCase):

  @parameterized.parameters(
      ("Count the words.", "", "text"),
      ("Count the words.", "hedging", "text"),
      ("Count the words.", "visual_symmetry", "visual_symmetry"),
      ("Test Visual Symmetry completion.", "", "visual_symmetry"),
      ("reflection SYMMETRY", "", "visual_symmetry"),
      ("Test visual symmetry.", "visual_symmetry", "visual_symmetry"),
      # A set task_suite decides; the seed only counts when it is empty.
      ("Test visual symmetry.", "hedging", "text"),
  )
  def test_resolve(self, seed, suite, expected):
    executor = task_executor.resolve_task_executor(
        _config(seed, suite), _noop_log
    )
    self.assertEqual(executor.name, expected)

  def test_seed_fallback_warns(self):
    with self.assertLogs(level="WARNING") as logs:
      task_executor.resolve_task_executor(
          _config("Visual symmetry task"), _noop_log
      )
    self.assertIn("inferred from the seed", logs.output[0])


class ExecutorFlagsTest(absltest.TestCase):

  def test_text(self):
    ex = task_executor.TextTaskExecutor(_config("s"), _noop_log)
    self.assertFalse(ex.exclude_analyze)
    self.assertTrue(ex.uses_preregistration)
    self.assertTrue(ex.uses_text_querygen)
    self.assertAlmostEqual(ex.null_rate(), 0.5)
    cfg = _config("s")
    cfg.task_null_proportion = 0.2
    self.assertAlmostEqual(
        task_executor.TextTaskExecutor(cfg, _noop_log).null_rate(), 0.2
    )

  def test_symmetry(self):
    ex = task_executor.SymmetryTaskExecutor(_config("symmetry"), _noop_log)
    self.assertTrue(ex.exclude_analyze)
    self.assertFalse(ex.uses_preregistration)
    self.assertFalse(ex.uses_text_querygen)

  def test_symmetry_null_rate_prefers_seed(self):
    ex = task_executor.SymmetryTaskExecutor(
        _config("Symmetry with a 70% null rate"), _noop_log
    )
    self.assertAlmostEqual(ex.null_rate(), 0.7)
    ex = task_executor.SymmetryTaskExecutor(_config("Symmetry"), _noop_log)
    self.assertAlmostEqual(ex.null_rate(), 0.9)


if __name__ == "__main__":
  absltest.main()
