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

"""Tests for the `random_seed` of the p-value randomization RNGs."""

import dataclasses
from typing import Any, cast

from absl.testing import absltest
import numpy as np

from disco import precision_estimation
from disco import runners
from disco.testing import disco_runner_smoke_cases


def _runner(out_dir: str, seed: int | None) -> runners.DiscoRunner:
  config = dataclasses.replace(
      disco_runner_smoke_cases.make_config("disco_post_hoc", out_dir),
      random_seed=seed,
  )
  return runners.DiscoRunner(config, cast(Any, None))


class RunnerSeedTest(absltest.TestCase):

  def test_same_seed_gives_same_draws(self):
    out_dir = self.create_tempdir().full_path
    np.testing.assert_array_equal(
        _runner(out_dir, 3)._pval_rng.random(4),
        _runner(out_dir, 3)._pval_rng.random(4),
    )

  def test_different_seeds_give_different_draws(self):
    out_dir = self.create_tempdir().full_path
    self.assertFalse(
        np.array_equal(
            _runner(out_dir, 3)._pval_rng.random(4),
            _runner(out_dir, 4)._pval_rng.random(4),
        )
    )

  def test_resume_continues_the_stream(self):
    out_dir = self.create_tempdir().full_path
    first = _runner(out_dir, 3)
    first._pval_rng.random(5)
    first._save_checkpoint()
    want = first._pval_rng.random(4)

    resumed = _runner(out_dir, 3)
    self.assertTrue(resumed._load_checkpoint())
    np.testing.assert_array_equal(resumed._pval_rng.random(4), want)

  def test_unseeded_run_saves_no_rng_state(self):
    out_dir = self.create_tempdir().full_path
    runner = _runner(out_dir, None)
    runner._save_checkpoint()
    self.assertIsNone(runner.state.pval_rng_state)
    resumed = _runner(out_dir, None)
    self.assertTrue(resumed._load_checkpoint())
    self.assertIsNone(resumed.state.pval_rng_state)


class HypothesisRngTest(absltest.TestCase):

  def test_same_seed_and_id_give_same_draws(self):
    np.testing.assert_array_equal(
        precision_estimation._hypothesis_rng(5, "h_1").random(4),
        precision_estimation._hypothesis_rng(5, "h_1").random(4),
    )

  def test_ids_get_different_streams(self):
    self.assertFalse(
        np.array_equal(
            precision_estimation._hypothesis_rng(5, "h_1").random(4),
            precision_estimation._hypothesis_rng(5, "h_2").random(4),
        )
    )

  def test_seeds_give_different_streams(self):
    self.assertFalse(
        np.array_equal(
            precision_estimation._hypothesis_rng(5, "h_1").random(4),
            precision_estimation._hypothesis_rng(6, "h_1").random(4),
        )
    )


if __name__ == "__main__":
  absltest.main()
