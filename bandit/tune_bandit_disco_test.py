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

"""Unit tests for the open-source Vizier bandit hyperparameter tuner."""

import os

from absl.testing import absltest
from absl.testing import flagsaver

from disco.bandit import tune_bandit_disco


class TuneBanditDiscoTest(absltest.TestCase):

  def test_build_study_config_single_objective(self):
    study_config = tune_bandit_disco.build_study_config(
        runner_type="disco",
        primary_metric="penalized_cluster_coverage",
        algorithm="RANDOM_SEARCH",
    )
    self.assertLen(study_config.metric_information, 1)
    metric = next(iter(study_config.metric_information))
    self.assertEqual(metric.name, "penalized_cluster_coverage")

  def test_run_study_two_trials_random_search(self):
    out_dir = self.create_tempdir().full_path
    with flagsaver.flagsaver(
        env_type="bernoulli",
        budget=20,
        chunk_size=10,
        num_mc_runs=2,
        n_sim=200,
        num_trials=2,
        algorithm="RANDOM_SEARCH",
        output_dir=out_dir,
    ):
      summary = tune_bandit_disco.run_study(
          runner_type="disco",
          study_id="test_disco_study",
          output_dir=out_dir,
          num_trials=2,
          primary_metric="penalized_cluster_coverage",
          algorithm="RANDOM_SEARCH",
          delta_threshold=0.05,
      )

    self.assertEqual(summary["runner_type"], "disco")
    self.assertEqual(summary["total_trials"], 2)
    self.assertEqual(summary["completed_trials"], 2)
    self.assertLen(summary["all_trials"], 2)
    self.assertIsNotNone(summary["best_observed_trial"])
    self.assertTrue(
        os.path.exists(os.path.join(out_dir, "disco_study_summary.json"))
    )


if __name__ == "__main__":
  absltest.main()
