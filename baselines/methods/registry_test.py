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

"""Tests for the discovery methods registry."""

import inspect

from absl.testing import absltest

from disco.baselines import eval_discovery_baselines
from disco.baselines.methods import base
from disco.baselines.methods import direct_asking
from disco.baselines.methods import looped
from disco.baselines.methods import red_team
from disco.baselines.methods import registry
from disco.baselines.methods import simple_probing

# The function each method's `run` forwards its plan to.
_RUN_FNS = {
    "direct_asking": direct_asking.run_direct_asking_baseline,
    "simple_probing": simple_probing.run_simple_probing_baseline,
    "red_team": red_team.run_red_team_probing_baseline,
    "looped_investigator": looped.run_looped_investigator_baseline,
    "looped_multisample": looped.run_looped_multisample_baseline,
}


class RegistryTest(absltest.TestCase):

  def test_registers_every_method_the_pipeline_knows(self):
    self.assertCountEqual(
        registry.METHODS,
        eval_discovery_baselines._METHOD_TO_CLASS_NAME,
    )

  def test_methods_are_registered_under_their_names(self):
    for name, method in registry.METHODS.items():
      with self.subTest(name):
        self.assertEqual(method.name, name)

  def test_plans_only_name_parameters_of_run(self):
    for name, method in registry.METHODS.items():
      with self.subTest(name):
        plan = method.plan(300, base.PlanOptions())
        self.assertContainsSubset(
            plan, inspect.signature(_RUN_FNS[name]).parameters
        )


if __name__ == "__main__":
  absltest.main()
