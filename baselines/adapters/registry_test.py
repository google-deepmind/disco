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

"""Tests for baseline task adapter registry."""

from absl.testing import absltest

from disco.baselines.adapters import registry
from disco.baselines.adapters import t2i
from disco.tasks import data_classes
from disco.tasks import registry as tasks_registry


class RegistryTest(absltest.TestCase):

  def test_every_registered_task_has_a_registered_adapter(self):
    for suite in tasks_registry.list_suites():
      for task in tasks_registry.list_specs(suite):
        with self.subTest(task=task.key):
          self.assertIn(task.adapter, registry.ADAPTERS)

  def test_options_name_registered_adapters(self):
    self.assertContainsSubset(registry._ADAPTER_OPTIONS, registry.ADAPTERS)

  def test_get_task_adapter_uses_registry(self):
    for name, adapter_cls in registry.ADAPTERS.items():
      with self.subTest(adapter=name):
        task = data_classes.Task(
            name="Test task", description="A task.", adapter=name
        )
        self.assertIs(type(registry.get_task_adapter(task)), adapter_cls)

  def test_get_task_adapter_passes_options(self):
    t2i_adapter = registry.get_task_adapter(
        data_classes.Task(name="t", description="d", adapter="t2i_bias"),
        judge_model="models/judge-x",
    )
    assert isinstance(t2i_adapter, t2i.T2IBiasTaskAdapter)
    self.assertEqual(t2i_adapter.judge_model, "judge-x")

  def test_missing_adapter_raises(self):
    task = data_classes.Task(
        name="Recipe length", description="d", suite="my_recipes"
    )
    with self.assertRaisesRegex(ValueError, "sets no adapter"):
      registry.get_task_adapter(task)

  def test_unknown_adapter_raises(self):
    task = data_classes.Task(
        name="Test task", description="A task.", adapter="nope"
    )
    with self.assertRaisesRegex(
        ValueError, "Task 'Test task' has unknown adapter 'nope'"
    ):
      registry.get_task_adapter(task)


if __name__ == "__main__":
  absltest.main()
