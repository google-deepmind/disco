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

"""Loads the built-in task suites from `tasks/data/<suite>.yaml`.

Each file holds one suite:

  suite: hedging                 # registry name of the suite
  default_task: <task key>       # what `registry.resolve("<suite>")` returns
  tasks:
    - name: ...                  # TaskSpec fields; every key that is not a
      description: ...           # TaskSpec field ends up in `extra`
      split: ...
      ...

Every task key except `name`, `description`, `split` and the spec-only fields
(`spec.SPEC_ONLY_FIELDS`) is a legacy metadata key, and is listed in the
spec's `metadata_keys` in file order, so `TaskSpec.metadata` reproduces the
dict the task had when it was a Python literal.

This module depends only on `spec`, so the task modules can derive their
constants from it without importing the registry.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import functools
import pathlib
from typing import Any

import yaml

from disco.tasks import spec

_DATA_DIR = pathlib.Path(__file__).parent / "data"

# Built-in suites, in registration order (which is the order of
# `registry.get_tasks(suite="all")`).
BUILTIN_SUITES: tuple[str, ...] = (
    "recipes",
    "hedging",
    "visual_symmetry",
    "t2i_cat_bias",
    "t2i_handedness_bias",
)

_CORE_KEYS: frozenset[str] = frozenset({"name", "description", "split"})
_TYPED_KEYS: frozenset[str] = frozenset(
    spec.METADATA_FIELDS + spec.SPEC_ONLY_FIELDS
)


@dataclasses.dataclass(frozen=True)
class SuiteData:
  """One suite as loaded from its YAML file.

  Attributes:
    suite: Registry name of the suite.
    default_task: Key of the task `registry.resolve(suite)` returns.
    tasks: The suite's tasks, in file order.
  """

  suite: str
  default_task: str
  tasks: tuple[spec.TaskSpec, ...]

  def by_key(self, key: str) -> spec.TaskSpec:
    """Returns the task whose key is `key`."""
    for task in self.tasks:
      if task.key == key:
        return task
    raise KeyError(f"{self.suite}: no task with key {key!r}")


def _task_from_entry(suite: str, entry: Mapping[str, Any]) -> spec.TaskSpec:
  """Builds a TaskSpec from one `tasks:` entry of a suite file."""
  missing = sorted(k for k in ("name", "description") if k not in entry)
  if missing:
    raise ValueError(f"{suite}: task entry lacks {missing}: {dict(entry)}")
  if "suite" in entry:
    raise ValueError(f"{suite}: tasks take their suite from the file")
  typed = {k: v for k, v in entry.items() if k in _TYPED_KEYS}
  extra = {
      k: v
      for k, v in entry.items()
      if k not in _TYPED_KEYS and k not in _CORE_KEYS
  }
  metadata_keys = [
      k for k in entry if k not in _CORE_KEYS and k not in spec.SPEC_ONLY_FIELDS
  ]
  return spec.TaskSpec(
      name=entry["name"],
      description=entry["description"],
      split=entry.get("split", "all"),
      suite=suite,
      extra=extra,
      metadata_keys=metadata_keys,
      **typed,
  )


def parse_suite(text: str) -> SuiteData:
  """Parses the contents of one suite YAML file.

  Args:
    text: YAML text.

  Returns:
    The parsed suite.

  Raises:
    ValueError: If the file is malformed, or `default_task` names no task.
  """
  doc = yaml.safe_load(text)
  if not isinstance(doc, dict) or set(doc) != {
      "suite",
      "default_task",
      "tasks",
  }:
    raise ValueError(
        "A suite file must have exactly the keys suite, default_task and"
        f" tasks; got {sorted(doc) if isinstance(doc, dict) else doc!r}"
    )
  suite = doc["suite"]
  entries: Sequence[Mapping[str, Any]] = doc["tasks"]
  tasks = tuple(_task_from_entry(suite, entry) for entry in entries)
  keys = [t.key for t in tasks]
  if len(set(keys)) != len(keys):
    raise ValueError(f"{suite}: duplicate task keys {keys}")
  if doc["default_task"] not in keys:
    raise ValueError(
        f"{suite}: default_task {doc['default_task']!r} is not one of {keys}"
    )
  return SuiteData(suite=suite, default_task=doc["default_task"], tasks=tasks)


@functools.cache
def load_suite(suite: str) -> SuiteData:
  """Loads built-in suite `suite` from `tasks/data/<suite>.yaml`.

  Cached, so every caller (the registry and the task modules) shares the same
  TaskSpec objects.

  Args:
    suite: One of `BUILTIN_SUITES`.

  Returns:
    The loaded suite.

  Raises:
    ValueError: If `suite` is not built in, or its file is malformed.
  """
  if suite not in BUILTIN_SUITES:
    raise ValueError(f"Unknown built-in suite {suite!r}: {BUILTIN_SUITES}")
  text = (_DATA_DIR / f"{suite}.yaml").read_text(encoding="utf-8")
  data = parse_suite(text)
  if data.suite != suite:
    raise ValueError(f"{suite}.yaml declares suite {data.suite!r}")
  return data
