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

"""Base data classes and data models for DISCO benchmark tasks and hypotheses."""

import dataclasses
from typing import Any

from disco.tasks import spec

# A single benchmark evaluation task. Still constructible as
# `Task(name, description, split, suite, metadata)`; see `spec.TaskSpec` for
# the typed fields and the legacy `metadata` view.
Task = spec.TaskSpec


@dataclasses.dataclass(frozen=True)
class Hypothesis:
  """A candidate hypothesis about model capability under specific conditions.

  Attributes:
    name: Human-readable name of the hypothesis (e.g., 'Symmetry English Letters
      1to1').
    claim: Full natural language claim / hypothesis assertion.
    task_suite: Benchmark task suite this hypothesis targets (e.g.,
      'visual_symmetry').
    condition: Structured parameters defining the test condition (e.g.
      shape_type, grid size).
    null_proportion: Null hypothesis proportion for statistical testing (e.g.,
      0.75).
    metadata: Optional extra metadata dictionary.
  """

  name: str
  claim: str
  task_suite: str = "visual_symmetry"
  condition: dict[str, Any] = dataclasses.field(default_factory=dict)
  null_proportion: float = 0.75
  metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

  @property
  def key(self) -> str:
    """Returns a normalized key suitable for indexing and lookup."""
    return self.name.strip().lower().replace(" ", "_").replace("-", "_")

  def to_dict(self) -> dict[str, Any]:
    """Returns a JSON-serializable dictionary representation of the hypothesis."""
    result: dict[str, Any] = {
        "name": self.name,
        "claim": self.claim,
        "task_suite": self.task_suite,
        "condition": self.condition,
        "null_proportion": self.null_proportion,
    }
    if self.metadata:
      result["metadata"] = self.metadata
    return result
