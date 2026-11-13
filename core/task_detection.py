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

"""Decides whether a run is a visual symmetry task.

Every site that needs to know (agent construction, the round's task executor,
the verifier, isolated query generation) uses `is_symmetry_task`, so they can
never disagree:

  * `task_suite` decides whenever it is set: a run is a symmetry task iff
    `task_suite == "visual_symmetry"`.
  * Only when `task_suite` is empty does the seed instruction count: a seed
    mentioning "symmetry" makes it a symmetry task, with a warning to set
    `task_suite` instead.

The sites used to apply three different rules (seed substring only; task_suite
or seed; never), so e.g. the verifier could treat a run as symmetry while the
discovery agent did not.
"""

from typing import Any

from absl import logging

SYMMETRY_SUITE = "visual_symmetry"


def is_symmetry_task(cfg: Any) -> bool:
  """Returns whether `cfg` (a DiscoConfig or VerificationConfig) is symmetry.

  Args:
    cfg: Run config with `task_suite` and `seed_instruction` (either may be
      missing or None).
  """
  suite = getattr(cfg, "task_suite", "") or ""
  if suite:
    return suite == SYMMETRY_SUITE
  seed_lower = (getattr(cfg, "seed_instruction", "") or "").lower()
  if "symmetry" not in seed_lower:
    return False
  logging.warning(
      "Visual symmetry task inferred from the seed instruction; set"
      " task_suite='visual_symmetry' instead."
  )
  return True
