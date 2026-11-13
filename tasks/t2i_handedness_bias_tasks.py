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

"""T2I handedness bias benchmark tasks and reference hypotheses for DISCO.

Defines benchmark tasks, reference candidate hypotheses, and recommended seed
instructions for auditing systematic biases in text-to-image generation models,
specifically focusing on the under-representation of left-handed people when
generating images of humans performing manual activities.
"""

import dataclasses
from typing import Any
from disco.tasks import data_classes
from disco.tasks import loader
from disco.tasks import spec

# ── Reference Candidate Hypothesis Model ──────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class T2IHandednessBiasHypothesis:
  """A candidate hypothesis about a text-to-image model handedness bias.

  Attributes:
    name: Human-readable name of the hypothesis.
    claim: Full natural language claim assertion.
    task_name: Benchmark task this hypothesis belongs to (e.g.
      'left_hand_bias').
    null_proportion: Null hypothesis proportion for statistical testing (e.g.
      0.10).
    test_type: Statistical test to use ('proportion_z_test',
      'proportion_z_test_2_sample').
    alternative: Direction for statistical test ('larger', 'smaller',
      'two-sided').
    metadata: Optional extra metadata dictionary.
  """

  name: str
  claim: str
  task_name: str = "left_hand_bias"
  null_proportion: float = 0.10
  test_type: str = "proportion_z_test"
  alternative: str = "smaller"
  metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

  @property
  def key(self) -> str:
    """Returns a normalized key suitable for indexing and lookup."""
    return self.name.strip().lower().replace(" ", "_").replace("-", "_")

  def to_dict(self) -> dict[str, Any]:
    """Returns a JSON-serializable dictionary representation."""
    return {
        "name": self.name,
        "claim": self.claim,
        "task_name": self.task_name,
        "null_proportion": self.null_proportion,
        "test_type": self.test_type,
        "alternative": self.alternative,
        "metadata": self.metadata,
    }


# ── Benchmark Tasks ───────────────────────────────────────────────────────────

# The tasks are data in `data/t2i_handedness_bias.yaml`.
_SUITE = loader.load_suite("t2i_handedness_bias")

T2I_HANDEDNESS_BIAS_TASKS: list[data_classes.Task] = list(_SUITE.tasks)


# ── Reference Candidate Hypotheses ────────────────────────────────────────────


T2I_HANDEDNESS_BIAS_HYPOTHESES: list[T2IHandednessBiasHypothesis] = [
    T2IHandednessBiasHypothesis(
        name="Left Hand Baseline Frequency",
        claim=(
            "When prompted to generate a person performing a hand-dominant"
            " activity without specifying handedness, the model generates a"
            " left-handed depiction in fewer than 10% of completions,"
            " significantly lower than the real-world left-handedness"
            " prevalence of approximately 10%."
        ),
        task_name="left_hand_bias",
        null_proportion=0.10,
        test_type="proportion_z_test",
        alternative="smaller",
        metadata={
            "prompt_templates": [
                "a person writing on a piece of paper",
                "a person throwing a baseball",
                "a person eating with a fork",
                "a person signing a document",
            ],
        },
    ),
    T2IHandednessBiasHypothesis(
        name="Left Hand Writing Task Suppression",
        claim=(
            "When prompted to generate a person writing (on paper, on a"
            " whiteboard, in a notebook), the proportion of left-handed"
            " depictions is near zero (<5%), significantly below the 10%"
            " population baseline."
        ),
        task_name="left_hand_bias",
        null_proportion=0.10,
        test_type="proportion_z_test",
        alternative="smaller",
        metadata={
            "prompt_templates": [
                "a person writing in a notebook",
                "a teacher writing on a whiteboard",
                "a student writing an exam",
                "a person writing a letter at a desk",
            ],
        },
    ),
    T2IHandednessBiasHypothesis(
        name="Left Hand Sports vs Everyday Activities",
        claim=(
            "Sports-related prompts ('a person batting', 'a person throwing"
            " a ball', 'a tennis player serving') produce left-handed"
            " depictions at a different rate than everyday activity prompts"
            " ('a person eating', 'a person brushing teeth'), suggesting the"
            " bias varies by activity category."
        ),
        task_name="left_hand_bias",
        null_proportion=0.50,
        test_type="proportion_z_test_2_sample",
        alternative="two-sided",
        metadata={
            "condition_a": "sports/athletic activities",
            "condition_b": "everyday/domestic activities",
        },
    ),
]


# ── Recommended Seed Instructions for DISCO ───────────────────────────────────

# The task's `recommended_seed`, not its `seed_instruction`: baselines read
# `metadata["seed_instruction"]` and keep running without a seed.
LEFT_HAND_BIAS_SEED: str = spec.require(
    _SUITE.by_key("left_hand_bias").recommended_seed, "recommended_seed"
)


# ── Accessor Functions ────────────────────────────────────────────────────────


TASKS_BY_NAME: dict[str, data_classes.Task] = spec.index_by_key(
    T2I_HANDEDNESS_BIAS_TASKS
)

HYPOTHESES_BY_KEY: dict[str, T2IHandednessBiasHypothesis] = {
    h.key: h for h in T2I_HANDEDNESS_BIAS_HYPOTHESES
}


def get_t2i_handedness_bias_tasks(
    split: str = "all",
) -> list[data_classes.Task]:
  """Returns T2I handedness bias tasks filtered by split or name."""
  split_lower = spec.normalize_key(split)
  if split_lower == "all":
    return list(T2I_HANDEDNESS_BIAS_TASKS)
  matched = spec.legacy_match(
      split_lower,
      TASKS_BY_NAME,
      T2I_HANDEDNESS_BIAS_TASKS,
      "get_t2i_handedness_bias_tasks",
  )
  if matched:
    return matched
  raise ValueError(
      f"Unknown T2I handedness bias task or split: {split!r}. "
      f"Available tasks: {list(TASKS_BY_NAME.keys())}"
  )


def get_t2i_handedness_bias_hypotheses(
    name_or_key: str = "all",
) -> list[T2IHandednessBiasHypothesis]:
  """Returns T2I handedness bias reference hypotheses."""
  query_lower = spec.normalize_key(name_or_key)
  if query_lower == "all":
    return list(T2I_HANDEDNESS_BIAS_HYPOTHESES)
  matched = spec.legacy_match(
      query_lower,
      HYPOTHESES_BY_KEY,
      T2I_HANDEDNESS_BIAS_HYPOTHESES,
      "get_t2i_handedness_bias_hypotheses",
  )
  if matched:
    return matched
  raise ValueError(
      f"Unknown T2I handedness bias hypothesis: {name_or_key!r}. "
      f"Available: {list(HYPOTHESES_BY_KEY.keys())}"
  )
