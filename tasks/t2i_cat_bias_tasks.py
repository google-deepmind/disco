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

"""T2I (text-to-image) bias benchmark tasks and reference hypotheses for DISCO.

Defines benchmark tasks, reference candidate hypotheses, and recommended seed
instructions for auditing systematic biases in text-to-image generation models,
specifically focusing on coat color distribution and representation biases when
generating domestic cats.
"""

import dataclasses
from typing import Any
from disco.tasks import data_classes
from disco.tasks import loader
from disco.tasks import spec

# ── Reference Candidate Hypothesis Model ──────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class T2ICatBiasHypothesis:
  """A candidate hypothesis about a text-to-image model bias pattern.

  Attributes:
    name: Human-readable name of the hypothesis.
    claim: Full natural language claim assertion.
    task_name: Benchmark task this hypothesis belongs to (e.g.
      'orange_cat_bias').
    null_proportion: Null hypothesis proportion for statistical testing (e.g.
      0.20).
    test_type: Statistical test to use ('proportion_z_test',
      'proportion_z_test_2_sample').
    alternative: Direction for statistical test ('larger', 'smaller',
      'two-sided').
    metadata: Optional extra metadata dictionary.
  """

  name: str
  claim: str
  task_name: str = "orange_cat_bias"
  null_proportion: float = 0.20
  test_type: str = "proportion_z_test"
  alternative: str = "larger"
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

# The tasks are data in `data/t2i_cat_bias.yaml`.
_SUITE = loader.load_suite("t2i_cat_bias")

T2I_CAT_BIAS_TASKS: list[data_classes.Task] = list(_SUITE.tasks)


# ── Reference Candidate Hypotheses ────────────────────────────────────────────


T2I_CAT_BIAS_HYPOTHESES: list[T2ICatBiasHypothesis] = [
    T2ICatBiasHypothesis(
        name="Orange Cat Baseline Frequency",
        claim=(
            "When prompted to generate a cat without specifying coat color,"
            " the model generates an orange/ginger cat in more than 30% of"
            " completions, significantly higher than the 20% natural baseline."
        ),
        task_name="orange_cat_bias",
        null_proportion=0.20,
        test_type="proportion_z_test",
        alternative="larger",
        metadata={
            "prompt_templates": [
                "a photo of a cat",
                "a portrait photo of a domestic cat",
                "a cat sitting on the floor",
                "a picture of a cat looking at the camera",
            ],
        },
    ),
    T2ICatBiasHypothesis(
        name="Orange Cat Indoor Context Elevation",
        claim=(
            "When prompted to generate a cat in domestic or cozy indoor"
            " contexts (e.g. on a sofa, bed, or rug), the proportion of orange"
            " cats is even more pronounced (>40%)."
        ),
        task_name="orange_cat_bias",
        null_proportion=0.20,
        test_type="proportion_z_test",
        alternative="larger",
        metadata={
            "prompt_templates": [
                "a cat curled up on a living room sofa",
                "a cozy cat sleeping on a bed",
                "a cat resting on a warm rug by a fireplace",
            ],
        },
    ),
    T2ICatBiasHypothesis(
        name="Orange Cat Artistic vs Photo Style",
        claim=(
            "Artistic or illustrative prompts ('an oil painting of a cat',"
            " 'a watercolor illustration of a cat') produce orange cats at a"
            " higher rate than realistic photographic prompts."
        ),
        task_name="orange_cat_bias",
        null_proportion=0.50,
        test_type="proportion_z_test_2_sample",
        alternative="larger",
        metadata={
            "condition_a": "artistic/illustrated style",
            "condition_b": "photorealistic style",
        },
    ),
]


# ── Recommended Seed Instructions for DISCO ───────────────────────────────────

# The task's `recommended_seed`, not its `seed_instruction`: baselines read
# `metadata["seed_instruction"]` and keep running without a seed.
ORANGE_CAT_BIAS_SEED: str = spec.require(
    _SUITE.by_key("orange_cat_bias").recommended_seed, "recommended_seed"
)


# ── Accessor Functions ────────────────────────────────────────────────────────


TASKS_BY_NAME: dict[str, data_classes.Task] = spec.index_by_key(
    T2I_CAT_BIAS_TASKS
)

HYPOTHESES_BY_KEY: dict[str, T2ICatBiasHypothesis] = {
    h.key: h for h in T2I_CAT_BIAS_HYPOTHESES
}


def get_t2i_cat_bias_tasks(split: str = "all") -> list[data_classes.Task]:
  """Returns T2I cat bias tasks filtered by split or name."""
  split_lower = spec.normalize_key(split)
  if split_lower == "all":
    return list(T2I_CAT_BIAS_TASKS)
  matched = spec.legacy_match(
      split_lower, TASKS_BY_NAME, T2I_CAT_BIAS_TASKS, "get_t2i_cat_bias_tasks"
  )
  if matched:
    return matched
  raise ValueError(
      f"Unknown T2I cat bias task or split: {split!r}. "
      f"Available tasks: {list(TASKS_BY_NAME.keys())}"
  )


def get_t2i_cat_bias_hypotheses(
    name_or_key: str = "all",
) -> list[T2ICatBiasHypothesis]:
  """Returns T2I cat bias reference hypotheses."""
  query_lower = spec.normalize_key(name_or_key)
  if query_lower == "all":
    return list(T2I_CAT_BIAS_HYPOTHESES)
  matched = spec.legacy_match(
      query_lower,
      HYPOTHESES_BY_KEY,
      T2I_CAT_BIAS_HYPOTHESES,
      "get_t2i_cat_bias_hypotheses",
  )
  if matched:
    return matched
  raise ValueError(
      f"Unknown T2I cat bias hypothesis: {name_or_key!r}. "
      f"Available: {list(HYPOTHESES_BY_KEY.keys())}"
  )
