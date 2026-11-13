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

"""Visual symmetry completion reference hypotheses and task definition for DISCO.

Defines:
1. The overarching ``VISUAL_SYMMETRY_TASK`` — evaluating spatial reasoning
   and reflectional symmetry completion on images by the target model
   (gemini-3.1-flash-image).
2. 2 reference candidate ``VISUAL_SYMMETRY_HYPOTHESES`` — reference assertions
   about specific pattern types where the model is claimed to achieve >=90%
   completion success.

Target model: gemini-3.1-flash-image

Evaluation metric:
  Tolerant F1 (2px match tolerance at 320px probe resolution) between the
  returned image's right half and the mirror of its own left half, conjoined
  with a left-half fidelity check against the probe. A completion attempt is
  marked as a Success when the symmetry term is >= 0.90 and the fidelity term
  is >= 0.80. See visual_symmetry_eval.score_direct_symmetry_completion.

Null accuracy:
  0.90. Raised from 0.75 once the corrected metric showed the target model
  clearing 92-98% on every shape family in the original probe set, which made
  a 0.75 null degenerate: every success hypothesis verified and no failure
  hypothesis could. The probe set was simultaneously extended with the harder
  families in visual_symmetry_image_gen.HARD_SHAPE_TYPES.
"""

from disco.tasks import data_classes
from disco.tasks import loader
from disco.tasks import spec

# ── The Visual Symmetry Evaluation Task ───────────────────────────────────────

# The task is data in `data/visual_symmetry.yaml`, including its
# `null_proportion` (0.90) and the reasoning behind it.
VISUAL_SYMMETRY_TASK: data_classes.Task = loader.load_suite(
    "visual_symmetry"
).by_key("visual_symmetry_completion")

# ── Canonical seed instruction ────────────────────────────────────────────────

# The default domain seed for visual symmetry discovery and verification.
# Stored in the task data so that any consumer resolving the task from
# `tasks/registry.py` gets a usable seed. It must keep mentioning "symmetry"
# and naming the two symmetry tools: several call sites (agent_factory,
# runners, precision_estimation) detect the visual symmetry task by
# substring-matching the seed instruction.
DEFAULT_SEED_INSTRUCTION: str = spec.require(
    VISUAL_SYMMETRY_TASK.seed_instruction, "seed_instruction"
)


# ── Reference Candidate Hypotheses ────────────────────────────────────────────

VISUAL_SYMMETRY_HYPOTHESES: list[data_classes.Hypothesis] = [
    # ── English letters ──────────────────────────────────────────────────
    data_classes.Hypothesis(
        name="Symmetry English Letters 1to1",
        claim=(
            "The model can perform vertical symmetry completion on"
            " symmetric English letters on 1:1 images, succeeding at"
            " least 90% of the time."
        ),
        task_suite="visual_symmetry",
        condition={
            "target_model": "gemini-3.1-flash-image",
            "shape_type": "english_letter",
            "aspect_ratio": "1:1",
            "size": 320,
            "bg_color": "white",
        },
        null_proportion=0.90,
    ),
    # ── Chinese characters ───────────────────────────────────────────────
    data_classes.Hypothesis(
        name="Symmetry Chinese Characters 1to1",
        claim=(
            "The model can perform vertical symmetry completion on"
            " symmetric Chinese characters on 1:1 images, succeeding at"
            " least 90% of the time."
        ),
        task_suite="visual_symmetry",
        condition={
            "target_model": "gemini-3.1-flash-image",
            "shape_type": "chinese_char",
            "aspect_ratio": "1:1",
            "size": 320,
            "bg_color": "white",
        },
        null_proportion=0.90,
    ),
]


# ── Convenience accessors ─────────────────────────────────────────────────────

HYPOTHESES_BY_KEY: dict[str, data_classes.Hypothesis] = {
    h.key: h for h in VISUAL_SYMMETRY_HYPOTHESES
}

# The text prompt sent to the target model along with the half-erased image.
# It does NOT disclose what the pattern is, and asks the model to reflect
# the existing pattern along the central vertical axis while keeping the
# existing pattern without modification.
SYMMETRY_PROMPT: str = (
    "Reflect the pattern on the left half along the central vertical axis to"
    " complete the blank right half. Keep the existing colored pattern on the"
    " left without modification. The resulting image should be perfectly"
    " symmetric about the vertical center axis."
)


def get_visual_symmetry_hypotheses(
    name_or_key: str = "all",
) -> list[data_classes.Hypothesis]:
  """Returns visual symmetry reference hypotheses, optionally filtered.

  Args:
    name_or_key: "all" to return all hypotheses, or a name/key for a specific
      hypothesis.

  Returns:
    List of matching Hypothesis objects.
  """
  query_lower = spec.normalize_key(name_or_key)
  if query_lower == "all":
    return list(VISUAL_SYMMETRY_HYPOTHESES)
  matched = spec.legacy_match(
      query_lower,
      HYPOTHESES_BY_KEY,
      VISUAL_SYMMETRY_HYPOTHESES,
      "get_visual_symmetry_hypotheses",
  )
  if matched:
    return matched
  raise ValueError(
      f"Unknown visual symmetry hypothesis: {name_or_key!r}. "
      f"Available: {list(HYPOTHESES_BY_KEY.keys())}"
  )
