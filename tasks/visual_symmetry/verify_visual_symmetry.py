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

"""Verification runner for Visual Symmetry Reasoning Hypotheses.

Discovers hypotheses across all methods (DISCO, e-BH Baseline, and Discovery
Baselines) in an experiment root directory, identifies which hypotheses have
already been verified to avoid duplicate work, runs fresh independent probe
evaluations against the target multimodal model, records comprehensive
verification parameters and metadata, and computes empirical FDR and
replicability.
"""

import asyncio
import dataclasses
import datetime
import hashlib
import inspect
import io
import json
import os
import random
import re
from typing import Any

from absl import app
from absl import flags
import numpy as np
import PIL.Image
from scipy import stats as sp_stats

from disco import backbone_client
from disco import standalone_agent
from disco import stats_utils
from disco.models import content as content_lib
from disco.models import registry as model_registry
from disco.stats import calibrators
from disco.tasks.visual_symmetry import eval as visual_symmetry_eval
from disco.tasks.visual_symmetry import hypotheses as visual_symmetry_hypotheses
from disco.tasks.visual_symmetry import image_gen as visual_symmetry_image_gen
from disco.verification import hypothesis_io
from disco.verification import min_response_fraction_flag
from disco.verification import scoring

_EXPERIMENT_DIR = flags.DEFINE_string(
    "experiment_dir",
    "",
    "Path to root experiment directory containing outputs across models/seeds."
    " Leave empty when verifying an explicit --hypotheses_path.",
)
_HYPOTHESES_PATH = flags.DEFINE_string(
    "hypotheses_path",
    "",
    "Optional path to a hypotheses JSON file (e.g. from"
    " standalone_precision_verifier or custom JSON). If provided, verifies"
    " these hypotheses in addition to or instead of experiment_dir.",
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    "",
    "Path for verification artifacts. Defaults to "
    "<experiment_dir>/verification/.",
)
_TARGET_MODEL = flags.DEFINE_string(
    "target_model",
    "gemini-3.1-flash-image",
    "Multimodal target model to test and verify against.",
)
_BACKBONE_MODEL = flags.DEFINE_string(
    "backbone_model",
    model_registry.DEFAULT_GEMINI_MODEL,
    "Backbone LLM that plans the probe design for each hypothesis. Recorded"
    " as `agent_model` in the verification metadata.",
)
_NUM_PROBES = flags.DEFINE_integer(
    "num_probes",
    100,
    "Number of fresh independent probe images to evaluate per hypothesis.",
)
_ALPHA = flags.DEFINE_float(
    "alpha",
    0.05,
    "Significance level for replication (e.g. 0.05).",
)
_NULL_ACCURACY = flags.DEFINE_float(
    "null_accuracy",
    0.90,
    "Null benchmark accuracy proportion for statistical testing. 0.90 rather"
    " than the historical 0.75: under the corrected tolerant-F1 metric the"
    " target model clears 92-98% on every shape family in the original probe"
    " set, so 0.75 is not a discriminating reference. MUST match the task's"
    " null_proportion and the rate stated in the seed, or the PLAN"
    " INCONSISTENCY guard fails every hypothesis.",
)
_IOU_THRESHOLD = flags.DEFINE_float(
    "iou_threshold",
    0.90,
    "Tolerant-F1 agreement threshold for probe completion success.",
)

# Query generator specification recorded in the verification metadata.
QUERY_GENERATOR = "visual_symmetry_image_gen (320px continuous)"

# Calibrator parameter for the p-to-e conversion. It only alters the stored
# `e_value` of each record; the verdict is decided by `p_value <= alpha`.
KAPPA = 0.1741

# Pacing for the target model's multimodal RPM limits: at most this many
# concurrent probe requests, each preceded by a short delay, with exponential
# backoff on quota errors and a cooldown between consecutive hypotheses.
CONCURRENCY = 2
REQUEST_DELAY_S = 0.5
MAX_RETRIES = 6
# Retries for a probe that returns successfully but carries no image. Matches
# the backfill budget the text scraping path uses. Bounded separately from
# MAX_RETRIES, which exists for quota backoff: a model that simply will not
# emit an image should not consume the full quota retry budget on every probe.
# Probes still empty after this many retries are recorded as unanswered
# (missing data) and excluded from scoring, never counted as failed
# completions.
MAX_EMPTY_RETRIES = 3
INTER_HYPOTHESIS_DELAY_S = 2.0

# The success criterion is fixed to the one discovery used (the self-mirror
# conjunction with its left-fidelity guard); verifying under a different
# criterion would test a different proposition than the hypothesis states.
LEFT_FIDELITY_THRESHOLD = visual_symmetry_eval.DEFAULT_LEFT_FIDELITY_THRESHOLD
SUCCESS_MODE = visual_symmetry_eval.DEFAULT_SUCCESS_MODE


def _safe_list_dir(path: str) -> list[str]:
  if not path:
    return []
  try:
    return list(os.listdir(path))
  except OSError:
    return []


def _load_json_safe(path: str) -> Any:
  if not path:
    return None
  try:
    with open(path, "r", encoding="utf-8") as f:
      return json.load(f)
  except (OSError, json.JSONDecodeError, ValueError, TypeError):
    return None


def _save_json(path: str, data: Any) -> None:
  dirname = os.path.dirname(path)
  if dirname:
    os.makedirs(dirname, exist_ok=True)
  with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)


def _load_hypotheses_from_file(path: str) -> list[tuple[str, str]]:
  """Loads hypotheses from a JSON file supporting multiple standard formats."""
  return hypothesis_io.load_hypotheses(path, strict_type=False)


@dataclasses.dataclass
class ProbeSpec:
  """Flexible specifications for procedural visual symmetry probe generation."""

  shape_type: str = "english_letter"
  canvas_size: int = 320
  bg_color: str = "white"
  fg_color: str | None = None
  letter: str | None = None
  show_midline: bool = True
  size: int | None = None
  radius: int | None = None
  n_dots: int | None = None
  text_prompt: str = ""


@dataclasses.dataclass
class VerificationTestPlan:
  """Full experimental design for verifying single or paired hypotheses."""

  is_paired: bool = False
  num_probes: int = 50
  direction: str = "smaller"
  null_accuracy: float = 0.90
  condition_a: ProbeSpec = dataclasses.field(default_factory=ProbeSpec)
  condition_b: ProbeSpec | None = None
  reasoning: str = ""


def _extract_shape_from_phrase(
    text: str, default: str = "english_letter"
) -> str:
  """Extracts the most relevant visual shape type from a text phrase."""
  lowered = text.lower()
  # Harder families first: several of them contain substrings that the older,
  # coarser rules below would match (e.g. "complex Chinese" contains "chinese",
  # "fine noise" contains nothing but "scatter rings" contains "ring"), so the
  # specific patterns have to win.
  if "complex_chinese" in lowered or (
      "chinese" in lowered
      and any(
          w in lowered for w in ("complex", "high-stroke", "intricate", "dense")
      )
  ):
    return "complex_chinese"
  if (
      "multi_glyph" in lowered
      or "multi-glyph" in lowered
      or "stacked glyph" in lowered
  ):
    return "multi_glyph"
  if "maze" in lowered or "labyrinth" in lowered:
    return "maze"
  if "snowflake" in lowered:
    return "snowflake"
  if "filigree" in lowered or "ornament" in lowered:
    return "filigree"
  if (
      "nested" in lowered
      or "concentric outline" in lowered
      or "outline" in lowered
  ):
    return "nested_outlines"
  if "starburst" in lowered or "ray" in lowered:
    return "starburst"
  if "ring" in lowered:
    return "scatter_rings"
  if "wave" in lowered or "sinusoid" in lowered:
    return "wave_lines"
  if (
      "fine_noise" in lowered
      or "fine noise" in lowered
      or "fine-grained" in lowered
  ):
    return "fine_noise"
  if "chinese" in lowered or "hanzi" in lowered or "kanji" in lowered:
    return "chinese_char"
  if "greek" in lowered:
    return "greek_letter"
  if "dense" in lowered:
    return "random_dense"
  if "sparse" in lowered:
    return "random_sparse"
  if "dot" in lowered or "lattice" in lowered:
    return "dots"
  if "stripe" in lowered:
    return "stripes"
  if "triangle" in lowered:
    return "triangle"
  if "plus" in lowered or "cross" in lowered:
    return "plus"
  if "heart" in lowered:
    return "heart"
  if "circle" in lowered or "arc" in lowered:
    return "circle"
  if "square" in lowered:
    return "square"
  if "letter" in lowered or "alphabet" in lowered or "english" in lowered:
    return "english_letter"
  return default


def _heuristic_test_plan(
    hypothesis_text: str,
    default_num_probes: int,
    default_null_accuracy: float,
) -> VerificationTestPlan:
  """Fallback rule-based plan for paired and single hypotheses."""
  lowered = hypothesis_text.lower()
  comparison_markers = [
      "compared to",
      "versus",
      " vs ",
      "relative to",
      "better than",
      "worse than",
      "higher than",
      "lower than",
      "drop when",
      "drops without",
      "compared with",
  ]
  is_paired = any(m in lowered for m in comparison_markers)
  positive_words = [
      "succeed",
      "above",
      "higher",
      "better",
      "accurate",
      "robust",
  ]
  direction = (
      "larger" if any(w in lowered for w in positive_words) else "smaller"
  )
  null_acc = stats_utils.extract_null_proportion_from_hypothesis(
      hypothesis_text, default=default_null_accuracy
  )

  if is_paired:
    part_a, part_b = lowered, lowered
    for marker in comparison_markers:
      if marker in lowered:
        parts = lowered.split(marker, 1)
        part_a, part_b = parts[0], parts[1]
        break

    shape_a = _extract_shape_from_phrase(part_a, default="chinese_char")
    shape_b = _extract_shape_from_phrase(part_b, default="english_letter")
    midline_a = (
        False
        if ("without midline" in part_a or "no midline" in part_a)
        else True
    )
    midline_b = (
        False
        if ("without midline" in part_b or "no midline" in part_b)
        else True
    )

    return VerificationTestPlan(
        is_paired=True,
        num_probes=200,
        direction=direction,
        null_accuracy=0.50,
        condition_a=ProbeSpec(shape_type=shape_a, show_midline=midline_a),
        condition_b=ProbeSpec(shape_type=shape_b, show_midline=midline_b),
        reasoning="Heuristic paired contrastive plan",
    )

  shape_type = _extract_shape_from_phrase(lowered, default="english_letter")
  midline = (
      False
      if ("without midline" in lowered or "no midline" in lowered)
      else True
  )
  return VerificationTestPlan(
      is_paired=False,
      num_probes=default_num_probes,
      direction=direction,
      null_accuracy=null_acc,
      condition_a=ProbeSpec(shape_type=shape_type, show_midline=midline),
      condition_b=None,
      reasoning="Heuristic single-condition plan",
  )


async def plan_verification_probes(
    hypothesis_text: str,
    agent_model: str,
    default_num_probes: int = 50,
    default_null_accuracy: float = 0.90,
) -> VerificationTestPlan:
  """Uses the agent model to flexibly plan single or paired probes."""
  english_letters = ", ".join(
      visual_symmetry_image_gen.SYMMETRIC_ENGLISH_LETTERS
  )
  greek_letters = ", ".join(
      visual_symmetry_image_gen.SYMMETRIC_GREEK_LETTERS[:4]
  )
  chinese_chars = ", ".join(
      visual_symmetry_image_gen.SYMMETRIC_CHINESE_CHARS[:8]
  )
  complex_chinese_chars = ", ".join(
      visual_symmetry_image_gen.COMPLEX_SYMMETRIC_CHINESE_CHARS[:6]
  )
  prompt = f"""You are an experimental design auditor for visual spatial reasoning and reflectional symmetry.
Given the following hypothesis about multimodal model capabilities or failure modes, design the exact verification experiment.

Supported visual pattern types:
- "english_letter": Reflectionally symmetric Latin letters ({english_letters}).
- "greek_letter": Reflectionally symmetric Greek letters ({greek_letters}, etc.).
- "chinese_char": Reflectionally symmetric Chinese/Hanzi characters ({chinese_chars}, etc.).
- "square": Concentric or centered geometric squares.
- "circle": Concentric or centered geometric circles.
- "triangle": Centered isosceles triangle.
- "plus": Centered cross / plus sign.
- "heart": Centered symmetric heart shape.
- "stripes": Horizontal reflection stripes.
- "dots": Discrete dot clusters or lattices.
- "random_dense": High-density stochastic texture lattice.
- "random_sparse": Low-density stochastic texture lattice.
Harder families (finer strokes, more parts, more internal structure):
- "complex_chinese": High-stroke-count symmetric Hanzi ({complex_chinese_chars}, ...).
- "multi_glyph": Three small symmetric glyphs stacked vertically.
- "maze": Comb of thin vertical bars with horizontal connectors.
- "snowflake": Radial arms with sub-branches, thin strokes.
- "filigree": Scattered thin arcs and ticks.
- "nested_outlines": Concentric rectangle/ellipse outlines (not filled).
- "starburst": Many thin rays of differing lengths from the centre.
- "scatter_rings": Small hollow rings of varying radius.
- "wave_lines": Stacked sinusoids with independent amplitude, frequency, phase.
- "fine_noise": Stochastic lattice on a 32x32 grid (half the cell size of random_*).

Configuration options per condition:
- shape_type: one of the types above or "dynamic".
- show_midline: boolean (whether the vertical central dashed midline is displayed, default true).
- bg_color: background color. One of "white", "black", "lightgray", "lightyellow", "lightblue", "mistyrose", "honeydew", "lavender", "navy", "darkslategray". Default "white". Only use a non-white background if the hypothesis is actually about the background; it must be a single named colour for the whole condition.
- fg_color: foreground color ("black", "blue", "purple", etc. or null).
- letter: specific character/letter if explicitly named (e.g. "A", "中", "M") or null.
- size: square size (pixels, 40-140) or null.
- radius: circle radius (pixels, 20-140) or null.
- text_prompt: custom text prompt if the hypothesis tests specific prompt phrasing, otherwise null.

Rules for Experiment Design:
1. Is it a PAIRED / contrastive hypothesis?
   - Set "is_paired": true if the hypothesis compares performance between TWO conditions (e.g. "Model fails on Chinese characters compared to English letters", "Accuracy is lower on random dense patterns than random sparse patterns", "Performance drops without a midline versus with a midline", "Shape A vs Shape B", "Condition A vs Condition B").
   - Set "is_paired": false if the hypothesis makes an assertion about a SINGLE condition against a baseline (e.g. "Model fails to achieve 90% accuracy on English letters").
2. Number of probes ("num_probes"):
   - If "is_paired" is true, ALWAYS set "num_probes" to 200 (100 pairs: 100 for condition_a, 100 for condition_b).
   - If "is_paired" is false, set "num_probes" to {default_num_probes}.
3. Conditions:
   - For single condition: populate "condition_a" only. "condition_b" should be null.
   - For paired condition: populate "condition_a" with the primary condition claimed to fail/differ, and "condition_b" with the reference/contrast condition.
4. Test Direction ("direction"):
   - "smaller": if Condition A is hypothesized to have lower accuracy / higher failure rate than Condition B (or lower than null benchmark).
   - "larger": if Condition A is hypothesized to have higher accuracy / superior capability than Condition B (or higher than null benchmark).
5. Null Accuracy ("null_accuracy"):
   - Baseline benchmark for single-condition tests (e.g. 0.90, or extracted from hypothesis text if stated). Default {default_null_accuracy}.

Hypothesis: "{hypothesis_text}"

Return ONLY a JSON object:
{{
  "is_paired": false,
  "num_probes": {default_num_probes},
  "direction": "smaller",
  "null_accuracy": {default_null_accuracy},
  "condition_a": {{
    "shape_type": "english_letter",
    "show_midline": true,
    "bg_color": "white",
    "fg_color": null,
    "letter": null,
    "size": null,
    "radius": null,
    "text_prompt": null
  }},
  "condition_b": null,
  "reasoning": "brief explanation"
}}
"""
  try:
    backbone = backbone_client.get_backbone_client()
    resp_text = await backbone.generate(
        model=agent_model,
        prompt=prompt,
        temperature=0.0,
    )
    json_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", resp_text)
    if json_match:
      raw_json = json_match.group(1)
    else:
      brace_start = resp_text.find("{")
      brace_end = resp_text.rfind("}") + 1
      raw_json = (
          resp_text[brace_start:brace_end]
          if brace_start >= 0 and brace_end > brace_start
          else ""
      )

    if raw_json:
      data = json.loads(raw_json)
      is_paired = bool(data.get("is_paired", False))
      direction = (
          "larger"
          if data.get("direction") in ("larger", "greater", "above")
          else "smaller"
      )
      null_acc = float(data.get("null_accuracy", default_null_accuracy))
      num_probes = (
          200 if is_paired else int(data.get("num_probes", default_num_probes))
      )

      cond_a_data = data.get("condition_a", {}) or {}
      cond_a = ProbeSpec(
          shape_type=cond_a_data.get("shape_type", "english_letter"),
          show_midline=bool(cond_a_data.get("show_midline", True)),
          bg_color=cond_a_data.get("bg_color", "white"),
          fg_color=cond_a_data.get("fg_color"),
          letter=cond_a_data.get("letter"),
          size=cond_a_data.get("size"),
          radius=cond_a_data.get("radius"),
          text_prompt=cond_a_data.get("text_prompt") or "",
      )

      cond_b = None
      if is_paired and data.get("condition_b"):
        cond_b_data = data.get("condition_b", {}) or {}
        cond_b = ProbeSpec(
            shape_type=cond_b_data.get("shape_type", "english_letter"),
            show_midline=bool(cond_b_data.get("show_midline", True)),
            bg_color=cond_b_data.get("bg_color", "white"),
            fg_color=cond_b_data.get("fg_color"),
            letter=cond_b_data.get("letter"),
            size=cond_b_data.get("size"),
            radius=cond_b_data.get("radius"),
            text_prompt=cond_b_data.get("text_prompt") or "",
        )

      return VerificationTestPlan(
          is_paired=is_paired,
          num_probes=num_probes,
          direction=direction,
          null_accuracy=null_acc,
          condition_a=cond_a,
          condition_b=cond_b,
          reasoning=str(data.get("reasoning", "")),
      )
  except backbone_client.LLM_CALL_ERRORS as e:
    print(f"    [Planner Note] Model planner fallback invoked ({e}).")

  return _heuristic_test_plan(
      hypothesis_text, default_num_probes, default_null_accuracy
  )


def _generate_probe_batch(
    spec: ProbeSpec,
    n_probes: int,
    seed_start: int,
) -> tuple[list[np.ndarray], list[np.ndarray], str]:
  """Generates a batch of continuous symmetry probes from a ProbeSpec."""
  gt_cells_list = []
  prompt_images = []
  # Reference, not a copy: the out-of-sample verifier must word the task exactly
  # as discovery did, or it is re-testing something else.
  default_prompt = visual_symmetry_hypotheses.SYMMETRY_PROMPT
  prompt_text = spec.text_prompt.strip() if spec.text_prompt else default_prompt

  for i in range(n_probes):
    seed = seed_start + i
    kwargs: dict[str, Any] = {
        "show_midline": spec.show_midline,
        "bg_color": spec.bg_color or "white",
    }
    if spec.fg_color:
      kwargs["fg_color"] = spec.fg_color
    if spec.letter:
      kwargs["letter"] = spec.letter
    if spec.size is not None:
      kwargs["size"] = spec.size
    if spec.radius is not None:
      kwargs["radius"] = spec.radius
    if spec.n_dots is not None:
      kwargs["n_dots"] = spec.n_dots

    gt_cells, prompt_cells, _ = (
        visual_symmetry_image_gen.generate_symmetry_image(
            shape_type=spec.shape_type,
            canvas_size=spec.canvas_size,
            seed=seed,
            allow_dynamic=True,
            **kwargs,
        )
    )
    gt_cells_list.append(gt_cells)
    prompt_images.append(prompt_cells)

  return gt_cells_list, prompt_images, prompt_text


generate_probe_batch = _generate_probe_batch


def _extract_response_image(resp: Any) -> np.ndarray | None:
  """Returns the first decodable RGB image in a response, else None.

  This is the single definition of "usable" for a symmetry probe response. The
  query retry loop, the high-failure-rate abort guard, and the scorer all go
  through it, so the three cannot disagree about which responses count as
  answered. They previously did: the retry loop only retried on exceptions and
  the abort guard only counted `None`, so a response that arrived carrying no
  image was treated as a healthy query by both and as a model failure by the
  scorer.

  Args:
    resp: Raw model response object.

  Returns:
    Decoded RGB uint8 array, or None if no valid image is present.
  """
  if resp is None or not resp.content:
    return None
  img_chunks = resp.content.as_images(ignore_unsupported_types=True)
  if not img_chunks:
    return None
  chunk = img_chunks[0]
  try:
    if hasattr(chunk, "data") and chunk.data:
      return np.array(PIL.Image.open(io.BytesIO(chunk.data)).convert("RGB"))
    if hasattr(chunk, "pil_image"):
      return np.array(chunk.pil_image.convert("RGB"))
  except (OSError, ValueError, TypeError, AttributeError):
    return None
  return None


extract_response_image = _extract_response_image


def _score_completions(
    raw_responses: list[Any],
    gt_cells_list: list[np.ndarray],
    iou_threshold: float,
    probe_images: list[np.ndarray] | None = None,
    left_fidelity_threshold: float = (
        visual_symmetry_eval.DEFAULT_LEFT_FIDELITY_THRESHOLD
    ),
    success_mode: str = visual_symmetry_eval.DEFAULT_SUCCESS_MODE,
    bg_color: str = "white",
) -> tuple[list[float | None], list[dict[str, Any] | None]]:
  """Scores raw model responses, using the same criterion discovery used.

  Returns `(scores, details)`, one entry each per response.

  `scores[i]` is `None` when the model returned nothing scoreable. None means
  *missing data* and must be excluded by callers rather than treated as a score
  of 0.0 -- a probe the model never answered is not a probe the model failed.

  Otherwise `scores[i]` is a **gated** score in [0, 1], defined so that
  `scores[i] >= iou_threshold` is exactly the success criterion. Callers may
  therefore keep comparing it to `iou_threshold` directly:

    * `success_mode="ground_truth"`: the score is the pixelwise IoU of the
      model's right half against the ground truth's right half.
    * `success_mode="self_mirror"`: success is a conjunction -- the returned
      image must be internally symmetric AND must have preserved the left half
      it was handed. A scalar cannot express a conjunction, so the self-mirror
      IoU is forced to 0.0 whenever the left-fidelity term fails. Read
      `details[i]` for the ungated numbers.

  `details[i]` carries the raw per-criterion values (`iou_self_mirror`,
  `left_fidelity`, `iou_vs_ground_truth` and the matching booleans) so the
  criteria can be compared against each other after the fact.

  Args:
    raw_responses: Model responses, in probe order.
    gt_cells_list: Ground-truth image per probe.
    iou_threshold: Minimum IoU for the symmetry term.
    probe_images: The images that were sent to the model, in probe order.
      Required for the left-fidelity term; when omitted, `self_mirror` mode
      silently degrades to the self-mirror term alone, which a blank canvas
      passes.
    left_fidelity_threshold: Minimum left-half IoU against the probe.
    success_mode: Which criterion decides success.
    bg_color: The background the probes were rendered with. This has to be the
      condition's actual background: the scorer extracts ink from the probe and
      the ground truth relative to it, so passing "white" for a probe drawn on
      navy classifies the entire canvas as ink and every criterion collapses.

  Returns:
    Tuple of (scores, details), one entry each per response.
  """
  scores: list[float | None] = []
  details: list[dict[str, Any] | None] = []
  for idx, resp in enumerate(raw_responses[: len(gt_cells_list)]):
    gt = gt_cells_list[idx]
    probe = (
        probe_images[idx]
        if probe_images is not None and idx < len(probe_images)
        else None
    )
    out_arr = _extract_response_image(resp)
    if out_arr is None:
      scores.append(None)
      details.append(None)
      continue
    try:
      score_res = visual_symmetry_eval.score_symmetry_completion(
          ground_truth_cells=gt,
          output_image=out_arr,
          iou_threshold=iou_threshold,
          bg_color=bg_color,
          probe_image=probe,
          left_fidelity_threshold=left_fidelity_threshold,
          success_mode=success_mode,
      )
      # `score_symmetry_completion` reports the continuous score under "iou"
      # for Option A (direct image) and "right_accuracy" for the legacy cell
      # grid. It has never emitted a "symmetry_iou" key, so reading that name
      # silently scored every probe as 0.0.
      iou_val = score_res.get("iou")
      if iou_val is None:
        iou_val = score_res.get("right_accuracy", 0.0)
      iou_val = float(iou_val)
      # Fold the conjunction into the scalar so that every downstream
      # `score >= iou_threshold` test agrees with `is_success`.
      if not score_res.get("is_success_left_fidelity", True):
        iou_val = 0.0
      scores.append(iou_val)
      details.append(score_res)
    except (
        ValueError,
        TypeError,
        RuntimeError,
        ArithmeticError,
        IndexError,
        KeyError,
    ) as e:
      # Log rather than swallow. A scoring exception is recorded as missing
      # data, which is statistically the right thing to do, but it is NOT the
      # same as "the model did not answer" and it must not be invisible: an
      # exception that fires on every probe of one condition would silently
      # shrink that condition's sample to zero while the run still reported a
      # clean result.
      print(
          f"    [Scoring Error] probe {idx} raised {type(e).__name__}: {e}."
          " Recorded as missing data."
      )
      scores.append(None)
      details.append(None)
  return scores, details


score_completions = _score_completions


def _log_criterion_breakdown(
    label: str,
    details: list[dict[str, Any] | None],
) -> dict[str, Any]:
  """Prints and returns how the three success criteria compare on this batch.

  The point of reporting all three is to see whether they actually agree. If
  `self_mirror` and `ground_truth` diverge sharply, the headline number is
  measuring registration error rather than symmetry, and the discovery that
  rests on it should be read with that in mind.

  Args:
    label: Identifier printed in the log line.
    details: Per-probe scoring detail dicts.

  Returns:
    Summary dictionary of per-criterion rates and means.
  """
  rows = [d for d in details if d]
  if not rows:
    return {}

  def _rate(key: str) -> float:
    return float(np.mean([1.0 if r.get(key) else 0.0 for r in rows]))

  def _mean(key: str) -> float:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else 0.0

  agreements = []
  for r in rows:
    both_ok = bool(r.get("is_success_self_mirror")) and bool(
        r.get("is_success_left_fidelity")
    )
    gt_ok = bool(r.get("is_success_ground_truth"))
    agreements.append(1.0 if both_ok == gt_ok else 0.0)

  summary = {
      "n_scored": len(rows),
      "rate_self_mirror": _rate("is_success_self_mirror"),
      "rate_left_fidelity": _rate("is_success_left_fidelity"),
      "rate_ground_truth": _rate("is_success_ground_truth"),
      "mean_iou_self_mirror": _mean("iou_self_mirror"),
      "mean_left_fidelity": _mean("left_fidelity"),
      "mean_iou_vs_ground_truth": _mean("iou_vs_ground_truth"),
      "criterion_agreement": float(np.mean(agreements)),
      "success_mode": rows[0].get("success_mode"),
  }
  print(
      f"    [Criteria:{label}] n={summary['n_scored']}"
      f" self_mirror={summary['rate_self_mirror']:.1%}"
      f" (mean {summary['mean_iou_self_mirror']:.3f})"
      f" | left_fidelity={summary['rate_left_fidelity']:.1%}"
      f" (mean {summary['mean_left_fidelity']:.3f})"
      f" | ground_truth={summary['rate_ground_truth']:.1%}"
      f" (mean {summary['mean_iou_vs_ground_truth']:.3f})"
      f" | agreement={summary['criterion_agreement']:.1%}"
      f" | deciding={summary['success_mode']}"
  )
  return summary


log_criterion_breakdown = _log_criterion_breakdown


# Statuses of records written before records carried `status`: their test
# produced no usable measurement.
_FAILED_VERIFICATION_STATUSES = frozenset(
    {"EXECUTION_ERROR", "RATE_LIMITED", scoring.INSUFFICIENT_RESPONSES_STATUS}
)


def record_status(rec: dict[str, Any]) -> str:
  """Returns `rec`'s scoring status, inferring it for older records."""
  if "status" in rec:
    return str(rec["status"])
  if rec.get("verification_status") in _FAILED_VERIFICATION_STATUSES:
    return scoring.LEGACY_UNMEASURED_STATUS
  return scoring.OK_STATUS


def _method_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
  result = scoring.score([{**r, "status": record_status(r)} for r in records])
  return {
      "total_tested": result.n_scored,
      "total_verified": result.n_verified,
      "total_failed": result.n_failed,
      "precision": round(result.precision or 0.0, 4),
      "empirical_fdr": round(result.empirical_fdr, 4),
  }


def summarize_records(
    records: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
  """Scores verification records per origin method and overall.

  Uses the shared rule (`verification.scoring.score`, failed records dropped),
  so a hypothesis whose test produced no usable measurement (e.g. an
  EXECUTION_ERROR record left in the registry by an earlier run) is reported
  as failed rather than counted as a false discovery.

  Args:
    records: The verification records.

  Returns:
    (per-method summary, overall `total_tested` / `total_verified` /
    `total_failed`). `total_tested` counts scored records only.
  """
  by_method: dict[str, list[dict[str, Any]]] = {}
  for r in records:
    by_method.setdefault(r.get("origin_method", "Unknown"), []).append(r)
  methods = {m: _method_summary(rs) for m, rs in by_method.items()}
  overall = _method_summary(records)
  return methods, {
      k: overall[k] for k in ("total_tested", "total_verified", "total_failed")
  }


def _collect_from_checkpoint_dict(
    ckpt: dict[str, Any],
    *,
    cell_prefix: str,
    origin_model: str,
    origin_seed: str,
    method_tag: str,
    origin_method: str,
    cell_dir: str,
    seen_keys: set[str],
    hypotheses: list[dict[str, Any]],
) -> None:
  """Appends hypotheses from a parsed `checkpoint.json` dict."""
  if "hypothesis_groups" not in ckpt:
    return
  extra = ckpt.get("extra", {})
  term = extra.get("terminal_statuses", {})
  e_vals = extra.get("final_e_values", {})
  for h_id, h_data in ckpt["hypothesis_groups"].items():
    status = term.get(h_id, "")
    e_val = e_vals.get(h_id, 0.0)
    h_text = h_data.get("text", "") if isinstance(h_data, dict) else ""
    if not h_text:
      continue
    unique_key = f"{cell_prefix}/{method_tag}/{h_id}"
    if unique_key not in seen_keys:
      seen_keys.add(unique_key)
      hypotheses.append({
          "unique_key": unique_key,
          "hypothesis_id": h_id,
          "text": h_text,
          "origin_model": origin_model,
          "origin_seed": origin_seed,
          "origin_method": origin_method,
          "discovery_status": status or "Discovered",
          "discovery_e_value": e_val,
          "cell_dir": cell_dir,
      })


def _collect_from_stage1_dict(
    disc_res: dict[str, Any],
    *,
    cell_prefix: str,
    origin_model: str,
    origin_seed: str,
    base_cell_dir: str,
    seen_keys: set[str],
    hypotheses: list[dict[str, Any]],
) -> None:
  """Appends hypotheses from a parsed `stage1_discovery_results.json` dict."""
  if "method_hypotheses" in disc_res and isinstance(
      disc_res["method_hypotheses"], dict
  ):
    for m_key, m_data in disc_res["method_hypotheses"].items():
      if not isinstance(m_data, dict):
        continue
      m_name = m_data.get("method_name", m_key)
      preds = m_data.get("predictions", [])
      for i, pred in enumerate(preds):
        h_text = (
            pred
            if isinstance(pred, str)
            else (pred.get("text", "") if isinstance(pred, dict) else "")
        )
        if not h_text:
          continue
        h_id = f"h_{i+1}"
        unique_key = f"{cell_prefix}/{m_name}/{h_id}"
        if unique_key not in seen_keys:
          seen_keys.add(unique_key)
          hypotheses.append({
              "unique_key": unique_key,
              "hypothesis_id": h_id,
              "text": h_text,
              "origin_model": origin_model,
              "origin_seed": origin_seed,
              "origin_method": m_name,
              "discovery_status": "Discovered",
              "discovery_e_value": None,
              "cell_dir": os.path.join(base_cell_dir, m_name),
          })
    return

  methods_list = hypothesis_io.method_entries(disc_res)
  if methods_list:
    for m_entry in methods_list:
      if not isinstance(m_entry, dict):
        continue
      m_name = m_entry.get("method_name") or m_entry.get("method") or "baseline"
      preds = hypothesis_io.first_truthy(
          m_entry, hypothesis_io.METHOD_PREDICTION_KEYS, []
      )
      for i, pred in enumerate(preds):
        if isinstance(pred, str):
          h_text = pred
        elif isinstance(pred, dict):
          h_text = hypothesis_io.first_truthy(pred, hypothesis_io.TEXT_KEYS, "")
        else:
          h_text = ""
        if not h_text:
          continue
        h_id = f"h_{i+1}"
        unique_key = f"{cell_prefix}/{m_name}/{h_id}"
        if unique_key not in seen_keys:
          seen_keys.add(unique_key)
          hypotheses.append({
              "unique_key": unique_key,
              "hypothesis_id": h_id,
              "text": str(h_text),
              "origin_model": origin_model,
              "origin_seed": origin_seed,
              "origin_method": str(m_name),
              "discovery_status": "Discovered",
              "discovery_e_value": None,
              "cell_dir": os.path.join(base_cell_dir, str(m_name)),
          })


def _collect_direct_run_dir(
    run_dir: str,
    run_label: str,
    *,
    seen_keys: set[str],
    hypotheses: list[dict[str, Any]],
) -> None:
  """Collects hypotheses when `run_dir` is a direct `run_disco` or baseline output dir."""
  ckpt = _load_json_safe(os.path.join(run_dir, "checkpoint.json"))
  if not (isinstance(ckpt, dict) and "hypothesis_groups" in ckpt):
    ckpt = _load_json_safe(os.path.join(run_dir, "test", "checkpoint.json"))
  if isinstance(ckpt, dict) and "hypothesis_groups" in ckpt:
    origin_model = str(ckpt.get("target_model") or run_label)
    origin_seed = str(ckpt.get("seed", "0"))
    _collect_from_checkpoint_dict(
        ckpt,
        cell_prefix=run_label,
        origin_model=origin_model,
        origin_seed=origin_seed,
        method_tag="DISCO",
        origin_method="DISCO",
        cell_dir=run_dir,
        seen_keys=seen_keys,
        hypotheses=hypotheses,
    )
  else:
    fdr_data = _load_json_safe(os.path.join(run_dir, "fdr_bounds.json"))
    if fdr_data is None:
      fdr_data = _load_json_safe(
          os.path.join(run_dir, "test", "fdr_bounds.json")
      )
    if fdr_data is not None:
      origin_model = (
          str(fdr_data.get("target_model") or run_label)
          if isinstance(fdr_data, dict)
          else run_label
      )
      for h_id, h_text in hypothesis_io.extract_hypotheses(
          fdr_data, strict_type=False
      ):
        unique_key = f"{run_label}/DISCO/{h_id}"
        if unique_key not in seen_keys:
          seen_keys.add(unique_key)
          hypotheses.append({
              "unique_key": unique_key,
              "hypothesis_id": h_id,
              "text": h_text,
              "origin_model": origin_model,
              "origin_seed": "0",
              "origin_method": "DISCO",
              "discovery_status": "H",
              "discovery_e_value": None,
              "cell_dir": run_dir,
          })

  disc_res = _load_json_safe(
      os.path.join(run_dir, "stage1_discovery_results.json")
  )
  if isinstance(disc_res, dict):
    origin_model = str(disc_res.get("target_model") or run_label)
    _collect_from_stage1_dict(
        disc_res,
        cell_prefix=run_label,
        origin_model=origin_model,
        origin_seed="0",
        base_cell_dir=run_dir,
        seen_keys=seen_keys,
        hypotheses=hypotheses,
    )


def collect_discovered_hypotheses(exp_dir: str) -> list[dict[str, Any]]:
  """Recursively collects all candidate hypotheses from experiment directory."""
  exp_dir = exp_dir.rstrip("/")
  hypotheses: list[dict[str, Any]] = []
  seen_keys: set[str] = set()

  # Support standard `run_disco` and `eval_discovery_baselines` layouts directly
  # in `exp_dir` or in immediate `<exp_dir>/<run_name>/` subdirectories.
  root_label = os.path.basename(exp_dir) or "experiment"
  _collect_direct_run_dir(
      exp_dir,
      root_label,
      seen_keys=seen_keys,
      hypotheses=hypotheses,
  )

  for model_entry in _safe_list_dir(exp_dir):
    model_path = os.path.join(exp_dir, model_entry)
    model_name = os.path.basename(model_path)
    if model_name in ("verification", "tmp", "inputs", "images", "test"):
      continue

    _collect_direct_run_dir(
        model_path,
        model_name,
        seen_keys=seen_keys,
        hypotheses=hypotheses,
    )

    for seed_entry in _safe_list_dir(model_path):
      seed_path = os.path.join(model_path, seed_entry)
      seed_name = os.path.basename(seed_path)
      if seed_name in ("verification", "tmp", "inputs", "images", "test"):
        continue

      cell_prefix = f"{model_name}/{seed_name}"

      # 1. DISCO Run
      disco_dir = os.path.join(seed_path, "disco", "test")
      disco_ckpt = _load_json_safe(os.path.join(disco_dir, "checkpoint.json"))
      if isinstance(disco_ckpt, dict):
        _collect_from_checkpoint_dict(
            disco_ckpt,
            cell_prefix=cell_prefix,
            origin_model=model_name,
            origin_seed=seed_name,
            method_tag="DISCO",
            origin_method="DISCO",
            cell_dir=os.path.join(seed_path, "disco"),
            seen_keys=seen_keys,
            hypotheses=hypotheses,
        )

      # 2. e-BH Baseline Run
      ebh_dir = os.path.join(seed_path, "ebh_baseline", "test")
      ebh_ckpt = _load_json_safe(os.path.join(ebh_dir, "checkpoint.json"))
      if isinstance(ebh_ckpt, dict):
        _collect_from_checkpoint_dict(
            ebh_ckpt,
            cell_prefix=cell_prefix,
            origin_model=model_name,
            origin_seed=seed_name,
            method_tag="eBH",
            origin_method="e-BH Baseline",
            cell_dir=os.path.join(seed_path, "ebh_baseline"),
            seen_keys=seen_keys,
            hypotheses=hypotheses,
        )

      # 3. Type-I Baseline Run
      t1_dir = os.path.join(seed_path, "type1_baseline", "test")
      t1_ckpt = _load_json_safe(os.path.join(t1_dir, "checkpoint.json"))
      if isinstance(t1_ckpt, dict):
        _collect_from_checkpoint_dict(
            t1_ckpt,
            cell_prefix=cell_prefix,
            origin_model=model_name,
            origin_seed=seed_name,
            method_tag="Type1",
            origin_method="Type-I Baseline",
            cell_dir=os.path.join(seed_path, "type1_baseline"),
            seen_keys=seen_keys,
            hypotheses=hypotheses,
        )

      # 4. Discovery Baselines Run
      base_dir = os.path.join(seed_path, "baselines", "baseline_run")
      disc_res = _load_json_safe(
          os.path.join(base_dir, "stage1_discovery_results.json")
      )
      if isinstance(disc_res, dict):
        _collect_from_stage1_dict(
            disc_res,
            cell_prefix=cell_prefix,
            origin_model=model_name,
            origin_seed=seed_name,
            base_cell_dir=os.path.join(seed_path, "baselines"),
            seen_keys=seen_keys,
            hypotheses=hypotheses,
        )

  return hypotheses


def _insufficient_record(
    h_entry: dict[str, Any],
    plan: Any,
    reason: str,
    **counts: Any,
) -> dict[str, Any]:
  """A record for a batch short of `--min_response_fraction`.

  Args:
    h_entry: The hypothesis entry.
    plan: The verification plan.
    reason: `scoring.short_batch_reason`'s explanation.
    **counts: Probe / pair counts to record (e.g. `num_probes_valid`).

  Returns:
    Verification record dict with insufficient-responses status.
  """
  return {
      "label": h_entry["unique_key"],
      "hypothesis_id": h_entry["hypothesis_id"],
      "hypothesis_text": h_entry["text"],
      "origin_model": h_entry["origin_model"],
      "origin_seed": h_entry["origin_seed"],
      "origin_method": h_entry["origin_method"],
      **counts,
      "successes": 0,
      "observed_accuracy": 0.0,
      "mean_iou": 0.0,
      "null_accuracy": plan.null_accuracy,
      "p_value": 1.0,
      "e_value": 0.0,
      "verified": False,
      "status": scoring.INSUFFICIENT_RESPONSES_STATUS,
      "verification_status": scoring.INSUFFICIENT_RESPONSES_STATUS,
      "error_message": reason,
      "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
  }


async def verify_single_hypothesis(
    h_entry: dict[str, Any],
    candidate_cls: Any,
    agent_model: str,
    num_probes: int,
    null_accuracy: float,
    alpha: float,
    kappa: float,
    iou_threshold: float,
    concurrency: int,
    probe_seed_offset: int,
    request_delay: float = REQUEST_DELAY_S,
    max_retries: int = MAX_RETRIES,
    max_empty_retries: int = MAX_EMPTY_RETRIES,
    left_fidelity_threshold: float = LEFT_FIDELITY_THRESHOLD,
    success_mode: str = SUCCESS_MODE,
    min_response_fraction: float = scoring.DEFAULT_MIN_RESPONSE_FRACTION,
) -> dict[str, Any]:
  """Executes flexible independent probe generation and statistical verification."""
  h_text = h_entry["text"]

  # 1. Flexible hypothesis analysis & test planning
  plan = await plan_verification_probes(
      hypothesis_text=h_text,
      agent_model=agent_model,
      default_num_probes=num_probes,
      default_null_accuracy=null_accuracy,
  )

  gt_cells_a: list[np.ndarray] = []
  gt_cells_b: list[np.ndarray] = []
  gt_cells_list: list[np.ndarray] = []
  prompt_imgs_a: list[np.ndarray] = []
  prompt_imgs_b: list[np.ndarray] = []
  prompt_images: list[np.ndarray] = []

  try:
    if plan.is_paired:
      n_pairs = plan.num_probes // 2  # 100 pairs = 200 probes
      print(
          f"  [Flexible Plan: PAIRED] Generating 200 probes ({n_pairs} pairs)."
      )
      print(
          f"    Condition A: shape={plan.condition_a.shape_type},"
          f" midline={plan.condition_a.show_midline}"
      )
      if plan.condition_b:
        print(
            f"    Condition B: shape={plan.condition_b.shape_type},"
            f" midline={plan.condition_b.show_midline}"
        )
      print(
          f"    Direction: {plan.direction} | Null: 50% discordant |"
          f" Plan Note: {plan.reasoning[:80]}"
      )
      gt_cells_a, prompt_imgs_a, prompt_text_a = _generate_probe_batch(
          spec=plan.condition_a,
          n_probes=n_pairs,
          seed_start=probe_seed_offset,
      )
      gt_cells_b, prompt_imgs_b, prompt_text_b = _generate_probe_batch(
          spec=plan.condition_b or ProbeSpec(),
          n_probes=n_pairs,
          seed_start=probe_seed_offset,
      )

      probes = []
      for pil_arr in prompt_imgs_a:
        pil_img = PIL.Image.fromarray(pil_arr)
        content = content_lib.Content([pil_img, prompt_text_a])
        probes.append(content_lib.Probe(content=content))
      for pil_arr in prompt_imgs_b:
        pil_img = PIL.Image.fromarray(pil_arr)
        content = content_lib.Content([pil_img, prompt_text_b])
        probes.append(content_lib.Probe(content=content))
    else:
      print(
          f"  [Flexible Plan: SINGLE] Generating {plan.num_probes} probes."
          f" shape={plan.condition_a.shape_type},"
          f" midline={plan.condition_a.show_midline},"
          f" direction={plan.direction},"
          f" null_acc={plan.null_accuracy:.1%}"
      )
      gt_cells_list, prompt_images, prompt_text = _generate_probe_batch(
          spec=plan.condition_a,
          n_probes=plan.num_probes,
          seed_start=probe_seed_offset,
      )
      probes = []
      for pil_arr in prompt_images:
        pil_img = PIL.Image.fromarray(pil_arr)
        content = content_lib.Content([pil_img, prompt_text])
        probes.append(content_lib.Probe(content=content))
  except (RuntimeError, ValueError, TypeError, OSError, KeyError) as exc:
    return {
        "label": h_entry["unique_key"],
        "hypothesis_id": h_entry["hypothesis_id"],
        "hypothesis_text": h_text,
        "origin_model": h_entry["origin_model"],
        "origin_seed": h_entry["origin_seed"],
        "origin_method": h_entry["origin_method"],
        "is_paired": plan.is_paired,
        "num_probes": plan.num_probes,
        "successes": 0,
        "observed_accuracy": 0.0,
        "mean_iou": 0.0,
        "null_accuracy": plan.null_accuracy,
        "p_value": 1.0,
        "e_value": 0.0,
        "verified": False,
        "status": scoring.LEGACY_UNMEASURED_STATUS,
        "verification_status": "EXECUTION_ERROR",
        "error_message": f"Probe generation failed: {exc}",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

  sem = asyncio.Semaphore(concurrency)
  backoff_lock = asyncio.Lock()
  global_backoff_until = [0.0]

  async def _query_with_sem(p: content_lib.Probe, probe_idx: int) -> Any:
    # Stagger probe start times slightly to avoid instant burst
    if request_delay > 0:
      await asyncio.sleep((probe_idx % concurrency) * request_delay)

    empty_retries = 0
    for attempt in range(max_retries):
      # Wait if a concurrent probe triggered a global rate-limit backoff
      now = asyncio.get_event_loop().time()
      if now < global_backoff_until[0]:
        wait_time = global_backoff_until[0] - now + random.uniform(0.5, 2.0)
        await asyncio.sleep(wait_time)

      async with sem:
        try:
          if request_delay > 0:
            await asyncio.sleep(request_delay)
          res = candidate_cls.respond(p.content)
          if inspect.isawaitable(res):
            res = await res
          if _extract_response_image(res) is not None:
            return res
          # A response that arrives carrying no image is exactly as
          # unscoreable as a failed query, so retry it instead of spending the
          # probe. This mirrors the backfill that the text scraping path does
          # in `standalone_agent._scrape_target_model`, and uses the same
          # budget of MAX_EMPTY_RETRIES attempts. Empty responses are bounded
          # separately from `max_retries` because that budget exists for quota
          # backoff, where retrying is nearly free; a model that simply will
          # not emit an image should not consume six calls per probe.
          empty_retries += 1
          if empty_retries > max_empty_retries:
            print(
                f"    [Probe Empty] Probe {probe_idx + 1} returned no image"
                f" after {max_empty_retries} retry(ies). Recording as"
                " unanswered (missing data), not as a failed completion."
            )
            return res
          backoff = (2.0 * empty_retries) + random.uniform(0.5, 1.5)
          print(
              f"    [Probe Empty] Probe {probe_idx + 1} returned no image"
              f" (retry {empty_retries}/{max_empty_retries}). Retrying in"
              f" {backoff:.1f}s..."
          )
          await asyncio.sleep(backoff)
        except backbone_client.LLM_CALL_ERRORS as e:
          err_msg = str(e)
          is_quota = any(
              k in err_msg.lower()
              for k in (
                  "quota",
                  "resource_exhausted",
                  "rate",
                  "429",
                  "exceeded your current quota",
              )
          )
          if is_quota:
            backoff = (5.0 * (2 ** min(attempt, 4))) + random.uniform(2.0, 5.0)
            print(
                f"    [Quota/Rate Limit] Hit quota on probe {probe_idx + 1}"
                f" (attempt {attempt + 1}/{max_retries})."
                f" Backing off for {backoff:.1f}s..."
            )
            async with backoff_lock:
              new_until = asyncio.get_event_loop().time() + backoff
              if new_until > global_backoff_until[0]:
                global_backoff_until[0] = new_until
            await asyncio.sleep(backoff)
          elif attempt < max_retries - 1:
            backoff = (2.0 * (attempt + 1)) + random.uniform(0.5, 1.5)
            print(
                f"    [Probe Error] Query failed: {err_msg[:100]} (attempt"
                f" {attempt + 1}/{max_retries}). Retrying in {backoff:.1f}s..."
            )
            await asyncio.sleep(backoff)
          else:
            print(
                f"    [Probe Failed] Probe {probe_idx + 1} exhausted"
                f" {max_retries} attempts: {err_msg[:140]}"
            )
            return None
    return None

  raw_responses = await asyncio.gather(
      *[_query_with_sem(p, idx) for idx, p in enumerate(probes)]
  )

  # Count responses that came back without an image, not just hard failures.
  # A response that returns successfully but carries no image is exactly as
  # unscoreable as a missing one, and counting only the latter let whole runs
  # complete with nothing scoreable at all.
  failed_queries = sum(
      1 for r in raw_responses if _extract_response_image(r) is None
  )
  if failed_queries > 0:
    print(
        f"    [Warning] {failed_queries}/{len(raw_responses)} probe queries"
        " returned no usable image."
    )

  # 2. Score completions and compute statistical verification
  if plan.is_paired:
    n_pairs = plan.num_probes // 2
    raw_responses_a = raw_responses[:n_pairs]
    raw_responses_b = raw_responses[n_pairs:]

    ious_a, details_a = _score_completions(
        raw_responses_a,
        gt_cells_a,
        iou_threshold,
        probe_images=prompt_imgs_a,
        left_fidelity_threshold=left_fidelity_threshold,
        success_mode=success_mode,
        bg_color=plan.condition_a.bg_color,
    )
    ious_b, details_b = _score_completions(
        raw_responses_b,
        gt_cells_b,
        iou_threshold,
        probe_images=prompt_imgs_b,
        left_fidelity_threshold=left_fidelity_threshold,
        success_mode=success_mode,
        bg_color=(
            plan.condition_b.bg_color
            if plan.condition_b is not None
            else plan.condition_a.bg_color
        ),
    )
    criteria_a = _log_criterion_breakdown("A", details_a)
    criteria_b = _log_criterion_breakdown("B", details_b)

    # A pair is only usable if BOTH arms returned something scoreable. Scoring a
    # missing arm as 0.0 would flip that pair into the discordant count in
    # whichever direction the arm went missing, biasing McNemar directly.
    paired_ious = [
        (a, b)
        for a, b in zip(ious_a, ious_b)
        if a is not None and b is not None
    ]
    n_pairs_valid = len(paired_ious)
    n_orphan_pairs = min(len(ious_a), len(ious_b)) - n_pairs_valid
    if n_pairs_valid == 0:
      # No pair had both arms answered. Falling through would report
      # "Failed Replication", which asserts evidence against the hypothesis
      # when in fact there is no data at all.
      return {
          "label": h_entry["unique_key"],
          "hypothesis_id": h_entry["hypothesis_id"],
          "hypothesis_text": h_text,
          "origin_model": h_entry["origin_model"],
          "origin_seed": h_entry["origin_seed"],
          "origin_method": h_entry["origin_method"],
          "is_paired": True,
          "num_probes": plan.num_probes,
          "num_pairs": n_pairs,
          "num_pairs_valid": 0,
          "num_orphan_pairs": n_orphan_pairs,
          "successes": 0,
          "observed_accuracy": 0.0,
          "mean_iou": 0.0,
          "null_accuracy": plan.null_accuracy,
          "p_value": 1.0,
          "e_value": 0.0,
          "verified": False,
          "status": scoring.LEGACY_UNMEASURED_STATUS,
          "verification_status": "EXECUTION_ERROR",
          "error_message": (
              f"No usable pairs: all {n_pairs} pair(s) had at least one arm"
              " return nothing scoreable."
          ),
          "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
      }
    short = scoring.short_batch_reason(
        is_paired=True,
        q_count=2 * n_pairs,
        n_responses=0,
        n_pairs=n_pairs_valid,
        min_fraction=min_response_fraction,
    )
    if short:
      print(f"    [Insufficient Responses] {short}")
      return _insufficient_record(
          h_entry,
          plan,
          short,
          is_paired=True,
          num_probes=plan.num_probes,
          num_pairs=n_pairs,
          num_pairs_valid=n_pairs_valid,
          num_orphan_pairs=n_orphan_pairs,
      )
    if n_orphan_pairs:
      print(
          f"    [Warning] Dropped {n_orphan_pairs} pair(s) where one or both"
          f" arms returned no usable image. Using {n_pairs_valid} of"
          f" {n_pairs} pairs."
      )

    valid_a = [a for a, _ in paired_ious]
    valid_b = [b for _, b in paired_ious]
    successes_a = sum(1 for iou in valid_a if iou >= iou_threshold)
    successes_b = sum(1 for iou in valid_b if iou >= iou_threshold)
    acc_a = successes_a / n_pairs_valid if n_pairs_valid > 0 else 0.0
    acc_b = successes_b / n_pairs_valid if n_pairs_valid > 0 else 0.0
    mean_iou_a = float(np.mean(valid_a)) if valid_a else 0.0
    mean_iou_b = float(np.mean(valid_b)) if valid_b else 0.0

    # Discordant pairs for McNemar's exact binomial test
    n_10 = sum(
        1 for a, b in paired_ious if a >= iou_threshold and b < iou_threshold
    )
    n_01 = sum(
        1 for a, b in paired_ious if a < iou_threshold and b >= iou_threshold
    )
    n_disc = n_10 + n_01

    if n_disc > 0:
      alt = "smaller" if plan.direction == "smaller" else "larger"
      p_value, _, ci_lower, ci_upper = (
          stats_utils.compute_binomial_replication_stats(
              successes=n_10,
              total=n_disc,
              null_prob=0.5,
              alpha=alpha,
              alternative=alt,
          )
      )
    else:
      p_value = 1.0
      ci_lower, ci_upper = 0.0, 1.0

    t_pval = 1.0
    if n_pairs_valid > 1 and np.any(np.array(valid_a) != np.array(valid_b)):
      try:
        t_res = sp_stats.ttest_rel(valid_a, valid_b)
        t_pval = float(t_res.pvalue)
      except (ValueError, TypeError, ZeroDivisionError, FloatingPointError):
        t_pval = 1.0

    safe_p = max(1e-12, min(1.0, p_value))
    e_value = calibrators.p_to_e(safe_p, kappa=kappa)
    threshold = calibrators.get_threshold_from_alpha(alpha)
    verified = bool(p_value <= alpha or e_value >= threshold)

    cond_b_dict = (
        dataclasses.asdict(plan.condition_b) if plan.condition_b else None
    )
    shape_b_name = (
        plan.condition_b.shape_type if plan.condition_b else "reference"
    )

    return {
        "label": h_entry["unique_key"],
        "hypothesis_id": h_entry["hypothesis_id"],
        "hypothesis_text": h_text,
        "origin_model": h_entry["origin_model"],
        "origin_seed": h_entry["origin_seed"],
        "origin_method": h_entry["origin_method"],
        "is_paired": True,
        "test_type": "paired_mcnemar_binomial",
        "shape_type": f"{plan.condition_a.shape_type} vs {shape_b_name}",
        "condition_a": dataclasses.asdict(plan.condition_a),
        "condition_b": cond_b_dict,
        "test_direction": plan.direction,
        "success_mode": success_mode,
        "iou_threshold": iou_threshold,
        "left_fidelity_threshold": left_fidelity_threshold,
        "criteria_a": criteria_a,
        "criteria_b": criteria_b,
        "num_probes": plan.num_probes,
        "num_pairs": n_pairs,
        "num_pairs_valid": n_pairs_valid,
        "num_orphan_pairs": n_orphan_pairs,
        "successes": successes_a,
        "observed_accuracy": round(acc_a, 4),
        "observed_accuracy_a": round(acc_a, 4),
        "observed_accuracy_b": round(acc_b, 4),
        "delta_accuracy": round(acc_a - acc_b, 4),
        "mean_iou": round(mean_iou_a, 4),
        "mean_iou_a": round(mean_iou_a, 4),
        "mean_iou_b": round(mean_iou_b, 4),
        "delta_iou": round(mean_iou_a - mean_iou_b, 4),
        "discordant_pairs": n_disc,
        "n_a_win_b_loss": n_10,
        "n_a_loss_b_win": n_01,
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "null_accuracy": 0.50,
        "p_value": p_value,
        "t_test_p_value": round(t_pval, 6),
        "e_value": round(e_value, 4),
        "verified": verified,
        "status": scoring.OK_STATUS,
        "verification_status": (
            "Replicated" if verified else "Failed Replication"
        ),
        "reasoning": plan.reasoning,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

  # Single condition evaluation
  ious, details = _score_completions(
      raw_responses,
      gt_cells_list,
      iou_threshold,
      probe_images=prompt_images,
      left_fidelity_threshold=left_fidelity_threshold,
      success_mode=success_mode,
      bg_color=plan.condition_a.bg_color,
  )
  criteria_summary = _log_criterion_breakdown("single", details)
  valid_ious = [iou for iou in ious if iou is not None]
  n_valid = len(valid_ious)
  n_unusable = len(ious) - n_valid
  successes = sum(1 for iou in valid_ious if iou >= iou_threshold)
  mean_iou = float(np.mean(valid_ious)) if valid_ious else 0.0

  if n_valid == 0:
    # Nothing was scoreable. Reporting 0/N here would produce the most extreme
    # possible p-value and record a spurious "Replicated" discovery.
    return {
        "label": h_entry["unique_key"],
        "hypothesis_id": h_entry["hypothesis_id"],
        "hypothesis_text": h_text,
        "origin_model": h_entry["origin_model"],
        "origin_seed": h_entry["origin_seed"],
        "origin_method": h_entry["origin_method"],
        "is_paired": False,
        "num_probes": len(ious),
        "num_probes_valid": 0,
        "num_probes_unusable": n_unusable,
        "successes": 0,
        "observed_accuracy": 0.0,
        "mean_iou": 0.0,
        "null_accuracy": plan.null_accuracy,
        "p_value": 1.0,
        "e_value": 0.0,
        "verified": False,
        "status": scoring.LEGACY_UNMEASURED_STATUS,
        "verification_status": "EXECUTION_ERROR",
        "error_message": (
            f"No usable completions: all {len(ious)} probe(s) returned"
            " nothing scoreable."
        ),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

  short = scoring.short_batch_reason(
      is_paired=False,
      q_count=len(ious),
      n_responses=n_valid,
      n_pairs=0,
      min_fraction=min_response_fraction,
  )
  if short:
    print(f"    [Insufficient Responses] {short}")
    return _insufficient_record(
        h_entry,
        plan,
        short,
        is_paired=False,
        num_probes=len(ious),
        num_probes_valid=n_valid,
        num_probes_unusable=n_unusable,
    )

  if n_unusable:
    print(
        f"    [Warning] {n_unusable}/{len(ious)} probe(s) returned no usable"
        f" image; excluding them. Denominator is {n_valid}, not"
        f" {plan.num_probes}."
    )

  p_value, observed_rate, ci_lower, ci_upper = (
      stats_utils.compute_binomial_replication_stats(
          successes=successes,
          total=n_valid,
          null_prob=plan.null_accuracy,
          alpha=alpha,
          alternative=plan.direction,
      )
  )

  safe_p = max(1e-12, min(1.0, p_value))
  e_value = calibrators.p_to_e(safe_p, kappa=kappa)
  threshold = calibrators.get_threshold_from_alpha(alpha)
  verified = bool(p_value <= alpha or e_value >= threshold)

  return {
      "label": h_entry["unique_key"],
      "hypothesis_id": h_entry["hypothesis_id"],
      "hypothesis_text": h_text,
      "origin_model": h_entry["origin_model"],
      "origin_seed": h_entry["origin_seed"],
      "origin_method": h_entry["origin_method"],
      "is_paired": False,
      "test_type": "one_sample_binomial",
      "shape_type": plan.condition_a.shape_type,
      "condition_a": dataclasses.asdict(plan.condition_a),
      "condition_b": None,
      "test_direction": plan.direction,
      "success_mode": success_mode,
      "iou_threshold": iou_threshold,
      "left_fidelity_threshold": left_fidelity_threshold,
      "criteria": criteria_summary,
      "num_probes": plan.num_probes,
      "num_probes_valid": n_valid,
      "num_probes_unusable": n_unusable,
      "successes": successes,
      "observed_accuracy": round(observed_rate, 4),
      "ci_lower": round(ci_lower, 4),
      "ci_upper": round(ci_upper, 4),
      "mean_iou": round(mean_iou, 4),
      "null_accuracy": plan.null_accuracy,
      "p_value": p_value,
      "e_value": round(e_value, 4),
      "verified": verified,
      "status": scoring.OK_STATUS,
      "verification_status": "Replicated" if verified else "Failed Replication",
      "reasoning": plan.reasoning,
      "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
  }


async def run_visual_verification() -> None:
  """Main execution loop for continuous visual reasoning verification."""
  exp_dir = _EXPERIMENT_DIR.value.rstrip("/")
  out_dir = (
      _OUTPUT_DIR.value.rstrip("/")
      if _OUTPUT_DIR.value
      else os.path.join(exp_dir, "verification")
  )

  print("=" * 66)
  print("Visual Reasoning Hypothesis Verifier")
  print(f"Target Model:       {_TARGET_MODEL.value}")
  print(f"Backbone Model:     {_BACKBONE_MODEL.value}")
  print(f"Query Generator:    {QUERY_GENERATOR}")
  print(f"Experiment Root:    {exp_dir}")
  print(f"Verification Dir:   {out_dir}")
  print(f"Probes per Hyp:     {_NUM_PROBES.value}")
  print(f"Alpha / Null Rate:  {_ALPHA.value} / {_NULL_ACCURACY.value}")
  print("=" * 66)

  # 1. Discover all candidate hypotheses
  all_hyps: list[dict[str, Any]] = []
  if _HYPOTHESES_PATH.value:
    resolved_path = hypothesis_io.resolve_file_path(_HYPOTHESES_PATH.value)
    print(f"Loading hypotheses from file: {resolved_path}")
    file_hyps = _load_hypotheses_from_file(resolved_path)
    for label, text in file_hyps:
      all_hyps.append({
          "unique_key": f"input/{label}",
          "hypothesis_id": label,
          "text": text,
          "origin_model": _TARGET_MODEL.value,
          "origin_seed": "0",
          "origin_method": "InputFile",
          "discovery_status": "Loaded",
          "discovery_e_value": None,
          "cell_dir": out_dir,
      })
    print(f"Loaded {len(file_hyps)} hypotheses from {resolved_path}.")

  if exp_dir and os.path.exists(exp_dir):
    dir_hyps = collect_discovered_hypotheses(exp_dir)
    all_hyps.extend(dir_hyps)
    print(f"Discovered {len(dir_hyps)} candidate hypotheses from {exp_dir}.")
  elif not _HYPOTHESES_PATH.value:
    print(f"Warning: Experiment directory does not exist: {exp_dir}")

  print(f"Total candidate hypotheses to process: {len(all_hyps)}.")

  # 2. Load cumulative registry to prevent duplicate verification
  registry_path = os.path.join(out_dir, "cumulative_verification_registry.json")
  existing_registry: dict[str, Any] = _load_json_safe(registry_path) or {
      "metadata": {},
      "records": {},
  }
  verified_records: dict[str, Any] = existing_registry.get("records", {})

  unverified: list[dict[str, Any]] = []
  for h in all_hyps:
    key = h["unique_key"]
    text_hash = hashlib.sha256(h["text"].strip().encode("utf-8")).hexdigest()
    rec = verified_records.get(key) or verified_records.get(text_hash)
    # Hypotheses whose previous run produced a usable measurement are done; a
    # record left by an execution error or quota failure is retried.
    if rec and record_status(rec) == scoring.OK_STATUS:
      continue
    h["text_hash"] = text_hash
    unverified.append(h)

  print(
      f"Identified {len(verified_records)} already verified hypotheses. "
      f"{len(unverified)} new hypotheses to verify."
  )

  # Initialize model candidate
  candidate_cls = standalone_agent.get_or_create_candidate(
      _TARGET_MODEL.value
  )

  # 3. Verify each new hypothesis
  new_records: list[dict[str, Any]] = []
  base_seed_start = 50000

  for idx, h_entry in enumerate(unverified):
    print(
        f"\n[{idx + 1}/{len(unverified)}] Verifying"
        f" [{h_entry['origin_method']}] {h_entry['hypothesis_id']}..."
    )
    print(f"  Hypothesis: {h_entry['text'][:110]}...")

    rec = await verify_single_hypothesis(
        h_entry=h_entry,
        candidate_cls=candidate_cls,
        agent_model=_BACKBONE_MODEL.value,
        num_probes=_NUM_PROBES.value,
        null_accuracy=_NULL_ACCURACY.value,
        alpha=_ALPHA.value,
        kappa=KAPPA,
        iou_threshold=_IOU_THRESHOLD.value,
        left_fidelity_threshold=LEFT_FIDELITY_THRESHOLD,
        success_mode=SUCCESS_MODE,
        concurrency=CONCURRENCY,
        probe_seed_offset=base_seed_start + idx * 1000,
        request_delay=REQUEST_DELAY_S,
        max_retries=MAX_RETRIES,
        max_empty_retries=MAX_EMPTY_RETRIES,
        min_response_fraction=min_response_fraction_flag.value(),
    )

    if record_status(rec) != scoring.OK_STATUS:
      print(
          "  [Skipped Registry Save] Verification produced no usable"
          f" measurement ({rec.get('error_message', '')}). Hypothesis will"
          " remain unverified and be retried on the next run."
      )
      continue

    ver_tag = "REPLICATED (PASS)" if rec["verified"] else "FAILED REPLICATION"
    if rec.get("is_paired"):
      print(
          f"  Result (PAIRED): {ver_tag} | Acc A:"
          f" {rec['observed_accuracy_a']:.1%} vs Acc B:"
          f" {rec['observed_accuracy_b']:.1%} (Δ={rec['delta_accuracy']:+.1%})"
          f" | Disc: {rec['discordant_pairs']} (A wins:"
          f" {rec['n_a_win_b_loss']}, B wins: {rec['n_a_loss_b_win']}) |"
          f" p={rec['p_value']:.4e} | E={rec['e_value']:.2f}"
      )
    else:
      print(
          f"  Result: {ver_tag} | Observed Acc:"
          f" {rec['observed_accuracy']:.2%} | p={rec['p_value']:.4e} |"
          f" E={rec['e_value']:.2f}"
      )

    verified_records[h_entry["unique_key"]] = rec
    verified_records[h_entry["text_hash"]] = rec
    new_records.append(rec)

    # Incrementally update per-cell verification output
    cell_ver_file = os.path.join(
        h_entry["cell_dir"], "verification", "verification_results.json"
    )
    _save_json(
        cell_ver_file,
        {
            "target_model": _TARGET_MODEL.value,
            "agent_model": _BACKBONE_MODEL.value,
            "query_generator": QUERY_GENERATOR,
            "null_accuracy": _NULL_ACCURACY.value,
            "alpha": _ALPHA.value,
            "latest_record": rec,
        },
    )

    if INTER_HYPOTHESIS_DELAY_S > 0 and idx < len(unverified) - 1:
      await asyncio.sleep(INTER_HYPOTHESIS_DELAY_S)

  # 4. Save consolidated verification results & registry
  all_flat_records = [
      r
      for k, r in verified_records.items()
      if not (len(k) == 64 and all(c in "0123456789abcdef" for c in k))
  ]

  methods_stats, overall = summarize_records(all_flat_records)

  if SUCCESS_MODE == visual_symmetry_eval.SUCCESS_MODE_SELF_MIRROR:
    scoring_metric = (
        f"self_mirror_tolerant_f1 >= {_IOU_THRESHOLD.value} AND"
        f" left_fidelity_tolerant_f1 >= {LEFT_FIDELITY_THRESHOLD}"
    )
  else:
    scoring_metric = f"{SUCCESS_MODE}_tolerant_f1 >= {_IOU_THRESHOLD.value}"

  final_summary = {
      "metadata": {
          "target_model": _TARGET_MODEL.value,
          "agent_model": _BACKBONE_MODEL.value,
          "query_generator": QUERY_GENERATOR,
          "scoring_metric": scoring_metric,
          "success_mode": SUCCESS_MODE,
          "left_fidelity_threshold": LEFT_FIDELITY_THRESHOLD,
          "num_probes_per_hypothesis": _NUM_PROBES.value,
          "null_accuracy": _NULL_ACCURACY.value,
          "alpha": _ALPHA.value,
          "experiment_dir": exp_dir,
          "last_updated": (
              datetime.datetime.now(datetime.timezone.utc).isoformat()
          ),
      },
      "summary_by_method": methods_stats,
      **overall,
      "records": all_flat_records,
  }

  summary_path = os.path.join(out_dir, "verification_results.json")
  _save_json(summary_path, final_summary)
  _save_json(
      registry_path,
      {"metadata": final_summary["metadata"], "records": verified_records},
  )
  print(f"\nVerification summary saved to: {summary_path}")


def main(argv: list[str]) -> None:
  del argv
  asyncio.run(run_visual_verification())


if __name__ == "__main__":
  app.run(main)
