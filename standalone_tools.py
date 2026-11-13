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

"""Tool declarations and implementations for the standalone DISCO runner.

Contains:
  - Gemini function calling declarations for all tools.
  - Statistical test implementations (proportion z-test, McNemar, etc.).
  - AskQuestionJudge for backbone-based response analysis.
"""

import asyncio
import json
import math
import random
import re
from typing import Any, Callable

from google.genai import types
import pydantic
from scipy import stats

from disco import backbone_client as backbone_lib
from disco import stats_utils
from disco.models import content as content_api
from disco.models import registry as model_registry
from disco.tasks import hedging_autorater

# ── Gemini function-calling tool declarations ────────────────────────────────

TOOL_DECLARATIONS: list[types.FunctionDeclaration] = [
    types.FunctionDeclaration(
        name="generate_single_turn_responses",
        description=(
            "Sends a list of text prompts to a candidate model and returns the"
            " responses (supports text and/or image generation models)."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "prompts": types.Schema(
                    type="ARRAY",
                    items=types.Schema(type="STRING"),
                    description="List of text prompts to send.",
                ),
                "candidate": types.Schema(
                    type="STRING",
                    description="The target model address.",
                ),
            },
            required=["prompts"],
        ),
    ),
    types.FunctionDeclaration(
        name="analyze_responses",
        description=(
            "Asks a question about each response (analyzing text or generated"
            " images) and classifies the answers. Returns counts per answer"
            " option."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "question": types.Schema(
                    type="STRING",
                    description="Question to ask about each response.",
                ),
                "answer_options": types.Schema(
                    type="ARRAY",
                    items=types.Schema(type="STRING"),
                    description="Possible answer options.",
                ),
            },
            required=["question", "answer_options"],
        ),
    ),
    types.FunctionDeclaration(
        name="run_proportion_z_test",
        description=(
            "One-sample exact binomial test for proportions."
            " Returns the p-value."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "n_successes": types.Schema(
                    type="INTEGER", description="Number of successes."
                ),
                "n_obs": types.Schema(
                    type="INTEGER", description="Total observations."
                ),
                "null_proportion": types.Schema(
                    type="NUMBER",
                    description="Hypothesized proportion. Default 0.5.",
                ),
                "alternative": types.Schema(
                    type="STRING",
                    description=(
                        "'larger' (if testing sample proportion > null),"
                        " 'smaller' (if testing sample proportion < null), or"
                        " 'two-sided'."
                    ),
                ),
            },
            required=["n_successes", "n_obs", "alternative"],
        ),
    ),
    types.FunctionDeclaration(
        name="run_proportion_z_test_2_sample",
        description=(
            "Two-sample Fisher's exact test for proportions."
            " Returns the p-value."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "n_successes_a": types.Schema(
                    type="INTEGER", description="Successes in sample A."
                ),
                "n_obs_a": types.Schema(
                    type="INTEGER", description="Observations in A."
                ),
                "n_successes_b": types.Schema(
                    type="INTEGER", description="Successes in sample B."
                ),
                "n_obs_b": types.Schema(
                    type="INTEGER", description="Observations in B."
                ),
                "alternative": types.Schema(
                    type="STRING",
                    description=(
                        "'larger' (A>B), 'smaller' (A<B), or 'two-sided'."
                    ),
                ),
            },
            required=[
                "n_successes_a",
                "n_obs_a",
                "n_successes_b",
                "n_obs_b",
                "alternative",
            ],
        ),
    ),
    types.FunctionDeclaration(
        name="run_mcnemar_test",
        description=(
            "McNemar's test for paired binary data (exact binomial)."
            " Returns the p-value."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "passed_a_failed_b": types.Schema(
                    type="INTEGER",
                    description="Count: A true, B false.",
                ),
                "failed_a_passed_b": types.Schema(
                    type="INTEGER",
                    description="Count: A false, B true.",
                ),
                "alternative": types.Schema(
                    type="STRING",
                    description=(
                        "'larger' (A>B), 'smaller' (A<B), or 'two-sided'."
                    ),
                ),
            },
            required=["passed_a_failed_b", "failed_a_passed_b", "alternative"],
        ),
    ),
    types.FunctionDeclaration(
        name="run_continuous_means_test",
        description=(
            "One-sample Student's t-test for the mean of continuous scores."
            " Returns the p-value."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "sample_mean": types.Schema(
                    type="NUMBER", description="Sample mean."
                ),
                "sample_std": types.Schema(
                    type="NUMBER", description="Sample std deviation."
                ),
                "n_obs": types.Schema(
                    type="INTEGER", description="Number of observations."
                ),
                "null_mean": types.Schema(
                    type="NUMBER",
                    description="Hypothesized mean. Default 0.",
                ),
                "alternative": types.Schema(
                    type="STRING",
                    description="'larger', 'smaller', or 'two-sided'.",
                ),
            },
            required=["sample_mean", "sample_std", "n_obs", "alternative"],
        ),
    ),
    types.FunctionDeclaration(
        name="run_paired_means_test",
        description=(
            "Paired Student's t-test for continuous difference scores between"
            " matched pairs (e.g. Group A minus Group B). Returns the p-value."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "mean_diff": types.Schema(
                    type="NUMBER",
                    description=(
                        "Mean of the paired differences (Group A - Group B)."
                    ),
                ),
                "std_diff": types.Schema(
                    type="NUMBER",
                    description=(
                        "Sample standard deviation of the paired differences"
                        " (ddof=1)."
                    ),
                ),
                "n_pairs": types.Schema(
                    type="INTEGER",
                    description="Number of matched pairs.",
                ),
                "null_mean": types.Schema(
                    type="NUMBER",
                    description=(
                        "Hypothesized mean difference under null. Default 0.0."
                    ),
                ),
                "alternative": types.Schema(
                    type="STRING",
                    description=(
                        "'larger' (if testing mean difference > null_mean),"
                        " 'smaller' (if testing mean difference < null_mean),"
                        " or 'two-sided'."
                    ),
                ),
            },
            required=["mean_diff", "std_diff", "n_pairs", "alternative"],
        ),
    ),
    types.FunctionDeclaration(
        name="run_procedural_autorater",
        description=(
            "Runs a deterministic (non-LLM) measurement function on each "
            "stored response. Returns a numeric value per response and "
            "summary statistics (mean, min, max, std). Use this instead of "
            "analyze_responses when exact programmatic measurement is needed."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "autorater_name": types.Schema(
                    type="STRING",
                    description=(
                        "Name of the autorater to use (e.g., 'count_words',"
                        " 'count_hedges', 'has_disclaimer',"
                        " 'verbalized_confidence')."
                    ),
                ),
                "params": types.Schema(
                    type="OBJECT",
                    description=(
                        "Optional parameters for the autorater. Contents"
                        " depend on the autorater."
                    ),
                ),
            },
            required=["autorater_name"],
        ),
    ),
    types.FunctionDeclaration(
        name="generate_symmetry_test_image",
        description=(
            "Generates a visual symmetry test image. Produces a grid with a "
            "symmetric pattern where the right half is erased (blank "
            "background). Returns a description of the generated image "
            "including its shape type, grid dimensions, and seed. The "
            "generated prompt image is stored internally for sending to the "
            "target model, and the ground-truth cells are stored for later "
            "scoring. "
            + "Supported shape_types. Base families: english_letter,"
            " greek_letter, chinese_char, square, circle, triangle, plus,"
            " heart, stripes, dots, random_sparse, random_dense. Harder"
            " families (finer strokes, more parts, more internal structure):"
            " complex_chinese (high-stroke-count symmetric Hanzi),"
            " multi_glyph (three small glyphs stacked), maze, snowflake,"
            " filigree, nested_outlines, starburst, scatter_rings,"
            " wave_lines, fine_noise."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "shape_type": types.Schema(
                    type="STRING",
                    description=(
                        "Supported shape_types. Base families: english_letter,"
                        " greek_letter, chinese_char, square, circle, triangle,"
                        " plus, heart, stripes, dots, random_sparse,"
                        " random_dense. Harder families (finer strokes, more"
                        " parts, more internal structure): complex_chinese"
                        " (high-stroke-count symmetric Hanzi), multi_glyph"
                        " (three small glyphs stacked), maze, snowflake,"
                        " filigree, nested_outlines, starburst, scatter_rings,"
                        " wave_lines, fine_noise."
                    ),
                ),
                "seed": types.Schema(
                    type="INTEGER",
                    description="Random seed for reproducibility. Required.",
                ),
                "bg_color": types.Schema(
                    type="STRING",
                    description=(
                        "Background color, one of 'white', 'black',"
                        " 'lightgray', 'lightyellow', 'lightblue', 'mistyrose',"
                        " 'honeydew', 'lavender', 'navy', 'darkslategray'."
                        " Default 'white'. It is a property of the whole batch,"
                        " so keep it fixed within a condition. A foreground"
                        " colour that does not contrast with the background is"
                        " rejected with an error rather than silently swapped."
                    ),
                ),
                "fg_color": types.Schema(
                    type="STRING",
                    description=(
                        "Foreground color for letter/shape types. "
                        "Default 'blue'."
                    ),
                ),
                "letter": types.Schema(
                    type="STRING",
                    description=(
                        "Specific letter/character to render (for letter "
                        "shape types). If omitted, one is chosen randomly."
                    ),
                ),
            },
            required=["shape_type", "seed"],
        ),
    ),
    types.FunctionDeclaration(
        name="generate_and_send_symmetry_probes",
        description=(
            "Generates multiple continuous symmetry test images and sends each"
            " one (with the symmetry completion text prompt) to the target"
            " model. For each probe: generates a half-erased test image (left"
            " half contains the pattern, right half is empty background), sends"
            " it along with the neutral text prompt (reflect existing pattern"
            " along central vertical axis without modification, without"
            " disclosing pattern identity) to the target model, and stores the"
            " model's output image for scoring. The text prompt is FIXED and"
            " identical for every probe in every run; it is not a parameter of"
            " this tool and cannot be varied, so do not form hypotheses about"
            " prompt wording. Returns the number of probes sent. After calling"
            " this, use score_symmetry_completions to evaluate the model's"
            " outputs."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "shape_type": types.Schema(
                    type="STRING",
                    description=(
                        "Supported shape_types. Base families: english_letter,"
                        " greek_letter, chinese_char, square, circle, triangle,"
                        " plus, heart, stripes, dots, random_sparse,"
                        " random_dense. Harder families (finer strokes, more"
                        " parts, more internal structure): complex_chinese"
                        " (high-stroke-count symmetric Hanzi), multi_glyph"
                        " (three small glyphs stacked), maze, snowflake,"
                        " filigree, nested_outlines, starburst, scatter_rings,"
                        " wave_lines, fine_noise."
                    ),
                ),
                "n_probes": types.Schema(
                    type="INTEGER",
                    description=(
                        "Number of test images to generate. Default 25."
                    ),
                ),
                "seed_start": types.Schema(
                    type="INTEGER",
                    description=(
                        "Starting seed. Seeds will be"
                        " seed_start..seed_start+n_probes-1. Default 0."
                    ),
                ),
                "bg_color": types.Schema(
                    type="STRING",
                    description=(
                        "Background color, one of 'white', 'black',"
                        " 'lightgray', 'lightyellow', 'lightblue', 'mistyrose',"
                        " 'honeydew', 'lavender', 'navy', 'darkslategray'."
                        " Default 'white'. It is a property of the whole batch,"
                        " so keep it fixed within a condition. A foreground"
                        " colour that does not contrast with the background is"
                        " rejected with an error rather than silently swapped."
                    ),
                ),
                "fg_color": types.Schema(
                    type="STRING",
                    description=(
                        "Foreground color for letter/shape types. Default"
                        " 'blue'."
                    ),
                ),
                "samples_per_probe": types.Schema(
                    type="INTEGER",
                    description=(
                        "Number of completion attempts to sample from the model"
                        " for each input probe image (for pass@k evaluation)."
                        " Default 1."
                    ),
                ),
            },
            required=["shape_type"],
        ),
    ),
    types.FunctionDeclaration(
        name="score_symmetry_completions",
        description=(
            "Scores the model's symmetry completion outputs from the most"
            " recent generate_and_send_symmetry_probes call. A completion"
            " counts as a Success when BOTH hold: (a) the returned image is"
            " symmetric about its own vertical centre axis, agreement >= 0.90"
            " between its right half and its mirrored left half, and (b) it"
            " preserved the left half it was given, agreement >= 0.80 against"
            " the probe. Criterion (b) is required because a blank canvas"
            " satisfies (a) perfectly. Agreement is a tolerant F1 with a 2"
            " pixel match tolerance at the 320px probe resolution, not a raw"
            " mask IoU: raw IoU charges the model for stroke weight, which"
            " made the score track how thin the shape was rather than how well"
            " it was mirrored."
            " The pixel-exact comparison against the ground-truth image is"
            " also computed and reported, but does NOT decide success: image"
            " models re-render the whole canvas, so that comparison penalises"
            " global shift and rescale rather than asymmetry."
            " Supports pass@k evaluation: an input probe is considered passed"
            " under pass@k if at least 1 of its sampled completions succeeded."
            " Returns pass@k success counts, pass@k rate, mean sample pass@1"
            " rate, the success rate under each of the three criteria"
            " separately, how often they agree, and per-probe details."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "iou_threshold": types.Schema(
                    type="NUMBER",
                    description=(
                        "Option A: Minimum symmetry agreement required for "
                        "success. Default 0.90 (90%)."
                    ),
                ),
                "delta_e_threshold": types.Schema(
                    type="NUMBER",
                    description=(
                        "ΔE threshold above which a cell is incorrect. "
                        "Default 10.0."
                    ),
                ),
                "max_incorrect_for_success": types.Schema(
                    type="INTEGER",
                    description=(
                        "Max incorrect cells for a trial to count as "
                        "success. Default 0."
                    ),
                ),
                "pass_at_k": types.Schema(
                    type="INTEGER",
                    description=(
                        "k value for pass@k (1 <= k <= samples_per_probe)."
                        " Evaluates whether at least 1 of the first k samples"
                        " passed. If omitted, evaluates over all available"
                        " samples per probe."
                    ),
                ),
                "success_criterion": types.Schema(
                    type="STRING",
                    description=(
                        "Which of the three criteria decides Success. The"
                        " three measure different things and do NOT agree, so"
                        " state which one your hypothesis is about and pass it"
                        " here. One of:"
                        " 'preservation' (the model kept the left half it was"
                        " given -- says nothing about reflecting);"
                        " 'symmetry' (the returned image is symmetric about"
                        " its own centre axis -- a rescaled but symmetric"
                        " image still passes);"
                        " 'reconstruction' (the right half matches the"
                        " intended reflection pixel for pixel -- the"
                        " strictest, because it also charges the model for"
                        " shifting or rescaling the whole canvas);"
                        " 'both' (the conjunction of preservation and"
                        " symmetry; the default, and what the probe prompt"
                        " literally asks for)."
                        " All three rates are reported whichever you pick, so"
                        " this only changes which one the Success count and"
                        " the statistical test are based on."
                    ),
                ),
            },
        ),
    ),
]


def get_tool_declarations(
    exclude_analyze: bool = False,
    allow_continuous_means: bool = True,
) -> list[types.FunctionDeclaration]:
  """Returns tool declarations, optionally excluding analyze_responses or run_continuous_means_test."""
  excluded = set()
  if exclude_analyze:
    excluded.add("analyze_responses")
  if not allow_continuous_means:
    excluded.add("run_continuous_means_test")
    excluded.add("run_paired_means_test")
  return [t for t in TOOL_DECLARATIONS if t.name not in excluded]


# ── Statistical test implementations ─────────────────────────────────────────


def _get_p_value(
    z_stat: float,
    alternative: str,
    dist: Any = stats.norm,
    **dist_kwargs: Any,
) -> float:
  """Calculates the p-value for a given test statistic."""
  if alternative == "two-sided":
    return float(2 * dist.sf(abs(z_stat), **dist_kwargs))
  elif alternative == "larger":
    return float(dist.sf(z_stat, **dist_kwargs))
  elif alternative == "smaller":
    return float(dist.cdf(z_stat, **dist_kwargs))
  else:
    raise ValueError(f"Invalid alternative: {alternative}")


def run_proportion_z_test(args: dict[str, Any]) -> str:
  """One-sample exact binomial test for proportions."""
  if "n_successes" not in args or "n_obs" not in args:
    return "Error: 'n_successes' and 'n_obs' parameters are required."
  n_successes = int(args["n_successes"])
  n_obs = int(args["n_obs"])
  null_proportion = float(args.get("null_proportion", 0.5))
  if "alternative" not in args or args["alternative"] not in (
      "larger",
      "smaller",
      "two-sided",
  ):
    return (
        "Error: 'alternative' parameter is required. You MUST specify"
        " 'larger' (if testing sample proportion > null), 'smaller' (if"
        " testing sample proportion < null), or 'two-sided'."
    )
  alternative = args["alternative"]

  if n_obs <= 0:
    return "Error: n_obs must be greater than 0"
  if n_successes < 0 or n_successes > n_obs:
    return "Error: n_successes must be between 0 and n_obs"
  if null_proportion <= 0 or null_proportion >= 1:
    return "Error: null_proportion must be in (0, 1)"

  p_hat = n_successes / n_obs
  method = "Binomial test (exact)"
  alt_map = {"smaller": "less", "larger": "greater"}
  mapped_alternative = alt_map.get(alternative, alternative)
  res = stats.binomtest(
      n_successes,
      n_obs,
      p=null_proportion,
      alternative=mapped_alternative,
  )
  p_value = res.pvalue

  note = ""
  if alternative == "larger" and p_hat < null_proportion:
    note = (
        f"\n- Note: Sample proportion ({p_hat:.4f}) is below null"
        f" ({null_proportion:.4f}); one-sided p-value for 'larger' is"
        f" {stats_utils.format_p_value(p_value)}."
    )
  elif alternative == "smaller" and p_hat > null_proportion:
    note = (
        f"\n- Note: Sample proportion ({p_hat:.4f}) is above null"
        f" ({null_proportion:.4f}); one-sided p-value for 'smaller' is"
        f" {stats_utils.format_p_value(p_value)}."
    )

  return (
      "One-Sample Proportion Test Results:\n"
      f"- Method: {method}\n"
      f"- Sample proportion: {n_successes/n_obs:.4f}"
      f" ({n_successes}/{n_obs})\n"
      f"- Null proportion: {null_proportion:.4f}\n"
      "- Statistic: N/A (exact)\n"
      f"- Alternative: {alternative}\n"
      f"- P-value: {stats_utils.format_p_value(p_value)}"
      f"{note}\n"
  )


def run_proportion_z_test_2_sample(args: dict[str, Any]) -> str:
  """Two-sample Fisher's exact test for proportions."""
  if (
      "n_successes_a" not in args
      or "n_obs_a" not in args
      or "n_successes_b" not in args
      or "n_obs_b" not in args
  ):
    return (
        "Error: 'n_successes_a', 'n_obs_a', 'n_successes_b', and 'n_obs_b' are"
        " required."
    )
  na_s = int(args["n_successes_a"])
  na = int(args["n_obs_a"])
  nb_s = int(args["n_successes_b"])
  nb = int(args["n_obs_b"])
  if "alternative" not in args or args["alternative"] not in (
      "larger",
      "smaller",
      "two-sided",
  ):
    return (
        "Error: 'alternative' parameter is required. You MUST specify"
        " 'larger' (A>B), 'smaller' (A<B), or 'two-sided'."
    )
  alternative = args["alternative"]

  if na <= 0 or nb <= 0:
    return "Error: Sample sizes must be greater than 0"
  if na_s < 0 or na_s > na or nb_s < 0 or nb_s > nb:
    return "Error: Success counts must be between 0 and total observations"

  p_a, p_b = na_s / na, nb_s / nb
  method = "Fisher's Exact Test"
  table = [[na_s, na - na_s], [nb_s, nb - nb_s]]
  alt_map = {"smaller": "less", "larger": "greater"}
  mapped_alternative = alt_map.get(alternative, alternative)
  _, p_value = stats.fisher_exact(table, alternative=mapped_alternative)

  return (
      "Two-Sample Proportion Test Results:\n"
      f"- Method: {method}\n"
      f"- Sample A: {na_s}/{na} ({p_a:.4f})\n"
      f"- Sample B: {nb_s}/{nb} ({p_b:.4f})\n"
      "- Statistic: N/A (exact)\n"
      f"- Alternative: {alternative}\n"
      f"- P-value: {stats_utils.format_p_value(p_value)}\n"
  )


def run_mcnemar_test(args: dict[str, Any]) -> str:
  """McNemar's test for paired binary data (exact binomial)."""
  if "passed_a_failed_b" not in args or "failed_a_passed_b" not in args:
    return "Error: 'passed_a_failed_b' and 'failed_a_passed_b' are required."
  ab = int(args["passed_a_failed_b"])
  ba = int(args["failed_a_passed_b"])
  if "alternative" not in args or args["alternative"] not in (
      "larger",
      "smaller",
      "two-sided",
  ):
    return (
        "Error: 'alternative' parameter is required. You MUST specify"
        " 'larger' (A>B), 'smaller' (A<B), or 'two-sided'."
    )
  alternative = args["alternative"]

  if ab < 0 or ba < 0:
    return "Error: Counts must be non-negative"

  n = ab + ba
  if n == 0:
    p_value = 1.0
  else:
    alt_map = {
        "smaller": "less",
        "larger": "greater",
        "two-sided": "two-sided",
    }
    mapped_alternative = alt_map.get(alternative, alternative)
    res = stats.binomtest(ab, n, p=0.5, alternative=mapped_alternative)
    p_value = res.pvalue

  return (
      "McNemar's Test Results:\n"
      f"- Passed A, Failed B: {ab}\n"
      f"- Failed A, Passed B: {ba}\n"
      f"- Alternative: {alternative}\n"
      "- Exact: True\n"
      f"- P-value: {stats_utils.format_p_value(p_value)}\n"
  )


def run_continuous_means_test(args: dict[str, Any]) -> str:
  """One-sample Student's t-test for the mean of continuous scores."""
  if (
      "sample_mean" not in args
      or "sample_std" not in args
      or "n_obs" not in args
  ):
    return "Error: 'sample_mean', 'sample_std', and 'n_obs' are required."
  sample_mean = float(args["sample_mean"])
  sample_std = float(args["sample_std"])
  n_obs = int(args["n_obs"])
  null_mean = float(args.get("null_mean", 0.0))
  if "alternative" not in args or args["alternative"] not in (
      "larger",
      "smaller",
      "two-sided",
  ):
    return (
        "Error: 'alternative' parameter is required. You MUST specify"
        " 'larger', 'smaller', or 'two-sided'."
    )
  alternative = args["alternative"]

  if n_obs <= 0:
    return "Error: n_obs must be greater than 0"
  if sample_std < 0:
    return "Error: sample_std must be non-negative"

  if sample_std == 0:
    method = "Deterministic (zero variance)"
    if sample_mean == null_mean:
      z = 0.0
      p_value = 1.0
    elif sample_mean > null_mean:
      z = float("inf")
      p_value = 0.0 if alternative in ("larger", "two-sided") else 1.0
    else:  # sample_mean < null_mean
      z = float("-inf")
      p_value = 0.0 if alternative in ("smaller", "two-sided") else 1.0
  else:
    df = max(1, n_obs - 1)
    z = (sample_mean - null_mean) / (sample_std / math.sqrt(n_obs))
    method = "T-test (Student's t)"
    p_value = _get_p_value(z, alternative, dist=stats.t, df=df)

  return (
      "One-Sample Means Test Results:\n"
      f"- Method: {method}\n"
      f"- Sample mean: {sample_mean:.4f}\n"
      f"- Sample std: {sample_std:.4f}\n"
      f"- Null mean: {null_mean:.4f}\n"
      f"- Statistic: {z:.4f}\n"
      f"- Alternative: {alternative}\n"
      f"- P-value: {stats_utils.format_p_value(p_value)}\n"
  )


def run_paired_means_test(args: dict[str, Any]) -> str:
  """Paired Student's t-test for differences between matched pairs."""
  if "mean_diff" not in args or "std_diff" not in args or "n_pairs" not in args:
    if "sample_mean" in args and "sample_std" in args and "n_obs" in args:
      mean_diff = float(args["sample_mean"])
      std_diff = float(args["sample_std"])
      n_pairs = int(args["n_obs"])
    else:
      return "Error: 'mean_diff', 'std_diff', and 'n_pairs' are required."
  else:
    mean_diff = float(args["mean_diff"])
    std_diff = float(args["std_diff"])
    n_pairs = int(args["n_pairs"])

  null_mean = float(args.get("null_mean", 0.0))
  if "alternative" not in args or args["alternative"] not in (
      "larger",
      "smaller",
      "two-sided",
  ):
    return (
        "Error: 'alternative' parameter is required. You MUST specify"
        " 'larger', 'smaller', or 'two-sided'."
    )
  alternative = args["alternative"]

  if n_pairs <= 0:
    return "Error: n_pairs must be greater than 0"
  if std_diff < 0:
    return "Error: std_diff must be non-negative"

  if std_diff == 0:
    method = "Deterministic (zero variance)"
    if mean_diff == null_mean:
      t_stat = 0.0
      p_value = 1.0
    elif mean_diff > null_mean:
      t_stat = float("inf")
      p_value = 0.0 if alternative in ("larger", "two-sided") else 1.0
    else:  # mean_diff < null_mean
      t_stat = float("-inf")
      p_value = 0.0 if alternative in ("smaller", "two-sided") else 1.0
  else:
    df = max(1, n_pairs - 1)
    t_stat = (mean_diff - null_mean) / (std_diff / math.sqrt(n_pairs))
    method = "Paired T-test (Student's t)"
    p_value = _get_p_value(t_stat, alternative, dist=stats.t, df=df)

  return (
      "Paired Means Test Results:\n"
      f"- Method: {method}\n"
      f"- Mean difference (A - B): {mean_diff:.4f}\n"
      f"- Std dev of differences: {std_diff:.4f}\n"
      f"- Number of pairs: {n_pairs}\n"
      f"- Null mean: {null_mean:.4f}\n"
      f"- Statistic (t): {t_stat:.4f}\n"
      f"- Alternative: {alternative}\n"
      f"- P-value: {stats_utils.format_p_value(p_value)}\n"
  )


# Statistical testing tools.
STAT_TOOLS: set[str] = {
    "run_proportion_z_test",
    "run_proportion_z_test_2_sample",
    "run_mcnemar_test",
    "run_continuous_means_test",
    "run_paired_means_test",
}

# Maps tool name → implementation function (sync stats tools).
SYNC_TOOL_DISPATCH: dict[str, Any] = {
    "run_proportion_z_test": run_proportion_z_test,
    "run_proportion_z_test_2_sample": run_proportion_z_test_2_sample,
    "run_mcnemar_test": run_mcnemar_test,
    "run_continuous_means_test": run_continuous_means_test,
    "run_paired_means_test": run_paired_means_test,
}


# ── Backbone-based judge for response analysis ───────────────────────────────


class JudgeOutput(pydantic.BaseModel):
  """Structured output schema for the backbone-based response judge."""

  chain_of_thought: str
  answer: str


LogFn = Callable[[str], Any]


class AskQuestionJudge:
  """Judges model responses by asking a question via BackboneClient.

  Routes text and multimodal prompts with interleaved images through the
  Gemini backbone client.

  Usage::

      backbone = backbone_lib.get_backbone_client("gemini-3.8-flash")
      judge = AskQuestionJudge(
          backbone=backbone,
          question_text="What hand is the person using?",
          answer_options=["Left", "Right"],
      )
      judgments = await judge.judge_all(responses, log_fn=print)
  """

  def __init__(
      self,
      backbone: backbone_lib.BackboneClient,
      question_text: str,
      answer_options: list[str],
      *,
      model: str = model_registry.DEFAULT_GEMINI_MODEL,
      max_retries: int = 2,
      jitter: float = 1.0,
      concurrency: int = 20,
  ):
    self._backbone = backbone
    self._question_text = question_text
    self._answer_options = answer_options
    self._model = (
        model_registry.normalize_model_address(model)
        if isinstance(model, str)
        else model
    )
    self._max_retries = max_retries
    self._jitter = jitter
    self._semaphore = asyncio.Semaphore(concurrency)

  # ── Helpers ──────────────────────────────────────────────────────────────

  @classmethod
  def extract_images_from_content(
      cls,
      content: content_api.Content,
  ) -> list[backbone_lib.ImagePart]:
    """Extracts image chunks from a ``content_api.Content`` as ``ImagePart``s.

    Iterates over the content chunks and collects any with an ``image/*``
    MIME type into ``ImagePart`` objects for the Gemini backbone client.

    Args:
      content: A ``content_api.Content`` object (iterable of chunks).

    Returns:
      A list of ``ImagePart`` instances (empty if the content has no images).
    """
    del cls
    images: list[backbone_lib.ImagePart] = []
    for chunk in content:
      mimetype = getattr(chunk, "mimetype", "") or ""
      if mimetype.startswith("image/"):
        data = getattr(chunk, "data", b"")
        if data:
          images.append(backbone_lib.ImagePart(data=data, mimetype=mimetype))
    return images

  @classmethod
  def build_judge_prompt(
      cls,
      prompt_text: str,
      response_text: str,
      question_text: str,
      answer_options: list[str],
      *,
      prompt_images: list[backbone_lib.ImagePart] | None = None,
      response_images: list[backbone_lib.ImagePart] | None = None,
  ) -> backbone_lib.Prompt:
    """Builds the judge prompt for the backbone model.

    When images are provided, returns a multimodal ``list[PromptPart]``
    with image parts interleaved at the appropriate positions. Otherwise
    returns a plain string for backward compatibility.
    """
    del cls
    answer_str = "[" + ", ".join(answer_options) + "]"

    header = (
        "You will review a conversation and answer a question.\n\n"
        "You will receive:\n"
        "  - <CONVERSATION_HISTORY>: the conversation.\n"
        "  - <QUESTION>: the question to answer.\n"
        "  - <ANSWER_OPTIONS>: your available options.\n\n"
        "Return a JSON object:\n```json\n{\n"
        '    "chain_of_thought": <your reasoning>,\n'
        '    "answer": <one of the options>\n}\n'
        "```\n"
    )

    user_section = f"<USER>\n{prompt_text}\n</USER>\n"
    model_section = f"<MODEL>\n{response_text}\n</MODEL>\n"
    footer = (
        "</CONVERSATION_HISTORY>\n"
        f"<QUESTION>\n{question_text}\n</QUESTION>\n\n"
        f"<ANSWER_OPTIONS>\n{answer_str}\n</ANSWER_OPTIONS>\n"
    )

    has_images = bool(prompt_images or response_images)

    if not has_images:
      return (
          header
          + "<CONVERSATION_HISTORY>\n"
          + user_section
          + model_section
          + footer
      )

    # Build multimodal prompt with interleaved images.
    parts: list[backbone_lib.PromptPart] = [
        header + "<CONVERSATION_HISTORY>\n" + user_section,
    ]
    for img in prompt_images or []:
      parts.append(img)
    parts.append(model_section)
    for img in response_images or []:
      parts.append(img)
    parts.append(footer)
    return parts

  # ── Judging ─────────────────────────────────────────────────────────────

  async def judge_one(
      self,
      idx: int,
      probe_content: content_api.Content | None,
      response_content: content_api.Content,
      *,
      log_fn: LogFn | None = None,
  ) -> dict[str, Any]:
    """Judges a single response via the backbone.

    Retries up to ``max_retries`` times with exponential backoff on
    transient backbone errors before falling back to ``"Unknown"``.

    Args:
      idx: Index of this response (for logging).
      probe_content: The original prompt content (may contain images).
      response_content: The model response content (may contain images).
      log_fn: Optional logging callback.

    Returns:
      A dict with keys: response_idx, prompt, response, judgment, reasoning.
    """
    if self._jitter > 0:
      await asyncio.sleep(random.uniform(0, self._jitter))
    async with self._semaphore:
      prompt_text = (
          probe_content.as_text(ignore_unsupported_types=True)
          if probe_content
          else ""
      )
      prompt_images = (
          self.extract_images_from_content(probe_content)
          if probe_content
          else []
      )
      resp_text = response_content.as_text(ignore_unsupported_types=True)
      resp_images = self.extract_images_from_content(response_content)
      judge_prompt = self.build_judge_prompt(
          prompt_text,
          resp_text,
          self._question_text,
          self._answer_options,
          prompt_images=prompt_images,
          response_images=resp_images,
      )
      ans = "Unknown"
      reason = ""
      for attempt in range(self._max_retries + 1):
        try:
          raw_json = await self._backbone.generate(
              model=self._model,
              prompt=judge_prompt,
              response_schema=JudgeOutput,
              temperature=0.0,
          )
          clean_text = raw_json.strip()
          if clean_text.startswith("```"):
            clean_text = re.sub(
                r"^```(?:json)?\s*\n?", "", clean_text, flags=re.IGNORECASE
            )
            clean_text = re.sub(r"\n?```\s*$", "", clean_text).strip()
          elif "{" in clean_text and "}" in clean_text:
            start = clean_text.find("{")
            end = clean_text.rfind("}") + 1
            candidate = clean_text[start:end]
            try:
              json.loads(candidate)
              clean_text = candidate
            except (json.JSONDecodeError, ValueError, TypeError):
              pass
          parsed = json.loads(clean_text)
          ans = parsed.get("answer", "Unknown")
          reason = parsed.get("chain_of_thought", "")
          # Validate answer is one of the options.
          if ans not in self._answer_options:
            ans = "Unknown"
          break
        except backbone_lib.LLM_CALL_ERRORS as e:
          if attempt < self._max_retries:
            delay = 2.0 * (2**attempt)
            if log_fn:
              log_fn(
                  f"  [Judge Retry] Response {idx}: attempt"
                  f" {attempt + 1}/{self._max_retries + 1}"
                  f" failed, retrying in {delay:.0f}s: {e}"
              )
            await asyncio.sleep(delay)
          else:
            if log_fn:
              log_fn(
                  f"  [Judge Error] Response {idx}: all"
                  f" {self._max_retries + 1} attempts failed: {e}"
              )
      return {
          "response_idx": idx,
          "prompt": prompt_text,
          "response": resp_text,
          "judgment": ans,
          "reasoning": reason,
      }

  async def judge_all(
      self,
      responses: list[Any],
      *,
      log_fn: LogFn | None = None,
  ) -> list[dict[str, Any]]:
    """Judges all responses concurrently via the backbone.

    Args:
      responses: List of ``content_api.Response`` objects. Each must have
        ``.content`` and optionally ``.probe.content``.
      log_fn: Optional logging callback.

    Returns:
      A list of judgment dicts, one per response.
    """
    tasks = [
        self.judge_one(
            idx=i,
            probe_content=r.probe.content if r.probe else None,
            response_content=r.content,
            log_fn=log_fn,
        )
        for i, r in enumerate(responses)
    ]
    return list(await asyncio.gather(*tasks))


# ── Procedural Autorater Registry ─────────────────────────────────────────────

ProceduralAutorater = Callable[[str, dict[str, Any]], float]

PROCEDURAL_AUTORATERS: dict[str, ProceduralAutorater] = {}


def register_autorater(
    name: str,
) -> Callable[[ProceduralAutorater], ProceduralAutorater]:
  """Decorator to register a procedural autorater by name."""

  def decorator(fn: ProceduralAutorater) -> ProceduralAutorater:
    PROCEDURAL_AUTORATERS[name] = fn
    return fn

  return decorator


@register_autorater("count_words")
def count_words(text: str, params: dict[str, Any]) -> float:
  """Counts whitespace-separated words in the response text."""
  del params  # Unused.
  return float(len(text.split()))


@register_autorater("count_hedges")
def count_hedges(text: str, params: dict[str, Any]) -> float:
  """Counts hedging markers, evaluates hedge density, or unhedged status."""
  return hedging_autorater.count_hedges(text, params)


@register_autorater("has_disclaimer")
def has_disclaimer(text: str, params: dict[str, Any]) -> float:
  """Checks whether the response contains boilerplate disclaimers."""
  return hedging_autorater.has_disclaimer(text, params)


@register_autorater("verbalized_confidence")
def verbalized_confidence(text: str, params: dict[str, Any]) -> float:
  """Extracts explicit verbalized confidence percentage (0-100), or -1.0."""
  return hedging_autorater.verbalized_confidence(text, params)
