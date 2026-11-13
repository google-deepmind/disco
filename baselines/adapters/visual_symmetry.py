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

"""Task adapter for visual symmetry (grid completion) discovery."""

import asyncio
import io
import os
from typing import Any

from google.genai import types
import numpy as np
from PIL import Image

from disco.baselines import common
from disco.baselines.adapters import base
from disco.tasks import data_classes
from disco.tasks.visual_symmetry import eval as visual_symmetry_eval
from disco.tasks.visual_symmetry import image_gen as visual_symmetry_image_gen


class VisualSymmetryTaskAdapter(base.TaskProbeAdapter):
  """Task adapter for Visual Symmetry grid completion."""

  def __init__(
      self,
      task: data_classes.Task,
      seed_instruction: str = "",
      image_save_dir: str = "",
  ):
    super().__init__(task, seed_instruction)
    self.image_save_dir = image_save_dir

  def get_direct_interrogation_prompts(self) -> list[str]:
    """Returns direct self-report questions for visual symmetry capabilities."""
    pct = self.task_null_pct
    return [
        (
            "What visual symmetry capabilities do you possess? Specifically,"
            " can you complete the vertical reflection of symmetric English"
            " letters, Greek letters, Chinese characters, geometric shapes"
            " (squares, circles, triangles, plus signs, hearts, stripes), and"
            " finer-grained patterns (high-stroke-count Hanzi, stacked glyphs,"
            " mazes, snowflakes, filigree, nested outlines, starbursts,"
            " scattered rings, wave lines, fine noise) in 1:1 square images,"
            " on both light and dark backgrounds?"
        ),
        (
            "Under what visual conditions (such as shape type, stroke density,"
            f" or background colour) do you reliably succeed (>= {pct}%"
            f" accuracy) or fail (< {pct}% accuracy) at vertical symmetry"
            " completion?"
        ),
        (
            "Are you better at completing vertical symmetry for geometric"
            " shapes (squares, circles, triangles) or linguistic glyphs"
            " (English letters, Chinese characters)?"
        ),
    ]

  def _shape_menu(self) -> list[str]:
    """Every shape family the probe generator supports."""
    return [
        s
        for s in visual_symmetry_image_gen.list_shape_types()
        if s != "dynamic"
    ]

  def _probe_menu(self) -> str:
    """The probe space, rendered for an LLM-driven baseline's prompt.

    Must match what the DISCO arm reaches through its probe tool. If DISCO can
    explore the harder families and the ten backgrounds and the baselines
    cannot, the arms are searching different spaces and the comparison
    measures the probe menu rather than the method.

    Returns:
      Bullet-list description of supported `shape_type` and `bg_color` values.
    """
    shapes = ", ".join(repr(s) for s in self._shape_menu())
    bgs = ", ".join(
        repr(b) for b in visual_symmetry_image_gen.BACKGROUND_PALETTE
    )
    return f"- shape_type: one of [{shapes}]\n- bg_color: one of [{bgs}]\n"

  def get_fixed_probe_categories(self) -> list[dict[str, Any]]:
    """Every shape family crossed with every background in the palette.

    The uniform-probing sweep groups configs by `shape_type`, so the ten
    background variants cycle *within* a category rather than fragmenting the
    budget across 220 of them.

    Returns:
      List of probe category dictionaries with `shape_type` and `bg_color`.
    """
    categories: list[dict[str, Any]] = []
    for s in self._shape_menu():
      for bg in visual_symmetry_image_gen.BACKGROUND_PALETTE:
        categories.append({"shape_type": s, "bg_color": bg})
    return categories

  def _build_red_team_system_prompt(self) -> str:
    return (
        "You are an autonomous behavioral auditor characterizing the visual"
        " reasoning and vertical symmetry completion capabilities of a target"
        " multimodal vision-language model. Your objective is to propose"
        " diverse shape types and test symmetry completion performance across"
        " categories.\nPropose JSON probe configurations with fields:\n"
        + self._probe_menu()
        + "Output your proposal wrapped in ```json [ { ... } ] ```."
    )

  def _build_looped_investigator_system_prompt(self) -> str:
    return (
        "You are an autonomous AI scientist investigating the vertical"
        " symmetry completion capability of a multimodal vision model. You"
        " design visual probes, analyze empirical IoU scores, maintain a"
        " research log, and synthesize accurate capability hypotheses."
    )

  async def execute_probe(
      self,
      client: Any,
      target_model: str,
      probe_config: dict[str, Any],
      round_idx: int,
      probe_idx: int,
      k_samples: int = 1,
      semaphore: asyncio.Semaphore | None = None,
      log: common.LogFn = common.default_log,
  ) -> common.GenericProbeObservation:
    """Generates a visual symmetry probe image, queries the target, and scores IoU."""
    shape_type = probe_config.get("shape_type", "square")
    rows = 10
    cols = 10
    seed = probe_config.get("seed", 42 + round_idx * 100 + probe_idx)
    # The background must be whatever the condition asked for, and the same
    # value must reach the scorer below: ink is extracted relative to it, so a
    # navy probe scored as if it were white reads as an all-ink canvas.
    bg_color = probe_config.get("bg_color", "white")
    # fg is left to the generator unless the condition names one, so that a
    # dark background does not end up with black ink on it.
    fg_color = probe_config.get("fg_color")

    gt_img, prompt_img, _ = visual_symmetry_image_gen.generate_symmetry_image(
        shape_type=shape_type,
        rows=rows,
        cols=cols,
        bg_color=bg_color,
        fg_color=fg_color,
        seed=seed,
        allow_dynamic=True,
    )

    save_dir = probe_config.get("image_save_dir") or self.image_save_dir
    if save_dir:
      try:
        os.makedirs(save_dir, exist_ok=True)
        p_path = os.path.join(
            save_dir,
            f"prompt_r{round_idx:02d}_p{probe_idx:04d}_{shape_type}.png",
        )
        gt_path = os.path.join(
            save_dir,
            f"gt_r{round_idx:02d}_p{probe_idx:04d}_{shape_type}.png",
        )
        if not os.path.exists(p_path):
          common.save_png(prompt_img, p_path, log)
        if not os.path.exists(gt_path):
          common.save_png(gt_img, gt_path, log)
      except (OSError, ValueError, TypeError) as e:
        log(f"  [Warning] Could not save probe images to {save_dir}: {e}")

    prompt_buf = io.BytesIO()
    Image.fromarray(prompt_img).save(prompt_buf, format="PNG")
    prompt_bytes = prompt_buf.getvalue()

    system_instruction = (
        "You are an expert at visual reflection symmetry. You are given an"
        " image containing the left half of a horizontally symmetric pattern on"
        f" a {bg_color} canvas. Your task is to complete the entire symmetric"
        " pattern by completing the right half."
    )
    prompt_text = (
        "Please complete the vertical symmetry of the shape shown in the"
        " image. Return only the final completed full image."
    )

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part(text=prompt_text),
                types.Part.from_bytes(data=prompt_bytes, mime_type="image/png"),
            ],
        )
    ]
    gen_config = self.target_generate_config(
        temperature=0.7 if k_samples > 1 else 0.0,
        system_instruction=system_instruction,
    )

    async def _sample_one(s_idx: int) -> tuple[float, str, str]:
      """Returns (iou, response_text, failure_reason); "" reason when valid.

      A failed call, an unreadable or missing image, or a scoring error is an
      invalid sample (missing data), as in the DISCO arm, which backfills
      unanswered probes and then excludes them. A returned image that
      discarded the left half it was given is a valid sample that failed.

      Args:
        s_idx: Index of the sample within this probe.
      """
      try:
        resp = await common.generate_content_with_retry(
            client=client,
            model=target_model,
            contents=contents,
            config=gen_config,
            semaphore=semaphore,
            log_fn=log,
        )
      except common.LLM_CALL_ERRORS as e:
        return 0.0, "", base.failure_reason_from_error(e)
      try:
        r_text = (
            resp.text if (resp and hasattr(resp, "text") and resp.text) else ""
        )
      except (AttributeError, ValueError, TypeError):
        r_text = ""
      try:
        pred_img = common.extract_image_from_response(resp)
      except (OSError, ValueError, TypeError, AttributeError, IndexError) as e:
        return (
            0.0,
            r_text,
            base.failure_reason_from_error(e, "unreadable image"),
        )
      if pred_img is None:
        return 0.0, r_text, "no image generated"
      if save_dir:
        out_path = os.path.join(
            save_dir,
            f"out_r{round_idx:02d}_p{probe_idx:04d}_s{s_idx}_{shape_type}.png",
        )
        common.save_png(pred_img, out_path, log)
      try:
        # Must use the same criterion as the DISCO arm. If the baseline is
        # scored against ground truth while DISCO is scored on self-mirror
        # symmetry, the two arms are measuring different things and the
        # comparison between them says nothing.
        score = visual_symmetry_eval.score_direct_symmetry_completion(
            output_image=np.array(pred_img, dtype=np.uint8),
            ground_truth_image=gt_img,
            bg_color=bg_color,
            probe_image=prompt_img,
        )
      except (
          ValueError,
          TypeError,
          RuntimeError,
          ArithmeticError,
          IndexError,
          KeyError,
      ) as e:
        return 0.0, r_text, base.failure_reason_from_error(e, "scoring error")
      iou = float(score.get("iou", 0.0))
      if not score.get("is_success_left_fidelity", True):
        # Keep the scalar consistent with `is_success`: a returned image
        # that discarded the left half it was given is not a success, no
        # matter how symmetric it is.
        iou = 0.0
      return iou, r_text, ""

    sample_results = await asyncio.gather(
        *[_sample_one(s_idx) for s_idx in range(k_samples)]
    )
    # Score over the valid samples only.
    reasons = [r[2] for r in sample_results]
    responses_text = [r[1] for r in sample_results]
    valid_idx = [i for i, r in enumerate(reasons) if not r]
    ious = [sample_results[i][0] for i in valid_idx]
    n_valid = len(ious)
    failure_reason = base.first_failure_reason(reasons)
    threshold = visual_symmetry_eval.DEFAULT_IOU_THRESHOLD

    avg_iou = float(np.mean(ious)) if ious else 0.0
    sample_passes = [iou >= threshold for iou in ious]
    passed = n_valid > 0 and base.majority_vote(sample_passes, n_valid)
    n_failed = k_samples - n_valid
    failed_note = f", {n_failed} sample(s) failed" if n_failed else ""

    if n_valid == 0:
      resp_summary = common.no_usable_response_summary(failure_reason)
    else:
      resp_summary = (
          f"Avg IoU={avg_iou:.4f}, Pass={passed} (samples={len(ious)}"
          f"{failed_note})"
      )

    if save_dir:
      output_images = [
          f"out_r{round_idx:02d}_p{probe_idx:04d}_s{s}_{shape_type}.png"
          for s in valid_idx
      ]
    else:
      output_images = []

    return common.GenericProbeObservation(
        round_idx=round_idx,
        probe_idx=probe_idx,
        category=shape_type,
        prompt_summary=f"Shape={shape_type}",
        response_summary=resp_summary,
        score=avg_iou,
        passed=passed,
        details={
            "shape_type": shape_type,
            "ious": [round(x, 4) for x in ious],
            "k_samples": k_samples,
            "iou_threshold": threshold,
            # Keep the full response text: when a probe scores oddly, the
            # target's own words are usually the fastest explanation.
            "responses_text": responses_text,
            # Names the images this probe wrote, so the record points at the
            # pixels instead of requiring them to be re-derived from the seed.
            "prompt_image": (
                f"prompt_r{round_idx:02d}_p{probe_idx:04d}_{shape_type}.png"
                if save_dir
                else ""
            ),
            "ground_truth_image": (
                f"gt_r{round_idx:02d}_p{probe_idx:04d}_{shape_type}.png"
                if save_dir
                else ""
            ),
            "output_images": output_images,
            **(
                {"sample_failure_reasons": reasons}
                if n_valid < k_samples
                else {}
            ),
        },
        n_samples=k_samples,
        n_valid_samples=n_valid,
        failure_reason=failure_reason,
    )

  def format_empirical_predictions(
      self, empirical_rates: dict[str, float]
  ) -> list[str]:
    """Formats empirical pass rates by shape category into natural-language hypotheses."""
    preds = []
    # Both directions are stated against the task's null proportion.
    pct = self.task_null_pct
    null = self.task_null_proportion
    for shape, rate in empirical_rates.items():
      clean_shape = shape.replace("_", " ")
      if rate >= null:
        preds.append(
            "The target model can perform vertical symmetry completion on"
            f" {clean_shape} shapes on 1:1 images, succeeding at least {pct}%"
            f" of the time (empirical pass rate: {rate:.1%})."
        )
      else:
        preds.append(
            "The target model struggles with vertical symmetry completion on"
            f" {clean_shape} shapes on 1:1 images, succeeding less than {pct}%"
            f" of the time (empirical pass rate: {rate:.1%})."
        )
    return preds
