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

"""Unit tests for the visual symmetry verification runner.

Focuses on `_score_completions`, which converts raw multimodal responses into
continuous IoU scores. This is the layer where a mis-named metric key silently
scored every probe as a failure, so it is covered here directly.
"""

import asyncio
import io
import json
import os
from typing import Any
from unittest import mock

from absl import flags
from absl.testing import absltest
import numpy as np
import PIL.Image

from disco.tasks.visual_symmetry import eval as visual_symmetry_eval
from disco.tasks.visual_symmetry import image_gen as visual_symmetry_image_gen
from disco.tasks.visual_symmetry import verify_visual_symmetry


def _as_float(value: float | None) -> float:
  """Narrows an optional score, failing loudly if the probe was unscoreable."""
  if value is None:
    raise AssertionError("Expected a scoreable probe, got None.")
  return value


def _as_detail(value: dict[str, Any] | None) -> dict[str, Any]:
  """Narrows an optional score record, failing loudly if the probe was skipped."""
  if value is None:
    raise AssertionError("Expected a scored probe, got None.")
  return value


def _png_bytes(arr: np.ndarray) -> bytes:
  buf = io.BytesIO()
  PIL.Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
  return buf.getvalue()


class _FakeImageChunk:
  """Mimics a backbone image chunk exposing raw PNG bytes."""

  def __init__(self, data: bytes):
    self.data = data


class _FakeContent:
  """Mimics response content, which may or may not carry images.

  `truthy` is tracked separately from `images` so that a response which
  returned text but no image -- the common real-world failure -- can be
  distinguished from a response with no content at all.
  """

  def __init__(self, images: list[Any], truthy: bool = True):
    self._images = images
    self._truthy = truthy

  def __bool__(self) -> bool:
    return self._truthy

  def as_images(self, ignore_unsupported_types: bool = False) -> list[Any]:
    del ignore_unsupported_types
    return list(self._images)


class _FakeResponse:

  def __init__(self, images: list[Any], truthy: bool = True):
    self.content = _FakeContent(images, truthy=truthy)


def _image_response(arr: np.ndarray) -> _FakeResponse:
  return _FakeResponse([_FakeImageChunk(_png_bytes(arr))])


def _text_only_response() -> _FakeResponse:
  """A response that came back successfully but carries no image."""
  return _FakeResponse([], truthy=True)


def _empty_response() -> _FakeResponse:
  """A response with no content at all."""
  return _FakeResponse([], truthy=False)


class ScoreCompletionsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.gt, self.prompt, _ = visual_symmetry_image_gen.generate_symmetry_image(
        shape_type="english_letter",
        seed=1,
        fg_color="black",
    )

  def test_perfect_completion_scores_near_one(self):
    """A correct completion must not score 0.0.

    This is the regression test for reading a metric key that
    `score_symmetry_completion` never emits, which made every probe score 0.0
    regardless of what the model returned.
    """
    ious, _ = verify_visual_symmetry.score_completions(
        raw_responses=[_image_response(self.gt)],
        gt_cells_list=[self.gt],
        iou_threshold=0.9,
    )
    self.assertLen(ious, 1)
    self.assertIsNotNone(ious[0])
    self.assertAlmostEqual(_as_float(ious[0]), 1.0, places=3)

  def test_poor_completion_scores_low_but_not_none(self):
    """A genuinely bad completion is data, and must be scored, not dropped."""
    ious, _ = verify_visual_symmetry.score_completions(
        raw_responses=[_image_response(self.prompt)],
        gt_cells_list=[self.gt],
        iou_threshold=0.9,
    )
    self.assertIsNotNone(ious[0])
    self.assertLess(_as_float(ious[0]), 0.5)

  def test_scores_are_not_all_identical(self):
    """Guards against any future regression that collapses every score."""
    ious, _ = verify_visual_symmetry.score_completions(
        raw_responses=[_image_response(self.gt), _image_response(self.prompt)],
        gt_cells_list=[self.gt, self.gt],
        iou_threshold=0.9,
    )
    self.assertLen(ious, 2)
    self.assertNotAlmostEqual(_as_float(ious[0]), _as_float(ious[1]))

  def test_missing_response_is_none_not_zero(self):
    """An unanswered probe is missing data, not a model failure."""
    ious, _ = verify_visual_symmetry.score_completions(
        raw_responses=[None],
        gt_cells_list=[self.gt],
        iou_threshold=0.9,
    )
    self.assertEqual(ious, [None])

  def test_response_without_image_is_none_not_zero(self):
    ious, _ = verify_visual_symmetry.score_completions(
        raw_responses=[_text_only_response()],
        gt_cells_list=[self.gt],
        iou_threshold=0.9,
    )
    self.assertEqual(ious, [None])

  def test_response_with_empty_content_is_none_not_zero(self):
    ious, _ = verify_visual_symmetry.score_completions(
        raw_responses=[_empty_response()],
        gt_cells_list=[self.gt],
        iou_threshold=0.9,
    )
    self.assertEqual(ious, [None])

  def test_undecodable_image_is_none_not_zero(self):
    ious, _ = verify_visual_symmetry.score_completions(
        raw_responses=[_FakeResponse([_FakeImageChunk(b"not-a-png")])],
        gt_cells_list=[self.gt],
        iou_threshold=0.9,
    )
    self.assertEqual(ious, [None])

  def test_mixed_batch_preserves_positional_alignment(self):
    """Scores must stay index-aligned with ground truth, gaps included."""
    ious, _ = verify_visual_symmetry.score_completions(
        raw_responses=[
            _image_response(self.gt),
            None,
            _image_response(self.gt),
        ],
        gt_cells_list=[self.gt, self.gt, self.gt],
        iou_threshold=0.9,
    )
    self.assertLen(ious, 3)
    self.assertIsNone(ious[1])
    self.assertAlmostEqual(_as_float(ious[0]), 1.0, places=3)
    self.assertAlmostEqual(_as_float(ious[2]), 1.0, places=3)

  def test_excess_responses_are_ignored(self):
    """More responses than ground truths must not index out of range."""
    ious, _ = verify_visual_symmetry.score_completions(
        raw_responses=[_image_response(self.gt), _image_response(self.gt)],
        gt_cells_list=[self.gt],
        iou_threshold=0.9,
    )
    self.assertLen(ious, 1)


class ExtractResponseImageTest(absltest.TestCase):
  """Tests the shared definition of a "usable" symmetry response.

  `_extract_response_image` is consulted by three places that must agree: the
  query retry loop, the high-failure-rate abort guard, and the scorer. If they
  disagree, a response can be retried but not scored, or scored but not
  counted toward the abort threshold.
  """

  def setUp(self):
    super().setUp()
    self.arr = np.zeros((40, 40, 3), dtype=np.uint8)

  def test_decodes_png_payload(self):
    out = verify_visual_symmetry.extract_response_image(
        _image_response(self.arr)
    )
    self.assertIsNotNone(out)
    assert out is not None
    self.assertEqual(out.shape, (40, 40, 3))

  def test_none_response(self):
    self.assertIsNone(verify_visual_symmetry.extract_response_image(None))

  def test_empty_content(self):
    self.assertIsNone(
        verify_visual_symmetry.extract_response_image(_empty_response())
    )

  def test_text_only_response(self):
    self.assertIsNone(
        verify_visual_symmetry.extract_response_image(_text_only_response())
    )

  def test_undecodable_payload(self):
    self.assertIsNone(
        verify_visual_symmetry.extract_response_image(
            _FakeResponse([_FakeImageChunk(b"not-a-png")])
        )
    )


class SuccessCriterionTest(absltest.TestCase):
  """Covers the self-mirror + left-fidelity criterion.

  The criterion exists because comparing pixel-for-pixel against ground truth
  charges an image model for re-rendering the canvas at a slightly different
  scale or offset -- registration error, not a symmetry error. Measuring
  symmetry inside the returned image removes that confound, but on its own it
  is trivially gamed, so the left half the model was handed must survive too.
  """

  def setUp(self):
    super().setUp()
    self.gt, self.prompt, _ = visual_symmetry_image_gen.generate_symmetry_image(
        shape_type="english_letter",
        seed=1,
        fg_color="black",
    )

  def test_blank_canvas_is_perfectly_self_mirrored_but_must_not_pass(self):
    """A blank image is symmetric about every axis.

    Without the fidelity term it would score a perfect 1.0, so this is the case
    the term must catch.
    """
    blank = np.full_like(self.gt, 255)
    scores, details = verify_visual_symmetry.score_completions(
        raw_responses=[_image_response(blank)],
        gt_cells_list=[self.gt],
        iou_threshold=0.9,
        probe_images=[self.prompt],
    )
    self.assertAlmostEqual(
        _as_detail(details[0])["iou_self_mirror"], 1.0, places=3
    )
    self.assertFalse(_as_detail(details[0])["is_success_left_fidelity"])
    self.assertFalse(_as_detail(details[0])["is_success"])
    # The gate must also be folded into the scalar, since every downstream
    # caller decides success by comparing that scalar to the threshold.
    self.assertEqual(_as_float(scores[0]), 0.0)

  def test_correct_completion_passes_both_terms(self):
    scores, details = verify_visual_symmetry.score_completions(
        raw_responses=[_image_response(self.gt)],
        gt_cells_list=[self.gt],
        iou_threshold=0.9,
        probe_images=[self.prompt],
    )
    self.assertTrue(_as_detail(details[0])["is_success_self_mirror"])
    self.assertTrue(_as_detail(details[0])["is_success_left_fidelity"])
    self.assertTrue(_as_detail(details[0])["is_success"])
    self.assertGreaterEqual(_as_float(scores[0]), 0.9)

  def test_all_three_criteria_are_always_reported(self):
    """All three criteria are reported regardless of the deciding mode."""
    _, details = verify_visual_symmetry.score_completions(
        raw_responses=[_image_response(self.gt)],
        gt_cells_list=[self.gt],
        iou_threshold=0.9,
        probe_images=[self.prompt],
    )
    for key in (
        "iou_self_mirror",
        "left_fidelity",
        "iou_vs_ground_truth",
        "is_success_self_mirror",
        "is_success_left_fidelity",
        "is_success_ground_truth",
    ):
      self.assertIn(key, _as_detail(details[0]))
    self.assertIsNotNone(_as_detail(details[0])["left_fidelity"])

  def test_ground_truth_mode_still_available(self):
    _, details = verify_visual_symmetry.score_completions(
        raw_responses=[_image_response(self.gt)],
        gt_cells_list=[self.gt],
        iou_threshold=0.9,
        probe_images=[self.prompt],
        success_mode="ground_truth",
    )
    self.assertEqual(_as_detail(details[0])["success_mode"], "ground_truth")
    self.assertAlmostEqual(
        _as_detail(details[0])["iou_vs_ground_truth"], 1.0, places=3
    )

  def test_left_fidelity_is_none_when_probe_not_supplied(self):
    """Makes the degraded state explicit when no probe is supplied."""
    _, details = verify_visual_symmetry.score_completions(
        raw_responses=[_image_response(self.gt)],
        gt_cells_list=[self.gt],
        iou_threshold=0.9,
    )
    self.assertIsNone(_as_detail(details[0])["left_fidelity"])
    self.assertTrue(_as_detail(details[0])["is_success_left_fidelity"])

  def test_criterion_breakdown_summarises_the_batch(self):
    _, details = verify_visual_symmetry.score_completions(
        raw_responses=[
            _image_response(self.gt),
            _image_response(np.full_like(self.gt, 255)),
        ],
        gt_cells_list=[self.gt, self.gt],
        iou_threshold=0.9,
        probe_images=[self.prompt, self.prompt],
    )
    summary = verify_visual_symmetry.log_criterion_breakdown("t", details)
    self.assertEqual(summary["n_scored"], 2)
    self.assertAlmostEqual(summary["rate_left_fidelity"], 0.5)
    self.assertAlmostEqual(summary["rate_self_mirror"], 1.0)


class MinResponseFractionTest(absltest.TestCase):
  """`verify_single_hypothesis` applies --min_response_fraction."""

  _ENTRY = {
      "text": "The model fails on circles.",
      "unique_key": "k",
      "hypothesis_id": "h",
      "origin_model": "m",
      "origin_seed": "s",
      "origin_method": "DISCO",
  }

  def _verify(self, plan, ious, min_fraction=0.9):
    probe = np.zeros((4, 4, 3), dtype=np.uint8)
    half = len(ious) // 2
    # `_score_completions` runs once, or once per arm of a paired plan.
    chunks = iter([ious[:half], ious[half:]] if plan.is_paired else [ious])

    def score(raw, *args, **kwargs):
      del raw, args, kwargs
      part = next(chunks)
      return part, [{} for _ in part]

    class Candidate:

      def respond(self, content):
        del content
        return object()

    with (
        mock.patch.object(
            verify_visual_symmetry,
            "plan_verification_probes",
            mock.AsyncMock(return_value=plan),
        ),
        mock.patch.object(
            verify_visual_symmetry,
            "_generate_probe_batch",
            lambda spec, n_probes, seed_start: (
                [probe] * n_probes,
                [probe] * n_probes,
                "complete it",
            ),
        ),
        mock.patch.object(
            verify_visual_symmetry, "_extract_response_image", lambda r: probe
        ),
        mock.patch.object(verify_visual_symmetry, "_score_completions", score),
        mock.patch.object(
            verify_visual_symmetry, "_log_criterion_breakdown", lambda *a: {}
        ),
    ):
      return asyncio.run(
          verify_visual_symmetry.verify_single_hypothesis(
              h_entry=dict(self._ENTRY),
              candidate_cls=Candidate(),
              agent_model="agent",
              num_probes=plan.num_probes,
              null_accuracy=0.9,
              alpha=0.05,
              kappa=0.5,
              iou_threshold=0.9,
              concurrency=4,
              probe_seed_offset=0,
              request_delay=0.0,
              min_response_fraction=min_fraction,
          )
      )

  def test_single_condition_short_batch_is_not_scored(self):
    plan = verify_visual_symmetry.VerificationTestPlan(num_probes=10)
    # 8 of 10 probes scored: 0.8 < 0.9.
    rec = self._verify(plan, [1.0] * 8 + [None] * 2)
    self.assertEqual(rec["status"], "INSUFFICIENT_RESPONSES")
    self.assertEqual(
        rec["error_message"],
        "INSUFFICIENT_RESPONSES: 8/10 responses received (< 0.9 required)",
    )
    self.assertFalse(rec["verified"])
    self.assertEqual(verify_visual_symmetry.record_status(rec), rec["status"])

  def test_single_condition_enough_is_scored(self):
    plan = verify_visual_symmetry.VerificationTestPlan(num_probes=10)
    rec = self._verify(plan, [1.0] * 9 + [None])
    self.assertEqual(rec["status"], "OK")
    # The old hard-coded 20% abort is gone: 8/10 passes when the flag is 0.8.
    rec = self._verify(plan, [1.0] * 8 + [None] * 2, min_fraction=0.8)
    self.assertEqual(rec["status"], "OK")

  def test_paired_counts_complete_pairs(self):
    plan = verify_visual_symmetry.VerificationTestPlan(
        is_paired=True,
        num_probes=20,
        direction="larger",
        condition_b=verify_visual_symmetry.ProbeSpec(shape_type="square"),
    )
    # Arm A: all 10 scored; arm B: 2 missing, so 8 of 10 pairs complete.
    rec = self._verify(plan, [1.0] * 10 + [0.0] * 8 + [None] * 2)
    self.assertEqual(rec["status"], "INSUFFICIENT_RESPONSES")
    self.assertEqual(
        rec["error_message"],
        "INSUFFICIENT_RESPONSES: 8/10 complete pairs received (< 0.9 required)",
    )


class SummarizeRecordsTest(absltest.TestCase):

  def test_failed_records_are_dropped_not_counted_as_false(self):
    records = [
        {"origin_method": "DISCO", "verified": True, "status": "OK"},
        {"origin_method": "DISCO", "verified": False, "status": "OK"},
        # Written by this version: an explicit failed status.
        {
            "origin_method": "DISCO",
            "verified": False,
            "status": "SCRAPE_FAILED",
            "verification_status": "EXECUTION_ERROR",
        },
        # Written before records carried `status`.
        {"origin_method": "ART", "verified": True},
        {
            "origin_method": "ART",
            "verified": False,
            "verification_status": "EXECUTION_ERROR",
        },
        {
            "origin_method": "ART",
            "verified": False,
            "verification_status": "RATE_LIMITED",
        },
    ]
    methods, overall = verify_visual_symmetry.summarize_records(records)
    self.assertEqual(
        methods["DISCO"],
        {
            "total_tested": 2,
            "total_verified": 1,
            "total_failed": 1,
            "precision": 0.5,
            "empirical_fdr": 0.5,
        },
    )
    self.assertEqual(
        methods["ART"],
        {
            "total_tested": 1,
            "total_verified": 1,
            "total_failed": 2,
            "precision": 1.0,
            "empirical_fdr": 0.0,
        },
    )
    self.assertEqual(
        overall, {"total_tested": 3, "total_verified": 2, "total_failed": 3}
    )

  def test_record_status(self):
    self.assertEqual(
        verify_visual_symmetry.record_status({"verified": False}), "OK"
    )
    self.assertEqual(
        verify_visual_symmetry.record_status(
            {"verification_status": "EXECUTION_ERROR"}
        ),
        "SCRAPE_FAILED",
    )
    self.assertEqual(
        verify_visual_symmetry.record_status(
            {"status": "OK", "verification_status": "EXECUTION_ERROR"}
        ),
        "OK",
    )


class CollectDiscoveredHypothesesTest(absltest.TestCase):

  def test_collects_from_run_disco_subdir_checkpoint(self):
    exp_dir = self.create_tempdir().full_path
    run_dir = os.path.join(exp_dir, "run_01")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "checkpoint.json"), "w") as f:
      json.dump(
          {
              "target_model": "gemini-3.1-flash-image",
              "seed": 42,
              "hypothesis_groups": {
                  "h_1": {"text": "The model fails on maze patterns."},
                  "h_2": {"text": "The model fails on filigree patterns."},
              },
              "extra": {
                  "terminal_statuses": {"h_1": "H", "h_2": "P"},
                  "final_e_values": {"h_1": 25.0, "h_2": 3.5},
              },
          },
          f,
      )

    all_hyps = verify_visual_symmetry.collect_discovered_hypotheses(exp_dir)
    self.assertLen(all_hyps, 2)
    by_id = {h["hypothesis_id"]: h for h in all_hyps}
    self.assertEqual(by_id["h_1"]["text"], "The model fails on maze patterns.")
    self.assertEqual(by_id["h_1"]["discovery_status"], "H")
    self.assertEqual(by_id["h_1"]["discovery_e_value"], 25.0)
    # Unconfirmed hypotheses are collected too: verification is what decides.
    self.assertEqual(by_id["h_2"]["discovery_status"], "P")
    self.assertEqual(by_id["h_2"]["origin_model"], "gemini-3.1-flash-image")
    self.assertEqual(by_id["h_2"]["origin_seed"], "42")

  def test_collects_from_direct_fdr_bounds_and_stage1_results(self):
    exp_dir = self.create_tempdir().full_path
    with open(os.path.join(exp_dir, "fdr_bounds.json"), "w") as f:
      json.dump(
          {"rejected_hypotheses": ["The model fails on complex_chinese."]}, f
      )
    with open(os.path.join(exp_dir, "stage1_discovery_results.json"), "w") as f:
      json.dump(
          {
              "methods": [{
                  "method_name": "ART",
                  "predictions": ["The model fails on snowflake patterns."],
              }]
          },
          f,
      )

    hyps = verify_visual_symmetry.collect_discovered_hypotheses(exp_dir)
    texts = [h["text"] for h in hyps]
    self.assertIn("The model fails on complex_chinese.", texts)
    self.assertIn("The model fails on snowflake patterns.", texts)


class ProbeGenerationErrorPropagationTest(absltest.TestCase):

  def test_generate_probe_batch_propagates_errors(self):
    spec = verify_visual_symmetry.ProbeSpec(shape_type="english_letter")
    with mock.patch.object(
        visual_symmetry_image_gen,
        "generate_symmetry_image",
        side_effect=RuntimeError("No usable glyph font found"),
    ):
      with self.assertRaises(RuntimeError):
        verify_visual_symmetry.generate_probe_batch(
            spec=spec, n_probes=1, seed_start=0
        )

  def test_success_mode_is_the_discovery_default(self):
    """The verifier must score under the criterion discovery used."""
    self.assertEqual(
        verify_visual_symmetry.SUCCESS_MODE,
        visual_symmetry_eval.DEFAULT_SUCCESS_MODE,
    )
    self.assertIn(
        verify_visual_symmetry.SUCCESS_MODE, visual_symmetry_eval.SUCCESS_MODES
    )
    self.assertEqual(
        verify_visual_symmetry.LEFT_FIDELITY_THRESHOLD,
        visual_symmetry_eval.DEFAULT_LEFT_FIDELITY_THRESHOLD,
    )

  def test_num_probes_flag_defaults_to_100(self):
    self.assertEqual(flags.FLAGS["num_probes"].default, 100)
    self.assertIn("backbone_model", flags.FLAGS)
    for removed in (
        "agent_model",
        "query_generator",
        "inter_hypothesis_delay",
        "left_fidelity_threshold",
        "success_mode",
        "reverify_failed",
        "only_confirmed",
    ):
      self.assertNotIn(removed, flags.FLAGS)


if __name__ == "__main__":
  absltest.main()
