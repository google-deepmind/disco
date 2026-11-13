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

"""Unit tests for visual symmetry task and hypothesis modules.

Tests image generation, evaluation scoring, and hypothesis definitions.
"""

from typing import Any

from absl.testing import absltest
import numpy as np
from PIL import Image as PIL_Image

from disco import standalone_agent
from disco import standalone_tools
from disco.tasks import data_classes
from disco.tasks.visual_symmetry import eval as visual_symmetry_eval
from disco.tasks.visual_symmetry import hypotheses as visual_symmetry_hypotheses
from disco.tasks.visual_symmetry import image_gen as visual_symmetry_image_gen
from disco.tasks.visual_symmetry import prompts as visual_symmetry_prompts


def _is_grid_symmetric(cells: np.ndarray) -> bool:
  half = cells.shape[1] // 2
  left = cells[:, :half, :]
  right = cells[:, half:, :]
  mirrored_left = np.flip(left, axis=1)
  return bool(np.array_equal(mirrored_left, right))


class VisualSymmetryImageGenTest(absltest.TestCase):
  """Tests for visual_symmetry_image_gen."""

  def test_every_shape_type_ground_truth_is_exactly_symmetric(self):
    """The core invariant: no probe may ask for an asymmetric target.

    Exhaustive over shape type x background x seed rather than spot-checked.
    The whole benchmark rests on the ground truth being symmetric, and a shape
    handler that draws after the mirror step (or one that is asymmetric in the
    font) produces a probe the model cannot possibly satisfy -- which reads as
    a model failure, not as our bug.
    """
    backgrounds = visual_symmetry_image_gen.BACKGROUND_PALETTE
    for shape_type in visual_symmetry_image_gen.list_shape_types():
      for bg_color in backgrounds:
        for seed in range(3):
          with self.subTest(shape=shape_type, bg=bg_color, seed=seed):
            gt, prompt, _ = visual_symmetry_image_gen.generate_symmetry_image(
                shape_type=shape_type,
                canvas_size=320,
                bg_color=bg_color,
                seed=seed,
                allow_dynamic=True,
            )
            half = gt.shape[1] // 2
            np.testing.assert_array_equal(
                gt[:, gt.shape[1] - half :],
                np.flip(gt[:, :half], axis=1),
                err_msg=(
                    f"{shape_type} on {bg_color} seed={seed} is not symmetric"
                ),
            )
            # And the explicit guard agrees.
            visual_symmetry_image_gen.assert_ground_truth_symmetric(
                gt, shape_type=shape_type, seed=seed
            )
            # The prompt must be the same image with the right half cleared.
            np.testing.assert_array_equal(
                prompt[:, :half],
                gt[:, :half],
                err_msg=f"{shape_type} prompt altered the given left half",
            )

  def test_every_source_glyph_is_mirror_symmetric_in_the_font(self):
    """Each *character* must be symmetric, not just the mirrored ground truth.

    `test_every_shape_type_ground_truth_is_exactly_symmetric` above cannot
    catch this. The ground truth is built as ``right := fliplr(left)``, so it
    is symmetric by construction and that assertion can never fail -- it
    reported a perfect score for all 880 probes while a third of the glyph
    list was asymmetric.

    The substantive invariant is on the source glyph. Mirroring the left half
    of an asymmetric character yields a chimera that is not the character, so
    a model that knows the language and completes the *real* glyph is marked
    wrong for being right, and the resulting "finding" is our bug.
    """
    failures = []
    groups = [
        (
            "english_letter",
            visual_symmetry_image_gen.SYMMETRIC_ENGLISH_LETTERS,
            False,
        ),
        (
            "greek_letter",
            visual_symmetry_image_gen.SYMMETRIC_GREEK_LETTERS,
            False,
        ),
        (
            "chinese_char",
            visual_symmetry_image_gen.SYMMETRIC_CHINESE_CHARS,
            False,
        ),
        (
            "complex_chinese",
            visual_symmetry_image_gen.COMPLEX_SYMMETRIC_CHINESE_CHARS,
            False,
        ),
    ]
    # `multi_glyph` renders at 0.26 of the canvas rather than 0.72, and thin
    # strokes lose proportionally more ink when small, so full-size success
    # does not imply stacked success.
    groups.append((
        "multi_glyph",
        [
            c * 3
            for c in (
                visual_symmetry_image_gen.SYMMETRIC_CHINESE_CHARS
                + visual_symmetry_image_gen.SYMMETRIC_GREEK_LETTERS
                + visual_symmetry_image_gen.SYMMETRIC_ENGLISH_LETTERS
            )
        ],
        True,
    ))

    for group, chars, stacked in groups:
      for char in chars:
        img = PIL_Image.new("RGB", (320, 320), (255, 255, 255))
        visual_symmetry_image_gen.draw_glyphs_centered(
            img,
            char,
            fg_rgb=(0, 0, 0),
            font_path=visual_symmetry_image_gen.FONT_PATH,
            stacked=stacked,
            shape_type=group,
        )
        score = visual_symmetry_image_gen.glyph_self_mirror_f1(
            img, (255, 255, 255)
        )
        if score < visual_symmetry_image_gen.MIN_GLYPH_SELF_MIRROR_F1:
          failures.append(f"{group}: {char} (U+{ord(char[0]):04X}) {score:.4f}")

    self.assertEmpty(
        failures,
        "These characters are not mirror symmetric in the rendering font, so "
        "mirroring their left half produces a shape that is not the "
        "character. Remove them from their list:\n"
        + "\n".join(failures),
    )

  def test_source_glyph_guard_rejects_a_known_asymmetric_character(self):
    """The guard must be able to fail, or it is another tautology.

    The bug this whole check replaces was an assertion that could not fail.
    A green suite means nothing unless the check fires on a real violation, so
    pin it to a character measured as asymmetric in this font: 磊 scores 0.84,
    its right-hand 石 components differing visibly from the left.
    """
    img = PIL_Image.new("RGB", (320, 320), (255, 255, 255))
    visual_symmetry_image_gen.draw_glyphs_centered(
        img,
        "\u78ca",  # 磊
        fg_rgb=(0, 0, 0),
        font_path=visual_symmetry_image_gen.FONT_PATH,
        stacked=False,
        shape_type="complex_chinese",
    )
    self.assertLess(
        visual_symmetry_image_gen.glyph_self_mirror_f1(img, (255, 255, 255)),
        visual_symmetry_image_gen.MIN_GLYPH_SELF_MIRROR_F1,
    )
    self.assertRaises(
        RuntimeError,
        visual_symmetry_image_gen.assert_source_glyph_symmetric,
        img,
        (255, 255, 255),
        "\u78ca",
        "complex_chinese",
    )

  def test_every_shape_type_renders_visible_ink(self):
    """A blank probe scores a perfect self-mirror, so blankness must be fatal.

    This is the failure mode that produced the `chinese_char` result the run
    was eventually thrown out over: the font fell back to a 4x7 bitmap box, the
    probes were effectively blank, and 0/100 looked like a finding about stroke
    complexity.
    """
    for shape_type in visual_symmetry_image_gen.list_shape_types():
      for bg_color in ("white", "navy", "lightyellow"):
        for seed in range(3):
          with self.subTest(shape=shape_type, bg=bg_color, seed=seed):
            gt, _, _ = visual_symmetry_image_gen.generate_symmetry_image(
                shape_type=shape_type,
                canvas_size=320,
                bg_color=bg_color,
                seed=seed,
                allow_dynamic=True,
            )
            bg_rgb = visual_symmetry_eval.resolve_color_rgb(bg_color)
            dist = np.sqrt(
                np.sum((gt.astype(np.float64) - bg_rgb) ** 2, axis=-1)
            )
            ink_fraction = float((dist > 30.0).mean())
            self.assertGreater(
                ink_fraction,
                0.002,
                f"{shape_type} on {bg_color} seed={seed} rendered a near-blank"
                f" probe (ink fraction {ink_fraction:.5f})",
            )

  def test_hard_shape_types_are_registered_and_distinct(self):
    types = visual_symmetry_image_gen.list_shape_types()
    for hard in visual_symmetry_image_gen.HARD_SHAPE_TYPES:
      self.assertIn(hard, types)
    self.assertEmpty(
        set(visual_symmetry_image_gen.HARD_SHAPE_TYPES)
        & set(visual_symmetry_image_gen.BASE_SHAPE_TYPES)
    )

  def test_hard_shapes_are_thinner_than_their_base_counterparts(self):
    """The hard families must actually be harder, not just differently named.

    Ink fraction is the proxy: these shapes are meant to be fine-stroked, so if
    one of them is rendering as a large solid blob the intent has been lost.
    """
    for shape_type in visual_symmetry_image_gen.HARD_SHAPE_TYPES:
      fractions = []
      for seed in range(5):
        gt, _, _ = visual_symmetry_image_gen.generate_symmetry_image(
            shape_type=shape_type, canvas_size=320, bg_color="white", seed=seed
        )
        bg_rgb = visual_symmetry_eval.resolve_color_rgb("white")
        dist = np.sqrt(np.sum((gt.astype(np.float64) - bg_rgb) ** 2, axis=-1))
        fractions.append(float((dist > 30.0).mean()))
      self.assertLess(
          float(np.mean(fractions)),
          0.35,
          f"{shape_type} covers {np.mean(fractions):.2%} of the canvas; it is"
          " a blob, not a fine-stroked pattern",
      )

  def test_low_contrast_foreground_is_rejected_not_substituted(self):
    with self.assertRaisesRegex(ValueError, "contrast"):
      visual_symmetry_image_gen.generate_symmetry_image(
          shape_type="square",
          canvas_size=320,
          bg_color="navy",
          fg_color="blue",
          seed=0,
      )

  def test_asymmetric_ground_truth_is_detected(self):
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    arr[4, 2] = 255  # Present on the left, absent on the right.
    self.assertRaisesRegex(
        RuntimeError,
        "NOT",
        visual_symmetry_image_gen.assert_ground_truth_symmetric,
        arr,
    )

  def test_list_shape_types(self):
    types = visual_symmetry_image_gen.list_shape_types()
    self.assertIsInstance(types, list)
    self.assertIn("english_letter", types)
    self.assertIn("square", types)
    self.assertIn("circle", types)
    self.assertIn("triangle", types)
    self.assertIn("plus", types)
    self.assertIn("heart", types)
    self.assertIn("dynamic", types)
    self.assertIn("random_sparse", types)
    self.assertIn("random_dense", types)
    self.assertIn("greek_letter", types)
    self.assertIn("chinese_char", types)
    self.assertIn("stripes", types)
    self.assertIn("dots", types)

  def test_generate_symmetry_image_english_letter_direct(self):
    gt_cells, prompt_cells, gt_image = (
        visual_symmetry_image_gen.generate_symmetry_image(
            shape_type="english_letter",
            seed=42,
            fg_color="blue",
        )
    )
    self.assertEqual(gt_cells.shape, (320, 320, 3))
    self.assertEqual(prompt_cells.shape, (320, 320, 3))
    self.assertEqual(gt_image.shape, (320, 320, 3))
    self.assertTrue(_is_grid_symmetric(gt_cells))

  def test_ground_truth_is_symmetric_all_shapes(self):
    for shape in visual_symmetry_image_gen.list_shape_types():
      gt, _, _ = visual_symmetry_image_gen.generate_symmetry_image(
          shape_type=shape,
          seed=123,
      )
      self.assertTrue(
          _is_grid_symmetric(gt),
          msg=f"Shape {shape} ground truth is not vertically symmetric",
      )

  def test_prompt_has_blank_right_half(self):
    _, prompt_arr, _ = visual_symmetry_image_gen.generate_symmetry_image(
        shape_type="english_letter",
        seed=42,
        bg_color="white",
    )
    white = np.array(
        visual_symmetry_image_gen._bg_color_rgb("white"), dtype=np.uint8
    )
    half = prompt_arr.shape[1] // 2
    # Midline guide exists by default at x=half
    self.assertTrue(
        np.any(prompt_arr[:, half, :] != white),
        msg="Expected midline guide at x=half by default, but none found",
    )
    # The right half beyond the midline is completely blank background
    right_beyond_midline = prompt_arr[:, half + 1 :, :]
    np.testing.assert_array_equal(
        right_beyond_midline,
        np.full_like(right_beyond_midline, white),
        err_msg="Right half beyond midline in prompt is not background color",
    )

    # When show_midline=False, entire right half from x=half is blank
    _, prompt_no_midline, _ = visual_symmetry_image_gen.generate_symmetry_image(
        shape_type="english_letter",
        seed=42,
        bg_color="white",
        show_midline=False,
    )
    right_half = prompt_no_midline[:, half:, :]
    np.testing.assert_array_equal(
        right_half,
        np.full_like(right_half, white),
        err_msg="Right half in prompt without midline is not background color",
    )

  def test_unknown_shape_raises(self):
    with self.assertRaises(ValueError):
      visual_symmetry_image_gen.generate_symmetry_image(
          shape_type="nonexistent_shape",
      )

  def test_unknown_shape_with_allow_dynamic_succeeds(self):
    gt, prompt, _ = visual_symmetry_image_gen.generate_symmetry_image(
        shape_type="novel_star_shape",
        allow_dynamic=True,
        seed=42,
    )
    self.assertEqual(gt.shape, (320, 320, 3))
    self.assertEqual(prompt.shape, (320, 320, 3))
    self.assertTrue(_is_grid_symmetric(gt))

  def test_probes_vary_across_seeds_for_all_shape_types(self):
    for shape in visual_symmetry_image_gen.list_shape_types():
      prompts = []
      for seed in range(10):
        _, prompt_cells, _ = visual_symmetry_image_gen.generate_symmetry_image(
            shape_type=shape, seed=seed
        )
        prompts.append(prompt_cells)
      # Check that at least some pairs of prompt grids across 10 seeds are
      # different.
      is_unique = any(not np.array_equal(prompts[0], p) for p in prompts[1:])
      self.assertTrue(
          is_unique,
          msg=(
              f"Shape type {shape!r} produced 10 identical prompt images across"
              " seeds!"
          ),
      )

  def test_every_shape_type_renders_substantial_ink(self):
    """Regression: a probe must never come out effectively blank.

    A missing font previously made PIL fall back to its built-in bitmap font,
    which ignores font_size and draws a ~4x7 pixel box. The resulting probes
    were blank, and the target model was scored 0/100 against them.
    """
    canvas = visual_symmetry_image_gen.DIRECT_CANVAS_SIZE
    for shape in visual_symmetry_image_gen.SUPPORTED_SHAPE_TYPES:
      with self.subTest(shape=shape):
        gt, _, _ = visual_symmetry_image_gen.render_direct_symmetry_image(
            shape_type=shape, canvas_size=canvas, seed=0
        )
        dist = np.sqrt(np.sum((gt.astype(np.float64) - 255.0) ** 2, axis=-1))
        ink_fraction = float((dist > 30.0).mean())
        self.assertGreater(
            ink_fraction,
            visual_symmetry_image_gen.MIN_GLYPH_INK_FRACTION,
            msg=(
                f"Shape {shape!r} covered only {ink_fraction:.5f} of the"
                " canvas, which means it did not really render."
            ),
        )

  def test_glyph_shapes_span_a_large_fraction_of_the_canvas(self):
    """Glyphs are drawn at 0.72 * canvas, so their bbox must be substantial."""
    canvas = visual_symmetry_image_gen.DIRECT_CANVAS_SIZE
    for shape in ("english_letter", "greek_letter", "chinese_char"):
      with self.subTest(shape=shape):
        gt, _, _ = visual_symmetry_image_gen.render_direct_symmetry_image(
            shape_type=shape, canvas_size=canvas, seed=0
        )
        dist = np.sqrt(np.sum((gt.astype(np.float64) - 255.0) ** 2, axis=-1))
        rows, cols = np.nonzero(dist > 30.0)
        self.assertGreater(rows.size, 0, msg=f"{shape!r} drew nothing at all.")
        height = int(rows.max() - rows.min() + 1)
        width = int(cols.max() - cols.min() + 1)
        # The bitmap fallback font produces a ~4x7 box; a real glyph at
        # font_size = 0.72 * 320 = 230px is an order of magnitude larger.
        self.assertGreater(height, canvas // 4, msg=f"{shape!r} bbox too short")
        self.assertGreater(width, canvas // 8, msg=f"{shape!r} bbox too narrow")

  def test_assert_glyph_rendered_raises_on_blank_canvas(self):
    blank = PIL_Image.new("RGB", (320, 320), (255, 255, 255))
    self.assertRaisesRegex(
        RuntimeError,
        "ink coverage",
        visual_symmetry_image_gen.assert_glyph_rendered,
        blank,
        (255, 255, 255),
        "chinese_char",
        "\u4e2d",
        "/fake/font.ttf",
    )


class VisualSymmetryEvalTest(absltest.TestCase):
  """Tests for visual_symmetry_eval."""

  def test_rgb_to_lab_white(self):
    white = np.array([[[255, 255, 255]]], dtype=np.uint8)
    lab = visual_symmetry_eval.rgb_to_lab(white)
    self.assertAlmostEqual(lab[0, 0, 0], 100.0, places=0)
    self.assertAlmostEqual(lab[0, 0, 1], 0.0, places=0)
    self.assertAlmostEqual(lab[0, 0, 2], 0.0, places=0)

  def test_rgb_to_lab_black(self):
    black = np.array([[[0, 0, 0]]], dtype=np.uint8)
    lab = visual_symmetry_eval.rgb_to_lab(black)
    self.assertAlmostEqual(lab[0, 0, 0], 0.0, places=0)

  def test_delta_e_identical(self):
    c1 = np.array([[[100, 150, 200]]], dtype=np.uint8)
    lab1 = visual_symmetry_eval.rgb_to_lab(c1)
    de = visual_symmetry_eval.delta_e_cie76(lab1, lab1)
    self.assertAlmostEqual(float(de[0, 0]), 0.0, places=4)

  def test_delta_e_white_vs_black(self):
    w = np.array([[[255, 255, 255]]], dtype=np.uint8)
    b = np.array([[[0, 0, 0]]], dtype=np.uint8)
    lab_w = visual_symmetry_eval.rgb_to_lab(w)
    lab_b = visual_symmetry_eval.rgb_to_lab(b)
    de = visual_symmetry_eval.delta_e_cie76(lab_w, lab_b)
    self.assertAlmostEqual(float(de[0, 0]), 100.0, places=0)

  def test_score_symmetry_completions_pass_at_k(self):
    # Ground truth: 2 probes
    gt1_img, prompt1_img, _ = visual_symmetry_image_gen.generate_symmetry_image(
        shape_type="square", seed=1
    )
    gt2_img, prompt2_img, _ = visual_symmetry_image_gen.generate_symmetry_image(
        shape_type="square", seed=2
    )

    # Probe 1 has 2 samples: [failure (prompt), success (gt)]
    # Probe 2 has 2 samples: [failure (prompt), failure (prompt)]
    output_by_probe = [
        [prompt1_img, gt1_img],
        [prompt2_img, prompt2_img],
    ]
    gt_by_probe = [gt1_img, gt2_img]

    res = visual_symmetry_eval.score_symmetry_completions_pass_at_k(
        output_images_by_probe=output_by_probe,
        ground_truth_cells_by_probe=gt_by_probe,
        probe_images_by_probe=[prompt1_img, prompt2_img],
    )

    self.assertEqual(res["n_probes"], 2)
    self.assertEqual(res["k"], 2)
    self.assertEqual(
        res["n_probes_passed"], 1
    )  # Probe 1 passed, probe 2 failed
    self.assertAlmostEqual(res["pass_at_k_rate"], 0.5)
    self.assertEqual(res["total_samples"], 4)
    self.assertEqual(res["total_samples_passed"], 1)
    self.assertAlmostEqual(res["pass_at_1_rate"], 0.25)
    self.assertTrue(res["per_probe_results"][0]["passed"])
    self.assertFalse(res["per_probe_results"][1]["passed"])

  def test_pass_at_k_excludes_probes_with_no_completion(self):
    """A probe the model never answered is missing data, not a failure."""
    gt1_img, prompt1_img, _ = visual_symmetry_image_gen.generate_symmetry_image(
        shape_type="square", seed=1
    )
    gt2_img, prompt2_img, _ = visual_symmetry_image_gen.generate_symmetry_image(
        shape_type="square", seed=2
    )

    # Probe 0 returned a perfect completion; probe 1 returned nothing at all.
    res = visual_symmetry_eval.score_symmetry_completions_pass_at_k(
        output_images_by_probe=[[gt1_img], []],
        ground_truth_cells_by_probe=[gt1_img, gt2_img],
        probe_images_by_probe=[prompt1_img, prompt2_img],
    )

    self.assertEqual(res["n_probes"], 2)
    self.assertEqual(res["n_probes_valid"], 1)
    self.assertEqual(res["n_probes_invalid"], 1)
    self.assertEqual(res["n_probes_passed"], 1)
    # 1/1, not 1/2: the unanswered probe must not be counted as a failure.
    self.assertAlmostEqual(res["pass_at_k_rate"], 1.0)
    # k is derived from valid probes, so it stays 1 rather than 1 // 2 == 0.
    self.assertEqual(res["k"], 1)
    self.assertTrue(res["per_probe_results"][0]["valid"])
    self.assertFalse(res["per_probe_results"][1]["valid"])

  def test_pass_at_k_all_probes_empty(self):
    """Zero usable completions yields a zero rate, not a division error."""
    gt_cells, prompt_cells, _ = (
        visual_symmetry_image_gen.generate_symmetry_image(
            shape_type="square", seed=1
        )
    )
    res = visual_symmetry_eval.score_symmetry_completions_pass_at_k(
        output_images_by_probe=[[], []],
        ground_truth_cells_by_probe=[gt_cells, gt_cells],
        probe_images_by_probe=[prompt_cells, prompt_cells],
    )
    self.assertEqual(res["n_probes_valid"], 0)
    self.assertEqual(res["n_probes_invalid"], 2)
    self.assertEqual(res["n_probes_passed"], 0)
    self.assertAlmostEqual(res["pass_at_k_rate"], 0.0)
    self.assertEqual(res["k"], 0)

  def test_option_a_direct_symmetry_completion_perfect(self):
    gt, prompt, _ = visual_symmetry_image_gen.generate_symmetry_image(
        shape_type="english_letter",
        seed=1,
        fg_color="black",
    )
    result = visual_symmetry_eval.score_symmetry_completion(
        output_image=gt,
        ground_truth_cells=gt,
        probe_image=prompt,
    )
    self.assertAlmostEqual(result["iou"], 1.0, places=3)
    self.assertTrue(result["is_success"])

  def test_option_a_direct_symmetry_completion_threshold_fail(self):
    gt, prompt, _ = visual_symmetry_image_gen.generate_symmetry_image(
        shape_type="english_letter",
        seed=1,
        fg_color="black",
    )
    # Returning the probe unchanged leaves the right half empty, so the
    # self-mirror term must reject it. The exact value is not pinned: under the
    # tolerant-F1 measure a near-empty right half against a full left half
    # scores around 0.27 rather than ~0.0, because the handful of midline
    # pixels that do exist are all within tolerance of the mirrored letter
    # (precision ~1.0) while recall stays tiny. What matters is that it is
    # nowhere near the 0.90 threshold.
    result = visual_symmetry_eval.score_symmetry_completion(
        output_image=prompt,
        ground_truth_cells=gt,
        probe_image=prompt,
    )
    self.assertLess(result["iou"], 0.5)
    self.assertLess(result["iou"], result["iou_threshold"])
    self.assertFalse(result["is_success"])
    self.assertFalse(result["is_success_self_mirror"])

  def test_blank_canvas_is_rejected_by_left_fidelity(self):
    """A blank canvas is perfectly self-symmetric and must still fail.

    This is the reason the success criterion is a conjunction rather than the
    symmetry term alone.
    """
    gt, prompt, _ = visual_symmetry_image_gen.generate_symmetry_image(
        shape_type="english_letter", seed=1, fg_color="black"
    )
    blank = np.full_like(gt, 255)
    result = visual_symmetry_eval.score_symmetry_completion(
        output_image=blank,
        ground_truth_cells=gt,
        probe_image=prompt,
    )
    self.assertAlmostEqual(result["iou_self_mirror"], 1.0, places=6)
    self.assertFalse(result["is_success_left_fidelity"])
    self.assertFalse(result["is_success"])

  def test_tolerant_f1_ignores_stroke_weight_but_not_displacement(self):
    """The property the tolerant metric exists for.

    A stroke drawn one pixel thicker must not be punished; a stroke drawn in
    the wrong place must be. Raw IoU could not tell those apart, which is what
    made the old metric's difficulty track shape sparsity.
    """
    thin = np.zeros((64, 64), dtype=bool)
    thin[10:54, 30] = True
    thicker = np.zeros((64, 64), dtype=bool)
    thicker[10:54, 29:32] = True
    displaced = np.zeros((64, 64), dtype=bool)
    displaced[10:54, 50] = True

    self.assertGreater(
        visual_symmetry_eval.tolerant_f1(thicker, thin, k=2.0), 0.9
    )
    self.assertLess(
        visual_symmetry_eval.tolerant_f1(displaced, thin, k=2.0), 0.1
    )
    # Raw IoU cannot make that distinction: the thicker stroke scores poorly.
    self.assertLess(visual_symmetry_eval._mask_iou(thicker, thin), 0.5)

  def test_odd_width_output_is_scored_not_dropped(self):
    """Models return odd-width images; those must not raise inside the scorer.

    The shape mismatch this guards against was caught by a bare `except` one
    level up and recorded as "the model did not answer", so the probes vanished
    from the sample rather than showing up as an error.
    """
    gt, prompt, _ = visual_symmetry_image_gen.generate_symmetry_image(
        shape_type="circle", seed=3
    )
    odd = np.array(
        PIL_Image.fromarray(gt).resize((513, 513), PIL_Image.Resampling.BOX),
        dtype=np.uint8,
    )
    result = visual_symmetry_eval.score_symmetry_completion(
        output_image=odd, ground_truth_cells=gt, probe_image=prompt
    )
    self.assertGreater(result["iou_self_mirror"], 0.9)


class VisualSymmetryHypothesesTest(absltest.TestCase):
  """Tests for visual_symmetry_hypotheses and task definitions."""

  def test_overarching_task_definition(self):
    task = visual_symmetry_hypotheses.VISUAL_SYMMETRY_TASK
    self.assertIsInstance(task, data_classes.Task)
    self.assertEqual(task.suite, "visual_symmetry")
    self.assertEqual(task.name, "Visual Symmetry Completion")
    self.assertEqual(task.key, "visual_symmetry_completion")

  def test_hypothesis_count(self):
    hyps = visual_symmetry_hypotheses.get_visual_symmetry_hypotheses("all")
    self.assertLen(hyps, 2)

  def test_all_hypotheses_are_hypotheses(self):
    for hyp in visual_symmetry_hypotheses.VISUAL_SYMMETRY_HYPOTHESES:
      self.assertIsInstance(hyp, data_classes.Hypothesis)
      self.assertEqual(hyp.task_suite, "visual_symmetry")

  def test_all_hypotheses_have_required_condition_keys(self):
    required_keys = {
        "target_model",
        "shape_type",
        "aspect_ratio",
        "size",
        "bg_color",
    }
    for hyp in visual_symmetry_hypotheses.VISUAL_SYMMETRY_HYPOTHESES:
      missing = required_keys - set(hyp.condition.keys())
      self.assertEmpty(
          missing,
          msg=f"Hypothesis {hyp.name!r} missing condition keys: {missing}",
      )

  def test_all_hypotheses_target_model(self):
    for hyp in visual_symmetry_hypotheses.VISUAL_SYMMETRY_HYPOTHESES:
      self.assertEqual(
          hyp.condition["target_model"],
          "gemini-3.1-flash-image",
      )

  def test_null_proportion_is_090(self):
    """0.90, and it must agree with the task metadata.

    A mismatch between the two is not cosmetic: the verifier reads one and the
    agent is told the other, and the PLAN INCONSISTENCY guard then fails every
    hypothesis while the run still looks healthy.
    """
    task_null = visual_symmetry_hypotheses.VISUAL_SYMMETRY_TASK.metadata[
        "null_proportion"
    ]
    self.assertAlmostEqual(task_null, 0.90)
    for hyp in visual_symmetry_hypotheses.VISUAL_SYMMETRY_HYPOTHESES:
      self.assertAlmostEqual(
          hyp.null_proportion,
          0.90,
          msg=f"Hypothesis {hyp.name!r} null_proportion != 0.90",
      )

  def test_seed_instruction_states_the_same_null_rate(self):
    seed = visual_symmetry_hypotheses.DEFAULT_SEED_INSTRUCTION
    self.assertIn("90 percent as the null accuracy rate", seed)
    self.assertNotIn("75", seed)
    # Spell out "percent" rather than using a literal '%' character.
    self.assertNotIn("%", seed)

  def test_default_aspect_ratio_and_size(self):
    for hyp in visual_symmetry_hypotheses.VISUAL_SYMMETRY_HYPOTHESES:
      self.assertEqual(hyp.condition["aspect_ratio"], "1:1")
      self.assertEqual(hyp.condition["size"], 320)

  def test_get_by_key(self):
    hyps = visual_symmetry_hypotheses.get_visual_symmetry_hypotheses(
        "symmetry_english_letters_1to1"
    )
    self.assertLen(hyps, 1)
    self.assertEqual(hyps[0].name, "Symmetry English Letters 1to1")

  def test_get_unknown_raises(self):
    with self.assertRaises(ValueError):
      visual_symmetry_hypotheses.get_visual_symmetry_hypotheses(
          "nonexistent_xyz_999"
      )

  def test_hypothesis_claims(self):
    hyps = visual_symmetry_hypotheses.get_visual_symmetry_hypotheses("all")
    self.assertLen(hyps, 2)
    for h in hyps:
      self.assertIn("symmetry", h.claim.lower())

  def test_unique_hypothesis_keys(self):
    keys = [
        h.key for h in visual_symmetry_hypotheses.VISUAL_SYMMETRY_HYPOTHESES
    ]
    self.assertEqual(
        len(keys), len(set(keys)), msg="Duplicate hypothesis keys found"
    )

  def test_symmetry_prompt_exists(self):
    self.assertIsInstance(visual_symmetry_hypotheses.SYMMETRY_PROMPT, str)
    self.assertIn(
        "vertical", visual_symmetry_hypotheses.SYMMETRY_PROMPT.lower()
    )

  def test_shape_types_match_image_gen(self):
    valid_types = visual_symmetry_image_gen.list_shape_types()
    for hyp in visual_symmetry_hypotheses.VISUAL_SYMMETRY_HYPOTHESES:
      self.assertIn(
          hyp.condition["shape_type"],
          valid_types,
          msg=f"Hypothesis {hyp.name!r} has unknown shape_type",
      )


class IntegrationTest(absltest.TestCase):
  """End-to-end tests: generate → render → score."""

  def test_pipeline_roundtrip(self):
    for shape in [
        "english_letter",
        "greek_letter",
        "chinese_char",
        "square",
        "circle",
        "random_sparse",
        "random_dense",
        "stripes",
        "dots",
    ]:
      gt_cells, prompt_cells, gt_img = (
          visual_symmetry_image_gen.generate_symmetry_image(
              shape_type=shape,
              seed=7,
          )
      )
      result = visual_symmetry_eval.score_symmetry_completion(
          output_image=gt_img,
          ground_truth_cells=gt_cells,
          probe_image=prompt_cells,
      )
      self.assertTrue(
          result["is_success"],
          msg=f"Shape {shape} failed roundtrip scoring with ground-truth image",
      )


class SuccessCriterionSeparationTest(absltest.TestCase):
  """The three criteria must disagree, or separating them is pointless.

  Each case below is built so that the correct verdict under each criterion is
  known without measuring anything. If a change makes two criteria agree on all
  of them, the separation has silently collapsed back to one number.
  """

  _SIZE = 320
  _WHITE = np.array([255, 255, 255], dtype=np.uint8)
  _BLUE = np.array([0, 0, 255], dtype=np.uint8)

  def _blank(self) -> np.ndarray:
    return np.tile(self._WHITE, (self._SIZE, self._SIZE, 1)).astype(np.uint8)

  def _probe(self) -> np.ndarray:
    """Left half carries a deliberately asymmetric pattern; right half empty."""
    img = self._blank()
    img[80:160, 40:120] = self._BLUE
    img[180:210, 70:100] = self._BLUE
    return img

  def _perfect(self) -> np.ndarray:
    """The probe's left half mirrored into its right half."""
    img = self._probe()
    half = self._SIZE // 2
    img[:, self._SIZE - half :] = np.fliplr(img[:, :half])
    return img

  def _score(self, output: np.ndarray, criterion: str) -> dict[str, Any]:
    return visual_symmetry_eval.score_direct_symmetry_completion(
        output_image=output,
        ground_truth_image=self._perfect(),
        bg_color="white",
        probe_image=self._probe(),
        success_mode=visual_symmetry_eval.resolve_success_mode(criterion),
    )

  def test_perfect_completion_passes_every_criterion(self):
    for criterion in ("preservation", "symmetry", "reconstruction", "both"):
      with self.subTest(criterion=criterion):
        self.assertTrue(self._score(self._perfect(), criterion)["is_success"])

  def test_blank_canvas_passes_symmetry_but_fails_preservation(self):
    """The gameability hole that made the conjunction the default.

    A blank canvas is perfectly symmetric about its own axis, so `symmetry`
    alone must pass it. That is not a bug in `symmetry`; it is the reason
    `symmetry` must never be read without `preservation` alongside.
    """
    blank = self._blank()
    self.assertTrue(self._score(blank, "symmetry")["is_success"])
    self.assertFalse(self._score(blank, "preservation")["is_success"])
    self.assertFalse(self._score(blank, "both")["is_success"])
    self.assertFalse(self._score(blank, "reconstruction")["is_success"])

  def test_preserved_but_unreflected_passes_only_preservation(self):
    """Returning the probe unchanged: kept the left half, reflected nothing."""
    unchanged = self._probe()
    self.assertTrue(self._score(unchanged, "preservation")["is_success"])
    self.assertFalse(self._score(unchanged, "symmetry")["is_success"])
    self.assertFalse(self._score(unchanged, "both")["is_success"])

  def test_criteria_are_reported_even_when_not_selected(self):
    """Re-reading a run under a different criterion must not need re-querying."""
    result = self._score(self._perfect(), "symmetry")
    for key in ("iou_self_mirror", "left_fidelity", "iou_vs_ground_truth"):
      self.assertIsNotNone(result[key])

  def test_headline_iou_follows_the_selected_criterion(self):
    """`iou` is what callers threshold, so it must track the chosen criterion."""
    out = self._probe()
    self.assertAlmostEqual(
        float(self._score(out, "preservation")["iou"]),
        float(self._score(out, "preservation")["left_fidelity"]),
    )
    self.assertAlmostEqual(
        float(self._score(out, "reconstruction")["iou"]),
        float(self._score(out, "reconstruction")["iou_vs_ground_truth"]),
    )

  def test_unknown_criterion_raises_rather_than_defaulting(self):
    """A typo used to fall through the `else` and score the conjunction."""
    with self.assertRaises(ValueError):
      visual_symmetry_eval.resolve_success_mode("self_mirrror")
    with self.assertRaises(ValueError):
      visual_symmetry_eval.validate_success_mode("groundtruth")

  def test_agent_names_and_internal_names_resolve_to_the_same_mode(self):
    self.assertEqual(
        visual_symmetry_eval.resolve_success_mode("reconstruction"),
        visual_symmetry_eval.SUCCESS_MODE_GROUND_TRUTH,
    )
    self.assertEqual(
        visual_symmetry_eval.resolve_success_mode("both"),
        visual_symmetry_eval.SUCCESS_MODE_SELF_MIRROR,
    )
    self.assertEqual(
        visual_symmetry_eval.resolve_success_mode("ground_truth"),
        visual_symmetry_eval.SUCCESS_MODE_GROUND_TRUTH,
    )
    self.assertEqual(
        visual_symmetry_eval.resolve_success_mode(None),
        visual_symmetry_eval.DEFAULT_SUCCESS_MODE,
    )

  def test_preservation_without_a_probe_image_raises(self):
    """Otherwise it would report a 100 percent rate from zero evidence."""
    with self.assertRaises(ValueError):
      visual_symmetry_eval.score_direct_symmetry_completion(
          output_image=self._perfect(),
          ground_truth_image=self._perfect(),
          bg_color="white",
          probe_image=None,
          success_mode=visual_symmetry_eval.SUCCESS_MODE_PRESERVATION,
      )

  def test_batch_scorer_rejects_preservation_without_probe_images(self):
    """Batch-level guard: the per-probe handler would record missing data."""
    with self.assertRaises(ValueError):
      visual_symmetry_eval.score_symmetry_completions_pass_at_k(
          output_images_by_probe=[[self._perfect()]],
          ground_truth_cells_by_probe=[self._perfect()],
          probe_images_by_probe=None,
          success_mode=visual_symmetry_eval.SUCCESS_MODE_PRESERVATION,
      )


class SymmetryPromptConsistencyTest(absltest.TestCase):
  """Guards the invariants that let discovery and verification be compared.

  Every failure in this class corresponds to a bug that previously shipped and
  was silent: the numbers still came out, they were just measuring something
  other than what was claimed.
  """

  def test_prompt_shape_catalog_matches_the_probe_generator(self):
    """The catalog quoted to the agent must be the one that can be rendered.

    The old hand-maintained list named 9 of 22 families. An agent cannot form a
    hypothesis about a family it is told does not exist, so ten hard families
    -- the whole reason the null was raised to 0.90 -- were unreachable.
    """
    self.assertEqual(
        visual_symmetry_prompts.SUPPORTED_SHAPE_TYPES,
        visual_symmetry_image_gen.SUPPORTED_SHAPE_TYPES,
    )
    for shape in visual_symmetry_image_gen.HARD_SHAPE_TYPES:
      self.assertIn(shape, visual_symmetry_prompts.SUPPORTED_SHAPE_TYPES)

  def test_execution_prompt_names_every_renderable_shape(self):
    """The prompt text itself must enumerate the full catalog."""
    prompt = visual_symmetry_prompts.probe_execution_instructions(
        q_count=50, h_text="h"
    )
    for shape in visual_symmetry_image_gen.SUPPORTED_SHAPE_TYPES:
      self.assertIn(shape, prompt, f"{shape} missing from execution prompt")

  def test_prompts_mention_background_colour(self):
    """Background was the only discriminating factor found in the first run.

    It has to be reachable from the prompt, or a background hypothesis is
    verified on default white probes.
    """
    execution = visual_symmetry_prompts.probe_execution_instructions(
        q_count=50, h_text="h"
    )
    spec = visual_symmetry_prompts.task_specification(
        target_model="m", q_count=50, null_rate=0.9
    )
    for text in (execution, spec):
      self.assertIn("bg_color", text)
    for colour in visual_symmetry_image_gen.BACKGROUND_PALETTE:
      self.assertIn(colour, execution)

  def test_prompts_do_not_describe_the_retired_iou_metric(self):
    """Scoring is a tolerant-F1 conjunction, not a raw IoU against ground truth.

    Both prompts used to state the retired description verbatim, which tells
    the agent its failure claims will be judged on reconstruction when they are
    actually judged on symmetry AND preservation.
    """
    execution = visual_symmetry_prompts.probe_execution_instructions(
        q_count=50, h_text="h"
    )
    spec = visual_symmetry_prompts.task_specification(
        target_model="m", q_count=50, null_rate=0.9
    )
    for text in (execution, spec):
      self.assertNotIn("ink mask IoU", text)
      self.assertIn("tolerant F1", text)
      self.assertIn("PRESERVATION", text)
      self.assertIn("SYMMETRY", text)

  def test_hard_families_are_not_absorbed_by_easy_lookalikes(self):
    """A hard-family claim must not be probed with the easy family.

    `complex_chinese` contains the substring "chinese", so the old inference
    returned `chinese_char` -- verifying a hard-family failure claim against
    EASY probes, which converts a true finding into a false negative.
    """
    self.assertEqual(
        visual_symmetry_prompts.infer_shape_type(
            "The model fails on complex_chinese glyphs"
        ),
        "complex_chinese",
    )
    self.assertEqual(
        visual_symmetry_prompts.infer_shape_type(
            "high-stroke-count Chinese characters degrade accuracy"
        ),
        "complex_chinese",
    )
    self.assertEqual(
        visual_symmetry_prompts.infer_shape_type(
            "multi_glyph stacks are harder than single letters"
        ),
        "multi_glyph",
    )
    for shape in visual_symmetry_image_gen.HARD_SHAPE_TYPES:
      self.assertEqual(
          visual_symmetry_prompts.infer_shape_type(
              f"accuracy drops on {shape}"
          ),
          shape,
      )

  def test_inference_always_returns_a_renderable_shape(self):
    """The fallback must be renderable, including for unrecognisable text."""
    for text in ("", "something entirely unrelated", "the null is 90 percent"):
      self.assertIn(
          visual_symmetry_prompts.infer_shape_type(text),
          visual_symmetry_image_gen.SUPPORTED_SHAPE_TYPES,
      )

  def test_canonical_probe_text_is_single_sourced(self):
    """Every probe-sending path must use the same string object.

    Four byte-identical copies used to exist. They agreed by luck; nothing
    stopped one from being reworded. A verifier that words the task differently
    from discovery is re-testing a different task.
    """
    self.assertEqual(
        visual_symmetry_prompts.CANONICAL_PROBE_PROMPT,
        visual_symmetry_hypotheses.SYMMETRY_PROMPT,
    )
    self.assertEqual(
        standalone_agent.StandaloneAgent.SYMMETRY_PROMPT,
        visual_symmetry_hypotheses.SYMMETRY_PROMPT,
    )

  def test_canonical_probe_text_has_the_required_wording(self):
    """Pins the exact instruction the experiments are defined against."""
    self.assertEqual(
        visual_symmetry_hypotheses.SYMMETRY_PROMPT,
        "Reflect the pattern on the left half along the central vertical axis"
        " to complete the blank right half. Keep the existing colored pattern"
        " on the left without modification. The resulting image should be"
        " perfectly symmetric about the vertical center axis.",
    )

  def test_prompts_quote_the_canonical_probe_text(self):
    """The agent must be shown the exact prompt the target model receives."""
    execution = visual_symmetry_prompts.probe_execution_instructions(
        q_count=50, h_text="h"
    )
    spec = visual_symmetry_prompts.task_specification(
        target_model="m", q_count=50, null_rate=0.9
    )
    for text in (execution, spec):
      self.assertIn(visual_symmetry_prompts.CANONICAL_PROBE_PROMPT, text)
      self.assertIn("FIXED", text)

  def test_probe_text_is_not_an_agent_settable_tool_parameter(self):
    """`text_prompt` was advertised but never read by the handler.

    A declared-but-ignored parameter is worse than no parameter: the agent
    believes it varied the wording and reasons about the results on that false
    premise.
    """
    for decl in standalone_tools.TOOL_DECLARATIONS:
      if decl.name == "generate_and_send_symmetry_probes":
        self.assertIsNotNone(decl.parameters)
        properties = (
            (decl.parameters.properties or {}) if decl.parameters else {}
        )
        self.assertIn("shape_type", properties)
        self.assertNotIn("text_prompt", properties)
        break
    else:
      self.fail("generate_and_send_symmetry_probes declaration not found")

  def test_discovery_and_verification_prompts_agree(self):
    """The two phases may differ only in wording of role, never in substance."""
    discovery = visual_symmetry_prompts.probe_execution_instructions(
        q_count=50, h_text="h", verifying=False
    )
    verification = visual_symmetry_prompts.probe_execution_instructions(
        q_count=50, h_text="h", verifying=True
    )
    for shape in visual_symmetry_image_gen.SUPPORTED_SHAPE_TYPES:
      self.assertIn(shape, discovery)
      self.assertIn(shape, verification)
    self.assertIn(visual_symmetry_prompts.scoring_description(), discovery)
    self.assertIn(visual_symmetry_prompts.scoring_description(), verification)

  def test_no_seed_or_prompt_contains_a_literal_percent_sign(self):
    """Checks that task_specification contains at most one '%' sign."""
    spec = visual_symmetry_prompts.task_specification(
        target_model="m", q_count=50, null_rate=0.9
    )
    # The task specification legitimately renders the null as a percentage for
    # readability; assert it is the ONLY occurrence.
    self.assertLessEqual(spec.count("%"), 1)


if __name__ == "__main__":
  absltest.main()
