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

"""Evaluation utilities for visual symmetry completion tasks.

``score_direct_symmetry_completion`` reports three criteria, of which the
default success rule is the conjunction of the first two:

  1. ``iou_self_mirror``      the returned image against its own mirror image,
                              measured at the model's native resolution.
  2. ``left_fidelity``        the returned left half against the probe's left
                              half. Required: a blank canvas scores a perfect
                              self-mirror.
  3. ``iou_vs_ground_truth``  the returned right half against the ground
                              truth's. Reported, but does not decide success --
                              image models re-render the whole canvas, so this
                              charges them for global shift and rescale.

Agreement is a **tolerant F1** (``tolerant_f1``) with a 2px match tolerance at
probe resolution, not a raw mask IoU. Raw IoU charges the model for stroke
weight, which made the score track how thin the shape was rather than how well
it was mirrored: on the same run `dots` scored 0/100 and `dense_noise` 80/100.
The raw zero-tolerance numbers are still reported under the ``raw_*`` keys.

Ink is extracted relative to **each image's own background**: ``bg_color`` for
the probe and the ground truth, which we rendered, and an estimate from the
image itself for the model's output, which it re-rendered and often re-tinted.
"""

from typing import Any

import matplotlib
import matplotlib.colors
import numpy as np
from PIL import Image as _PIL_Image
from scipy import ndimage

# ── Constants ─────────────────────────────────────────────────────────────────

# Default agreement threshold for continuous symmetry completion.
# A completion attempt is marked as a Success if the tolerant F1 (see
# `tolerant_f1`) is >= 0.90.
DEFAULT_IOU_THRESHOLD: float = 0.90

# Minimum agreement between the model's left half and the probe's left half for
# the completion to count. This is the anti-gaming term: without it, a model
# could score a perfect self-mirror symmetry by returning a blank canvas, or by
# drawing any symmetric picture it likes while ignoring the given pattern.
DEFAULT_LEFT_FIDELITY_THRESHOLD: float = 0.80

# Match tolerance, in pixels at *probe* resolution (the 320px canvas we render),
# used by the tolerant-F1 agreement measure below.
#
# Why a tolerance at all. Raw mask IoU charges the model for stroke weight and
# sub-pixel placement. Image models do not edit the canvas they are given, they
# re-render it, and measurement on real runs showed their strokes carry
# 1.68-1.69x the ink of the ground truth (that ratio is stable across every
# resampling strategy tried, so it is the model's drawing style, not an
# artifact). On a small round dot, IoU is dominated by boundary pixels, so a
# correct mirror drawn slightly thicker scores near zero; on a dense texture the
# same thickening costs almost nothing. The metric's difficulty therefore
# tracked how thin the shape was rather than how well the model mirrored it --
# `dots` scored 0/100 while `dense_noise` scored 80/100 on the same run.
#
# Why 2. A sweep over k in {0, 1, 2, 3, 5, 8} on 100 real completions: at k=0
# (which is exactly the Dice coefficient) the median self-mirror score is 0.932;
# one pixel of slack takes it to 0.992, and the curve is then flat out to k=8.
# One pixel cannot rescue a model that mirrored the wrong thing, so that step is
# sub-pixel rasterisation error, not leniency. k=2 sits on the plateau rather
# than on its edge. Negative controls (scoring a completion against a
# *different* probe's left half) stay at median <= 0.53 with 0/100 above 0.80 at
# every k in the sweep, so the separation between a real completion and a wrong
# one is preserved. 2px is 0.6% of the 320px canvas width.
DEFAULT_MATCH_TOLERANCE_PX: float = 2.0

# Canvas size the probes are rendered at, used to scale the tolerance when a
# criterion is evaluated at the model's own (usually much larger) resolution.
REFERENCE_CANVAS_SIZE: int = 320

# Which criterion decides `is_success`.
#
# The three criteria answer different questions and do not agree, so which one
# is selected changes the measured rate substantially -- on dark backgrounds one
# run read self_mirror 90 / left_fidelity 70 / ground_truth 60. Four modes are
# offered: the three criteria individually, plus the conjunction that has been
# the default since the blank-canvas hole was closed.
#
#   "preservation"  - the returned image must have kept the left half it was
#                     given. Says nothing about whether anything was reflected.
#   "symmetry"      - the returned image must be internally symmetric about its
#                     own vertical centre. Measured within the returned image,
#                     so a global rescale does not count against it. On its own
#                     this is trivially satisfied by a blank canvas, which is
#                     why it is NOT the default -- use it when the hypothesis is
#                     specifically about symmetry and preservation is being
#                     reported alongside.
#   "ground_truth"  - "reconstruction": the model's right half against the
#                     ground truth right half. Penalises any global shift or
#                     rescale the model applies when it re-renders the canvas,
#                     so it measures registration as much as symmetry, and is
#                     the strictest of the three.
#   "self_mirror"   - the conjunction of "symmetry" AND "preservation". This is
#                     what the probe prompt actually asks for, and stays the
#                     default so existing runs remain comparable.
SUCCESS_MODE_GROUND_TRUTH: str = "ground_truth"
SUCCESS_MODE_SELF_MIRROR: str = "self_mirror"
SUCCESS_MODE_PRESERVATION: str = "preservation"
SUCCESS_MODE_SYMMETRY: str = "symmetry"
DEFAULT_SUCCESS_MODE: str = SUCCESS_MODE_SELF_MIRROR

# Validated rather than pattern-matched. Before the single-criterion modes
# existed the combine step was `if mode == ground_truth ... else conjunction`,
# so any unrecognised string -- including a typo, and including the new mode
# names if they reach an older binary -- silently scored the conjunction while
# reporting the requested mode back in `success_mode`. With the agent now
# choosing a criterion per hypothesis that failure would be invisible and would
# corrupt the rate, so an unknown mode raises instead.
SUCCESS_MODES: frozenset[str] = frozenset({
    SUCCESS_MODE_GROUND_TRUTH,
    SUCCESS_MODE_SELF_MIRROR,
    SUCCESS_MODE_PRESERVATION,
    SUCCESS_MODE_SYMMETRY,
})


def validate_success_mode(success_mode: str) -> str:
  """Returns `success_mode` unchanged, or raises if it is not a known mode.

  Args:
    success_mode: The requested success criterion.

  Returns:
    The validated mode.

  Raises:
    ValueError: If `success_mode` is not one of `SUCCESS_MODES`.
  """
  if success_mode not in SUCCESS_MODES:
    raise ValueError(
        f"Unknown success_mode {success_mode!r}. Expected one of:"
        f" {sorted(SUCCESS_MODES)}."
    )
  return success_mode


# The names the agent sees in the `score_symmetry_completions` tool schema are
# not the internal mode strings. Two of the internal names are actively
# misleading to a reader who has not read this file: `ground_truth` sounds like
# the authoritative measure when it is in fact the one we deliberately do not
# decide success on, and `self_mirror` names the conjunction after only one of
# its two terms. The agent is given `reconstruction` and `both` instead.
#
# Both vocabularies are accepted, so an internal caller passing
# `SUCCESS_MODE_GROUND_TRUTH` and an agent passing "reconstruction" land on the
# same mode rather than one of them silently falling back to the default.
AGENT_CRITERION_ALIASES: dict[str, str] = {
    "preservation": SUCCESS_MODE_PRESERVATION,
    "symmetry": SUCCESS_MODE_SYMMETRY,
    "reconstruction": SUCCESS_MODE_GROUND_TRUTH,
    "both": SUCCESS_MODE_SELF_MIRROR,
    # Conjunction synonyms the model is likely to emit unprompted.
    "conjunction": SUCCESS_MODE_SELF_MIRROR,
    "both_criteria": SUCCESS_MODE_SELF_MIRROR,
}


def resolve_success_mode(criterion: str | None) -> str:
  """Resolves an agent-facing criterion name to an internal success mode.

  Args:
    criterion: An agent-facing name (`preservation`, `symmetry`,
      `reconstruction`, `both`), an internal mode string, or None/empty for the
      default.

  Returns:
    A validated internal success mode.

  Raises:
    ValueError: If `criterion` is neither a known alias nor a known mode.
  """
  if not criterion:
    return DEFAULT_SUCCESS_MODE
  key = str(criterion).strip().lower()
  if key in AGENT_CRITERION_ALIASES:
    return AGENT_CRITERION_ALIASES[key]
  if key in SUCCESS_MODES:
    return key
  raise ValueError(
      f"Unknown success criterion {criterion!r}. Expected one of"
      f" {sorted(AGENT_CRITERION_ALIASES)} or {sorted(SUCCESS_MODES)}."
  )


# ── Color space conversion ────────────────────────────────────────────────────


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
  """Converts sRGB [0, 255] uint8 values to linear RGB [0, 1]."""
  c_norm = c.astype(np.float64) / 255.0
  return np.where(
      c_norm <= 0.04045,
      c_norm / 12.92,
      ((c_norm + 0.055) / 1.055) ** 2.4,
  )


def _linear_rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
  """Converts linear RGB to CIE XYZ (D65 illuminant).

  Args:
    rgb: Array of shape (..., 3) with linear RGB values in [0, 1].

  Returns:
    Array of same shape with XYZ values.
  """
  # sRGB to XYZ (D65) matrix.
  m = np.array([
      [0.4124564, 0.3575761, 0.1804375],
      [0.2126729, 0.7151522, 0.0721750],
      [0.0193339, 0.1191920, 0.9503041],
  ])
  return rgb @ m.T


def _xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
  """Converts CIE XYZ to CIELAB (D65 reference white).

  Args:
    xyz: Array of shape (..., 3) with XYZ values.

  Returns:
    Array of same shape with L*a*b* values.
  """
  # D65 reference white.
  ref = np.array([0.95047, 1.00000, 1.08883])
  xyz_n = xyz / ref

  delta = 6.0 / 29.0
  f = np.where(
      xyz_n > delta**3,
      np.cbrt(xyz_n),
      xyz_n / (3 * delta**2) + 4.0 / 29.0,
  )

  l_star = 116.0 * f[..., 1] - 16.0
  a_star = 500.0 * (f[..., 0] - f[..., 1])
  b_star = 200.0 * (f[..., 1] - f[..., 2])
  return np.stack([l_star, a_star, b_star], axis=-1)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
  """Converts sRGB uint8 image to CIELAB.

  Args:
    rgb: Array of shape (..., 3) with uint8 sRGB values.

  Returns:
    Array of same shape with float64 L*a*b* values.
  """
  linear = _srgb_to_linear(rgb)
  xyz = _linear_rgb_to_xyz(linear)
  return _xyz_to_lab(xyz)


def delta_e_cie76(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
  """Computes CIE76 ΔE (Euclidean distance in CIELAB).

  Args:
    lab1: Array of shape (..., 3) with L*a*b* values.
    lab2: Array of same shape with L*a*b* values.

  Returns:
    Array of ΔE values with the leading dimensions of the inputs.
  """
  diff = lab1 - lab2
  return np.sqrt(np.sum(diff**2, axis=-1))


def resolve_color_rgb(name: str) -> np.ndarray:
  """Resolves a colour name or hex string to a float64 (3,) RGB array.

  Uses the same matplotlib resolver the image generator uses, so the scorer and
  the generator can never disagree about what "lavender" means.

  Args:
    name: Any matplotlib-recognised colour name or "#rrggbb" string.

  Returns:
    (3,) float64 array of RGB values in [0, 255].
  """
  rgba = matplotlib.colors.to_rgba(name)
  return np.array([rgba[0] * 255.0, rgba[1] * 255.0, rgba[2] * 255.0])


def estimate_bg_rgb(
    image: np.ndarray,
    nominal_rgb: np.ndarray,
    ring_frac: float = 0.08,
) -> np.ndarray:
  """Estimates an image's background colour from its border.

  Needed because the model does not edit the canvas it is handed, it re-renders
  it, and it frequently shifts the background hue slightly (or substantially)
  while doing so. Extracting ink relative to the *nominal* background would then
  classify the whole canvas as ink and collapse every criterion to nonsense. On
  a white background the drift is invisible; the moment probes use coloured
  backgrounds it is not.

  Method: take the outer `ring_frac` frame, quantise to 16 levels per channel,
  and consider the two most common bins. The background is the majority colour
  of the frame for every shape family we render except `stripes`, where the
  frame is close to 50/50 -- hence taking two candidates and breaking the tie by
  proximity to the nominal background rather than trusting the mode outright.

  Args:
    image: (H, W, 3) uint8 image.
    nominal_rgb: The background the probe was rendered with, used as the
      tie-break and as the fallback.
    ring_frac: Width of the border frame, as a fraction of the smaller side.

  Returns:
    (3,) float64 RGB estimate.
  """
  h, w = image.shape[:2]
  band = max(1, int(round(min(h, w) * ring_frac)))
  if 2 * band >= min(h, w):
    ring = image.reshape(-1, 3)
  else:
    ring = np.concatenate([
        image[:band, :, :].reshape(-1, 3),
        image[-band:, :, :].reshape(-1, 3),
        image[band:-band, :band, :].reshape(-1, 3),
        image[band:-band, -band:, :].reshape(-1, 3),
    ])
  if ring.size == 0:
    return np.asarray(nominal_rgb, dtype=np.float64)

  quantized = ring.astype(np.int64) >> 4
  keys = quantized[:, 0] * 256 + quantized[:, 1] * 16 + quantized[:, 2]
  uniq, counts = np.unique(keys, return_counts=True)
  order = np.argsort(counts)[::-1][:2]

  best: np.ndarray | None = None
  best_dist = np.inf
  nominal = np.asarray(nominal_rgb, dtype=np.float64)
  for i in order:
    members = ring[keys == uniq[i]].astype(np.float64)
    candidate = members.mean(axis=0)
    dist = float(np.sqrt(((candidate - nominal) ** 2).sum()))
    if dist < best_dist:
      best_dist = dist
      best = candidate
  return nominal if best is None else best


def ink_mask(
    image: np.ndarray, bg_rgb: np.ndarray, tol: float = 30.0
) -> np.ndarray:
  """Boolean mask of pixels that differ from the background color.

  Public because the probe *generator* audits its glyphs with it. Scoring a
  model against an ink mask built one way while validating the probe with an
  ink mask built another way is how a defect hides: the audit passes on ink the
  scorer never sees. One definition, both sides.

  Args:
    image: (H, W, 3) uint8 RGB image.
    bg_rgb: (3,) background RGB colour in [0, 255].
    tol: Euclidean RGB distance threshold for classifying a pixel as ink.

  Returns:
    2D boolean array where True indicates ink.
  """
  dist = np.sqrt(np.sum((image.astype(np.float64) - bg_rgb) ** 2, axis=-1))
  return dist > tol


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
  """IoU of two boolean masks. Empty-vs-empty is 1.0, empty-vs-nonempty 0.0.

  Retained only so the raw, zero-tolerance numbers stay reportable alongside the
  tolerant ones. It no longer decides success; see `tolerant_f1`.

  Args:
    a: First boolean mask.
    b: Second boolean mask.

  Returns:
    Intersection-over-union in [0.0, 1.0].
  """
  union = int(np.logical_or(a, b).sum())
  if union == 0:
    return 1.0
  return float(np.logical_and(a, b).sum()) / float(union)


def distance_to_ink(mask: np.ndarray) -> np.ndarray:
  """Euclidean distance from every pixel to the nearest ink pixel in `mask`.

  Computing this once per mask answers every tolerance at once. The obvious
  alternative -- dilate by a disc of radius k -- is what this replaces: at the
  model's native 1024px resolution a 2px probe tolerance is a ~6px disc, and
  scipy's `binary_dilation` made a 100-probe sweep take longer than 15 minutes,
  versus 2 minutes with distance transforms.

  An all-background mask yields +inf everywhere, which makes every tolerance
  comparison against it correctly fail.

  Args:
    mask: Boolean ink mask.

  Returns:
    float64 array of the same shape.
  """
  if not mask.any():
    return np.full(mask.shape, np.inf)
  return ndimage.distance_transform_edt(~mask)


def tolerant_f1(
    a: np.ndarray,
    b: np.ndarray,
    k: float,
    dist_a: np.ndarray | None = None,
    dist_b: np.ndarray | None = None,
) -> float:
  """Agreement between two ink masks, counting a match within `k` pixels.

  Precision is the fraction of `a`'s ink lying within `k` pixels of some ink in
  `b`; recall is the fraction of `b`'s ink within `k` of some ink in `a`. The
  returned value is their F1, so the measure is symmetric in `a` and `b`.

  This is the standard boundary-F measure. It is indifferent to stroke weight
  and to sub-pixel placement -- the two things that made raw IoU's difficulty
  track shape sparsity rather than symmetry -- while still requiring the ink to
  be in the right place: `k` is 2px out of 320, so ink that is genuinely
  mis-mirrored is still scored as absent.

  At k=0 this reduces exactly to the Dice coefficient (not to IoU), so k=0
  numbers are NOT comparable to the historical IoU numbers even though both are
  "zero tolerance".

  Args:
    a: Boolean mask being scored.
    b: Boolean reference mask, same shape as `a`.
    k: Tolerance in pixels, in the resolution `a` and `b` share.
    dist_a: Optional precomputed `distance_to_ink(a)`.
    dist_b: Optional precomputed `distance_to_ink(b)`.

  Returns:
    F1 in [0, 1]. Empty-vs-empty is 1.0; empty-vs-nonempty is 0.0.
  """
  n_a, n_b = int(a.sum()), int(b.sum())
  if n_a == 0 and n_b == 0:
    return 1.0
  if n_a == 0 or n_b == 0:
    return 0.0
  if dist_a is None:
    dist_a = distance_to_ink(a)
  if dist_b is None:
    dist_b = distance_to_ink(b)
  precision = float((dist_b[a] <= k).sum()) / float(n_a)
  recall = float((dist_a[b] <= k).sum()) / float(n_b)
  if precision + recall == 0.0:
    return 0.0
  return 2.0 * precision * recall / (precision + recall)


def _resize_to(image: np.ndarray, height: int, width: int) -> np.ndarray:
  """Resizes to (height, width), area-averaging when downsampling.

  Image models return ~1024x1024 for a 320x320 probe. Nearest-neighbour
  downsampling of a 3.2x reduction drops most of a thin stroke, which by itself
  can push a correct completion below any threshold. BOX (area average)
  preserves stroke mass; BILINEAR is used when upsampling.

  Args:
    image: Input uint8 image array of shape (H, W, 3).
    height: Target height in pixels.
    width: Target width in pixels.

  Returns:
    Resized uint8 image array of shape (height, width, 3).
  """
  if image.shape[0] == height and image.shape[1] == width:
    return image
  shrinking = image.shape[0] >= height and image.shape[1] >= width
  resample = (
      _PIL_Image.Resampling.BOX if shrinking else _PIL_Image.Resampling.BILINEAR
  )
  pil = _PIL_Image.fromarray(image).resize((width, height), resample=resample)
  return np.array(pil, dtype=np.uint8)


def score_direct_symmetry_completion(
    output_image: np.ndarray,
    ground_truth_image: np.ndarray | None = None,
    bg_color: str = "white",
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    probe_image: np.ndarray | None = None,
    left_fidelity_threshold: float = DEFAULT_LEFT_FIDELITY_THRESHOLD,
    success_mode: str = DEFAULT_SUCCESS_MODE,
    match_tolerance_px: float = DEFAULT_MATCH_TOLERANCE_PX,
) -> dict[str, Any]:
  """Scores a symmetry completion using three independent criteria.

  The three criteria answer different questions, and they do **not** agree:

  1. `iou_self_mirror` - is the image the model returned internally symmetric
     about its own vertical centre axis? Computed entirely within the returned
     image, at its native resolution, so a global rescale does not affect it.
     This is literally what the probe prompt asks for.
  2. `left_fidelity` - did the model preserve the left half it was given?
     Computed against the probe image. Without this term, criterion 1 is
     trivially gamed by returning a blank canvas or any unrelated symmetric
     picture.
  3. `iou_vs_ground_truth` - the model's right half against the ground truth's
     right half. Because image models regenerate the whole canvas rather than
     editing it, this charges the model for global translation and scale error
     that has nothing to do with symmetry.

  All three are always reported. `is_success` is decided by `success_mode`.

  Two properties of the measurement are worth stating explicitly, because both
  were wrong in earlier versions of this function and both silently changed the
  answer rather than raising:

  * **Agreement is tolerant F1, not raw IoU.** See `tolerant_f1` and
    `DEFAULT_MATCH_TOLERANCE_PX`. Raw IoU made the score track stroke sparsity
    rather than symmetry. The raw numbers are still reported under the
    `raw_*` keys so the two can be compared on any run.
  * **Ink is taken relative to each image's own background.** The probe and the
    ground truth are ours, so their background is known exactly (`bg_color`).
    The model's output is not: it re-renders the canvas and routinely shifts the
    background. Its background is therefore estimated from the output itself
    (`estimate_bg_rgb`). Using the nominal colour for the output would classify
    the entire canvas as ink on any non-white background.

  Args:
    output_image: (H, W, 3) uint8 image produced by the model.
    ground_truth_image: (H, W, 3) uint8 ground-truth image (optional).
    bg_color: Background color name or hex the probe was rendered with.
    iou_threshold: Minimum agreement required for success. Default is 0.90.
    probe_image: (H, W, 3) uint8 image that was sent to the model. Required for
      `left_fidelity`; when absent, fidelity is reported as None and
      `success_mode="self_mirror"` falls back to the self-mirror term alone.
    left_fidelity_threshold: Minimum left-half agreement against the probe.
    success_mode: Which criterion decides `is_success`. One of
      `SUCCESS_MODE_PRESERVATION` (left half kept), `SUCCESS_MODE_SYMMETRY`
      (returned image internally symmetric), `SUCCESS_MODE_GROUND_TRUTH`
      (reconstruction: right half matches the intended reflection), or
      `SUCCESS_MODE_SELF_MIRROR` (the conjunction of the first two; default).
      All three criteria are reported regardless of which is selected.
    match_tolerance_px: Match tolerance in pixels at probe resolution.

  Returns:
    Dictionary containing `iou`, `iou_self_mirror`, `left_fidelity`,
    `iou_vs_ground_truth`, their `raw_*` zero-tolerance counterparts, the
    corresponding per-criterion booleans (`is_success_self_mirror`,
    `is_success_left_fidelity`, `is_success_ground_truth`), the selected
    `is_success`, and the legacy `precision` / `recall` / `accuracy` /
    `n_incorrect` / `mean_delta_e` keys.
  """
  nominal_bg_rgb = resolve_color_rgb(bg_color)
  # The model's canvas is its own; ours is known. Mixing the two is what breaks
  # on coloured backgrounds.
  out_bg_rgb = estimate_bg_rgb(output_image, nominal_bg_rgb)
  bg_drift = float(np.sqrt(((out_bg_rgb - nominal_bg_rgb) ** 2).sum()))

  # Everything except criterion 1 needs a common resolution. Prefer the probe/GT
  # geometry, which is also the resolution `match_tolerance_px` is defined in.
  reference = (
      ground_truth_image if ground_truth_image is not None else probe_image
  )
  if reference is not None:
    resized = _resize_to(output_image, reference.shape[0], reference.shape[1])
  else:
    resized = output_image
  height, width = resized.shape[:2]
  half = width // 2
  out_ink = ink_mask(resized, out_bg_rgb)
  d_out_left = distance_to_ink(out_ink[:, :half])

  # ---- Criterion 1: internal symmetry, at the model's native resolution. ----
  # Scale the tolerance so that "2 pixels" always means the same physical
  # fraction of the canvas regardless of what resolution the model returned.
  #
  # The right slice starts at `w - half`, not at `half`. They are the same when
  # the width is even, but models do return odd widths, and `ink[:, w // 2:]` is
  # then one column wider than `fliplr(ink[:, :w // 2])`. Comparing them raises
  # on the shape mismatch, and `_score_completions` catches every exception and
  # records the probe as missing data -- so odd-width responses were being
  # silently dropped rather than scored. Excluding the centre column (which is
  # the reflection axis itself, and belongs to neither side) makes both halves
  # exactly `half` columns wide.
  native_ink = ink_mask(output_image, out_bg_rgb)
  native_w = output_image.shape[1]
  native_half = native_w // 2
  native_scale = native_w / float(width) if width else 1.0
  mirror_right = native_ink[:, native_w - native_half :]
  mirror_left = np.fliplr(native_ink[:, :native_half])
  iou_self_mirror = tolerant_f1(
      mirror_right, mirror_left, match_tolerance_px * native_scale
  )
  raw_self_mirror = _mask_iou(mirror_right, mirror_left)

  # ---- Criterion 2: did the model keep the left half it was handed? ----
  left_fidelity: float | None = None
  raw_left_fidelity: float | None = None
  if probe_image is not None:
    probe_resized = _resize_to(probe_image, height, width)
    probe_ink = ink_mask(probe_resized, nominal_bg_rgb)
    left_fidelity = tolerant_f1(
        out_ink[:, :half],
        probe_ink[:, :half],
        match_tolerance_px,
        dist_a=d_out_left,
    )
    raw_left_fidelity = _mask_iou(out_ink[:, :half], probe_ink[:, :half])

  # ---- Criterion 3: comparison to the ground truth's right half. ----
  # `width - half` rather than `half`, for the odd-width reason above.
  right_ink = out_ink[:, width - half :]
  expected_right_ink = np.fliplr(out_ink[:, :half])
  if ground_truth_image is not None:
    gt_resized = _resize_to(ground_truth_image, height, width)
    expected_right_ink = ink_mask(gt_resized, nominal_bg_rgb)[:, width - half :]

  intersection = int(np.logical_and(right_ink, expected_right_ink).sum())
  union = int(np.logical_or(right_ink, expected_right_ink).sum())
  n_right = int(right_ink.sum())
  n_expected = int(expected_right_ink.sum())

  raw_iou_vs_gt = 0.0 if (union == 0 or n_right == 0) else intersection / union
  f1_vs_gt = tolerant_f1(right_ink, expected_right_ink, match_tolerance_px)
  precision = (
      float(intersection) / float(n_right)
      if n_right > 0
      else (1.0 if n_expected == 0 else 0.0)
  )
  recall = (
      float(intersection) / float(n_expected)
      if n_expected > 0
      else (1.0 if n_right == 0 else 0.0)
  )

  # ---- Combine. ----
  validate_success_mode(success_mode)
  ok_self_mirror = bool(iou_self_mirror >= iou_threshold)
  # `None` means no probe image was supplied, so preservation is unmeasurable.
  # Treating that as True is right for the conjunction (it degrades to the
  # self-mirror term alone, which is documented) but wrong for `preservation`,
  # where it would report a 100 percent rate from no evidence at all. That is
  # exactly the silent-success failure this separation is meant to expose, so it
  # raises.
  if left_fidelity is None and success_mode == SUCCESS_MODE_PRESERVATION:
    raise ValueError(
        "success_mode='preservation' requires `probe_image`: without it there"
        " is nothing to compare the returned left half against, and the"
        " criterion would pass unconditionally."
    )
  ok_left = (
      True
      if left_fidelity is None
      else bool(left_fidelity >= left_fidelity_threshold)
  )
  ok_gt = bool(f1_vs_gt >= iou_threshold)
  if success_mode == SUCCESS_MODE_GROUND_TRUTH:
    is_ok = ok_gt
  elif success_mode == SUCCESS_MODE_PRESERVATION:
    is_ok = ok_left
  elif success_mode == SUCCESS_MODE_SYMMETRY:
    is_ok = ok_self_mirror
  else:
    is_ok = ok_self_mirror and ok_left

  mean_de = 0.0
  if ground_truth_image is not None:
    gt_resized = _resize_to(ground_truth_image, height, width)
    per_pixel_de = delta_e_cie76(
        rgb_to_lab(gt_resized[:, half:]), rgb_to_lab(resized[:, half:])
    )
    mean_de = float(per_pixel_de.mean())

  n_total = int(right_ink.size)
  n_incorrect = int(np.sum(right_ink != expected_right_ink))

  # `iou` stays bound to whichever criterion drives `is_success`, so existing
  # callers that threshold `iou` remain self-consistent. `preservation` reports
  # left fidelity here; the `left_fidelity is None` case cannot reach this line
  # in that mode because it raised above.
  if success_mode == SUCCESS_MODE_GROUND_TRUTH:
    headline = f1_vs_gt
  elif success_mode == SUCCESS_MODE_PRESERVATION:
    # Unreachable: the combine step above raises when preservation is selected
    # without a probe image. Restated here so the type checker can see that
    # `headline` is never None, rather than defaulting it to 0.0 and turning a
    # programming error into a plausible-looking score of zero.
    if left_fidelity is None:
      raise ValueError("success_mode='preservation' requires `probe_image`.")
    headline = left_fidelity
  else:
    headline = iou_self_mirror

  return {
      "iou": float(headline),
      "iou_self_mirror": float(iou_self_mirror),
      "left_fidelity": None if left_fidelity is None else float(left_fidelity),
      "iou_vs_ground_truth": float(f1_vs_gt),
      # Zero-tolerance counterparts, reported so any run can be re-read under
      # the old measure without re-querying the target model.
      "raw_iou_self_mirror": float(raw_self_mirror),
      "raw_left_fidelity": (
          None if raw_left_fidelity is None else float(raw_left_fidelity)
      ),
      "raw_iou_vs_ground_truth": float(raw_iou_vs_gt),
      "is_success": is_ok,
      "is_success_self_mirror": ok_self_mirror,
      "is_success_left_fidelity": ok_left,
      "is_success_ground_truth": ok_gt,
      "success_mode": success_mode,
      "match_tolerance_px": float(match_tolerance_px),
      "bg_color": bg_color,
      "output_bg_rgb": [float(c) for c in out_bg_rgb],
      "output_bg_drift": bg_drift,
      "precision": float(precision),
      "recall": float(recall),
      "accuracy": float(headline),
      "right_accuracy": float(headline),
      "iou_threshold": float(iou_threshold),
      "left_fidelity_threshold": float(left_fidelity_threshold),
      "n_incorrect": n_incorrect,
      "n_total": n_total,
      "n_incorrect_right": n_incorrect,
      "n_right_total": n_total,
      "mean_delta_e": mean_de,
      "max_delta_e": mean_de,
  }


def score_symmetry_completion(
    output_image: np.ndarray,
    ground_truth_cells: np.ndarray,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    bg_color: str = "white",
    probe_image: np.ndarray | None = None,
    left_fidelity_threshold: float = DEFAULT_LEFT_FIDELITY_THRESHOLD,
    success_mode: str = DEFAULT_SUCCESS_MODE,
    match_tolerance_px: float = DEFAULT_MATCH_TOLERANCE_PX,
    **_unused_kwargs: Any,
) -> dict[str, Any]:
  """Scores a model's symmetry completion output.

  Evaluates via `score_direct_symmetry_completion`, which reports self-mirror
  agreement, left-half fidelity against the probe, and agreement against the
  ground truth, and picks between them using `success_mode`.

  Args:
    output_image: (H, W, 3) uint8 image produced by the model.
    ground_truth_cells: (H, W, 3) uint8 ground-truth image array.
    iou_threshold: Minimum agreement required for success (default 0.90).
    bg_color: Background color name.
    probe_image: (H, W, 3) uint8 image that was sent to the model. Required for
      the left-fidelity term.
    left_fidelity_threshold: Minimum left-half agreement against the probe.
    success_mode: Which criterion decides `is_success`.
    match_tolerance_px: Match tolerance in pixels at probe resolution; see
      `tolerant_f1`.
    **_unused_kwargs: Ignored legacy keyword arguments.

  Returns:
    Dictionary of evaluation metrics including 'is_success', 'accuracy', etc.
  """
  del _unused_kwargs
  return score_direct_symmetry_completion(
      output_image=output_image,
      ground_truth_image=ground_truth_cells,
      bg_color=bg_color,
      iou_threshold=iou_threshold,
      probe_image=probe_image,
      left_fidelity_threshold=left_fidelity_threshold,
      success_mode=success_mode,
      match_tolerance_px=match_tolerance_px,
  )


def score_symmetry_completions_pass_at_k(
    output_images_by_probe: list[list[np.ndarray]],
    ground_truth_cells_by_probe: list[np.ndarray],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    bg_color: str = "white",
    probe_images_by_probe: list[np.ndarray] | None = None,
    left_fidelity_threshold: float = DEFAULT_LEFT_FIDELITY_THRESHOLD,
    success_mode: str = DEFAULT_SUCCESS_MODE,
    match_tolerance_px: float = DEFAULT_MATCH_TOLERANCE_PX,
    **_unused_kwargs: Any,
) -> dict[str, Any]:
  """Computes pass@k metrics across multiple candidate samples per input probe.

  Success is decided by `success_mode`, one of:

    `preservation`  the returned image kept the probe's left half
                    (agreement >= `left_fidelity_threshold`). Requires
                    `probe_images_by_probe`.
    `symmetry`      the returned image is symmetric about its own centre axis
                    (agreement >= `iou_threshold`).
    `ground_truth`  reconstruction: its right half matches the ground truth's
                    right half (agreement >= `iou_threshold`).
    `self_mirror`   the conjunction of `symmetry` and `preservation`. Default.

  All three underlying numbers are reported per sample whichever mode is
  selected, so a run can be re-read under a different criterion without
  re-querying the target model.

  An input probe is considered a pass@k success if AT LEAST ONE of its sampled
  completions succeeded.

  A probe for which the target model returned no completion at all is treated
  as missing data rather than as a failure: it is reported via
  `n_probes_invalid` and excluded from `n_probes_valid`, `pass_at_k_rate` and
  `k`. Callers must use `n_probes_valid` (not `n_probes`) as the denominator of
  any statistical test.

  Args:
    output_images_by_probe: List of sampled completion image lists per probe.
    ground_truth_cells_by_probe: List of ground-truth images per probe.
    iou_threshold: Minimum agreement threshold for symmetry/reconstruction.
    bg_color: Nominal background colour name or hex string.
    probe_images_by_probe: Optional list of input probe images per probe.
    left_fidelity_threshold: Minimum agreement threshold for left-half
      preservation.
    success_mode: Criterion deciding `is_success`.
    match_tolerance_px: Match tolerance in pixels at reference resolution.
    **_unused_kwargs: Ignored legacy keyword arguments.

  Returns:
    Dictionary of aggregate pass@k metrics and per-probe results.
  """
  del _unused_kwargs
  # Validate once, here, rather than relying on the per-probe call to raise.
  # `_score_completions` wraps each probe in a try/except and records anything
  # that throws as missing data, so a bad mode would otherwise be reported as
  # "the target model returned nothing" for every probe -- a wrong answer that
  # looks like an infrastructure problem.
  validate_success_mode(success_mode)
  if success_mode == SUCCESS_MODE_PRESERVATION and not probe_images_by_probe:
    raise ValueError(
        "success_mode='preservation' requires `probe_images_by_probe`:"
        " preservation is measured against the probe that was sent, and"
        " without it every probe would pass unconditionally."
    )
  n_probes = min(len(output_images_by_probe), len(ground_truth_cells_by_probe))
  if n_probes == 0:
    return {
        "n_probes": 0,
        "n_probes_valid": 0,
        "n_probes_invalid": 0,
        "k": 0,
        "n_probes_passed": 0,
        "pass_at_k_rate": 0.0,
        "total_samples": 0,
        "total_samples_passed": 0,
        "pass_at_1_rate": 0.0,
        "per_probe_results": [],
    }

  n_probes_passed = 0
  n_probes_valid = 0
  total_samples = 0
  total_samples_passed = 0
  per_probe_results: list[dict[str, Any]] = []

  for probe_idx in range(n_probes):
    samples = output_images_by_probe[probe_idx]
    gt = ground_truth_cells_by_probe[probe_idx]
    probe_img = None
    if probe_images_by_probe is not None and probe_idx < len(
        probe_images_by_probe
    ):
      probe_img = probe_images_by_probe[probe_idx]
    sample_scores: list[dict[str, Any]] = []
    probe_sample_successes = 0

    for sample_idx, img in enumerate(samples):
      score = score_symmetry_completion(
          output_image=img,
          ground_truth_cells=gt,
          iou_threshold=iou_threshold,
          bg_color=bg_color,
          probe_image=probe_img,
          left_fidelity_threshold=left_fidelity_threshold,
          success_mode=success_mode,
          match_tolerance_px=match_tolerance_px,
      )
      is_ok = bool(score["is_success"])

      if is_ok:
        probe_sample_successes += 1
        total_samples_passed += 1
      total_samples += 1
      sample_scores.append({
          "sample_idx": sample_idx,
          "success": is_ok,
          "iou": score.get("iou", 0.0),
          # All three criteria are recorded on every sample so the choice of
          # success criterion can be re-evaluated after the fact without
          # re-querying the target model.
          "iou_self_mirror": score.get("iou_self_mirror"),
          "left_fidelity": score.get("left_fidelity"),
          "iou_vs_ground_truth": score.get("iou_vs_ground_truth"),
          "raw_iou_self_mirror": score.get("raw_iou_self_mirror"),
          "raw_left_fidelity": score.get("raw_left_fidelity"),
          "raw_iou_vs_ground_truth": score.get("raw_iou_vs_ground_truth"),
          "output_bg_drift": score.get("output_bg_drift"),
          "success_self_mirror": score.get("is_success_self_mirror"),
          "success_left_fidelity": score.get("is_success_left_fidelity"),
          "success_ground_truth": score.get("is_success_ground_truth"),
          "n_incorrect": score["n_incorrect"],
          "n_incorrect_right": score["n_incorrect_right"],
          "accuracy": score["accuracy"],
          "mean_delta_e": score["mean_delta_e"],
      })

    # A probe with no returned completion is *missing data*, not a failed
    # completion: the target model gave us nothing to score. Counting it as a
    # failure would silently manufacture evidence for hypotheses of the form
    # "the model's accuracy is below X", because every such hypothesis is
    # tested with alternative='smaller'. Invalid probes are therefore excluded
    # from both the numerator and the denominator, and reported separately.
    probe_valid = bool(samples)
    probe_passed = probe_sample_successes > 0
    if probe_valid:
      n_probes_valid += 1
      if probe_passed:
        n_probes_passed += 1

    per_probe_results.append({
        "probe_idx": probe_idx,
        "valid": probe_valid,
        "passed": probe_passed,
        "n_samples": len(samples),
        "n_samples_passed": probe_sample_successes,
        "sample_scores": sample_scores,
    })

  n_probes_invalid = n_probes - n_probes_valid
  pass_at_k_rate = (
      n_probes_passed / n_probes_valid if n_probes_valid > 0 else 0.0
  )
  pass_at_1_rate = (
      total_samples_passed / total_samples if total_samples > 0 else 0.0
  )
  k_val = total_samples // n_probes_valid if n_probes_valid > 0 else 0

  # Per-criterion tallies over every scored sample. These let the choice of
  # success criterion be audited after the fact: if `criterion_agreement` is
  # well below 1.0, the criteria are measuring materially different things and
  # the headline number depends on which one was selected.
  all_samples: list[dict[str, Any]] = []
  for p in per_probe_results:
    all_samples.extend(p["sample_scores"])

  def _rate(key: str) -> float | None:
    vals = [s[key] for s in all_samples if s.get(key) is not None]
    return (sum(1 for v in vals if v) / len(vals)) if vals else None

  agreement = None
  comparable = [
      s
      for s in all_samples
      if s.get("success_self_mirror") is not None
      and s.get("success_ground_truth") is not None
  ]
  if comparable:
    both = sum(
        1
        for s in comparable
        if bool(s["success_self_mirror"] and s.get("success_left_fidelity"))
        == bool(s["success_ground_truth"])
    )
    agreement = both / len(comparable)

  return {
      "n_probes": n_probes,
      "n_probes_valid": n_probes_valid,
      "n_probes_invalid": n_probes_invalid,
      "k": k_val,
      "n_probes_passed": n_probes_passed,
      "pass_at_k_rate": pass_at_k_rate,
      "total_samples": total_samples,
      "total_samples_passed": total_samples_passed,
      "pass_at_1_rate": pass_at_1_rate,
      "success_mode": success_mode,
      "match_tolerance_px": float(match_tolerance_px),
      "rate_self_mirror": _rate("success_self_mirror"),
      "rate_left_fidelity": _rate("success_left_fidelity"),
      "rate_ground_truth": _rate("success_ground_truth"),
      "criterion_agreement": agreement,
      "per_probe_results": per_probe_results,
  }
