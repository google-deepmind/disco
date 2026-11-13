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

"""Image generation utilities for visual symmetry completion tasks.

Generates continuous symmetric patterns and produces two images:
  1. A ground-truth fully symmetric image with exact horizontal reflection.
  2. A prompt image with only the left half filled (right half is empty
     background).

Base pattern types (see ``BASE_SHAPE_TYPES``):
  - Symmetric English letters (A, H, I, M, O, T, U, V, X, Y)
  - Symmetric Greek letters (Α, Δ, Θ, Λ, Ξ, Π, Τ, Υ, Φ, Ψ, Ω)
  - Symmetric Chinese characters (中, 十, 王, 田, 人, 木, 土, 丰, 工, 口, 日, 三,
    干, 甲, 由, 申, 早, 回, 曲)
  - Concentric or centered squares, circles, triangles, plus signs, hearts
  - Horizontal stripes
  - Discrete dots and dot clusters
  - Stochastic texture lattices (random_sparse, random_dense)

Harder pattern types (see ``HARD_SHAPE_TYPES``). These exist because the target
model clears 92-98% on every base family under the corrected metric, which makes
the base set degenerate as a discovery benchmark. They are harder along the two
axes that actually cost the model accuracy -- stroke fineness and part count --
rather than by being larger or more colourful:
  - ``complex_chinese``  high-stroke-count symmetric Hanzi (囍, 業, 晶, 品, 黃, ...)
  - ``multi_glyph``      three small glyphs stacked vertically
  - ``maze``             thin vertical bars with horizontal connectors
  - ``snowflake``        radial arms with sub-branches
  - ``filigree``         scattered thin arcs and ticks
  - ``nested_outlines``  concentric outlines, not filled
  - ``starburst``        many thin rays of differing lengths
  - ``scatter_rings``    small hollow rings of varying radius
  - ``wave_lines``       stacked sinusoids, independent
  amplitude/frequency/phase
  - ``fine_noise``       stochastic lattice at half the cell size of random_*

Backgrounds: ``BACKGROUND_PALETTE`` lists the supported canvas colours. There is
deliberately no per-probe random background -- the scorer must know the exact
colour the probe was drawn with.

Symmetry guarantee: the right half of the ground truth is *replaced* by the
mirror of the left half, and ``assert_ground_truth_symmetric`` then verifies it.
The verification is not redundant: every score downstream assumes the target is
symmetric, so a rendering bug that broke the assumption would be attributed to
the model rather than to us.

All images are rendered at native canvas resolution (default 320x320 pixels)
and returned as uint8 NumPy arrays with shape ``(canvas_size, canvas_size, 3)``.
"""

import os
import random as _random
from typing import Any

import matplotlib
import matplotlib.colors
import matplotlib.font_manager as fm
import numpy as np
from PIL import Image as _PIL_Image
from PIL import ImageDraw as _PIL_ImageDraw
from PIL import ImageFont as _PIL_ImageFont

from disco.tasks.visual_symmetry import eval as visual_symmetry_eval

matplotlib.use("Agg")  # Non-interactive backend.


# ── Constants ─────────────────────────────────────────────────────────────────

# Canvas dimension (pixels) for direct continuous images.
DIRECT_CANVAS_SIZE: int = 320

# Candidate font paths for rendering all character types (English, Greek, CJK).
_FONT_CANDIDATES: list[str] = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
]


def _find_font() -> str | None:
  """Locates a font file for character rendering, returning its path or None."""
  env_font = os.environ.get("DISCO_FONT_PATH", "").strip()
  if env_font and os.path.exists(env_font):
    return env_font
  for path in _FONT_CANDIDATES:
    if os.path.exists(path):
      return path
  # Ask matplotlib's font_manager for CJK or sans-serif fonts.
  for family in (
      "Noto Sans CJK JP",
      "Droid Sans Fallback",
      "DejaVu Sans",
      "sans-serif",
  ):
    try:
      prop = fm.FontProperties(family=family)
      fallback = fm.findfont(prop, fallback_to_default=False)
      if fallback and os.path.exists(fallback):
        return fallback
    except (ValueError, RuntimeError, OSError):
      pass
  return None


_FONT_PATH: str | None = _find_font()
FONT_PATH: str | None = _FONT_PATH

# A correctly rendered glyph at font_size = 0.72 * canvas covers a few percent
# of the canvas at minimum. PIL's built-in bitmap fallback font ignores the
# requested size and draws a ~4x7 pixel box, which is ~0.02% of a 320x320
# canvas. Anything below this fraction means the glyph did not really render,
# which would silently turn the probe into a blank image and make the target
# model look incapable. Fail loudly instead.
_MIN_GLYPH_INK_FRACTION: float = 0.005
MIN_GLYPH_INK_FRACTION: float = _MIN_GLYPH_INK_FRACTION


# Vertically symmetric English letters (symmetric about the vertical axis).
#
# W is deliberately absent: standard fonts do not draw its middle vertex
# as a mirror image, and it measures worse against its own reflection
# than the letters listed here.
SYMMETRIC_ENGLISH_LETTERS: list[str] = [
    "A",
    "H",
    "I",
    "M",
    "O",
    "T",
    "U",
    "V",
    "X",
    "Y",
]

# Vertically symmetric Greek letters.
SYMMETRIC_GREEK_LETTERS: list[str] = [
    "\u0391",  # Α (Alpha)
    "\u0394",  # Δ (Delta)
    "\u0398",  # Θ (Theta)
    "\u039b",  # Λ (Lambda)
    "\u039e",  # Ξ (Xi)
    "\u03a0",  # Π (Pi)
    "\u03a4",  # Τ (Tau)
    "\u03a5",  # Υ (Upsilon)
    "\u03a6",  # Φ (Phi)
    "\u03a8",  # Ψ (Psi)
    "\u03a9",  # Ω (Omega)
]

# Vertically symmetric Chinese characters (all <= 7 strokes).
#
# Every entry is verified by `glyph_self_mirror_f1`, which
# renders the glyph and scores it against its own reflection with the
# production scorer. Do not add a character on the strength of "it looks
# symmetric" -- 天, 大, 八, 山, 丫 and 目 all read as symmetric and all failed,
# because 撇 (falling-left) and 捺 (falling-right) are different strokes that
# the font does not draw as mirror images.
SYMMETRIC_CHINESE_CHARS: list[str] = [
    "\u4e2d",  # 中 (4 strokes)
    "\u5341",  # 十 (2 strokes)
    "\u738b",  # 王 (4 strokes)
    "\u7530",  # 田 (5 strokes)
    "\u4eba",  # 人 (2 strokes)
    "\u6728",  # 木 (4 strokes)
    "\u571f",  # 土 (3 strokes)
    "\u4e30",  # 丰 (4 strokes)
    "\u5de5",  # 工 (3 strokes)
    "\u53e3",  # 口 (3 strokes)
    "\u65e5",  # 日 (4 strokes)
    "\u4e09",  # 三 (3 strokes)
    "\u5e72",  # 干 (3 strokes)
    "\u7532",  # 甲 (5 strokes)
    "\u7531",  # 由 (5 strokes)
    "\u7533",  # 申 (5 strokes)
    "\u65e9",  # 早 (6 strokes)
    "\u56de",  # 回 (6 strokes)
    "\u66f2",  # 曲 (6 strokes)
]

# High-stroke-count Chinese characters that are still mirror symmetric about a
# vertical axis. These are deliberately much harder than SYMMETRIC_CHINESE_CHARS
# (which is capped at 7 strokes): the interior structure is dense enough that
# getting the reflection right requires reproducing stroke topology, not just
# the silhouette.
#
# Every entry must be symmetric *in the rendering font*, not merely "symmetric
# in principle", and that is enforced by `assert_source_glyph_symmetric` rather
# than by judgement. Judgement is not good enough here: 磊, 燕, 樂, 意, 鑫, 森, 高
# and 器 were all originally listed as symmetric and all failed, because of an
# asymmetric component (石, 北, 白, 心, 犬) that is easy to overlook.
#
# The replacements were chosen to hold the mean stroke count constant at 13.4,
# so that dropping the failures did not quietly make this family easier and
# turn "the model struggles with complex glyphs" into an artefact of the fix.
COMPLEX_SYMMETRIC_CHINESE_CHARS: list[str] = [
    "\u56cd",  # 囍 (12 strokes)
    "\u696d",  # 業 (13 strokes)
    "\u6676",  # 晶 (12 strokes)
    "\u54c1",  # 品 (9 strokes)
    "\u9ec3",  # 黃 (12 strokes)
    "\u7ae5",  # 童 (12 strokes)
    "\u58f9",  # 壹 (12 strokes)
    "\u7ae0",  # 章 (11 strokes)
    "\u9748",  # 靈 (24 strokes)
    "\u8c50",  # 豐 (18 strokes)
    "\u5669",  # 噩 (16 strokes)
    "\u8cde",  # 賞 (15 strokes)
    "\u96f7",  # 雷 (13 strokes)
    "\u55ae",  # 單 (12 strokes)
    "\u8983",  # 覃 (12 strokes)
    "\u7c9f",  # 粟 (12 strokes)
]

# Named colors for patterns.
_PALETTE: list[str] = [
    "red",
    "blue",
    "green",
    "orange",
    "purple",
    "cyan",
    "magenta",
    "brown",
    "olive",
    "teal",
]

# Background colours the probes may be rendered on, beyond plain white.
#
# There is deliberately no "random" option. The background has to be a named,
# fixed property of a condition: the scorer needs to know the exact colour the
# probe was drawn with in order to extract ink from the probe and the ground
# truth, and a per-probe random background would mean the scorer is handed a
# colour that is wrong for all but one probe in the batch.
BACKGROUND_PALETTE: list[str] = [
    "white",
    "black",
    "lightgray",
    "lightyellow",
    "lightblue",
    "mistyrose",
    "honeydew",
    "lavender",
    "navy",
    "darkslategray",
]

# Minimum RGB Euclidean distance between foreground and background.
#
# The ink mask counts a pixel as ink when it is more than 30.0 from the
# background, so a foreground closer than that is invisible to the scorer and
# the probe silently becomes a blank canvas. The margin here is far wider than
# 30 because the target model re-renders the canvas and shifts colours while
# doing so; a pair that only just clears the mask threshold on our rendering can
# fall under it on the model's.
_MIN_FG_BG_CONTRAST: float = 140.0


def _rgb_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
  """Euclidean distance between two RGB triples."""
  return float(np.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))))


def _is_light(rgb: tuple[int, int, int]) -> bool:
  """True if `rgb` is a light colour by Rec. 601 luma."""
  return (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) > 140.0


def _pick_fg_rgb(
    rng: _random.Random,
    bg_rgb: tuple[int, int, int],
    requested: str | None,
    default_for_glyphs: bool = False,
) -> tuple[int, int, int]:
  """Chooses a foreground colour that is actually visible on `bg_rgb`.

  Args:
    rng: Seeded RNG, so the choice is reproducible.
    bg_rgb: The background the shape will be drawn on.
    requested: An explicit colour asked for by the caller, or None.
    default_for_glyphs: If True and no colour was requested, use plain
      black-or-white (whichever contrasts) rather than a palette colour, which
      is what character probes have always looked like.

  Returns:
    An (R, G, B) tuple.

  Raises:
    ValueError: If `requested` is too close to the background to be seen. This
      is raised rather than silently substituted: a substituted colour would
      mean the condition actually run is not the condition that was asked for,
      and every number downstream would be attributed to the wrong condition.
  """
  if requested:
    fg = _color_name_to_rgb(requested)
    dist = _rgb_distance(fg, bg_rgb)
    if dist < _MIN_FG_BG_CONTRAST:
      usable = [
          name
          for name in _PALETTE + ["black", "white"]
          if _rgb_distance(_color_name_to_rgb(name), bg_rgb)
          >= _MIN_FG_BG_CONTRAST
      ]
      raise ValueError(
          f"fg_color={requested!r} (RGB {fg}) is only {dist:.0f} from the "
          f"background (RGB {bg_rgb}), below the {_MIN_FG_BG_CONTRAST} minimum "
          "contrast. The pattern would be invisible to the scorer's ink mask "
          "and the probe would silently measure a blank canvas. Usable "
          f"foreground colours on this background: {usable}."
      )
    return fg

  if default_for_glyphs:
    return (0, 0, 0) if _is_light(bg_rgb) else (255, 255, 255)

  candidates = [
      name
      for name in _PALETTE
      if _rgb_distance(_color_name_to_rgb(name), bg_rgb) >= _MIN_FG_BG_CONTRAST
  ]
  if not candidates:
    return (0, 0, 0) if _is_light(bg_rgb) else (255, 255, 255)
  return _color_name_to_rgb(rng.choice(candidates))


def _midline_color(bg_rgb: tuple[int, int, int]) -> tuple[int, int, int]:
  """A subtle but visible axis guide colour for the given background.

  The guide used to be hard-coded to a pale blue-grey, which is invisible on a
  light background and nearly invisible on several others. The probe would then
  be missing the axis hint the prompt implicitly relies on, for some conditions
  but not others.

  Args:
    bg_rgb: Background (R, G, B) uint8 tuple.

  Returns:
    (R, G, B) uint8 tuple for the midline axis guide.
  """
  if _is_light(bg_rgb):
    return (
        int(bg_rgb[0] * 0.62),
        int(bg_rgb[1] * 0.62),
        int(bg_rgb[2] * 0.62),
    )
  return (
      int(bg_rgb[0] + (255 - bg_rgb[0]) * 0.45),
      int(bg_rgb[1] + (255 - bg_rgb[1]) * 0.45),
      int(bg_rgb[2] + (255 - bg_rgb[2]) * 0.45),
  )


# ── Color helpers ─────────────────────────────────────────────────────────────


def _color_name_to_rgb(name: str) -> tuple[int, int, int]:
  """Converts a matplotlib color name to an (R, G, B) uint8 tuple."""
  rgba = matplotlib.colors.to_rgba(name)
  return (int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))


def _bg_color_rgb(bg_color: str) -> tuple[int, int, int]:
  return _color_name_to_rgb(bg_color)


# Shape types that were in the original probe set. Gemini clears 92-98% on
# essentially all of them under the corrected metric, so on their own they make
# a degenerate discovery benchmark: every "success" hypothesis verifies and no
# "failure" hypothesis can.
BASE_SHAPE_TYPES: list[str] = [
    "chinese_char",
    "circle",
    "dots",
    "english_letter",
    "greek_letter",
    "heart",
    "plus",
    "random_dense",
    "random_sparse",
    "square",
    "stripes",
    "triangle",
]

# Deliberately harder families: finer strokes, more parts, and more internal
# structure to keep registered across the reflection axis.
HARD_SHAPE_TYPES: list[str] = [
    "complex_chinese",
    "fine_noise",
    "filigree",
    "maze",
    "multi_glyph",
    "nested_outlines",
    "scatter_rings",
    "snowflake",
    "starburst",
    "wave_lines",
]

# Supported shape types for direct continuous image generation.
SUPPORTED_SHAPE_TYPES: list[str] = sorted(
    BASE_SHAPE_TYPES + HARD_SHAPE_TYPES + ["dynamic"]
)


def list_shape_types() -> list[str]:
  """Returns the list of supported shape type names."""
  return list(SUPPORTED_SHAPE_TYPES)


def _assert_glyph_rendered(
    img: Any,
    bg_rgb: tuple[int, ...],
    shape_type: str,
    char: str,
    font_file: str,
) -> None:
  """Raises if a drawn character covers implausibly little of the canvas.

  A font that loads successfully but lacks a glyph for `char` renders either
  nothing or a tiny `.notdef` box. Either way the probe becomes a blank canvas,
  and the downstream symmetry score then measures nothing at all while looking
  like a legitimate model failure.

  Args:
    img: The PIL image that was just drawn on.
    bg_rgb: Background color the canvas was filled with.
    shape_type: Shape type being rendered, for the error message.
    char: The character that was drawn, for the error message.
    font_file: Font that was used, for the error message.

  Raises:
    RuntimeError: If the glyph covers less than `_MIN_GLYPH_INK_FRACTION`.
  """
  arr = np.array(img, dtype=np.float64)
  dist = np.sqrt(np.sum((arr - np.array(bg_rgb, dtype=np.float64)) ** 2, -1))
  ink_fraction = float((dist > 30.0).mean())
  if ink_fraction < _MIN_GLYPH_INK_FRACTION:
    raise RuntimeError(
        f"Rendering shape_type={shape_type!r} char={char!r} with font"
        f" {font_file!r} produced only {ink_fraction:.5f} ink coverage, below"
        f" the {_MIN_GLYPH_INK_FRACTION} minimum. The font almost certainly has"
        " no glyph for this character, which would yield a blank probe image."
        " Set DISCO_FONT_PATH or install a CJK-capable font (e.g. Noto Sans"
        " CJK)."
    )


assert_glyph_rendered = _assert_glyph_rendered


def draw_glyphs_centered(
    img: "_PIL_Image.Image",
    char: str,
    fg_rgb: tuple[int, int, int],
    font_path: str | None,
    stacked: bool = False,
    shape_type: str = "",
) -> str:
  """Draws `char` centred on the canvas's vertical axis, in place.

  Shared by the probe generator and the glyph audit so the thing audited is
  the thing rendered. Auditing a reimplementation of this would verify the
  reimplementation.

  Args:
    img: RGB canvas to draw on, modified in place.
    char: One character, or several when `stacked` is set.
    fg_rgb: Ink colour.
    font_path: TrueType font file. None raises; see below.
    stacked: Draw each character on its own row instead of one large glyph.
    shape_type: Name used in the error message only.

  Returns:
    The resolved font path. Returned rather than discarded so callers get a
    non-optional value to pass on without re-checking for None.

  Raises:
    RuntimeError: If no usable font was found.
  """
  size = img.size[0]
  if font_path is None:
    raise RuntimeError(
        "No usable character font was found, so "
        f"shape_type={shape_type!r} cannot be rendered. PIL's built-in "
        "bitmap font ignores font_size and would draw a ~4x7 pixel box, "
        "silently producing a blank probe. Set DISCO_FONT_PATH or install a "
        f"CJK/sans-serif system font. Searched paths {_FONT_CANDIDATES}."
    )
  font_size = int(size * (0.26 if stacked else 0.72))
  font = _PIL_ImageFont.truetype(font_path, font_size)
  draw = _PIL_ImageDraw.Draw(img)

  chars = list(char) if stacked else [char]
  n_rows = len(chars)
  # Centred on size/2.0. Centring on (size-1)/2.0 -- the axis `np.fliplr`
  # actually mirrors about -- was tried and measured *worse*: Latin glyphs
  # lost ~9 points of self-mirror IoU (A 0.979 -> 0.849) and CJK was
  # bit-identical. The residual asymmetry is sub-pixel jitter from
  # `font.getbbox` returning integer bounds, so the true glyph centre is
  # quantised to within half a pixel and no single global offset fixes it.
  # That is why the symmetry check below is tolerant rather than exact.
  for row, glyph in enumerate(chars):
    bbox = font.getbbox(glyph)
    glyph_cx = (bbox[0] + bbox[2]) / 2.0
    glyph_cy = (bbox[1] + bbox[3]) / 2.0
    # Every glyph is centred on the canvas's vertical axis, so a stack as a
    # whole is symmetric about the same axis every other shape uses.
    cy = (size / 2.0) if n_rows == 1 else size * (row + 0.5) / n_rows
    draw.text(
        ((size / 2.0) - glyph_cx, cy - glyph_cy),
        glyph,
        font=font,
        fill=fg_rgb,
    )

  return font_path


def glyph_self_mirror_spectrum(
    img: "_PIL_Image.Image",
    bg_rgb: tuple[int, int, int],
    ks: "list[float] | tuple[float, ...]",
) -> list[float]:
  """`glyph_self_mirror_f1` at several tolerances, sharing one transform.

  The expensive part of `tolerant_f1` is the Euclidean distance transform, and
  it does not depend on `k` at all. Because the second mask here is the first
  one mirrored, its transform is likewise just the mirror of the first -- so a
  whole spectrum costs exactly one transform rather than two per tolerance.
  That is what makes a full-list audit cheap enough to run routinely, and an
  audit nobody runs is the state this probe set was already in.

  Args:
    img: Rendered RGB canvas, *before* any mirroring.
    bg_rgb: Background colour, used to extract the ink mask.
    ks: Match tolerances in pixels.

  Returns:
    One score per entry of `ks`; all 0.0 if the canvas has no ink.
  """
  arr = np.asarray(img)
  ink = visual_symmetry_eval.ink_mask(arr, np.array(bg_rgb, dtype=np.float64))
  if not ink.any():
    return [0.0] * len(ks)
  mirrored = np.fliplr(ink)
  dist = visual_symmetry_eval.distance_to_ink(ink)
  return [
      visual_symmetry_eval.tolerant_f1(
          ink, mirrored, k=k, dist_a=dist, dist_b=np.fliplr(dist)
      )
      for k in ks
  ]


def glyph_self_mirror_f1(
    img: "_PIL_Image.Image",
    bg_rgb: tuple[int, int, int],
    k: float = visual_symmetry_eval.DEFAULT_MATCH_TOLERANCE_PX,
) -> float:
  """How well a rendered glyph's ink agrees with its own mirror image.

  This is deliberately the *same* function that scores the target model
  (`visual_symmetry_eval.tolerant_f1`) over the *same* ink mask
  (`visual_symmetry_eval.ink_mask`), at the *same* default tolerance. That
  matters more than it looks: the number returned here is exactly the
  self-mirror score a flawless, language-aware model would earn on this glyph
  if it ignored the instruction and redrew the real character. So it answers
  the operational question directly -- can the experiment's own scorer even
  tell the mirrored-left chimera apart from the true glyph? -- rather than
  answering a proxy question with a proxy metric.

  What the residual is NOT: rasterisation. `font.getbbox` returns integer
  bounds, so a glyph's centre is quantised to within half a pixel of the mirror
  axis, which does depress the k=0 numbers. But that is at most one pixel of
  displacement after mirroring and is fully absorbed by the default tolerance.
  Re-rendering the whole list at 4x resolution, where any quantisation error
  shrinks fourfold, moved every score by less than 0.01 (磊 0.8409 -> 0.8341,
  天 0.9604 -> 0.9619). Scale invariance means the asymmetry lives in the font
  outlines, not in the rasteriser, so a glyph that fails here fails for real
  and cannot be rescued by rendering it differently.

  Args:
    img: Rendered RGB canvas, *before* any mirroring.
    bg_rgb: Background colour, used to extract the ink mask.
    k: Match tolerance in pixels. Defaults to the production scorer's.

  Returns:
    Tolerant F1 against its own reflection, in [0, 1]; 0.0 if there is no ink.
  """
  return glyph_self_mirror_spectrum(img, bg_rgb, (k,))[0]


# Minimum self-mirror agreement for a source glyph to count as symmetric.
#
# Set well above the experiment's own pass threshold
# (`visual_symmetry_eval.DEFAULT_IOU_THRESHOLD`, 0.90) on purpose. A glyph
# sitting at 0.90 would mean a model that redrew the true character landed
# exactly on the pass/fail boundary, so the probe would be measuring the
# rendering's asymmetry as much as the model's behaviour. Requiring near-
# perfect agreement keeps the chimera and the real character indistinguishable
# to the scorer, which is the property the probe actually depends on.
#
# Measured across every character in every list using `glyph_self_mirror_f1`.
MIN_GLYPH_SELF_MIRROR_F1: float = 0.97


def assert_source_glyph_symmetric(
    img: "_PIL_Image.Image",
    bg_rgb: tuple[int, int, int],
    char: str,
    shape_type: str = "",
) -> None:
  """Verifies the rendered glyph is mirror symmetric before it is mirrored.

  This is the check that `assert_ground_truth_symmetric` cannot make. The
  ground truth is constructed as `right := fliplr(left)`, so its symmetry is a
  tautology; the substantive question is whether the character being mirrored
  was symmetric in the first place. When it is not, the probe shows the model
  the left half of a real character and scores it against a mirrored-left
  chimera that does not exist in the language.

  Args:
    img: Rendered RGB canvas, before mirroring.
    bg_rgb: Background colour.
    char: Character(s) drawn, for the error message.
    shape_type: Shape name, for the error message.

  Raises:
    RuntimeError: If the glyph is not mirror symmetric in this font.
  """
  score = glyph_self_mirror_f1(img, bg_rgb)
  if score < MIN_GLYPH_SELF_MIRROR_F1:
    raise RuntimeError(
        f"Source glyph {char!r} (shape_type={shape_type!r}) is not mirror "
        "symmetric in the rendering font: self-mirror tolerant-F1 "
        f"{score:.4f} < {MIN_GLYPH_SELF_MIRROR_F1}. Mirroring its left half "
        "would produce a shape that is not this character, and the target "
        "model would be scored against it. Remove this character from its "
        "list."
    )


def assert_ground_truth_symmetric(
    gt_arr: np.ndarray,
    shape_type: str = "",
    seed: int | None = None,
    max_mismatched_pixels: int = 0,
) -> None:
  """Verifies a ground-truth image is exactly mirror symmetric, or raises.

  Scope, stated because it is narrow: the ground truth is *constructed* as
  ``right := fliplr(left)``, so this check is close to a tautology and will
  essentially never fire. It guards only against something overwriting the
  array afterwards. It does NOT establish that the underlying pattern is a
  sensible target -- for glyphs that is
  ``assert_source_glyph_symmetric``, which runs before the mirror step.

  The task the target model is asked to perform is "make this image symmetric".
  Every score is computed on the premise that the ground truth already is. If
  that premise is ever false, the model is penalised for our rendering bug and
  the failure is indistinguishable from a real capability limit -- which is
  exactly how the earlier font fallback survived: the glyph probes were all
  identical blank boxes, and the resulting 0/100 read as a plausible finding
  about stroke complexity.

  The check is on the reflection of the left half onto the right, excluding the
  centre column for odd widths (the axis belongs to neither side).

  Args:
    gt_arr: (H, W, 3) uint8 ground-truth image.
    shape_type: Shape name, for the error message only.
    seed: Seed, for the error message only.
    max_mismatched_pixels: Tolerance, in pixels. Zero by default, because the
      construction is an exact array copy and anything nonzero means something
      else wrote to the array afterwards.

  Raises:
    RuntimeError: If the image is not mirror symmetric.
  """
  width = gt_arr.shape[1]
  half = width // 2
  right = gt_arr[:, width - half :]
  mirrored_left = np.fliplr(gt_arr[:, :half])
  if right.shape != mirrored_left.shape:
    raise RuntimeError(
        f"Ground truth for shape_type={shape_type!r} seed={seed!r} has "
        f"mismatched halves: right {right.shape} vs mirrored left "
        f"{mirrored_left.shape}."
    )
  differing = np.any(right != mirrored_left, axis=-1)
  n_bad = int(differing.sum())
  if n_bad > max_mismatched_pixels:
    rows, cols = np.nonzero(differing)
    first = (int(rows[0]), int(cols[0])) if n_bad else None
    raise RuntimeError(
        f"Ground truth for shape_type={shape_type!r} seed={seed!r} is NOT "
        f"mirror symmetric: {n_bad} of {differing.size} pixels in the right "
        f"half differ from the mirrored left half (first at {first}). The "
        "probe would be asking the model to reproduce an asymmetric target, "
        "and every score computed against it would be meaningless."
    )


def _assert_has_ink(
    gt_arr: np.ndarray,
    bg_rgb: tuple[int, ...],
    shape_type: str,
    seed: int | None,
    min_ink_fraction: float = 0.002,
) -> None:
  """Raises if a rendered probe is effectively blank.

  Args:
    gt_arr: (H, W, 3) uint8 ground-truth image.
    bg_rgb: Background the canvas was filled with.
    shape_type: Shape name, for the error message.
    seed: Seed, for the error message.
    min_ink_fraction: Minimum fraction of non-background pixels.

  Raises:
    RuntimeError: If ink coverage is below `min_ink_fraction`.
  """
  dist = np.sqrt(
      np.sum(
          (gt_arr.astype(np.float64) - np.array(bg_rgb, dtype=np.float64)) ** 2,
          axis=-1,
      )
  )
  ink_fraction = float((dist > 30.0).mean())
  if ink_fraction < min_ink_fraction:
    raise RuntimeError(
        f"Probe for shape_type={shape_type!r} seed={seed!r} rendered only "
        f"{ink_fraction:.5f} ink coverage, below the {min_ink_fraction} "
        "minimum. A blank probe scores a perfect self-mirror and would be "
        "recorded as a model success."
    )


def render_direct_symmetry_image(
    shape_type: str,
    canvas_size: int = DIRECT_CANVAS_SIZE,
    bg_color: str = "white",
    seed: int | None = None,
    show_midline: bool = True,
    font_path: str | None = _FONT_PATH,
    allow_dynamic: bool = False,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Generates a continuous, high-resolution direct symmetry image.

  Exact Symmetry Guarantee:
    The right half of the ground truth image is constructed as the exact
    horizontal mirror reflection of the left half:
      gt_arr[:, half:] = np.fliplr(gt_arr[:, :half])
    This guarantees 100.0% subpixel symmetry.

  Args:
    shape_type: One of the registered shape types or dynamic string.
    canvas_size: Canvas height and width in pixels (default 320).
    bg_color: Background color name or hex.
    seed: Random seed for reproducibility.
    show_midline: Whether to draw a subtle vertical reflection line in prompt.
    font_path: Path to TTF font file.
    allow_dynamic: If True, unknown shape types are rendered dynamically as
      procedural symmetric patterns instead of raising ValueError.
    **kwargs: Additional parameters (e.g. letter, fg_color).

  Returns:
    (ground_truth_image, prompt_image, ground_truth_image): (size, size, 3)
    uint8 arrays.

  Raises:
    ValueError: If shape_type is not supported and allow_dynamic is False.
  """
  allow_dynamic = kwargs.pop("allow_dynamic", allow_dynamic)
  if shape_type not in SUPPORTED_SHAPE_TYPES and not allow_dynamic:
    raise ValueError(
        f"Unknown shape_type: {shape_type!r}. Supported:"
        f" {SUPPORTED_SHAPE_TYPES}"
    )

  rng = _random.Random(seed)
  size = canvas_size
  bg_rgb = _bg_color_rgb(bg_color)
  half = size // 2

  img = _PIL_Image.new("RGB", (size, size), bg_rgb)
  draw = _PIL_ImageDraw.Draw(img)

  if shape_type in (
      "english_letter",
      "greek_letter",
      "chinese_char",
      "complex_chinese",
      "multi_glyph",
  ):
    if shape_type == "english_letter":
      char = kwargs.get("letter", rng.choice(SYMMETRIC_ENGLISH_LETTERS))
    elif shape_type == "greek_letter":
      char = kwargs.get("letter", rng.choice(SYMMETRIC_GREEK_LETTERS))
    elif shape_type == "complex_chinese":
      char = kwargs.get("letter", rng.choice(COMPLEX_SYMMETRIC_CHINESE_CHARS))
    elif shape_type == "multi_glyph":
      # Three smaller glyphs stacked vertically. Shrinking the glyph is what
      # makes this hard: the strokes thin out roughly in proportion, so the
      # reflection has to be registered at a finer scale than a single large
      # character requires.
      char = kwargs.get(
          "letter",
          "".join(
              rng.choice(
                  SYMMETRIC_CHINESE_CHARS
                  + SYMMETRIC_GREEK_LETTERS
                  + SYMMETRIC_ENGLISH_LETTERS
              )
              for _ in range(3)
          ),
      )
    else:
      char = kwargs.get("letter", rng.choice(SYMMETRIC_CHINESE_CHARS))

    fg_rgb = _pick_fg_rgb(
        rng, bg_rgb, kwargs.get("fg_color"), default_for_glyphs=True
    )

    font_file = draw_glyphs_centered(
        img,
        char,
        fg_rgb=fg_rgb,
        font_path=font_path or _FONT_PATH,
        stacked=(shape_type == "multi_glyph"),
        shape_type=shape_type,
    )

    # Guard against a font that loads but has no glyph for `char` (e.g. a
    # Latin-only font asked for a CJK codepoint), which renders as .notdef or
    # nothing at all.
    _assert_glyph_rendered(img, bg_rgb, shape_type, char, font_file)

    # The source glyph must itself be mirror symmetric. The ground truth is
    # built by overwriting the right half with the mirrored left half, so an
    # asymmetric character does not merely render imperfectly -- it is
    # silently replaced by a left-mirrored chimera that is not the character.
    # The model is then shown a real character's left half and scored against
    # a shape that does not exist, and its "failure" is our bug.
    assert_source_glyph_symmetric(img, bg_rgb, char, shape_type)

  elif shape_type == "square":
    fg_rgb = _pick_fg_rgb(rng, bg_rgb, kwargs.get("fg_color"))
    side_arg = kwargs.get("size", None)
    if side_arg is not None:
      side = int(side_arg * (size / 10.0)) if side_arg < 15 else int(side_arg)
    else:
      side = rng.randint(int(size * 0.25), int(size * 0.60))
    x0 = (size - side) // 2
    y0 = (size - side) // 2
    draw.rectangle([x0, y0, x0 + side, y0 + side], fill=fg_rgb)

  elif shape_type == "circle":
    fg_rgb = _pick_fg_rgb(rng, bg_rgb, kwargs.get("fg_color"))
    rad_arg = kwargs.get("radius", None)
    if rad_arg is not None:
      rad = int(rad_arg * (size / 10.0)) if rad_arg < 15 else int(rad_arg)
    else:
      rad = rng.randint(int(size * 0.15), int(size * 0.35))
    cx, cy = size // 2, size // 2
    draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=fg_rgb)

  elif shape_type == "stripes":
    fg_rgb = _pick_fg_rgb(rng, bg_rgb, kwargs.get("fg_color"))
    stripe_h = kwargs.get(
        "stripe_width", rng.choice([size // 14, size // 10, size // 7])
    )
    y_start = rng.randint(0, stripe_h)
    for y in range(y_start, size, stripe_h * 2):
      draw.rectangle([0, y, size, min(y + stripe_h, size)], fill=fg_rgb)

  elif shape_type == "dots":
    fg_rgb = _pick_fg_rgb(rng, bg_rgb, kwargs.get("fg_color"))
    n_dots = kwargs.get("n_dots", rng.randint(2, 6))
    dot_r = max(4, size // 25)
    for _ in range(n_dots):
      dx = rng.randint(dot_r, half - dot_r - 2)
      dy = rng.randint(dot_r, size - dot_r - 2)
      draw.ellipse(
          [dx - dot_r, dy - dot_r, dx + dot_r, dy + dot_r], fill=fg_rgb
      )

  elif shape_type in ("random_sparse", "random_dense"):
    density = 0.12 if shape_type == "random_sparse" else 0.38
    grid_n = 16
    cell_s = size // grid_n
    half_cols = grid_n // 2
    fg_rgb = _pick_fg_rgb(rng, bg_rgb, kwargs.get("fg_color"))
    for r in range(grid_n):
      for c in range(half_cols):
        if rng.random() < density:
          draw.rectangle(
              [c * cell_s, r * cell_s, (c + 1) * cell_s, (r + 1) * cell_s],
              fill=fg_rgb,
          )

  elif shape_type == "triangle":
    fg_rgb = _pick_fg_rgb(rng, bg_rgb, kwargs.get("fg_color"))
    side_arg = kwargs.get("size", None)
    if side_arg is not None:
      side = int(side_arg * (size / 10.0)) if side_arg < 15 else int(side_arg)
    else:
      side = rng.randint(int(size * 0.35), int(size * 0.70))
    cx, cy = size // 2, size // 2
    h = int(side * np.sqrt(3) / 2)
    top = (cx, cy - 2 * h // 3)
    bl = (cx - side // 2, cy + h // 3)
    br = (cx + side // 2, cy + h // 3)
    draw.polygon([top, bl, br], fill=fg_rgb)

  elif shape_type in ("plus", "cross"):
    fg_rgb = _pick_fg_rgb(rng, bg_rgb, kwargs.get("fg_color"))
    side_arg = kwargs.get("size", None)
    if side_arg is not None:
      arm = int(side_arg * (size / 10.0)) if side_arg < 15 else int(side_arg)
    else:
      arm = rng.randint(int(size * 0.35), int(size * 0.65))
    thick = max(6, arm // 4)
    cx, cy = size // 2, size // 2
    draw.rectangle(
        [cx - arm // 2, cy - thick // 2, cx + arm // 2, cy + thick // 2],
        fill=fg_rgb,
    )
    draw.rectangle(
        [cx - thick // 2, cy - arm // 2, cx + thick // 2, cy + arm // 2],
        fill=fg_rgb,
    )

  elif shape_type == "heart":
    fg_rgb = _pick_fg_rgb(rng, bg_rgb, kwargs.get("fg_color"))
    scale = size * 0.015
    cx, cy = size // 2, int(size * 0.48)
    t = np.linspace(0, 2 * np.pi, 120)
    xs = cx + scale * (16.0 * (np.sin(t) ** 3))
    ys = cy - scale * (
        13.0 * np.cos(t)
        - 5.0 * np.cos(2 * t)
        - 2.0 * np.cos(3 * t)
        - np.cos(4 * t)
    )
    pts = list(zip(xs, ys))
    draw.polygon(pts, fill=fg_rgb)

  elif shape_type == "fine_noise":
    # Same construction as random_sparse/random_dense but on a 32x32 lattice
    # instead of 16x16, so each cell is 10px rather than 20px. Halving the
    # feature size is the whole point: it doubles the number of independent
    # elements whose reflection has to be placed correctly.
    density = float(kwargs.get("density", 0.25))
    grid_n = 32
    cell_s = size // grid_n
    half_cols = grid_n // 2
    fg_rgb = _pick_fg_rgb(rng, bg_rgb, kwargs.get("fg_color"))
    for r in range(grid_n):
      for c in range(half_cols):
        if rng.random() < density:
          draw.rectangle(
              [c * cell_s, r * cell_s, (c + 1) * cell_s, (r + 1) * cell_s],
              fill=fg_rgb,
          )

  elif shape_type == "maze":
    # A comb of thin vertical bars with horizontal connectors. Thin strokes at
    # irregular positions: there is no silhouette to fall back on, the model
    # has to place each bar.
    fg_rgb = _pick_fg_rgb(rng, bg_rgb, kwargs.get("fg_color"))
    stroke = int(kwargs.get("stroke_width", max(2, size // 120)))
    n_bars = int(kwargs.get("n_bars", rng.randint(7, 11)))
    margin = int(size * 0.08)
    for i in range(n_bars):
      x = margin + int((half - 2 * margin) * i / max(1, n_bars - 1))
      y0 = rng.randint(margin, int(size * 0.5))
      y1 = rng.randint(int(size * 0.55), size - margin)
      draw.line([(x, y0), (x, y1)], fill=fg_rgb, width=stroke)
      # Connector to the next bar, at a random height along the shared span.
      if i < n_bars - 1:
        x_next = margin + int(
            (half - 2 * margin) * (i + 1) / max(1, n_bars - 1)
        )
        y_link = rng.randint(max(y0, margin), max(y0 + 1, y1))
        draw.line([(x, y_link), (x_next, y_link)], fill=fg_rgb, width=stroke)

  elif shape_type == "snowflake":
    # Radial arms with sub-branches, all emanating from the canvas centre so
    # the figure is symmetric about the same vertical axis as everything else.
    fg_rgb = _pick_fg_rgb(rng, bg_rgb, kwargs.get("fg_color"))
    stroke = int(kwargs.get("stroke_width", max(2, size // 130)))
    cx, cy = size / 2.0, size / 2.0
    n_arms = int(kwargs.get("n_arms", rng.choice([5, 6, 7, 8])))
    arm_len = size * rng.uniform(0.30, 0.44)
    # Only the left half is meaningful (the right is overwritten by the
    # mirror), so arms are laid out over the left semicircle.
    for i in range(n_arms):
      theta = np.pi / 2.0 + np.pi * (i + 0.5) / n_arms
      ex, ey = cx + arm_len * np.cos(theta), cy + arm_len * np.sin(theta)
      draw.line([(cx, cy), (ex, ey)], fill=fg_rgb, width=stroke)
      for frac in (0.45, 0.72):
        bx, by = cx + arm_len * frac * np.cos(
            theta
        ), cy + arm_len * frac * np.sin(theta)
        blen = arm_len * 0.22
        for sign in (-1.0, 1.0):
          phi = theta + sign * np.pi / 4.0
          draw.line(
              [(bx, by), (bx + blen * np.cos(phi), by + blen * np.sin(phi))],
              fill=fg_rgb,
              width=max(1, stroke - 1),
          )

  elif shape_type == "filigree":
    # Thin arcs plus small rings and ticks, scattered. Dense curvature with
    # very little ink mass.
    fg_rgb = _pick_fg_rgb(rng, bg_rgb, kwargs.get("fg_color"))
    stroke = int(kwargs.get("stroke_width", max(2, size // 140)))
    for _ in range(int(kwargs.get("n_arcs", rng.randint(6, 10)))):
      r = rng.randint(int(size * 0.06), int(size * 0.20))
      ax = rng.randint(r, max(r + 1, half - 4))
      ay = rng.randint(r, max(r + 1, size - r - 2))
      start = rng.randint(0, 359)
      draw.arc(
          [ax - r, ay - r, ax + r, ay + r],
          start=start,
          end=start + rng.randint(90, 280),
          fill=fg_rgb,
          width=stroke,
      )
    for _ in range(int(kwargs.get("n_ticks", rng.randint(5, 9)))):
      tx = rng.randint(4, max(5, half - 4))
      ty = rng.randint(4, size - 5)
      tl = rng.randint(int(size * 0.02), int(size * 0.06))
      draw.line([(tx, ty), (tx + tl, ty + tl)], fill=fg_rgb, width=stroke)

  elif shape_type == "nested_outlines":
    # Concentric outlines, alternating between rectangles and ellipses. Filled
    # shapes are forgiving because only the boundary can be wrong; outlines are
    # boundary all the way through.
    fg_rgb = _pick_fg_rgb(rng, bg_rgb, kwargs.get("fg_color"))
    stroke = int(kwargs.get("stroke_width", max(2, size // 140)))
    cx, cy = size // 2, size // 2
    n_rings = int(kwargs.get("n_rings", rng.randint(4, 6)))
    max_r = int(size * 0.42)
    for i in range(n_rings):
      r = int(max_r * (i + 1) / n_rings)
      box = [cx - r, cy - r, cx + r, cy + r]
      if (i + rng.randint(0, 1)) % 2 == 0:
        draw.ellipse(box, outline=fg_rgb, width=stroke)
      else:
        draw.rectangle(box, outline=fg_rgb, width=stroke)

  elif shape_type == "starburst":
    # Many thin rays of differing lengths from the centre.
    fg_rgb = _pick_fg_rgb(rng, bg_rgb, kwargs.get("fg_color"))
    stroke = int(kwargs.get("stroke_width", max(1, size // 160)))
    cx, cy = size / 2.0, size / 2.0
    n_rays = int(kwargs.get("n_rays", rng.randint(9, 15)))
    for i in range(n_rays):
      theta = np.pi / 2.0 + np.pi * (i + 0.5) / n_rays
      length = size * rng.uniform(0.18, 0.46)
      draw.line(
          [
              (cx, cy),
              (cx + length * np.cos(theta), cy + length * np.sin(theta)),
          ],
          fill=fg_rgb,
          width=stroke,
      )

  elif shape_type == "scatter_rings":
    # Small hollow rings of varying radius. Sparse and thin at the same time,
    # which is the combination raw IoU used to be worst at and which the model
    # genuinely does find harder than filled dots.
    fg_rgb = _pick_fg_rgb(rng, bg_rgb, kwargs.get("fg_color"))
    stroke = int(kwargs.get("stroke_width", max(2, size // 150)))
    for _ in range(int(kwargs.get("n_rings", rng.randint(5, 9)))):
      r = rng.randint(int(size * 0.03), int(size * 0.10))
      rx = rng.randint(r + 2, max(r + 3, half - r - 2))
      ry = rng.randint(r + 2, max(r + 3, size - r - 2))
      draw.ellipse(
          [rx - r, ry - r, rx + r, ry + r], outline=fg_rgb, width=stroke
      )

  elif shape_type == "wave_lines":
    # Stacked sinusoids with independent amplitude, frequency and phase.
    fg_rgb = _pick_fg_rgb(rng, bg_rgb, kwargs.get("fg_color"))
    stroke = int(kwargs.get("stroke_width", max(2, size // 140)))
    n_waves = int(kwargs.get("n_waves", rng.randint(4, 7)))
    for i in range(n_waves):
      y0 = size * (i + 0.5) / n_waves
      amp = size * rng.uniform(0.015, 0.045)
      freq = rng.uniform(1.5, 4.0)
      phase = rng.uniform(0.0, 2.0 * np.pi)
      pts = []
      for x in range(0, half + 1, 2):
        y = y0 + amp * np.sin(2.0 * np.pi * freq * x / size + phase)
        pts.append((float(x), float(y)))
      draw.line(pts, fill=fg_rgb, width=stroke, joint="curve")

  elif (
      shape_type == "dynamic"
      or allow_dynamic
      or shape_type not in SUPPORTED_SHAPE_TYPES
  ):
    fg_rgb = _pick_fg_rgb(rng, bg_rgb, kwargs.get("fg_color"))
    if len(shape_type) == 1 and not shape_type.isalnum():
      font_file = draw_glyphs_centered(
          img,
          shape_type,
          fg_rgb=fg_rgb,
          font_path=font_path or _FONT_PATH,
          stacked=False,
          shape_type=shape_type,
      )
      _assert_glyph_rendered(img, bg_rgb, shape_type, shape_type, font_file)
    else:
      num_pts = rng.randint(3, 7)
      pts = []
      for _ in range(num_pts):
        px = rng.randint(int(size * 0.08), half - 2)
        py = rng.randint(int(size * 0.08), int(size * 0.92))
        pts.append((px, py))
      if len(pts) >= 3:
        c_px = sum(p[0] for p in pts) / len(pts)
        c_py = sum(p[1] for p in pts) / len(pts)
        pts.sort(key=lambda p: np.arctan2(p[1] - c_py, p[0] - c_px))
        draw.polygon(pts, fill=fg_rgb)

  # Convert to numpy array
  gt_arr = np.array(img, dtype=np.uint8)

  # Exact mathematical symmetry guarantee: the right half is *replaced* by the
  # mirror of the left half, so whatever was drawn on the right is discarded.
  #
  # The destination starts at `size - half`, not at `half`. They coincide for an
  # even canvas, but for an odd one `gt_arr[:, half:]` is one column wider than
  # `fliplr(gt_arr[:, :half])` and the assignment raises. Writing it this way
  # leaves the centre column (the reflection axis itself) untouched, which is
  # the geometrically correct thing to do.
  gt_arr[:, size - half :] = np.fliplr(gt_arr[:, :half])

  # Assert rather than assume. Every downstream number is computed against this
  # array on the premise that it is exactly symmetric; if a future shape handler
  # draws after this point, or a resize sneaks in, the premise breaks silently
  # and the target model gets blamed for our own asymmetry. This is the same
  # class of bug as the font fallback that turned every glyph probe into an
  # identical 4x7 box while the scores still looked plausible.
  assert_ground_truth_symmetric(gt_arr, shape_type=shape_type, seed=seed)

  # A probe with (almost) no ink cannot measure anything: a blank canvas scores
  # a perfect self-mirror, and the left-fidelity term that is supposed to reject
  # it is itself comparing two near-empty masks. Glyph shapes have their own
  # stricter check above; this catches the procedural shapes, where a bad
  # parameter combination (zero rings, zero-length rays) would otherwise produce
  # an empty probe that silently reads as a model success.
  _assert_has_ink(gt_arr, bg_rgb, shape_type, seed)

  # Generate prompt image: right half erased with background
  prompt_arr = gt_arr.copy()
  prompt_arr[:, size - half :] = bg_rgb

  # Vertical reflection axis guide (middle line)
  if show_midline:
    prompt_img = _PIL_Image.fromarray(prompt_arr)
    draw_prompt = _PIL_ImageDraw.Draw(prompt_img)
    dash_len = int(kwargs.get("dash_len", 6))
    space_len = int(kwargs.get("space_len", 4))
    # Derived from the background rather than hard-coded: the old constant
    # (180, 190, 205) is invisible on a light background and barely visible on
    # several others, so the axis hint was silently present for some conditions
    # and absent for others.
    guide_col = kwargs.get("midline_color", _midline_color(bg_rgb))
    midline_width = int(kwargs.get("midline_width", 1))
    if space_len == 0 or kwargs.get("midline_style") == "solid":
      draw_prompt.line(
          [(half, 0), (half, size)],
          fill=guide_col,
          width=midline_width,
      )
    else:
      y = 0
      while y < size:
        draw_prompt.line(
            [(half, y), (half, min(y + dash_len, size))],
            fill=guide_col,
            width=midline_width,
        )
        y += dash_len + space_len
    prompt_arr = np.array(prompt_img, dtype=np.uint8)

  return gt_arr, prompt_arr, gt_arr


def generate_symmetry_image(
    shape_type: str,
    canvas_size: int = DIRECT_CANVAS_SIZE,
    bg_color: str = "white",
    seed: int | None = None,
    show_midline: bool = True,
    allow_dynamic: bool = False,
    *args: Any,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Generates a symmetric continuous image and its half-erased prompt version.

  Generates a continuous high-resolution image pair without grid cells
  with 100.0% exact horizontal reflection.

  Args:
    shape_type: One of the registered shape types (see list_shape_types()).
    canvas_size: Canvas dimension in pixels (default 320).
    bg_color: Background color name.
    seed: Random seed for reproducibility.
    show_midline: Whether to draw a subtle midline guide in direct mode.
    allow_dynamic: Whether to allow procedural fallback for unknown shapes.
    *args: Ignored (legacy positional parameters).
    **kwargs: Additional keyword arguments passed to the shape generator.

  Returns:
    A tuple of:
      - ground_truth: (canvas_size, canvas_size, 3) ground-truth array.
      - prompt: (canvas_size, canvas_size, 3) prompt array with right half
        erased.
      - ground_truth_image: (canvas_size, canvas_size, 3) rendered image of the
        full pattern.
  """
  del args
  # Gracefully consume legacy grid arguments if passed by older callers
  kwargs.pop("rows", None)
  kwargs.pop("cols", None)
  kwargs.pop("render_mode", None)
  midline = kwargs.pop("show_midline", show_midline)
  allow_dyn = kwargs.pop("allow_dynamic", allow_dynamic)
  return render_direct_symmetry_image(
      shape_type=shape_type,
      canvas_size=canvas_size,
      bg_color=bg_color,
      seed=seed,
      show_midline=midline,
      allow_dynamic=allow_dyn,
      **kwargs,
  )
