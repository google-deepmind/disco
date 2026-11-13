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

"""Single source of truth for the prompt text used in visual symmetry runs.

Discovery and verification drive the same target model through the same tools.
Centralising the prompts and catalogs here ensures that both phases agree by
construction on:

  * All 22 probe shape families (both base and hard families).
  * The default scoring criterion (the `self_mirror`/`preservation` conjunction
    measured with tolerant F1).
  * The background colour palette (`bg_color`).

The catalogs are DERIVED, never re-typed -- `SUPPORTED_SHAPE_TYPES` and
`BACKGROUND_PALETTE` are imported from the generator that actually renders the
probes, so adding a shape family updates every prompt automatically.
"""

import dataclasses
import re

from disco.tasks.visual_symmetry import eval as visual_symmetry_eval
from disco.tasks.visual_symmetry import hypotheses as visual_symmetry_hypotheses
from disco.tasks.visual_symmetry import image_gen as visual_symmetry_image_gen

# ── Canonical probe text ──────────────────────────────────────────────────────

# The exact text sent to the target model alongside every half-erased probe, in
# BOTH discovery and verification, in-sample and out-of-sample.
#
# Re-exported from `visual_symmetry_hypotheses` so discovery and verification
# are guaranteed to use the identical prompt text.
CANONICAL_PROBE_PROMPT: str = visual_symmetry_hypotheses.SYMMETRY_PROMPT

# Shape families the probe generator can actually render, and the background
# colours it accepts. Derived from the generator, so prompts cannot go stale.
SUPPORTED_SHAPE_TYPES: list[str] = (
    visual_symmetry_image_gen.SUPPORTED_SHAPE_TYPES
)
BASE_SHAPE_TYPES: list[str] = visual_symmetry_image_gen.BASE_SHAPE_TYPES
HARD_SHAPE_TYPES: list[str] = visual_symmetry_image_gen.HARD_SHAPE_TYPES
BACKGROUND_PALETTE: list[str] = visual_symmetry_image_gen.BACKGROUND_PALETTE


def shape_catalog_sentence() -> str:
  """Returns a prose description of the probe families, split base vs hard."""
  base = ", ".join(f"'{s}'" for s in BASE_SHAPE_TYPES)
  hard = ", ".join(f"'{s}'" for s in HARD_SHAPE_TYPES)
  return (
      f"{len(BASE_SHAPE_TYPES)} base families ({base}) and"
      f" {len(HARD_SHAPE_TYPES)} deliberately harder, finer-stroked families"
      f" with more parts and more internal structure ({hard})"
  )


def background_palette_sentence() -> str:
  """Returns a prose description of the selectable background colours."""
  colours = ", ".join(f"'{c}'" for c in BACKGROUND_PALETTE)
  return (
      f"a fixed palette of {len(BACKGROUND_PALETTE)} background colours"
      f" ({colours})"
  )


def scoring_description() -> str:
  """Returns an accurate description of how a completion is scored.

  Kept faithful to `visual_symmetry_eval`: the default is the CONJUNCTION of
  self-mirror symmetry and left-half preservation, measured with a tolerant F1,
  not a raw ink-mask IoU against ground truth. The distinction matters in both
  directions -- a blank canvas is perfectly self-symmetric (so symmetry alone is
  gameable), and raw IoU charges the model for stroke weight (so it tracked how
  thin the shape was rather than how well it reflected).
  """
  return (
      "A completion counts as a Success when BOTH hold: (a) SYMMETRY -- the"
      " returned image is symmetric about its own vertical centre axis,"
      f" agreement >= {visual_symmetry_eval.DEFAULT_IOU_THRESHOLD:.2f} between"
      " its right half and its mirrored left half; and (b) PRESERVATION -- it"
      " kept the left half it was given, agreement >="
      f" {visual_symmetry_eval.DEFAULT_LEFT_FIDELITY_THRESHOLD:.2f} against the"
      " probe. Criterion (b) is required because a blank canvas satisfies (a)"
      " perfectly. Agreement is a tolerant F1 with a"
      f" {visual_symmetry_eval.DEFAULT_MATCH_TOLERANCE_PX:.0f} pixel match"
      f" tolerance at the {visual_symmetry_eval.REFERENCE_CANVAS_SIZE}px probe"
      " resolution, NOT a raw ink-mask IoU: raw IoU charges the model for"
      " stroke weight, which made the score track how thin the shape was"
      " rather than how well it reflected."
  )


# Instruction block for naming the criterion a hypothesis is actually about.
#
# `reconstruction` (match against ground truth) is always computed and reported,
# but is deliberately NOT claimable: the 0.90 null is calibrated against the
# conjunction, so a reconstruction claim would be tested against a rate that was
# never measured for it.
SUCCESS_CRITERION_INSTRUCTION: str = (
    "`success_criterion`: name the criterion your HYPOTHESIS is actually"
    " about.\n"
    "     * 'both' (default): the conjunction of symmetry AND preservation."
    " Use this unless your hypothesis is specifically about one half of it.\n"
    "     * 'symmetry': only whether the output is self-symmetric. Note a"
    " blank canvas passes this, so prefer it for FAILURE claims.\n"
    "     * 'preservation': only whether the given left half survived"
    " unmodified.\n"
    "     Do NOT base a claim on 'reconstruction' (similarity to the hidden"
    " ground truth): it is reported for reference but the reference accuracy"
    " rate is not calibrated for it."
)


def infer_shape_type(h_text: str, s_text: str = "") -> str:
  """Infers which probe family a hypothesis is about.

  Ordering matters: the most specific family must win. `complex_chinese` is
  checked before `chinese_char` because a hypothesis about high-stroke-count
  Hanzi contains the word "chinese", and matching it to the easy `chinese_char`
  family silently verifies a hard-family claim against easy probes. Likewise
  `multi_glyph` before `english_letter`.

  Args:
    h_text: The hypothesis text.
    s_text: Optional additional condition/scope text.

  Returns:
    A shape type guaranteed to be in `SUPPORTED_SHAPE_TYPES`.
  """
  combined_text = f"{h_text} {s_text}".lower()

  # Hard families first, so a specific hard claim is never absorbed by the
  # easy family whose name it happens to contain.
  for candidate in HARD_SHAPE_TYPES:
    if (
        candidate in combined_text
        or candidate.replace("_", " ") in combined_text
    ):
      return candidate

  # Distinctive keywords for hard families that may be described rather than
  # named. Checked before the base families for the same reason.
  hard_keywords: list[tuple[str, tuple[str, ...]]] = [
      (
          "complex_chinese",
          (
              "high-stroke",
              "high stroke",
              "intricate stroke",
              "stroke-dense",
              "complex hanzi",
              "complex chinese",
          ),
      ),
      ("multi_glyph", ("multi-glyph", "multiple glyphs", "stacked glyph")),
      ("fine_noise", ("fine noise", "fine-grained noise")),
      ("nested_outlines", ("nested outline", "concentric outline")),
      ("scatter_rings", ("scattered ring", "scatter ring")),
      ("wave_lines", ("wave line", "wavy line")),
  ]
  for candidate, keywords in hard_keywords:
    if any(kw in combined_text for kw in keywords):
      return candidate

  base_keywords: list[tuple[str, tuple[str, ...]]] = [
      ("chinese_char", ("chinese", "hanzi", "kanji", "character")),
      ("english_letter", ("english", "letter", "alphabet", "latin")),
      ("greek_letter", ("greek",)),
      ("dots", ("dot",)),
      ("stripes", ("stripe",)),
      ("square", ("square",)),
      ("circle", ("circle",)),
      ("triangle", ("triangle",)),
      ("plus", ("plus sign", "cross shape")),
      ("heart", ("heart",)),
  ]
  for candidate in BASE_SHAPE_TYPES:
    if (
        candidate in combined_text
        or candidate.replace("_", " ") in combined_text
    ):
      return candidate
  for candidate, keywords in base_keywords:
    if any(kw in combined_text for kw in keywords):
      return candidate

  return "english_letter"


# ── Contrastive (two-condition) hypotheses ────────────────────────────────────
#
# `infer_shape_type` answers "which single family is this hypothesis about",
# which is the only question an ABSOLUTE claim ("complex_chinese succeeds below
# 90 percent") asks. A CONTRASTIVE claim ("complex_chinese fails more often than
# chinese_char") names two conditions and asserts something about the difference
# between them, and collapsing it to one family does not under-power the test --
# it tests a different hypothesis, against a reference rate that was never
# calibrated for a difference.
#
# Everything below is additive. `infer_shape_type` is unchanged and remains the
# sole path whenever `infer_contrast` finds no contrast, so absolute claims are
# resolved exactly as before.


@dataclasses.dataclass(frozen=True)
class ProbeCondition:
  """One arm of a probe batch: a shape family drawn on a background colour."""

  shape_type: str
  bg_color: str = "white"

  @property
  def label(self) -> str:
    """Short human-readable name, used in logs and record diagnostics."""
    if self.bg_color == "white":
      return self.shape_type
    return f"{self.shape_type}@{self.bg_color}"

  def probe_args(self, n_probes: int) -> dict[str, object]:
    """Returns kwargs for `generate_and_send_symmetry_probes`."""
    return {
        "shape_type": self.shape_type,
        "bg_color": self.bg_color,
        "n_probes": n_probes,
    }


@dataclasses.dataclass(frozen=True)
class ContrastPlan:
  """Which condition(s) a hypothesis must be probed under.

  `condition_b is None` means the claim is absolute and is tested against the
  task's reference rate. Otherwise the claim is about A relative to B and must
  be tested with a two-sample comparison.
  """

  condition_a: ProbeCondition
  condition_b: ProbeCondition | None = None
  # "shape" or "background" -- which axis the two arms differ on. Empty for an
  # absolute claim.
  axis: str = ""

  @property
  def is_contrastive(self) -> bool:
    return self.condition_b is not None

  @property
  def conditions(self) -> list[ProbeCondition]:
    if self.condition_b is None:
      return [self.condition_a]
    return [self.condition_a, self.condition_b]


# Wording that signals a COMPARISON rather than a conjunction.
#
# Required before two mentioned conditions are treated as a contrast. Without
# it, "circles and squares both fail" -- a single claim about two families --
# would be split into two half-size arms and tested as a difference, which is
# not what it says.
_CONTRAST_MARKERS: tuple[str, ...] = (
    "than",
    "versus",
    "vs",
    "compared",
    "relative to",
    "whereas",
)


def _contrast_marker_present(text: str) -> bool:
  """Checks for comparative wording, matching whole words only."""
  return any(
      re.search(rf"\b{re.escape(marker)}\b", text)
      for marker in _CONTRAST_MARKERS
  )


# Wording that pins a hypothesis to one half of the success conjunction.
_PRESERVATION_KEYWORDS: tuple[str, ...] = (
    "preservation",
    "preserve",
    "preserved",
    "preserving",
    "left half",
    "left-half",
)
_SYMMETRY_ONLY_KEYWORDS: tuple[str, ...] = (
    "self-symmetr",
    "self symmetr",
    "self-mirror",
    "self mirror",
)


def infer_success_criterion(h_text: str, s_text: str = "") -> str:
  """Infers the scoring criterion a hypothesis is about.

  Needed only on the contrastive path, where the verifier drives the scoring
  tool itself and so cannot rely on the agent to pass `success_criterion`.

  Defaults to `both`, the conjunction the 0.90 reference rate is calibrated
  against, and narrows only when the text names exactly one half. Naming both
  halves, or neither, means the claim is about the conjunction.

  Args:
    h_text: The hypothesis text.
    s_text: Optional additional condition/scope text.

  Returns:
    One of "both", "symmetry", "preservation".
  """
  combined = f"{h_text} {s_text}".lower()
  wants_preservation = any(k in combined for k in _PRESERVATION_KEYWORDS)
  wants_symmetry_only = any(k in combined for k in _SYMMETRY_ONLY_KEYWORDS)
  if wants_preservation and not wants_symmetry_only:
    return "preservation"
  if wants_symmetry_only and not wants_preservation:
    return "symmetry"
  return "both"


# Shape families whose bare name is also a common English word, and which
# therefore may only be selected by a qualified keyword.
#
# "plus" is the live case: `infer_shape_type` matches bare substrings, so any
# hypothesis containing the conjunction "plus" selects the `plus` family. That
# is tolerable when the job is to pick one family out of 22, and not tolerable
# here, where a spurious second family splits the probe batch into two half-size
# arms and tests a contrast the hypothesis never claimed.
_AMBIGUOUS_BARE_NAMES: frozenset[str] = frozenset({"plus"})


def _shape_alias_table() -> list[tuple[str, tuple[str, ...]]]:
  """Returns (shape_type, aliases) in descending specificity order.

  Mirrors the precedence `infer_shape_type` encodes: hard families before the
  easy look-alikes whose names they contain, so `complex_chinese` is never
  absorbed by `chinese_char`.
  """
  table: list[tuple[str, tuple[str, ...]]] = []

  hard_keywords: dict[str, tuple[str, ...]] = {
      "complex_chinese": (
          "high-stroke",
          "high stroke",
          "intricate stroke",
          "stroke-dense",
          "complex hanzi",
          "complex chinese",
      ),
      "multi_glyph": ("multi-glyph", "multiple glyphs", "stacked glyph"),
      "fine_noise": ("fine noise", "fine-grained noise"),
      "nested_outlines": ("nested outline", "concentric outline"),
      "scatter_rings": ("scattered ring", "scatter ring"),
      "wave_lines": ("wave line", "wavy line"),
  }
  for shape in HARD_SHAPE_TYPES:
    aliases = [shape, shape.replace("_", " ")]
    aliases.extend(hard_keywords.get(shape, ()))
    table.append((shape, tuple(aliases)))

  # Base families. Keywords are deliberately narrower than `infer_shape_type`'s:
  # a false positive there picks a slightly wrong family, whereas here it
  # invents a second arm and halves the real one.
  base_keywords: dict[str, tuple[str, ...]] = {
      "chinese_char": ("chinese", "hanzi", "kanji"),
      "english_letter": ("english letter", "alphabet", "latin letter"),
      "greek_letter": ("greek",),
      "dots": ("dot",),
      "stripes": ("stripe",),
      "square": ("square",),
      "circle": ("circle",),
      "triangle": ("triangle",),
      "plus": ("plus sign", "cross shape"),
      "heart": ("heart",),
      "random_dense": ("dense random",),
      "random_sparse": ("sparse random",),
  }
  for shape in BASE_SHAPE_TYPES:
    if shape in _AMBIGUOUS_BARE_NAMES:
      # The family name is also an ordinary English word, so only the
      # qualified keywords may select it. Matching the bare word would turn
      # "circles plus noisy backgrounds" into a two-arm contrast.
      aliases = []
    else:
      aliases = [shape, shape.replace("_", " ")]
    aliases.extend(base_keywords.get(shape, ()))
    table.append((shape, tuple(aliases)))

  return table


def _find_mentions(
    text: str, table: list[tuple[str, tuple[str, ...]]]
) -> list[str]:
  """Finds every distinct entry mentioned in `text`, in order of appearance.

  Matches on word boundaries rather than bare substrings. `infer_shape_type`
  uses substring matching, which makes the English word "plus" select the
  `plus` shape family -- harmless when the answer is one family among many,
  but here it would fabricate a second arm out of ordinary prose.

  Entries are consumed in specificity order and matched spans are masked, so a
  hit on `complex_chinese` cannot be re-reported as `chinese_char` via the
  "chinese" keyword sitting inside it.

  Args:
    text: Lowercased text to scan.
    table: (name, aliases) pairs in descending specificity order.

  Returns:
    Distinct names, ordered by the position of their earliest match.
  """
  consumed = [False] * len(text)
  hits: list[tuple[int, str]] = []

  for name, aliases in table:
    best: tuple[int, int] | None = None
    for alias in aliases:
      # Tolerate a plural suffix: hypotheses are naturally written "circles
      # fail more than squares", and an exact-word match would find neither.
      for match in re.finditer(rf"\b{re.escape(alias)}s?\b", text):
        start, end = match.span()
        if any(consumed[start:end]):
          continue
        if best is None or start < best[0]:
          best = (start, end)
        break
    if best is not None:
      start, end = best
      for k in range(start, end):
        consumed[k] = True
      hits.append((start, name))

  hits.sort()
  ordered: list[str] = []
  for _, name in hits:
    if name not in ordered:
      ordered.append(name)
  return ordered


def find_shape_mentions(h_text: str, s_text: str = "") -> list[str]:
  """Returns the distinct shape families named in the text, in order."""
  return _find_mentions(f"{h_text} {s_text}".lower(), _shape_alias_table())


def find_background_mentions(h_text: str, s_text: str = "") -> list[str]:
  """Returns the distinct background colours named in the text, in order."""
  table: list[tuple[str, tuple[str, ...]]] = [
      (colour, (colour,)) for colour in BACKGROUND_PALETTE
  ]
  return _find_mentions(f"{h_text} {s_text}".lower(), table)


def infer_contrast(h_text: str, s_text: str = "") -> ContrastPlan:
  """Resolves the condition(s) a hypothesis must be probed under.

  A contrast is declared only when the text names two distinct conditions AND
  uses comparative wording. Both are required: two names alone may be a
  conjunction ("circles and squares both fail"), and comparative wording alone
  may be a comparison against the reference rate ("worse than 90 percent").

  When no contrast is found the result is an absolute plan whose single
  condition comes from `infer_shape_type`, i.e. byte-for-byte the previous
  behaviour.

  Args:
    h_text: The hypothesis text.
    s_text: Optional additional condition/scope text.

  Returns:
    A `ContrastPlan`. `is_contrastive` is False for an absolute claim.
  """
  combined = f"{h_text} {s_text}".lower()
  shapes = find_shape_mentions(h_text, s_text)
  backgrounds = find_background_mentions(h_text, s_text)

  if not _contrast_marker_present(combined):
    return _absolute_plan(h_text, s_text, shapes, backgrounds)

  # Shape axis wins when two families are named: a hypothesis that varies both
  # the family and the background confounds them, and the family is the
  # coarser, more commonly intended factor. Holding the background fixed at the
  # single mentioned colour (if any) keeps the comparison clean.
  if len(shapes) >= 2:
    fixed_bg = backgrounds[0] if len(backgrounds) == 1 else "white"
    return ContrastPlan(
        condition_a=ProbeCondition(shapes[0], fixed_bg),
        condition_b=ProbeCondition(shapes[1], fixed_bg),
        axis="shape",
    )

  if len(backgrounds) >= 2:
    fixed_shape = shapes[0] if shapes else infer_shape_type(h_text, s_text)
    return ContrastPlan(
        condition_a=ProbeCondition(fixed_shape, backgrounds[0]),
        condition_b=ProbeCondition(fixed_shape, backgrounds[1]),
        axis="background",
    )

  return _absolute_plan(h_text, s_text, shapes, backgrounds)


def _absolute_plan(
    h_text: str,
    s_text: str,
    shapes: list[str],
    backgrounds: list[str],
) -> ContrastPlan:
  """Builds the single-condition plan, deferring to `infer_shape_type`."""
  del shapes  # The legacy inferrer owns this decision; see module docstring.
  return ContrastPlan(
      condition_a=ProbeCondition(
          shape_type=infer_shape_type(h_text, s_text),
          bg_color=backgrounds[0] if len(backgrounds) == 1 else "white",
      )
  )


def task_specification(
    target_model: str,
    q_count: int,
    null_rate: float,
    *,
    verifying: bool = False,
) -> str:
  """Returns the shared orientation prompt describing the symmetry task.

  Args:
    target_model: Name of the model being probed.
    q_count: Exact number of probes the agent must use.
    null_rate: Reference accuracy the proportion test runs against.
    verifying: True for the verification phase (wording becomes "re-testing",
      and steps are not numbered against the discovery pipeline's Step 4/5).

  Returns:
    The orientation prompt text.
  """
  role = (
      "an automated visual symmetry verification system"
      if verifying
      else "an automated visual symmetry capability discovery system"
  )
  verb = "re-testing" if verifying else "testing"
  step_prefix = "" if verifying else "In Step 4, "
  test_step_prefix = "" if verifying else "In Step 5, "

  return f"""\
You are a sub-agent in {role}.
You are {verb} the target model '{target_model}' on visual reflection symmetry completion.

TASK SPECIFICATION:
- The input to the target model is a text and image combination.
- The image contains a pattern on the left side and is empty (background) on the right side.
  The image is created by first creating a left-right vertically symmetric image and deleting the right half.
- The text prompt sent to the target model is FIXED and identical for every probe. It is:
      "{CANONICAL_PROBE_PROMPT}"
  It does NOT disclose what the pattern is, and it cannot be changed.
- Your probing tool is `generate_and_send_symmetry_probes`:
  It generates multimodal probes for a given shape_type, drawn from
  {shape_catalog_sentence()}.
  It also accepts a `bg_color` drawn from {background_palette_sentence()};
  the background is a property of the whole batch, so hold it fixed within a condition.
  It pairs each image with the fixed prompt above, sends them to the target model,
  and stores the generated output images.
- Your evaluation tool is `score_symmetry_completions`:
  {scoring_description()}
- Your statistical test tool is `run_proportion_z_test`:
  Tests the success rate against the null baseline accuracy of {null_rate:.0%} ({null_rate}).

Your workflow for {verb} the assigned Hypothesis (H) with EXACTLY {q_count} probes will be:
1. Receive the assigned Hypothesis H and context.
2. {step_prefix}run `generate_and_send_symmetry_probes` with the shape_type (and bg_color) that specifically tests H.
3. {step_prefix}run `score_symmetry_completions` to algorithmically evaluate the output images.
4. {test_step_prefix}report the P-value from `run_proportion_z_test` using the algorithmic success count.

Please acknowledge that you have received this task specification and wait for the hypothesis and testing instructions. Do not run any tools yet.
"""


def probe_execution_instructions(
    q_count: int,
    h_text: str,
    *,
    verifying: bool = False,
) -> str:
  """Returns the shared execution prompt that runs the probes and scores them.

  Args:
    q_count: Exact number of probes to send.
    h_text: The hypothesis under test.
    verifying: True for the verification phase, which defers the statistical
      test to "the next step" rather than the discovery pipeline's "Step 5".

  Returns:
    The execution prompt text.
  """
  verb = "verify" if verifying else "test"
  next_step = "the next step" if verifying else "Step 5"
  shape_list = ", ".join(f"'{s}'" for s in SUPPORTED_SHAPE_TYPES)
  bg_list = ", ".join(f"'{c}'" for c in BACKGROUND_PALETTE)

  return f"""\
Now use your symmetry tools to {verb} this hypothesis with EXACTLY {q_count} visual probes.

HYPOTHESIS: {h_text}

TASK REQUIREMENTS:
- The input to the target model must be a text and image combination.
- The image contains a pattern in the left side of the image and is empty (background) in the right side.
  The image is created by first creating a left-right vertically symmetric image and deleting the right half.
- The text prompt is FIXED and identical for every probe:
      "{CANONICAL_PROBE_PROMPT}"
  It does NOT disclose what the pattern is, and you cannot change it.
- Scoring is algorithmic. {scoring_description()}

CRITICAL EXECUTION INSTRUCTIONS:
1. Call `generate_and_send_symmetry_probes` with:
   - `shape_type`: choose the family that directly matches your HYPOTHESIS.
     If the hypothesis is about a hard, fine-stroked or many-part pattern, you MUST
     use the specific hard family ({", ".join(f"'{s}'" for s in HARD_SHAPE_TYPES)})
     rather than its easier look-alike -- e.g. use 'complex_chinese', NOT 'chinese_char',
     for high-stroke-count Hanzi.
     Available shape types: [{shape_list}].
   - `bg_color`: if your HYPOTHESIS is about background colour, set it explicitly.
     Otherwise leave it at the default 'white'. Hold it FIXED across the batch.
     Available backgrounds: [{bg_list}].
   - `n_probes`: {q_count}
2. Next, call `score_symmetry_completions()` to algorithmically score the completions.
   Pass {SUCCESS_CRITERION_INSTRUCTION}
3. Do NOT call statistical testing tools (`run_proportion_z_test`) during this step.
   Statistical testing will be performed in {next_step}.
4. Summarize the resulting pass rate, the per-criterion rates, and observations.

Run your tools now."""


def contrastive_execution_briefing(
    h_text: str,
    plan: ContrastPlan,
    n_a: int,
    n_b: int,
) -> str:
  """Announces a two-arm probe batch that the verifier will run itself.

  Contrastive batches are executed deterministically rather than by the agent.
  The single-arm prompt tells the agent to "hold `bg_color` FIXED across the
  batch" and to pick one `shape_type`, which is correct for an absolute claim
  and makes a contrast impossible to express; and the arms must be balanced and
  scored between generations, since generating arm B overwrites arm A's stored
  images. Neither is something to leave to a tool-calling loop.

  Args:
    h_text: The hypothesis under test.
    plan: The resolved two-condition plan.
    n_a: Probes allocated to arm A.
    n_b: Probes allocated to arm B.

  Returns:
    The briefing text.
  """
  if plan.condition_b is None:
    raise ValueError("contrastive_execution_briefing requires two conditions.")
  return f"""\
This hypothesis is CONTRASTIVE: it compares two conditions rather than
measuring one against a fixed reference rate.

HYPOTHESIS: {h_text}

CONTRAST AXIS: {plan.axis}
  Condition A: shape_type='{plan.condition_a.shape_type}', bg_color='{plan.condition_a.bg_color}' ({n_a} probes)
  Condition B: shape_type='{plan.condition_b.shape_type}', bg_color='{plan.condition_b.bg_color}' ({n_b} probes)

I am running both arms for you now, balanced and scored independently, so that
the comparison is not confounded by an unbalanced or overwritten batch. Do NOT
call `generate_and_send_symmetry_probes` or `score_symmetry_completions`
yourself for this hypothesis.

Scoring is algorithmic. {scoring_description()}
"""


def contrastive_stats_prompt(
    h_text: str,
    plan: ContrastPlan,
    n_success_a: int,
    n_a: int,
    n_success_b: int,
    n_b: int,
) -> str:
  """Asks for the two-sample test, supplying the observed counts.

  The counts are given rather than elicited: they were measured algorithmically
  and re-deriving them from the transcript is a chance to get them wrong. The
  agent supplies only the DIRECTION, which is the one part that depends on
  reading the hypothesis.

  Args:
    h_text: The hypothesis under test.
    plan: The resolved two-condition plan.
    n_success_a: Successes observed in arm A.
    n_a: Scored observations in arm A.
    n_success_b: Successes observed in arm B.
    n_b: Scored observations in arm B.

  Returns:
    The statistical testing prompt.
  """
  if plan.condition_b is None:
    raise ValueError("contrastive_stats_prompt requires two conditions.")
  rate_a = (n_success_a / n_a) if n_a else 0.0
  rate_b = (n_success_b / n_b) if n_b else 0.0
  return f"""\
Both arms have been probed and scored. Observed results:

  Condition A ({plan.condition_a.label}): {n_success_a}/{n_a} successes ({rate_a:.1%})
  Condition B ({plan.condition_b.label}): {n_success_b}/{n_b} successes ({rate_b:.1%})

HYPOTHESIS: {h_text}

Call `run_proportion_z_test_2_sample` EXACTLY ONCE with:
  - `n_successes_a`={n_success_a}, `n_obs_a`={n_a}
  - `n_successes_b`={n_success_b}, `n_obs_b`={n_b}
  - `alternative`: the direction the HYPOTHESIS claims, as a statement about
    the SUCCESS RATE of condition A relative to condition B.
      * 'smaller' if the hypothesis says A fails more / performs worse than B.
      * 'larger'  if the hypothesis says A succeeds more / performs better than B.
    Choose the one-sided direction the hypothesis actually states. Do NOT use
    'two-sided' -- the hypothesis is directional.

Do NOT use the 90 percent reference rate here. That rate calibrates an ABSOLUTE
claim about one condition; this hypothesis is about the DIFFERENCE between two
conditions, and testing a difference against 0.90 would answer a question
nobody asked.

Report the test type and the p-value.
"""
