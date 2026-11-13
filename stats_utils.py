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

"""Shared statistical utilities for DISCO discovery and verification."""

from collections.abc import Sequence
import dataclasses
import re

import numpy as np
from scipy import stats as sp_stats

from disco.stats import hypothesis_tests as _hypothesis_tests
from disco.stats import plan

# Sentinel p-value to indicate statistical parsing failure.
STATS_PARSING_FAILURE_P_VALUE = -1.0


def extract_null_proportion_from_seed(seed: str, default: float = 0.5) -> float:
  """Extracts the null proportion from a seed instruction if present.

  Parses percentages or decimals such as "75%" -> 0.75, "10 percent" -> 0.10,
  or "0.10" -> 0.10 when followed by "null".

  Args:
    seed: Seed instruction string to search.
    default: Fallback null proportion if none is found in `seed`.

  Returns:
    Extracted null proportion in [0, 1], or `default`.
  """
  match = re.search(
      r"(\d+(?:\.\d+)?)\s*(%|percent)?\s*(?:as the null|null)",
      seed,
      re.IGNORECASE,
  )
  if match:
    try:
      val = float(match.group(1))
      has_percent = bool(match.group(2))
      if has_percent or val > 1.0:
        val /= 100.0
      return val
    except ValueError:
      pass
  return default


def extract_null_proportion_from_hypothesis(
    hypothesis: str, seed: str = "", default: float = 0.5
) -> float:
  """Extracts null proportion from hypothesis text, falling back to seed instruction if absent."""
  # 1. Try explicit null phrases in hypothesis
  prop = extract_null_proportion_from_seed(hypothesis, default=-1.0)
  if prop != -1.0:
    return prop

  # 2. Try matching comparison percentage patterns in hypothesis (e.g. "> 10%",
  # "greater than 10%", "more than 10 percent", "proportion of 0.75").
  #
  # A bare comparator ("> 0.90") must carry an explicit percent marker to count.
  # Without one, a bare decimal after a comparator is far more often a metric
  # threshold than a rate (e.g., visual symmetry hypotheses state their scoring
  # cutoff as "(Ink Mask IoU >= 0.90)" while testing against a 0.90 null).
  # Explicit rate phrases ("proportion of", "rate of", "baseline of") are
  # unambiguous, so for those the percent marker stays optional.
  match = re.search(
      r"(?:"
      r"(?:>=|>|greater than|more than|exceeds|at least)"
      r"\s*(\d+(?:\.\d+)?)\s*(%|percent)"
      r"|"
      r"(?:baseline of|rate of|proportion of)"
      r"\s*(\d+(?:\.\d+)?)\s*(%|percent)?"
      r")",
      hypothesis,
      re.IGNORECASE,
  )
  if match:
    try:
      # Exactly one of the two branches matched; take whichever produced a
      # number, along with its own percent marker.
      raw, pct = (
          (match.group(1), match.group(2))
          if match.group(1) is not None
          else (match.group(3), match.group(4))
      )
      val = float(raw)
      has_percent = bool(pct)
      if has_percent or val > 1.0:
        val /= 100.0
      if 0.0 < val < 1.0:
        return val
    except ValueError:
      pass

  # 3. Fall back to seed instruction if available
  if seed:
    seed_prop = extract_null_proportion_from_seed(seed, default=-1.0)
    if seed_prop != -1.0:
      return seed_prop

  return default


def format_p_value(p: float) -> str:
  """Formats a p-value in scientific notation if 0 < p < 1e-4, else 6 decimal places."""
  if 0.0 < p < 1e-4:
    return f"{p:.4e}"
  return f"{p:.6f}"


get_t_test_p_value = _hypothesis_tests.get_t_test_p_value
compute_binomial_replication_stats = (
    _hypothesis_tests.compute_binomial_replication_stats
)
compute_continuous_means_ci = _hypothesis_tests.compute_continuous_means_ci


@dataclasses.dataclass(frozen=True)
class PairedReduction:
  """Pair-level summary of an A/B-alternating batch of measurements.

  Attributes:
    n_pairs: Number of complete pairs (both arms produced a measurement).
    n_tied: Complete pairs whose two arms were equal within tolerance.
    n_positive: Complete pairs where Group A exceeded Group B.
    n_negative: Complete pairs where Group B exceeded Group A.
    diffs: Group A minus Group B for each complete pair, in pair order.
    n_orphans: Measurements whose partner arm was missing, and which therefore
      contribute to no pair.
  """

  n_pairs: int
  n_tied: int
  n_positive: int
  n_negative: int
  diffs: list[float]
  n_orphans: int

  @property
  def tie_rate(self) -> float:
    """Fraction of complete pairs that were ties, or 0.0 if there are none."""
    return self.n_tied / self.n_pairs if self.n_pairs else 0.0


def reduce_pairs(
    values: Sequence[float],
    indices: Sequence[int] | None = None,
    *,
    eps: float = 0.0,
) -> PairedReduction:
  """Reduces per-response measurements to per-pair difference scores.

  Paired query batches are emitted in strict A/B alternation, so the prompt at
  even index `2k` is Group A of pair `k` and the prompt at odd index `2k + 1`
  is Group B of the same pair. This function inverts that layout.

  `indices` matters whenever a response failed: the surviving measurements are
  compacted, so their position in `values` no longer implies their arm. Pass
  the originating prompt indices (see `StandaloneAgent.last_response_indices`)
  and a pair whose partner dropped out is reported as an orphan instead of
  being silently matched against the wrong response.

  Args:
    values: Measurement per surviving response, in response order.
    indices: Prompt-list index each measurement came from. Defaults to
      `range(len(values))`, which is only correct if nothing was dropped.
    eps: Absolute tolerance below which a difference counts as a tie. Use 0.0
      for integer-valued metrics such as word counts.

  Returns:
    A `PairedReduction` over the complete pairs.

  Raises:
    ValueError: If `indices` is given and its length differs from `values`.
  """
  if indices is None:
    indices = range(len(values))
  elif len(indices) != len(values):
    raise ValueError(
        f"reduce_pairs got {len(values)} values but {len(indices)} indices;"
        " they must be parallel."
    )

  arms_by_pair: dict[int, dict[str, float]] = {}
  for idx, value in zip(indices, values):
    arm = "A" if idx % 2 == 0 else "B"
    arms_by_pair.setdefault(idx // 2, {})[arm] = float(value)

  diffs: list[float] = []
  n_orphans = 0
  for pair_id in sorted(arms_by_pair):
    arms = arms_by_pair[pair_id]
    if "A" in arms and "B" in arms:
      diffs.append(arms["A"] - arms["B"])
    else:
      n_orphans += len(arms)

  return PairedReduction(
      n_pairs=len(diffs),
      n_tied=sum(1 for d in diffs if abs(d) <= eps),
      n_positive=sum(1 for d in diffs if d > eps),
      n_negative=sum(1 for d in diffs if d < -eps),
      diffs=diffs,
      n_orphans=n_orphans,
  )


def count_complete_pairs(indices: Sequence[int] | None) -> int:
  """Counts pairs for which both arms survived, from prompt indices alone.

  Used to cross-check an agent-reported observation count without needing the
  measurements themselves: under strict A/B alternation, pair `k` is complete
  exactly when prompt indices `2k` and `2k + 1` are both present.

  Args:
    indices: Prompt-list indices of the surviving responses, or None.

  Returns:
    The number of complete pairs, or 0 if `indices` is None or empty.
  """
  if not indices:
    return 0
  seen = set(indices)
  return sum(1 for i in seen if i % 2 == 0 and i + 1 in seen)


def maybe_randomize_p_value(
    p_value: float,
    res: plan.StatsResult,
    rng: np.random.Generator,
) -> float:
  """Applies Tocher PIT randomization to make discrete p-values exact U(0,1) under H0.

  For discrete statistical tests (1-sample Binomial test, Fisher's Exact Test,
  McNemar's test), the non-randomized p-value is p = P(T >= t_obs) which is
  super-uniform (conservative). Tocher randomization replaces it with
  p_rand = P(T > t_obs) + U * P(T = t_obs) where U ~ U(0,1), yielding an exact
  uniform null distribution under H0.

  For continuous-means tests (Student's t-test), the p-value is inherently
  continuous and is returned unchanged.

  Args:
    p_value: The non-randomized p-value from the statistical test.
    res: The parsed StatsResult containing test metadata.
    rng: A numpy random generator for the uniform draw.

  Returns:
    The randomized p-value if applicable, otherwise the original p-value.
  """
  test_type = (res.test_type or "").lower()

  # Decide discreteness from the tool the test routes to, not from substrings;
  # pairing says nothing about whether the test statistic is discrete.
  # Unrecognized test types are left alone, since randomizing a test we cannot
  # identify would be worse than returning the conservative p-value.
  if not plan.is_discrete_test(test_type):
    return p_value

  # 1. Two-sample Fisher's exact test.
  is_fisher = "fisher" in test_type or (
      ("2-sample" in test_type or "two-sample" in test_type)
      and "proportion" in test_type
  )
  if is_fisher:
    if (
        res.group_a_obs is None
        or res.group_a_mean is None
        or res.group_b_obs is None
        or res.group_b_mean is None
    ):
      return p_value
    na = res.group_a_obs
    nb = res.group_b_obs
    if na <= 0 or nb <= 0:
      return p_value
    na_s = round(res.group_a_mean * na)
    nb_s = round(res.group_b_mean * nb)
    # Under H0, conditional distribution of successes in Group A is
    # Hypergeometric. M = total items (na + nb), n = total successes
    # (na_s + nb_s), N = draws in A (na).
    p_equal = float(sp_stats.hypergeom.pmf(na_s, na + nb, na_s + nb_s, na))
    p_strict = max(0.0, p_value - p_equal)
    u = float(rng.uniform())
    p_rand = p_strict + u * p_equal
    return float(np.clip(p_rand, 1e-20, 1.0))

  # 2. Paired binary McNemar's test.
  if "mcnemar" in test_type:
    if res.group_a_obs is None or res.group_a_mean is None:
      return p_value
    n_disc = res.group_a_obs
    if n_disc <= 0:
      return p_value
    n_succ = round(res.group_a_mean * n_disc)
    p_equal = float(sp_stats.binom.pmf(n_succ, n_disc, 0.5))
    p_strict = max(0.0, p_value - p_equal)
    u = float(rng.uniform())
    p_rand = p_strict + u * p_equal
    return float(np.clip(p_rand, 1e-20, 1.0))

  # 3. One-sample Binomial / proportion test.
  is_binomial = any(
      k in test_type for k in ("binomial", "exact", "proportion", "z-test")
  )
  if is_binomial:
    if res.group_a_obs is None or res.group_a_mean is None:
      return p_value

    n_obs = res.group_a_obs
    if n_obs <= 0:
      return p_value
    n_succ = round(res.group_a_mean * n_obs)
    null_p = res.null_proportion if res.null_proportion is not None else 0.5

    p_equal = float(sp_stats.binom.pmf(n_succ, n_obs, null_p))
    p_strict = max(0.0, p_value - p_equal)

    u = float(rng.uniform())
    p_rand = p_strict + u * p_equal
    return float(np.clip(p_rand, 1e-20, 1.0))

  return p_value


# Aliases for backwards compatibility with private function names
_extract_null_proportion_from_seed = extract_null_proportion_from_seed
_maybe_randomize_p_value = maybe_randomize_p_value
