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

"""Statistical test plan models and routing utilities."""

import dataclasses
import enum
from typing import Callable, Literal

AlternativeType = Literal["larger", "smaller", "two-sided"]


class TiePolicy(enum.Enum):
  """How a paired analysis treats pairs whose two arms measured equal.

  Tie handling is not uniform across test families, and getting it wrong is
  silently anti-conservative, so the policy is derived from the pre-registered
  plan rather than chosen once the data is in view.
  """

  # Exclude tied pairs from both numerator and denominator. The standard sign
  # test / McNemar "discordant pairs only" convention, valid when the null is
  # symmetry (0.5) because a tie carries no directional evidence.
  DROP = "drop"

  # Keep tied pairs in the denominator and count them as non-successes. The
  # right choice when the null is an arbitrary rate over the population of
  # items: dropping ties there would shrink the denominator while leaving the
  # numerator alone, inflating the observed rate against a fixed null.
  COUNT_AS_FAILURE = "count_as_failure"

  # Keep tied pairs as-is. Required for continuous paired tests, where a zero
  # difference is real data that pulls the mean toward the null.
  RETAIN = "retain"

  # The analysis is not paired, so tie handling does not apply.
  NOT_APPLICABLE = "not_applicable"


class StatsTool(enum.Enum):
  """Names of the statistical tools available to the agent."""

  PROPORTION_Z_TEST = "run_proportion_z_test"
  PROPORTION_Z_TEST_2_SAMPLE = "run_proportion_z_test_2_sample"
  MCNEMAR = "run_mcnemar_test"
  CONTINUOUS_MEANS = "run_continuous_means_test"
  PAIRED_MEANS = "run_paired_means_test"


def match_test_to_tool(test_name: str) -> StatsTool | None:
  """Matches a test name string to the most specific available StatsTool.

  Uses an explicit if-elif chain to handle priority, checking more specific
  tests (like 2-sample) before generic ones (like 1-sample proportion).

  Args:
    test_name: The name of the test to match.

  Returns:
    The matching StatsTool, or None if no match is found.
  """
  test_name = test_name.lower()

  # McNemar is very specific.
  if "mcnemar" in test_name:
    return StatsTool.MCNEMAR

  # Paired continuous means test. "continuous" is included so that the
  # 'Paired continuous test' pre-registration option routes here rather than
  # falling through to the unpaired CONTINUOUS_MEANS branch below.
  if "paired" in test_name and any(
      k in test_name
      for k in ("means", "t-test", "diff", "student", "continuous")
  ):
    return StatsTool.PAIRED_MEANS

  # Check 2-sample keywords first, as they often overlap with 1-sample names.
  # e.g. "Two-Sample Proportion Test" should match the 2-sample tool.
  if any(k in test_name for k in ("2-sample", "two-sample", "fisher")):
    return StatsTool.PROPORTION_Z_TEST_2_SAMPLE

  # Continuous data tests.
  if any(k in test_name for k in ("continuous", "means", "t-test", "student")):
    return StatsTool.CONTINUOUS_MEANS

  # General proportion/binomial tests (1-sample defaults).
  if any(k in test_name for k in ("z-test", "proportion", "binomial")):
    return StatsTool.PROPORTION_Z_TEST

  return None


# Tools whose test statistic has a discrete null distribution, and whose
# p-values are therefore super-uniform and eligible for Tocher randomization.
_DISCRETE_TOOLS = frozenset({
    StatsTool.PROPORTION_Z_TEST,
    StatsTool.PROPORTION_Z_TEST_2_SAMPLE,
    StatsTool.MCNEMAR,
})

# Tools that consume real-valued measurements and already produce continuous
# p-values.
_CONTINUOUS_TOOLS = frozenset({
    StatsTool.CONTINUOUS_MEANS,
    StatsTool.PAIRED_MEANS,
})


def is_discrete_test(test_name: str | None) -> bool:
  """Reports whether a test name routes to a tool with a discrete null.

  Prefer this over ad-hoc substring matching on the test name. Checking for
  keywords like "paired" directly is unreliable because pairing is orthogonal
  to discreteness: "paired McNemar's test" is discrete while "paired continuous
  test" is not. Routing through `match_test_to_tool` asks the question that
  actually matters -- which tool will run -- and stays consistent with the tool
  the agent is required to invoke.

  Args:
    test_name: A declared or reported test type, e.g. "Binomial test".

  Returns:
    True if the routed tool has a discrete test statistic. Unrecognized test
    names return False, so callers treat them as continuous and skip
    randomization rather than randomizing something they can't identify.
  """
  return match_test_to_tool(test_name or "") in _DISCRETE_TOOLS


def is_continuous_test(test_name: str | None) -> bool:
  """Reports whether a test name routes to a means-style continuous tool.

  Note this is not the negation of `is_discrete_test`: an unrecognized test
  name is neither discrete nor continuous, and both predicates return False.

  Args:
    test_name: A declared or reported test type, e.g. "Continuous means test".

  Returns:
    True if the routed tool is `CONTINUOUS_MEANS` or `PAIRED_MEANS`.
  """
  return match_test_to_tool(test_name or "") in _CONTINUOUS_TOOLS


@dataclasses.dataclass(frozen=True)
class PreRegisteredTestPlan:
  """The parsed pre-registered statistical test plan."""

  test_type: str
  target_metric: str
  null_baseline: float = 0.5
  alternative: AlternativeType = "larger"
  reasoning: str | None = None
  is_paired: bool = False

  @property
  def implies_paired(self) -> bool:
    """Whether this plan requires paired data collection and analysis.

    `is_paired` is the agent's explicit declaration, but a plan can also imply
    pairing through its test type alone: McNemar's test and a paired continuous
    test are undefined on unpaired data. Treating those as paired even when the
    agent left IS_PAIRED False is the conservative reading, and it keeps a
    self-contradictory pre-registration from silently collecting the wrong
    shape of data.
    """
    if self.is_paired:
      return True
    return match_test_to_tool(self.test_type or "") in (
        StatsTool.MCNEMAR,
        StatsTool.PAIRED_MEANS,
    )

  @property
  def unit_of_analysis(self) -> Literal["pair", "response"]:
    """The thing one observation counts as under this plan.

    A paired plan generates two responses per item and reduces them to a single
    measurement, so its sample size is the number of pairs -- half the number
    of responses. Downstream consumers cross-check reported observation counts
    against this.
    """
    return "pair" if self.implies_paired else "response"

  @property
  def tie_policy(self) -> TiePolicy:
    """Tie handling implied by this plan's test type, null, and pairing.

    Derived deterministically so it is fixed before any data is observed. The
    null baseline drives the binary case: a symmetry null (0.5) asks "which
    side wins, given that one does", so ties are uninformative and dropped;
    any other null asks "how often does A beat B across all items", so the
    denominator must stay the full item population and ties count against.
    """
    if not self.implies_paired:
      return TiePolicy.NOT_APPLICABLE

    tool = match_test_to_tool(self.test_type or "")

    # Continuous differences: a zero difference is a real measurement.
    if tool in (StatsTool.PAIRED_MEANS, StatsTool.CONTINUOUS_MEANS):
      return TiePolicy.RETAIN

    # McNemar is defined on discordant pairs, so ties are dropped by
    # construction regardless of what null was declared.
    if tool is StatsTool.MCNEMAR:
      return TiePolicy.DROP

    if tool in (
        StatsTool.PROPORTION_Z_TEST,
        StatsTool.PROPORTION_Z_TEST_2_SAMPLE,
    ):
      is_symmetry_null = (
          self.null_baseline is not None
          and abs(self.null_baseline - 0.5) < 1e-9
      )
      return TiePolicy.DROP if is_symmetry_null else TiePolicy.COUNT_AS_FAILURE

    # Unrecognized test type: retain, which is the choice that leaves the data
    # untouched rather than silently reshaping the denominator.
    return TiePolicy.RETAIN


@dataclasses.dataclass(frozen=True)
class StatsResult:
  """The parsed statistical test result returned by StatsParser."""

  test_type: str
  p_value: float
  group_a_mean: float | None = None
  group_a_obs: int | None = None
  group_b_mean: float | None = None
  group_b_obs: int | None = None
  null_proportion: float | None = None
  alternative: AlternativeType | str | None = None
  reasoning: str | None = None
  sample_std: float | None = None


def guard_p_value_direction(
    res: StatsResult,
    log_fn: Callable[..., None],
    null_value: float = 0.5,
    use_direction_guard: bool = True,
) -> float:
  """Guards against two-sided or wrong-tail p-values when observed effect is in the wrong direction.

  If a test is directional (e.g. alternative="larger" testing group A > null or
  group A > group B), and the observed effect delta is negative (delta < 0),
  then any valid one-sided p-value in the hypothesized direction MUST be >= 0.5.
  If the reported p-value is < 0.5 (indicating a 2-sided test was run or the
  opposite tail was selected), we symmetrically map p -> 1.0 - p to obtain the
  true upper-tail p-value.

  Conversely, if alternative="smaller" and delta > 0, reported p-values < 0.5
  are similarly mapped p -> 1.0 - p.

  Args:
    res: Parsed stats result.
    log_fn: Logging function.
    null_value: Fallback null threshold for 1-sample tests if not in res
      (default 0.5).
    use_direction_guard: Whether to apply the direction guard.

  Returns:
    Directionally adjusted p-value.
  """
  p = res.p_value
  if not use_direction_guard or p < 0 or p > 1:
    return p
  if res.group_a_mean is None:
    return p

  alternative = (res.alternative or "larger").lower()
  if alternative in ("two-sided", "twosided", "two_sided"):
    return p

  if res.group_b_mean is not None:
    # 2-sample test: Delta = group_a_mean - group_b_mean
    delta = res.group_a_mean - res.group_b_mean
  else:
    # 1-sample test: prefer res.null_proportion, fallback to null_value
    actual_null = (
        res.null_proportion if res.null_proportion is not None else null_value
    )
    delta = res.group_a_mean - actual_null

  # For "larger" (H1: effect > 0): delta < 0 means wrong direction
  # For "smaller" (H1: effect < 0): delta > 0 means wrong direction
  is_wrong_direction = (alternative == "larger" and delta < 0) or (
      alternative == "smaller" and delta > 0
  )

  if is_wrong_direction and p < 0.5:
    p_adjusted = 1.0 - p
    log_fn(
        "  [Direction Guard] Observed effect is in wrong direction for"
        f" alternative='{alternative}' (Delta = {delta:+.4f}) but reported"
        f" p-value is {p:.6f}. Adjusting p-value to {p_adjusted:.6f} to enforce"
        " correct one-sided directional testing."
    )
    return p_adjusted

  return p
