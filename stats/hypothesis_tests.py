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

"""Standard statistical hypothesis tests and confidence intervals."""

import numpy as np
from scipy import stats as sp_stats


def get_t_test_p_value(t_stat: float, alternative: str, df: int) -> float:
  """Calculates directional p-value for Student's t-test."""
  if df <= 0:
    return 1.0
  alt = alternative.lower()
  if alt in ("larger", "greater"):
    return float(sp_stats.t.sf(t_stat, df=df))
  elif alt in ("smaller", "less"):
    return float(sp_stats.t.cdf(t_stat, df=df))
  else:
    return float(2.0 * sp_stats.t.sf(abs(t_stat), df=df))


def compute_binomial_replication_stats(
    successes: int,
    total: int,
    null_prob: float = 0.5,
    alpha: float = 0.05,
    alternative: str = "greater",
) -> tuple[float, float, float, float]:
  """Computes exact binomial p-value and exact Clopper-Pearson confidence interval.

  The confidence interval is the exact mathematical inversion of the binomial
  test (Clopper-Pearson), ensuring 100% duality with the p-value:
  p_value <= alpha iff null_prob is outside the confidence interval.

  Args:
    successes: Number of observed successes/failures.
    total: Total number of observations.
    null_prob: Null baseline probability in (0, 1).
    alpha: Significance level for the test (and corresponding (1 - alpha) CI).
    alternative: 'greater'/'larger', 'less'/'smaller', or 'two-sided'.

  Returns:
    (p_value, observed_rate, ci_lower, ci_upper)
  """
  if total <= 0:
    return 1.0, 0.0, 0.0, 0.0

  observed_rate = successes / total
  alt_map = {"larger": "greater", "smaller": "less"}
  mapped_alternative = alt_map.get(alternative, alternative)
  res = sp_stats.binomtest(
      successes, total, p=null_prob, alternative=mapped_alternative
  )
  p_value = float(res.pvalue)

  ci = res.proportion_ci(confidence_level=1.0 - alpha, method="exact")
  ci_lower = float(ci.low)
  ci_upper = float(ci.high)

  return p_value, observed_rate, ci_lower, ci_upper


def compute_continuous_means_ci(
    sample_mean: float,
    sample_std: float,
    n_obs: int,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> tuple[float, float]:
  """Computes Student's t confidence interval for a continuous sample mean.

  Args:
    sample_mean: The observed sample mean.
    sample_std: The sample standard deviation (ddof=1).
    n_obs: Number of observations.
    alpha: Significance level for the CI (coverage 1 - alpha).
    alternative: 'greater'/'larger', 'less'/'smaller', or 'two-sided'.

  Returns:
    (ci_lower, ci_upper)
  """
  if n_obs <= 1 or sample_std <= 0:
    return sample_mean, sample_mean

  df = n_obs - 1
  is_one_sided = alternative in ("greater", "larger", "less", "smaller")
  t_crit = float(
      sp_stats.t.ppf(1.0 - (alpha if is_one_sided else alpha / 2.0), df=df)
  )
  sem = sample_std / np.sqrt(n_obs)
  margin = t_crit * sem

  ci_lower = (
      float("-inf")
      if alternative in ("less", "smaller")
      else float(sample_mean - margin)
  )
  ci_upper = (
      float("inf")
      if alternative in ("greater", "larger")
      else float(sample_mean + margin)
  )
  return ci_lower, ci_upper
