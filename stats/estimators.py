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

"""Statistical point and interval estimators (BCa bootstrap, Student-t, Storey)."""

from collections.abc import Mapping, Sequence
import dataclasses
import math
from typing import Any
import numpy as np
from scipy import stats


@dataclasses.dataclass
class MetricEstimate:
  """Point estimate and confidence interval for a metric."""

  mean: float
  ci_low: float
  ci_high: float
  ci_valid: bool
  ci_method: (
      str  # "bca", "bootstrap_percentile", "point", "undefined", "student_t"
  )
  # Standard error of the mean across runs, used to propagate within-task
  # uncertainty into cross-task averages. NaN when undefined (n < 2).
  se: float = float("nan")


def compute_bca_percentiles(
    sample: np.ndarray,
    draws: np.ndarray,
    point: float,
    confidence: float,
) -> tuple[float, float] | None:
  """Computes BCa-adjusted percentile levels in [0, 100].

  Args:
    sample: Original 1D array of sample observations.
    draws: 1D array of bootstrap resample means.
    point: Observed sample mean.
    confidence: Two-sided confidence level in (0, 1).

  Returns:
    Tuple of (low_percentile, high_percentile) in [0, 100], or None if
    corrections are undefined (e.g. singular jackknife or no spread in
    resamples).
  """
  n_draws = draws.size
  below = float(np.count_nonzero(draws < point))
  ties = float(np.count_nonzero(draws == point))
  # Mid-rank proportion to avoid reading pile-up at observed mean as bias
  proportion = (below + 0.5 * ties) / n_draws
  if not (0.0 < proportion < 1.0):
    return None
  z0 = float(stats.norm.ppf(proportion))

  # Jackknife acceleration estimate: (x_i - mean) / (n - 1)
  deviations = sample - point
  sum_sq = float(np.sum(deviations**2))
  if sum_sq <= 0.0:
    return None
  accel = float(np.sum(deviations**3)) / (6.0 * sum_sq**1.5)

  tail = (1.0 - confidence) / 2.0
  levels = []
  for z_alpha in (stats.norm.ppf(tail), stats.norm.ppf(1.0 - tail)):
    adjusted = z0 + z_alpha
    denom = 1.0 - accel * adjusted
    if denom <= 0.0:
      return None
    levels.append(100.0 * float(stats.norm.cdf(z0 + adjusted / denom)))

  low, high = levels[0], levels[1]
  if not math.isfinite(low) or not math.isfinite(high) or low > high:
    return None
  return low, high


def bca_bootstrap_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
    val_min: float | None = None,
    val_max: float | None = None,
) -> MetricEstimate:
  """Computes sample mean and two-sided BCa bootstrap confidence interval.

  Args:
    values: Sequence of numeric values across runs.
    confidence: Two-sided confidence level (default 0.95 for 95% CI).
    resamples: Number of bootstrap draws (default 10,000).
    seed: Random seed for deterministic reproducibility.
    val_min: Optional lower bound to clamp CI (e.g. 0.0 for counts and FDR).
    val_max: Optional upper bound to clamp CI (e.g. 1.0 for FDR).

  Returns:
    MetricEstimate with mean, ci_low, ci_high, ci_valid, and ci_method.
  """
  n = len(values)
  if n == 0:
    return MetricEstimate(
        mean=float("nan"),
        ci_low=float("nan"),
        ci_high=float("nan"),
        ci_valid=False,
        ci_method="undefined",
    )

  sample = np.asarray(values, dtype=float)
  mean = float(np.mean(sample))
  se = float(np.std(sample, ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
  if n == 1:
    return MetricEstimate(
        mean=mean,
        ci_low=float("nan"),
        ci_high=float("nan"),
        ci_valid=False,
        ci_method="undefined",
    )

  std = float(np.std(sample, ddof=1))
  if std == 0.0:
    # All runs identical (e.g. all zero). Point interval.
    lo = mean if val_min is None else max(val_min, mean)
    hi = mean if val_max is None else min(val_max, mean)
    return MetricEstimate(
        mean=mean,
        ci_low=lo,
        ci_high=hi,
        ci_valid=True,
        ci_method="point",
        se=se,
    )

  rng = np.random.default_rng(seed)
  draws = rng.choice(sample, size=(resamples, n), replace=True).mean(axis=1)

  levels = compute_bca_percentiles(sample, draws, mean, confidence)
  if levels is None:
    tail = 100.0 * (1.0 - confidence) / 2.0
    levels = (tail, 100.0 - tail)
    method = "bootstrap_percentile"
  else:
    method = "bca"

  ci_low, ci_high = np.percentile(draws, list(levels))
  ci_low = float(ci_low)
  ci_high = float(ci_high)

  if val_min is not None:
    ci_low = max(val_min, ci_low)
    ci_high = max(val_min, ci_high)
  if val_max is not None:
    ci_low = min(val_max, ci_low)
    ci_high = min(val_max, ci_high)

  return MetricEstimate(
      mean=mean,
      ci_low=ci_low,
      ci_high=ci_high,
      ci_valid=True,
      ci_method=method,
      se=se,
  )


def storey_pi0_estimator(
    w_sum: float,
    r_total: int,
    lam: float,
    plus_one: bool = True,
) -> float | None:
  """Storey pi0 / FDP estimator: (plus_one + sum(w_i)) / (r_total * (1 - lambda))."""
  if r_total <= 0 or not (0.0 < lam < 1.0):
    return None
  numerator = (1.0 + w_sum) if plus_one else w_sum
  denominator = r_total * (1.0 - lam)
  raw_est = numerator / denominator
  return float(np.clip(raw_est, 0.0, 1.0))


def _extract_valid_p_values(
    records_or_p_values: Sequence[float | Mapping[str, Any]],
) -> list[float]:
  """Extracts valid p-values from a sequence of floats or verification record dicts."""
  p_values: list[float] = []
  for item in records_or_p_values:
    if isinstance(item, Mapping):
      if item.get("status") not in ("OK", None):
        continue
      p_val = item.get("p_value")
      if p_val is not None:
        p_values.append(float(p_val))
    else:
      p_values.append(float(item))
  return p_values


def compute_storey_fdr(
    records_or_p_values: Sequence[float | Mapping[str, Any]],
    lam: float = 0.5,
    plus_one: bool = True,
) -> float | None:
  """Computes Storey's pi0 / FDP estimate across verified discovery records or p-values."""
  p_values = _extract_valid_p_values(records_or_p_values)
  r_total = len(p_values)
  if r_total <= 0:
    return None
  w_sum = float(sum(1 for p in p_values if p > lam))
  return storey_pi0_estimator(
      w_sum=w_sum, r_total=r_total, lam=lam, plus_one=plus_one
  )


def compute_thresholded_fdr(
    records_or_p_values: Sequence[float | Mapping[str, Any]],
    alpha: float = 0.05,
) -> float | None:
  """Computes the thresholded empirical FDR (fraction with p > alpha) across records or p-values."""
  p_values = _extract_valid_p_values(records_or_p_values)
  r_total = len(p_values)
  if r_total <= 0:
    return None
  n_unverified = sum(1 for p in p_values if p > alpha)
  return float(n_unverified / r_total)
