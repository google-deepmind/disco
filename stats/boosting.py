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

"""Boosting factor and staircase truncation computation for e-values."""

import functools
import math
import typing

import numpy as np
from scipy import optimize

from disco.stats import calibrators


@typing.overload
def staircase_t(x: float, k: int) -> float:
  ...


@typing.overload
def staircase_t(x: np.ndarray, k: int) -> np.ndarray:
  ...


def staircase_t(x: float | np.ndarray, k: int) -> float | np.ndarray:
  """Wang-Ramdas staircase truncation function.

  T_k(x) = k / ceil(k/x) * 1(x >= 1).

  Args:
    x: Input value(s). Scalar or array.
    k: Staircase resolution parameter (number of hypotheses).

  Returns:
    T_k(x), same shape as input.
  """
  x = np.asarray(x, dtype=float)
  scalar_input = x.ndim == 0
  x = np.atleast_1d(x)

  result = np.where(
      x >= 1.0,
      k / np.ceil(k / np.maximum(x, 1e-300)),
      0.0,
  )

  if scalar_input:
    return float(result[0])
  return result


def _boost_expectation(b: float, alpha: float, kappa: float, m: int) -> float:
  """Computes expectation of truncated boosted product of m calibrated e-values.

  Computes E[alpha * b * M^g * 1(alpha * b * M^g >= 1)], i.e., using the
  simplified truncation T_hat(x) = x * 1(x >= 1).

  Args:
    b: Boost factor (>= 1).
    alpha: Significance level.
    kappa: Calibrator parameter.
    m: Number of probes (product length).

  Returns:
    Expected value of the truncated boosted e-value product.
  """
  term_arg = alpha * b * (kappa**m)
  if term_arg >= 1.0 or term_arg <= 0.0:
    # Sentinel: the formula below requires term_arg < 1 (so that the
    # integration lower bound c = log(1/term_arg)/(1-kappa) > 0).
    # When term_arg >= 1, the true expectation is alpha*b (finite), but we
    # return inf as a signal to compute_boost_factor's brentq that the
    # expectation exceeds alpha at this b. Do NOT use this function for
    # reporting outside of root-finding.
    return float("inf")
  log_term = math.log(1.0 / term_arg)
  poly_sum = 0.0
  term = 1.0
  for j in range(m):
    poly_sum += term
    term *= log_term / (j + 1)

  coef = (term_arg ** (1.0 / (1.0 - kappa))) / ((1.0 - kappa) ** m)
  return coef * poly_sum


def _boost_expectation_exact_t(
    b: float, alpha: float, kappa: float, m: int, k_total: int
) -> float:
  """Computes E[T(alpha * b * M^g)] using the paper's exact staircase T.

  T(x) = K / ceil(K/x) * 1(x >= 1), which truncates x down to the nearest
  value in {K/k : k = 1, ..., K} union {0}.

  E[T(alpha*b*M)] = sum_{k=1}^{K} (K/k) * P(alpha*b*M in [K/k, K/(k-1)))

  Fully vectorized using numpy + scipy.special.gammaincc for O(K) compiled
  operations instead of O(K*m) pure-Python float ops.

  Args:
    b: Boost factor (>= 1).
    alpha: Significance level.
    kappa: Calibrator parameter.
    m: Number of probes (product length).
    k_total: Total number of hypotheses K.

  Returns:
    E[T(alpha * b * M^g)].
  """
  k_arr = np.arange(1, k_total + 1, dtype=float)

  # Staircase bins are indexed k=1..K where lower_m[k] = K/(k*alpha*b).
  # Since lower_m is DECREASING in k (largest threshold at k=1, smallest
  # at k=K), s_lower is INCREASING in k. The probability of falling in bin k is
  # P(M >= lower_m[k]) - P(M >= lower_m[k-1]) = s_lower[k] - s_lower[k-1].
  lower_m = k_total / (k_arr * alpha * b)
  s_lower = np.asarray(
      calibrators.calibrated_evalue_survival(lower_m, kappa, m)
  )

  # s_upper[0] = 0 because P(M >= inf) = 0 (no upper boundary for k=1).
  # s_upper[k] = s_lower[k-1] since upper boundary of bin k equals the lower
  # boundary of bin k-1.
  s_upper = np.zeros(k_total)
  s_upper[1:] = s_lower[:-1]

  prob_in_bin = s_lower - s_upper
  weights = k_total / k_arr
  return float(np.sum(weights * prob_in_bin))


@functools.lru_cache(maxsize=4096)
def compute_boost_factor(
    alpha: float,
    kappa: float,
    m: int,
    k_total: int,
    use_exact_t: bool = True,
) -> float:
  """Computes per-hypothesis boosting factor b >= 1 via Brent's root-finding.

  Args:
    alpha: Significance level.
    kappa: Calibrator parameter.
    m: Number of probes.
    k_total: Total number of hypotheses K.
    use_exact_t: If True, use the paper's exact staircase T function. If False,
      use the simplified T_hat(x) = x * 1(x >= 1).

  Returns:
    Boost factor b >= 1.

  Raises:
    RuntimeError: If expectation evaluation or root-finding fails.
  """
  if m <= 0 or k_total <= 0:
    return 1.0

  if use_exact_t:
    expect_fn = lambda b: _boost_expectation_exact_t(
        b, alpha, kappa, m, k_total
    )
  else:
    expect_fn = lambda b: _boost_expectation(b, alpha, kappa, m)

  # Test b = 1.0
  try:
    exp_1 = expect_fn(1.0)
  except (ValueError, OverflowError, ZeroDivisionError) as e:
    raise RuntimeError(
        f"compute_boost_factor: expect_fn(1.0) raised {type(e).__name__} with"
        f" alpha={alpha}, kappa={kappa}, m={m}, k_total={k_total}"
    ) from e

  if math.isnan(exp_1):
    raise RuntimeError(
        "compute_boost_factor: expect_fn(1.0) returned NaN with"
        f" alpha={alpha}, kappa={kappa}, m={m}, k_total={k_total}"
    )

  if exp_1 > alpha:
    return 1.0

  denom = alpha * (kappa**m)
  if denom == 0.0:
    b_max = 1e12
  else:
    b_max = min((1.0 - 1e-9) / denom, 1e12)
  if b_max <= 1.0:
    return 1.0

  try:
    b_star = optimize.brentq(
        lambda b: expect_fn(b) - alpha,
        1.0,
        b_max,
        xtol=1e-5,
    )
    return max(1.0, float(b_star))
  except ValueError:
    # f(1) and f(b_max) have same sign. Since we verified f(1) <= 0,
    # this means f(b_max) < 0 too — the expectation is below alpha even
    # at b_max, so the full boost is valid.
    return float(b_max)
  except Exception as e:
    raise RuntimeError(
        "compute_boost_factor: brentq failed unexpectedly with"
        f" alpha={alpha}, kappa={kappa}, m={m}, k_total={k_total},"
        f" b_max={b_max}, exp_at_1={exp_1:.6g}"
    ) from e


def boost_probe_count(n_g: int, n_max: int) -> int:
  """Returns the probe count m that boosting and phi use for a hypothesis.

  The null tables and the boost certificate are only defined for
  1 <= m <= n_max, so every DISCO site has to clip m the same way before
  computing a boost factor.

  Args:
    n_g: Observed probe count.
    n_max: The run's probe cap.
  """
  return min(max(1, int(n_g)), int(n_max))


def generate_k_grid(k_min: int, k_max: int, n_points: int = 5) -> list[int]:
  """Generates a geometric grid of K values for sweeping the unconditional bound.

  The grid is spaced so consecutive values have a constant ratio, ensuring
  equal relative change in staircase resolution. The staircase T_K has
  resolution proportional to 1/K, so a geometric grid samples uniformly in
  log-resolution space.

  Args:
    k_min: Minimum K (e.g. initial_active_count).
    k_max: Maximum K (e.g. budget / chunk_size).
    n_points: Number of grid points (default 5).

  Returns:
    Sorted, deduplicated list of K values including k_min and k_max.
  """
  if k_min >= k_max or n_points <= 1:
    return sorted(set([k_min, k_max]))
  r = (k_max / k_min) ** (1.0 / (n_points - 1))
  grid = set()
  for i in range(n_points):
    grid.add(max(k_min, min(k_max, round(k_min * r**i))))
  grid.add(k_min)
  grid.add(k_max)
  return sorted(grid)
