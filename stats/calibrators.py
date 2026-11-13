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

"""Calibrators for converting p-values to e-values and thresholding."""

import math
import numpy as np
from scipy import special


def p_to_e(p_value: float, kappa: float = 0.5) -> float:
  """Converts a p-value to an e-value using the p-to-e calibrator.

  Formula: e = kappa * p^(kappa - 1)

  Args:
    p_value: The p-value to convert. Must be between 0 and 1.
    kappa: Calibrator parameter in (0, 1). Default is 0.5.

  Returns:
    The e-value.
  """
  if p_value < 0 or p_value > 1:
    raise ValueError(f"p-value must be between 0 and 1 (got {p_value})")
  if kappa <= 0 or kappa >= 1:
    raise ValueError(f"kappa must be between 0 and 1 exclusive (got {kappa})")

  p_value = max(p_value, 1e-20)

  return kappa * (p_value ** (kappa - 1))


def get_threshold_from_alpha(alpha: float) -> float:
  """Calculates the e-value threshold for a given alpha level.

  Args:
    alpha: The significance level for the e-value.

  Returns:
    The e-value threshold.
  """
  return 1.0 / alpha


def calibrated_evalue_survival(
    x: float | np.ndarray, kappa: float, m: int
) -> float | np.ndarray:
  """Computes P(M^g >= x) for a product of m calibrated e-values.

  M^g = prod_{i=1}^m kappa * p_i^{kappa-1} where p_i ~ U(0,1).
  Uses the Gamma distribution representation: M^g = kappa^m * exp((1-kappa)*G)
  where G ~ Gamma(m, 1).

  Supports both scalar and numpy array inputs for x. Uses
  scipy.special.gammaincc for efficient compiled evaluation of the
  regularized upper incomplete gamma function.

  Args:
    x: Threshold value (scalar or array).
    kappa: Calibrator parameter in (0, 1).
    m: Number of e-values in the product.

  Returns:
    P(M^g >= x), same shape as x.
  """
  x = np.asarray(x, dtype=float)
  scalar_input = x.ndim == 0
  x = np.atleast_1d(x)

  log_ratio = np.where(
      x > 0,
      np.log(np.maximum(x, 1e-300)) - m * math.log(kappa),
      -1.0,
  )
  c = log_ratio / (1.0 - kappa)

  # P(Gamma(m,1) >= c) = gammaincc(m, c) for c > 0.
  # When c <= 0: P = 1.0 (Gamma(m,1) >= 0 always).
  result = np.where(
      (x <= 0) | (log_ratio <= 0),
      1.0,
      special.gammaincc(m, np.maximum(c, 0.0)),
  )

  if scalar_input:
    return float(result[0])
  return result
