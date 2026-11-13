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

"""Bandit environments for evaluating FDR control and recall in DISCO.

Provides:
- BernoulliBanditEnv: Multi-armed bandit with independent null and alternative
arms.
- ContextualBanditEnv: Contextual bandit where arms have d-dimensional feature
vectors.
"""

import abc
import dataclasses
import numpy as np
from scipy import stats


@dataclasses.dataclass
class ArmInfo:
  """Metadata and ground truth properties of a bandit arm."""

  arm_id: str
  is_alternative: bool
  true_rate: float
  feature: np.ndarray | None = None
  cluster_id: int | None = None


@dataclasses.dataclass
class QueryResult:
  """Result of querying a bandit arm with a batch of samples."""

  arm_id: str
  queries: int
  failures: int
  p_value: float
  is_alternative: bool


class BanditEnv(abc.ABC):
  """Abstract base class for bandit environments."""

  def __init__(self, baseline_rate_logit: float = -4.6, seed: int = 42):
    self.baseline_rate_logit = baseline_rate_logit
    self.baseline_rate = 1.0 / (1.0 + np.exp(-baseline_rate_logit))
    self.rng = np.random.default_rng(seed)
    self.arms: dict[str, ArmInfo] = {}

  @abc.abstractmethod
  def get_or_create_arm(self, arm_id: str) -> ArmInfo:
    """Returns or creates the arm with the specified ID."""

  @abc.abstractmethod
  def generate_candidate_pool(
      self, size: int, exclude_ids: set[str] | None = None
  ) -> list[ArmInfo]:
    """Generates a candidate pool of unvisited arms."""

  def pre_generate_pool(self, num_rounds: int) -> tuple[int, dict[int, int]]:
    """Pre-generates a fixed pool of num_rounds arms.

    Must be called before any simulation so that the finite pool is
    materialized and ``total_true_positives`` is known.

    Args:
      num_rounds: Number of candidate arms to pre-generate in the pool.

    Returns:
      A tuple of (total_true_positives, cluster_true_positives_dict).
    """
    pool = self.generate_candidate_pool(num_rounds)
    ttp = sum(1 for a in pool if a.is_alternative)
    cluster_ttp: dict[int, int] = {}
    for a in pool:
      if a.is_alternative and a.cluster_id is not None:
        cluster_ttp[a.cluster_id] = cluster_ttp.get(a.cluster_id, 0) + 1
    return ttp, cluster_ttp

  def sample_queries(
      self, arm_id: str, count: int, randomized_p: bool = True
  ) -> QueryResult:
    """Queries an arm with count samples and computes one-sided p-value."""
    arm = self.get_or_create_arm(arm_id)
    failures = int(self.rng.binomial(count, arm.true_rate))

    # Compute one-sided Binomial p-value
    # Under H0, randomized p-value is exact Uniform(0, 1)
    p_strictly_greater = 1.0 - float(
        stats.binom.cdf(failures, count, self.baseline_rate)
    )
    p_equal = float(stats.binom.pmf(failures, count, self.baseline_rate))

    if randomized_p:
      u = float(self.rng.uniform(0.0, 1.0))
      p_val = p_strictly_greater + u * p_equal
    else:
      p_val = p_strictly_greater + p_equal

    p_val = float(np.clip(p_val, 1e-20, 1.0))

    return QueryResult(
        arm_id=arm_id,
        queries=count,
        failures=failures,
        p_value=p_val,
        is_alternative=arm.is_alternative,
    )


class BernoulliBanditEnv(BanditEnv):
  """Multi-armed Bernoulli bandit with independent null and alternative arms.

  Each arm is drawn from an infinite pool:
  - With probability 1 - pi_1: arm is Null (true_rate = baseline_rate).
  - With probability pi_1: arm is Alternative (true_rate = alternative_rate >
  baseline_rate).
  """

  def __init__(
      self,
      pi_1: float = 0.10,
      baseline_rate_logit: float = -4.6,
      alternative_rate: float = 0.20,
      seed: int = 42,
  ):
    super().__init__(baseline_rate_logit=baseline_rate_logit, seed=seed)
    self.pi_1 = pi_1
    self.alternative_rate = alternative_rate
    self._next_arm_counter = 0

  def get_or_create_arm(self, arm_id: str) -> ArmInfo:
    """Returns an existing arm or samples a new Bernoulli arm."""
    if arm_id in self.arms:
      return self.arms[arm_id]

    # Sample whether arm is alternative
    is_alt = bool(self.rng.random() < self.pi_1)
    rate = self.alternative_rate if is_alt else self.baseline_rate
    arm = ArmInfo(
        arm_id=arm_id,
        is_alternative=is_alt,
        true_rate=rate,
        feature=None,
        cluster_id=0 if is_alt else None,
    )
    self.arms[arm_id] = arm
    return arm

  def generate_candidate_pool(
      self, size: int, exclude_ids: set[str] | None = None
  ) -> list[ArmInfo]:
    """Generates a pool of new candidate Bernoulli arms."""
    exclude = exclude_ids or set()
    pool = []
    while len(pool) < size:
      arm_id = f"arm_{self._next_arm_counter}"
      self._next_arm_counter += 1
      if arm_id not in exclude:
        pool.append(self.get_or_create_arm(arm_id))
    return pool


class ContextualBanditEnv(BanditEnv):
  """Contextual bandit where each arm has a d-dimensional feature vector.

  Vulnerability is clustered in feature space:
  theta(x) = sigma(z_0 + sum_{c in vuln} w_c * exp(-||x - mu_c||^2 / (2 *
  var_c)))
  """

  def __init__(
      self,
      dimension: int = 3,
      num_vulnerable_clusters: int = 1,
      baseline_rate_logit: float = -4.6,
      cluster_weight_min: float = 7.0,
      cluster_weight_max: float = 10.0,
      cluster_var_min: float = 0.05,
      cluster_var_max: float = 0.15,
      seed: int = 42,
  ):
    super().__init__(baseline_rate_logit=baseline_rate_logit, seed=seed)
    self.dimension = dimension
    self.num_vulnerable_clusters = num_vulnerable_clusters

    # Generate cluster centers and parameters for vulnerable clusters
    if num_vulnerable_clusters > 0:
      self.cluster_centers = self.rng.uniform(
          -0.8, 0.8, size=(num_vulnerable_clusters, dimension)
      )
      self.cluster_variances = self.rng.uniform(
          cluster_var_min, cluster_var_max, size=num_vulnerable_clusters
      )
      self.cluster_weights = self.rng.uniform(
          cluster_weight_min, cluster_weight_max, size=num_vulnerable_clusters
      )
    else:
      self.cluster_centers = np.empty((0, dimension))
      self.cluster_variances = np.empty((0,))
      self.cluster_weights = np.empty((0,))

    self._next_arm_counter = 0

  def _compute_failure_probability(
      self, x: np.ndarray
  ) -> tuple[float, bool, int | None]:
    """Computes failure rate theta(x) and whether x is in a vulnerable cluster."""
    logit = self.baseline_rate_logit
    is_vuln = False
    best_c_idx = None
    min_dist_sq = float("inf")
    for c_idx in range(self.num_vulnerable_clusters):
      diff = x - self.cluster_centers[c_idx]
      dist_sq = float(np.sum(diff * diff))
      var = self.cluster_variances[c_idx]
      boost = float(
          self.cluster_weights[c_idx] * np.exp(-dist_sq / (2.0 * var))
      )
      logit += boost
      if dist_sq <= 3.0 * var:  # Within 3-sigma radius
        is_vuln = True
      if dist_sq < min_dist_sq:
        min_dist_sq = dist_sq
        best_c_idx = c_idx

    rate = float(1.0 / (1.0 + np.exp(-logit)))
    # If failure rate is meaningfully elevated above baseline, mark as
    # alternative.
    if rate >= self.baseline_rate + 0.001:
      is_vuln = True
    cluster_id = (
        best_c_idx if (is_vuln and self.num_vulnerable_clusters > 0) else None
    )
    return rate, is_vuln, cluster_id

  def get_or_create_arm(
      self, arm_id: str, feature: np.ndarray | None = None
  ) -> ArmInfo:
    """Returns an existing arm or creates a new contextual arm with features."""
    if arm_id in self.arms:
      return self.arms[arm_id]

    if feature is None:
      feature = self.rng.uniform(-1.0, 1.0, size=self.dimension)

    rate, is_alt, cluster_id = self._compute_failure_probability(feature)
    arm = ArmInfo(
        arm_id=arm_id,
        is_alternative=is_alt,
        true_rate=rate,
        feature=feature,
        cluster_id=cluster_id,
    )
    self.arms[arm_id] = arm
    return arm

  def generate_candidate_pool(
      self, size: int, exclude_ids: set[str] | None = None
  ) -> list[ArmInfo]:
    """Generates a pool of new candidate contextual arms."""
    exclude = exclude_ids or set()
    pool = []
    while len(pool) < size:
      arm_id = f"arm_ctx_{self._next_arm_counter}"
      self._next_arm_counter += 1
      if arm_id not in exclude:
        feat = self.rng.uniform(-1.0, 1.0, size=self.dimension)
        pool.append(self.get_or_create_arm(arm_id, feature=feat))
    return pool
