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

"""Multiple testing procedures (e-BH, BH, BY) for FDR control."""

from collections.abc import Sequence
from typing import Any

from absl import logging
import numpy as np

from disco.stats import boosting
from disco.stats import calibrators


class DegeneratePValuesError(Exception):
  """Raised when p-values are degenerate (e.g. all unpopulated defaults)."""


def _ebh_k_star(
    sorted_items: Sequence[tuple[Any, float]],
    k_total: int,
    alpha: float,
) -> int:
  """Finds k* = max{k : k * e_(k) / k_total >= 1 / alpha} for descending e-values."""
  k_star = 0
  for k_idx, (_, e_val) in enumerate(sorted_items, start=1):
    if (k_idx * e_val / k_total) >= (1.0 / alpha):
      k_star = k_idx
  return k_star


def compute_harmonic_number(n: int) -> float:
  """Computes harmonic number c(n) = sum_{j=1}^n 1/j."""
  if n <= 0:
    return 1.0
  return sum(1.0 / j for j in range(1, n + 1))


def compute_bh_procedure(
    p_values: Sequence[float],
    alpha: float,
    m_total: int,
) -> dict[str, Any]:
  """Applies Benjamini-Hochberg (1995) step-up FDR control procedure.

  Args:
    p_values: Sequence of p-values in [0, 1].
    alpha: FDR significance level in (0, 1).
    m_total: Family size denominator m (>= len(p_values)).

  Returns:
    Dict containing:
      - 'rejected_indices': Set of 0-based indices into p_values that reject.
      - 'rejection_threshold': Maximum p-value threshold (tau_BH) for rejection.
      - 'k_star': Number of rejections (k*).
      - 'q_values': List of step-up adjusted p-values matching input order.
      - 'ranks': List of 1-based ranks (ascending by p-value).
  """
  n = len(p_values)
  if n == 0 or m_total <= 0 or alpha <= 0.0:
    return {
        "rejected_indices": set(),
        "rejection_threshold": 0.0,
        "k_star": 0,
        "q_values": [],
        "ranks": [],
    }

  # Sort by p-value ascending, keeping track of original indices
  indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])

  # Find k* = max{k : p_(k) <= (k / m) * alpha}
  k_star = 0
  for k, (_, p_val) in enumerate(indexed_p, start=1):
    cutoff = (k / m_total) * alpha
    if p_val <= cutoff:
      k_star = k

  threshold = (k_star / m_total) * alpha if k_star > 0 else 0.0
  rejected_indices = {orig_idx for orig_idx, _ in indexed_p[:k_star]}

  # Compute adjusted p-values (q-values) via backward step-up:
  # q_(k) = min_{j >= k} min(1.0, (m / j) * p_(j))
  sorted_q = [0.0] * n
  cum_min = 1.0
  for rank_rev, (_, p_val) in enumerate(reversed(indexed_p), start=1):
    k = n - rank_rev + 1
    raw_q = (m_total / k) * p_val
    cum_min = min(cum_min, raw_q)
    sorted_q[k - 1] = min(1.0, cum_min)

  # Map back to original indices
  q_values = [0.0] * n
  ranks = [0] * n
  for rank, (orig_idx, _) in enumerate(indexed_p, start=1):
    q_values[orig_idx] = sorted_q[rank - 1]
    ranks[orig_idx] = rank

  return {
      "rejected_indices": rejected_indices,
      "rejection_threshold": threshold,
      "k_star": k_star,
      "q_values": q_values,
      "ranks": ranks,
  }


def compute_by_procedure(
    p_values: Sequence[float],
    alpha: float,
    m_total: int,
) -> dict[str, Any]:
  """Applies Benjamini-Yekutieli (2001) FDR control procedure under arbitrary dependence.

  Scales the significance level by the harmonic penalty c(m) = sum_{j=1}^m 1/j.

  Args:
    p_values: Sequence of p-values in [0, 1].
    alpha: FDR significance level in (0, 1).
    m_total: Family size denominator m (>= len(p_values)).

  Returns:
    Dict containing:
      - 'rejected_indices': Set of 0-based indices into p_values that reject.
      - 'rejection_threshold': Maximum p-value threshold (tau_BY) for rejection.
      - 'k_star': Number of rejections.
      - 'q_values': List of BY adjusted p-values matching input order.
      - 'harmonic_c': Harmonic constant c(m).
  """
  n = len(p_values)
  c_m = compute_harmonic_number(m_total)
  if n == 0 or m_total <= 0 or alpha <= 0.0:
    return {
        "rejected_indices": set(),
        "rejection_threshold": 0.0,
        "k_star": 0,
        "q_values": [],
        "harmonic_c": c_m,
    }

  effective_alpha = alpha / c_m
  bh_res = compute_bh_procedure(
      p_values=p_values, alpha=effective_alpha, m_total=m_total
  )

  # BY q-values are min(1.0, bh_q * c_m)
  by_q_values = [min(1.0, q * c_m) for q in bh_res["q_values"]]

  return {
      "rejected_indices": bh_res["rejected_indices"],
      "rejection_threshold": bh_res["rejection_threshold"],
      "k_star": bh_res["k_star"],
      "q_values": by_q_values,
      "harmonic_c": c_m,
  }


def compute_ebh_unboosted(
    p_values: Sequence[float],
    alpha: float,
    kappa: float,
    m_total: int,
) -> dict[str, Any]:
  """Applies un-boosted e-BH procedure (Wang & Ramdas 2022) with p-to-e calibrator.

  Args:
    p_values: Sequence of p-values in [0, 1].
    alpha: Significance level in (0, 1).
    kappa: Calibrator exponent in (0, 1).
    m_total: Total hypothesis count K.

  Returns:
    Dict containing:
      - 'rejected_indices': Set of 0-based indices into p_values that reject.
      - 'rejection_threshold': Minimum e-value threshold for rejection.
      - 'equivalent_p_cutoff': Equivalent p-value cutoff (p <= cutoff rejects).
      - 'k_star': Number of rejections.
      - 'e_values': Calibrated e-values in original input order.
      - 'ranks': 1-based ranks (descending by e-value).
  """
  n = len(p_values)
  if n == 0 or m_total <= 0 or alpha <= 0.0:
    return {
        "rejected_indices": set(),
        "rejection_threshold": float("inf"),
        "equivalent_p_cutoff": 0.0,
        "k_star": 0,
        "e_values": [],
        "ranks": [],
    }

  e_values = [calibrators.p_to_e(p, kappa=kappa) for p in p_values]

  # Sort by e-value descending
  indexed_e = sorted(enumerate(e_values), key=lambda x: x[1], reverse=True)

  k_star = _ebh_k_star(indexed_e, m_total, alpha)

  if k_star > 0:
    threshold = m_total / (alpha * k_star)
    # p_cutoff = (k* * alpha * kappa / m)^(1 / (1 - kappa))
    p_cutoff = (k_star * alpha * kappa / m_total) ** (1.0 / (1.0 - kappa))
    rejected_indices = {orig_idx for orig_idx, _ in indexed_e[:k_star]}
  else:
    threshold = float("inf")
    p_cutoff = 0.0
    rejected_indices = set()

  ranks = [0] * n
  for rank, (orig_idx, _) in enumerate(indexed_e, start=1):
    ranks[orig_idx] = rank

  return {
      "rejected_indices": rejected_indices,
      "rejection_threshold": threshold,
      "equivalent_p_cutoff": p_cutoff,
      "k_star": k_star,
      "e_values": e_values,
      "ranks": ranks,
  }


def compute_ebh_boosted(
    p_values: Sequence[float],
    alpha: float,
    kappa: float,
    m_total: int,
    probe_counts: Sequence[int] | None = None,
    *,
    n_max: int | None = None,
) -> dict[str, Any]:
  """Applies boosted e-BH procedure using per-hypothesis boosting factor b >= 1.

  Args:
    p_values: Sequence of p-values in [0, 1].
    alpha: Significance level in (0, 1).
    kappa: Calibrator exponent in (0, 1).
    m_total: Total hypothesis count K.
    probe_counts: Optional sequence of probe counts m per hypothesis (default
      1).
    n_max: If set, each m is clipped to [1, n_max] via
      `boosting.boost_probe_count`. None keeps the raw counts.

  Returns:
    Dict containing:
      - 'rejected_indices': Set of 0-based indices that reject.
      - 'rejection_threshold': Minimum boosted e-value threshold.
      - 'equivalent_p_cutoff': Equivalent p-value cutoff.
      - 'k_star': Number of rejections.
      - 'boost_factors': Boost factors b in original order.
      - 'boosted_e_values': Boosted e-values (b * e) in original order.
      - 'ranks': 1-based ranks (descending by boosted e-value).
  """
  n = len(p_values)
  if n == 0 or m_total <= 0 or alpha <= 0.0:
    return {
        "rejected_indices": set(),
        "rejection_threshold": float("inf"),
        "equivalent_p_cutoff": 0.0,
        "k_star": 0,
        "boost_factors": [],
        "boosted_e_values": [],
        "ranks": [],
    }

  if probe_counts is None:
    probe_counts = [1] * n
  if n_max is not None:
    probe_counts = [boosting.boost_probe_count(m, n_max) for m in probe_counts]

  e_values = [calibrators.p_to_e(p, kappa=kappa) for p in p_values]
  boost_factors = [
      boosting.compute_boost_factor(
          alpha=alpha, kappa=kappa, m=m_probes, k_total=m_total
      )
      for m_probes in probe_counts
  ]
  boosted_e = [b * e for b, e in zip(boost_factors, e_values)]

  # Sort by boosted e-value descending
  indexed_boosted = sorted(
      enumerate(boosted_e), key=lambda x: x[1], reverse=True
  )

  k_star = _ebh_k_star(indexed_boosted, m_total, alpha)

  if k_star > 0:
    threshold = m_total / (alpha * k_star)
    # Representative boost factor for p-cutoff
    mean_b = float(np.mean(boost_factors))
    p_cutoff = (k_star * alpha * kappa * mean_b / m_total) ** (
        1.0 / (1.0 - kappa)
    )
    rejected_indices = {orig_idx for orig_idx, _ in indexed_boosted[:k_star]}
  else:
    threshold = float("inf")
    p_cutoff = 0.0
    rejected_indices = set()

  ranks = [0] * n
  for rank, (orig_idx, _) in enumerate(indexed_boosted, start=1):
    ranks[orig_idx] = rank

  return {
      "rejected_indices": rejected_indices,
      "rejection_threshold": threshold,
      "equivalent_p_cutoff": p_cutoff,
      "k_star": k_star,
      "boost_factors": boost_factors,
      "boosted_e_values": boosted_e,
      "ranks": ranks,
  }


def e_bh_rejection(
    e_values: dict[str, float],
    alpha: float,
    k_total_override: int | None = None,
) -> tuple[set[str], float]:
  """Applies offline e-BH procedure at level alpha.

  Args:
    e_values: Dictionary mapping hypothesis_id -> final e-value.
    alpha: Target False Discovery Rate level in (0, 1).
    k_total_override: If provided, use this as K instead of len(e_values). This
      is useful when the number of hypotheses is fixed in advance (e.g., K_max =
      budget // chunk_size) and may differ from the number of hypotheses
      actually tested.

  Returns:
    Tuple of (rejected_hypothesis_ids_set, rejection_threshold).
    If no hypotheses are rejected, returns (set(), inf).
  """
  k_total = k_total_override if k_total_override is not None else len(e_values)
  if k_total == 0 or alpha <= 0.0:
    return set(), float("inf")

  sorted_items = sorted(e_values.items(), key=lambda kv: kv[1], reverse=True)

  k_star = _ebh_k_star(sorted_items, k_total, alpha)

  if k_star == 0:
    return set(), float("inf")

  rejection_threshold = k_total / (alpha * k_star)
  rejected_set = {h_id for h_id, _ in sorted_items[:k_star]}
  return rejected_set, rejection_threshold


def compute_all_fdr_corrections(
    records: list[dict[str, Any]],
    alpha: float = 0.05,
    kappa: float = 0.1741,
    verification_mode: str = "single_batch_p_value",
    ebh_family_size: int | None = None,
) -> dict[str, Any]:
  """Performs all four FDR procedures (BH, BY, e-BH, Boosted e-BH) on records.

  The BH / BY family is the successfully evaluated records (status "OK");
  errored candidates carry no p-value to rank and are left out.

  Args:
    records: List of record dicts from verification or precision check.
    alpha: Significance level alpha.
    kappa: Calibrator exponent kappa.
    verification_mode: Verification mode ('single_batch_p_value').
    ebh_family_size: Optional fixed family size K for the two e-BH procedures.
      e-BH's guarantee requires K to be chosen before seeing the data, but an
      agent proposes a data-dependent number of hypotheses, so deriving K from
      the observed count makes it adaptive and voids the guarantee. Passing the
      pre-registered cap (max_predictions) restores validity: the unfilled slots
      behave exactly like hypotheses carrying e = 0, which are valid e-variables
      and can never be rejected. BH and BY are unaffected.

  Returns:
    Dict with:
      - 'annotated_records': List of record dicts augmented with FDR fields.
      - 'summary': Summary dictionary comparing all methods.

  Raises:
    DegeneratePValuesError: If single-batch mode has only default 1.0 p-values.
  """
  n_total = len(records)
  valid_indices = [
      i for i, r in enumerate(records) if r.get("status") in ("OK", None)
  ]
  n_eval = len(valid_indices)

  m_family = n_eval

  # Raising K only ever raises the e-BH threshold K / (alpha * k), so a cap
  # above the observed count is the conservative direction. A cap *below* it
  # would be anti-conservative and means the cap was misreported, so fall back
  # to the observed count rather than silently inflating the rejection set.
  if ebh_family_size is not None and ebh_family_size > 0:
    m_ebh = max(int(ebh_family_size), m_family)
    if m_ebh != int(ebh_family_size):
      logging.warning(
          "ebh_family_size=%d is below the observed family size %d; using the"
          " observed count instead.",
          int(ebh_family_size),
          m_family,
      )
  else:
    m_ebh = m_family

  # Single batch p-value mode
  valid_p_values = []
  valid_probe_counts = []
  for i in valid_indices:
    p_val = records[i].get("p_value")
    if p_val is None:
      raise DegeneratePValuesError(
          f"Record at index {i} is missing 'p_value' in single_batch mode."
      )
    valid_p_values.append(float(p_val))
    valid_probe_counts.append(int(records[i].get("n_subhyps_used", 1)))

  # Check for degenerate p-values: all p_values are identically 1.0 and no
  # sample statistics are present.
  if n_eval > 0 and all(p >= 1.0 for p in valid_p_values):
    has_stats = any(
        "observed_rate" in records[i] or "sample_size" in records[i]
        for i in valid_indices
    )
    if not has_stats:
      raise DegeneratePValuesError(
          "All p-values are 1.0 and no sample statistics are present."
      )

  # Run all four procedures. BH and BY use the observed family size; the e-BH
  # variants use the pre-registered one, which may be larger.
  bh_res = compute_bh_procedure(
      p_values=valid_p_values, alpha=alpha, m_total=m_family
  )
  by_res = compute_by_procedure(
      p_values=valid_p_values, alpha=alpha, m_total=m_family
  )
  ebh_u_res = compute_ebh_unboosted(
      p_values=valid_p_values, alpha=alpha, kappa=kappa, m_total=m_ebh
  )
  ebh_b_res = compute_ebh_boosted(
      p_values=valid_p_values,
      alpha=alpha,
      kappa=kappa,
      m_total=m_ebh,
      probe_counts=valid_probe_counts,
  )

  annotated_records = []
  valid_ptr = 0
  for idx, r in enumerate(records):
    r_copy = dict(r)
    if idx in valid_indices:
      p_val = valid_p_values[valid_ptr]
      # Deliberately independent of the stored `verified` flag: this is a fresh
      # procedure at the caller's alpha, not a replay of the verifier's own
      # decision at its own alpha.
      r_copy["unadjusted_verified"] = bool(p_val <= alpha)
      r_copy["bh_verified"] = valid_ptr in bh_res["rejected_indices"]
      r_copy["bh_q_value"] = bh_res["q_values"][valid_ptr]
      r_copy["bh_rank"] = bh_res["ranks"][valid_ptr]

      r_copy["by_verified"] = valid_ptr in by_res["rejected_indices"]
      r_copy["by_q_value"] = by_res["q_values"][valid_ptr]

      r_copy["e_value"] = ebh_u_res["e_values"][valid_ptr]
      r_copy["ebh_unboosted_verified"] = (
          valid_ptr in ebh_u_res["rejected_indices"]
      )
      r_copy["ebh_unboosted_rank"] = ebh_u_res["ranks"][valid_ptr]

      r_copy["boost_factor"] = ebh_b_res["boost_factors"][valid_ptr]
      r_copy["boosted_e_value"] = ebh_b_res["boosted_e_values"][valid_ptr]
      r_copy["ebh_boosted_verified"] = (
          valid_ptr in ebh_b_res["rejected_indices"]
      )
      r_copy["ebh_boosted_rank"] = ebh_b_res["ranks"][valid_ptr]
      valid_ptr += 1
    else:
      r_copy["unadjusted_verified"] = False
      r_copy["bh_verified"] = False
      r_copy["bh_q_value"] = None
      r_copy["bh_rank"] = None
      r_copy["by_verified"] = False
      r_copy["by_q_value"] = None
      r_copy["e_value"] = None
      r_copy["ebh_unboosted_verified"] = False
      r_copy["ebh_unboosted_rank"] = None
      r_copy["boost_factor"] = None
      r_copy["boosted_e_value"] = None
      r_copy["ebh_boosted_verified"] = False
      r_copy["ebh_boosted_rank"] = None
    annotated_records.append(r_copy)

  summary = {
      "verification_mode": verification_mode,
      "n_total": n_total,
      "n_evaluated": n_eval,
      "family_size_m": m_family,
      "ebh_family_size_m": m_ebh,
      "methods": {
          "unadjusted": {
              "n_verified": sum(
                  1 for r in annotated_records if r.get("unadjusted_verified")
              ),
              "rejection_threshold": alpha,
              "assumptions": "None (uncontrolled)",
          },
          "bh": {
              "n_verified": len(bh_res["rejected_indices"]),
              "rejection_threshold": bh_res["rejection_threshold"],
              "k_star": bh_res["k_star"],
              "assumptions": "PRDS (positive dependence)",
          },
          "by": {
              "n_verified": len(by_res["rejected_indices"]),
              "rejection_threshold": by_res["rejection_threshold"],
              "harmonic_penalty_c": by_res["harmonic_c"],
              "k_star": by_res["k_star"],
              "assumptions": "Arbitrary dependence",
          },
          "ebh_unboosted": {
              "n_verified": len(ebh_u_res["rejected_indices"]),
              "rejection_threshold": ebh_u_res["rejection_threshold"],
              "equivalent_p_cutoff": ebh_u_res["equivalent_p_cutoff"],
              "k_star": ebh_u_res["k_star"],
              "kappa": kappa,
              "assumptions": "Arbitrary dependence",
          },
          "ebh_boosted": {
              "n_verified": len(ebh_b_res["rejected_indices"]),
              "rejection_threshold": ebh_b_res["rejection_threshold"],
              "equivalent_p_cutoff": ebh_b_res["equivalent_p_cutoff"],
              "boost_factor": (
                  float(np.mean(ebh_b_res["boost_factors"]))
                  if ebh_b_res["boost_factors"]
                  else 1.0
              ),
              "k_star": ebh_b_res["k_star"],
              "kappa": kappa,
              "assumptions": "Arbitrary dependence",
          },
      },
  }

  return {"annotated_records": annotated_records, "summary": summary}
