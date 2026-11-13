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

"""Shape of `evaluation_results.json` written by DISCO and the baselines.

Writers build these payloads here so the key set and key order live in one
place; `to_json()` reproduces the canonical dict structure.

Kinds of payload:
  * "disco": eval_disco's run-level file.
  * "discovery_method": eval_discovery_baselines' per-method file.
  * "discovery_task": eval_discovery_baselines' per-task file; the method list
    is written twice, under `methods` and `baseline_results`.
  * "discovery_suite": eval_discovery_baselines' multi-task file.
"""

import dataclasses
from typing import Any

from disco.core import records as records_lib

# Keys every DISCO payload starts with, in write order.
DISCO_BASE_KEYS: tuple[str, ...] = (
    "config",
    "disco_eval_dir",
    "rejected_hypotheses",
    "unrejected_hypotheses",
    "success",
    "error",
    "precision_check",
)
# Optional keys a DISCO payload may gain later in the run.
DISCO_OPTIONAL_KEYS: tuple[str, ...] = (
    "baseline_fdr_fields",
    "disco_fdr_fields",
    "bounds_report",
    "alpha_sweep",
    # Diagnostic coverage summary (verification/scoring.summarize_data_quality).
    "data_quality",
)
DISCOVERY_METHOD_KEYS: tuple[str, ...] = (
    "method_name",
    "method",
    "origin",
    "num_predictions",
    "predictions",
    "rejected_hypotheses",
    "verified_hypotheses",
    "hypotheses",
    "n_discoveries",
    "total_target_queries",
    "precision_check",
    "config",
)
DISCOVERY_TASK_KEYS: tuple[str, ...] = (
    "task_suite",
    "task_name",
    "stage",
    "methods",
    "baseline_results",
    "config",
)
DISCOVERY_SUITE_KEYS: tuple[str, ...] = (
    "task_suite",
    "task_name",
    "suite_results",
)


@dataclasses.dataclass
class DiscoResults:
  """Initial eval_disco payload; later stages fill the placeholder fields."""

  config: dict[str, Any]
  disco_eval_dir: str
  rejected_hypotheses: list[str]
  unrejected_hypotheses: list[str]

  def to_json(self) -> dict[str, Any]:
    """Serializes the DISCO evaluation results to a JSON-compatible dictionary."""
    return {
        "config": self.config,
        "disco_eval_dir": self.disco_eval_dir,
        "rejected_hypotheses": self.rejected_hypotheses,
        "unrejected_hypotheses": self.unrejected_hypotheses,
        "success": False,
        "error": "",
        "precision_check": {},
    }


def baseline_fdr_fields(
    fdr_result: records_lib.DiscoResult, target_alpha: float
) -> dict[str, Any]:
  """`baseline_fdr_fields` block for baseline_mode / baseline_type1_mode."""
  return {
      "total_hypotheses_k": fdr_result.total_hypotheses_k,
      "n_rejected": len(fdr_result.rejected_hypotheses),
      "rejected_ids": sorted(fdr_result.rejected_hypotheses),
      "rejection_threshold": fdr_result.rejection_threshold,
      "final_e_values": fdr_result.final_e_values,
      "probe_counts": dict(fdr_result.probe_counts),
      "k_reference": fdr_result.total_hypotheses_k,
      "target_alpha": target_alpha,
  }


def disco_fdr_fields(fdr_result: records_lib.DiscoResult) -> dict[str, Any]:
  """`disco_fdr_fields` block for DISCO runs."""
  return {
      "total_hypotheses_k": fdr_result.total_hypotheses_k,
      "n_rejected": len(fdr_result.rejected_hypotheses),
      "rejected_ids": sorted(fdr_result.rejected_hypotheses),
      "rejection_threshold": fdr_result.rejection_threshold,
      "final_e_values": fdr_result.final_e_values,
      "boosted_e_values": fdr_result.boosted_e_values,
      "boost_factors": fdr_result.boost_factors,
      "probe_counts": dict(fdr_result.probe_counts),
      "terminal_statuses": fdr_result.terminal_statuses,
      "nominal_alpha_used": fdr_result.nominal_alpha_used,
      "epsilon_post": fdr_result.epsilon_post,
      "delta_post": fdr_result.delta_post,
      "posthoc_certificate": fdr_result.posthoc_certificate,
      "posthoc_certificate_boost": fdr_result.posthoc_certificate_boost,
      "confidence": fdr_result.confidence,
  }


@dataclasses.dataclass
class DiscoveryMethodResult:
  """Per-method payload of a generic discovery baseline."""

  method_name: str
  origin: str
  predictions: list[str]
  verified_hypotheses: list[str]
  total_target_queries: int
  precision_check: dict[str, Any] | None
  config: dict[str, Any]

  def to_json(self) -> dict[str, Any]:
    """Serializes the method result to a JSON-compatible dictionary."""
    # The predictions are repeated under every key readers have looked for.
    return {
        "method_name": self.method_name,
        "method": self.method_name,
        "origin": self.origin,
        "num_predictions": len(self.predictions),
        "predictions": self.predictions,
        "rejected_hypotheses": self.predictions,
        "verified_hypotheses": self.verified_hypotheses,
        "hypotheses": self.predictions,
        "n_discoveries": len(self.predictions),
        "total_target_queries": self.total_target_queries,
        "precision_check": self.precision_check,
        "config": self.config,
    }


@dataclasses.dataclass
class DiscoveryTaskResults:
  """Per-task payload of a generic discovery baseline run."""

  task_suite: str
  task_name: str
  methods: list[dict[str, Any]]
  config: dict[str, Any]
  stage: str = "evaluation_complete"

  def to_json(self) -> dict[str, Any]:
    """Serializes the task results to a JSON-compatible dictionary."""
    return {
        "task_suite": self.task_suite,
        "task_name": self.task_name,
        "stage": self.stage,
        "methods": self.methods,
        "baseline_results": self.methods,
        "config": self.config,
    }


@dataclasses.dataclass
class DiscoverySuiteResults:
  """Multi-task payload of a generic discovery baseline run."""

  task_suite: str
  task_name: str
  suite_results: dict[str, list[dict[str, Any]]]

  def to_json(self) -> dict[str, Any]:
    """Serializes the suite results to a JSON-compatible dictionary."""
    return {
        "task_suite": self.task_suite,
        "task_name": self.task_name,
        "suite_results": self.suite_results,
    }
