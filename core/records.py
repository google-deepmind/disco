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

"""Record types for hypothesis tracking and run outcomes in DISCO."""

import dataclasses
from typing import Any, Callable

from disco.stats import plan as stats_plan


@dataclasses.dataclass
class EValueEntry:
  """A single round's e-value measurement with its budget watermark.

  `sub_hypothesis` holds the hypothesis text: hypotheses are operationalized
  directly, so there is no sub-hypothesis step. The field keeps its name so
  that old results and checkpoints, which were written with sub-hypothesis
  texts, still load.
  """

  sub_hypothesis: str
  e_value: float
  cumulative_queries: int = 0

  def __iter__(self):
    return iter((self.sub_hypothesis, self.e_value))

  def __getitem__(self, idx):
    return (self.sub_hypothesis, self.e_value, self.cumulative_queries)[idx]


@dataclasses.dataclass
class HypothesisRecord:
  """Evidence for a single hypothesis accumulated across rounds."""

  text: str
  e_values: list[EValueEntry] = dataclasses.field(default_factory=list)
  # No longer written (see `EValueEntry`); kept so old checkpoints load.
  sub_hypotheses: list[str] = dataclasses.field(default_factory=list)
  failures: list[dict[str, str]] = dataclasses.field(default_factory=list)
  failure_reflections: list[str] = dataclasses.field(default_factory=list)
  executed_queries: set[str] = dataclasses.field(default_factory=set)
  queries: int = 0
  # Pre-registered statistical plan, elicited once on the hypothesis's first
  # round and reused for every later round. Re-eliciting per round would let
  # the agent revise its test after seeing data, which is the exact degree of
  # freedom pre-registration exists to remove.
  test_plan: stats_plan.PreRegisteredTestPlan | None = None

  def format_failures(self, indent: str = "  ") -> str:
    """Formats concrete failure examples for prompt context."""
    if not self.failures:
      return ""
    lines: list[str] = [f"{indent}Concrete Failure Patterns discovered:\n"]
    for i, fail in enumerate(self.failures):
      lines.append(
          f"{indent}  Example {i+1} ({fail.get('judgment', 'FAILURE')}):\n"
          f"{indent}    - Prompt: \"{fail.get('prompt', '')}\"\n"
          f"{indent}    - Model Response: \"{fail.get('response', '')}\"\n"
          f"{indent}    - Judge Reasoning: \"{fail.get('reasoning', '')}\"\n"
      )
    return "".join(lines)


@dataclasses.dataclass
class DiscoResult:
  """Result container for an FDR-controlled DISCO run (proof-based)."""

  all_records: dict[str, HypothesisRecord]
  rejected_hypotheses: set[str]
  final_e_values: dict[str, float]
  boosted_e_values: dict[str, float]
  boost_factors: dict[str, float]
  probe_counts: dict[str, int]
  terminal_statuses: dict[str, str]  # 'A', 'H', 'L', 'B_j'
  total_hypotheses_k: int
  rejection_threshold: float
  nominal_alpha_used: float  # Calibrated alpha_BH used for e-BH

  # DISCO certificates (Theorem 5.2)
  epsilon_post: float = 0.0
  delta_post: float = 0.0
  posthoc_certificate: float = 0.0  # epsilon_post + delta_post
  confidence: float = 1.0  # 1 - delta_post

  # Boosted variants
  epsilon_post_boost: float = 0.0
  posthoc_certificate_boost: float = 0.0

  # Bounds report & pre-hoc bounds
  prehoc_bounds: dict[str, Any] = dataclasses.field(default_factory=dict)
  bounds_report: dict[str, Any] = dataclasses.field(default_factory=dict)

  # Alpha sweep results
  alpha_sweep_results: list[Any] = dataclasses.field(default_factory=list)

  # Discovery rounds skipped without an e-value, by kind
  # (`scoring.SKIP_KINDS`). Diagnostic only.
  skip_counts: dict[str, int] = dataclasses.field(default_factory=dict)


def log_hypothesis_details(
    h_id: str,
    h_data: HypothesisRecord,
    agg_e: float,
    log_fn: Callable[[str], None],
    unboosted_e: float | None = None,
) -> None:
  """Logs details of a single hypothesis for the final summary."""
  log_fn(f"\nID: {h_id} (Hypothesis: {h_data.text})")
  log_fn(f"  Final Aggregated E-value: {agg_e:.4f}")
  if unboosted_e is not None:
    log_fn(f"  Unboosted E-value: {unboosted_e:.4f}")
  log_fn(f"  Total Queries Used: {h_data.queries}")
  if h_data.e_values:
    log_fn("  Tested via:")
    for entry in h_data.e_values:
      s_text = (
          entry.sub_hypothesis if hasattr(entry, "sub_hypothesis") else entry[0]
      )
      e_val = entry.e_value if hasattr(entry, "e_value") else entry[1]
      log_fn(f"    - Sub-H: {s_text} (E-val: {e_val:.4f})")
  else:
    log_fn("  (No tests run yet)")
