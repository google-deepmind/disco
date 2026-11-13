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

"""Tests for eval/output_schema.

The `_legacy_*` builders are frozen copies of the dict literals in
eval_disco._init_results_summary and eval_discovery_baselines'
run_discovery_pipeline; `to_json()` must serialize byte-identically.
"""

import json
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized

from disco.core import records as records_lib
from disco.eval import output_schema


def _fdr_result() -> records_lib.DiscoResult:
  return records_lib.DiscoResult(
      all_records={},
      rejected_hypotheses={"h_2"},
      final_e_values={"h_1": 0.5, "h_2": 30.0},
      boosted_e_values={"h_1": 0.6, "h_2": 31.0},
      boost_factors={"h_1": 1.2, "h_2": 1.03},
      probe_counts={"h_1": 3, "h_2": 6},
      terminal_statuses={"h_1": "L", "h_2": "A"},
      total_hypotheses_k=4,
      rejection_threshold=20.0,
      nominal_alpha_used=0.04,
      epsilon_post=0.01,
      delta_post=0.02,
      posthoc_certificate=0.03,
      confidence=0.98,
      posthoc_certificate_boost=0.035,
  )


def _legacy_disco(config_summary, experiment_dir, rejected, unrejected):
  return {
      "config": config_summary,
      "disco_eval_dir": experiment_dir,
      "rejected_hypotheses": rejected,
      "unrejected_hypotheses": unrejected,
      "success": False,
      "error": "",
      "precision_check": {},
  }


def _legacy_baseline_fields(fdr_result, alpha):
  return {
      "total_hypotheses_k": fdr_result.total_hypotheses_k,
      "n_rejected": len(fdr_result.rejected_hypotheses),
      "rejected_ids": list(fdr_result.rejected_hypotheses),
      "rejection_threshold": fdr_result.rejection_threshold,
      "final_e_values": fdr_result.final_e_values,
      "probe_counts": dict(fdr_result.probe_counts),
      "k_reference": fdr_result.total_hypotheses_k,
      "target_alpha": alpha,
  }


def _legacy_disco_fields(fdr_result):
  return {
      "total_hypotheses_k": fdr_result.total_hypotheses_k,
      "n_rejected": len(fdr_result.rejected_hypotheses),
      "rejected_ids": list(fdr_result.rejected_hypotheses),
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


def _legacy_method(name, origin, preds, verified, queries, prec, flags):
  return {
      "method_name": name,
      "method": name,
      "origin": origin,
      "num_predictions": len(preds),
      "predictions": preds,
      "rejected_hypotheses": preds,
      "verified_hypotheses": verified,
      "hypotheses": preds,
      "n_discoveries": len(preds),
      "total_target_queries": queries,
      "precision_check": prec,
      "config": flags,
  }


def _dumps(d: Any) -> str:
  return json.dumps(d, indent=2)


class WriterParityTest(parameterized.TestCase):

  def test_disco_results(self):
    cfg = {"alpha": 0.1, "name": "x"}
    got = output_schema.DiscoResults(
        config=cfg,
        disco_eval_dir="/out/x",
        rejected_hypotheses=["A"],
        unrejected_hypotheses=["B", "C"],
    ).to_json()
    self.assertEqual(
        _dumps(got), _dumps(_legacy_disco(cfg, "/out/x", ["A"], ["B", "C"]))
    )
    self.assertEqual(tuple(got), output_schema.DISCO_BASE_KEYS)

  def test_fdr_fields(self):
    fdr = _fdr_result()
    self.assertEqual(
        _dumps(output_schema.baseline_fdr_fields(fdr, 0.1)),
        _dumps(_legacy_baseline_fields(fdr, 0.1)),
    )
    self.assertEqual(
        _dumps(output_schema.disco_fdr_fields(fdr)),
        _dumps(_legacy_disco_fields(fdr)),
    )

  @parameterized.named_parameters(
      ("with_precision", {"precision": 0.5, "records": []}),
      ("no_precision", None),
  )
  def test_discovery_method(self, prec):
    flags = {"budget": 100}
    got = output_schema.DiscoveryMethodResult(
        method_name="direct_asking",
        origin="baseline",
        predictions=["P1", "P2"],
        verified_hypotheses=["P2"],
        total_target_queries=42,
        precision_check=prec,
        config=flags,
    ).to_json()
    self.assertEqual(
        _dumps(got),
        _dumps(
            _legacy_method(
                "direct_asking",
                "baseline",
                ["P1", "P2"],
                ["P2"],
                42,
                prec,
                flags,
            )
        ),
    )
    self.assertEqual(tuple(got), output_schema.DISCOVERY_METHOD_KEYS)

  def test_discovery_task_and_suite(self):
    methods = [{"method_name": "m"}]
    flags = {"budget": 1}
    task = output_schema.DiscoveryTaskResults(
        task_suite="recipes", task_name="t", methods=methods, config=flags
    ).to_json()
    self.assertEqual(
        _dumps(task),
        _dumps({
            "task_suite": "recipes",
            "task_name": "t",
            "stage": "evaluation_complete",
            "methods": methods,
            "baseline_results": methods,
            "config": flags,
        }),
    )
    self.assertEqual(tuple(task), output_schema.DISCOVERY_TASK_KEYS)
    suite = output_schema.DiscoverySuiteResults(
        task_suite="recipes", task_name="", suite_results={"t": methods}
    ).to_json()
    self.assertEqual(
        _dumps(suite),
        _dumps({
            "task_suite": "recipes",
            "task_name": "",
            "suite_results": {"t": methods},
        }),
    )
    self.assertEqual(tuple(suite), output_schema.DISCOVERY_SUITE_KEYS)


if __name__ == "__main__":
  absltest.main()
