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

"""Evaluation script for DISCO behavior discovery and out-of-sample verification."""

import asyncio
from collections.abc import Sequence
import dataclasses
import json
import os
from typing import Any

from absl import app

from disco import flags_lib
from disco import precision_estimation
from disco import runners
from disco.core import config as config_lib
from disco.core import fdr_outputs
from disco.core import records as records_lib
from disco.core import run_config
from disco.eval import output_schema
from disco.verification import failed_records_flag
from disco.verification import min_response_fraction_flag
from disco.verification import scoring

# The precision check always runs in this mode; recorded in the results so
# `apply_fdr_control` can tell how the verification records were produced.
_VERIFICATION_MODE = "single_batch_p_value"

DISCO_FLAGS = flags_lib.define_disco_flags(
    default_seed="Inconsistent shapes and colors when drawing shapes",
    default_budget=30,
    default_output_dir="./disco_eval",
    default_name="disco_eval",
)


@dataclasses.dataclass
class EvalLoopResult:
  """Container for the raw outputs of the DISCO evaluation loop."""

  rejected_hypotheses: list[str]
  unrejected_hypotheses: list[str]
  rejected_hypotheses_records: list[tuple[str, records_lib.HypothesisRecord]]
  all_records: dict[str, records_lib.HypothesisRecord] = dataclasses.field(
      default_factory=dict
  )
  fdr_result: records_lib.DiscoResult | None = None


def _build_disco_config() -> config_lib.DiscoConfig:
  """Constructs and validates the DiscoConfig from flags and `--config`."""
  overrides = run_config.apply_yaml_config(
      field_names=run_config.config_field_names(config_lib.DiscoConfig),
  )
  abs_output_dir = os.path.abspath(DISCO_FLAGS.output_dir.value)
  os.makedirs(abs_output_dir, exist_ok=True)

  config = config_lib.DiscoConfig.from_flags(
      DISCO_FLAGS,
      output_dir=abs_output_dir,
      experiment_name=f"disco_eval_{DISCO_FLAGS.name.value}",
      min_response_fraction=min_response_fraction_flag.value(),
      overrides=overrides,
  )

  print(f"Initializing DISCO runner with target model: {config.target_model}")
  print(f"Seed: {config.seed_instruction}")
  print(f"Budget: {config.budget}")
  if config.method == "type1_baseline":
    print(
        "Baseline Mode: Type-I Error Rate (threshold=1/alpha="
        f"{1.0/config.alpha:.2f}, alpha={config.alpha}, no e-BH)"
    )
  elif config.method == "ebh_baseline":
    k_bar = max(1, config.budget // config.chunk_size)
    print(
        "Baseline Mode: Unrestricted exploration + offline e-BH"
        f" (K_bar={k_bar}, alpha={config.alpha})"
    )
  print(f"Output directory: {config.output_dir}")
  return config


async def _run_disco_loop(config: config_lib.DiscoConfig) -> EvalLoopResult:
  """Executes the DISCO run loop and collects outputs."""
  parsers = config.build_parsers()
  runner = runners.DiscoRunner(config, parsers)

  print("Starting DISCO run loop...")
  rejected_hypotheses: list[str] = []
  rejected_hypotheses_records: list[
      tuple[str, records_lib.HypothesisRecord]
  ] = []
  unrejected_hypotheses: list[str] = []

  fdr_result = await runner.run_loop()
  all_records = fdr_result.all_records
  # Proposal order, not set order: set order depends on PYTHONHASHSEED.
  for h_id, rec in fdr_result.all_records.items():
    if h_id in fdr_result.rejected_hypotheses:
      rejected_hypotheses.append(rec.text)
      rejected_hypotheses_records.append((h_id, rec))
    else:
      unrejected_hypotheses.append(rec.text)

  return EvalLoopResult(
      rejected_hypotheses=rejected_hypotheses,
      unrejected_hypotheses=unrejected_hypotheses,
      rejected_hypotheses_records=rejected_hypotheses_records,
      all_records=all_records,
      fdr_result=fdr_result,
  )


def _print_detection_summary(
    run_result: EvalLoopResult, config: config_lib.DiscoConfig
) -> None:
  """Prints defect detection and FDR summary to console."""
  print("\n=== Checking DISCO Defect Detection Results ===")
  print(f"DISCO Rejected Hypotheses: {run_result.rejected_hypotheses}")
  print(f"DISCO Unrejected Hypotheses: {run_result.unrejected_hypotheses}")

  fdr_result = run_result.fdr_result
  if config.baseline_type1_mode and fdr_result is not None:
    print("\n=== Baseline Type-I Error Rate Summary ===")
    print(f"Total hypotheses tested: {len(fdr_result.final_e_values)}")
    print(f"Rejected (E >= 1/alpha): {len(fdr_result.rejected_hypotheses)}")
    print(
        f"Rejection threshold (1/alpha): {fdr_result.rejection_threshold:.4f}"
    )
    print(f"Target alpha: {config.alpha:.4f}")
  elif config.baseline_mode and fdr_result is not None:
    print("\n=== Baseline Offline e-BH Summary ===")
    print(
        f"Conservative K (budget / chunk_size): {fdr_result.total_hypotheses_k}"
    )
    print(f"Total hypotheses tested: {len(fdr_result.final_e_values)}")
    print(f"Rejected by e-BH: {len(fdr_result.rejected_hypotheses)}")
    print(f"e-BH rejection threshold: {fdr_result.rejection_threshold:.4f}")
    print(f"Target alpha: {config.alpha:.4f}")
  elif fdr_result is not None and isinstance(
      fdr_result, records_lib.DiscoResult
  ):
    print("\n=== DISCO FDR Control Summary ===")
    print(f"Total hypotheses tested (K): {fdr_result.total_hypotheses_k}")
    print(f"Rejected by e-BH: {len(fdr_result.rejected_hypotheses)}")
    print(f"e-BH rejection threshold: {fdr_result.rejection_threshold:.4f}")
    print(f"Calibrated alpha_BH: {fdr_result.nominal_alpha_used:.6f}")
    print(f"Post-hoc cert (unboosted): {fdr_result.posthoc_certificate:.6f}")
    print(
        f"Post-hoc cert (boosted): {fdr_result.posthoc_certificate_boost:.6f}"
    )
    print(f"Confidence (1 - delta): {fdr_result.confidence:.6f}")


def _init_results_summary(
    run_result: EvalLoopResult, config: config_lib.DiscoConfig
) -> tuple[dict[str, Any], dict[str, Any]]:
  """Initializes results and config summaries."""
  # The full effective config (flags and YAML-only fields alike); `name` is
  # the key older readers of `flags.json` look for.
  config_summary = dataclasses.asdict(config)
  config_summary["name"] = config.experiment_name

  experiment_dir = os.path.join(config.output_dir, config.experiment_name)
  results_summary = output_schema.DiscoResults(
      config=config_summary,
      disco_eval_dir=experiment_dir,
      rejected_hypotheses=run_result.rejected_hypotheses,
      unrejected_hypotheses=run_result.unrejected_hypotheses,
  ).to_json()

  fdr_result = run_result.fdr_result
  if (
      config.baseline_mode or config.baseline_type1_mode
  ) and fdr_result is not None:
    results_summary["baseline_fdr_fields"] = output_schema.baseline_fdr_fields(
        fdr_result, config.alpha
    )
  elif fdr_result is not None and isinstance(
      fdr_result, records_lib.DiscoResult
  ):
    results_summary["disco_fdr_fields"] = output_schema.disco_fdr_fields(
        fdr_result
    )

  if fdr_result is not None:
    if fdr_result.bounds_report:
      results_summary["bounds_report"] = fdr_result.bounds_report
    if fdr_result.alpha_sweep_results:
      results_summary["alpha_sweep"] = fdr_result.alpha_sweep_results

  return results_summary, config_summary


async def _run_precision_check(
    hypotheses_to_check: list[tuple[str, records_lib.HypothesisRecord]],
    config: config_lib.DiscoConfig,
    results_summary: dict[str, Any],
    final_verified_hypotheses: list[str] | None = None,
) -> None:
  """Executes precision estimation for verified hypotheses."""
  if not hypotheses_to_check:
    return
  experiment_dir = os.path.join(config.output_dir, config.experiment_name)
  os.makedirs(experiment_dir, exist_ok=True)
  log_file = os.path.join(experiment_dir, "verifier.log")
  log = fdr_outputs.make_log_function(log_file, config.log_to_console)

  precision_res = await precision_estimation.run_precision_check(
      verified_hypotheses=hypotheses_to_check,
      config=config,
      log_fn=log,
  )
  final_verified_set = set(
      final_verified_hypotheses
      if final_verified_hypotheses is not None
      else [rec.text for _, rec in hypotheses_to_check]
  )
  # Only discoveries enter precision/FDR. Records whose test produced no usable
  # measurement follow --failed_records (dropped by default).
  score = scoring.score(
      [
          r
          for r in precision_res.records
          if r.hypothesis_text in final_verified_set
      ],
      failed_records=failed_records_flag.value(),
  )
  if score.n_failed:
    log(f"[Precision] WARNING: {score.describe_failed()}.")

  results_summary["precision_check"] = {
      **score.summary_fields(),
      "n_total": score.n_scored,
      "verification_mode": _VERIFICATION_MODE,
      "max_concurrent_hypotheses": config.max_concurrent_hypotheses,
      "scrape_bandwidth": config.scrape_bandwidth,
      "records": [
          {
              "hypothesis_text": r.hypothesis_text,
              "verified": r.verified,
              # Without `status`, downstream cannot tell an uninterpretable test
              # apart from a genuine non-verification and will inflate FDR.
              "status": r.status,
              "block_reasons": r.block_reasons,
              # `precision_res.records` also holds hypotheses that were scored
              # but never rejected by e-BH. Those are not discoveries and must
              # stay out of FDR; they skew verified-positive, so a consumer that
              # ignores this flag will under-report FDR.
              "in_discovery_set": r.hypothesis_text in final_verified_set,
              "final_e_product": r.final_e_product,
              "n_subhyps_used": r.n_subhyps_used,
              "e_values": r.e_values,
              "p_value": r.p_value,
              "sample_size": r.sample_size,
              "observed_successes": r.observed_successes,
              "observed_rate": r.observed_rate,
              "null_rate": r.null_rate,
              "ci_95": [r.ci_lower, r.ci_upper],
              "contrast_diagnostics": r.contrast_diagnostics,
              "n_requested": r.n_requested,
              "n_received": r.n_received,
              "failure_reason": r.failure_reason,
          }
          for r in precision_res.records
      ],
  }


def _save_evaluation_artifacts(
    results_summary: dict[str, Any],
    config_summary: dict[str, Any],
    config: config_lib.DiscoConfig,
) -> None:
  """Saves evaluation results JSON and flags JSON to the experiment directory."""
  experiment_dir = os.path.join(config.output_dir, config.experiment_name)
  os.makedirs(experiment_dir, exist_ok=True)

  results_path = os.path.join(experiment_dir, "evaluation_results.json")
  with open(results_path, "w", encoding="utf-8") as f:
    json.dump(results_summary, f, indent=2)
  print(f"\nSaved evaluation results to: {results_path}")

  flags_path = os.path.join(experiment_dir, "flags.json")
  with open(flags_path, "w", encoding="utf-8") as f:
    json.dump(config_summary, f, indent=2)
  print(f"Saved evaluation flags to: {flags_path}")


async def run_evaluation() -> None:
  """Runs the DISCO evaluation pipeline."""
  config = _build_disco_config()
  run_result = await _run_disco_loop(config)
  _print_detection_summary(run_result, config)

  results_summary, config_summary = _init_results_summary(run_result, config)

  evaluation_hypotheses = run_result.rejected_hypotheses_records
  if run_result.rejected_hypotheses:
    results_summary["success"] = True
  else:
    results_summary["error"] = "No rejected hypotheses were discovered."

  if config.run_precision_check and evaluation_hypotheses:
    await _run_precision_check(
        evaluation_hypotheses,
        config,
        results_summary,
        final_verified_hypotheses=run_result.rejected_hypotheses,
    )

  results_summary["data_quality"] = _data_quality(run_result, results_summary)
  scoring.log_data_quality(results_summary["data_quality"], print)

  _save_evaluation_artifacts(results_summary, config_summary, config)


def _data_quality(
    run_result: EvalLoopResult, results_summary: dict[str, Any]
) -> dict[str, Any]:
  """Returns the payload's `data_quality` block (diagnostic only).

  Args:
    run_result: Outputs of the DISCO loop; supplies the skipped-round counts.
    results_summary: Payload so far; supplies the precision-check records.
  """
  records = results_summary.get("precision_check", {}).get("records", [])
  skip_counts = getattr(run_result.fdr_result, "skip_counts", None)
  return scoring.summarize_data_quality(records, skip_counts)


def main(argv: Sequence[str]) -> None:
  """CLI entry point for DISCO evaluation."""
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  asyncio.run(run_evaluation())


if __name__ == "__main__":
  app.run(main)
