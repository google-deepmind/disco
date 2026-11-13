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

r"""Standalone precision verifier for hypotheses from any source.

Loads hypotheses from a JSON file (or dictionary), runs independent verification
probes against a target model using the e-value statistical framework, and
estimates precision (replicability).

Only the settings worth changing per run are flags; every other
`VerifierConfig` field (per-role models, `paired`, `p_value_mode`,
`total_verification_budget`, `random_seed`, concurrency, ...) is set from the
`--config` YAML file.

Usage:
  python -m disco.standalone_precision_verifier \
    --hypotheses_path=/path/to/hypotheses.json \
    --target_model=gemini-3.8-flash \
    --backbone_model=gemini-3.8-flash \
    --output_dir=./outputs/verification_run/ \
    --num_queries=100 \
    --alpha=0.05
"""

import asyncio
from collections.abc import Callable, Mapping, Sequence
import dataclasses
import datetime
import json
import os
from typing import Any

from absl import app
from absl import flags
from absl import logging

from disco import precision_estimation
from disco import verifier_config
from disco.core import fdr_outputs
from disco.core import records as records_lib
from disco.core import run_config
from disco.models import registry as model_registry
from disco.stats import plan as stats_plan
from disco.verification import failed_records_flag
from disco.verification import hypothesis_io
from disco.verification import min_response_fraction_flag
from disco.verification import retry
from disco.verification import scoring
from disco.verification import task_resolver

_HYPOTHESES_PATH = flags.DEFINE_string(
    "hypotheses_path",
    "",
    "Path to a JSON file containing hypotheses to verify.",
)
_TARGET_MODEL = flags.DEFINE_string(
    "target_model",
    model_registry.DEFAULT_GEMINI_MODEL,
    "Model address to test against.",
)
_BACKBONE_MODEL = flags.DEFINE_string(
    "backbone_model",
    "",
    "Model for every auxiliary role (agent, parser, rating and summarizer)"
    " not set explicitly in the --config YAML. Empty uses the default Gemini"
    " model.",
)
_ALPHA = flags.DEFINE_float(
    "alpha",
    0.05,
    "Significance level of the per-hypothesis test.",
)
_SEED_INSTRUCTION = flags.DEFINE_string(
    "seed_instruction",
    "",
    "Task or domain description for context (auto-inherits from metadata if"
    " empty).",
)
_TASK_SUITE = flags.DEFINE_string(
    "task_suite",
    "",
    "Task suite the hypotheses came from (e.g. 'visual_symmetry'). Selects the"
    " verification modality; auto-inherits from the run metadata if empty.",
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    "./outputs/precision_verifier/",
    "Directory to write probe responses and verification results.",
)
_EXPERIMENT_NAME = flags.DEFINE_string(
    "experiment_name",
    "verification",
    "Experiment name used in subdirectories.",
)
_NUM_QUERIES = flags.DEFINE_integer(
    "num_queries",
    100,
    "Base number of queries per hypothesis (doubled if paired). A positive"
    " `total_verification_budget` in the --config YAML is split across the"
    " hypotheses instead.",
)
_RETRY_FAILED = flags.DEFINE_boolean(
    "retry_failed",
    False,
    "If an existing verification_results.json is found under --output_dir,"
    " rerun only its records whose status is one of"
    f" {list(retry.DEFAULT_RETRY_STATUSES)}, merge the new records over them"
    " (by label, then text), recompute the summary, append to"
    " `retry_history`, and keep the previous file as <name>.pre_retry[.N].json."
    " Uses the current flags; a WARNING lists any config fields that differ"
    " from the stored run. --hypotheses_path is then optional. Without an"
    " existing result, runs everything as usual.",
)

# Name of the verification summary JSON written under `--output_dir`.
OUTPUT_FILENAME = "verification_results.json"
# Name of the log file written under `--output_dir`.
LOG_FILENAME = "verifier.log"

# Stored config keys that are expected to differ between runs.
_DRIFT_IGNORED_KEYS = frozenset({"output_dir", "log_to_console"})


def load_hypotheses(path: str) -> list[tuple[str, str]]:
  """Loads hypotheses from a JSON file supporting multiple formats.

  Supported formats:
    1. Dict mapping label -> text:
       {"h_counting": "Model miscounts...", "h_refusal": "Model refuses..."}
    2. Dict mapping label -> dict:
       {"h_counting": {"text": "Model miscounts...", "description": "..."}}
    3. Dict containing a top-level "hypotheses", "rejected_hypotheses",
    "verified_hypotheses", or "predictions" key:
       {"hypotheses": {"h_1": "..."}, ...}, {"rejected_hypotheses": ["..."]}, or
       {"verified_hypotheses": ["..."]}
    4. Dict containing a "methods" or "baseline_results" list of baseline method
    objects:
       {"methods": [{"method_name": "DirectAskingBaseline", "predictions":
       [...]}, ...]}
    5. List of strings or dicts:
       ["Hypothesis 1", "Hypothesis 2"]
       or [{"label": "h_1", "text": "Hypothesis 1"}, ...]

  See verification/hypothesis_io for the shared implementation.

  Args:
    path: Path to the JSON file.

  Returns:
    List of (label, hypothesis_text) pairs.
  """
  return hypothesis_io.load_hypotheses(path, strict_type=True)


def load_hypotheses_and_metadata(
    path: str,
) -> tuple[list[tuple[str, str]], dict[str, Any] | list[Any]]:
  """Loads hypotheses and returns both extracted hypotheses and raw JSON data."""
  return hypothesis_io.load_hypotheses_and_data(path, strict_type=True)


# Task-setting resolution lives in verification/task_resolver; these names are
# kept for existing callers.
resolve_task_autorater = task_resolver.resolve_task_autorater
resolve_seed_instruction = task_resolver.resolve_seed_instruction
resolve_task_suite = task_resolver.resolve_task_suite
resolve_task_null_proportion = task_resolver.resolve_task_null_proportion


def _record_to_dict(
    r: precision_estimation.HypothesisPrecisionRecord,
) -> dict[str, Any]:
  """The JSON form of one record in `verification_results.json`."""
  return {
      "label": r.hypothesis_id,
      "hypothesis_text": r.hypothesis_text,
      "verified": r.verified,
      "status": r.status,
      "block_reasons": r.block_reasons,
      "final_e_product": r.final_e_product,
      "n_subhyps_used": r.n_subhyps_used,
      "e_values": r.e_values,
      "p_value": r.p_value,
      "sample_size": r.sample_size,
      "observed_successes": r.observed_successes,
      "observed_rate": r.observed_rate,
      "null_rate": r.null_rate,
      "ci_95": [r.ci_lower, r.ci_upper],
      "n_requested": r.n_requested,
      "n_received": r.n_received,
      "failure_reason": r.failure_reason,
      "test_plan": (
          dataclasses.asdict(r.test_plan) if r.test_plan is not None else None
      ),
      "allocated_queries": r.allocated_queries,
  }


def _retry_budget(
    records: Sequence[Mapping[str, Any]],
    labelled_positions: Sequence[tuple[str, int]],
    log: Callable[[str], None],
) -> precision_estimation.RetryBudget:
  """Original per-hypothesis query counts for `--retry_failed`.

  Args:
    records: The `records` list of the previous result.
    labelled_positions: (label, index into `records`) of the retried records.
    log: Logging function.

  Returns:
    RetryBudget tracking recorded allocations and original hypothesis count.
  """
  allocated: dict[str, int] = {}
  for label, i in labelled_positions:
    try:
      allocated[label] = max(int(records[i].get("allocated_queries") or 0), 0)
    except (TypeError, ValueError):
      allocated[label] = 0
  n_known = sum(1 for q in allocated.values() if q > 0)
  log(
      f"  Keeping the original query count for {n_known}/"
      f"{len(labelled_positions)} retried record(s); the rest get the original"
      f" per-hypothesis share of a total budget split over {len(records)}."
  )
  return precision_estimation.RetryBudget(
      allocated_queries=allocated, n_original=len(records)
  )


def _stored_test_plans(
    records: Sequence[Mapping[str, Any]],
    labelled_positions: Sequence[tuple[str, int]],
    log: Callable[[str], None],
) -> dict[str, stats_plan.PreRegisteredTestPlan]:
  """Test plans saved on `records`, keyed by label, for `--retry_failed`.

  Records written before plans were saved, or whose check had no single plan,
  have none; those hypotheses pre-register again.

  Args:
    records: The `records` list of the previous result.
    labelled_positions: (label, index into `records`) of the retried records.
    log: Logging function.

  Returns:
    Mapping from hypothesis label to its stored PreRegisteredTestPlan.
  """
  plans: dict[str, stats_plan.PreRegisteredTestPlan] = {}
  for label, i in labelled_positions:
    raw = records[i].get("test_plan")
    if not isinstance(raw, dict):
      continue
    try:
      plans[label] = stats_plan.PreRegisteredTestPlan(**raw)
    except TypeError as e:
      log(
          f"  WARNING: [{label}] unreadable stored test plan ({e}); it will"
          " pre-register again."
      )
  log(
      f"  Reusing the stored test plan for {len(plans)}/"
      f"{len(labelled_positions)} retried record(s); the rest pre-register"
      " again."
  )
  return plans


def _stored_config(config: verifier_config.VerifierConfig) -> dict[str, Any]:
  """The `config` block of `verification_results.json`."""
  return dataclasses.asdict(config)


async def _run_precision_check(
    hypotheses_to_check: list[tuple[str, records_lib.HypothesisRecord]],
    config: verifier_config.VerifierConfig,
    log: Callable[[str], None],
    stored_plans: Mapping[str, stats_plan.PreRegisteredTestPlan] | None = None,
    retry_budget: precision_estimation.RetryBudget | None = None,
) -> precision_estimation.PrecisionResult:
  """Logs the mode and runs `precision_estimation.run_precision_check`."""
  log(
      "\nStarting Single-Batch Precision Verification"
      f" (N={config.single_batch_num_queries},"
      f" alpha={config.single_batch_alpha})..."
  )

  return await precision_estimation.run_precision_check(
      verified_hypotheses=hypotheses_to_check,
      config=config,
      log_fn=log,
      stored_plans=stored_plans,
      retry_budget=retry_budget,
  )


def _load_existing_results(path: str) -> dict[str, Any] | None:
  """Returns a previous result with a `records` list at `path`, if any."""
  if not os.path.exists(path):
    return None
  try:
    with open(path, "r", encoding="utf-8") as f:
      data = json.loads(f.read())
  except (OSError, ValueError) as e:
    logging.warning("Ignoring unreadable result %s: %s", path, e)
    return None
  if not isinstance(data, dict) or not isinstance(data.get("records"), list):
    logging.warning("Ignoring result without a records list: %s", path)
    return None
  return data


def _config_drift(stored: Any, current: dict[str, Any]) -> list[str]:
  """Config keys whose stored value differs from the current one.

  Keys missing from the stored config (fields added since) are not reported.

  Args:
    stored: The `config` block of the previous result.
    current: The config this retry runs with (`_stored_config`).

  Returns:
    Sorted list of config keys that differ between stored and current runs.
  """
  if not isinstance(stored, dict):
    return []
  # Round-trip so tuples and lists compare equal.
  current = json.loads(json.dumps(current))
  return sorted(
      k
      for k, v in current.items()
      if k in stored and k not in _DRIFT_IGNORED_KEYS and stored[k] != v
  )


async def _retry_failed_records(
    existing: dict[str, Any],
    config: verifier_config.VerifierConfig,
    statuses: tuple[str, ...],
    log: Callable[[str], None],
) -> None:
  """`--retry_failed`: reruns failed records of `existing` and rewrites it."""
  out_path = os.path.join(config.output_dir, OUTPUT_FILENAME)
  records: list[dict[str, Any]] = existing["records"]
  positions = retry.select(records, statuses)
  log(
      f"\n=== Retry Failed ===\n{len(positions)}/{len(records)} record(s) in"
      f" {out_path} have status in {list(statuses)}."
  )
  drift = _config_drift(existing.get("config"), _stored_config(config))
  if drift:
    log(
        "  WARNING: the current config differs from the stored run in:"
        f" {', '.join(drift)}. Retried records use the current config."
    )
  if not positions:
    log("  Nothing to retry; leaving the result unchanged.")
    return

  labels = [str(records[i].get("label") or f"h_{i}") for i in positions]
  hypotheses_to_check = [
      (
          label,
          records_lib.HypothesisRecord(
              text=str(records[i].get("hypothesis_text", ""))
          ),
      )
      for label, i in zip(labels, positions)
  ]
  for label, h in hypotheses_to_check:
    log(f"  [{label}]: {h.text}")
  # Reuse each record's original plan: picking a new test after seeing the
  # first attempt fail would let the retry shop for a test.
  stored_plans = _stored_test_plans(records, list(zip(labels, positions)), log)
  # Keep each record's original sample size: a retry must not test a
  # hypothesis harder just because fewer hypotheses are being retried.
  retry_budget = _retry_budget(records, list(zip(labels, positions)), log)
  precision_res = await _run_precision_check(
      hypotheses_to_check,
      config,
      log,
      stored_plans=stored_plans,
      retry_budget=retry_budget,
  )
  merged, changes = retry.merge(
      records, positions, [_record_to_dict(r) for r in precision_res.records]
  )

  backup = retry.backup_path(out_path, os.path.exists)
  with (
      open(out_path, "r", encoding="utf-8") as src,
      open(backup, "w", encoding="utf-8") as dst,
  ):
    dst.write(src.read())
  log(f"  Kept the previous result as: {backup}")

  score = scoring.score(merged, failed_records=failed_records_flag.value())
  entry = retry.history_entry(statuses, changes)
  if drift:
    entry["config_drift"] = drift
  results_data = dict(existing)
  results_data.update({
      **score.summary_fields(),
      **retry.summary_counts(merged),
      "records": merged,
      "retry_history": [*existing.get("retry_history", []), entry],
      "data_quality": scoring.summarize_data_quality(merged),
  })

  log("\n=== Retry Complete ===")
  log(
      f"  Recovered {entry['n_recovered']}/{entry['n_retried']} retried"
      " record(s)."
  )
  log(f"  Verified: {score.n_verified}/{score.n_scored} scored")
  if score.n_failed:
    log(f"  WARNING: {score.describe_failed()}.")
  log(f"  Precision: {scoring.format_precision(score.precision)}")
  scoring.log_data_quality(results_data["data_quality"], log)

  with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results_data, f, indent=2)
  log(f"\nSaved verification results to: {out_path}")


async def run_verifier() -> None:
  """Main async pipeline for standalone precision verification."""
  field_overrides = run_config.apply_yaml_config(
      field_names=run_config.config_field_names(verifier_config.VerifierConfig)
  )
  hypotheses_path = _HYPOTHESES_PATH.value
  existing: dict[str, Any] | None = None
  retry_statuses: tuple[str, ...] = ()
  if _RETRY_FAILED.value:
    retry_statuses = retry.DEFAULT_RETRY_STATUSES
    existing = _load_existing_results(
        os.path.join(_OUTPUT_DIR.value, OUTPUT_FILENAME)
    )
  if not hypotheses_path and existing is None:
    raise ValueError(
        "--hypotheses_path is required (unless --retry_failed finds an"
        " existing result)."
    )

  hypotheses_with_labels: list[tuple[str, str]] = []
  raw_data: dict[str, Any] | list[Any] | None = None
  if hypotheses_path:
    hypotheses_with_labels, raw_data = load_hypotheses_and_metadata(
        hypotheses_path
    )
  raw_data_dict = raw_data if isinstance(raw_data, dict) else None
  # `procedural_autorater` is YAML-only; a YAML value beats the run metadata
  # and the task registry.
  resolved_autorater = resolve_task_autorater(
      raw_data_dict,
      str(field_overrides.get("procedural_autorater") or ""),
  )
  resolved_seed_instruction = resolve_seed_instruction(
      raw_data_dict,
      _SEED_INSTRUCTION.value,
  )
  resolved_task_suite = resolve_task_suite(
      raw_data_dict,
      _TASK_SUITE.value,
  )
  resolved_task_null_prop = resolve_task_null_proportion(
      raw_data_dict,
      resolved_task_suite,
  )

  config = verifier_config.VerifierConfig(
      target_model=model_registry.resolve_model_url(_TARGET_MODEL.value),
      # Fans out to the agent, parser, rating and summarizer models unless a
      # YAML override sets one of them explicitly.
      backbone_model=model_registry.resolve_model_url(_BACKBONE_MODEL.value),
      alpha=_ALPHA.value,
      output_dir=_OUTPUT_DIR.value,
      experiment_name=_EXPERIMENT_NAME.value,
      seed_instruction=resolved_seed_instruction,
      task_suite=resolved_task_suite,
      task_null_proportion=resolved_task_null_prop,
      procedural_autorater=resolved_autorater,
      # A procedural autorater scores every response itself, so the agent's
      # LLM-judge analysis tool is withheld.
      exclude_analyze=bool(resolved_autorater),
      single_batch_num_queries=_NUM_QUERIES.value,
      single_batch_alpha=_ALPHA.value,
      min_response_fraction=min_response_fraction_flag.value(),
  )
  config = run_config.apply_field_overrides(config, field_overrides)

  os.makedirs(config.output_dir, exist_ok=True)

  log_file = os.path.join(config.output_dir, LOG_FILENAME)
  log = fdr_outputs.make_log_function(log_file, config.log_to_console)

  flags_path = os.path.join(config.output_dir, "flags.json")
  with open(flags_path, "w", encoding="utf-8") as f:
    json.dump(dataclasses.asdict(config), f, indent=2)

  # Append, never truncate: a run can be restarted or resumed from a
  # checkpoint, and the earlier attempt's log is the primary debugging
  # artifact. Each attempt is delimited by the banner below.
  log(
      "\n\n=== Standalone Precision Verifier Log Start"
      f" ({datetime.datetime.now().isoformat(timespec='seconds')}) ==="
  )
  log(f"Log File: {log_file}")
  if hypotheses_path:
    log(
        f"Loaded {len(hypotheses_with_labels)} hypothesis(es) from:"
        f" {hypotheses_path}"
    )
  for label, text in hypotheses_with_labels:
    log(f"  [{label}]: {text}")

  log("\n=== Verification Configuration ===")
  log(f"Verification Mode: {config.verification_mode}")
  log(f"Seed Instruction:  {config.seed_instruction}")
  if config.procedural_autorater:
    log(f"Autorater:         {config.procedural_autorater}")
  log(f"Target Model:      {config.target_model}")
  log(f"Agent Model:       {config.agent_model}")
  log(f"Parser Model:      {config.parser_model}")
  if not config.procedural_autorater:
    log(f"Rating Model:      {config.rating_model}")
    if config.summarizer_model:
      log(f"Summarizer Model:  {config.summarizer_model}")
  log(f"Single Batch N: {config.single_batch_num_queries} (base)")
  log(f"Single Batch Alpha: {config.single_batch_alpha}")
  if config.single_batch_null_prob is not None:
    log(f"Single Batch Null:  {config.single_batch_null_prob}")
  log(f"Scrape Bandwidth:  {config.scrape_bandwidth}")
  log(f"Output Dir:        {config.output_dir}")

  if existing is not None:
    await _retry_failed_records(existing, config, retry_statuses, log)
    return
  if _RETRY_FAILED.value:
    log("\n[Retry] No existing result to retry; verifying every hypothesis.")

  checked_records: list[precision_estimation.HypothesisPrecisionRecord] = []
  if not hypotheses_with_labels:
    log("\n[Notice] No non-empty hypotheses to check. Writing empty result.")
    results_data = {
        **scoring.score(
            [], failed_records=failed_records_flag.value()
        ).summary_fields(),
        "n_total": 0,
        "records": [],
        "config": dataclasses.asdict(config),
    }
  else:
    # Build hypothesis records with preserved labels
    hypotheses_to_check: list[tuple[str, records_lib.HypothesisRecord]] = [
        (label, records_lib.HypothesisRecord(text=text))
        for label, text in hypotheses_with_labels
    ]

    precision_res = await _run_precision_check(hypotheses_to_check, config, log)
    checked_records = precision_res.records

    score = scoring.score(
        precision_res.records, failed_records=failed_records_flag.value()
    )
    n_v = score.n_verified
    n_t = precision_res.n_total
    n_eval = precision_res.n_evaluated
    n_sf = precision_res.n_statistical_failures
    n_ee = precision_res.n_execution_errors
    n_vf = precision_res.n_verification_failures

    results_data = {
        **score.summary_fields(),
        "n_total": n_t,
        "n_evaluated": n_eval,
        "n_statistical_failures": n_sf,
        "n_execution_errors": n_ee,
        "n_verification_failures": n_vf,
        "records": [_record_to_dict(r) for r in precision_res.records],
        "config": _stored_config(config),
    }

    log("\n=== Precision Verification Complete ===")
    log(f"  Verified: {n_v}/{score.n_scored} scored (Total: {n_t})")
    if n_sf > 0:
      log(f"  Statistical Failures: {n_sf}/{n_eval}")
    if n_ee > 0:
      log(f"  Execution Errors: {n_ee}/{n_t}")
    if score.n_failed:
      log(f"  WARNING: {score.describe_failed()}.")
    log(f"  Precision: {scoring.format_precision(score.precision)}")
    log(f"  Empirical FDR (1 - Precision): {score.empirical_fdr:.4f}")

  results_data["data_quality"] = scoring.summarize_data_quality(checked_records)
  scoring.log_data_quality(results_data["data_quality"], log)

  out_path = os.path.join(config.output_dir, OUTPUT_FILENAME)
  with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results_data, f, indent=2)
  log(f"\nSaved verification results to: {out_path}")


def main(argv: list[str]) -> None:
  """CLI entry point for standalone precision verification."""
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  run_config.apply_yaml_config(
      field_names=run_config.config_field_names(verifier_config.VerifierConfig)
  )
  if not _HYPOTHESES_PATH.value and not _RETRY_FAILED.value:
    raise app.UsageError("--hypotheses_path is required.")

  asyncio.run(run_verifier())


if __name__ == "__main__":
  app.run(main)
