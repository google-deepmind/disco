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

r"""Unified CLI Runner for Generic Behavior & Capability Discovery Baselines.

Runs any or all of the 5 discovery baselines across any task from `tasks/registry.py`
(Recipe length comparisons, Epistemic Hedging, T2I biases, Visual Symmetry, etc.):

Methods:
  - direct_asking: Interrogates target model directly (Self-Report, Sheshadri et al., 2026)
  - simple_probing: Non-adaptive fixed probing (Uniform Probing, Sheshadri et al., 2026)
  - red_team: Adversarial auditor probing (Adaptive Red Teaming, Perez et al., 2022)
  - looped_investigator: Multi-round autonomous investigator (Investigator Agent, Sheshadri et al., 2026)
  - looped_multisample: Multi-round multi-sample pass@k probing (Repeated-Sample Probing, Sheshadri et al., 2026)
  - all: Runs all 5 baseline methods.

All requested baselines run Stage 1 hypothesis discovery IN PARALLEL with
per-method checkpoint/restart support. By default (`run_precision_check:
true`), Stage 2 precision verification runs in-job on each method's discovered
hypotheses; set `run_precision_check: false` in the `--config` YAML for
discovery-only runs (Stage 2 can then be run separately via
standalone_precision_verifier.py).

Only the settings worth changing per run have a command-line flag; every other
setting is a `BaselineRunSettings` or `VerifierConfig` field, set from the
`--config` YAML (see `configs/recipes_baselines.yaml`).

Usage:
  # Run all baselines for Visual Symmetry with budget=400:
  python -m disco.baselines.eval_discovery_baselines \
    --task_suite=visual_symmetry \
    --target_model=gemini-3.1-flash-image \
    --baseline_method=all \
    --budget=400 \
    --output_dir=./outputs/vs_baselines/

  # Run only LLM-based methods:
  python -m disco.baselines.eval_discovery_baselines \
    --task_suite=visual_symmetry \
    --target_model=gemini-3.1-flash-image \
    --baseline_method=llm_methods \
    --budget=400 \
    --output_dir=./outputs/visual_baselines/
"""

import asyncio
from collections.abc import Mapping
import dataclasses
import json
import os
import time
import traceback
from typing import Any, Callable

from absl import app
from absl import flags

from disco import precision_estimation
from disco import standalone_agent
from disco import verifier_config
from disco.baselines import common as baselines_common
from disco.baselines.adapters import base as adapters_base
from disco.baselines.adapters import registry as adapters_registry
from disco.baselines.methods import base as methods_base
from disco.baselines.methods import registry as methods_registry
from disco.core import records as records_lib
from disco.core import run_config
from disco.eval import output_schema
from disco.models import genai_client
from disco.models import registry as model_registry
from disco.tasks import registry as task_registry
from disco.verification import failed_records_flag
from disco.verification import min_response_fraction_flag
from disco.verification import scoring

_TARGET_MODEL = flags.DEFINE_string(
    "target_model",
    model_registry.DEFAULT_GEMINI_MODEL,
    "Target model to discover capabilities/behaviors for.",
)
_BACKBONE_MODEL = flags.DEFINE_string(
    "backbone_model",
    "",
    "Model for every auxiliary role (investigator, T2I judge, and the"
    " verifier's agent, parser, rating and summarizer models) that is not"
    " set explicitly. Empty keeps each role's default.",
)
_INVESTIGATOR_MODEL = flags.DEFINE_string(
    "investigator_model",
    model_registry.DEFAULT_GEMINI_MODEL,
    "Investigator / auxiliary LLM model. Defaults to --backbone_model when"
    " that is set.",
)
_TASK_SUITE = flags.DEFINE_string(
    "task_suite",
    "recipes",
    "Task suite name from registry (e.g. 'recipes', 'hedging',"
    " 'visual_symmetry', 't2i_cat_bias', 't2i_handedness_bias').",
)
_TASK_NAME = flags.DEFINE_string(
    "task_name",
    "all",
    "Specific task key/name within the suite or 'all'.",
)
_BASELINE_METHOD = flags.DEFINE_string(
    "baseline_method",
    "all",
    "Which discovery baseline method(s) to run: 'all', 'llm_methods',"
    " 'empirical_methods', or a comma-separated list of methods.",
)
_BUDGET = flags.DEFINE_integer(
    "budget",
    20,
    "Target probe query budget per baseline method.",
)
_MAX_PREDICTIONS = flags.DEFINE_integer(
    "max_predictions",
    10,
    "Maximum number of candidate predictions to parse from synthesis.",
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    "./outputs/generic_discovery_baselines/",
    "Directory to save evaluation results.",
)
_EXPERIMENT_NAME = flags.DEFINE_string(
    "experiment_name",
    "baseline_run",
    "Name for this experiment run.",
)
_SEED_INSTRUCTION = flags.DEFINE_string(
    "seed_instruction",
    "",
    "Explicit seed instruction override (defaults to task metadata if empty).",
)
_REPLICA = flags.DEFINE_integer(
    "replica",
    0,
    "Replica index for random sampling and seed variation.",
)

# ---------------------------------------------------------------------------
# Precision Verification Flags
# ---------------------------------------------------------------------------
_ALPHA = flags.DEFINE_float(
    "alpha",
    0.05,
    "Significance level of the Stage 2 single-batch p-value test.",
)
_NUM_QUERIES = flags.DEFINE_integer(
    "num_queries",
    100,
    "Base number of Stage 2 verification queries per hypothesis (doubled if"
    " paired).",
)
_TOTAL_VERIFICATION_BUDGET = flags.DEFINE_integer(
    "total_verification_budget",
    0,
    "Total verification query budget split across all predictions. If > 0,"
    " overrides num_queries to (total_verification_budget // num_predictions).",
)


@dataclasses.dataclass(frozen=True)
class BaselineRunSettings:
  """Run settings with no command-line flag, settable from the `--config` YAML.

  Attributes:
    red_team_batch_size: Probes the red-team baseline asks the investigator for
      per generation round.
    random_seed: Base random seed for probe sampling (`--replica` is added).
    procedural_autorater: Name of the Stage 2 procedural autorater (e.g.
      'count_words', 'count_hedges'). Empty inherits the task definition's.
    run_precision_check: Whether to run Stage 2 precision verification in-job on
      each method's discovered hypotheses.
    direct_asking_total_verification_budget: Separate total verification query
      budget for the direct_asking baseline. -1 inherits
      `--total_verification_budget`.
    single_batch_null_prob: Manual override of the Stage 2 null rate. None
      auto-extracts it per hypothesis.
    concurrency: Maximum concurrent GenAI API requests in flight.
    max_retries: Attempts per model call (target and investigator) before a
      failing call is given up on. Only raised errors are retried here; empty
      responses are resampled at the probe level (see baselines/README.md).
    base_retry_delay: Seconds before the first retry of a failing model call
      (target and investigator). Each further retry doubles it, with +/-20%
      jitter, up to 60s.
  """

  red_team_batch_size: int = 15
  random_seed: int = 42
  procedural_autorater: str = ""
  run_precision_check: bool = True
  direct_asking_total_verification_budget: int = -1
  single_batch_null_prob: float | None = None
  concurrency: int = 12
  max_retries: int = 10
  base_retry_delay: float = 3.0


def load_run_settings() -> tuple[BaselineRunSettings, dict[str, Any]]:
  """Applies the `--config` YAML and returns the YAML-only settings.

  YAML keys that name a flag set the flag (see `run_config.apply_yaml_config`).
  The remaining keys must name a `BaselineRunSettings` or `VerifierConfig`
  field; a key both have (e.g. `procedural_autorater`,
  `single_batch_null_prob`) is a `BaselineRunSettings` field, which
  `_build_verifier_config` passes on to the verifier.

  Returns:
    The run settings, and the `VerifierConfig` field overrides for
    `_build_verifier_config`.
  """
  settings_fields = run_config.config_field_names(BaselineRunSettings)
  overrides = run_config.apply_yaml_config(
      field_names=settings_fields
      | run_config.config_field_names(verifier_config.VerifierConfig)
  )
  settings = run_config.apply_field_overrides(
      BaselineRunSettings(),
      {k: v for k, v in overrides.items() if k in settings_fields},
  )
  verifier_overrides = {
      k: v for k, v in overrides.items() if k not in settings_fields
  }
  return settings, verifier_overrides


def resolve_auxiliary_model(flag_value: str) -> str:
  """Resolves an auxiliary model flag, deferring to `--backbone_model`.

  Args:
    flag_value: The flag's value. Its default, `DEFAULT_GEMINI_MODEL`, means the
      flag was not set explicitly.

  Returns:
    The resolved model ID: the flag's value if set explicitly, else
    `--backbone_model` if set, else the default model.
  """
  if flag_value and flag_value != model_registry.DEFAULT_GEMINI_MODEL:
    return model_registry.resolve_model_url(flag_value)
  return model_registry.resolve_model_url(
      _BACKBONE_MODEL.value or model_registry.DEFAULT_GEMINI_MODEL
  )


# ---------------------------------------------------------------------------
# Method name <-> class name mapping for checkpoint matching
# ---------------------------------------------------------------------------
_METHOD_TO_CLASS_NAME: dict[str, str] = {
    "direct_asking": "DirectAskingBaseline",
    "simple_probing": "SimpleProbingBaseline",
    "red_team": "RedTeamProbingBaseline",
    "looped_investigator": "LoopedInvestigatorBaseline",
    "looped_multisample": "LoopedMultiSampleBaseline",
}


def _make_log_fn(
    log_file_path: str, log_to_console: bool = True
) -> Callable[..., None]:
  """Creates a logger writing to file and optionally to console."""

  def log(*args: Any, mode: str = "a", **kwargs: Any) -> None:
    del kwargs
    msg = " ".join(str(a) for a in args)
    if log_to_console:
      print(msg)
    try:
      with open(log_file_path, mode, encoding="utf-8") as f:
        f.write(msg + "\n")
    except OSError:
      pass

  return log


def _checkpoint_exists(method_dir: str) -> bool:
  """Checks if a method has a completed Stage 1 checkpoint."""
  return os.path.exists(os.path.join(method_dir, "round_summary.json"))


def _load_precision_checkpoint(
    method_dir: str,
    expected_budget: int,
    expected_n_hyps: int,
) -> dict[str, Any] | None:
  """Loads a completed Stage 2 precision result, if one is reusable.

  Stage 2 spends the full verification budget against the target model, so a
  job that is restarted after verification already finished (e.g. preempted
  while writing the task-level results) must not pay for it twice.

  The checkpoint is only reused when it was produced under the same budget and
  covers the same number of hypotheses, mirroring the Stage 1 guard: a changed
  budget means a different measurement, not a resumable one.

  Args:
    method_dir: Per-method output directory.
    expected_budget: `total_verification_budget` for the current run.
    expected_n_hyps: Number of hypotheses about to be verified.

  Returns:
    The cached precision summary, or None if absent or not reusable.
  """
  eval_path = os.path.join(method_dir, "evaluation_results.json")
  if not os.path.exists(eval_path):
    return None
  try:
    with open(eval_path, "r", encoding="utf-8") as f:
      data = json.load(f)
  except (OSError, json.JSONDecodeError, ValueError, TypeError):
    return None

  if not isinstance(data, dict):
    return None
  cached = data.get("precision_check")
  if not isinstance(cached, dict) or not cached.get("records"):
    return None
  if cached.get("total_verification_budget") != expected_budget:
    return None
  if len(cached["records"]) != expected_n_hyps:
    return None
  return cached


def _load_checkpoint(
    method_dir: str,
) -> baselines_common.GenericBaselineResult | None:
  """Loads a completed method's result from its checkpoint files."""
  summary_path = os.path.join(method_dir, "round_summary.json")
  hyp_path = os.path.join(method_dir, "hypotheses.json")
  if not os.path.exists(summary_path):
    return None
  try:
    with open(summary_path, "r", encoding="utf-8") as f:
      summary_data = json.load(f)
    predictions = summary_data.get("predictions", [])
    if not predictions and os.path.exists(hyp_path):
      with open(hyp_path, "r", encoding="utf-8") as f:
        hyp_data = json.load(f)
        predictions = hyp_data.get("hypotheses", [])
    return baselines_common.GenericBaselineResult(
        method_name=summary_data.get("method_name", ""),
        origin=summary_data.get("origin", ""),
        predictions=predictions,
        total_target_queries=summary_data.get("total_target_queries", 0),
        empirical_rates=summary_data.get("empirical_rates", {}),
    )
  except (
      OSError,
      json.JSONDecodeError,
      ValueError,
      TypeError,
      KeyError,
      AttributeError,
  ):
    return None


def _save_stage1_results(
    res: baselines_common.GenericBaselineResult,
    m_dir: str,
    flags_summary: dict[str, Any],
) -> None:
  """Saves Stage 1 (discovery) results for a single method."""
  os.makedirs(m_dir, exist_ok=True)

  # Save detailed probe round summary (this is the checkpoint marker)
  with open(
      os.path.join(m_dir, "round_summary.json"), "w", encoding="utf-8"
  ) as f:
    json.dump(res.to_dict(), f, indent=2)

  # Save per-method flags
  method_flags = dict(flags_summary)
  method_flags["method"] = res.method_name
  method_flags["method_name"] = res.method_name
  method_flags["origin"] = res.origin
  method_flags["hypotheses_count"] = len(res.predictions)
  with open(os.path.join(m_dir, "flags.json"), "w", encoding="utf-8") as f:
    json.dump(method_flags, f, indent=2)

  # Save per-method hypotheses.json (for Stage 2 consumption)
  with open(os.path.join(m_dir, "hypotheses.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "hypotheses": res.predictions,
            "method": res.method_name,
            "method_name": res.method_name,
            "origin": res.origin,
            "n_discoveries": len(res.predictions),
            "total_target_queries": res.total_target_queries,
        },
        f,
        indent=2,
    )


def _build_verifier_config(
    task: Any,
    class_name: str,
    task_dir: str,
    target_model: str,
    resolved_seed_instruction: str,
    settings: BaselineRunSettings,
    method_name: str = "",
    field_overrides: Mapping[str, Any] | None = None,
) -> verifier_config.VerifierConfig:
  """Builds a VerifierConfig dataclass instance for a baseline method verification run.

  Args:
    task: The registry task being verified.
    class_name: Baseline method class name (output subdirectory).
    task_dir: Output directory of the task.
    target_model: Resolved target model.
    resolved_seed_instruction: Seed instruction for the verifier prompts.
    settings: The run's YAML-only settings.
    method_name: Baseline method key, for method-specific budgets.
    field_overrides: YAML-only `VerifierConfig` fields from `--config` (see
      `load_run_settings`), applied last.

  Returns:
    Configured `VerifierConfig` instance.
  """
  if settings.procedural_autorater:
    resolved_autorater = settings.procedural_autorater
  else:
    resolved_autorater = (
        getattr(task, "procedural_autorater", "") or ""
    ) or task.metadata.get("procedural_autorater", "")

  v_output_dir = os.path.join(task_dir, class_name, "precision")

  is_direct_asking = method_name == "direct_asking" or class_name in (
      "DirectAskingBaseline",
      "direct_asking",
  )
  if is_direct_asking and settings.direct_asking_total_verification_budget >= 0:
    v_budget = settings.direct_asking_total_verification_budget
  else:
    v_budget = _TOTAL_VERIFICATION_BUDGET.value

  return run_config.apply_field_overrides(
      verifier_config.VerifierConfig(
          target_model=model_registry.resolve_model_url(target_model),
          # Fans out to the agent, parser, rating and summarizer models unless
          # a YAML override sets one of them explicitly.
          backbone_model=(
              model_registry.resolve_model_url(_BACKBONE_MODEL.value)
              if _BACKBONE_MODEL.value
              else ""
          ),
          alpha=_ALPHA.value,
          output_dir=v_output_dir,
          experiment_name=class_name,
          seed_instruction=resolved_seed_instruction,
          # Lets the verifier dispatch on modality (e.g. visual_symmetry probes
          # are images) instead of inferring it from the seed instruction text.
          task_suite=_TASK_SUITE.value,
          # Reference rate the task is defined against (0.90 for visual
          # symmetry, from VISUAL_SYMMETRY_TASK.metadata). Only a fallback: an
          # explicit `single_batch_null_prob` or a rate stated in the
          # hypothesis text wins.
          task_null_proportion=float(
              task.metadata.get("null_proportion", 0.0) or 0.0
          ),
          procedural_autorater=resolved_autorater,
          single_batch_num_queries=_NUM_QUERIES.value,
          total_verification_budget=v_budget,
          single_batch_alpha=_ALPHA.value,
          single_batch_null_prob=settings.single_batch_null_prob,
          min_response_fraction=min_response_fraction_flag.value(),
      ),
      field_overrides or {},
  )


async def _run_in_job_precision_check(
    method_name: str,
    predictions: list[str],
    v_cfg: verifier_config.VerifierConfig,
    log: Callable[..., None],
) -> dict[str, Any]:
  """Runs in-job precision verification on discovered hypotheses."""
  if not predictions:
    return {
        **scoring.score(
            [], failed_records=failed_records_flag.value()
        ).summary_fields(),
        "n_total": 0,
        "n_evaluated": 0,
        "n_statistical_failures": 0,
        "n_execution_errors": 0,
        "verification_mode": v_cfg.verification_mode,
        "records": [],
    }

  log(f"\n{'=' * 60}")
  log(
      f" Stage 2: Precision Verification for {method_name}"
      f" ({len(predictions)} hypotheses)"
  )
  log(f"{'=' * 60}")
  log(f"Verification mode: {v_cfg.verification_mode}")
  if v_cfg.total_verification_budget > 0:
    log(
        f"Total verif budget: {v_cfg.total_verification_budget}"
        f" ({max(1, v_cfg.total_verification_budget // len(predictions))}/hyp"
        " base)"
    )
  else:
    log(f"Single batch N:    {v_cfg.single_batch_num_queries} (per hyp base)")
  log(f"Target model:      {v_cfg.target_model}")
  log(f"Agent model:       {v_cfg.agent_model}")
  log(f"Parser model:      {v_cfg.parser_model}")
  log(f"Rating model:      {v_cfg.rating_model}")
  if v_cfg.procedural_autorater:
    log(f"Procedural autorater: {v_cfg.procedural_autorater}")
  log(f"Output directory:  {v_cfg.output_dir}")

  hyps_to_check = [
      (f"{method_name}_h{i}", records_lib.HypothesisRecord(text=h))
      for i, h in enumerate(predictions)
  ]

  prec_res = await precision_estimation.run_precision_check(
      verified_hypotheses=hyps_to_check,
      config=v_cfg,
      log_fn=log,
  )

  score = scoring.score(
      prec_res.records, failed_records=failed_records_flag.value()
  )
  precision_summary = {
      **score.summary_fields(),
      "n_total": prec_res.n_total,
      "n_evaluated": prec_res.n_evaluated,
      "n_statistical_failures": prec_res.n_statistical_failures,
      "n_execution_errors": prec_res.n_execution_errors,
      "verification_mode": v_cfg.verification_mode,
      "total_verification_budget": v_cfg.total_verification_budget,
      "records": [
          {
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
          }
          for r in prec_res.records
      ],
  }
  precision_summary["data_quality"] = scoring.summarize_data_quality(
      prec_res.records
  )

  log(f"\n[SUMMARY] {method_name} Verification Results:")
  log(
      f"  Verified: {score.n_verified}/{score.n_scored} scored"
      f" (Total: {prec_res.n_total})"
  )
  if score.n_failed:
    log(f"  WARNING: {score.describe_failed()}.")
  log(f"  Precision: {scoring.format_precision(score.precision)}")
  log(f"  Empirical FDR: {score.empirical_fdr:.4f}")
  scoring.log_data_quality(precision_summary["data_quality"], log)

  return precision_summary


async def _run_single_method(
    method_name: str,
    adapter: adapters_base.TaskProbeAdapter,
    client: Any,
    target_model: str,
    investigator_model: str,
    budget: int,
    max_predictions: int,
    seed: int,
    red_team_batch_size: int,
    semaphore: asyncio.Semaphore,
    task_dir: str,
    flags_summary: dict[str, Any],
    shared_log: Callable[..., None],
) -> tuple[str, baselines_common.GenericBaselineResult | None, str | None]:
  """Runs a single baseline method with checkpoint/restart support.

  Args:
    method_name: Name of the baseline method to run.
    adapter: Task-specific probe adapter.
    client: GenAI client instance.
    target_model: Target model URL.
    investigator_model: Investigator model URL.
    budget: Query budget for this method.
    max_predictions: Maximum predictions to keep.
    seed: Sampling seed of simple probing.
    red_team_batch_size: Batch size for red team probing.
    semaphore: Concurrency semaphore.
    task_dir: Output directory for this task.
    flags_summary: Experiment configuration dict.
    shared_log: Logging function for the experiment-wide `baseline.log`. It is
      teed into a per-method log below, once the method directory is known.

  Returns:
    Tuple of (method_name, result_or_None, error_message_or_None).
  """
  class_name = _METHOD_TO_CLASS_NAME.get(method_name, method_name)
  m_dir = os.path.join(task_dir, class_name)

  # Check for existing checkpoint (restart logic)
  if _checkpoint_exists(m_dir):
    cached = _load_checkpoint(m_dir)
    if cached is not None:
      # If requested budget is already fulfilled by the checkpoint, reuse it.
      if cached.total_target_queries >= budget:
        shared_log(
            f"[CHECKPOINT] Reusing {method_name} — found completed"
            f" checkpoint at {m_dir} with {cached.total_target_queries} queries"
            f" (requested budget: {budget},"
            f" {len(cached.predictions)} predictions)"
        )
        return (method_name, cached, None)
      else:
        shared_log(
            f"[CHECKPOINT] Found existing checkpoint for {method_name} with"
            f" lower queries ({cached.total_target_queries} < {budget})."
            f" Continuing with higher budget {budget}..."
        )
    else:
      shared_log(
          "[CHECKPOINT] Found corrupt checkpoint for"
          f" {method_name}, re-running."
      )

  # Create a per-method adapter copy to isolate images and observation caches
  method_adapter = adapters_registry.get_task_adapter(
      adapter.task,
      seed_instruction=adapter.seed_instruction,
      judge_model=_BACKBONE_MODEL.value,
  )
  if hasattr(method_adapter, "image_save_dir"):
    setattr(method_adapter, "image_save_dir", os.path.join(m_dir, "images"))

  # The methods run concurrently and all write into the one shared
  # `baseline.log`, so its lines interleave and cannot be attributed back to a
  # method. Tee each method's output into its own file as well; the shared log
  # is still written, so nothing that worked before stops working.
  os.makedirs(m_dir, exist_ok=True)
  method_log_path = os.path.join(m_dir, "method.log")
  method_file_log = _make_log_fn(method_log_path, log_to_console=False)

  def log(*args: Any, **kwargs: Any) -> None:
    shared_log(*args, **kwargs)
    method_file_log(*args, **kwargs)

  log(f"\n>>> Running baseline: {method_name} ...")
  start_time = time.time()
  try:
    method = methods_registry.METHODS.get(method_name)
    if method is None:
      return (method_name, None, f"Unknown method: {method_name}")
    plan = method.plan(
        budget,
        methods_base.PlanOptions(
            seed=seed, red_team_batch_size=red_team_batch_size
        ),
    )
    res = await method.run(
        method_adapter,
        methods_base.RunContext(
            client=client,
            target_model=target_model,
            investigator_model=investigator_model,
            max_predictions=max_predictions,
            checkpoint_dir=m_dir,
            semaphore=semaphore,
            log=log,
            min_response_fraction=min_response_fraction_flag.value(),
        ),
        **plan,
    )

    if max_predictions > 0 and len(res.predictions) > max_predictions:
      res.predictions = res.predictions[:max_predictions]
      res.structured_hypotheses = res.structured_hypotheses[:max_predictions]

    elapsed = time.time() - start_time
    log(
        f"\n[DONE] {method_name} completed in {elapsed:.1f}s:"
        f" {len(res.predictions)} predictions,"
        f" {res.total_target_queries} target queries"
    )

    # Save Stage 1 checkpoint immediately
    _save_stage1_results(res, m_dir, flags_summary)
    log(f"[CHECKPOINT] Saved Stage 1 results to {m_dir}")

    return (method_name, res, None)

  except Exception as e:  # pylint: disable=broad-exception-caught  # Last-resort parallel runner guard: isolate per-method failures so completed sibling methods keep saved checkpoints.
    elapsed = time.time() - start_time
    err_msg = (
        f"{method_name} failed after {elapsed:.1f}s:"
        f" {e}\n{traceback.format_exc()}"
    )
    # The method is reported under `methods_failed`; say so where it is seen.
    log(
        f"\n[ERROR] Baseline method {method_name} FAILED and produced no"
        f" hypotheses (reported under methods_failed): {err_msg}"
    )
    return (method_name, None, err_msg)


run_single_method = _run_single_method


_LLM_SYNTHESIS_METHODS: tuple[str, ...] = (
    "direct_asking",
    "red_team",
    "looped_investigator",
    "looped_multisample",
)
_EMPIRICAL_CATEGORY_METHODS: tuple[str, ...] = ("simple_probing",)


def parse_baseline_methods(raw_method_str: str) -> list[str]:
  """Resolves a `--baseline_method` spec into an ordered list of method names."""
  all_known_methods = list(methods_registry.METHODS.keys())
  group_aliases: dict[str, list[str]] = {
      "all": all_known_methods,
      "llm_methods": list(_LLM_SYNTHESIS_METHODS),
      "empirical_methods": list(_EMPIRICAL_CATEGORY_METHODS),
  }
  tokens = [
      t.strip() for t in raw_method_str.strip().lower().split(",") if t.strip()
  ]
  if not tokens:
    return all_known_methods
  methods_to_run: list[str] = []
  for token in tokens:
    if token in group_aliases:
      for m in group_aliases[token]:
        if m not in methods_to_run:
          methods_to_run.append(m)
    elif token in all_known_methods:
      if token not in methods_to_run:
        methods_to_run.append(token)
    else:
      raise ValueError(
          f"Unknown baseline method {token!r}. Valid options:"
          f" {all_known_methods} or 'llm_methods', 'empirical_methods', 'all'."
      )
  return methods_to_run


async def run_discovery_pipeline() -> None:
  """Runs all requested discovery baselines in parallel with checkpointing."""
  settings, verifier_field_overrides = load_run_settings()
  target_model = model_registry.resolve_model_url(_TARGET_MODEL.value)
  investigator_model = resolve_auxiliary_model(_INVESTIGATOR_MODEL.value)

  experiment_dir = os.path.join(
      os.path.abspath(_OUTPUT_DIR.value), _EXPERIMENT_NAME.value
  )
  os.makedirs(experiment_dir, exist_ok=True)

  log_file = os.path.join(experiment_dir, "baseline.log")
  log = _make_log_fn(log_file, log_to_console=True)

  log("=" * 60)
  log(" Generic Behavior & Capability Discovery Baselines")
  log("=" * 60)
  log(f"Task suite:        {_TASK_SUITE.value}")
  log(f"Task name:         {_TASK_NAME.value}")
  log(f"Baseline method:   {_BASELINE_METHOD.value}")
  log(f"Target model:      {target_model}")
  log(f"Investigator:      {investigator_model}")
  if _BACKBONE_MODEL.value:
    log(f"Backbone model:    {_BACKBONE_MODEL.value}")
  log(f"Query budget:      {_BUDGET.value}")
  log(f"Concurrency:       {settings.concurrency} max in-flight requests")
  # Configure retry parameters for model calls.
  baselines_common.set_target_max_retries(settings.max_retries)
  log(f"Max retries:       {settings.max_retries} attempts per model call")
  baselines_common.set_target_base_retry_delay(settings.base_retry_delay)
  log(f"Base retry delay:  {settings.base_retry_delay:g}s, doubling per retry")
  log(
      "Min response frac: "
      f"{min_response_fraction_flag.value():g} of each probe's samples"
  )
  log(f"Output directory:  {experiment_dir}")

  if settings.run_precision_check:
    # Stage 2 is the first thing that builds a candidate for the target model,
    # so an unusable target model would otherwise only surface once the whole
    # Stage 1 discovery budget has already been spent. Pay that validation now.
    log("Validating target model for Stage 2 precision verification...")
    standalone_agent.prewarm_candidate(target_model)

  semaphore = asyncio.Semaphore(settings.concurrency)

  # Load tasks
  tasks = task_registry.get_tasks(
      suite=_TASK_SUITE.value, split=_TASK_NAME.value
  )
  if not tasks:
    raise ValueError(
        f"No tasks found for suite={_TASK_SUITE.value!r},"
        f" name={_TASK_NAME.value!r}"
    )

  client = genai_client.get_genai_client()

  methods_to_run = parse_baseline_methods(_BASELINE_METHOD.value)
  seed = settings.random_seed + _REPLICA.value

  suite_results: dict[str, list[dict[str, Any]]] = {}

  for task in tasks:
    log(f"\n{'=' * 60}")
    log(f"Active Task:       {task.name} ({task.description})")
    log(f"{'=' * 60}")

    task_dir = (
        os.path.join(experiment_dir, task.key)
        if len(tasks) > 1
        else experiment_dir
    )
    os.makedirs(task_dir, exist_ok=True)

    resolved_seed_instruction = (
        _SEED_INSTRUCTION.value
        if _SEED_INSTRUCTION.value
        else task.metadata.get("seed_instruction", "")
    )
    if resolved_seed_instruction:
      log(f"Seed Instruction:  {resolved_seed_instruction}")

    adapter = adapters_registry.get_task_adapter(
        task,
        seed_instruction=resolved_seed_instruction,
        judge_model=_BACKBONE_MODEL.value,
    )

    # The effective configuration: the flags, then the YAML-only settings.
    # `max_predictions` is read back by `stats/ebh_family.py`.
    flags_summary = {
        "task_suite": _TASK_SUITE.value,
        "task_name": task.name,
        "target_model": target_model,
        "backbone_model": _BACKBONE_MODEL.value,
        "investigator_model": investigator_model,
        "baseline_method": _BASELINE_METHOD.value,
        "seed_instruction": resolved_seed_instruction,
        "budget": _BUDGET.value,
        "max_predictions": _MAX_PREDICTIONS.value,
        "replica": _REPLICA.value,
        "seed": seed,
        "verification_mode": "single_batch_p_value",
        "alpha": _ALPHA.value,
        "num_queries": _NUM_QUERIES.value,
        "total_verification_budget": _TOTAL_VERIFICATION_BUDGET.value,
        **dataclasses.asdict(settings),
        "verifier_field_overrides": dict(verifier_field_overrides),
    }
    with open(os.path.join(task_dir, "flags.json"), "w", encoding="utf-8") as f:
      json.dump(flags_summary, f, indent=2)

    # =================================================================
    # Run discovery baselines IN PARALLEL with checkpointing
    # =================================================================
    log(
        "\n--- Hypothesis Discovery"
        f" ({len(methods_to_run)} methods in parallel) ---"
    )

    method_tasks = [
        _run_single_method(
            method_name=m,
            adapter=adapter,
            client=client,
            target_model=target_model,
            investigator_model=investigator_model,
            budget=_BUDGET.value,
            max_predictions=_MAX_PREDICTIONS.value,
            seed=seed,
            red_team_batch_size=settings.red_team_batch_size,
            semaphore=semaphore,
            task_dir=task_dir,
            flags_summary=flags_summary,
            shared_log=log,
        )
        for m in methods_to_run
    ]

    parallel_results = await asyncio.gather(*method_tasks)

    # Collect results and report summary
    method_results: dict[str, baselines_common.GenericBaselineResult] = {}
    errors: list[str] = []
    for method_name, result, error in parallel_results:
      if result is not None:
        method_results[method_name] = result
      if error is not None:
        errors.append(error)

    log("\n--- Discovery Summary ---")
    log(f"  Completed: {len(method_results)}/{len(methods_to_run)} methods")
    for m_name, res in method_results.items():
      log(
          f"  {m_name}: {len(res.predictions)} predictions,"
          f" {res.total_target_queries} queries"
      )
    if errors:
      log(f"  Errors: {len(errors)}")
      for err in errors:
        log(f"    - {err[:200]}")

    # Save discovery summary (all methods' hypotheses in one file)
    discovery_summary = {
        "task_suite": _TASK_SUITE.value,
        "task_name": task.name,
        "stage": "discovery",
        "methods_completed": list(method_results.keys()),
        "methods_failed": [e[:100] for e in errors],
        "config": flags_summary,
        "method_hypotheses": {
            m_name: {
                "method_name": res.method_name,
                "origin": res.origin,
                "predictions": res.predictions,
                "n_discoveries": len(res.predictions),
                "total_target_queries": res.total_target_queries,
            }
            for m_name, res in method_results.items()
        },
    }
    discovery_path = os.path.join(task_dir, "stage1_discovery_results.json")
    with open(discovery_path, "w", encoding="utf-8") as f:
      json.dump(discovery_summary, f, indent=2)
    log(f"\nSaved discovery results to: {discovery_path}")

    # =================================================================
    # Stage 2: In-Job Precision Verification (if enabled)
    # =================================================================
    task_results: list[dict[str, Any]] = []

    for m_name, res in method_results.items():
      class_name = _METHOD_TO_CLASS_NAME.get(m_name, m_name)
      m_dir = os.path.join(task_dir, class_name)
      precision_summary = None

      if settings.run_precision_check and res.predictions:
        v_cfg = _build_verifier_config(
            task=task,
            class_name=class_name,
            task_dir=task_dir,
            target_model=target_model,
            resolved_seed_instruction=resolved_seed_instruction,
            settings=settings,
            method_name=m_name,
            field_overrides=verifier_field_overrides,
        )
        # Stage 2 spends the full verification budget against the target model.
        # Reuse a matching completed result so a restart after verification
        # finished does not pay for it a second time.
        precision_summary = _load_precision_checkpoint(
            m_dir,
            expected_budget=v_cfg.total_verification_budget,
            expected_n_hyps=len(res.predictions),
        )
        if precision_summary is not None:
          log(
              f"[CHECKPOINT] Reusing {m_name} precision check — found completed"
              f" verification at {m_dir} with"
              f" {len(precision_summary['records'])} records"
              f" (budget: {v_cfg.total_verification_budget})"
          )
        else:
          precision_summary = await _run_in_job_precision_check(
              method_name=m_name,
              predictions=res.predictions,
              v_cfg=v_cfg,
              log=log,
          )

      if precision_summary and "records" in precision_summary:
        verified_hyps = [
            r["hypothesis_text"]
            for r in precision_summary["records"]
            if r.get("verified")
        ]
      else:
        verified_hyps = res.predictions

      method_eval_dict: dict[str, Any] = output_schema.DiscoveryMethodResult(
          method_name=res.method_name,
          origin=res.origin,
          predictions=res.predictions,
          verified_hypotheses=verified_hyps,
          total_target_queries=res.total_target_queries,
          precision_check=precision_summary,
          config=flags_summary,
      ).to_json()

      # Save per-method evaluation_results.json
      with open(
          os.path.join(m_dir, "evaluation_results.json"), "w", encoding="utf-8"
      ) as f:
        json.dump(method_eval_dict, f, indent=2)
      log(
          "[EVALUATION] Saved method results to"
          f" {m_dir}/evaluation_results.json"
      )

      task_results.append(method_eval_dict)

    # Save task-level evaluation_results.json
    task_eval_path = os.path.join(task_dir, "evaluation_results.json")
    with open(task_eval_path, "w", encoding="utf-8") as f:
      json.dump(
          output_schema.DiscoveryTaskResults(
              task_suite=_TASK_SUITE.value,
              task_name=task.name,
              methods=task_results,
              config=flags_summary,
          ).to_json(),
          f,
          indent=2,
      )
    log(f"\nSaved task results to: {task_eval_path}")

    suite_results[task.key] = task_results

  if len(tasks) > 1:
    suite_path = os.path.join(experiment_dir, "evaluation_results.json")
    with open(suite_path, "w", encoding="utf-8") as f:
      json.dump(
          output_schema.DiscoverySuiteResults(
              task_suite=_TASK_SUITE.value,
              task_name=_TASK_NAME.value,
              suite_results=suite_results,
          ).to_json(),
          f,
          indent=2,
      )
    log(f"\nSaved suite results to: {suite_path}")


def main(argv: list[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  asyncio.run(run_discovery_pipeline())


if __name__ == "__main__":
  app.run(main)
