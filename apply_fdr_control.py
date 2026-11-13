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

r"""Single-run post-hoc FDR control and estimation on verification outputs.

Applies Benjamini-Hochberg (BH), Benjamini-Yekutieli (BY), un-boosted e-BH,
and boosted e-BH False Discovery Rate (FDR) control procedures, as well as
Storey pi0 and thresholded empirical FDR estimators, to a verification or
evaluation output file or single run directory.

Statistical foundations:
  1. Benjamini-Hochberg (BH 1995): Controls FDR at level alpha under the
     Positive Regression Dependence on a Subset (PRDS) condition.
  2. Benjamini-Yekutieli (BY 2001): Controls FDR at level alpha under arbitrary
     dependence by scaling the significance level by the harmonic penalty
     c(m) = sum_{j=1}^m 1/j.
  3. Un-boosted e-BH (Wang & Ramdas 2022): Calibrates p-values to e-values via
     e = kappa * p^(kappa - 1) and applies e-BH. Controls FDR under arbitrary
     dependence with no assumptions.
  4. Boosted e-BH: Applies per-hypothesis boosting factor b >= 1 derived from
     e-value calibrator properties (stats/boosting.compute_boost_factor).
     Substantially closes the efficiency gap with BY under arbitrary
     dependence while remaining assumption-free.

The reported "unadjusted" baseline is recomputed from scratch at the caller's
--alpha (p <= alpha). It deliberately ignores the `verified` flag stored by the
original verification run, which was decided at that run's own alpha; this is a
separate procedure, not a replay.

Supported input layouts:
  Layout A: Nested `precision_check` inside `evaluation_results.json`.
  Layout B: Standalone `verification_results.json` in the run directory or a
            verification subdirectory.

Results are written to an `ebh_fdr/` subdirectory placed beside the input file.
"""

from collections.abc import Sequence
import dataclasses
import json
import os
from typing import Any

from absl import app
from absl import flags
from absl import logging

from disco.stats import ebh_family
from disco.stats import estimators
from disco.stats import multiple_testing

# Name of the output subdirectory written beside the input file.
OUTPUT_SUBDIR = "ebh_fdr"

# Recorded in the outputs: the BH / BY family is always the successfully
# evaluated hypotheses (status == "OK"), never the errored candidates.
FAMILY_SIZE_MODE = "evaluated"

_INPUT_PATH = flags.DEFINE_string(
    "input_path",
    "",
    "Direct path to a verification_results.json or evaluation_results.json"
    " file to process.",
)
_RUN_DIR = flags.DEFINE_string(
    "run_dir", "", "Single run directory to process."
)
_SOURCE = flags.DEFINE_enum(
    "source",
    "auto",
    ["auto", "evaluation", "verification"],
    "Which input layout to read: 'evaluation' (evaluation_results.json),"
    " 'verification' (verification_results.json), or 'auto'.",
)
_ALPHA = flags.DEFINE_float(
    "alpha", 0.05, "Target FDR significance level alpha in (0, 1)."
)
_KAPPA = flags.DEFINE_float(
    "kappa", 0.1741, "p-to-e calibrator exponent parameter kappa in (0, 1)."
)
_EBH_FAMILY_SIZE = flags.DEFINE_enum(
    "ebh_family_size",
    ebh_family.DEFAULT_MODE,
    list(ebh_family.MODES),
    ebh_family.HELP,
)
_MAX_PREDICTIONS = flags.DEFINE_integer(
    "max_predictions",
    0,
    "Pre-registered hypothesis cap used when --ebh_family_size=max_predictions."
    " Leave at 0 to read it from the run's flags.json.",
)


@dataclasses.dataclass(frozen=True)
class FdrTarget:
  """One unit of FDR correction work (input file -> output dir)."""

  run_name: str
  input_path: str
  source_type: str  # "evaluation" | "verification"
  source_subdir: str  # "" for evaluation, else the verification subdir name
  output_dir: str  # directory where outputs will be saved
  method_name: str
  n_hypotheses: int
  verification_mode: str = "single_batch_p_value"


def _extract_verification_subdirs(dir_path: str) -> list[str]:
  """Returns names of immediate subdirectories containing verification_results.json."""
  if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
    return []
  subdirs = []
  try:
    for entry in os.listdir(dir_path):
      sub_path = os.path.join(dir_path, entry)
      if os.path.isdir(sub_path) and os.path.exists(
          os.path.join(sub_path, "verification_results.json")
      ):
        subdirs.append(entry)
  except OSError as e:
    logging.warning("Error listing subdirs in %s: %s", dir_path, e)
  return sorted(subdirs)


def _targets_from_file(
    file_path: str,
    run_name: str,
    source_subdir: str = "",
) -> list[FdrTarget]:
  """Builds FdrTarget(s) from a single evaluation or verification JSON file."""
  with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)
  parent_dir = os.path.dirname(file_path)

  # Multi-method evaluation container
  methods_list = (
      data.get("baseline_results") or data.get("methods")
      if isinstance(data, dict)
      else None
  )
  if isinstance(methods_list, list) and methods_list:
    targets: list[FdrTarget] = []
    for m_info in methods_list:
      if not isinstance(m_info, dict):
        continue
      m_prec = m_info.get("precision_check")
      if isinstance(m_prec, dict) and "records" in m_prec:
        m_name = m_info.get("method_name") or m_info.get("method", "baseline")
        targets.append(
            FdrTarget(
                run_name=f"{run_name}/{m_name}",
                input_path=file_path,
                source_type="evaluation",
                source_subdir="",
                output_dir=os.path.join(parent_dir, m_name, OUTPUT_SUBDIR),
                method_name=str(m_name),
                n_hypotheses=len(m_prec.get("records", [])),
                verification_mode=str(
                    m_prec.get("verification_mode", "single_batch_p_value")
                ),
            )
        )
    return targets

  # Single-method evaluation_results.json with precision_check
  if isinstance(data, dict) and isinstance(data.get("precision_check"), dict):
    p_check = data["precision_check"]
    m_name = data.get("method_name") or data.get("method") or run_name
    return [
        FdrTarget(
            run_name=run_name,
            input_path=file_path,
            source_type="evaluation",
            source_subdir="",
            output_dir=os.path.join(parent_dir, OUTPUT_SUBDIR),
            method_name=str(m_name),
            n_hypotheses=len(p_check.get("records", [])),
            verification_mode=str(
                p_check.get("verification_mode", "single_batch_p_value")
            ),
        )
    ]

  # Standalone verification_results.json
  if isinstance(data, dict) and "records" in data:
    v_mode = (
        data.get("verification_mode")
        or data.get("config", {}).get("verification_mode")
        or "single_batch_p_value"
    )
    m_name = (
        data.get("config", {}).get("class_name")
        or data.get("config", {}).get("method_name")
        or run_name
    )
    return [
        FdrTarget(
            run_name=run_name,
            input_path=file_path,
            source_type="verification",
            source_subdir=source_subdir,
            output_dir=os.path.join(parent_dir, OUTPUT_SUBDIR),
            method_name=str(m_name),
            n_hypotheses=len(data.get("records", [])),
            verification_mode=str(v_mode),
        )
    ]

  return []


def _multiple_subdirs_error(run_dir: str, subdirs: Sequence[str]) -> ValueError:
  """The error raised when a run directory holds several verification subdirs."""
  return ValueError(
      f"Multiple ({len(subdirs)}) verification subdirs found in {run_dir}:"
      f" {', '.join(subdirs)}. Pass the one to process explicitly via"
      f" --input_path={run_dir}/<subdir>/verification_results.json instead."
  )


def resolve_single_target(
    input_path: str = "",
    run_dir: str = "",
    source: str = "auto",
) -> list[FdrTarget]:
  """Resolves FdrTarget(s) for a single input JSON file or a single run directory.

  Args:
    input_path: Path to an input JSON file or directory.
    run_dir: Path to a run directory containing evaluation or verification JSON.
    source: Target source filter ("auto", "evaluation", or "verification").

  Returns:
    A list of resolved FdrTarget objects.

  Raises:
    FileNotFoundError: If `input_path` or `run_dir` does not exist on disk.
    ValueError: If neither path is given, or if `run_dir` holds more than one
      `<subdir>/verification_results.json` and no other input resolves first;
      pass the specific file via `input_path` in that case.
  """
  if input_path:
    clean_input = input_path.rstrip("/")
    if not os.path.exists(clean_input):
      raise FileNotFoundError(f"Input file does not exist: {clean_input}")
    if os.path.isdir(clean_input):
      run_dir = clean_input
    else:
      run_name = os.path.basename(os.path.dirname(clean_input)) or "run"
      return _targets_from_file(clean_input, run_name)

  if not run_dir:
    raise ValueError("Must provide either --input_path or --run_dir.")

  clean_dir = run_dir.rstrip("/")
  if not os.path.exists(clean_dir):
    raise FileNotFoundError(f"Run directory does not exist: {clean_dir}")
  if not os.path.isdir(clean_dir):
    run_name = os.path.basename(os.path.dirname(clean_dir)) or "run"
    return _targets_from_file(clean_dir, run_name)

  run_name = os.path.basename(clean_dir) or "run"
  eval_path = os.path.join(clean_dir, "evaluation_results.json")
  direct_verif = os.path.join(clean_dir, "verification_results.json")
  verif_subdirs = _extract_verification_subdirs(clean_dir)

  if source == "evaluation":
    if os.path.exists(eval_path):
      return _targets_from_file(eval_path, run_name)
    return []

  if source == "verification":
    if os.path.exists(direct_verif):
      return _targets_from_file(direct_verif, run_name)
    if len(verif_subdirs) == 1:
      v_path = os.path.join(
          clean_dir, verif_subdirs[0], "verification_results.json"
      )
      return _targets_from_file(
          v_path, run_name, source_subdir=verif_subdirs[0]
      )
    if len(verif_subdirs) > 1:
      raise _multiple_subdirs_error(clean_dir, verif_subdirs)
    return []

  # source == "auto"
  if os.path.exists(direct_verif):
    return _targets_from_file(direct_verif, run_name)
  if len(verif_subdirs) == 1:
    v_path = os.path.join(
        clean_dir, verif_subdirs[0], "verification_results.json"
    )
    return _targets_from_file(v_path, run_name, source_subdir=verif_subdirs[0])
  if os.path.exists(eval_path):
    return _targets_from_file(eval_path, run_name)
  if len(verif_subdirs) > 1:
    raise _multiple_subdirs_error(clean_dir, verif_subdirs)
  return []


def process_target(
    target: FdrTarget,
    alpha: float = 0.05,
    kappa: float = 0.1741,
    ebh_family_size: str = "max_predictions",
    max_predictions: int = 0,
) -> dict[str, Any] | None:
  """Processes one FdrTarget and writes output files."""
  if not os.path.exists(target.input_path):
    logging.warning("Target input does not exist: %s", target.input_path)
    return None

  with open(target.input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

  if target.source_type == "evaluation":
    if "precision_check" in data:
      prec = data["precision_check"]
      records = prec.get("records", [])
      verification_mode = prec.get(
          "verification_mode", target.verification_mode
      )
    elif "methods" in data or "baseline_results" in data:
      methods = data.get("methods") or data.get("baseline_results") or []
      matching_m = next(
          (
              m
              for m in methods
              if isinstance(m, dict)
              and (
                  m.get("method_name") == target.method_name
                  or m.get("method") == target.method_name
              )
          ),
          None,
      )
      if matching_m and "precision_check" in matching_m:
        prec = matching_m["precision_check"]
        records = prec.get("records", [])
        verification_mode = prec.get(
            "verification_mode", target.verification_mode
        )
      else:
        logging.warning(
            "Could not find method %s in %s",
            target.method_name,
            target.input_path,
        )
        return None
    else:
      logging.warning("No precision_check in %s", target.input_path)
      return None
  else:
    records = data.get("records", [])
    verification_mode = (
        data.get("verification_mode")
        or data.get("config", {}).get("verification_mode")
        or target.verification_mode
    )

  failed_records = [r for r in records if r.get("status") not in ("OK", None)]
  resolved_ebh_m, ebh_source = ebh_family.resolve(
      input_path=target.input_path,
      mode=ebh_family_size,
      max_predictions_override=max_predictions,
      n_failed=len(failed_records),
  )
  if failed_records:
    logging.warning(
        ebh_family.failed_slots_warning(
            target.input_path, failed_records, ebh_family_size, resolved_ebh_m
        )
    )
  try:
    correction_result = multiple_testing.compute_all_fdr_corrections(
        records=records,
        alpha=alpha,
        kappa=kappa,
        verification_mode=verification_mode,
        ebh_family_size=resolved_ebh_m,
    )
  except (
      ValueError,
      TypeError,
      KeyError,
      ZeroDivisionError,
      RuntimeError,
  ) as e:
    logging.error("Failed FDR correction for %s: %s", target.input_path, e)
    return None

  annotated_records = correction_result["annotated_records"]
  methods_summary = correction_result["summary"]["methods"]
  n_eval = correction_result["summary"]["n_evaluated"]
  n_total = correction_result["summary"]["n_total"]
  m_family = correction_result["summary"]["family_size_m"]
  m_ebh = correction_result["summary"]["ebh_family_size_m"]

  # Compute Storey pi0 FDR and thresholded empirical FDR estimates across the
  # verified discovery records via stats/estimators.py.
  storey_fdr = estimators.compute_storey_fdr(records, lam=0.5, plus_one=True)
  thresholded_fdr = estimators.compute_thresholded_fdr(records, alpha=alpha)

  unadj_info = methods_summary.get("unadjusted", {})
  bh_info = methods_summary.get("bh", {})
  by_info = methods_summary.get("by", {})
  ebh_u_info = methods_summary.get("ebh_unboosted", {})
  ebh_b_info = methods_summary.get("ebh_boosted", {})

  fdr_results_data = {
      "run_name": target.run_name,
      "method_name": target.method_name,
      "source_type": target.source_type,
      "source_subdir": target.source_subdir,
      "input_path": target.input_path,
      "verification_mode": verification_mode,
      "n_total": n_total,
      "n_evaluated": n_eval,
      "n_verified": unadj_info.get("n_verified", 0),
      "n_bh_verified": bh_info.get("n_verified"),
      "n_by_verified": by_info.get("n_verified"),
      "n_ebh_unboosted_verified": ebh_u_info.get("n_verified"),
      "n_ebh_boosted_verified": ebh_b_info.get("n_verified"),
      "bh_rejection_threshold": bh_info.get("rejection_threshold"),
      "by_rejection_threshold": by_info.get("rejection_threshold"),
      "ebh_unboosted_threshold": ebh_u_info.get("rejection_threshold"),
      "ebh_boosted_threshold": ebh_b_info.get("rejection_threshold"),
      "ebh_boost_factor": ebh_b_info.get("boost_factor"),
      "storey_fdr": storey_fdr,
      "thresholded_fdr": thresholded_fdr,
      "fdr_config": {
          "alpha": alpha,
          "kappa": kappa,
          "family_size_mode": FAMILY_SIZE_MODE,
          "family_size_m": m_family,
          "ebh_family_size_m": m_ebh,
          "ebh_family_size_source": ebh_source,
      },
      "records": annotated_records,
  }

  rollup_row: dict[str, Any] = {
      "run_name": target.run_name,
      "method_name": target.method_name,
      "source_type": target.source_type,
      "source_subdir": target.source_subdir,
      "verification_mode": verification_mode,
      "n_total": n_total,
      "n_evaluated": n_eval,
      "n_unadjusted": unadj_info.get("n_verified", 0),
      "n_bh": bh_info.get("n_verified"),
      "n_by": by_info.get("n_verified"),
      "n_ebh_unboosted": ebh_u_info.get("n_verified"),
      "n_ebh_boosted": ebh_b_info.get("n_verified"),
      "bh_threshold": bh_info.get("rejection_threshold"),
      "by_threshold": by_info.get("rejection_threshold"),
      "ebh_unboosted_threshold": ebh_u_info.get("rejection_threshold"),
      "ebh_boosted_threshold": ebh_b_info.get("rejection_threshold"),
      "boost_factor": ebh_b_info.get("boost_factor"),
      "storey_fdr": storey_fdr,
      "thresholded_fdr": thresholded_fdr,
      "alpha": alpha,
      "kappa": kappa,
      "family_size_mode": FAMILY_SIZE_MODE,
      "family_size_m": m_family,
      "ebh_family_size_m": m_ebh,
      "ebh_family_size_source": ebh_source,
  }

  compact_summary: dict[str, Any] = {
      "run_name": target.run_name,
      "method_name": target.method_name,
      "source_type": target.source_type,
      "source_subdir": target.source_subdir,
      "verification_mode": verification_mode,
      "n_total": n_total,
      "n_evaluated": n_eval,
      "storey_fdr": storey_fdr,
      "thresholded_fdr": thresholded_fdr,
      "methods": methods_summary,
      "config": {
          "alpha": alpha,
          "kappa": kappa,
          "family_size_mode": FAMILY_SIZE_MODE,
          "family_size_m": m_family,
          "ebh_family_size_m": m_ebh,
          "ebh_family_size_source": ebh_source,
          "input_path": target.input_path,
      },
      "rollup_row": rollup_row,
  }

  os.makedirs(target.output_dir, exist_ok=True)

  v_out = os.path.join(target.output_dir, "verification_results_fdr.json")
  with open(v_out, "w", encoding="utf-8") as f:
    json.dump(fdr_results_data, f, indent=2)

  s_out = os.path.join(target.output_dir, "fdr_summary.json")
  with open(s_out, "w", encoding="utf-8") as f:
    json.dump(compact_summary, f, indent=2)

  csv_out = os.path.join(target.output_dir, "hypotheses_ranked.csv")
  csv_cols = [
      "rank",
      "label",
      "hypothesis_text",
      "status",
      "p_value",
      "e_value",
      "boost_factor",
      "boosted_e_value",
      "bh_rank",
      "bh_q_value",
      "bh_verified",
      "by_q_value",
      "by_verified",
      "ebh_unboosted_rank",
      "ebh_unboosted_verified",
      "ebh_boosted_rank",
      "ebh_boosted_verified",
      "unadjusted_verified",
  ]
  with open(csv_out, "w", encoding="utf-8") as f:
    f.write(",".join(csv_cols) + "\n")
    for idx, r in enumerate(annotated_records, start=1):
      text_escaped = (
          '"' + str(r.get("hypothesis_text", "")).replace('"', '""') + '"'
      )
      row_vals = [
          str(idx),
          str(r.get("label", "")),
          text_escaped,
          str(r.get("status", "")),
          str(r.get("p_value", "")),
          str(r.get("e_value", "")),
          str(r.get("boost_factor", "")),
          str(r.get("boosted_e_value", "")),
          str(r.get("bh_rank", "")),
          str(r.get("bh_q_value", "")),
          str(r.get("bh_verified", "")),
          str(r.get("by_q_value", "")),
          str(r.get("by_verified", "")),
          str(r.get("ebh_unboosted_rank", "")),
          str(r.get("ebh_unboosted_verified", "")),
          str(r.get("ebh_boosted_rank", "")),
          str(r.get("ebh_boosted_verified", "")),
          str(r.get("unadjusted_verified", "")),
      ]
      f.write(",".join(row_vals) + "\n")

  return rollup_row


def main(argv: Sequence[str]) -> None:
  """CLI entry point for post-hoc FDR control."""
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if not _INPUT_PATH.value and not _RUN_DIR.value:
    raise app.UsageError("Must provide either --input_path or --run_dir.")

  targets = resolve_single_target(
      input_path=_INPUT_PATH.value,
      run_dir=_RUN_DIR.value,
      source=_SOURCE.value,
  )
  if not targets:
    logging.info("No FDR targets found.")
    return

  for target in targets:
    row = process_target(
        target=target,
        alpha=_ALPHA.value,
        kappa=_KAPPA.value,
        ebh_family_size=_EBH_FAMILY_SIZE.value,
        max_predictions=_MAX_PREDICTIONS.value,
    )
    if row is not None:
      print(
          f"[FDR] {row['run_name']} ({row['method_name']}):"
          f" evaluated={row['n_evaluated']}/{row['n_total']},"
          f" unadjusted={row['n_unadjusted']},"
          f" BH={row['n_bh']}, BY={row['n_by']},"
          f" e-BH={row['n_ebh_unboosted']},"
          f" boosted e-BH={row['n_ebh_boosted']},"
          f" thresholded_fdr={row['thresholded_fdr']},"
          f" storey_fdr={row['storey_fdr']}",
          flush=True,
      )


if __name__ == "__main__":
  app.run(main)
