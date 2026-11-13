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

"""Shared writers for the end-of-run FDR outputs of a DISCO run.

Every finalization mode (DISCO, unrestricted e-BH baseline, Type-I baseline)
writes `<output_dir>/<experiment>/fdr_bounds.json` with the same layout:

  params: run parameters (common prefix, mode-specific middle, sweep suffix)
  bounds_report: mode-specific bounds and calibration summary
  alpha_sweep: one record per alpha on the sweep grid
  hypotheses_data: probe counts, e-values, boost factors, terminal statuses

`fdr_bounds.json` is read by downstream tooling and pinned by the smoke
goldens, so key names and insertion order must stay stable.
"""

from collections.abc import Callable, Collection, Iterable, Mapping
import json
import logging
import os
import threading
import time
from typing import Any

import numpy as np

from disco.core import config as config_lib
from disco.core import records as records_lib

FDR_BOUNDS_FILENAME = "fdr_bounds.json"


def make_log_function(
    log_file: str, log_to_console: bool
) -> Callable[..., None]:
  """Returns a function that logs to a file and optionally to the console."""
  lock = threading.Lock()

  def log(msg: Any, end: str = "\n", mode: str = "a") -> None:
    """Writes `msg` to `log_file` (with retry) and optionally to stdout."""
    if mode not in ["a", "w"]:
      raise ValueError(f"Invalid logging mode: {mode}")
    if log_to_console:
      print(msg, end=end)
    with lock:
      for attempt in range(3):
        try:
          with open(log_file, mode, encoding="utf-8") as f:
            f.write(str(msg) + end)
          break
        except OSError as e:
          if attempt == 2:
            logging.warning("Failed to write to log file %s: %s", log_file, e)
          else:
            time.sleep(0.05 * (2**attempt))

  return log


def sweep_alphas(config: config_lib.DiscoConfig) -> list[float]:
  """Returns the alpha sweep grid: square-spaced, always containing alpha."""
  alphas = [
      float(x)
      for x in (
          np.linspace(
              np.sqrt(config.alpha_sweep_min),
              np.sqrt(config.alpha_sweep_max),
              config.alpha_sweep_steps,
          )
          ** 2
      )
  ]
  if config.alpha not in alphas:
    alphas.append(config.alpha)
    alphas.sort()
  return alphas


def fdr_params(
    config: config_lib.DiscoConfig, mode_params: Mapping[str, Any]
) -> dict[str, Any]:
  """Returns the `params` block of fdr_bounds.json.

  Args:
    config: The run config.
    mode_params: Mode-specific keys, inserted between the common run keys and
      the alpha-sweep keys (in the given order).
  """
  params: dict[str, Any] = {
      "target_model": config.target_model,
      "seed": config.seed_instruction,
      "budget": config.budget,
      "agent_model": config.agent_model,
      "output_dir": config.output_dir,
      "name": config.experiment_name,
      "alpha": config.alpha,
      "kappa": config.kappa,
  }
  params.update(mode_params)
  params["alpha_sweep_min"] = config.alpha_sweep_min
  params["alpha_sweep_max"] = config.alpha_sweep_max
  params["alpha_sweep_steps"] = config.alpha_sweep_steps
  return params


def hypotheses_data(
    probe_counts: Mapping[str, int],
    final_e_values: Mapping[str, float],
    boosted_e_values: Mapping[str, float],
    boost_factors: Mapping[str, float],
    terminal_statuses: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
  """Returns the `hypotheses_data` block of fdr_bounds.json."""
  return {
      "probe_counts": {str(k): int(v) for k, v in probe_counts.items()},
      "final_e_values": {str(k): float(v) for k, v in final_e_values.items()},
      "boosted_e_values": {
          str(k): float(v) for k, v in boosted_e_values.items()
      },
      "boost_factors": {str(k): float(v) for k, v in boost_factors.items()},
      "terminal_statuses": {
          str(k): str(v) for k, v in terminal_statuses.items()
      },
  }


def write_fdr_bounds(
    config: config_lib.DiscoConfig,
    *,
    params: Mapping[str, Any],
    bounds_report: Mapping[str, Any],
    alpha_sweep: list[dict[str, Any]],
    hyp_data: Mapping[str, Any],
    log_fn: Callable[[str], None],
) -> str:
  """Writes fdr_bounds.json into the experiment dir and returns its path."""
  fdr_bounds_data = {
      "params": params,
      "bounds_report": bounds_report,
      "alpha_sweep": alpha_sweep,
      "hypotheses_data": hyp_data,
  }
  output_dir = os.path.join(config.output_dir, config.experiment_name)
  os.makedirs(output_dir, exist_ok=True)
  bounds_path = os.path.join(output_dir, FDR_BOUNDS_FILENAME)
  with open(bounds_path, "w", encoding="utf-8") as f:
    json.dump(fdr_bounds_data, f, indent=2)
  log_fn(f"Saved FDR bounds report to: {bounds_path}")
  return bounds_path


def log_rejections(
    hypothesis_groups: Mapping[str, records_lib.HypothesisRecord],
    rejected_set: Collection[str],
    tested_ids: Iterable[str],
    *,
    rejected_header: str,
    rejected_e: Callable[[str], float],
    non_rejected_e: Callable[[str], float],
    unboosted_e: Callable[[str], float | None] | None = None,
    log_fn: Callable[[str], None],
) -> None:
  """Logs the final rejected / non-rejected hypothesis summaries.

  Args:
    hypothesis_groups: Per-hypothesis records.
    rejected_set: Rejected hypothesis ids (logged in iteration order).
    tested_ids: All tested hypothesis ids (logged in iteration order, minus
      rejected ones).
    rejected_header: Header line for the rejected block.
    rejected_e: Displayed e-value for a rejected hypothesis.
    non_rejected_e: Displayed e-value for a non-rejected hypothesis.
    unboosted_e: If given, the unboosted e-value shown alongside.
    log_fn: Logger.
  """
  log_fn(rejected_header)
  for h_id in rejected_set:
    records_lib.log_hypothesis_details(
        h_id,
        hypothesis_groups[h_id],
        rejected_e(h_id),
        log_fn,
        unboosted_e=unboosted_e(h_id) if unboosted_e else None,
    )

  log_fn("\n=== Non-rejected Hypotheses ===")
  for h_id in tested_ids:
    if h_id not in rejected_set:
      records_lib.log_hypothesis_details(
          h_id,
          hypothesis_groups[h_id],
          non_rejected_e(h_id),
          log_fn,
          unboosted_e=unboosted_e(h_id) if unboosted_e else None,
      )
