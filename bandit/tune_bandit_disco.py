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

"""Standalone Bayesian optimization script for DISCO bandit hyperparameters using open-source Google Vizier."""

from collections.abc import Mapping, Sequence
import dataclasses
import json
import os
from typing import Any

from absl import app
from absl import flags
from absl import logging
from vizier.service import clients
from vizier.service import pyvizier as vz

from disco.bandit import bandit_disco
from disco.bandit import run_bandit_disco

# Reuse simulation and environment flags defined in `run_bandit_disco`.
_OUTPUT_DIR = run_bandit_disco.OUTPUT_DIR
_RUNNER_TYPE = run_bandit_disco.RUNNER_TYPE
_DELTA_THRESHOLD = run_bandit_disco.DELTA_THRESHOLD
_DELTA_MULTIPLIER = run_bandit_disco.DELTA_MULTIPLIER
_NUM_MC_RUNS = run_bandit_disco.NUM_MC_RUNS

_EXPERIMENT_NAME = flags.DEFINE_string(
    "experiment_name",
    "bandit_disco_vizier_bo_cov",
    "Vizier study experiment name prefix.",
)
_STUDY_ID = flags.DEFINE_string(
    "study_id",
    "",
    "Optional explicit Vizier study ID (overrides --experiment_name prefix"
    " when set).",
)
_PRIMARY_METRIC = flags.DEFINE_enum(
    "primary_metric",
    "penalized_cluster_coverage",
    [
        "penalized_cluster_coverage",
        "cluster_coverage",
        "penalized_recall",
        "recall",
        "macro_recall",
        "true_positives",
    ],
    "Primary objective metric to maximize in Vizier Bayesian Optimization.",
)
_NUM_TRIALS = flags.DEFINE_integer(
    "num_trials", 200, "Total number of Bayesian optimization trials per study."
)
_ALGORITHM = flags.DEFINE_enum(
    "algorithm",
    "DEFAULT",
    [
        "DEFAULT",
        "GP_UCB_PE",
        "GAUSSIAN_PROCESS_BANDIT",
        "RANDOM_SEARCH",
        "QUASI_RANDOM_SEARCH",
    ],
    "Vizier search algorithm.",
)

# =============================================================================
# Customizable Hyperparameter Search Space Ranges
# =============================================================================
# 1. kappa (p-to-e calibration parameter)
_KAPPA_MIN = flags.DEFINE_float("kappa_min", 0.1, "Minimum kappa value.")
_KAPPA_MAX = flags.DEFINE_float("kappa_max", 0.9, "Maximum kappa value.")
_KAPPA_DEFAULT = flags.DEFINE_float(
    "kappa_default", 0.5, "Default kappa value."
)

# 2. e_max (upper deactivation threshold)
_E_MAX_MIN = flags.DEFINE_float("e_max_min", 10.0, "Minimum e_max value.")
_E_MAX_MAX = flags.DEFINE_float("e_max_max", 1000.0, "Maximum e_max value.")
_E_MAX_DEFAULT = flags.DEFINE_float(
    "e_max_default", 100.0, "Default e_max value."
)

# 3. n_max (maximum probes per hypothesis)
_N_MAX_VALUES = flags.DEFINE_list(
    "n_max_values",
    ["2", "3", "5", "10", "20", "30"],
    "Feasible discrete values for n_max.",
)

# 4. num_active_bins conditional search space:
# 'continuous' -> num_active_bins = 0 (unbinned e-values)
# 'binned' -> discrete log-spaced values between 1 and 100
_BINNED_ACTIVE_BINS_VALUES = flags.DEFINE_list(
    "binned_active_bins_values",
    ["1", "2", "3", "5", "8", "13", "22", "36", "60", "100"],
    "Feasible discrete log-spaced values for num_active_bins under 'binned'"
    " mode (1 to 100).",
)

# 5. e_min (lower deactivation threshold for DISCO)
_E_MIN_MIN = flags.DEFINE_float("e_min_min", 0.01, "Minimum e_min value.")
_E_MIN_MAX = flags.DEFINE_float("e_min_max", 2.0, "Maximum e_min value.")
_E_MIN_DEFAULT = flags.DEFINE_float(
    "e_min_default", 0.01, "Default e_min value."
)


def build_study_config(
    runner_type: str,
    primary_metric: str,
    algorithm: str,
) -> vz.StudyConfig:
  """Constructs the Vizier StudyConfig defining the search space over the 5 variables."""
  study_config = vz.StudyConfig(algorithm=algorithm)
  root = study_config.search_space.select_root()

  # 1. kappa: p-to-e calibration parameter in [kappa_min, kappa_max]
  root.add_float_param(
      "args.kappa",
      min_value=_KAPPA_MIN.value,
      max_value=_KAPPA_MAX.value,
      scale_type=vz.ScaleType.LINEAR,
      default_value=_KAPPA_DEFAULT.value,
  )

  # 2. e_max: upper deactivation threshold in [e_max_min, e_max_max] (log scale)
  if runner_type != "type1":
    root.add_float_param(
        "args.e_max",
        min_value=_E_MAX_MIN.value,
        max_value=_E_MAX_MAX.value,
        scale_type=vz.ScaleType.LOG,
        default_value=_E_MAX_DEFAULT.value,
    )

  # 3. n_max: probe cap per hypothesis (discrete choices)
  n_max_ints = [int(x) for x in _N_MAX_VALUES.value]
  root.add_discrete_param(
      "args.n_max",
      feasible_values=n_max_ints,
      default_value=10 if 10 in n_max_ints else n_max_ints[0],
  )

  # DISCO-Specific Hyperparameters to Optimize
  if runner_type == "disco":
    # 4. Active bins: 'continuous' (0 bins) vs 'binned' (discrete log-spaced).
    # Added at root level because open-source Vizier designers require a flat
    # search space (`evaluate_trial_params` ignores `num_active_bins` when
    # `bin_mode == "continuous"`).
    root.add_categorical_param(
        "args.bin_mode",
        feasible_values=["continuous", "binned"],
        default_value="binned",
    )
    binned_values = [int(x) for x in _BINNED_ACTIVE_BINS_VALUES.value]
    root.add_discrete_param(
        "args.num_active_bins",
        feasible_values=binned_values,
        default_value=1 if 1 in binned_values else binned_values[0],
    )

    # 5. e_min: lower threshold for hypothesis deactivation
    root.add_float_param(
        "args.e_min",
        min_value=_E_MIN_MIN.value,
        max_value=_E_MIN_MAX.value,
        scale_type=vz.ScaleType.LINEAR,
        default_value=_E_MIN_DEFAULT.value,
    )

  study_config.metric_information = [
      vz.MetricInformation(
          name=primary_metric, goal=vz.ObjectiveMetricGoal.MAXIMIZE
      ),
  ]
  return study_config


def _clean_trial_params(raw_params: Mapping[str, Any]) -> dict[str, Any]:
  """Strips 'args.' prefix from Vizier parameter names."""
  params: dict[str, Any] = {}
  for k, v in raw_params.items():
    key = k[len("args.") :] if k.startswith("args.") else k
    params[key] = v
  return params


def evaluate_trial_params(
    runner_type: str,
    trial_params: dict[str, Any],
) -> dict[str, float]:
  """Evaluates one trial configuration across `_NUM_MC_RUNS` simulations."""
  bin_mode = trial_params.get("bin_mode", "binned")
  if bin_mode == "continuous":
    active_bins = 0
  else:
    active_bins = int(trial_params.get("num_active_bins", 1))

  base_config = run_bandit_disco.base_config_from_flags()
  config = dataclasses.replace(
      base_config,
      kappa=float(trial_params.get("kappa", _KAPPA_DEFAULT.value)),
      e_min=float(trial_params.get("e_min", _E_MIN_DEFAULT.value)),
      e_max=float(trial_params.get("e_max", _E_MAX_DEFAULT.value)),
      n_max=int(trial_params.get("n_max", 10)),
      num_active_bins=active_bins,
      use_boosting=(runner_type == "disco"),
  )

  k_bar = bandit_disco.resolve_k_bar(config)
  null_tables = (
      bandit_disco.build_null_tables(config, k_bar)
      if runner_type == "disco"
      else None
  )

  n_runs = _NUM_MC_RUNS.value
  num_rounds = max(1, config.budget // config.chunk_size)
  runs: list[bandit_disco.BanditResult] = []
  for i in range(n_runs):
    run_seed = config.seed + i * 1000
    pool_env = run_bandit_disco.build_env(config, run_seed)
    ttp, cluster_ttp = pool_env.pre_generate_pool(num_rounds)
    res = run_bandit_disco.run_single(
        config,
        runner_type,
        run_seed,
        null_tables=null_tables,
        total_true_positives=ttp,
        cluster_true_positives=cluster_ttp,
    )
    runs.append(res)

  m_data = run_bandit_disco.summarize_runs(runs, config)
  rec = float(m_data["recall"]["mean"])
  fdr = float(m_data["fdr"]["mean"])
  cov = float(m_data.get("cluster_coverage", {}).get("mean", 0.0))
  penalized_cov = float(
      m_data.get("penalized_cluster_coverage", {}).get("mean", cov)
  )
  delta_post_mean = float(m_data.get("delta_post", {}).get("mean", 0.0))
  delta_excess_pct = max(
      0.0, (delta_post_mean - _DELTA_THRESHOLD.value) * 100.0
  )
  delta_penalty = _DELTA_MULTIPLIER.value * delta_excess_pct
  penalized_rec = rec - delta_penalty

  return {
      "penalized_cluster_coverage": penalized_cov,
      "cluster_coverage": cov,
      "penalized_recall": penalized_rec,
      "recall": rec,
      "empirical_fdr": fdr,
      "delta_post": delta_post_mean,
      "delta_penalty": delta_penalty,
      "macro_recall": float(m_data.get("macro_recall", {}).get("mean", rec)),
      "true_positives": float(m_data["tp"]["mean"]),
      "false_positives": float(m_data["fp"]["mean"]),
  }


def run_study(
    runner_type: str,
    study_id: str,
    output_dir: str,
    num_trials: int,
    primary_metric: str,
    algorithm: str,
    delta_threshold: float,
) -> dict[str, Any]:
  """Runs a local open-source Vizier study and writes a summary JSON."""
  if "database_url" not in clients.environment_variables.servicer_kwargs:
    clients.environment_variables.servicer_use_sql_ram()
  study_config = build_study_config(
      runner_type=runner_type,
      primary_metric=primary_metric,
      algorithm=algorithm,
  )
  study = clients.Study.from_study_config(
      study_config,
      owner="disco",
      study_id=study_id,
  )

  trial_records: list[dict[str, Any]] = []
  for trial_idx in range(num_trials):
    suggestions = study.suggest(count=1)
    for suggestion in suggestions:
      params = _clean_trial_params(suggestion.parameters)
      metrics = evaluate_trial_params(runner_type, params)

      obj_val = float(metrics.get(primary_metric, -float("inf")))
      delta_val = metrics.get("delta_post", None)
      is_feasible = (
          (delta_val is not None and delta_val <= delta_threshold)
          if runner_type == "disco"
          else True
      )
      vizier_obj_val = obj_val
      if (
          runner_type == "disco"
          and not is_feasible
          and primary_metric not in (
              "penalized_cluster_coverage",
              "penalized_recall",
          )
      ):
        vizier_obj_val = obj_val - float(metrics.get("delta_penalty", 0.0))

      suggestion.complete(vz.Measurement({primary_metric: vizier_obj_val}))

      trial_records.append({
          "trial_id": suggestion.id,
          "parameters": params,
          "observed_objective": obj_val,
          "vizier_objective": vizier_obj_val,
          "metrics": metrics,
          "is_feasible": is_feasible,
      })
      print(
          f"[{runner_type} trial {trial_idx + 1}/{num_trials}] "
          f"{primary_metric}={obj_val:.2f}, params={params}"
      )

  ranked_by_observed = sorted(
      trial_records, key=lambda x: x["observed_objective"], reverse=True
  )
  feasible_trials = [r for r in trial_records if r["is_feasible"]]
  ranked_feasible = sorted(
      feasible_trials, key=lambda x: x["observed_objective"], reverse=True
  )

  summary_data = {
      "study_id": study_id,
      "runner_type": runner_type,
      "primary_metric": primary_metric,
      "total_trials": len(trial_records),
      "completed_trials": len(trial_records),
      "feasible_trials_count": len(feasible_trials),
      "feasibility_rate": (
          len(feasible_trials) / len(trial_records) if trial_records else 0.0
      ),
      "best_observed_trial": (
          ranked_by_observed[0] if ranked_by_observed else None
      ),
      "best_feasible_trial": ranked_feasible[0] if ranked_feasible else None,
      "top_5_observed": ranked_by_observed[:5],
      "all_trials": trial_records,
  }

  if output_dir:
    os.makedirs(output_dir, exist_ok=True)
    summary_file = os.path.join(output_dir, f"{runner_type}_study_summary.json")
    try:
      with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)
      logging.info("Saved Vizier study summary to %s", summary_file)
    except (OSError, TypeError, ValueError) as io_err:
      logging.warning(
          "Failed writing study summary to %s: %s", summary_file, io_err
      )

  return summary_data


def main(argv: Sequence[str]) -> None:
  """Runs Vizier Bayesian optimization studies across requested runner types."""
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  runner_mode = _RUNNER_TYPE.value
  if runner_mode == "all":
    runners = ["disco", "offline_ebh", "type1"]
  elif runner_mode == "sxs":
    runners = ["disco", "offline_ebh"]
  else:
    runners = [runner_mode]
  exp_name = _STUDY_ID.value or _EXPERIMENT_NAME.value

  clean_folder = (
      exp_name.replace("_disco", "")
      .replace("_offline_ebh", "")
      .replace("_type1", "")
  )
  base_output_dir = _OUTPUT_DIR.value or "./bandit_vizier"
  exp_output_dir = (
      os.path.join(base_output_dir, clean_folder) if base_output_dir else ""
  )

  for r_type in runners:
    study_name = (
        exp_name
        if (_STUDY_ID.value and len(runners) == 1)
        else f"{exp_name}_{r_type}"
    )
    print(
        f"Running Vizier Study '{study_name}' ({r_type.upper()}) with"
        f" {_NUM_TRIALS.value} trials..."
    )
    run_study(
        runner_type=r_type,
        study_id=study_name,
        output_dir=exp_output_dir,
        num_trials=_NUM_TRIALS.value,
        primary_metric=_PRIMARY_METRIC.value,
        algorithm=_ALGORITHM.value,
        delta_threshold=_DELTA_THRESHOLD.value,
    )


if __name__ == "__main__":
  app.run(main)
