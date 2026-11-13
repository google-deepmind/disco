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

"""Executable binary to run FDR-controlled bandit simulations (Bernoulli & Contextual)."""

from collections.abc import Sequence
import json
import os
import sys
import time
from typing import Any

from absl import app
from absl import flags
import numpy as np

from disco.bandit import bandit_disco
from disco.bandit import bandit_env
from disco.stats.coarsened import calibration as calibration_lib
from disco.stats.coarsened import certificates as cert_lib
from disco.stats.coarsened import status as status_lib

_OUTPUT_PATH = flags.DEFINE_string(
    "output_path",
    "",
    "If set, write structured JSON results to this path.",
)
OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    "",
    "If set, write structured JSON results to this directory.",
)
_OUTPUT_DIR = OUTPUT_DIR
_SAVE_PER_RUN_DATA = flags.DEFINE_boolean(
    "save_per_run_data",
    True,
    "Whether to write per-run output arrays to a separate *_runs.json file.",
)
RUNNER_TYPE = flags.DEFINE_enum(
    "runner_type",
    "sxs",
    ["disco", "offline_ebh", "type1", "sxs", "all"],
    "Which runner variant to execute: 'disco', 'offline_ebh', 'type1', 'sxs'"
    " (Offline vs DISCO), or 'all' (Offline vs Type-I vs DISCO).",
)
_RUNNER_TYPE = RUNNER_TYPE
_ENV_TYPE = flags.DEFINE_enum(
    "env_type",
    "bernoulli",
    ["bernoulli", "contextual"],
    "Bandit environment type: 'bernoulli' or 'contextual'.",
)

_BUDGET = flags.DEFINE_integer(
    "budget", 500, "Total query budget for the simulation."
)
_CHUNK_SIZE = flags.DEFINE_integer(
    "chunk_size", 10, "Number of queries to allocate per probe chunk."
)
_P_VALUE_MODE = flags.DEFINE_enum(
    "p_value_mode",
    "randomized",
    ["randomized", "discrete", "discrete_matched"],
    "P-value computation mode: 'randomized' (Tocher PIT, exact uniform null),"
    " 'discrete' (non-randomized Binomial p-values with continuous null"
    " tables), 'discrete_matched' (non-randomized with discrete-calibrated null"
    " tables).",
)
_ALPHA = flags.DEFINE_float(
    "alpha", 0.05, "Target FDR level alpha for e-BH rejection."
)
_KAPPA = flags.DEFINE_float(
    "kappa", 0.5, "Kappa parameter for p-to-e calibration."
)
_E_MIN = flags.DEFINE_float(
    "e_min", 0.01, "Lower threshold e_min for hypothesis deactivation."
)
_ENFORCE_EXIT_LOW = flags.DEFINE_boolean(
    "enforce_exit_low",
    True,
    "Whether to deactivate hypotheses immediately when running e-value drops"
    " below e_min.",
)
_E_MAX = flags.DEFINE_float(
    "e_max", 100.0, "Upper threshold e_max for hypothesis deactivation."
)
_N_MAX = flags.DEFINE_integer(
    "n_max", 10, "Maximum probes per hypothesis before timeout cap."
)
_K_BAR = flags.DEFINE_integer(
    "k_bar",
    None,
    "Pre-hoc upper bound on total hypotheses K_bar. If not set, defaults to"
    " budget // chunk_size.",
)
_NUM_ACTIVE_BINS = flags.DEFINE_integer(
    "num_active_bins",
    1,
    "Number of active bins J (0 = unbinned/exact, 1 = binary active strip, J >"
    " 1 = multi-bin refinement).",
)
_ALPHA_CALIBRATION_MODE = flags.DEFINE_enum(
    "alpha_calibration_mode",
    "post_hoc",
    ["nominal", "pre_hoc", "post_hoc", "good_event"],
    "Alpha calibration mode for DISCO.",
)
_USE_BOOSTING = flags.DEFINE_boolean(
    "use_boosting",
    True,
    "Whether to apply per-hypothesis e-value boosting before e-BH.",
)
_N_SIM = flags.DEFINE_integer(
    "n_sim",
    100_000,
    "Number of Monte Carlo simulations for null table precomputation.",
)

_ENABLE_ALPHA_SWEEP = flags.DEFINE_boolean(
    "enable_alpha_sweep",
    False,
    "Whether to compute and display a post-run alpha sweep.",
)
_ALPHA_SWEEP_MIN = flags.DEFINE_float(
    "alpha_sweep_min", 0.01, "Minimum alpha for post-run sweep."
)
_ALPHA_SWEEP_MAX = flags.DEFINE_float(
    "alpha_sweep_max", 0.50, "Maximum alpha for post-run sweep."
)
_ALPHA_SWEEP_STEPS = flags.DEFINE_integer(
    "alpha_sweep_steps", 8, "Number of sqrt-spaced alpha sweep steps."
)

_GP_BETA = flags.DEFINE_float(
    "gp_beta", 2.0, "Exploration factor beta for UCB."
)
_RIDGE_LAMBDA = flags.DEFINE_float(
    "ridge_lambda", 1.0, "Ridge regularization parameter lambda."
)

_DIMENSION = flags.DEFINE_integer(
    "dimension", 3, "Dimension d of the feature space for contextual bandit."
)
_NUM_VULNERABLE_CLUSTERS = flags.DEFINE_integer(
    "num_vulnerable_clusters", 1, "Number of vulnerable clusters."
)
_CLUSTER_WEIGHT_MIN = flags.DEFINE_float(
    "cluster_weight_min",
    7.0,
    "Minimum Gaussian RBF cluster weight for contextual bandit.",
)
_CLUSTER_WEIGHT_MAX = flags.DEFINE_float(
    "cluster_weight_max",
    10.0,
    "Maximum Gaussian RBF cluster weight for contextual bandit.",
)
_CLUSTER_VAR_MIN = flags.DEFINE_float(
    "cluster_var_min",
    0.05,
    "Minimum Gaussian RBF cluster variance for contextual bandit.",
)
_CLUSTER_VAR_MAX = flags.DEFINE_float(
    "cluster_var_max",
    0.15,
    "Maximum Gaussian RBF cluster variance for contextual bandit.",
)
_PI_1 = flags.DEFINE_float(
    "pi_1",
    0.10,
    "Fraction of alternative (vulnerable) arms in Bernoulli bandit.",
)
_BASELINE_RATE_LOGIT = flags.DEFINE_float(
    "baseline_rate_logit",
    -4.6,
    "Background baseline failure logit z_0 (e.g. -4.6 gives ~1% baseline"
    " rate).",
)
_ALTERNATIVE_RATE = flags.DEFINE_float(
    "alternative_rate",
    0.20,
    "Failure rate for alternative arms in Bernoulli bandit.",
)
DELTA_THRESHOLD = flags.DEFINE_float(
    "delta_threshold",
    0.05,
    "Tolerated post-hoc delta threshold before hinge penalty applies.",
)
_DELTA_THRESHOLD = DELTA_THRESHOLD
DELTA_MULTIPLIER = flags.DEFINE_float(
    "delta_multiplier",
    5.0,
    "Multiplier slope for hinge penalty on excess post-hoc delta.",
)
_DELTA_MULTIPLIER = DELTA_MULTIPLIER

NUM_MC_RUNS = flags.DEFINE_integer(
    "num_mc_runs", 1, "Number of Monte Carlo simulation repetitions to run."
)
_NUM_MC_RUNS = NUM_MC_RUNS
_SEED = flags.DEFINE_integer("seed", 42, "Random seed for reproducibility.")


# ---------------------------------------------------------------------------
# Helper: config assembly
# ---------------------------------------------------------------------------


def base_config_from_flags() -> bandit_disco.BanditConfig:
  """Assembles a BanditConfig from the module-level flag values."""
  return bandit_disco.BanditConfig(
      env_type=_ENV_TYPE.value,
      budget=_BUDGET.value,
      chunk_size=_CHUNK_SIZE.value,
      alpha=_ALPHA.value,
      kappa=_KAPPA.value,
      e_min=_E_MIN.value,
      e_max=_E_MAX.value,
      enforce_exit_low=_ENFORCE_EXIT_LOW.value,
      n_max=_N_MAX.value,
      k_bar=_K_BAR.value,
      num_active_bins=_NUM_ACTIVE_BINS.value,
      alpha_calibration_mode=_ALPHA_CALIBRATION_MODE.value,
      use_boosting=_USE_BOOSTING.value,
      n_sim=_N_SIM.value,
      gp_beta=_GP_BETA.value,
      ridge_lambda=_RIDGE_LAMBDA.value,
      p_value_mode=_P_VALUE_MODE.value,
      dimension=_DIMENSION.value,
      num_vulnerable_clusters=_NUM_VULNERABLE_CLUSTERS.value,
      cluster_weight_min=_CLUSTER_WEIGHT_MIN.value,
      cluster_weight_max=_CLUSTER_WEIGHT_MAX.value,
      cluster_var_min=_CLUSTER_VAR_MIN.value,
      cluster_var_max=_CLUSTER_VAR_MAX.value,
      pi_1=_PI_1.value,
      baseline_rate_logit=_BASELINE_RATE_LOGIT.value,
      alternative_rate=_ALTERNATIVE_RATE.value,
      delta_threshold=_DELTA_THRESHOLD.value,
      delta_multiplier=_DELTA_MULTIPLIER.value,
      seed=_SEED.value,
  )


_config_from_flags = base_config_from_flags


def _runners_for(runner_type: str) -> list[str]:
  """Returns the ordered list of runner keys to execute for a given runner_type."""
  if runner_type == "all":
    return ["offline_ebh", "type1", "disco"]
  elif runner_type == "sxs":
    return ["offline_ebh", "disco"]
  else:
    return [runner_type]


# ---------------------------------------------------------------------------
# Environment / single-run helpers
# ---------------------------------------------------------------------------


def build_env(
    config: bandit_disco.BanditConfig, seed: int
) -> bandit_env.BanditEnv:
  """Constructs a bandit environment (`ContextualBanditEnv` or `BernoulliBanditEnv`)."""
  if config.env_type == "contextual":
    return bandit_env.ContextualBanditEnv(
        dimension=config.dimension,
        num_vulnerable_clusters=config.num_vulnerable_clusters,
        baseline_rate_logit=config.baseline_rate_logit,
        cluster_weight_min=config.cluster_weight_min,
        cluster_weight_max=config.cluster_weight_max,
        cluster_var_min=config.cluster_var_min,
        cluster_var_max=config.cluster_var_max,
        seed=seed,
    )
  else:
    return bandit_env.BernoulliBanditEnv(
        pi_1=config.pi_1,
        baseline_rate_logit=config.baseline_rate_logit,
        alternative_rate=config.alternative_rate,
        seed=seed,
    )


_build_env = build_env


def run_single(
    config: bandit_disco.BanditConfig,
    runner_type: str,
    seed: int,
    null_tables: Any | None = None,
    total_true_positives: int = 0,
    cluster_true_positives: dict[int, int] | None = None,
) -> bandit_disco.BanditResult:
  """Executes a single bandit simulation run for `runner_type`."""
  env = build_env(config, seed)
  num_rounds = max(1, config.budget // config.chunk_size)
  if total_true_positives == 0 and cluster_true_positives is None:
    ttp, cluster_ttp = env.pre_generate_pool(num_rounds)
  else:
    ttp = total_true_positives
    cluster_ttp = cluster_true_positives or {}
  if runner_type == "disco":
    return bandit_disco.run_disco_bandit(
        config,
        env,
        null_tables=null_tables,
        total_true_positives=ttp,
        cluster_true_positives=cluster_ttp,
    )
  elif runner_type == "type1":
    return bandit_disco.run_type1_bandit(
        config,
        env,
        total_true_positives=ttp,
        cluster_true_positives=cluster_ttp,
    )
  else:
    return bandit_disco.run_offline_ebh_bandit(
        config,
        env,
        total_true_positives=ttp,
        cluster_true_positives=cluster_ttp,
    )


_run_single = run_single


# ---------------------------------------------------------------------------
# Summary / statistics helpers
# ---------------------------------------------------------------------------


def _stats(values: list[float], sqrt_n: float, n: int) -> dict[str, float]:
  """Computes mean/std/sem/ci95/ci_low/ci_high for a list of floats."""
  mean = float(np.mean(values))
  std = float(np.std(values))
  sem = std / sqrt_n if n > 1 else 0.0
  ci95 = 1.96 * sem
  return {
      "mean": mean,
      "std": std,
      "sem": sem,
      "ci95": ci95,
      "ci_low": max(0.0, mean - ci95),
      "ci_high": mean + ci95,
  }


def _get_fdr_bound(r: bandit_disco.BanditResult) -> float:
  if r.runner_type == "disco" and r.epsilon_post is not None:
    return r.epsilon_post * 100.0
  return r.nominal_alpha_used * 100.0


def summarize_runs(
    runs: list[bandit_disco.BanditResult],
    config: bandit_disco.BanditConfig,
) -> dict[str, Any]:
  """Aggregates a list of BanditResult objects into a summary statistics dict."""
  n = len(runs)
  sqrt_n = np.sqrt(n) if n > 0 else 1.0

  def st(values: list[float]) -> dict[str, float]:
    return _stats(values, sqrt_n, n)

  fdr_exceed_list = [
      1.0 if r.empirical_fdr > (_get_fdr_bound(r) + 1e-9) else 0.0 for r in runs
  ]
  fdr_exceed_count = int(sum(fdr_exceed_list))

  delta_post_vals = [
      float(r.delta_post) for r in runs if r.delta_post is not None
  ]
  if delta_post_vals:
    delta_post_stats = st(delta_post_vals)
  else:
    delta_post_stats = {
        "mean": 0.0,
        "std": 0.0,
        "sem": 0.0,
        "ci95": 0.0,
        "ci_low": 0.0,
        "ci_high": 0.0,
    }

  return {
      "fdr": st([r.empirical_fdr for r in runs]),
      "fdr_exceedances": fdr_exceed_count,
      "fdr_exceed_rate": st(fdr_exceed_list),
      "recall": st([r.recall for r in runs]),
      "macro_recall": st([r.macro_recall for r in runs]),
      "cluster_coverage": st([r.cluster_coverage for r in runs]),
      "penalized_cluster_coverage": st(
          [r.penalized_cluster_coverage for r in runs]
      ),
      "delta_post": delta_post_stats,
      "k": st([float(r.total_hypotheses_tested) for r in runs]),
      "tp": st([float(r.true_positives) for r in runs]),
      "fp": st([float(r.false_positives) for r in runs]),
      "probes_null": st([r.avg_probes_null for r in runs]),
      "probes_alt": st([r.avg_probes_alt for r in runs]),
      "nominal_alpha": st([r.nominal_alpha_used for r in runs]),
      "exits_high": st([
          float(sum(1 for s in r.terminal_statuses.values() if s == "H"))
          for r in runs
      ]),
      "exits_low": st([
          float(sum(1 for s in r.terminal_statuses.values() if s == "L"))
          for r in runs
      ]),
      "exits_timeout": st([
          float(
              sum(
                  1
                  for h, s in r.terminal_statuses.items()
                  if s not in ("H", "L")
                  and r.probe_counts.get(h, 0) >= config.n_max
              )
          )
          for r in runs
      ]),
      "exits_budget_expired": st([
          float(
              sum(
                  1
                  for h, s in r.terminal_statuses.items()
                  if s not in ("H", "L")
                  and r.probe_counts.get(h, 0) < config.n_max
              )
          )
          for r in runs
      ]),
  }


_summarize_runs = summarize_runs


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------


def _print_comparison_table(
    runner_type: str,
    summary_data: dict[str, Any],
    results_by_runner: dict[str, list[bandit_disco.BanditResult]],
    config: bandit_disco.BanditConfig,
) -> None:
  """Prints the multi-runner comparison table (all / sxs) or single-runner summary."""
  if runner_type == "all":
    off = summary_data["offline_ebh"]
    t1 = summary_data["type1"]
    smt = summary_data["disco"]
    n_off = len(results_by_runner["offline_ebh"])
    n_t1 = len(results_by_runner["type1"])
    n_smt = len(results_by_runner["disco"])
    off_exceed_str = (
        f"{off['fdr_exceedances']} / {n_off}"
        f" ({(off['fdr_exceedances']/max(1, n_off))*100.0:.1f}%)"
    )
    t1_exceed_str = (
        f"{t1['fdr_exceedances']} / {n_t1}"
        f" ({(t1['fdr_exceedances']/max(1, n_t1))*100.0:.1f}%)"
    )
    smt_exceed_str = (
        f"{smt['fdr_exceedances']} / {n_smt}"
        f" ({(smt['fdr_exceedances']/max(1, n_smt))*100.0:.1f}%)"
    )

    print("=" * 122)
    print(
        f"{'Metric (Mean ± 95% CI)':<36} | {'Offline e-BH':<26} |"
        f" {'Type-I (1/alpha)':<26} | {'DISCO':<26}"
    )
    print("-" * 122)
    print(
        f"{'Empirical FDR (%)':<36} | {off['fdr']['mean']:5.2f}% ±"
        f" {off['fdr']['ci95']:4.2f}%      | {t1['fdr']['mean']:5.2f}% ±"
        f" {t1['fdr']['ci95']:4.2f}%      | {smt['fdr']['mean']:5.2f}% ±"
        f" {smt['fdr']['ci95']:4.2f}%"
    )
    print(
        f"{'FDR Exceedances (> Bound)':<36} | {off_exceed_str:<26} |"
        f" {t1_exceed_str:<26} | {smt_exceed_str:<26}"
    )
    print(
        f"{'Recall (Micro / Total) (%)':<36} | {off['recall']['mean']:5.2f}% ±"
        f" {off['recall']['ci95']:4.2f}%      | {t1['recall']['mean']:5.2f}% ±"
        f" {t1['recall']['ci95']:4.2f}%      | {smt['recall']['mean']:5.2f}% ±"
        f" {smt['recall']['ci95']:4.2f}%"
    )
    if config.env_type == "contextual":
      print(
          f"{'Macro Recall (Cluster-Balanced) (%)':<36} |"
          f" {off['macro_recall']['mean']:5.2f}% ±"
          f" {off['macro_recall']['ci95']:4.2f}%      |"
          f" {t1['macro_recall']['mean']:5.2f}% ±"
          f" {t1['macro_recall']['ci95']:4.2f}%      |"
          f" {smt['macro_recall']['mean']:5.2f}% ±"
          f" {smt['macro_recall']['ci95']:4.2f}%"
      )
      print(
          f"{'Cluster Coverage (Unique Hits) (%)':<36} |"
          f" {off['cluster_coverage']['mean']:5.2f}% ±"
          f" {off['cluster_coverage']['ci95']:4.2f}%      |"
          f" {t1['cluster_coverage']['mean']:5.2f}% ±"
          f" {t1['cluster_coverage']['ci95']:4.2f}%      |"
          f" {smt['cluster_coverage']['mean']:5.2f}% ±"
          f" {smt['cluster_coverage']['ci95']:4.2f}%"
      )
      print(
          f"{f'Penalized Coverage (Delta<={_DELTA_THRESHOLD.value}) (%)':<36} |"
          f" {off['penalized_cluster_coverage']['mean']:5.2f}% ±"
          f" {off['penalized_cluster_coverage']['ci95']:4.2f}%      |"
          f" {t1['penalized_cluster_coverage']['mean']:5.2f}% ±"
          f" {t1['penalized_cluster_coverage']['ci95']:4.2f}%      |"
          f" {smt['penalized_cluster_coverage']['mean']:5.2f}% ±"
          f" {smt['penalized_cluster_coverage']['ci95']:4.2f}%"
      )
    print(
        f"{'True Discoveries (TP)':<36} | {off['tp']['mean']:5.2f} ±"
        f" {off['tp']['ci95']:4.2f}          | {t1['tp']['mean']:5.2f} ±"
        f" {t1['tp']['ci95']:4.2f}          | {smt['tp']['mean']:5.2f} ±"
        f" {smt['tp']['ci95']:4.2f}"
    )
    print(
        f"{'False Discoveries (FP)':<36} | {off['fp']['mean']:5.2f} ±"
        f" {off['fp']['ci95']:4.2f}          | {t1['fp']['mean']:5.2f} ±"
        f" {t1['fp']['ci95']:4.2f}          | {smt['fp']['mean']:5.2f} ±"
        f" {smt['fp']['ci95']:4.2f}"
    )
    print(
        f"{'Avg Hypotheses Tested (K)':<36} | {off['k']['mean']:5.2f} ±"
        f" {off['k']['ci95']:4.2f}          | {t1['k']['mean']:5.2f} ±"
        f" {t1['k']['ci95']:4.2f}          | {smt['k']['mean']:5.2f} ±"
        f" {smt['k']['ci95']:4.2f}"
    )
    print(
        f"{'Avg Probes on Nulls':<36} | {off['probes_null']['mean']:5.2f} ±"
        f" {off['probes_null']['ci95']:4.2f}          |"
        f" {t1['probes_null']['mean']:5.2f} ± {t1['probes_null']['ci95']:4.2f} "
        f"         | {smt['probes_null']['mean']:5.2f} ±"
        f" {smt['probes_null']['ci95']:4.2f}"
    )
    print(
        f"{'Avg Probes on Alternatives':<36} | {off['probes_alt']['mean']:5.2f}"
        f" ± {off['probes_alt']['ci95']:4.2f}          |"
        f" {t1['probes_alt']['mean']:5.2f} ± {t1['probes_alt']['ci95']:4.2f}   "
        f"       | {smt['probes_alt']['mean']:5.2f} ±"
        f" {smt['probes_alt']['ci95']:4.2f}"
    )
    print(
        f"{'Nominal Calibrated Alpha':<36} |"
        f" {off['nominal_alpha']['mean']:6.4f} ±"
        f" {off['nominal_alpha']['ci95']:6.4f}      |"
        f" {t1['nominal_alpha']['mean']:6.4f} ±"
        f" {t1['nominal_alpha']['ci95']:6.4f}      |"
        f" {smt['nominal_alpha']['mean']:6.4f} ±"
        f" {smt['nominal_alpha']['ci95']:6.4f}"
    )
    print("=" * 122)
  elif runner_type == "sxs":
    off = summary_data["offline_ebh"]
    smt = summary_data["disco"]
    n_off = len(results_by_runner["offline_ebh"])
    n_smt = len(results_by_runner["disco"])
    off_exceed_str = (
        f"{off['fdr_exceedances']} / {n_off}"
        f" ({(off['fdr_exceedances']/max(1, n_off))*100.0:.1f}%)"
    )
    smt_exceed_str = (
        f"{smt['fdr_exceedances']} / {n_smt}"
        f" ({(smt['fdr_exceedances']/max(1, n_smt))*100.0:.1f}%)"
    )

    print("=" * 92)
    print(
        f"{'Metric (Mean ± 95% CI)':<36} | {'Offline e-BH':<26} | {'DISCO':<26}"
    )
    print("-" * 92)
    print(
        f"{'Empirical FDR (%)':<36} | {off['fdr']['mean']:5.2f}% ±"
        f" {off['fdr']['ci95']:4.2f}%      | {smt['fdr']['mean']:5.2f}% ±"
        f" {smt['fdr']['ci95']:4.2f}%"
    )
    print(
        f"{'FDR Exceedances (> Bound)':<36} | {off_exceed_str:<26} |"
        f" {smt_exceed_str:<26}"
    )
    print(
        f"{'Recall (Micro / Total) (%)':<36} | {off['recall']['mean']:5.2f}% ±"
        f" {off['recall']['ci95']:4.2f}%      | {smt['recall']['mean']:5.2f}% ±"
        f" {smt['recall']['ci95']:4.2f}%"
    )
    if config.env_type == "contextual":
      print(
          f"{'Macro Recall (Cluster-Balanced) (%)':<36} |"
          f" {off['macro_recall']['mean']:5.2f}% ±"
          f" {off['macro_recall']['ci95']:4.2f}%      |"
          f" {smt['macro_recall']['mean']:5.2f}% ±"
          f" {smt['macro_recall']['ci95']:4.2f}%"
      )
      print(
          f"{'Cluster Coverage (Unique Hits) (%)':<36} |"
          f" {off['cluster_coverage']['mean']:5.2f}% ±"
          f" {off['cluster_coverage']['ci95']:4.2f}%      |"
          f" {smt['cluster_coverage']['mean']:5.2f}% ±"
          f" {smt['cluster_coverage']['ci95']:4.2f}%"
      )
      print(
          f"{f'Penalized Coverage (Delta<={_DELTA_THRESHOLD.value}) (%)':<36} |"
          f" {off['penalized_cluster_coverage']['mean']:5.2f}% ±"
          f" {off['penalized_cluster_coverage']['ci95']:4.2f}%      |"
          f" {smt['penalized_cluster_coverage']['mean']:5.2f}% ±"
          f" {smt['penalized_cluster_coverage']['ci95']:4.2f}%"
      )
    print(
        f"{'True Discoveries (TP)':<36} | {off['tp']['mean']:5.2f} ±"
        f" {off['tp']['ci95']:4.2f}          | {smt['tp']['mean']:5.2f} ±"
        f" {smt['tp']['ci95']:4.2f}"
    )
    print(
        f"{'False Discoveries (FP)':<36} | {off['fp']['mean']:5.2f} ±"
        f" {off['fp']['ci95']:4.2f}          | {smt['fp']['mean']:5.2f} ±"
        f" {smt['fp']['ci95']:4.2f}"
    )
    print(
        f"{'Avg Hypotheses Tested (K)':<36} | {off['k']['mean']:5.2f} ±"
        f" {off['k']['ci95']:4.2f}          | {smt['k']['mean']:5.2f} ±"
        f" {smt['k']['ci95']:4.2f}"
    )
    print(
        f"{'Avg Probes on Nulls':<36} | {off['probes_null']['mean']:5.2f} ±"
        f" {off['probes_null']['ci95']:4.2f}          |"
        f" {smt['probes_null']['mean']:5.2f} ±"
        f" {smt['probes_null']['ci95']:4.2f}"
    )
    print(
        f"{'Avg Probes on Alternatives':<36} | {off['probes_alt']['mean']:5.2f}"
        f" ± {off['probes_alt']['ci95']:4.2f}          |"
        f" {smt['probes_alt']['mean']:5.2f} ± {smt['probes_alt']['ci95']:4.2f}"
    )
    print(
        f"{'Nominal Calibrated Alpha':<36} |"
        f" {off['nominal_alpha']['mean']:6.4f} ±"
        f" {off['nominal_alpha']['ci95']:6.4f}      |"
        f" {smt['nominal_alpha']['mean']:6.4f} ±"
        f" {smt['nominal_alpha']['ci95']:6.4f}"
    )
    print("=" * 92)
  else:
    data = summary_data[runner_type]
    n_run = len(results_by_runner[runner_type])
    print("=" * 92)
    print(f"=== Summary for {runner_type.upper()} ({n_run} runs) ===")
    print(
        f"Empirical FDR:            {data['fdr']['mean']:.2f}% ±"
        f" {data['fdr']['ci95']:.2f}% (95% CI)"
    )
    print(
        f"FDR Exceedances (> Bound): {data['fdr_exceedances']} / {n_run}"
        f" ({(data['fdr_exceedances']/max(1, n_run))*100.0:.1f}%)"
    )
    print(
        f"Recall (Micro / Total):   {data['recall']['mean']:.2f}% ±"
        f" {data['recall']['ci95']:.2f}% (95% CI)"
    )
    if config.env_type == "contextual":
      print(
          f"Macro Recall (Cluster):   {data['macro_recall']['mean']:.2f}% ±"
          f" {data['macro_recall']['ci95']:.2f}% (95% CI)"
      )
      print(
          f"Cluster Coverage (Hits):  {data['cluster_coverage']['mean']:.2f}% ±"
          f" {data['cluster_coverage']['ci95']:.2f}% (95% CI)"
      )
      print(
          "Penalized Coverage:      "
          f" {data['penalized_cluster_coverage']['mean']:.2f}% ±"
          f" {data['penalized_cluster_coverage']['ci95']:.2f}% (95% CI)"
      )
    print(
        f"True Positives:           {data['tp']['mean']:.2f} ±"
        f" {data['tp']['ci95']:.2f}"
    )
    print(
        f"False Positives:          {data['fp']['mean']:.2f} ±"
        f" {data['fp']['ci95']:.2f}"
    )
    print(
        f"Tested Arms (K):          {data['k']['mean']:.2f} ±"
        f" {data['k']['ci95']:.2f}"
    )
    print(
        f"Avg Probes on Nulls:      {data['probes_null']['mean']:.2f} ±"
        f" {data['probes_null']['ci95']:.2f}"
    )
    print(
        f"Avg Probes on Alt:        {data['probes_alt']['mean']:.2f} ±"
        f" {data['probes_alt']['ci95']:.2f}"
    )
    print(
        f"Nominal Calibrated Alpha: {data['nominal_alpha']['mean']:.4f} ±"
        f" {data['nominal_alpha']['ci95']:.4f}"
    )
    print("=" * 92)


def _print_exit_breakdown(
    summary_data: dict[str, Any],
    results_by_runner: dict[str, list[bandit_disco.BanditResult]],
    config: bandit_disco.BanditConfig,
) -> None:
  """Prints the DISCO deactivation / exit breakdown (and optional bin breakdown)."""
  if "disco" not in results_by_runner:
    return

  smt = summary_data["disco"]
  print("\n--- DISCO Deactivation / Exit Breakdown ---")
  print(
      f"  High Exits (H: e >= e_max):        {smt['exits_high']['mean']:5.2f}"
      f" ± {smt['exits_high']['ci95']:4.2f}"
  )
  print(
      f"  Low Exits (L: e < e_min):          {smt['exits_low']['mean']:5.2f}"
      f" ± {smt['exits_low']['ci95']:4.2f}"
  )
  print(
      "  Timeout Exits (n >= n_max):       "
      f" {smt['exits_timeout']['mean']:5.2f} ±"
      f" {smt['exits_timeout']['ci95']:4.2f}"
  )
  print(
      "  Budget Expired (n < n_max):       "
      f" {smt['exits_budget_expired']['mean']:5.2f} ±"
      f" {smt['exits_budget_expired']['ci95']:4.2f}"
  )

  if config.num_active_bins > 1:
    j_total = config.num_active_bins
    bin_edges = status_lib.compute_strip_bins(
        config.e_min, config.e_max, j_total
    )
    num_groups = 10 if j_total > 20 else j_total
    group_counts = [0.0] * num_groups
    disco_runs = results_by_runner["disco"]
    n_runs_disco = max(1, len(disco_runs))
    for r in disco_runs:
      for st in r.terminal_statuses.values():
        parsed_st = status_lib.parse_status(st)
        if parsed_st.is_binned and parsed_st.bin_idx is not None:
          g_idx = min(
              int((parsed_st.bin_idx - 1) * num_groups / j_total),
              num_groups - 1,
          )
          group_counts[g_idx] += 1.0 / n_runs_disco

    def _fmt_e(x: float) -> str:
      if x < 0.1:
        return f"{x:.3f}"
      if x < 10.0:
        return f"{x:.2f}"
      if x < 100.0:
        return f"{x:.1f}"
      return f"{x:.0f}"

    group_strs = []
    bin_breakdown_dict = {}
    for g in range(num_groups):
      b_start_idx = int(g * j_total / num_groups)
      b_end_idx = int((g + 1) * j_total / num_groups)
      e_lo = bin_edges[b_start_idx]
      e_hi = bin_edges[b_end_idx]
      label = f"[{_fmt_e(e_lo)}, {_fmt_e(e_hi)})"
      cnt = group_counts[g]
      group_strs.append(f"{label}: {cnt:4.2f}")
      bin_breakdown_dict[label] = cnt

    smt["bin_breakdown"] = bin_breakdown_dict

    if len(group_strs) <= 5:
      print(
          "  Active Sub-Bin Breakdown (All Active):      "
          f" {' | '.join(group_strs)}"
      )
    else:
      half = (len(group_strs) + 1) // 2
      print(
          "  Active Sub-Bin Breakdown (All Active 1/2):  "
          f" {' | '.join(group_strs[:half])}"
      )
      print(
          "  Active Sub-Bin Breakdown (All Active 2/2):  "
          f" {' | '.join(group_strs[half:])}"
      )


def _compute_bounds_report(
    results_by_runner: dict[str, list[bandit_disco.BanditResult]],
    config: bandit_disco.BanditConfig,
    null_tables: Any | None,
    k_bar: int,
) -> tuple[dict[str, Any], dict[str, float] | None]:
  """Computes FDR certificates / bounds for DISCO runs.

  Args:
    results_by_runner: Simulation results keyed by runner type.
    config: Bandit simulation configuration.
    null_tables: Precomputed null coarsened tables, or None.
    k_bar: Pre-hoc upper bound on total hypotheses.

  Returns:
    (bounds_report_dict, prehoc_bounds) — both empty/None if DISCO not in run.
  """
  if null_tables is None or "disco" not in results_by_runner:
    return {}, None

  disco_runs = results_by_runner["disco"]
  nom_alpha_mean = float(np.mean([r.nominal_alpha_used for r in disco_runs]))
  eps_post_mean = float(
      np.mean(
          [r.epsilon_post for r in disco_runs if r.epsilon_post is not None]
      )
  )
  delta_post_mean = float(
      np.mean([r.delta_post for r in disco_runs if r.delta_post is not None])
  )
  cert_post_mean = float(
      np.mean([
          r.posthoc_certificate
          for r in disco_runs
          if r.posthoc_certificate is not None
      ])
  )

  alpha_for_prehoc = (
      nom_alpha_mean
      if config.alpha_calibration_mode == "pre_hoc"
      else config.alpha
  )
  prehoc_bounds = cert_lib.compute_prehoc_fdr_bounds(
      null_tables=null_tables,
      k_bar=k_bar,
      alpha_star=alpha_for_prehoc,
  )
  if config.num_active_bins == 0:
    # For J=infinity, E_B^max = e_max, not the ternary E_A_max
    prehoc_bounds["unboosted"] = (
        alpha_for_prehoc * config.e_max + prehoc_bounds["baseline_Kbar_PH"]
    )

  j_label = (
      "continuous"
      if config.num_active_bins == 0
      else f"J={config.num_active_bins}"
  )
  print(f"\n=== FDR Certificates & Bounds (DISCO, {j_label}) ===")
  print(
      f"Nominal Calibrated Alpha:    {nom_alpha_mean:.4f} (Target Alpha*:"
      f" {config.alpha:.4f}, Kappa: {config.kappa:.2f})"
  )
  print("--- Pre-Hoc Bounds (Stated Before Running) ---")
  print(f"Pre-Hoc Unboosted Bound:     {prehoc_bounds['unboosted']:.4f}")
  print(f"Pre-Hoc Boosted Bound:       {prehoc_bounds['boosted']:.4f}")
  print(
      f"Pre-Hoc Analytical Ceiling:  {prehoc_bounds['analytical_ceiling']:.4f}"
  )
  print("--- Post-Hoc Bounds (Observed Run Average) ---")
  print(
      f"Post-Hoc Error Bound (eps):  {eps_post_mean:.4f} (delta ="
      f" {delta_post_mean:.6f})"
  )
  print(
      f"Marginal Post-Hoc (eps+del): {cert_post_mean:.4f} (delta ="
      f" {delta_post_mean:.6f})"
  )

  bounds_report_dict = {
      "nominal_calibrated_alpha": nom_alpha_mean,
      "prehoc_unboosted": prehoc_bounds["unboosted"],
      "prehoc_boosted": prehoc_bounds["boosted"],
      "prehoc_analytical_ceiling": prehoc_bounds["analytical_ceiling"],
      "posthoc_epsilon_mean": eps_post_mean,
      "posthoc_delta_mean": delta_post_mean,
      "posthoc_marginal_mean": cert_post_mean,
  }
  return bounds_report_dict, prehoc_bounds


def _compute_alpha_sweep(
    results_by_runner: dict[str, list[bandit_disco.BanditResult]],
    config: bandit_disco.BanditConfig,
    null_tables: Any | None,
    prehoc_bounds: dict[str, float] | None,
    n_runs: int,
) -> list[dict[str, Any]]:
  """Computes and prints the post-run alpha sweep table.

  Args:
    results_by_runner: Simulation results keyed by runner type.
    config: Bandit simulation configuration.
    null_tables: Precomputed null coarsened tables, or None.
    prehoc_bounds: Unused; kept for signature compatibility.
    n_runs: Number of Monte Carlo runs executed.

  Returns:
    List of per-alpha-level record dicts (empty if sweep not enabled).
  """
  del prehoc_bounds
  if (
      not _ENABLE_ALPHA_SWEEP.value
      or "disco" not in results_by_runner
      or null_tables is None
  ):
    return []

  sweep_alphas = [
      float(x)
      for x in (
          np.linspace(
              np.sqrt(_ALPHA_SWEEP_MIN.value),
              np.sqrt(_ALPHA_SWEEP_MAX.value),
              _ALPHA_SWEEP_STEPS.value,
          )
          ** 2
      )
  ]
  if _ALPHA.value not in sweep_alphas:
    sweep_alphas.append(_ALPHA.value)
    sweep_alphas.sort()

  print(
      f"\n=== Post-Run Alpha Sweep ({len(sweep_alphas)} Alpha Levels across"
      f" {n_runs} runs) ==="
  )
  header = (
      f"{'Alpha*':>8} | {'Nominal Alpha':>13} | {'Discoveries':>11} |"
      f" {'Empirical FDR':>13} | {'Recall':>8} |"
      f" {'Post-Hoc (eps)':>16} | {'Marginal Post-Hoc':>19} |"
      f" {'Delta':>8}"
  )
  print(header)
  print("-" * len(header))

  # Pre-compute sweep entries for all alphas in one call per run.
  all_run_entries: list[list[calibration_lib.DiscoAlphaSweepEntry]] = []
  for r in results_by_runner["disco"]:
    sw_entries = calibration_lib.compute_disco_alpha_sweep(
        final_e_values=r.final_e_values,
        terminal_statuses=r.terminal_statuses,
        probe_counts=r.probe_counts,
        null_tables=null_tables,
        sweep_alphas=sweep_alphas,
        alpha_calibration_mode=config.alpha_calibration_mode,
        use_boosting=config.use_boosting,
        kappa=config.kappa,
    )
    all_run_entries.append(sw_entries)

  alpha_sweep_records = []
  for ai, a_star in enumerate(sweep_alphas):
    nom_vals = []
    disc_vals = []
    fdr_vals = []
    rec_vals = []
    eps_vals = []
    cert_vals = []
    delta_vals = []

    for ri, r in enumerate(results_by_runner["disco"]):
      entries = all_run_entries[ri]
      if ai >= len(entries):
        continue
      entry = entries[ai]
      nom_vals.append(entry.nominal_alpha)
      disc_vals.append(entry.num_discoveries)
      eps_vals.append(
          entry.epsilon_post_boost
          if config.use_boosting
          else entry.epsilon_post
      )
      cert_vals.append(
          entry.posthoc_boosted
          if config.use_boosting
          else entry.posthoc_unboosted
      )
      delta_vals.append(entry.delta_post)

      sw_tp = 0
      sw_fp = 0
      for h_id in entry.rejected_hypotheses:
        if r.is_alternative_map.get(h_id, False):
          sw_tp += 1
        else:
          sw_fp += 1
      num_d = entry.num_discoveries
      sw_fdr = (sw_fp / max(1, num_d)) * 100.0 if num_d > 0 else 0.0
      sw_rec = (
          (sw_tp / max(1, r.total_true_positives)) * 100.0
          if r.total_true_positives > 0
          else 0.0
      )
      fdr_vals.append(sw_fdr)
      rec_vals.append(sw_rec)

    m_nom = float(np.mean(nom_vals)) if nom_vals else 0.0
    m_disc = float(np.mean(disc_vals)) if disc_vals else 0.0
    m_fdr = float(np.mean(fdr_vals)) if fdr_vals else 0.0
    m_rec = float(np.mean(rec_vals)) if rec_vals else 0.0
    m_eps = float(np.mean(eps_vals)) if eps_vals else 0.0
    m_cert = float(np.mean(cert_vals)) if cert_vals else 0.0
    m_delta = float(np.mean(delta_vals)) if delta_vals else 0.0

    is_run_alpha = " *" if abs(a_star - config.alpha) < 1e-6 else ""

    print(
        f"{a_star:8.4f} | {m_nom:13.4f} | {m_disc:11.2f} |"
        f" {m_fdr:12.1f}% | {m_rec:7.1f}% | {m_eps:16.4f} |"
        f" {m_cert:19.4f} | {m_delta:8.4f}{is_run_alpha}"
    )

    alpha_sweep_records.append({
        "alpha_target": a_star,
        "nominal_alpha": m_nom,
        "num_discoveries": m_disc,
        "empirical_fdr": m_fdr,
        "recall": m_rec,
        "epsilon_post": m_eps,
        "marginal_posthoc": m_cert,
        "delta_post": m_delta,
    })

  return alpha_sweep_records


def _resolve_output_path(
    runner_type: str, config: bandit_disco.BanditConfig
) -> str:
  """Resolves the target output file path from flags."""
  if _OUTPUT_PATH.value:
    return _OUTPUT_PATH.value
  if not _OUTPUT_DIR.value:
    return ""

  filename = f"{runner_type}_seed{config.seed}_{int(time.time()*1000)}.json"
  return os.path.join(_OUTPUT_DIR.value, filename)


def _write_outputs(
    target_output_file: str,
    summary_data: dict[str, Any],
    bounds_report_dict: dict[str, Any],
    alpha_sweep_records: list[dict[str, Any]],
    results_by_runner: dict[str, list[bandit_disco.BanditResult]],
) -> None:
  """Writes the main JSON results file and optional per-run file."""
  results_dict = {
      "params": {
          flag.name: flag.value
          for flag in flags.FLAGS.get_key_flags_for_module(__name__)
      },
      "summary": summary_data,
      "bounds_report": bounds_report_dict,
      "alpha_sweep": alpha_sweep_records,
  }
  out_dir = os.path.dirname(target_output_file)
  if out_dir and not os.path.exists(out_dir):
    os.makedirs(out_dir, exist_ok=True)
  with open(target_output_file, "w") as f:
    json.dump(results_dict, f, indent=2)
  print(f"\nResults written to: {target_output_file}")

  if _SAVE_PER_RUN_DATA.value and results_by_runner:
    per_run_data = {}
    for r_type, runs in results_by_runner.items():
      per_run_data[r_type] = {
          "cluster_coverage": [float(r.cluster_coverage) for r in runs],
          "penalized_cluster_coverage": [
              float(r.penalized_cluster_coverage) for r in runs
          ],
          "recall": [float(r.recall) for r in runs],
          "macro_recall": [float(r.macro_recall) for r in runs],
          "empirical_fdr": [float(r.empirical_fdr) for r in runs],
          "delta_post": [
              float(r.delta_post) if r.delta_post is not None else 0.0
              for r in runs
          ],
          "epsilon_post": [
              float(r.epsilon_post) if r.epsilon_post is not None else None
              for r in runs
          ],
          "true_positives": [int(r.true_positives) for r in runs],
          "false_positives": [int(r.false_positives) for r in runs],
          "total_hypotheses_tested": [
              int(r.total_hypotheses_tested) for r in runs
          ],
          "avg_probes_null": [float(r.avg_probes_null) for r in runs],
          "avg_probes_alt": [float(r.avg_probes_alt) for r in runs],
          "nominal_alpha_used": [float(r.nominal_alpha_used) for r in runs],
      }
    raw_runs_dict = {
        "params": {
            flag.name: flag.value
            for flag in flags.FLAGS.get_key_flags_for_module(__name__)
        },
        "num_mc_runs": (
            len(list(results_by_runner.values())[0]) if results_by_runner else 0
        ),
        "runs": per_run_data,
    }
    raw_output_file = (
        target_output_file[:-5] + "_runs.json"
        if target_output_file.endswith(".json")
        else target_output_file + "_runs.json"
    )
    with open(raw_output_file, "w") as f:
      json.dump(raw_runs_dict, f, indent=2)
    print(f"Per-run details written to: {raw_output_file}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str]) -> None:
  """Executes FDR-controlled bandit simulations from CLI flags."""
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  config = _config_from_flags()
  n_runs = _NUM_MC_RUNS.value
  runner_type = _RUNNER_TYPE.value

  print(
      f"\n=== FDR-Controlled Bandit Evaluation ({_ENV_TYPE.value.upper()}) ==="
  )
  print(
      f"Budget: {config.budget}, Chunk: {config.chunk_size}, Alpha*:"
      f" {config.alpha}, Kappa: {config.kappa}, P-Value Mode:"
      f" {config.p_value_mode}"
  )
  if config.env_type == "contextual":
    print(f"Beta: {config.gp_beta}, Lambda: {config.ridge_lambda}")
  print(f"Strip: [{config.e_min}, {config.e_max}), n_max: {config.n_max}")
  print(
      f"Boosting: {config.use_boosting}, Calibration:"
      f" {config.alpha_calibration_mode}, MC Runs: {n_runs}\n"
  )
  if config.use_boosting and runner_type in (
      "offline_ebh",
      "type1",
      "sxs",
      "all",
  ):
    print(
        "[Note] Boosting is only mathematically valid with coarsened"
        " e-processes (DISCO). Offline baselines will run with boosting=False,"
        " while DISCO will use boosting=True.\n"
    )

  # Precompute null tables once for all MC runs
  k_bar = bandit_disco.resolve_k_bar(config)
  null_tables = (
      bandit_disco.build_null_tables(config, k_bar)
      if runner_type in ("disco", "sxs", "all")
      else None
  )

  runners_to_execute = _runners_for(runner_type)
  results_by_runner: dict[str, list[bandit_disco.BanditResult]] = {
      r: [] for r in runners_to_execute
  }

  # Live Simulation Progress Bar
  start_time = time.time()
  num_rounds = max(1, config.budget // config.chunk_size)
  for i in range(n_runs):
    run_seed = config.seed + i * 1000
    # Pre-generate pool once to get fixed total_true_positives and cluster
    # breakdown for this MC run.
    pool_env = _build_env(config, run_seed)
    ttp, cluster_ttp = pool_env.pre_generate_pool(num_rounds)
    for r_type in runners_to_execute:
      res = _run_single(
          config,
          r_type,
          run_seed,
          null_tables=null_tables,
          total_true_positives=ttp,
          cluster_true_positives=cluster_ttp,
      )
      results_by_runner[r_type].append(res)

    done = i + 1
    elapsed = time.time() - start_time
    rate = done / max(1e-5, elapsed)
    eta = (n_runs - done) / max(1e-5, rate)
    pct = (done / n_runs) * 100.0
    bar_len = 30
    filled = int(bar_len * done // n_runs)
    bar = "█" * filled + "-" * (bar_len - filled)
    sys.stdout.write(
        f"\r[Simulation Progress] |{bar}| {done}/{n_runs} runs ({pct:.1f}%)"
        f" [{elapsed:04.1f}s, {rate:4.1f} runs/s, ETA: {eta:04.1f}s]"
    )
    sys.stdout.flush()

  sys.stdout.write("\n\n")
  sys.stdout.flush()

  # Aggregate results across MC runs
  summary_data: dict[str, Any] = {}
  for r_type in runners_to_execute:
    summary_data[r_type] = _summarize_runs(results_by_runner[r_type], config)

  _print_comparison_table(runner_type, summary_data, results_by_runner, config)
  _print_exit_breakdown(summary_data, results_by_runner, config)

  bounds_report_dict, prehoc_bounds = _compute_bounds_report(
      results_by_runner, config, null_tables, k_bar
  )
  alpha_sweep_records = _compute_alpha_sweep(
      results_by_runner,
      config,
      null_tables,
      prehoc_bounds,
      n_runs,
  )

  target_output_file = _resolve_output_path(runner_type, config)
  if target_output_file:
    _write_outputs(
        target_output_file,
        summary_data,
        bounds_report_dict,
        alpha_sweep_records,
        results_by_runner,
    )


if __name__ == "__main__":
  app.run(main)
