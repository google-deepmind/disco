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

"""Compute typical values for the elements of the DISCO FDR bound.

Outputs E_A (expected value of surviving null walk), P_H (probability of null
walk exiting high), multi-bin conditional expectations, boosted phi
expectations, and calibrated pre-hoc FDR bounds across standard values of e_max,
n_max, kappa, num_bins, budget, and chunk_size.
"""

from collections.abc import Sequence
import dataclasses
import json
import os
import re
import time
from typing import Any
from absl import app
from absl import flags
import numpy as np
from disco.stats import boosting

_CALCULATION_MODE = flags.DEFINE_enum(
    "calculation_mode",
    "grid",
    ["grid", "tuples"],
    "Calculation mode: 'grid' to compute over Cartesian product of"
    " hyperparameters, or 'tuples' to evaluate only specified hyperparameter"
    " tuples.",
)
_TUPLES = flags.DEFINE_string(
    "tuples",
    "",
    "Comma- or semicolon-separated list of tuples '(kappa, e_max, n_max)' or"
    " '(kappa, e_max, n_max, num_bins)' or '(kappa, e_max, n_max, num_bins,"
    " e_min)', or a path to a file containing tuples. When provided or when"
    " --calculation_mode=tuples, computes only these tuples.",
)
_E_MAX = flags.DEFINE_string(
    "e_max",
    "50.0,100.0,500.0,1000.0,2000.0",
    "Comma-separated list of upper thresholds e_max.",
)
_N_MAX = flags.DEFINE_string(
    "n_max",
    "5,10,15,20",
    "Comma-separated list of max probes n_max (query cap).",
)
_KAPPA = flags.DEFINE_string(
    "kappa",
    "0.5,0.1741",
    "Comma-separated list of kappa calibrator values.",
)
_NUM_BINS = flags.DEFINE_string(
    "num_bins",
    "1,2,3,5",
    "Comma-separated list of active bin counts (1 = binary active strip [0,"
    " e_max)).",
)
_E_MIN = flags.DEFINE_float(
    "e_min",
    0.0,
    "Lower threshold e_min for multi-bin active partitions (J > 1). When set >"
    " 0, the lowest active bin is [0, e_min) and the remaining J-1 bins are"
    " log-spaced between e_min and e_max.",
)
_ALPHA = flags.DEFINE_float(
    "alpha",
    0.05,
    "Target FDR significance level alpha*.",
)
_BUDGET = flags.DEFINE_integer(
    "budget",
    30,
    "Total probing budget (number of probing rounds, or raw query budget"
    " when chunk_size > 1).",
)
_CHUNK_SIZE = flags.DEFINE_integer(
    "chunk_size",
    1,
    "Number of queries per probe round (chunk size). Effective probe budget"
    " and hypothesis cap K_bar are computed as N_probes = budget //"
    " chunk_size.",
)
_N_SIM = flags.DEFINE_integer(
    "n_sim",
    500_000,
    "Number of Monte Carlo simulations for null random walks.",
)
_SEED = flags.DEFINE_integer(
    "seed",
    42,
    "Random seed for reproducibility.",
)
_MODE = flags.DEFINE_enum(
    "mode",
    "all",
    ["all", "summary", "detailed", "multibin", "latex", "json"],
    "Output display mode.",
)
_OUTPUT_FILE = flags.DEFINE_string(
    "output_file",
    "",
    "Optional file path to save output.",
)


@dataclasses.dataclass(frozen=True)
class ParamTuple:
  """Parameter configuration tuple for DISCO bound computation."""

  kappa: float
  e_max: float
  n_max: int
  num_bins: int = 1
  e_min: float = 0.0
  num_bins_explicit: bool = False


DEFAULT_TUPLES: tuple[ParamTuple, ...] = (
    ParamTuple(kappa=0.5, e_max=50.0, n_max=5, num_bins=1),
    ParamTuple(kappa=0.5, e_max=100.0, n_max=10, num_bins=1),
    ParamTuple(kappa=0.5, e_max=500.0, n_max=15, num_bins=1),
    ParamTuple(kappa=0.1741, e_max=50.0, n_max=5, num_bins=1),
    ParamTuple(kappa=0.1741, e_max=100.0, n_max=10, num_bins=1),
    ParamTuple(kappa=0.1741, e_max=500.0, n_max=15, num_bins=1),
)


def _list_to_param_tuple(
    parts: Sequence[Any],
    default_e_min: float = 0.0,
    default_num_bins: int = 1,
) -> ParamTuple:
  """Converts a sequence of string/numeric elements to a ParamTuple."""
  if len(parts) == 3:
    return ParamTuple(
        kappa=float(parts[0]),
        e_max=float(parts[1]),
        n_max=int(parts[2]),
        num_bins=default_num_bins,
        e_min=default_e_min,
        num_bins_explicit=False,
    )
  elif len(parts) == 4:
    return ParamTuple(
        kappa=float(parts[0]),
        e_max=float(parts[1]),
        n_max=int(parts[2]),
        num_bins=int(parts[3]),
        e_min=default_e_min,
        num_bins_explicit=True,
    )
  elif len(parts) >= 5:
    return ParamTuple(
        kappa=float(parts[0]),
        e_max=float(parts[1]),
        n_max=int(parts[2]),
        num_bins=int(parts[3]),
        e_min=float(parts[4]),
        num_bins_explicit=True,
    )
  else:
    raise ValueError(
        "Tuple must have 3, 4, or 5 elements (kappa, e_max, n_max, [num_bins,"
        f" [e_min]]), got: {parts}"
    )


def parse_tuples(
    tuples_str: str,
    default_e_min: float = 0.0,
    default_num_bins: int = 1,
) -> list[ParamTuple]:
  """Parses a string of tuples in various formats into a list of ParamTuples.

  Supports:
  - Parenthesized tuples: '(0.5, 100, 5), (0.5, 500, 10, 2)'
  - Square-bracketed tuples: '[(0.5, 100, 5), (0.5, 500, 10)]'
  - Semicolon-separated tuples: '0.5, 100, 5; 0.5, 500, 10, 2'
  - JSON array of arrays: '[[0.5, 100, 5], [0.5, 500, 10, 2]]'
  - Path to a file containing any of the above.

  Args:
    tuples_str: String of tuples or path to a file containing tuples.
    default_e_min: Default lower threshold if not specified in the tuple.
    default_num_bins: Default number of bins if not specified in the tuple.

  Returns:
    List of parsed ParamTuple instances.
  """
  tuples_str = tuples_str.strip()
  if not tuples_str:
    return []

  if os.path.isfile(tuples_str):
    with open(tuples_str) as f:
      tuples_str = f.read().strip()

  # JSON array of arrays
  if tuples_str.startswith("[") and not re.search(r"\([^)]+\)", tuples_str):
    try:
      raw_list = json.loads(tuples_str)
      if isinstance(raw_list, list):
        result = []
        for item in raw_list:
          if isinstance(item, (list, tuple)):
            result.append(
                _list_to_param_tuple(item, default_e_min, default_num_bins)
            )
        if result:
          return result
    except json.JSONDecodeError:
      pass

  # Parenthesized groups
  groups = re.findall(r"\(([^)]+)\)", tuples_str)
  if not groups:
    # Bracket groups
    groups = re.findall(r"\[([^\]]+)\]", tuples_str)

  if groups:
    items = groups
  else:
    if ";" in tuples_str:
      items = [x for x in tuples_str.split(";") if x.strip()]
    elif "\n" in tuples_str:
      items = [x for x in tuples_str.split("\n") if x.strip()]
    else:
      items = [tuples_str]

  result = []
  for item in items:
    parts = [x.strip() for x in item.split(",") if x.strip()]
    if not parts:
      continue
    result.append(_list_to_param_tuple(parts, default_e_min, default_num_bins))
  return result


def pareto_conditional_mean(a: float, b: float, kappa: float) -> float:
  """Computes analytical conditional expectation E[X | X in [a, b)] for X ~ Pareto tail."""
  if a <= 0.0:
    return 0.5 * b
  if b <= a:
    return a
  beta = 1.0 / (1.0 - kappa)
  if abs(beta - 2.0) < 1e-6:
    return 2.0 * a * b / (a + b)
  if abs(beta - 1.0) < 1e-6:
    return float((b - a) / (np.log(b) - np.log(a)))
  num = (beta / (beta - 1.0)) * (a ** (1.0 - beta) - b ** (1.0 - beta))
  den = a ** (-beta) - b ** (-beta)
  if den == 0.0:
    return 0.5 * (a + b)
  return float(num / den)


def precompute_boost_factors(
    alpha: float,
    kappa: float,
    max_n_max: int,
    k_bar: int,
) -> dict[int, float]:
  """Precomputes boost factors b(m, k_bar) once for all m = 1..max_n_max."""
  boost_map = {}
  for m in range(1, max_n_max + 1):
    b = boosting.compute_boost_factor(alpha, kappa, m, k_bar, use_exact_t=True)
    boost_map[m] = b
  return boost_map


def compute_base_walk_stats(
    log_m_all: np.ndarray,
    kappa: float,
    e_max: float,
    n_max: int,
    budget_probes: int,
    k_bar: int,
) -> dict[str, Any]:
  """Computes core binary random walk statistics for (kappa, e_max, n_max)."""
  log_m = log_m_all[:, :n_max]
  log_e_max = np.log(e_max)

  hit_high = log_m >= log_e_max
  has_exit = np.any(hit_high, axis=1)
  first_exit_idx = np.where(has_exit, np.argmax(hit_high, axis=1), n_max)

  p_h_table = np.zeros(n_max)
  p_surv_table = np.zeros(n_max)
  e_a_mc_table = np.zeros(n_max)
  e_a_formula_table = np.zeros(n_max)

  e_h = e_max / kappa

  for m_idx in range(n_max):
    exited_by_m = first_exit_idx <= m_idx
    p_h = float(np.mean(exited_by_m))
    p_h_table[m_idx] = p_h
    p_surv = 1.0 - p_h
    p_surv_table[m_idx] = p_surv

    surv_mask = first_exit_idx > m_idx
    if np.any(surv_mask):
      m_surv = np.exp(log_m[surv_mask, m_idx])
      e_a_mc = float(np.mean(m_surv))
    else:
      e_a_mc = 0.0
    e_a_mc_table[m_idx] = e_a_mc

    # Exact martingale formula (proof Eq. 11)
    if p_surv > 0:
      e_a_formula = max(0.0, min(1.0, (1.0 - p_h * e_h) / p_surv))
    else:
      e_a_formula = 0.0
    e_a_formula_table[m_idx] = e_a_formula

  del budget_probes
  e_a_worst = float(np.max(e_a_formula_table))
  p_h_terminal = float(p_h_table[-1])
  baseline_bad_event = float(k_bar * p_h_terminal)

  return {
      "kappa": kappa,
      "e_max": e_max,
      "n_max": n_max,
      "E_H": e_h,
      "P_H_table": p_h_table,
      "P_surv_table": p_surv_table,
      "P_H": p_h_terminal,
      "E_A_table": e_a_mc_table,
      "E_A_formula_table": e_a_formula_table,
      "E_A": e_a_worst,
      "baseline_K_PH": baseline_bad_event,
      "first_exit_idx": first_exit_idx,
  }


def compute_multibin_stats(
    kappa: float,
    e_max: float,
    n_max: int,
    num_bins: int,
    e_min: float = 0.0,
) -> tuple[float, np.ndarray, np.ndarray]:
  """Computes multi-bin active partition edges and top sub-bin expectation."""
  if num_bins <= 1:
    return 1.0, np.array([0.0, e_max]), np.zeros((1, n_max))

  if e_min > 0.0:
    upper_edges = np.geomspace(e_min, e_max, num_bins)
    bin_edges = np.concatenate([[0.0], upper_edges])
  else:
    b_1 = max(1.0, e_max ** (1.0 / num_bins))
    upper_edges = np.geomspace(b_1, e_max, num_bins)
    bin_edges = np.concatenate([[0.0], upper_edges])

  e_bin_table = np.zeros((num_bins, n_max))
  top_low = bin_edges[num_bins - 1]
  top_high = bin_edges[num_bins]
  top_pareto = pareto_conditional_mean(top_low, top_high, kappa)

  for m_idx in range(n_max):
    e_bin_table[num_bins - 1, m_idx] = top_pareto

  e_bin_max = float(np.max(e_bin_table[num_bins - 1, :]))
  return e_bin_max, bin_edges, e_bin_table


def format_2sigfigs(val: float, is_sci: bool = False) -> str:
  """Formats a float to 2 significant figures for clean LaTeX presentation."""
  if val == 0.0 or val == 0:
    return "0.0e+00" if is_sci else "0"
  if is_sci:
    return f"{val:.1e}"
  abs_v = abs(val)
  if abs_v < 0.001 or abs_v >= 1000.0:
    return f"{val:.1e}"
  if abs_v < 0.01:
    return f"{val:.4f}"
  if abs_v < 0.1:
    return f"{val:.3f}"
  if abs_v < 1.0:
    return f"{val:.2f}"
  if abs_v < 10.0:
    return f"{val:.1f}"
  if abs_v < 100.0:
    return f"{val:.0f}"
  return f"{round(val, -1):.0f}"


class _NumpyEncoder(json.JSONEncoder):
  """Custom encoder for numpy data types."""

  def default(self, o: Any) -> Any:
    if isinstance(o, np.ndarray):
      return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
      return o.item()
    return super().default(o)


def generate_text_reports(
    records: list[dict[str, Any]],
    alpha: float,
    budget_raw: int,
    budget_probes: int,
    chunk_size: int,
    k_bar: int,
    mode: str,
    e_min: float = 0.0,
    is_tuple_mode: bool = False,
) -> str:
  """Formats comprehensive text, markdown, or JSON output."""
  if mode == "json":
    metadata: dict[str, Any] = {
        "alpha": alpha,
        "budget_raw": budget_raw,
        "budget_probes": budget_probes,
        "chunk_size": chunk_size,
        "k_bar": k_bar,
        "calculation_mode": "tuples" if is_tuple_mode else "grid",
    }
    if e_min > 0.0:
      metadata["e_min"] = e_min
    json_data = {
        "metadata": metadata,
        "records": records,
    }
    return json.dumps(json_data, cls=_NumpyEncoder, indent=2)

  lines = []
  lines.append("=" * 95)
  lines.append("DISCO FDR BOUND: TYPICAL VALUES OF THE ELEMENTS")
  if chunk_size > 1:
    chunk_info = (
        f" (effective probes N_probes = {budget_probes} at chunk_size ="
        f" {chunk_size})"
    )
  else:
    chunk_info = ""
  lines.append(
      f"Parameters: Target alpha* = {alpha}, Budget N ="
      f" {budget_raw}{chunk_info}, Cap K_bar = {k_bar}"
  )
  lines.append("=" * 95)

  has_multibin = any(r["num_bins"] > 1 for r in records)

  if mode in ("all", "summary"):
    lines.append("\n" + "#" * 95)
    if has_multibin or is_tuple_mode:
      lines.append(
          "TABLE 1: TYPICAL VALUES FOR DISCO (Active Strip [0, e_max))"
      )
      lines.append("#" * 95)
      header = (
          f"{'kappa':<6} {'e_max':<7} {'n_max':<5} {'J':<3} | {'P_H':<11}"
          f" {'E_A':<8} | {'K_bar*P_H':<10} | {'FDR_Bound':<12}"
          f" {'alpha_pre':<10}"
      )
      lines.append(header)
      lines.append("-" * len(header))
      for r in records:
        lines.append(
            f"{r['kappa']:<6.4f} {r['e_max']:<7.1f} {r['n_max']:<5d}"
            f" {r['num_bins']:<3d} | {r['P_H']:<11.3e} {r['E_bin_max']:<8.4f} |"
            f" {r['baseline_K_PH']:<10.3e} | {r['prehoc_baseline']:<12.4f}"
            f" {r['calibrated_alpha_baseline']:<10.4f}"
        )
    else:
      lines.append(
          "TABLE 1: TYPICAL VALUES FOR BINARY DISCO (J = 1, Active Strip [0,"
          " e_max))"
      )
      lines.append("#" * 95)
      header = (
          f"{'kappa':<6} {'e_max':<7} {'n_max':<5} | {'P_H':<11} {'E_A':<8}"
          f" | {'K_bar*P_H':<10} | {'FDR_Bound':<12} {'alpha_pre':<10}"
      )
      lines.append(header)
      lines.append("-" * len(header))
      for r in records:
        lines.append(
            f"{r['kappa']:<6.4f} {r['e_max']:<7.1f} {r['n_max']:<5d} |"
            f" {r['P_H']:<11.3e} {r['E_A']:<8.4f} |"
            f" {r['baseline_K_PH']:<10.3e} | {r['prehoc_baseline']:<12.4f}"
            f" {r['calibrated_alpha_baseline']:<10.4f}"
        )

  if mode in ("all", "detailed"):
    lines.append("\n" + "#" * 95)
    lines.append(
        "TABLE 2: STEP-BY-STEP TRAJECTORY (m = 1..n_max) FOR SURVIVING WALKS"
    )
    lines.append("#" * 95)
    detail_records = [r for r in records if r["num_bins"] == 1]
    # In "all" mode, show representative first configuration. In "detailed"
    # mode, show all.
    target_records = (
        detail_records if mode == "detailed" else detail_records[:1]
    )
    for target_rec in target_records:
      lines.append(
          f"\nConfiguration: kappa = {target_rec['kappa']}, e_max ="
          f" {target_rec['e_max']}, n_max = {target_rec['n_max']}, J = 1"
      )
      lines.append(
          f"{'m':<3} | {'P_H(m)':<12} {'P_surv(m)':<12} | {'E_A_MC(m)':<11}"
          f" {'E_A_formula(m)':<14} | {'b(m, K_bar)':<12}"
          f" {'Ceiling: a/P_surv':<18}"
      )
      lines.append("-" * 95)
      for m in range(1, target_rec["n_max"] + 1):
        idx = m - 1
        lines.append(
            f"{m:<3d} | {target_rec['P_H_table'][idx]:<12.4e}"
            f" {target_rec['P_surv_table'][idx]:<12.6f} |"
            f" {target_rec['E_A_table'][idx]:<11.5f}"
            f" {target_rec['E_A_formula_table'][idx]:<14.5f} |"
            f" {target_rec['boost_factors'][idx]:<12.3f}"
            f" {target_rec['phi_analytical_ceiling'][idx]:<18.6f}"
        )

  if mode in ("all", "multibin"):
    lines.append("\n" + "#" * 95)
    lines.append(
        "TABLE 3: SHRINKAGE TRADEOFF: BINARY VS MULTI-BIN ACTIVE PARTITIONS"
    )
    lines.append("#" * 95)
    lines.append(
        f"{'J (Bins)':<8} | {'Partition Style':<30} | {'E_A':<12}"
        f" {'P_H':<11} | {'Pre-hoc Bound':<18}"
    )
    lines.append("-" * 95)
    for r in records:
      if (
          r["kappa"] == records[0]["kappa"]
          and r["e_max"] == records[0]["e_max"]
          and r["n_max"] == records[0]["n_max"]
      ):
        j_val = r["num_bins"]
        if j_val == 1:
          style = "Binary [0, e_max)"
        elif r.get("e_min", 0.0) > 0.0:
          emi = r["e_min"]
          style = (
              f"[0, {emi:g}) + {j_val-1} log-spaced in [{emi:g},"
              f" {r['e_max']:.0f})"
          )
        else:
          style = f"Log-spaced {j_val} sub-bins in [0, {r['e_max']:.0f})"
        lines.append(
            f"{j_val:<8d} | {style:<30} | {r['E_bin_max']:<12.4f}"
            f" {r['P_H']:<11.3e} | {r['prehoc_baseline']:<18.4f}"
        )

  if mode in ("all", "latex"):
    lines.append("\n" + "#" * 95)
    lines.append("LATEX TABLE FRAGMENTS (Ready for paper / appendix)")
    lines.append("#" * 95)
    lines.append("% Table: Typical values for DISCO FDR bound elements")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\small")
    show_partition_col = (
        has_multibin
        or is_tuple_mode
        or any(r.get("num_bins_explicit", False) for r in records)
    )
    if show_partition_col:
      lines.append("\\begin{tabular}{@{}rrrrrrrr@{}} \\toprule")
      lines.append(
          "$\\kappa$ & $e_{\\max}$ & $n_{\\max}$ & $|\\mathcal{P}|$ &"
          " $P_H(n_{\\max})$ & $E_A$ & $\\mathrm{Bound}$ &"
          " $\\alpha^\\star$ \\\\ \\midrule"
      )
      for r in records:
        kp_str = f"{r['kappa']:.2g}"
        emax_str = f"{r['e_max']:.0f}"
        nmax_str = f"{r['n_max']}"
        j_str = f"{r['num_bins']}"
        ph_str = format_2sigfigs(r["P_H"], is_sci=True)
        ea_str = format_2sigfigs(r["E_bin_max"])
        fdr_base_str = format_2sigfigs(r["prehoc_baseline"])
        apre_str = format_2sigfigs(r["calibrated_alpha_baseline"])
        lines.append(
            f"{kp_str} & {emax_str} & {nmax_str} & {j_str} &"
            f" {ph_str} & {ea_str} &"
            f" {fdr_base_str} &"
            f" {apre_str} \\\\"
        )
    else:
      lines.append("\\begin{tabular}{@{}rrrrrrr@{}} \\toprule")
      lines.append(
          "$\\kappa$ & $e_{\\max}$ & $n_{\\max}$ & $P_H(n_{\\max})$ & $E_A$ &"
          " $\\mathrm{Bound}$ &"
          " $\\alpha^\\star$ \\\\ \\midrule"
      )
      for r in records:
        kp_str = f"{r['kappa']:.2g}"
        emax_str = f"{r['e_max']:.0f}"
        nmax_str = f"{r['n_max']}"
        ph_str = format_2sigfigs(r["P_H"], is_sci=True)
        ea_str = format_2sigfigs(r["E_A"])
        fdr_base_str = format_2sigfigs(r["prehoc_baseline"])
        apre_str = format_2sigfigs(r["calibrated_alpha_baseline"])
        lines.append(
            f"{kp_str} & {emax_str} & {nmax_str} &"
            f" {ph_str} & {ea_str} &"
            f" {fdr_base_str} &"
            f" {apre_str} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(
        "\\caption{\n"
        "Typical values for \\textsc{DISCO} pre-hoc FDR bound and"
        f" $\\alpha^\\star$ ($\\alpha={alpha:g}$, $B={budget_raw}$,"
        f" $\\bar{{K}}={k_bar}$)}}"
    )
    lines.append("\\label{tab:disco_bound_typical_values}")
    lines.append("\\end{table}")

  return "\n".join(lines)


def compute_bounds_for_tuples(
    param_tuples: Sequence[ParamTuple],
    alpha: float = 0.05,
    budget_raw: int = 30,
    chunk_size: int = 1,
    n_sim: int = 500_000,
    seed: int = 42,
) -> list[dict[str, Any]]:
  """Computes DISCO FDR bound statistics for an explicit sequence of ParamTuples."""
  if not param_tuples:
    return []

  budget_probes = max(1, budget_raw // chunk_size)
  k_bar = budget_probes

  # Group unique kappas preserving first-appearance order
  unique_kappas = []
  for t in param_tuples:
    if t.kappa not in unique_kappas:
      unique_kappas.append(t.kappa)

  base_stats_cache: dict[tuple[float, float, int], dict[str, Any]] = {}
  multibin_cache: dict[
      tuple[float, float, int, int, float],
      tuple[float, np.ndarray, np.ndarray],
  ] = {}
  sim_data_by_kappa: dict[float, tuple[np.ndarray, dict[int, float]]] = {}

  for kappa in unique_kappas:
    max_n_max = max(t.n_max for t in param_tuples if t.kappa == kappa)
    boost_map = precompute_boost_factors(alpha, kappa, max_n_max, k_bar)

    rng = np.random.default_rng(seed)
    p = rng.uniform(0.0, 1.0, size=(n_sim, max_n_max))
    log_e = np.log(kappa) + (kappa - 1.0) * np.log(p)
    log_m_all = np.cumsum(log_e, axis=1)

    sim_data_by_kappa[kappa] = (log_m_all, boost_map)

  records = []
  for t in param_tuples:
    log_m_all, boost_map = sim_data_by_kappa[t.kappa]
    base_key = (t.kappa, t.e_max, t.n_max)
    if base_key not in base_stats_cache:
      base_stats_cache[base_key] = compute_base_walk_stats(
          log_m_all=log_m_all,
          kappa=t.kappa,
          e_max=t.e_max,
          n_max=t.n_max,
          budget_probes=budget_probes,
          k_bar=k_bar,
      )
    base_stats = base_stats_cache[base_key]

    boost_factors = np.array([boost_map[m] for m in range(1, t.n_max + 1)])
    p_surv_table = base_stats["P_surv_table"]
    phi_analytical_ceiling = np.array(
        [alpha / max(p_surv_table[m - 1], 1e-12) for m in range(1, t.n_max + 1)]
    )

    if t.num_bins == 1:
      e_bin_max = base_stats["E_A"]
      bin_edges = np.array([0.0, t.e_max])
    else:
      mb_key = (t.kappa, t.e_max, t.n_max, t.num_bins, t.e_min)
      if mb_key not in multibin_cache:
        multibin_cache[mb_key] = compute_multibin_stats(
            kappa=t.kappa,
            e_max=t.e_max,
            n_max=t.n_max,
            num_bins=t.num_bins,
            e_min=t.e_min,
        )
      e_bin_max, bin_edges, _ = multibin_cache[mb_key]

    prehoc_baseline = alpha * e_bin_max + base_stats["baseline_K_PH"]
    calibrated_alpha_baseline = max(
        0.0, min(1.0, (alpha - base_stats["baseline_K_PH"]) / e_bin_max)
    )

    rec = {
        "kappa": t.kappa,
        "e_min": t.e_min,
        "e_max": t.e_max,
        "n_max": t.n_max,
        "num_bins": t.num_bins,
        "num_bins_explicit": t.num_bins_explicit,
        "bin_edges": bin_edges,
        "E_H": base_stats["E_H"],
        "P_H_table": base_stats["P_H_table"],
        "P_surv_table": base_stats["P_surv_table"],
        "P_H": base_stats["P_H"],
        "E_A_table": base_stats["E_A_table"],
        "E_A_formula_table": base_stats["E_A_formula_table"],
        "E_A": base_stats["E_A"],
        "E_bin_max": e_bin_max,
        "baseline_K_PH": base_stats["baseline_K_PH"],
        "prehoc_baseline": prehoc_baseline,
        "calibrated_alpha_baseline": calibrated_alpha_baseline,
        "boost_factors": boost_factors,
        "phi_analytical_ceiling": phi_analytical_ceiling,
    }
    records.append(rec)

  return records


def main(argv: Sequence[str]) -> None:
  """CLI entry point for computing DISCO theoretical FDR bound values."""
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  calculation_mode = _CALCULATION_MODE.value
  tuples_str = _TUPLES.value.strip()
  if tuples_str:
    calculation_mode = "tuples"

  e_min = _E_MIN.value
  alpha = _ALPHA.value
  budget_raw = _BUDGET.value
  chunk_size = max(1, _CHUNK_SIZE.value)
  budget_probes = max(1, budget_raw // chunk_size)
  k_bar = budget_probes
  n_sim = _N_SIM.value
  seed = _SEED.value
  mode = _MODE.value

  if calculation_mode == "tuples":
    if tuples_str:
      num_bins_fallback = int(_NUM_BINS.value.split(",")[0].strip())
      param_tuples = parse_tuples(
          tuples_str,
          default_e_min=e_min,
          default_num_bins=num_bins_fallback,
      )
    else:
      param_tuples = list(DEFAULT_TUPLES)
  else:
    e_max_list = sorted(
        list(set(float(x.strip()) for x in _E_MAX.value.split(",")))
    )
    n_max_list = sorted(
        list(set(int(x.strip()) for x in _N_MAX.value.split(",")))
    )
    kappa_list = sorted(
        list(set(float(x.strip()) for x in _KAPPA.value.split(",")))
    )
    num_bins_list = sorted(
        list(set(int(x.strip()) for x in _NUM_BINS.value.split(",")))
    )
    param_tuples = []
    for k in kappa_list:
      for em in e_max_list:
        for nm in n_max_list:
          for nb in num_bins_list:
            param_tuples.append(
                ParamTuple(
                    kappa=k,
                    e_max=em,
                    n_max=nm,
                    num_bins=nb,
                    e_min=e_min,
                    num_bins_explicit=(nb > 1),
                )
            )

  t0 = time.time()
  unique_kappas = set(t.kappa for t in param_tuples)
  print(
      f"[disco_bound] Simulating {n_sim:,} paths across {len(unique_kappas)}"
      f" kappas ({len(param_tuples)} configurations)...",
      flush=True,
  )

  records = compute_bounds_for_tuples(
      param_tuples=param_tuples,
      alpha=alpha,
      budget_raw=budget_raw,
      chunk_size=chunk_size,
      n_sim=n_sim,
      seed=seed,
  )

  elapsed = time.time() - t0
  print(
      f"[disco_bound] Completed in {elapsed:.2f}s ({len(records)}"
      " configurations).\n",
      flush=True,
  )

  report = generate_text_reports(
      records,
      alpha,
      budget_raw,
      budget_probes,
      chunk_size,
      k_bar,
      mode,
      e_min=e_min,
      is_tuple_mode=(calculation_mode == "tuples"),
  )
  print(report)

  if _OUTPUT_FILE.value:
    with open(_OUTPUT_FILE.value, "w") as f:
      f.write(report)
    print(f"\n[disco_bound] Saved report to {_OUTPUT_FILE.value}")


if __name__ == "__main__":
  app.run(main)
