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

"""Deterministic case functions whose outputs are pinned as golden files.

Each public `*_cases` function evaluates a fixed grid of inputs through the
statistics modules and returns a JSON-serializable dict. The golden test
compares these against checked-in JSON; `regenerate_goldens` rewrites them.
"""

from collections.abc import Callable
import dataclasses
import itertools
from typing import Any

import numpy as np

from disco.stats import boosting
from disco.stats import calibrators
from disco.stats import multiple_testing
from disco.stats.coarsened import calibration as calibration_lib
from disco.stats.coarsened import certificates as cert_lib
from disco.stats.coarsened import null_tables as null_tables_lib
from disco.stats.coarsened import status as status_lib

# Paper defaults for the DISCO configuration (see core/config.DiscoConfig).
_KAPPA = 0.1741
_E_MIN = 0.01
_E_MAX = 100.0
_N_MAX = 10
_K_MAX = 6
_N_SIM = 20_000
_SEED = 42

_MC_FIELDS = (
    "mc_m_free",
    "mc_survived_cum",
    "mc_first_exit_idx",
    "mc_exited_low",
)

_E_VALUES = {
    "h0": 50.0,
    "h1": 30.0,
    "h2": 12.0,
    "h3": 5.0,
    "h4": 1.5,
    "h5": 0.3,
    "h6": 25.0,
    "h7": 0.0,
}

# Synthetic DISCO run outcomes. h5 has more probes than n_max to exercise
# clamping; h2 exited high.
_DISCO_COUNTS = {"h0": 3, "h1": 2, "h2": 5, "h3": 10, "h4": 1, "h5": 12}
_DISCO_FINAL_E = {
    "h0": 4.0,
    "h1": 0.005,
    "h2": 150.0,
    "h3": 20.0,
    "h4": 0.008,
    "h5": 1.2,
}
_DISCO_STATUSES_TERNARY = {
    "h0": "A",
    "h1": "L",
    "h2": "H",
    "h3": "A",
    "h4": "L",
    "h5": "A",
}
_DISCO_STATUSES_BINNED = {
    "h0": "B_2",
    "h1": "L",
    "h2": "H",
    "h3": "B_3",
    "h4": "L",
    "h5": "B_1",
}


def _fmt(**kwargs: Any) -> str:
  """Builds a stable string key such as 'alpha=0.05,m=1'."""
  return ",".join(f"{k}={v}" for k, v in kwargs.items())


def _capture(fn: Callable[[], Any]) -> Any:
  """Returns fn() or a description of the exception it raised."""
  try:
    return fn()
  except (ValueError, RuntimeError) as e:
    return {"raised": type(e).__name__}


def e_value_core_cases() -> dict[str, Any]:
  """Evaluates the calibrator, e-BH and boosting functions on a fixed grid."""
  out: dict[str, Any] = {}

  p_grid = [0.0, 1e-20, 1e-6, 1e-3, 0.01, 0.05, 0.1, 0.3, 0.5, 0.9, 1.0]
  out["p_to_e"] = {
      _fmt(p=p, kappa=kappa): calibrators.p_to_e(p, kappa)
      for kappa, p in itertools.product((_KAPPA, 0.5), p_grid)
  }
  out["p_to_e_invalid"] = {
      _fmt(p=p, kappa=kappa): _capture(
          lambda p=p, kappa=kappa: calibrators.p_to_e(p, kappa)
      )
      for p, kappa in ((-0.1, 0.5), (1.1, 0.5), (0.5, 0.0), (0.5, 1.0))
  }

  out["get_threshold_from_alpha"] = {
      _fmt(alpha=a): calibrators.get_threshold_from_alpha(a)
      for a in (0.05, 0.1, 0.2)
  }

  ebh = {}
  for alpha in (0.05, 0.1, 0.2):
    for k_override in (None, 8, 20):
      rejected, threshold = multiple_testing.e_bh_rejection(
          _E_VALUES, alpha, k_total_override=k_override
      )
      ebh[_fmt(alpha=alpha, k_override=k_override)] = {
          "rejected": rejected,
          "threshold": threshold,
      }
  ebh["empty"] = list(multiple_testing.e_bh_rejection({}, 0.1))
  ebh["alpha_zero"] = list(multiple_testing.e_bh_rejection(_E_VALUES, 0.0))
  out["e_bh_rejection"] = ebh

  out["calibrated_evalue_survival"] = {
      _fmt(kappa=kappa, m=m): calibrators.calibrated_evalue_survival(
          np.array([0.0, 0.5, 1.0, 2.0, 10.0, 100.0, 1e4]), kappa, m
      )
      for kappa, m in itertools.product((_KAPPA, 0.5), (1, 3, 10))
  }
  out["calibrated_evalue_survival_scalar"] = (
      calibrators.calibrated_evalue_survival(20.0, _KAPPA, 4)
  )

  boost_grid = itertools.product(
      (0.05, 0.1), (0, 1, 2, 3, 5, 10), (1, 5, 20), (True, False)
  )
  out["compute_boost_factor"] = {
      _fmt(alpha=alpha, m=m, k=k, exact=exact): boosting.compute_boost_factor(
          alpha, _KAPPA, m, k, use_exact_t=exact
      )
      for alpha, m, k, exact in boost_grid
  }

  out["generate_k_grid"] = {
      _fmt(k_min=a, k_max=b, n=n): boosting.generate_k_grid(a, b, n)
      for a, b, n in (
          (1, 10, 5),
          (3, 50, 5),
          (5, 5, 5),
          (2, 100, 3),
          (1, 10, 1),
      )
  }

  x_grid = np.array([0.0, 0.5, 0.999, 1.0, 1.5, 2.5, 7.0, 100.0])
  out["staircase_t"] = {
      _fmt(k=k): boosting.staircase_t(x_grid, k) for k in (1, 5, 20)
  }
  out["staircase_t_scalar"] = boosting.staircase_t(3.3, 4)
  return out


def _tables_to_dict(
    tables: null_tables_lib.NullCoarsenedTables,
) -> dict[str, Any]:
  """Serializes NullCoarsenedTables, summarizing the large raw Monte Carlo arrays."""
  result = {}
  for field in dataclasses.fields(tables):
    value = getattr(tables, field.name)
    if field.name in _MC_FIELDS:
      if value is None:
        result[field.name] = None
      else:
        arr = np.asarray(value)
        result[field.name] = {
            "shape": list(arr.shape),
            "sum": float(np.sum(arr.astype(float))),
        }
    else:
      result[field.name] = value
  return result


def _strip_mc(
    tables: null_tables_lib.NullCoarsenedTables,
) -> null_tables_lib.NullCoarsenedTables:
  """Returns a copy with raw MC arrays removed (exercises fallback paths)."""
  return dataclasses.replace(
      tables,
      mc_m_free=None,
      mc_survived_cum=None,
      mc_first_exit_idx=None,
      mc_exited_low=None,
  )


def build_null_tables() -> dict[str, null_tables_lib.NullCoarsenedTables]:
  """Builds the small null tables used by `fdr_coarsened_cases`."""

  def build(
      num_bins: int, discrete_null: bool
  ) -> null_tables_lib.NullCoarsenedTables:
    return null_tables_lib.compute_null_coarsened_tables(
        kappa=_KAPPA,
        e_min=_E_MIN,
        e_max=_E_MAX,
        n_max=_N_MAX,
        k_max=_K_MAX,
        num_bins=num_bins,
        alpha=0.05,
        n_sim=_N_SIM,
        seed=_SEED,
        discrete_null=discrete_null,
    )

  return {
      "J1": build(num_bins=1, discrete_null=False),
      "J3": build(num_bins=3, discrete_null=False),
      "J1_discrete": build(num_bins=1, discrete_null=True),
  }


def _statuses_for(table_name: str) -> dict[str, str]:
  return (
      _DISCO_STATUSES_BINNED if table_name == "J3" else _DISCO_STATUSES_TERNARY
  )


def _coarsened_cases_for_table(
    name: str, tables: null_tables_lib.NullCoarsenedTables
) -> dict[str, Any]:
  """Runs every stats/coarsened entry point against one NullCoarsenedTables instance."""
  out: dict[str, Any] = {}
  statuses = _statuses_for(name)
  status_list = [statuses[h] for h in _DISCO_COUNTS]
  count_list = [_DISCO_COUNTS[h] for h in _DISCO_COUNTS]
  k_obs = len(status_list)
  no_mc = _strip_mc(tables)

  out["tables"] = _tables_to_dict(tables)

  out["prehoc_fdr_bounds"] = {
      _fmt(alpha_star=a): cert_lib.compute_prehoc_fdr_bounds(tables, _K_MAX, a)
      for a in (0.05, 0.1)
  }

  phi_statuses = ["A", "L", "H"]
  if tables.num_bins > 1:
    phi_statuses += [f"B_{j}" for j in range(1, tables.num_bins + 1)]
  else:
    phi_statuses += ["B_1"]
  m_values = [1, 2, 5, 10, 15]
  recompute = {}
  for label, t in (("mc", tables), ("no_mc", no_mc)):
    for alpha_bh in (0.05, 0.2):
      for status in phi_statuses:
        recompute[_fmt(src=label, alpha=alpha_bh, status=status)] = (
            cert_lib.recompute_phi_at_alpha(
                alpha_bh, _KAPPA, m_values, k_obs, t, status=status
            )
        )
  recompute["alpha_zero"] = cert_lib.recompute_phi_at_alpha(
      0.0, _KAPPA, m_values, k_obs, tables
  )
  out["recompute_phi_at_alpha"] = recompute

  out["phi_bar"] = {
      _fmt(alpha=a): cert_lib.compute_phi_bar(
          a, _KAPPA, status_list, count_list, k_obs, tables
      )
      for a in (0.05, 0.1, 0.5)
  }
  out["continuous_phi_bar"] = {
      _fmt(alpha=a): cert_lib.compute_continuous_phi_bar(
          a, _KAPPA, statuses, _DISCO_COUNTS, _DISCO_FINAL_E, tables, _N_MAX
      )
      for a in (0.05, 0.1, 0.5)
  }
  out["prehoc_phi_max"] = {
      _fmt(alpha=a): cert_lib.compute_prehoc_phi_max(a, _KAPPA, _K_MAX, tables)
      for a in (0.05, 0.2)
  }

  delta_post = float(
      sum(
          tables.P_H_table[min(max(1, c), tables.n_max) - 1] for c in count_list
      )
  )
  optimal = {}
  for alpha_star in (0.05, 0.1, 0.2):
    for mode in ("good_event", "post_hoc"):
      optimal[_fmt(alpha_star=alpha_star, mode=mode)] = list(
          calibration_lib.find_optimal_alpha_bh(
              alpha_star,
              _KAPPA,
              status_list,
              count_list,
              k_obs,
              tables,
              delta_post=delta_post,
              mode=mode,
          )
      )
    optimal[_fmt(alpha_star=alpha_star, mode="continuous")] = list(
        calibration_lib.find_optimal_alpha_bh(
            alpha_star,
            _KAPPA,
            status_list,
            count_list,
            k_obs,
            tables,
            delta_post=delta_post,
            mode="post_hoc",
            final_e_values=_DISCO_FINAL_E,
            terminal_statuses_dict=statuses,
            probe_counts_dict=_DISCO_COUNTS,
            n_max=_N_MAX,
        )
    )
    optimal[_fmt(alpha_star=alpha_star, mode="pre_hoc")] = list(
        calibration_lib.find_optimal_prehoc_alpha_bh(
            alpha_star, _KAPPA, _K_MAX, tables
        )
    )
  out["find_optimal_alpha_bh"] = optimal

  out["posthoc_certificate"] = {
      "list": cert_lib.compute_disco_posthoc_certificate(
          status_list, count_list, tables, 0.1
      ),
      "dict": cert_lib.compute_disco_posthoc_certificate(
          statuses, _DISCO_COUNTS, tables, 0.1
      ),
      "continuous": cert_lib.compute_disco_posthoc_certificate(
          statuses,
          _DISCO_COUNTS,
          tables,
          0.1,
          final_e_values=_DISCO_FINAL_E,
          kappa=_KAPPA,
          n_max_override=_N_MAX,
      ),
      "empty": cert_lib.compute_disco_posthoc_certificate([], [], tables, 0.1),
  }

  sweep = {}
  for mode in ("post_hoc", "good_event", "pre_hoc"):
    for use_boosting in (True, False):
      sweep[_fmt(mode=mode, boosting=use_boosting)] = (
          calibration_lib.compute_disco_alpha_sweep(
              _DISCO_FINAL_E,
              statuses,
              _DISCO_COUNTS,
              tables,
              [0.05, 0.1, 0.2],
              alpha_calibration_mode=mode,
              use_boosting=use_boosting,
              kappa=_KAPPA,
          )
      )
  # List inputs are positional in `_DISCO_FINAL_E` order; they must match the
  # dict case's boost factors (each hypothesis's own n_g).
  sweep["list_inputs"] = calibration_lib.compute_disco_alpha_sweep(
      _DISCO_FINAL_E,
      status_list,
      count_list,
      tables,
      [0.1],
      alpha_calibration_mode="post_hoc",
      use_boosting=True,
      kappa=_KAPPA,
  )
  # Unbinned mode uses the exact e-values, like the headline certificate
  # (`posthoc_certificate.continuous`) and `find_optimal_alpha_bh` above.
  for mode in ("post_hoc", "pre_hoc"):
    for use_boosting in (True, False):
      sweep[_fmt(mode=mode, boosting=use_boosting, continuous=True)] = (
          calibration_lib.compute_disco_alpha_sweep(
              _DISCO_FINAL_E,
              statuses,
              _DISCO_COUNTS,
              tables,
              [0.05, 0.1, 0.2],
              alpha_calibration_mode=mode,
              use_boosting=use_boosting,
              kappa=_KAPPA,
              continuous=True,
              n_max=_N_MAX,
          )
      )
  out["alpha_sweep"] = sweep

  out["expected_null_e"] = {
      _fmt(status=s, n=n): null_tables_lib.get_expected_null_e(s, n, tables)
      for s, n in itertools.product(phi_statuses, (1, 4, 15))
  }
  out["expected_null_e_exact"] = null_tables_lib.get_expected_null_e(
      "A", 3, tables, exact_e=7.5
  )

  bin_pairs = list(itertools.product(range(tables.num_bins + 2), repeat=2))
  out["lookup_transition_score"] = {
      _fmt(prev=p, curr=c): null_tables_lib.lookup_transition_score(
          tables, p, c
      )
      for p, c in bin_pairs
  }
  no_centers = dataclasses.replace(tables, bin_log_centers=None)
  out["lookup_transition_score_no_centers"] = {
      _fmt(prev=p, curr=c): null_tables_lib.lookup_transition_score(
          no_centers, p, c
      )
      for p, c in bin_pairs
  }
  return out


def fdr_coarsened_cases() -> dict[str, Any]:
  """Evaluates the stats/coarsened functions on small, seeded null tables."""
  out: dict[str, Any] = {}

  status_inputs = [
      "L",
      "h",
      "A",
      "active",
      "survived",
      "exit_low",
      "B_1",
      "bin_2",
      "B3",
      "BIN4",
      "0",
      "2",
      "",
      "B_0",
      "-1",
      "garbage",
      0,
      3,
      -1,
  ]
  out["parse_status"] = {
      repr(s): _capture(lambda s=s: status_lib.parse_status(s))
      for s in status_inputs
  }

  edges = status_lib.compute_strip_bins(_E_MIN, _E_MAX, 3)
  out["compute_strip_bins"] = edges
  out["map_to_bin_idx"] = status_lib.map_to_bin_idx(
      np.array([0.001, 0.01, 0.05, 0.2, 1.0, 5.0, 99.9, 100.0, 1000.0]),
      _E_MIN,
      _E_MAX,
      edges,
      3,
  )
  out["map_to_bin_idx_scalar"] = status_lib.map_to_bin_idx(
      3.0, _E_MIN, _E_MAX, edges, 3
  )

  for name, tables in build_null_tables().items():
    out[name] = _coarsened_cases_for_table(name, tables)
  return out


ALL_CASES: dict[str, Callable[[], dict[str, Any]]] = {
    "e_value_core": e_value_core_cases,
    "fdr_coarsened": fdr_coarsened_cases,
}
