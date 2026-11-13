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

"""Characterization (golden) cases for the DISCO bandit simulators.

Shared between ``generate_bandit_goldens`` (writes testdata) and
``bandit_golden_test`` (compares against testdata). Two kinds of cases:

1. ``RESULT_CASES``: a ``BanditConfig`` + runner, executed in-process via the
   library runners in ``bandit_disco``. The full ``BanditResult`` is captured.
2. ``E2E_CASES``: a flag dict for ``run_bandit_disco.main``. The written JSON
   result file, the ``*_runs.json`` side file, and whitespace-normalized
   stdout are captured.

The point of these goldens is to pin *current* behavior bit-for-bit (floats to
~1e-9 relative) so that refactors can be verified as behavior-preserving. They
intentionally cover every branch of the runners rather than asserting anything
about statistical correctness.
"""

from collections.abc import Mapping, Sequence
import contextlib
import dataclasses
import io
import json
import math
import os
from typing import Any

from absl import flags
from absl.testing import flagsaver
import numpy as np

from disco.bandit import bandit_disco
from disco.bandit import bandit_env
from disco.bandit import run_bandit_disco

FLAGS = flags.FLAGS

RESULT_GOLDENS_FILENAME = "bandit_result_goldens.json"
E2E_GOLDENS_FILENAME = "e2e_goldens.json"

# Placeholder substituted for temp-dir dependent paths before comparison.
_PATH_PLACEHOLDER = "<output_path>"


# =============================================================================
# Library-level cases (BanditResult)
# =============================================================================


@dataclasses.dataclass(frozen=True)
class ResultCase:
  """Specification for a library-level bandit golden test case."""

  runner: str  # "disco" | "offline_ebh" | "type1"
  config: dict[str, Any]


_BERN = dict(
    env_type="bernoulli",
    budget=120,
    chunk_size=10,
    n_max=5,
    n_sim=1000,
    pi_1=0.3,
    seed=42,
)
_CTX = dict(
    env_type="contextual",
    budget=120,
    chunk_size=10,
    n_max=5,
    n_sim=1000,
    dimension=3,
    num_vulnerable_clusters=2,
    seed=42,
)


def _c(base: dict[str, Any], **overrides: Any) -> dict[str, Any]:
  return {**base, **overrides}


RESULT_CASES: dict[str, ResultCase] = {
    # ---- DISCO, Bernoulli ----------------------------------------------------
    "bern_disco_default": ResultCase("disco", _c(_BERN)),
    "bern_disco_seed7": ResultCase("disco", _c(_BERN, seed=7)),
    "bern_disco_j0": ResultCase("disco", _c(_BERN, num_active_bins=0)),
    "bern_disco_j3": ResultCase("disco", _c(_BERN, num_active_bins=3)),
    "bern_disco_nominal": ResultCase(
        "disco", _c(_BERN, alpha_calibration_mode="nominal")
    ),
    "bern_disco_prehoc": ResultCase(
        "disco", _c(_BERN, alpha_calibration_mode="pre_hoc")
    ),
    "bern_disco_goodevent": ResultCase(
        "disco", _c(_BERN, alpha_calibration_mode="good_event")
    ),
    "bern_disco_noboost_posthoc": ResultCase(
        "disco", _c(_BERN, use_boosting=False)
    ),
    "bern_disco_noboost_prehoc": ResultCase(
        "disco",
        _c(_BERN, use_boosting=False, alpha_calibration_mode="pre_hoc"),
    ),
    "bern_disco_noboost_goodevent": ResultCase(
        "disco",
        _c(_BERN, use_boosting=False, alpha_calibration_mode="good_event"),
    ),
    "bern_disco_noboost_nominal": ResultCase(
        "disco",
        _c(_BERN, use_boosting=False, alpha_calibration_mode="nominal"),
    ),
    "bern_disco_noboost_j0": ResultCase(
        "disco", _c(_BERN, use_boosting=False, num_active_bins=0)
    ),
    "bern_disco_noboost_j3": ResultCase(
        "disco", _c(_BERN, use_boosting=False, num_active_bins=3)
    ),
    "bern_disco_discrete": ResultCase(
        "disco", _c(_BERN, p_value_mode="discrete")
    ),
    "bern_disco_discrete_matched": ResultCase(
        "disco", _c(_BERN, p_value_mode="discrete_matched")
    ),
    "bern_disco_no_exit_low": ResultCase(
        "disco", _c(_BERN, enforce_exit_low=False, e_min=0.4)
    ),
    "bern_disco_no_exit_low_j0": ResultCase(
        "disco", _c(_BERN, enforce_exit_low=False, e_min=0.4, num_active_bins=0)
    ),
    "bern_disco_e_min_high": ResultCase("disco", _c(_BERN, e_min=0.4)),
    "bern_disco_kbar": ResultCase("disco", _c(_BERN, k_bar=5)),
    "bern_disco_all_null": ResultCase("disco", _c(_BERN, pi_1=0.0)),
    "bern_disco_high_alt": ResultCase(
        "disco", _c(_BERN, alternative_rate=0.8, alpha=0.1)
    ),
    # ---- DISCO, Contextual ---------------------------------------------------
    "ctx_disco_ucb": ResultCase("disco", _c(_CTX)),
    "ctx_disco_j0": ResultCase("disco", _c(_CTX, num_active_bins=0)),
    "ctx_disco_j3": ResultCase("disco", _c(_CTX, num_active_bins=3)),
    "ctx_disco_j0_no_exit_low": ResultCase(
        "disco", _c(_CTX, num_active_bins=0, enforce_exit_low=False, e_min=0.4)
    ),
    "ctx_disco_clusters1": ResultCase(
        "disco", _c(_CTX, num_vulnerable_clusters=1)
    ),
    "ctx_disco_clusters0": ResultCase(
        "disco", _c(_CTX, num_vulnerable_clusters=0)
    ),
    "ctx_disco_dim5_prehoc": ResultCase(
        "disco", _c(_CTX, dimension=5, alpha_calibration_mode="pre_hoc")
    ),
    "ctx_disco_goodevent_j3": ResultCase(
        "disco",
        _c(_CTX, num_active_bins=3, alpha_calibration_mode="good_event"),
    ),
    # ---- Offline e-BH --------------------------------------------------------
    "bern_offline_default": ResultCase("offline_ebh", _c(_BERN)),
    "bern_offline_no_exit_low": ResultCase(
        "offline_ebh", _c(_BERN, enforce_exit_low=False)
    ),
    "bern_offline_kbar": ResultCase("offline_ebh", _c(_BERN, k_bar=5)),
    "bern_offline_discrete": ResultCase(
        "offline_ebh", _c(_BERN, p_value_mode="discrete")
    ),
    "ctx_offline_ucb_loge": ResultCase("offline_ebh", _c(_CTX)),
    "ctx_offline_no_exit_low": ResultCase(
        "offline_ebh", _c(_CTX, enforce_exit_low=False)
    ),
    # ---- Type-I --------------------------------------------------------------
    "bern_type1_default": ResultCase("type1", _c(_BERN)),
    "bern_type1_high_alt": ResultCase(
        "type1", _c(_BERN, alternative_rate=0.8, alpha=0.1, pi_1=0.5)
    ),
    "bern_type1_no_exit_low": ResultCase(
        "type1", _c(_BERN, enforce_exit_low=False)
    ),
    "ctx_type1_default": ResultCase("type1", _c(_CTX)),
}


def build_env(
    config: bandit_disco.BanditConfig, seed: int
) -> bandit_env.BanditEnv:
  """Builds the environment exactly as ``run_bandit_disco._build_env`` does."""
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
  return bandit_env.BernoulliBanditEnv(
      pi_1=config.pi_1,
      baseline_rate_logit=config.baseline_rate_logit,
      alternative_rate=config.alternative_rate,
      seed=seed,
  )


def run_result_case(case: ResultCase) -> dict[str, Any]:
  """Runs one library-level case and returns the serialized BanditResult."""
  config = bandit_disco.BanditConfig(**case.config)
  env = build_env(config, config.seed)
  num_rounds = max(1, config.budget // config.chunk_size)
  ttp, cluster_ttp = env.pre_generate_pool(num_rounds)
  if case.runner == "disco":
    result = bandit_disco.run_disco_bandit(
        config,
        env,
        null_tables=None,
        total_true_positives=ttp,
        cluster_true_positives=cluster_ttp,
    )
  elif case.runner == "offline_ebh":
    result = bandit_disco.run_offline_ebh_bandit(
        config,
        env,
        total_true_positives=ttp,
        cluster_true_positives=cluster_ttp,
    )
  elif case.runner == "type1":
    result = bandit_disco.run_type1_bandit(
        config,
        env,
        total_true_positives=ttp,
        cluster_true_positives=cluster_ttp,
    )
  else:
    raise ValueError(f"Unknown runner: {case.runner}")
  return serialize_result(result)


def _json_default(obj: Any) -> Any:
  if isinstance(obj, (set, frozenset)):
    return sorted(obj)
  if isinstance(obj, np.ndarray):
    return obj.tolist()
  if isinstance(obj, np.integer):
    return int(obj)
  if isinstance(obj, np.floating):
    return float(obj)
  if isinstance(obj, np.bool_):
    return bool(obj)
  raise TypeError(f"Unserializable type {type(obj)}")


def _encode_nonfinite(obj: Any) -> Any:
  """Replaces inf/nan floats with sentinel strings (strict JSON has none)."""
  if isinstance(obj, Mapping):
    return {k: _encode_nonfinite(v) for k, v in obj.items()}
  if isinstance(obj, (list, tuple)):
    return [_encode_nonfinite(v) for v in obj]
  if isinstance(obj, (float, np.floating)) and not math.isfinite(obj):
    if math.isnan(obj):
      return "__nan__"
    return "__inf__" if obj > 0 else "__-inf__"
  return obj


def to_jsonable(obj: Any) -> Any:
  """Round-trips through JSON so goldens and live values share one shape."""
  return json.loads(
      json.dumps(
          _encode_nonfinite(obj),
          default=_json_default,
          sort_keys=True,
          allow_nan=False,
      )
  )


def serialize_result(result: bandit_disco.BanditResult) -> dict[str, Any]:
  d = dataclasses.asdict(result)
  d["rejected_hypotheses"] = sorted(d["rejected_hypotheses"])
  return to_jsonable(d)


# =============================================================================
# End-to-end cases (run_bandit_disco.main -> JSON files + stdout)
# =============================================================================

_E2E_BERN = dict(
    env_type="bernoulli",
    budget=100,
    chunk_size=10,
    n_max=5,
    n_sim=1000,
    pi_1=0.3,
    num_mc_runs=2,
    seed=42,
)
_E2E_CTX = dict(
    env_type="contextual",
    budget=100,
    chunk_size=10,
    n_max=5,
    n_sim=1000,
    dimension=3,
    num_vulnerable_clusters=2,
    num_mc_runs=2,
    seed=42,
)

E2E_CASES: dict[str, dict[str, Any]] = {
    "e2e_all_bernoulli_j3": _c(
        _E2E_BERN,
        runner_type="all",
        num_active_bins=3,
        enable_alpha_sweep=True,
        alpha_sweep_steps=4,
    ),
    "e2e_sxs_contextual_continuous": _c(
        _E2E_CTX,
        runner_type="sxs",
        num_active_bins=0,
        enable_alpha_sweep=True,
        alpha_sweep_steps=3,
    ),
    "e2e_disco_bernoulli_prehoc": _c(
        _E2E_BERN,
        runner_type="disco",
        alpha_calibration_mode="pre_hoc",
        num_mc_runs=3,
    ),
    "e2e_disco_noboost_goodevent_discrete_matched": _c(
        _E2E_BERN,
        runner_type="disco",
        alpha_calibration_mode="good_event",
        use_boosting=False,
        p_value_mode="discrete_matched",
    ),
    "e2e_disco_j25_bin_breakdown": _c(
        _E2E_BERN,
        runner_type="disco",
        num_active_bins=25,
        n_sim=500,
    ),
    "e2e_offline_ebh_contextual": _c(_E2E_CTX, runner_type="offline_ebh"),
    "e2e_type1_bernoulli": _c(_E2E_BERN, runner_type="type1"),
    "e2e_disco_no_per_run": _c(
        _E2E_BERN,
        runner_type="disco",
        save_per_run_data=False,
        num_mc_runs=1,
    ),
}

# Stdout lines that are timing- or path-dependent and therefore excluded.
_VOLATILE_STDOUT_MARKERS = (
    "[Simulation Progress]",
    "Results written to:",
    "Per-run details written to:",
    "[disco]",  # stats/coarsened timing chatter, e.g. "complete in 0.0s"
)


def normalize_stdout(text: str) -> list[str]:
  """Collapses whitespace and drops timing/path dependent lines."""
  out = []
  for raw in text.replace("\r", "\n").split("\n"):
    if any(m in raw for m in _VOLATILE_STDOUT_MARKERS):
      continue
    line = " ".join(raw.split())
    if line:
      out.append(line)
  return out


def run_e2e_case(
    flag_overrides: Mapping[str, Any], tmp_dir: str
) -> dict[str, Any]:
  """Runs ``run_bandit_disco.main`` with the given flags; captures outputs."""
  output_path = os.path.join(tmp_dir, "result.json")
  runs_path = os.path.join(tmp_dir, "result_runs.json")
  overrides = dict(flag_overrides)
  overrides["output_path"] = output_path
  overrides["output_dir"] = ""
  buf = io.StringIO()
  with flagsaver.flagsaver(**overrides):
    with contextlib.redirect_stdout(buf):
      run_bandit_disco.main(["run_bandit_disco"])
  with open(output_path, "r") as f:
    result = json.load(f)
  runs = None
  if os.path.exists(runs_path):
    with open(runs_path, "r") as f:
      runs = json.load(f)
  for d in (result, runs):
    if d is not None and "params" in d:
      d["params"]["output_path"] = _PATH_PLACEHOLDER
  return to_jsonable({
      "result": result,
      "runs": runs,
      "stdout": normalize_stdout(buf.getvalue()),
  })


# =============================================================================
# Comparison
# =============================================================================


def diff_structures(
    expected: Any,
    actual: Any,
    path: str = "",
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-12,
) -> list[str]:
  """Returns human-readable mismatches between two JSON-like structures."""
  diffs: list[str] = []
  if isinstance(expected, Mapping) and isinstance(actual, Mapping):
    for k in sorted(set(expected) | set(actual)):
      p = f"{path}.{k}" if path else str(k)
      if k not in expected:
        diffs.append(f"{p}: unexpected key (actual={actual[k]!r})")
      elif k not in actual:
        diffs.append(f"{p}: missing key (expected={expected[k]!r})")
      else:
        diffs.extend(
            diff_structures(expected[k], actual[k], p, rel_tol, abs_tol)
        )
    return diffs
  if (
      isinstance(expected, Sequence)
      and not isinstance(expected, str)
      and isinstance(actual, Sequence)
      and not isinstance(actual, str)
  ):
    if len(expected) != len(actual):
      diffs.append(
          f"{path}: length {len(expected)} != {len(actual)}"
          f" (expected={expected!r}, actual={actual!r})"
      )
      return diffs
    for i, (e, a) in enumerate(zip(expected, actual)):
      diffs.extend(diff_structures(e, a, f"{path}[{i}]", rel_tol, abs_tol))
    return diffs
  if isinstance(expected, bool) or isinstance(actual, bool):
    if expected != actual:
      diffs.append(f"{path}: {expected!r} != {actual!r}")
    return diffs
  if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
    e, a = float(expected), float(actual)
    if math.isnan(e) and math.isnan(a):
      return diffs
    if not math.isclose(e, a, rel_tol=rel_tol, abs_tol=abs_tol):
      diffs.append(f"{path}: {expected!r} != {actual!r}")
    return diffs
  if expected != actual:
    diffs.append(f"{path}: {expected!r} != {actual!r}")
  return diffs
