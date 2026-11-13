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

"""Golden cases for DISCO flag definitions and `DiscoConfig.from_flags`.

Pins:

*   every flag `flags_lib.define_disco_flags` registers -- name, type,
    default, enum values, help text -- for the argument sets the entry points
    pass (`run_disco`, `eval_disco`);
*   the `DiscoConfig` that `from_flags` builds for real holders and for
    hand-made holders, covering the model chain (backbone fan-out, explicit
    roles, parser falling back to the agent), the keyword overrides and the
    YAML-only `overrides`.
"""

from collections.abc import Callable, Mapping
import dataclasses
import itertools
import types
from typing import Any

from absl import flags

from disco import flags_lib
from disco.core import config as config_lib

_BACKBONE_MODEL = "gemini-3.8-flash"
_PREFIXES = itertools.count()

# `define_disco_flags` arguments of each entry point.
_ENTRY_POINT_ARGS: dict[str, dict[str, Any]] = {
    "run_disco": {},
    "eval_disco": {
        "default_seed": "<SEED>",
        "default_budget": 30,
        "default_output_dir": "<OUTPUT_DIR>",
        "default_name": "disco_eval",
    },
}


def _define(**kwargs: Any) -> tuple[str, flags_lib.DiscoFlagHolders]:
  """Defines the DISCO flags under a fresh prefix; returns (prefix, holders)."""
  prefix = f"config_golden_{next(_PREFIXES)}_"
  return prefix, flags_lib.define_disco_flags(prefix=prefix, **kwargs)


def _flag_definitions(**kwargs: Any) -> dict[str, Any]:
  """Returns holder attribute -> definition of its flag."""
  prefix, holders = _define(**kwargs)
  out = {}
  for field in dataclasses.fields(holders):
    holder = getattr(holders, field.name)
    flag = flags.FLAGS[holder.name]
    definition = {
        "name": holder.name.removeprefix(prefix),
        "type": flag.flag_type(),
        "default": flag.default,
        "help": flag.help,
        "allow_override_cpp": flag.allow_override_cpp,
    }
    enum_values = getattr(flag.parser, "enum_values", None)
    if enum_values is not None:
      definition["enum_values"] = list(enum_values)
    out[field.name] = definition
  return out


def _holder(value: Any, present: bool = False) -> types.SimpleNamespace:
  return types.SimpleNamespace(value=value, present=present)


def _holders_with(**overrides: Any) -> types.SimpleNamespace:
  """Default-valued holders for every DISCO flag, with `overrides` present."""
  _, holders = _define()
  values = {
      f.name: _holder(getattr(holders, f.name).value)
      for f in dataclasses.fields(holders)
  }
  for name, value in overrides.items():
    if name not in values:
      raise KeyError(f"No DISCO flag holder {name!r}")
    values[name] = _holder(value, present=True)
  return types.SimpleNamespace(**values)


def _config(holders: Any, **kwargs: Any) -> dict[str, Any]:
  config = config_lib.DiscoConfig.from_flags(holders, **kwargs)
  return dataclasses.asdict(config)


def _from_flags_cases() -> dict[str, dict[str, Any]]:
  """Returns case name -> `dataclasses.asdict` of the built DiscoConfig."""
  return {
      "defined_defaults": _config(_define()[1]),
      "defined_eval_disco": _config(
          _define(**_ENTRY_POINT_ARGS["eval_disco"])[1]
      ),
      "keyword_overrides": _config(
          _define()[1],
          log_to_console=False,
          validate_stats_test=True,
          experiment_name="<EXPERIMENT>",
          output_dir="<OUT>",
          min_response_fraction=0.25,
      ),
      "backbone_only": _config(_holders_with(backbone_model=_BACKBONE_MODEL)),
      "backbone_with_explicit_roles": _config(
          _holders_with(backbone_model=_BACKBONE_MODEL),
          overrides={
              "agent_model": "<AGENT_MODEL>",
              "parser_model": "<PARSER_MODEL>",
              "rating_model": "<RATING_MODEL>",
              "summarizer_model": "<SUMMARIZER_MODEL>",
          },
      ),
      "backbone_with_explicit_agent": _config(
          _holders_with(backbone_model=_BACKBONE_MODEL),
          overrides={"agent_model": "<AGENT_MODEL>"},
      ),
      "explicit_roles_without_backbone": _config(
          _holders_with(),
          overrides={
              "agent_model": "<AGENT_MODEL>",
              "parser_model": "<PARSER_MODEL>",
              "rating_model": "<RATING_MODEL>",
              "summarizer_model": "",
          },
      ),
      "explicit_agent_without_backbone": _config(
          _holders_with(), overrides={"agent_model": "<AGENT_MODEL>"}
      ),
      "all_flags_set": _config(
          _holders_with(
              target_model="<TARGET_MODEL>",
              seed="<SEED>",
              budget=77,
              alpha=0.2,
              backbone_model=_BACKBONE_MODEL,
              name="<NAME>",
              output_dir="<OUTPUT_DIR>",
              e_max=30.0,
              chunk_size=6,
              alpha_calibration_mode="pre_hoc",
              n_max=5,
              num_active_bins=0,
          )
      ),
      "yaml_only_overrides": _config(
          _holders_with(),
          overrides={
              "kappa": 0.3,
              "method": "ebh_baseline",
              "e_min": 0.4,
              "enforce_exit_low": False,
              "use_boosting": False,
              "alpha_sweep_min": 0.03,
              "alpha_sweep_max": 0.9,
              "alpha_sweep_steps": 9,
              "n_sim": 123,
              "procedural_autorater": "count_words",
              "show_intra_hypothesis_failures": False,
              "show_all_failures": True,
              "allow_continuous_means_test": False,
              "query_generation_mode": "taxonomy",
              "taxonomy_depth": 2,
              "taxonomy_use_critic": False,
              "taxonomy_best_of_n": 4,
              "paired": True,
              "use_direction_guard": False,
              "p_value_mode": "discrete",
              "random_seed": 11,
              "baseline_enforce_exit_low": True,
              "baseline_exit_high": True,
              "enable_hypothesis_checker": False,
              "hypothesis_checker_max_retries": 4,
              "hypothesis_checker_rules": ["<RULE_1>", "<RULE_2>"],
              "isolated_query_gen_temperature": 0.3,
              "run_precision_check": False,
              "single_batch_num_queries": 40,
              "single_batch_alpha": 0.2,
              "single_batch_null_prob": 0.0,
              "max_concurrent_hypotheses": 3,
              "scrape_bandwidth": 2,
              "max_consecutive_skips": 0,
              "agent_thinking_level": "HIGH",
              "agent_thinking_budget": 256,
              "task_suite": "visual_symmetry",
          },
      ),
  }


def config_flags_snapshot() -> dict[str, Any]:
  """Returns the flag definitions per entry point and the from_flags cases."""
  definitions: Mapping[str, Any] = {
      name: _flag_definitions(**kwargs)
      for name, kwargs in _ENTRY_POINT_ARGS.items()
  }
  return {
      "flag_definitions": definitions,
      "from_flags": _from_flags_cases(),
  }


# Golden name -> case function, in the same shape as
# stats_golden_cases.ALL_CASES.
ALL_CASES: dict[str, Callable[[], Any]] = {
    "config_flags": config_flags_snapshot,
}
