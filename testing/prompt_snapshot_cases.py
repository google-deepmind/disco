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

"""Snapshot cases for prompt builders and the task registry.

Each prompt module gets one golden file mapping case name -> rendered prompt.
Cases are generated from the builders' signatures:

*   `<fn>`: required arguments filled from `_SAMPLES`, everything else at its
    default.
*   `<fn>[<param>=<label>]`: the default case with one optional argument
    changed. Bool arguments are flipped; other arguments take each labeled
    sample value from `_SAMPLES`. Optional arguments without a sample keep
    their default.

`random` is reseeded before every call so builders that subsample (e.g.
excluded queries) are deterministic.
"""

from collections.abc import Callable, Mapping
import dataclasses
import inspect
import random
import types
from typing import Any

from disco import behavior_discovery_prompts
from disco import verification_prompts
from disco.prompting import parser
from disco.prompting import querygen
from disco.prompting import taxonomy
from disco.stats import plan
from disco.tasks import registry


@dataclasses.dataclass(frozen=True)
class _Choices:
  """Several labeled sample values for one parameter."""

  values: Mapping[str, Any]


_PLANS = _Choices({
    "binomial_larger": plan.PreRegisteredTestPlan(
        test_type="Binomial test",
        target_metric="<METRIC>",
        null_baseline=0.2,
        alternative="larger",
    ),
    "paired_mcnemar": plan.PreRegisteredTestPlan(
        test_type="McNemar's Test",
        target_metric="<METRIC>",
        null_baseline=0.5,
        alternative="two-sided",
        is_paired=True,
    ),
    "paired_binomial_null_0.3": plan.PreRegisteredTestPlan(
        test_type="Binomial test",
        target_metric="<METRIC>",
        null_baseline=0.3,
        alternative="smaller",
        is_paired=True,
    ),
    "paired_continuous": plan.PreRegisteredTestPlan(
        test_type="Paired continuous test",
        target_metric="<METRIC>",
        null_baseline=0.0,
        alternative="larger",
        is_paired=True,
    ),
})

# Sample values by parameter name. Required parameters must appear here.
_SAMPLES: dict[str, Any] = {
    "active_hypotheses": {"H1": "<HYPOTHESIS_1>", "H2": "<HYPOTHESIS_2>"},
    "autorater_name": "<AUTORATER>",
    "best_of_n": 5,
    "chunk_size": 10,
    "e_max": 20.0,
    "e_min": 0.05,
    "error_msg": "<ERROR_MESSAGE>",
    "excluded_queries": ["<EXCLUDED_QUERY_1>", "<EXCLUDED_QUERY_2>"],
    "failure_reflections": ["<REFLECTION_1>", "<REFLECTION_2>"],
    "failure_summary": "<FAILURE_SUMMARY>",
    "failures": [{
        "judgment": "EXHIBITS_BEHAVIOR",
        "prompt": "<PROBE_PROMPT>",
        "reasoning": "<JUDGE_REASONING>",
        "response": "<MODEL_RESPONSE>",
    }],
    "h_text": "<HYPOTHESIS>",
    "history_summary": "<HISTORY_SUMMARY>",
    "hypothesis": "<HYPOTHESIS>",
    "hypothesis_text": "<HYPOTHESIS>",
    "disco_history_summary": "<DISCO_HISTORY_SUMMARY>",
    "num_queries": 12,
    "original_prompt": "<ORIGINAL_PROMPT>",
    "plan": _PLANS,
    "prior_e_values": [
        ("<SUB_HYPOTHESIS_1>", 1.5),
        ("<SUB_2>", 0.25),
    ],
    "procedural_autorater": "<AUTORATER>",
    "q_count": 12,
    "queries": ["<QUERY_1>", "<QUERY_2>"],
    "remaining_queries": 40,
    "seed_instruction": "<SEED_INSTRUCTION>",
    "disco_seed": "<DISCO_SEED>",
    "target_model": "<TARGET_MODEL>",
    "taxonomy_depth": 2,
    "taxonomy_nodes": ["<NODE_1>", "<NODE_2>"],
    "threshold": 0.05,
}

# Golden name -> modules the builders of that golden are looked up in.
PROMPT_MODULES: dict[str, tuple[types.ModuleType, ...]] = {
    "behavior_discovery_prompts": (behavior_discovery_prompts,),
    "prompts": (parser, querygen, taxonomy),
    "verification_prompts": (verification_prompts,),
}

# Builders snapshotted in each module's golden.
PROMPT_GROUPS: dict[str, frozenset[str]] = {
    "behavior_discovery_prompts": frozenset({
        "get_default_preregistration_retry_prompt",
        "get_default_retry_prompt",
        "get_fresh_agent_intro_prompt",
        "get_disco_fresh_agent_intro_prompt",
        "get_disco_select_or_propose_prompt",
        "get_precision_round_context_prompt",
        "get_procedural_autorater_addendum",
        "get_procedural_scrape_and_measure_prompt",
        "get_recent_failures_reflection_prompt",
        "get_scrape_and_judge_execution_prompt",
        "get_stats_significance_prompt",
        "get_test_preregistration_prompt",
    }),
    "prompts": frozenset({
        "get_isolated_query_generation_prompt",
        "get_parser_system_instruction",
        "get_taxonomy_critic_prompt",
        "get_taxonomy_generation_prompt",
        "get_taxonomy_meta_prompt",
    }),
    "verification_prompts": frozenset({
        "get_fresh_agent_intro_prompt",
        "get_precision_round_context_prompt",
    }),
}


def _labeled_samples(name: str) -> dict[str, Any]:
  sample = _SAMPLES[name]
  if isinstance(sample, _Choices):
    return dict(sample.values)
  return {"sample": sample}


def _call(fn: Callable[..., str], kwargs: Mapping[str, Any]) -> str:
  random.seed(0)
  return fn(**kwargs)


def _builder_cases(fn: Callable[..., str]) -> dict[str, str]:
  """Renders the default case and single-argument variants of `fn`."""
  params = inspect.signature(fn).parameters.values()
  required = [p for p in params if p.default is inspect.Parameter.empty]
  optional = [p for p in params if p.default is not inspect.Parameter.empty]

  base = {}
  for p in required:
    samples = _labeled_samples(p.name)
    base[p.name] = next(iter(samples.values()))

  cases = {fn.__name__: _call(fn, base)}
  for p in optional:
    if isinstance(p.default, bool):
      variants = {str(not p.default): not p.default}
    elif p.name in _SAMPLES:
      variants = _labeled_samples(p.name)
    else:
      continue
    for label, value in variants.items():
      cases[f"{fn.__name__}[{p.name}={label}]"] = _call(
          fn, {**base, p.name: value}
      )
  return cases


def render_prompt_module(
    modules: tuple[types.ModuleType, ...], builders: frozenset[str]
) -> dict[str, str]:
  """Returns case name -> rendered prompt for each listed builder of `modules`.

  Args:
    modules: Modules to look the builders up on. Each builder must be a function
      in exactly one of them.
    builders: Names of the builders to render (see `PROMPT_GROUPS`).

  Raises:
    ValueError: If a listed builder is missing from `modules`, or is in more
      than one of them.
  """
  found: dict[str, list[Callable[..., str]]] = {name: [] for name in builders}
  for module in modules:
    for name in builders:
      fn = vars(module).get(name)
      if inspect.isfunction(fn):
        found[name].append(fn)
  module_names = [module.__name__ for module in modules]
  missing = sorted(name for name, fns in found.items() if not fns)
  if missing:
    raise ValueError(f"{module_names} lack builders: {missing}")
  ambiguous = sorted(name for name, fns in found.items() if len(fns) > 1)
  if ambiguous:
    raise ValueError(f"several of {module_names} define builders: {ambiguous}")
  cases = {}
  for _, (fn,) in sorted(found.items()):
    cases.update(_builder_cases(fn))
  return cases


def _task_snapshot(task: Any) -> dict[str, Any]:
  """Returns a task's legacy public fields plus its key.

  Spelled out rather than `dataclasses.asdict` so the snapshot does not depend
  on how the task dataclass stores those fields.

  Args:
    task: A registered task.
  """
  return {
      "name": task.name,
      "description": task.description,
      "split": task.split,
      "suite": task.suite,
      "metadata": task.metadata,
      "key": task.key,
  }


def task_registry_snapshot() -> dict[str, Any]:
  """Returns the registered tasks and hypotheses, plus lookup results.

  Lookups cover exact keys, aliases, splits and the substring fallback, so a
  registry rewrite shows exactly which lookups change.
  """
  tasks = {s: registry.get_suite(s) for s in registry.list_suites()}
  hypotheses = {
      s: registry.get_hypotheses(suite=s)
      for s in (
          "recipes",
          "hedging",
          "visual_symmetry",
          "t2i_cat_bias",
          "t2i_handedness_bias",
      )
  }
  lookups = {}
  for suite, suite_tasks in tasks.items():
    for split in ("iteration", "evaluation"):
      lookups[f"get_tasks({suite},{split})"] = [
          t.key for t in registry.get_tasks(suite=suite, split=split)
      ]
    for task in suite_tasks:
      for alias in task.metadata.get("aliases", []):
        lookups[f"get_task({alias})"] = registry.get_task(alias).key
  return {
      "hypotheses": {
          s: [dataclasses.asdict(h) | {"key": h.key} for h in hs]
          for s, hs in hypotheses.items()
      },
      "lookups": lookups,
      "tasks": {s: [_task_snapshot(t) for t in ts] for s, ts in tasks.items()},
  }


# Golden name -> case function, in the same shape as
# stats_golden_cases.ALL_CASES.
ALL_CASES: dict[str, Callable[[], Any]] = {
    f"prompts_{name}": lambda m=module, b=PROMPT_GROUPS[name]: (
        render_prompt_module(m, b)
    )
    for name, module in PROMPT_MODULES.items()
}
ALL_CASES["task_registry"] = task_registry_snapshot
