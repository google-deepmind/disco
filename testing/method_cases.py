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

"""Golden cases pinning the discovery-baseline methods end to end.

Pins the five discovery-baseline methods (direct asking, simple probing, red
team, looped investigator, looped multi-sample) on a text adapter (hedging) and
an image adapter (visual symmetry):

  * every LLM call each method makes, in order (model, contents, config),
  * every checkpoint it writes, and a resume from a partial checkpoint,
  * its log lines and the `GenericBaselineResult` it returns,

and, through `eval_discovery_baselines._run_single_method`, the probe plan
(rounds, probes per round, samples per probe) each method gets from a budget.

Target responses come from `adapter_cases.FakeModels`; investigator responses
are scripted from the request text, so nothing depends on call scheduling.
"""

import asyncio
from collections.abc import Callable
import hashlib
import json
import os
import shutil
import tempfile
import types as pytypes
from typing import Any
from unittest import mock

from disco.baselines import common
from disco.baselines import eval_discovery_baselines
from disco.baselines.adapters import registry as adapters_registry
from disco.baselines.methods import base as methods_base
from disco.baselines.methods import direct_asking
from disco.baselines.methods import looped
from disco.baselines.methods import red_team
from disco.baselines.methods import simple_probing
from disco.tasks import hedging_tasks
from disco.tasks.visual_symmetry import hypotheses as visual_symmetry_hypotheses
from disco.tasks.visual_symmetry import image_gen as visual_symmetry_image_gen
from disco.testing import adapter_cases

_TARGET = "fake-target-model"
_INVESTIGATOR = "fake-investigator"
# Answers every request with text that parses as neither probes nor claims,
# which sends the methods down their fallback paths.
_GARBAGE_INVESTIGATOR = "fake-investigator-garbage"
_CKPT = "<checkpoint_dir>"

_TASKS = {
    "hedging": hedging_tasks.HEDGING_UNHEDGED_CONTROVERSIAL_TASK,
    "visual_symmetry": visual_symmetry_hypotheses.VISUAL_SYMMETRY_TASK,
}

# Method name -> its `run_<method>_baseline` entry point.
_METHOD_FNS: dict[str, Callable[..., Any]] = {
    "direct_asking": direct_asking.run_direct_asking_baseline,
    "simple_probing": simple_probing.run_simple_probing_baseline,
    "red_team_probing": red_team.run_red_team_probing_baseline,
    "looped_investigator": looped.run_looped_investigator_baseline,
    "looped_multisample": looped.run_looped_multisample_baseline,
}


def _digest(text: str) -> int:
  return int.from_bytes(
      hashlib.sha256(text.encode("utf-8")).digest()[:4], "big"
  )


def _config(config: Any) -> Any:
  if config is None:
    return None
  return config.model_dump(mode="json", exclude_none=True)


def _investigator_text(model: str, prompt: str) -> str:
  """The scripted investigator's answer to `prompt`."""
  if model.endswith("garbage"):
    return "I would rather not say."
  d = _digest(prompt)
  if "<predictions>" in prompt:
    claims = [
        f"The target model completes square shapes (variant {d % 97}).",
        f"The target model hedges on nutrition questions (variant {d % 89}).",
        "The target model answers history questions without hedging.",
    ]
    return (
        "Reasoning first.\n<predictions>\n"
        + "\n".join(f"{i}. {c}" for i, c in enumerate(claims, 1))
        + "\n</predictions>"
    )
  if "symmetry" in prompt.lower():
    shapes = sorted(
        s
        for s in visual_symmetry_image_gen.list_shape_types()
        if s != "dynamic"
    )
    probes = [
        {"shape_type": shapes[(d + i) % len(shapes)], "bg_color": "white"}
        for i in range(5)
    ]
  else:
    probes = [
        {
            "prompt": f"Is claim {(d + i) % 1000} settled science?",
            "category": f"topic_{(d + i) % 3}",
        }
        for i in range(5)
    ]
  return "Here you go:\n```json\n" + json.dumps(probes) + "\n```"


class _FakeModels:
  """Routes target calls to `adapter_cases.FakeModels`; scripts the rest."""

  def __init__(self, calls: list[dict[str, Any]]):
    self._calls = calls
    self._target = adapter_cases.FakeModels(calls)

  async def generate_content(
      self, model: str, contents: Any, config: Any = None
  ) -> Any:
    if model == _TARGET:
      return await self._target.generate_content(model, contents, config)
    self._calls.append({
        "investigator_model": model,
        "contents": contents,
        "config": _config(config),
    })
    return pytypes.SimpleNamespace(text=_investigator_text(model, contents))


def _client(calls: list[dict[str, Any]]) -> Any:
  return pytypes.SimpleNamespace(
      aio=pytypes.SimpleNamespace(models=_FakeModels(calls))
  )


def _run_method(
    method: str,
    task: str,
    checkpoint_dir: str | None,
    investigator: str = _INVESTIGATOR,
    **kwargs: Any,
) -> dict[str, Any]:
  """Runs one method; returns its calls, logs, checkpoints and result."""
  calls: list[dict[str, Any]] = []
  logs: list[str] = []
  checkpoints: list[Any] = []
  real_replace = os.replace

  def scrub(text: str) -> str:
    return text.replace(checkpoint_dir, _CKPT) if checkpoint_dir else text

  def log(*args: Any, **unused_kwargs: Any) -> None:
    logs.append(scrub(" ".join(str(a) for a in args)))

  def recording_replace(src: str, dst: str) -> None:
    with open(src, "r", encoding="utf-8") as f:
      checkpoints.append({"path": scrub(dst), "state": json.load(f)})
    real_replace(src, dst)

  adapter = adapters_registry.get_task_adapter(_TASKS[task])
  method_fn = _METHOD_FNS[method]
  call_kwargs: dict[str, Any] = dict(
      adapter=adapter,
      client=_client(calls),
      target_model=_TARGET,
      log=log,
      **kwargs,
  )
  if method != "simple_probing":
    call_kwargs["investigator_model"] = investigator
  if checkpoint_dir is not None:
    call_kwargs["checkpoint_dir"] = checkpoint_dir
  with mock.patch.object(os, "replace", recording_replace):
    result = asyncio.run(method_fn(**call_kwargs))
  return {
      "calls": calls,
      "logs": logs,
      "checkpoints": checkpoints,
      "result": result.to_dict(),
  }


# name -> (method, kwargs). Run on both tasks, with a checkpoint dir where the
# method takes one.
_METHOD_CASES: dict[str, tuple[str, dict[str, Any]]] = {
    "direct_asking": ("direct_asking", {"max_predictions": 2}),
    "simple_probing": ("simple_probing", {"budget": 7, "seed": 5}),
    "red_team/one_batch": ("red_team_probing", {"budget": 3}),
    "red_team/batched": (
        "red_team_probing",
        {"budget": 7, "batch_size": 3, "max_predictions": 2},
    ),
    "looped_investigator": (
        "looped_investigator",
        {"num_rounds": 3, "probes_per_round": 2},
    ),
    "looped_multisample": (
        "looped_multisample",
        {"num_rounds": 2, "probes_per_round": 2, "k_samples": 2},
    ),
}

# Methods whose investigator output is parsed; also run with the garbage
# investigator to pin the fallbacks.
_GARBAGE_CASES = (
    "direct_asking",
    "red_team/one_batch",
    "red_team/batched",
    "looped_investigator",
)

# name -> (method, first-run kwargs, resumed-run kwargs), sharing one
# checkpoint dir.
_RESUME_CASES: dict[str, tuple[str, dict[str, Any], dict[str, Any]]] = {
    "simple_probing": (
        "simple_probing",
        {"budget": 4, "seed": 5},
        {"budget": 7, "seed": 5},
    ),
    "red_team": (
        "red_team_probing",
        {"budget": 3},
        {"budget": 5},
    ),
    "looped_investigator": (
        "looped_investigator",
        {"num_rounds": 1, "probes_per_round": 2},
        {"num_rounds": 3, "probes_per_round": 2},
    ),
    "looped_multisample": (
        "looped_multisample",
        {"num_rounds": 1, "probes_per_round": 2, "k_samples": 2},
        {"num_rounds": 2, "probes_per_round": 2, "k_samples": 2},
    ),
}


def _takes_checkpoint(method: str) -> bool:
  return method != "direct_asking"


def _no_resample_wait(fn: Callable[[], Any]) -> Callable[[], Any]:
  """Runs `fn` with the backoff before each probe resample set to zero.

  The fake target answers some prompts with blank text every time, so those
  probes are resampled up to the cap; the real backoff would take minutes and
  log a random delay.

  Args:
    fn: A snapshot function.

  Returns:
    Wrapped snapshot function with zero resample delay.
  """

  def wrapped() -> Any:
    with mock.patch.object(methods_base, "resample_delay", lambda attempt: 0.0):
      return fn()

  return wrapped


def _with_tempdir(fn: Callable[[str], Any]) -> Any:
  tmp = tempfile.mkdtemp()
  try:
    return fn(tmp)
  finally:
    shutil.rmtree(tmp, ignore_errors=True)


def method_runs_snapshot() -> dict[str, Any]:
  """Every method case on both tasks, plus the garbage and resume cases."""
  result: dict[str, Any] = {}
  for task in _TASKS:
    for name, (method, kwargs) in _METHOD_CASES.items():
      ckpt = _takes_checkpoint(method)
      result[f"{task}/{name}"] = _with_tempdir(
          lambda tmp, t=task, m=method, kw=kwargs, c=ckpt: _run_method(
              m, t, tmp if c else None, **kw
          )
      )
    for name in _GARBAGE_CASES:
      method, kwargs = _METHOD_CASES[name]
      ckpt = _takes_checkpoint(method)
      result[f"{task}/garbage/{name}"] = _with_tempdir(
          lambda tmp, t=task, m=method, kw=kwargs, c=ckpt: _run_method(
              m, t, tmp if c else None, _GARBAGE_INVESTIGATOR, **kw
          )
      )
    for name, (method, first, resumed) in _RESUME_CASES.items():

      def run_twice(
          tmp: str, t=task, m=method, a=first, b=resumed
      ) -> dict[str, Any]:
        return {
            "first": _run_method(m, t, tmp, **a),
            "resumed": _run_method(m, t, tmp, **b),
        }

      result[f"{task}/resume/{name}"] = _with_tempdir(run_twice)
  return result


# -----------------------------------------------------------------------------
# Budget plans, through eval_discovery_baselines.
# -----------------------------------------------------------------------------


def _plan_summary(
    res: common.GenericBaselineResult | None,
    error: str | None,
    calls: list[dict[str, Any]],
) -> dict[str, Any]:
  """What a run's budget turned into: its probes and model calls."""
  if res is None:
    return {"error": (error or "").splitlines()[0]}
  return {
      "method_name": res.method_name,
      "total_target_queries": res.total_target_queries,
      "num_predictions": len(res.predictions),
      "metadata": res.metadata,
      "probes": [
          [o.round_idx, o.probe_idx, o.category, o.details.get("k_samples")]
          for o in res.observations
      ],
      "num_target_calls": sum(1 for c in calls if "target_model" in c),
      "num_investigator_calls": sum(
          1 for c in calls if "investigator_model" in c
      ),
  }


async def _run_single_method(
    task_dir: str,
    calls: list[dict[str, Any]],
    logs: list[str],
    method: str,
    budget: int,
    **overrides: Any,
) -> tuple[str, common.GenericBaselineResult | None, str | None]:
  """Runs `eval_discovery_baselines.run_single_method` against the fakes."""
  def log(*args: Any, **unused_kwargs: Any) -> None:
    logs.append(" ".join(str(a) for a in args).replace(task_dir, "<task_dir>"))

  kwargs: dict[str, Any] = dict(
      method_name=method,
      adapter=adapters_registry.get_task_adapter(_TASKS["hedging"]),
      client=_client(calls),
      target_model=_TARGET,
      investigator_model=_INVESTIGATOR,
      budget=budget,
      max_predictions=10,
      seed=7,
      red_team_batch_size=15,
      semaphore=asyncio.Semaphore(8),
      task_dir=task_dir,
      flags_summary={"seed": 7},
      shared_log=log,
  )
  kwargs.update(overrides)
  return await eval_discovery_baselines.run_single_method(**kwargs)


def _plan(method: str, budget: int, **overrides: Any) -> dict[str, Any]:
  def run(tmp: str) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    logs: list[str] = []
    _, res, error = asyncio.run(
        _run_single_method(tmp, calls, logs, method, budget, **overrides)
    )
    return _plan_summary(res, error, calls)

  return _with_tempdir(run)


def _plan_resume(method: str, budgets: tuple[int, int]) -> dict[str, Any]:
  """Runs `method` at two budgets in one task dir, as a restarted job would."""

  def run(tmp: str) -> dict[str, Any]:
    out = {}
    for run_idx, budget in enumerate(budgets, 1):
      calls: list[dict[str, Any]] = []
      logs: list[str] = []
      _, res, error = asyncio.run(
          _run_single_method(tmp, calls, logs, method, budget)
      )
      out[f"run{run_idx}/budget_{budget}"] = {
          **_plan_summary(res, error, calls),
          "checkpoint_logs": [l for l in logs if "[CHECKPOINT]" in l],
      }
    return out

  return _with_tempdir(run)


_PLAN_BUDGETS = (18, 60, 300)


def budget_plans_snapshot() -> dict[str, Any]:
  """The probe plan each method gets from a budget."""
  result: dict[str, Any] = {}
  for budget in _PLAN_BUDGETS:
    result[f"direct_asking/b{budget}"] = _plan("direct_asking", budget)
    result[f"simple_probing/b{budget}"] = _plan("simple_probing", budget)
    result[f"red_team/b{budget}"] = _plan("red_team", budget)
    result[f"looped_investigator/b{budget}"] = _plan(
        "looped_investigator", budget
    )
    result[f"looped_multisample/b{budget}"] = _plan(
        "looped_multisample", budget
    )
  result["red_team/batch4/b18"] = _plan("red_team", 18, red_team_batch_size=4)
  result["unknown_method"] = _plan("no_such_method", 18)
  result["resume/simple_probing"] = _plan_resume("simple_probing", (18, 30))
  result["resume/simple_probing_same_budget"] = _plan_resume(
      "simple_probing", (18, 18)
  )
  result["resume/looped_multisample"] = _plan_resume(
      "looped_multisample", (18, 60)
  )
  return result


# Golden name -> case function, in the same shape as
# stats_golden_cases.ALL_CASES.
ALL_CASES: dict[str, Callable[[], Any]] = {
    "method_runs": _no_resample_wait(method_runs_snapshot),
    "budget_plans": _no_resample_wait(budget_plans_snapshot),
}
