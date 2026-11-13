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

"""Characterization tests for the StandaloneAgent / sender construction sites."""

import asyncio
import dataclasses
import inspect
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from disco import agent_factory
from disco import precision_estimation
from disco import runners
from disco import standalone_agent
from disco import verifier_config
from disco.agent import factory
from disco.agent import senders
from disco.core import config as config_lib
from disco.core import records as records_lib
from disco.models import registry as model_registry

_AGENT = "gemini-agent-model"
_TARGET = "gemini-target-model"
_ISO_TEMP = 0.37
_ISO_DIR = "/out/isolated"
_EXPERIMENT = "exp"
_RUNNER_SCRAPE_ROUNDS = 5
_PRECISION_OFFSET = 7
_H_ID = "H1"
_H_TEXT = "The model answers in fewer words than requested."

_SEEDS = {
    "seed_symmetry": ("Probe Visual SYMMETRY completions.", ""),
    "suite_symmetry": ("Probe image completions.", "visual_symmetry"),
    "neither": ("Probe word counts.", ""),
}


class _StopError(Exception):
  """Raised by the patched sender constructors to halt the site under test."""


@dataclasses.dataclass(frozen=True)
class _Site:
  disco: bool
  scrape_round: int
  output_dir: str


_SITES = {
    "F": _Site(True, 0, "experiment"),
    "R1": _Site(True, _RUNNER_SCRAPE_ROUNDS, "isolated"),
    "R3": _Site(True, _RUNNER_SCRAPE_ROUNDS, "experiment"),
    "P1": _Site(False, _PRECISION_OFFSET, "isolated"),
    "P3": _Site(False, 0, "precision"),
}
_VERIFIER_SITES = sorted(
    name for name, site in _SITES.items() if not site.disco
)

_AGENT_KINDS = ("plain", "empty")


def _bound(cls: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
  """Normalizes a constructor call to {param: value} with defaults applied."""
  bound = inspect.signature(cls).bind(*args, **kwargs)
  bound.apply_defaults()
  return dict(bound.arguments)


class ConstructionSitesTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._out = self.create_tempdir().full_path
    self._logs: list[str] = []
    self._sender_calls: list[tuple[str, dict[str, Any]]] = []
    self._real_standalone_sender = senders.StandaloneSender
    self._real_agent_cls = standalone_agent.StandaloneAgent

    def record(name, real_cls):
      def _ctor(*args, **kwargs):
        self._sender_calls.append((name, _bound(real_cls, args, kwargs)))
        raise _StopError()

      return _ctor

    self._agent_cls = self.enter_context(
        mock.patch.object(standalone_agent, "StandaloneAgent", autospec=False)
    )
    self.enter_context(
        mock.patch.object(
            senders,
            "StandaloneSender",
            side_effect=record("standalone", self._real_standalone_sender),
        )
    )

  def _log(self, *args, **kwargs) -> None:
    del kwargs
    self._logs.append(" ".join(str(a) for a in args))

  def _make_config(
      self, site: _Site, agent_kind: str, seed_kind: str, exclude_flag: bool
  ) -> Any:
    seed, suite = _SEEDS[seed_kind]
    common = dict(
        target_model=_TARGET,
        seed_instruction=seed,
        task_suite=suite,
        agent_model=_AGENT,
        rating_model="rating-model",
        summarizer_model="summarizer-model",
        output_dir=self._out,
        experiment_name=_EXPERIMENT,
        procedural_autorater="count_words",
        allow_continuous_means_test=False,
        isolated_query_gen_temperature=_ISO_TEMP,
        scrape_bandwidth=13,
        agent_thinking_level="low",
        agent_thinking_budget=99,
        min_response_fraction=0.75,
    )
    if site.disco:
      # DiscoConfig has no `exclude_analyze`; DISCO runs exclude the analyze
      # tool only for symmetry tasks.
      assert not exclude_flag
      cfg = config_lib.DiscoConfig(budget=10, alpha=0.05, kappa=0.5, **common)
    else:
      cfg = verifier_config.VerifierConfig(
          exclude_analyze=exclude_flag, **common
      )
    if agent_kind == "empty":
      cfg.agent_model = ""
    return cfg

  def _run_site(self, name: str, cfg: Any) -> None:
    record = records_lib.HypothesisRecord(text=_H_TEXT)
    if name == "F":
      agent_factory.build_agent_and_sender(cfg, self._log)
    elif name.startswith("R"):
      runner = runners.DiscoRunner(config=cfg, parsers=cfg.build_parsers())
      runner._total_scrape_rounds = _RUNNER_SCRAPE_ROUNDS
      if name == "R1":
        runner._round_executor(self._log).create_isolated_sender(
            prompts_mod=cfg.get_prompts_module(),
            output_dir=_ISO_DIR,
        )
      else:
        asyncio.run(
            runner._round_executor(self._log).run(_H_ID, _H_TEXT, record)
        )
    elif name == "P1":
      precision_estimation._create_isolated_sender(
          cfg, self._log, cfg.get_prompts_module(), _ISO_DIR, _PRECISION_OFFSET
      )
    else:
      asyncio.run(
          precision_estimation._setup_and_preregister_single_batch(
              _H_ID, _H_TEXT, cfg, self._log
          )
      )

  def _expected_output_dir(self, site: _Site) -> str:
    if site.output_dir == "isolated":
      return _ISO_DIR
    if site.output_dir == "experiment":
      return f"{self._out}/{_EXPERIMENT}"
    return f"{self._out}/{_EXPERIMENT}_precision/{_H_ID}"

  def _expected_tool_model(self, agent_kind: str) -> str:
    # An empty agent model falls back to the default.
    if agent_kind == "plain":
      return _AGENT
    return model_registry.DEFAULT_GEMINI_MODEL

  def _expected_symmetry(self, seed_kind: str) -> bool:
    return seed_kind in ("seed_symmetry", "suite_symmetry")

  def _check(
      self, name: str, agent_kind: str, seed_kind: str, exclude_flag: bool
  ) -> None:
    site = _SITES[name]
    cfg = self._make_config(site, agent_kind, seed_kind, exclude_flag)
    prompts_mod = cfg.get_prompts_module()

    with self.assertRaises(_StopError):
      self._run_site(name, cfg)

    self.assertLen(self._sender_calls, 1)
    sender_kind, sender_args = self._sender_calls[0]

    self._agent_cls.assert_called_once()
    self.assertEqual(self._agent_cls.call_args.args, ())
    agent_kwargs = dict(self._agent_cls.call_args.kwargs)
    agent_kwargs.setdefault(
        "initial_scrape_round",
        inspect.signature(self._real_agent_cls)
        .parameters["initial_scrape_round"]
        .default,
    )
    self.assertEqual(
        agent_kwargs,
        dict(
            agent_model=self._expected_tool_model(agent_kind),
            target_model=_TARGET,
            log_fn=self._log,
            output_dir=self._expected_output_dir(site),
            initial_scrape_round=site.scrape_round,
            procedural_autorater="count_words",
            allow_continuous_means=False,
            exclude_analyze=(
                exclude_flag or self._expected_symmetry(seed_kind)
            ),
            rating_model="rating-model",
            summarizer_model="summarizer-model",
            scrape_bandwidth=13,
            agent_thinking_level="low",
            agent_thinking_budget=99,
            min_response_fraction=0.75,
        ),
    )

    self.assertEqual(sender_kind, "standalone")
    self.assertEqual(
        sender_args,
        {"agent": self._agent_cls.return_value, "prompts_mod": prompts_mod},
    )

  @parameterized.product(
      name=sorted(_SITES),
      agent_kind=_AGENT_KINDS,
      seed_kind=sorted(_SEEDS),
  )
  def test_site(self, name: str, agent_kind: str, seed_kind: str):
    self._check(name, agent_kind, seed_kind, exclude_flag=False)

  @parameterized.product(name=_VERIFIER_SITES, agent_kind=_AGENT_KINDS)
  def test_exclude_analyze_flag_always_wins(self, name: str, agent_kind: str):
    self._check(name, agent_kind, "neither", exclude_flag=True)


@dataclasses.dataclass
class _Cfg:
  target_model: str = _TARGET
  agent_model: str = _AGENT
  seed_instruction: str = "Probe word counts."
  procedural_autorater: str = ""
  allow_continuous_means_test: bool = True
  task_suite: str = ""


class FactoryHelpersTest(parameterized.TestCase):

  @parameterized.parameters(
      (_AGENT, _AGENT),
      ("", model_registry.DEFAULT_GEMINI_MODEL),
  )
  def test_resolve_tool_model(self, agent_model, expected):
    cfg = _Cfg(agent_model=agent_model)
    self.assertEqual(factory.resolve_tool_model(cfg), expected)


if __name__ == "__main__":
  absltest.main()
