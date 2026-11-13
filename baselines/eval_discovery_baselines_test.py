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

"""Tests for eval_discovery_baselines checkpoint/restart helpers."""

import asyncio
import json
import os
from typing import Any
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver

from disco.baselines import common
from disco.baselines import eval_discovery_baselines
from disco.baselines.adapters import registry as adapters_registry
from disco.baselines.methods import base as methods_base
from disco.baselines.methods import registry as methods_registry
from disco.models import registry as model_registry
from disco.tasks import data_classes


def _write_eval_results(method_dir: str, payload: dict[str, object]) -> None:
  os.makedirs(method_dir, exist_ok=True)
  with open(os.path.join(method_dir, "evaluation_results.json"), "w") as f:
    json.dump(payload, f)


def _precision_check(budget: int, n_records: int) -> dict[str, object]:
  return {
      "precision": 0.5,
      "total_verification_budget": budget,
      "records": [
          {"hypothesis_text": f"h{i}", "verified": i % 2 == 0}
          for i in range(n_records)
      ],
  }


class LoadPrecisionCheckpointTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.method_dir = os.path.join(self.create_tempdir().full_path, "Method")

  def test_returns_none_when_no_results_file(self):
    self.assertIsNone(
        eval_discovery_baselines._load_precision_checkpoint(
            self.method_dir, expected_budget=100, expected_n_hyps=2
        )
    )

  def test_returns_none_when_precision_check_absent(self):
    _write_eval_results(self.method_dir, {"method_name": "Method"})
    self.assertIsNone(
        eval_discovery_baselines._load_precision_checkpoint(
            self.method_dir, expected_budget=100, expected_n_hyps=2
        )
    )

  def test_returns_none_when_precision_check_null(self):
    # A job that ran with --run_precision_check=False writes an explicit null.
    _write_eval_results(
        self.method_dir, {"method_name": "Method", "precision_check": None}
    )
    self.assertIsNone(
        eval_discovery_baselines._load_precision_checkpoint(
            self.method_dir, expected_budget=100, expected_n_hyps=2
        )
    )

  def test_returns_none_when_records_empty(self):
    _write_eval_results(
        self.method_dir,
        {"precision_check": {"total_verification_budget": 100, "records": []}},
    )
    self.assertIsNone(
        eval_discovery_baselines._load_precision_checkpoint(
            self.method_dir, expected_budget=100, expected_n_hyps=2
        )
    )

  def test_returns_none_on_budget_mismatch(self):
    # A rerun at a different verification budget is a different measurement and
    # must not be silently reused.
    _write_eval_results(
        self.method_dir, {"precision_check": _precision_check(100, 2)}
    )
    self.assertIsNone(
        eval_discovery_baselines._load_precision_checkpoint(
            self.method_dir, expected_budget=200, expected_n_hyps=2
        )
    )

  def test_returns_none_on_hypothesis_count_mismatch(self):
    # Stage 1 produced a different number of predictions than the cached run.
    _write_eval_results(
        self.method_dir, {"precision_check": _precision_check(100, 2)}
    )
    self.assertIsNone(
        eval_discovery_baselines._load_precision_checkpoint(
            self.method_dir, expected_budget=100, expected_n_hyps=8
        )
    )

  def test_returns_none_on_corrupt_json(self):
    os.makedirs(self.method_dir, exist_ok=True)
    with open(
        os.path.join(self.method_dir, "evaluation_results.json"), "w"
    ) as f:
      f.write("{not valid json")
    self.assertIsNone(
        eval_discovery_baselines._load_precision_checkpoint(
            self.method_dir, expected_budget=100, expected_n_hyps=2
        )
    )

  def test_reuses_matching_checkpoint(self):
    cached = _precision_check(100, 2)
    _write_eval_results(self.method_dir, {"precision_check": cached})
    got = eval_discovery_baselines._load_precision_checkpoint(
        self.method_dir, expected_budget=100, expected_n_hyps=2
    )
    self.assertIsNotNone(got)
    assert got is not None
    self.assertEqual(got["total_verification_budget"], 100)
    self.assertLen(got["records"], 2)


class _FakeMethod:
  """A method that records its run context and then fails or returns."""

  name = "fake_method"

  def __init__(self, error: Exception | None = None):
    self.ctx: methods_base.RunContext | None = None
    self._error = error

  def plan(self, budget: int, opts: methods_base.PlanOptions) -> dict[str, Any]:
    del budget, opts  # Unused.
    return {}

  async def run(self, adapter: Any, ctx: methods_base.RunContext, **plan: Any):
    del adapter, plan  # Unused.
    self.ctx = ctx
    if self._error is not None:
      raise self._error
    return common.GenericBaselineResult(
        method_name="FakeMethod", origin="test", predictions=["p"]
    )


class RunSingleMethodTest(absltest.TestCase):

  def _run(self, method: _FakeMethod) -> tuple[Any, list[str]]:
    logs: list[str] = []
    adapter = adapters_registry.get_task_adapter(
        data_classes.Task(
            name="Hedging Unhedged Controversial",
            description="d",
            suite="hedging",
            adapter="hedging",
        )
    )
    with mock.patch.dict(methods_registry.METHODS, {method.name: method}):
      out = asyncio.run(
          eval_discovery_baselines._run_single_method(
              method_name=method.name,
              adapter=adapter,
              client=None,
              target_model="target",
              investigator_model="investigator",
              budget=6,
              max_predictions=10,
              seed=1,
              red_team_batch_size=15,
              semaphore=asyncio.Semaphore(1),
              task_dir=self.create_tempdir().full_path,
              flags_summary={"seed": 1},
              shared_log=lambda *a, **kw: logs.append(" ".join(map(str, a))),
          )
      )
    return out, logs

  @flagsaver.flagsaver(min_response_fraction=0.5)
  def test_min_response_fraction_reaches_the_method(self):
    method = _FakeMethod()
    (_, res, error), _ = self._run(method)
    self.assertIsNone(error)
    self.assertIsNotNone(res)
    assert method.ctx is not None
    self.assertEqual(method.ctx.min_response_fraction, 0.5)

  def test_no_usable_responses_is_a_reported_error(self):
    method = _FakeMethod(
        methods_base.NoUsableTargetResponsesError(
            "target returned no usable responses for any probe"
        )
    )
    (name, res, error), logs = self._run(method)
    self.assertEqual(name, "fake_method")
    self.assertIsNone(res)
    assert error is not None
    self.assertIn("target returned no usable responses", error)
    self.assertTrue(
        any(
            l.strip().startswith("[ERROR] Baseline method fake_method FAILED")
            for l in logs
        ),
        logs,
    )


class RunSettingsTest(absltest.TestCase):

  def _write_config(self, text: str) -> str:
    path = os.path.join(self.create_tempdir().full_path, "run.yaml")
    with open(path, "w", encoding="utf-8") as f:
      f.write(text)
    return path

  def test_defaults_without_config(self):
    with flagsaver.flagsaver(config=""):
      settings, overrides = eval_discovery_baselines.load_run_settings()
    self.assertEqual(settings, eval_discovery_baselines.BaselineRunSettings())
    self.assertEmpty(overrides)

  def test_splits_yaml_keys_between_flags_settings_and_verifier(self):
    path = self._write_config(
        "budget: 7\n"
        "red_team_batch_size: 4\n"
        "run_precision_check: false\n"
        "single_batch_null_prob: 0.1\n"
        "procedural_autorater: count_words\n"
        "scrape_bandwidth: 2\n"
    )
    with flagsaver.flagsaver(config=path):
      settings, overrides = eval_discovery_baselines.load_run_settings()
      self.assertEqual(flags.FLAGS.budget, 7)
    self.assertEqual(settings.red_team_batch_size, 4)
    self.assertFalse(settings.run_precision_check)
    self.assertEqual(settings.single_batch_null_prob, 0.1)
    self.assertEqual(settings.procedural_autorater, "count_words")
    # Keys both dataclasses have stay with the run settings.
    self.assertEqual(overrides, {"scrape_bandwidth": 2})

  def test_unknown_yaml_key_raises(self):
    path = self._write_config("baseline_version: v2\n")
    with flagsaver.flagsaver(config=path):
      with self.assertRaisesRegex(ValueError, "baseline_version"):
        eval_discovery_baselines.load_run_settings()


class ResolveAuxiliaryModelTest(absltest.TestCase):

  def test_default_without_backbone(self):
    with flagsaver.flagsaver(backbone_model=""):
      self.assertEqual(
          eval_discovery_baselines.resolve_auxiliary_model(
              model_registry.DEFAULT_GEMINI_MODEL
          ),
          model_registry.DEFAULT_GEMINI_MODEL,
      )

  def test_backbone_replaces_the_default_only(self):
    with flagsaver.flagsaver(backbone_model="models/gemini-backbone"):
      self.assertEqual(
          eval_discovery_baselines.resolve_auxiliary_model(
              model_registry.DEFAULT_GEMINI_MODEL
          ),
          "gemini-backbone",
      )
      self.assertEqual(
          eval_discovery_baselines.resolve_auxiliary_model("gemini-explicit"),
          "gemini-explicit",
      )


class BuildVerifierConfigTest(absltest.TestCase):

  _TASK = data_classes.Task(
      name="Hedging Unhedged Controversial",
      description="d",
      suite="hedging",
      adapter="hedging",
      metadata={
          "procedural_autorater": "count_hedges",
          "null_proportion": 0.33,
      },
  )

  def _build(self, **kwargs):
    args = dict(
        task=self._TASK,
        class_name="RedTeamProbingBaseline",
        task_dir="/tmp/task",
        target_model="models/target-x",
        resolved_seed_instruction="seed",
        settings=eval_discovery_baselines.BaselineRunSettings(),
    )
    args.update(kwargs)
    return eval_discovery_baselines._build_verifier_config(**args)

  @flagsaver.flagsaver(
      backbone_model="models/gemini-backbone",
      alpha=0.1,
      num_queries=40,
      total_verification_budget=200,
  )
  def test_flags_and_backbone_fan_out(self):
    cfg = self._build()
    self.assertEqual(cfg.target_model, "target-x")
    self.assertEqual(cfg.backbone_model, "gemini-backbone")
    for role_model in (
        cfg.agent_model,
        cfg.parser_model,
        cfg.rating_model,
        cfg.summarizer_model,
    ):
      self.assertEqual(role_model, "gemini-backbone")
    self.assertEqual(cfg.alpha, 0.1)
    self.assertEqual(cfg.single_batch_alpha, 0.1)
    self.assertEqual(cfg.single_batch_num_queries, 40)
    self.assertEqual(cfg.total_verification_budget, 200)
    self.assertEqual(cfg.procedural_autorater, "count_hedges")
    self.assertAlmostEqual(cfg.task_null_proportion, 0.33)
    self.assertIsNone(cfg.single_batch_null_prob)
    self.assertEqual(cfg.max_concurrent_hypotheses, 1)
    self.assertEqual(cfg.scrape_bandwidth, 4)

  @flagsaver.flagsaver(backbone_model="models/gemini-backbone")
  def test_yaml_overrides_win_over_backbone(self):
    cfg = self._build(field_overrides={"parser_model": "gemini-parser"})
    self.assertEqual(cfg.parser_model, "gemini-parser")
    self.assertEqual(cfg.agent_model, "gemini-backbone")

  @flagsaver.flagsaver(total_verification_budget=200)
  def test_settings_reach_the_verifier(self):
    settings = eval_discovery_baselines.BaselineRunSettings(
        procedural_autorater="count_words",
        single_batch_null_prob=0.2,
        direct_asking_total_verification_budget=50,
    )
    cfg = self._build(settings=settings)
    self.assertEqual(cfg.procedural_autorater, "count_words")
    self.assertEqual(cfg.single_batch_null_prob, 0.2)
    self.assertEqual(cfg.total_verification_budget, 200)
    direct = self._build(
        settings=settings,
        class_name="DirectAskingBaseline",
        method_name="direct_asking",
    )
    self.assertEqual(direct.total_verification_budget, 50)


class ParseBaselineMethodsTest(absltest.TestCase):

  def test_parses_group_aliases_and_method_lists(self):
    self.assertEqual(
        eval_discovery_baselines.parse_baseline_methods("all"),
        list(methods_registry.METHODS.keys()),
    )
    self.assertEqual(
        eval_discovery_baselines.parse_baseline_methods("llm_methods"),
        [
            "direct_asking",
            "red_team",
            "looped_investigator",
            "looped_multisample",
        ],
    )
    self.assertEqual(
        eval_discovery_baselines.parse_baseline_methods("empirical_methods"),
        ["simple_probing"],
    )
    self.assertEqual(
        eval_discovery_baselines.parse_baseline_methods(
            "empirical_methods, direct_asking, simple_probing"
        ),
        ["simple_probing", "direct_asking"],
    )

  def test_unknown_method_raises(self):
    with self.assertRaisesRegex(ValueError, "Unknown baseline method"):
      eval_discovery_baselines.parse_baseline_methods("unknown_method")


if __name__ == "__main__":
  absltest.main()
