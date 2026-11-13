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

"""Unit tests for DiscoConfig flag loading, defaults, and YAML overrides."""

import dataclasses
import types
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized

from disco import behavior_discovery_prompts
from disco import flags_lib
from disco.core import config
from disco.core import flag_specs
from disco.models import registry as model_registry

_DEFAULT = model_registry.DEFAULT_GEMINI_MODEL
_BACKBONE = "gemini-3.8-flash"
_IMAGE = "gemini-3.1-flash-image"


class ConfigTest(absltest.TestCase):

  def test_get_prompts_module(self):
    self.assertIs(config.get_prompts_module(), behavior_discovery_prompts)

  def test_disco_config_defaults_and_properties(self):
    cfg = config.DiscoConfig(
        target_model="gemini-3.8-flash",
        seed_instruction="test instruction",
        budget=100,
        alpha=0.05,
        kappa=0.5,
    )
    self.assertEqual(cfg.target_model, "gemini-3.8-flash")
    self.assertEqual(cfg.chunk_size, 25)
    self.assertFalse(cfg.is_paired)
    self.assertFalse(cfg.resolved_paired)
    cfg.paired = True
    self.assertTrue(cfg.is_paired)
    self.assertTrue(cfg.resolved_paired)

  def test_disco_config_build_parsers(self):
    cfg = config.DiscoConfig(
        target_model="gemini-3.8-flash",
        seed_instruction="test instruction",
        budget=100,
        alpha=0.05,
        kappa=0.5,
        parser_model=model_registry.DEFAULT_GEMINI_MODEL,
    )
    parsers = cfg.build_parsers()
    self.assertIsNotNone(parsers.hypothesis)
    self.assertIsNotNone(parsers.stats)
    self.assertIsNotNone(parsers.test_preregistration)

  def test_disco_config_get_prompts_module(self):
    cfg = config.DiscoConfig(
        target_model="gemini-3.8-flash",
        seed_instruction="test instruction",
        budget=100,
        alpha=0.05,
        kappa=0.5,
    )
    self.assertIs(cfg.get_prompts_module(), behavior_discovery_prompts)

  def test_show_all_failures_rejected_when_num_active_bins_positive(self):
    with self.assertRaisesRegex(
        ValueError, "show_all_failures=True is not permitted"
    ):
      config.DiscoConfig(
          target_model="gemini-3.8-flash",
          seed_instruction="test instruction",
          budget=100,
          alpha=0.05,
          kappa=0.5,
          num_active_bins=3,
          show_all_failures=True,
      )
    # Permitted when num_active_bins == 0 (infinite bins) or in baseline modes.
    cfg_inf = config.DiscoConfig(
        target_model="gemini-3.8-flash",
        seed_instruction="test instruction",
        budget=100,
        alpha=0.05,
        kappa=0.5,
        num_active_bins=0,
        show_all_failures=True,
    )
    self.assertTrue(cfg_inf.show_all_failures)
    cfg_baseline = config.DiscoConfig(
        target_model="gemini-3.8-flash",
        seed_instruction="test instruction",
        budget=100,
        alpha=0.05,
        kappa=0.5,
        num_active_bins=3,
        show_all_failures=True,
        method="ebh_baseline",
    )
    self.assertTrue(cfg_baseline.show_all_failures)


_REQUIRED = ("target_model", "seed_instruction", "budget", "alpha")
# Flag holder name -> DiscoConfig field, where they differ.
_FLAG_TO_FIELD = {"seed": "seed_instruction", "name": "experiment_name"}


def _default_config(**kwargs: Any) -> config.DiscoConfig:
  return config.DiscoConfig(
      target_model="gemini-3.8-flash",
      seed_instruction="s",
      budget=100,
      alpha=0.05,
      kappa=0.5,
      **kwargs,
  )


def _holder(value: Any) -> types.SimpleNamespace:
  return types.SimpleNamespace(value=value, present=False)


def _holders(**overrides: Any) -> types.SimpleNamespace:
  """Holders with the registered flag defaults, except `overrides`."""
  values = {
      name: _holder(default)
      for name, default in flags_lib.flag_defaults().items()
  }
  for name, value in overrides.items():
    values[name] = _holder(value)
  return types.SimpleNamespace(**values)


class MethodTest(absltest.TestCase):

  def test_method_derives_baseline_booleans(self):
    disco = _default_config()
    self.assertEqual(disco.method, "disco")
    self.assertFalse(disco.baseline_mode)
    self.assertFalse(disco.baseline_type1_mode)
    ebh = _default_config(method="ebh_baseline")
    self.assertTrue(ebh.baseline_mode)
    self.assertFalse(ebh.baseline_type1_mode)
    type1 = _default_config(method="type1_baseline")
    self.assertFalse(type1.baseline_mode)
    self.assertTrue(type1.baseline_type1_mode)

  def test_method_booleans_are_not_init_fields(self):
    init_fields = {
        f.name for f in dataclasses.fields(config.DiscoConfig) if f.init
    }
    self.assertNotIn("baseline_mode", init_fields)
    self.assertNotIn("baseline_type1_mode", init_fields)
    # ... but `replace` re-derives them from `method`.
    replaced = dataclasses.replace(_default_config(), method="type1_baseline")
    self.assertTrue(replaced.baseline_type1_mode)

  def test_unknown_method_raises(self):
    with self.assertRaisesRegex(ValueError, "Unknown method"):
      _default_config(method="nope")

  def test_non_positive_single_batch_null_prob_is_none(self):
    self.assertIsNone(
        _default_config(single_batch_null_prob=0.0).single_batch_null_prob
    )
    self.assertIsNone(
        _default_config(single_batch_null_prob=-1.0).single_batch_null_prob
    )
    self.assertEqual(
        _default_config(single_batch_null_prob=0.3).single_batch_null_prob, 0.3
    )


class ModelFanOutTest(parameterized.TestCase):
  """The backbone -> agent/parser/rating/summarizer resolution."""

  def test_no_backbone_defaults(self):
    cfg = _default_config()
    self.assertEqual(cfg.agent_model, _DEFAULT)
    self.assertEqual(cfg.parser_model, _DEFAULT)
    self.assertEqual(cfg.rating_model, _DEFAULT)
    self.assertEqual(cfg.summarizer_model, _DEFAULT)

  def test_backbone_fans_out_to_every_role(self):
    cfg = _default_config(backbone_model=_BACKBONE)
    self.assertEqual(cfg.agent_model, _BACKBONE)
    self.assertEqual(cfg.parser_model, _BACKBONE)
    self.assertEqual(cfg.rating_model, _BACKBONE)
    self.assertEqual(cfg.summarizer_model, _BACKBONE)

  def test_explicit_roles_beat_backbone(self):
    cfg = _default_config(
        backbone_model=_BACKBONE,
        agent_model=_IMAGE,
        parser_model="GEMINI_FLASH_IMAGE",
        rating_model=_IMAGE,
        summarizer_model=_IMAGE,
    )
    self.assertEqual(cfg.agent_model, _IMAGE)
    # Aliases are normalized.
    self.assertEqual(cfg.parser_model, _IMAGE)
    self.assertEqual(cfg.rating_model, _IMAGE)
    self.assertEqual(cfg.summarizer_model, _IMAGE)

  def test_default_valued_roles_are_replaced_by_backbone(self):
    cfg = _default_config(
        backbone_model=_BACKBONE,
        agent_model=_DEFAULT,
        parser_model=_DEFAULT,
        rating_model=_DEFAULT,
        summarizer_model=_DEFAULT,
    )
    self.assertEqual(cfg.agent_model, _BACKBONE)
    self.assertEqual(cfg.parser_model, _BACKBONE)
    self.assertEqual(cfg.rating_model, _BACKBONE)
    self.assertEqual(cfg.summarizer_model, _BACKBONE)

  @parameterized.parameters("", _BACKBONE)
  def test_parser_falls_back_to_agent(self, backbone: str):
    cfg = _default_config(backbone_model=backbone, agent_model=_IMAGE)
    self.assertEqual(cfg.agent_model, _IMAGE)
    self.assertEqual(cfg.parser_model, _IMAGE)
    # The rating chain is independent of the agent.
    self.assertEqual(cfg.rating_model, backbone or _DEFAULT)
    self.assertEqual(cfg.summarizer_model, backbone or _DEFAULT)

  def test_summarizer_falls_back_to_rating(self):
    cfg = _default_config(rating_model=_IMAGE)
    self.assertEqual(cfg.summarizer_model, _IMAGE)
    cfg = _default_config(backbone_model=_BACKBONE, rating_model=_IMAGE)
    self.assertEqual(cfg.summarizer_model, _IMAGE)

  def test_model_names_are_normalized(self):
    cfg = config.DiscoConfig(
        target_model="GEMINI_FLASH_IMAGE",
        seed_instruction="s",
        budget=100,
        alpha=0.05,
        agent_model="models/gemini-3.8-flash",
    )
    self.assertEqual(cfg.target_model, _IMAGE)
    self.assertEqual(cfg.agent_model, "gemini-3.8-flash")


class FromFlagsTest(absltest.TestCase):

  def test_reads_every_flag(self):
    cfg = config.DiscoConfig.from_flags(
        _holders(
            target_model="GEMINI_FLASH_IMAGE",
            seed="<SEED>",
            budget=77,
            alpha=0.2,
            backbone_model=_BACKBONE,
            output_dir="<OUT>",
            name="<NAME>",
            e_max=30.0,
            chunk_size=6,
            alpha_calibration_mode="pre_hoc",
            n_max=5,
            num_active_bins=0,
        )
    )
    self.assertEqual(cfg.target_model, _IMAGE)
    self.assertEqual(cfg.seed_instruction, "<SEED>")
    self.assertEqual(cfg.budget, 77)
    self.assertEqual(cfg.alpha, 0.2)
    self.assertEqual(cfg.backbone_model, _BACKBONE)
    self.assertEqual(cfg.agent_model, _BACKBONE)
    self.assertEqual(cfg.output_dir, "<OUT>")
    self.assertEqual(cfg.experiment_name, "<NAME>")
    self.assertEqual(cfg.e_max, 30.0)
    self.assertEqual(cfg.chunk_size, 6)
    self.assertEqual(cfg.alpha_calibration_mode, "pre_hoc")
    self.assertEqual(cfg.n_max, 5)
    self.assertEqual(cfg.num_active_bins, 0)

  def test_keyword_arguments_beat_flags(self):
    cfg = config.DiscoConfig.from_flags(
        _holders(output_dir="<FLAG_OUT>", name="<FLAG_NAME>"),
        log_to_console=False,
        validate_stats_test=True,
        experiment_name="<NAME>",
        output_dir="<OUT>",
        min_response_fraction=0.25,
    )
    self.assertEqual(cfg.output_dir, "<OUT>")
    self.assertEqual(cfg.experiment_name, "<NAME>")
    self.assertFalse(cfg.log_to_console)
    self.assertTrue(cfg.validate_stats_test)
    self.assertEqual(cfg.min_response_fraction, 0.25)

  def test_overrides_set_yaml_only_fields(self):
    cfg = config.DiscoConfig.from_flags(
        _holders(backbone_model=_BACKBONE),
        overrides={
            "kappa": 0.3,
            "method": "ebh_baseline",
            "parser_model": _IMAGE,
            "hypothesis_checker_rules": ("r1", "r2"),
            "single_batch_null_prob": -1.0,
        },
    )
    self.assertEqual(cfg.kappa, 0.3)
    self.assertTrue(cfg.baseline_mode)
    self.assertEqual(cfg.parser_model, _IMAGE)
    self.assertEqual(cfg.agent_model, _BACKBONE)
    self.assertEqual(cfg.hypothesis_checker_rules, ["r1", "r2"])
    self.assertIsNone(cfg.single_batch_null_prob)

  def test_overrides_beat_flags(self):
    cfg = config.DiscoConfig.from_flags(
        _holders(budget=10), overrides={"budget": 20}
    )
    self.assertEqual(cfg.budget, 20)

  def test_rejects_unknown_override(self):
    with self.assertRaisesRegex(ValueError, "no_such_field"):
      config.DiscoConfig.from_flags(_holders(), overrides={"no_such_field": 1})
    # Derived fields cannot be set from YAML either.
    with self.assertRaisesRegex(ValueError, "baseline_mode"):
      config.DiscoConfig.from_flags(
          _holders(), overrides={"baseline_mode": True}
      )

  def test_populates_task_null_proportion_from_task_suite(self):
    cfg = config.DiscoConfig.from_flags(
        _holders(), overrides={"task_suite": "visual_symmetry"}
    )
    self.assertAlmostEqual(cfg.task_null_proportion, 0.9)
    explicit = config.DiscoConfig.from_flags(
        _holders(),
        overrides={
            "task_suite": "visual_symmetry",
            "task_null_proportion": 0.5,
        },
    )
    self.assertAlmostEqual(explicit.task_null_proportion, 0.5)
    self.assertEqual(
        config.DiscoConfig.from_flags(_holders()).task_null_proportion, 0.0
    )


class DefaultDriftTest(absltest.TestCase):
  """Pins the remaining differences between flag, from_flags and dataclass.

  Any new drift (or a fix of an existing one) must update the expected tables
  here deliberately.
  """

  def test_dataclass_defaults_are_paper_values(self):
    cfg = config.DiscoConfig(
        target_model="gemini-3.8-flash",
        seed_instruction="s",
        budget=100,
        alpha=0.05,
    )
    self.assertEqual(
        {
            "kappa": cfg.kappa,
            "e_min": cfg.e_min,
            "e_max": cfg.e_max,
            "enforce_exit_low": cfg.enforce_exit_low,
            "enable_hypothesis_checker": cfg.enable_hypothesis_checker,
            "n_max": cfg.n_max,
            "alpha_calibration_mode": cfg.alpha_calibration_mode,
            "alpha_sweep_min": cfg.alpha_sweep_min,
            "alpha_sweep_max": cfg.alpha_sweep_max,
            "alpha_sweep_steps": cfg.alpha_sweep_steps,
            "show_intra_hypothesis_failures": (
                cfg.show_intra_hypothesis_failures
            ),
            "max_concurrent_hypotheses": cfg.max_concurrent_hypotheses,
            "scrape_bandwidth": cfg.scrape_bandwidth,
        },
        {
            "kappa": 0.1741,
            "e_min": 0.99,
            "e_max": 50.0,
            "enforce_exit_low": True,
            "enable_hypothesis_checker": True,
            "n_max": 3,
            "alpha_calibration_mode": "good_event",
            "alpha_sweep_min": 0.01,
            "alpha_sweep_max": 0.99,
            "alpha_sweep_steps": 40,
            "show_intra_hypothesis_failures": True,
            "max_concurrent_hypotheses": 1,
            "scrape_bandwidth": 4,
        },
    )
    self.assertEqual(flags_lib.flag_defaults()["alpha"], 0.1)

  def test_flag_defaults_vs_dataclass_defaults(self):
    holders = flags_lib.define_disco_flags(prefix="drift_test_")
    cfg = _default_config()
    drift = {}
    for f in dataclasses.fields(holders):
      field = _FLAG_TO_FIELD.get(f.name, f.name)
      if field in _REQUIRED:
        continue
      flag_default = getattr(holders, f.name).default
      config_default = getattr(cfg, field)
      if flag_default != config_default:
        drift[field] = (flag_default, config_default)
    self.assertEqual(
        drift,
        {
            "output_dir": (
                "./disco_runs",
                "",
            ),
        },
    )

  def test_from_flags_defaults_match_dataclass_defaults(self):
    cfg = _default_config()
    from_flags = config.DiscoConfig.from_flags(
        _holders(
            target_model=cfg.target_model,
            seed=cfg.seed_instruction,
            budget=cfg.budget,
            alpha=cfg.alpha,
            name=cfg.experiment_name,
            output_dir=cfg.output_dir,
        ),
        overrides={"kappa": cfg.kappa},
    )
    self.assertEqual(dataclasses.asdict(from_flags), dataclasses.asdict(cfg))


class FlagSpecsTest(absltest.TestCase):

  def test_holders_match_specs(self):
    self.assertEqual(
        [f.name for f in dataclasses.fields(flags_lib.DiscoFlagHolders)],
        list(flag_specs.FLAG_SPECS),
    )

  def test_flag_set_is_the_core_set(self):
    self.assertEqual(
        list(flag_specs.FLAG_SPECS),
        [
            "target_model",
            "seed",
            "budget",
            "alpha",
            "backbone_model",
            "output_dir",
            "name",
            "e_max",
            "chunk_size",
            "alpha_calibration_mode",
            "n_max",
            "num_active_bins",
        ],
    )

  def test_every_spec_field_is_a_config_init_field(self):
    fields = {f.name for f in dataclasses.fields(config.DiscoConfig) if f.init}
    for name, spec in flag_specs.FLAG_SPECS.items():
      self.assertIn(spec.field_name(name), fields, name)

  def test_required_fields_have_flags_or_defaults(self):
    flagged = {
        spec.field_name(name) for name, spec in flag_specs.FLAG_SPECS.items()
    }
    for f in dataclasses.fields(config.DiscoConfig):
      if (
          f.default is dataclasses.MISSING
          and f.default_factory is dataclasses.MISSING
      ):
        self.assertIn(f.name, flagged)

  def test_enum_specs_have_values(self):
    for name, spec in flag_specs.FLAG_SPECS.items():
      self.assertEqual(spec.kind == "enum", bool(spec.enum_values), name)

  def test_flag_defaults_rejects_unknown_overrides(self):
    with self.assertRaisesRegex(TypeError, "default_no_such_flag"):
      flags_lib.flag_defaults(default_no_such_flag=1)
    with self.assertRaisesRegex(TypeError, "seed"):
      flags_lib.flag_defaults(seed="s")
    with self.assertRaisesRegex(TypeError, "default_kappa"):
      flags_lib.flag_defaults(default_kappa=0.5)

  def test_flag_defaults_overrides(self):
    defaults = flags_lib.flag_defaults(default_budget=4, default_name="n")
    self.assertEqual(defaults["budget"], 4)
    self.assertEqual(defaults["name"], "n")
    self.assertEqual(defaults["chunk_size"], 25)


if __name__ == "__main__":
  absltest.main()
