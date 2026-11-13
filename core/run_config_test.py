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

"""Tests for core.run_config YAML configuration loader."""

import dataclasses
import os
import tempfile

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized

from disco import eval_disco
from disco import verifier_config
from disco.core import config as config_lib
from disco.core import run_config


@dataclasses.dataclass
class _FakeConfig:
  name: str = "default"
  ratio: float = 0.5
  count: int = 1
  tags: tuple[str, ...] = ()
  derived: str = dataclasses.field(init=False, default="")

  def __post_init__(self):
    self.derived = f"{self.name}:{self.count}"


class RunConfigTest(parameterized.TestCase):

  def _make_flag_values(self) -> flags.FlagValues:
    fv = flags.FlagValues()
    flags.DEFINE_string("str_flag", "default_str", "str", flag_values=fv)
    flags.DEFINE_integer("int_flag", 10, "int", flag_values=fv)
    flags.DEFINE_float("float_flag", 0.5, "float", flag_values=fv)
    flags.DEFINE_boolean("bool_flag", False, "bool", flag_values=fv)
    flags.DEFINE_enum("enum_flag", "a", ["a", "b", "c"], "enum", flag_values=fv)
    flags.DEFINE_list("list_flag", ["x"], "list", flag_values=fv)
    flags.DEFINE_multi_string(
        "multi_str_flag", [], "multi_string", flag_values=fv
    )
    fv.mark_as_parsed()
    return fv

  def _write_yaml(self, yaml_text: str) -> str:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
      f.write(yaml_text)
      path = f.name
    self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
    return path

  def test_apply_yaml_config_sets_all_flag_types(self):
    fv = self._make_flag_values()
    path = self._write_yaml("""
str_flag: "from_yaml"
int_flag: 42
float_flag: 0.25
bool_flag: true
enum_flag: "b"
list_flag:
  - "item1"
  - "item2"
multi_str_flag:
  - "rule 1"
  - "rule 2"
""")

    overrides = run_config.apply_yaml_config(path, flag_values=fv)
    self.assertEmpty(overrides)
    self.assertEqual(fv["str_flag"].value, "from_yaml")
    self.assertEqual(fv["int_flag"].value, 42)
    self.assertEqual(fv["float_flag"].value, 0.25)
    self.assertTrue(fv["bool_flag"].value)
    self.assertEqual(fv["enum_flag"].value, "b")
    self.assertEqual(fv["list_flag"].value, ["item1", "item2"])
    self.assertEqual(fv["multi_str_flag"].value, ["rule 1", "rule 2"])
    self.assertTrue(fv["multi_str_flag"].present)

  def test_cli_flags_override_yaml_config(self):
    fv = self._make_flag_values()
    # Simulate explicit CLI flags for int_flag and bool_flag.
    fv["int_flag"].parse(999)
    fv["bool_flag"].parse("false")
    self.assertTrue(fv["int_flag"].present)
    self.assertTrue(fv["bool_flag"].present)

    path = self._write_yaml("""
str_flag: "from_yaml"
int_flag: 42
bool_flag: true
""")

    run_config.apply_yaml_config(path, flag_values=fv)
    # str_flag was not on CLI -> updated from YAML
    self.assertEqual(fv["str_flag"].value, "from_yaml")
    # int_flag and bool_flag were on CLI -> CLI wins over YAML
    self.assertEqual(fv["int_flag"].value, 999)
    self.assertFalse(fv["bool_flag"].value)

  def test_unknown_key_in_yaml_raises_value_error(self):
    fv = self._make_flag_values()
    path = self._write_yaml("nonexistent_flag: 123\n")
    with self.assertRaisesRegex(ValueError, "Unknown key"):
      run_config.apply_yaml_config(path, flag_values=fv)

  def test_field_names_are_returned_as_overrides_not_applied_to_flags(self):
    fv = self._make_flag_values()
    path = self._write_yaml("""
int_flag: 42
ratio: 0.75
count: 3
""")
    overrides = run_config.apply_yaml_config(
        path, flag_values=fv, field_names=("ratio", "count")
    )
    self.assertEqual(overrides, {"ratio": 0.75, "count": 3})
    self.assertEqual(fv["int_flag"].value, 42)

  def test_flag_prefix_is_stripped_from_field_keys_only(self):
    fv = flags.FlagValues()
    flags.DEFINE_integer("disco_budget", 10, "budget", flag_values=fv)
    fv.mark_as_parsed()
    path = self._write_yaml("""
disco_budget: 50
disco_ratio: 0.1
count: 2
""")
    overrides = run_config.apply_yaml_config(
        path,
        flag_values=fv,
        field_names=("ratio", "count"),
        flag_prefix="disco_",
    )
    self.assertEqual(overrides, {"ratio": 0.1, "count": 2})
    self.assertEqual(fv["disco_budget"].value, 50)

  def test_field_key_that_is_not_a_field_raises(self):
    fv = self._make_flag_values()
    path = self._write_yaml("not_a_field: 1\n")
    with self.assertRaisesRegex(ValueError, "not_a_field"):
      run_config.apply_yaml_config(path, flag_values=fv, field_names=("ratio",))

  def test_missing_config_file_raises_file_not_found(self):
    fv = self._make_flag_values()
    with self.assertRaises(FileNotFoundError):
      run_config.apply_yaml_config(
          "/nonexistent/path/to/config.yaml", flag_values=fv
      )

  def test_config_field_names_lists_init_fields_only(self):
    self.assertEqual(
        run_config.config_field_names(_FakeConfig),
        frozenset({"name", "ratio", "count", "tags"}),
    )

  def test_apply_field_overrides_replaces_and_reruns_post_init(self):
    cfg = _FakeConfig()
    updated = run_config.apply_field_overrides(
        cfg, {"name": "x", "ratio": 1, "count": 7, "tags": ["a", "b"]}
    )
    self.assertEqual(updated.name, "x")
    self.assertEqual(updated.ratio, 1.0)
    self.assertIsInstance(updated.ratio, float)
    self.assertEqual(updated.count, 7)
    self.assertEqual(updated.tags, ("a", "b"))
    self.assertEqual(updated.derived, "x:7")
    # The original is untouched.
    self.assertEqual(cfg.name, "default")

  def test_apply_field_overrides_empty_returns_same_object(self):
    cfg = _FakeConfig()
    self.assertIs(run_config.apply_field_overrides(cfg, {}), cfg)

  def test_apply_field_overrides_unknown_field_raises(self):
    with self.assertRaisesRegex(ValueError, "nope"):
      run_config.apply_field_overrides(_FakeConfig(), {"nope": 1})

  def test_apply_field_overrides_on_verifier_config_fans_out_backbone(self):
    cfg = verifier_config.VerifierConfig(target_model="gemini-3.8-flash")
    updated = run_config.apply_field_overrides(
        cfg, {"backbone_model": "gemini-3.1-flash-image"}
    )
    self.assertEqual(updated.agent_model, "gemini-3.1-flash-image")
    self.assertEqual(updated.rating_model, "gemini-3.1-flash-image")

  def test_apply_field_overrides_rating_model_updates_defaulted_summarizer(
      self,
  ):
    cfg = verifier_config.VerifierConfig(
        target_model="gemini-3.8-flash",
        backbone_model="gemini-2.5-flash",
    )
    updated = run_config.apply_field_overrides(
        cfg, {"rating_model": "gemini-2.5-pro"}
    )
    self.assertEqual(updated.agent_model, "gemini-2.5-flash")
    self.assertEqual(updated.rating_model, "gemini-2.5-pro")
    self.assertEqual(updated.summarizer_model, "gemini-2.5-pro")

  def test_disco_config_from_flags_applies_yaml_only_overrides(self):
    out_dir = self.create_tempdir().full_path
    path = self._write_yaml(f"""
output_dir: "{out_dir}"
budget: 12
# YAML-only DiscoConfig fields (no flag of their own):
log_to_console: false
kappa: 0.3
method: "ebh_baseline"
rating_model: "GEMINI_FLASH_IMAGE"
single_batch_null_prob: 0.33
""")
    with flagsaver.flagsaver():
      for name in list(flags.FLAGS):
        flags.FLAGS[name].present = False
      flags.FLAGS["config"].parse(path)
      cfg = eval_disco._build_disco_config()
    self.assertEqual(cfg.budget, 12)
    self.assertFalse(cfg.log_to_console)
    self.assertEqual(cfg.kappa, 0.3)
    self.assertEqual(cfg.method, "ebh_baseline")
    self.assertTrue(cfg.baseline_mode)
    self.assertEqual(cfg.single_batch_null_prob, 0.33)
    # Model names set from YAML are normalized like flag values.
    self.assertEqual(cfg.rating_model, "gemini-3.1-flash-image")

  def test_disco_config_from_flags_rejects_unknown_override(self):
    with self.assertRaisesRegex(ValueError, "Unknown DiscoConfig field"):
      config_lib.DiscoConfig.from_flags(
          eval_disco.DISCO_FLAGS,
          overrides={"no_such_field": 1},
      )

  def test_yaml_rejects_removed_flag_names(self):
    path = self._write_yaml("disco_budget: 12\n")
    with flagsaver.flagsaver():
      for name in list(flags.FLAGS):
        flags.FLAGS[name].present = False
      flags.FLAGS["config"].parse(path)
      with self.assertRaisesRegex(ValueError, "disco_budget"):
        eval_disco._build_disco_config()

  @parameterized.parameters(
      "configs/recipes_disco.yaml",
      "configs/hedging_disco.yaml",
      "configs/t2i_cat_bias_disco.yaml",
      "configs/t2i_handedness_bias_disco.yaml",
      "configs/visual_symmetry_broad_disco.yaml",
      "configs/visual_symmetry_contrastive_disco.yaml",
  )
  def test_example_disco_yaml_configs_are_valid(self, rel_path: str):
    out_dir = self.create_tempdir().full_path
    with flagsaver.flagsaver():
      for name in list(flags.FLAGS):
        flags.FLAGS[name].present = False
      flags.FLAGS["config"].parse(rel_path)
      flags.FLAGS["output_dir"].parse(out_dir)
      cfg = eval_disco._build_disco_config()
      self.assertEqual(cfg.output_dir, out_dir)
      self.assertGreater(cfg.budget, 0)
      self.assertTrue(cfg.seed_instruction)
      self.assertTrue(cfg.run_precision_check)
      self.assertEqual(cfg.single_batch_num_queries, 100)


if __name__ == "__main__":
  absltest.main()
