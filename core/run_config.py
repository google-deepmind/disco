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

"""YAML configuration loader for DISCO CLI entrypoints.

Supports launching single runs from a flat YAML file
(`--config=path/to/run.yaml`).
Each top-level key is either

*   the name of a registered command-line flag, in which case the value is
    applied to the flag unless the flag was given explicitly on the command
    line (`CLI > YAML > flag default`), or
*   the name of a field of the entrypoint's config dataclass that has no flag
    of its own (a "YAML-only" setting). `apply_yaml_config` returns these as a
    mapping of field name -> value for the entrypoint to apply with
    `apply_field_overrides` (or `DiscoConfig.from_flags(overrides=...)`).

Entrypoints that prefix their flags accept the prefix on YAML-only keys too
(`flag_prefix`); the DISCO entrypoints (`run_disco`, `eval_disco`) use no
prefix, so their YAML keys are the bare flag and field names.
"""

from collections.abc import Collection, Mapping, Sequence
import dataclasses
import os
from typing import Any, TypeVar

from absl import flags
import yaml

CONFIG_FLAG = flags.DEFINE_string(
    "config",
    "",
    "Optional path to a YAML configuration file mapping flag names (or"
    " YAML-only config field names) to values. Explicit command-line flags"
    " override values in the YAML file.",
)

_ConfigT = TypeVar("_ConfigT")


def config_field_names(config_cls: type[Any]) -> frozenset[str]:
  """Returns the names of the init fields of a config dataclass."""
  return frozenset(f.name for f in dataclasses.fields(config_cls) if f.init)


def resolve_config_path(config_path: str) -> str:
  """Resolves a config path relative to cwd, BUILD_WORKING_DIRECTORY, or package root."""
  if not config_path or os.path.exists(config_path):
    return config_path

  pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  build_wd = os.environ.get("BUILD_WORKING_DIRECTORY", "")
  search_bases = [b for b in (os.getcwd(), build_wd, pkg_root) if b]
  for base in search_bases:
    candidate = os.path.join(base, config_path)
    if os.path.exists(candidate):
      return candidate

  return config_path


def load_yaml_config(config_path: str) -> dict[str, Any]:
  """Loads and validates a flat YAML configuration dictionary from disk."""
  resolved_path = resolve_config_path(config_path)
  if not os.path.exists(resolved_path):
    raise FileNotFoundError(
        f"Config file not found: {config_path!r} (resolved as"
        f" {resolved_path!r})"
    )
  with open(resolved_path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

  if data is None:
    return {}
  if not isinstance(data, Mapping):
    raise ValueError(
        f"YAML config {config_path!r} must be a top-level mapping of"
        f" flag_name -> value, got {type(data).__name__}."
    )
  return {str(k): v for k, v in data.items()}


def _apply_flag_value(flag_obj: flags.Flag, value: Any) -> None:
  """Applies a single YAML value to an absl Flag object."""
  if value is None:
    flag_obj.value = None
    flag_obj.present = True
    return

  if isinstance(flag_obj, flags.MultiFlag):
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
      flag_obj.value = [str(item) for item in value]
    else:
      flag_obj.value = [str(value)]
    flag_obj.present = True
    return

  if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
    flag_obj.value = [str(item) for item in value]
    flag_obj.present = True
    return

  if isinstance(value, bool):
    flag_obj.parse("true" if value else "false")
    return

  flag_obj.parse(value)


def apply_yaml_config(
    config_path: str | None = None,
    flag_values: flags.FlagValues = flags.FLAGS,
    *,
    field_names: Collection[str] = (),
    flag_prefix: str = "",
) -> dict[str, Any]:
  """Applies a YAML configuration file to `flag_values` without overriding CLI flags.

  Args:
    config_path: Path to the YAML file. Defaults to `--config` (`CONFIG_FLAG`).
    flag_values: The `absl.flags.FlagValues` instance to update.
    field_names: Names of config dataclass fields that may be set from YAML even
      though they have no flag. A YAML key names such a field either directly or
      with `flag_prefix` in front.
    flag_prefix: Flag-name prefix of the entrypoint (e.g. `"disco_"`), stripped
      from YAML keys before matching them against `field_names`.

  Returns:
    The YAML-only entries, keyed by field name (prefix stripped), to apply to
    the config object once it is built. Empty if no config path was provided.

  Raises:
    FileNotFoundError: If `config_path` does not exist.
    ValueError: If the YAML file is not a mapping, or a key is neither a
      registered flag nor a name in `field_names`.
  """
  path = CONFIG_FLAG.value if config_path is None else config_path
  if not path:
    return {}

  data = load_yaml_config(path)
  field_overrides: dict[str, Any] = {}
  unknown_keys: list[str] = []
  for name, value in data.items():
    if name == "config":
      continue
    if name in flag_values:
      flag_obj = flag_values[name]
      if not flag_obj.present:
        # Explicit CLI flag takes precedence over YAML config.
        _apply_flag_value(flag_obj, value)
      continue
    field = name.removeprefix(flag_prefix) if flag_prefix else name
    if field in field_names:
      field_overrides[field] = value
    else:
      unknown_keys.append(name)

  if unknown_keys:
    raise ValueError(
        f"Unknown key(s) in YAML config {path!r}: {unknown_keys}. Keys must"
        " name a command-line flag or a config field."
    )
  return field_overrides


def apply_field_overrides(
    config: _ConfigT, overrides: Mapping[str, Any]
) -> _ConfigT:
  """Returns a copy of dataclass `config` with `overrides` applied.

  Values are used as parsed from YAML; an int given for a float field is
  widened and a list given for a tuple field is converted.

  Args:
    config: A dataclass instance.
    overrides: Field name -> value, as returned by `apply_yaml_config`.

  Raises:
    ValueError: If an override names no init field of `config`.
  """
  if not overrides:
    return config
  instance: Any = config
  fields_by_name = {f.name: f for f in dataclasses.fields(instance) if f.init}
  unknown = sorted(name for name in overrides if name not in fields_by_name)
  if unknown:
    raise ValueError(
        f"Unknown config field(s) {unknown} for {type(config).__name__}."
    )
  coerced: dict[str, Any] = {}
  for name, value in overrides.items():
    field_type = fields_by_name[name].type
    if (
        field_type in (float, "float")
        and isinstance(value, int)
        and not isinstance(value, bool)
    ):
      value = float(value)
    elif isinstance(value, list) and isinstance(getattr(instance, name), tuple):
      value = tuple(value)
    coerced[name] = value
  return dataclasses.replace(instance, **coerced)
