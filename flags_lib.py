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

"""Shared flag definitions for DISCO entry points.

The flags are generated from `core.flag_specs.FLAG_SPECS`; their defaults
come from the `DiscoConfig` field defaults unless a spec overrides them.
"""

from collections.abc import Callable
import dataclasses
from typing import Any

from absl import flags

from disco.core import config as config_lib
from disco.core import flag_specs


@dataclasses.dataclass
class DiscoFlagHolders:
  """Container for registered DISCO flag holders."""

  target_model: flags.FlagHolder[str]
  seed: flags.FlagHolder[str]
  budget: flags.FlagHolder[int]
  alpha: flags.FlagHolder[float]
  backbone_model: flags.FlagHolder[str]
  output_dir: flags.FlagHolder[str]
  name: flags.FlagHolder[str]
  e_max: flags.FlagHolder[float]
  chunk_size: flags.FlagHolder[int]
  alpha_calibration_mode: flags.FlagHolder[str]
  n_max: flags.FlagHolder[int]
  num_active_bins: flags.FlagHolder[int]


_DEFINE: dict[str, Callable[..., flags.FlagHolder[Any]]] = {
    "string": flags.DEFINE_string,
    "integer": flags.DEFINE_integer,
    "float": flags.DEFINE_float,
    "boolean": flags.DEFINE_boolean,
    "multi_string": flags.DEFINE_multi_string,
}


def _field_defaults() -> dict[str, Any]:
  """Returns the default of every `DiscoConfig` field that has one."""
  defaults = {}
  for field in dataclasses.fields(config_lib.DiscoConfig):
    if field.default is not dataclasses.MISSING:
      defaults[field.name] = field.default
    elif field.default_factory is not dataclasses.MISSING:
      defaults[field.name] = field.default_factory()
  return defaults


def flag_defaults(**default_overrides: Any) -> dict[str, Any]:
  """Returns flag name -> default, as `define_disco_flags` would register.

  Args:
    **default_overrides: `default_<flag>=<value>` replaces the default of
      `<flag>`.

  Raises:
    TypeError: If an override does not name a DISCO flag.
  """
  unknown = sorted(
      key
      for key in default_overrides
      if key.removeprefix("default_") not in flag_specs.FLAG_SPECS
      or not key.startswith("default_")
  )
  if unknown:
    raise TypeError(f"Unknown DISCO flag default overrides: {unknown}")

  field_defaults = _field_defaults()
  defaults: dict[str, Any] = {}
  for name, spec in flag_specs.FLAG_SPECS.items():
    if f"default_{name}" in default_overrides:
      default = default_overrides[f"default_{name}"]
    elif spec.flag_default is not flag_specs.FIELD_DEFAULT:
      default = spec.flag_default
    else:
      default = field_defaults[spec.field_name(name)]
    if spec.kind == "multi_string" and default is None:
      default = []
    defaults[name] = default
  return defaults


def define_disco_flags(
    prefix: str = "", **default_overrides: Any
) -> DiscoFlagHolders:
  """Defines the DISCO command-line flags.

  Args:
    prefix: Prefix for every flag name (e.g. "disco_").
    **default_overrides: `default_<flag>=<value>` replaces the default of
      `<flag>` (see `flag_defaults`).

  Returns:
    DiscoFlagHolders containing all registered flag holders.
  """
  defaults = flag_defaults(**default_overrides)
  holders: dict[str, flags.FlagHolder[Any]] = {}
  for name, spec in flag_specs.FLAG_SPECS.items():
    kwargs: dict[str, Any] = {}
    if spec.allow_override_cpp:
      kwargs["allow_override_cpp"] = True
    if spec.kind == "enum":
      holders[name] = flags.DEFINE_enum(
          f"{prefix}{name}",
          defaults[name],
          list(spec.enum_values),
          spec.help,
          **kwargs,
      )
    else:
      holders[name] = _DEFINE[spec.kind](
          f"{prefix}{name}", defaults[name], spec.help, **kwargs
      )
  return DiscoFlagHolders(**holders)
