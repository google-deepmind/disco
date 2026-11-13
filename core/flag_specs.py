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

"""Command-line flags for `DiscoConfig`, one spec per flag.

`flags_lib.define_disco_flags` registers one flag per entry of `FLAG_SPECS`,
and `DiscoConfig.from_flags` reads them back field by field. A flag's default
is the `DiscoConfig` field default unless the spec sets `flag_default`.

A flag exists only for the settings worth overriding from the command line.
Every other `DiscoConfig` field is set from the `--config` YAML file (see
`core.run_config`).

Required fields (`seed_instruction`, `budget`, `alpha`) have no dataclass
default, so their specs set `flag_default`; `alpha` defaults to the value the
paper runs used. `target_model` and `output_dir` also set one: the flag
supplies a default target and output location.
"""

import dataclasses
from typing import Any, Final, Literal

from disco.models import registry as model_registry


class _FieldDefault:
  """Sentinel: the flag default is the `DiscoConfig` field default."""

  def __repr__(self) -> str:
    return "FIELD_DEFAULT"


FIELD_DEFAULT: Final[Any] = _FieldDefault()

FlagKind = Literal[
    "string", "integer", "float", "boolean", "enum", "multi_string"
]


@dataclasses.dataclass(frozen=True)
class FlagSpec:
  """How one DISCO flag is defined.

  Attributes:
    kind: Which `flags.DEFINE_*` registers the flag.
    help: Help text.
    field: The `DiscoConfig` field the flag sets. Defaults to the flag name.
    enum_values: Allowed values of an enum flag.
    flag_default: The flag's default, if it differs from the field default.
    allow_override_cpp: Passed through to `flags.DEFINE_*`.
  """

  kind: FlagKind
  help: str
  field: str | _FieldDefault = FIELD_DEFAULT
  enum_values: tuple[str, ...] = ()
  flag_default: Any = FIELD_DEFAULT
  allow_override_cpp: bool = False

  def field_name(self, flag_name: str) -> str:
    """Returns the `DiscoConfig` field this flag sets."""
    if isinstance(self.field, _FieldDefault):
      return flag_name
    return self.field


# Flag name (without prefix) -> spec, in definition order.
FLAG_SPECS: Final[dict[str, FlagSpec]] = {
    "target_model": FlagSpec(
        kind="string",
        flag_default=model_registry.DEFAULT_GEMINI_MODEL,
        help="Target model to evaluate",
    ),
    "seed": FlagSpec(
        kind="string",
        field="seed_instruction",
        flag_default="Compare English and French recipe word counts",
        help="Seed instruction for red teaming.",
    ),
    "budget": FlagSpec(
        kind="integer",
        flag_default=200,
        help="Max total queries to send across all rounds.",
    ),
    "alpha": FlagSpec(
        kind="float",
        flag_default=0.1,
        help="Significance level for hypothesis testing.",
    ),
    "backbone_model": FlagSpec(
        kind="string",
        help=(
            "Master Gemini model for all DISCO auditor roles (hypothesis"
            " selection, query generation, reflection, rating, parsing). If"
            " set, other model roles default to this unless explicitly"
            " overridden."
        ),
    ),
    "output_dir": FlagSpec(
        kind="string",
        flag_default="./disco_runs",
        help="Directory to save logs for this experiment.",
    ),
    "name": FlagSpec(
        kind="string",
        field="experiment_name",
        help="Name of this experiment. Will be appended to output dir.",
    ),
    "e_max": FlagSpec(
        kind="float",
        help=(
            "Upper deactivation threshold for e-values. Hypotheses with running"
            " e-value at or above this are deactivated."
        ),
    ),
    "chunk_size": FlagSpec(
        kind="integer",
        help=(
            "Exact number of queries to allocate per hypothesis investigation"
            " round."
        ),
    ),
    "alpha_calibration_mode": FlagSpec(
        kind="enum",
        enum_values=(
            "pre_hoc",
            "post_hoc",
            "good_event",
        ),
        help="Alpha calibration mode for DISCO runner.",
    ),
    "n_max": FlagSpec(
        kind="integer",
        help="Maximum probes per hypothesis before timeout (DISCO runner).",
    ),
    "num_active_bins": FlagSpec(
        kind="integer",
        help=(
            "Number of active bins for coarsened e-processes"
            " (0=unbinned/exact, 1=binary active strip, >1=multi-bin)."
        ),
    ),
}
