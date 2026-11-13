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

"""Typed task specification for DISCO benchmark tasks.

`TaskSpec` replaces the old `Task(name, description, split, suite, metadata)`
record, where every task-specific setting was an untyped `metadata` key. The
settings are now typed fields, and `metadata` is a read-only view that
reconstructs the old dict exactly (same keys, in the same order, with the same
values), because many call sites still read `task.metadata[...]` and the
registry golden pins it.

Which keys the view contains is recorded in `metadata_keys`: a typed field only
shows up in `metadata` if the task had that key before. Fields that are new in
`TaskSpec` (`recommended_seed`, `adapter`) are only listed when they came from a
legacy `metadata` dict, so filling them in from the YAML data does not change
`metadata` or `to_dict()`.

The legacy constructor still works: `TaskSpec(name=..., description=...,
split=..., suite=..., metadata={...})` splits the dict into typed fields and
`extra`, so `data_classes.Task` is an alias of `TaskSpec` and
`dataclasses.replace(task, metadata=...)` keeps working.
"""

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any, Protocol, Self, TypeVar

from absl import logging

_T = TypeVar("_T")


def normalize_key(key: str) -> str:
  """Normalizes a name or alias to lowercase with underscores."""
  return key.strip().lower().replace(" ", "_").replace("-", "_")


# Typed fields that mirror a legacy metadata key of the same name, in the order
# they are listed in `metadata_keys` when a spec is built without one.
METADATA_FIELDS: tuple[str, ...] = (
    "seed_instruction",
    "checker_rules",
    "procedural_autorater",
    "autorater_params",
    "null_proportion",
    "target_model",
    "modality",
    "modalities",
    "recommended_target_models",
    "aliases",
)

# Typed fields with no legacy metadata key. They are excluded from `metadata`
# unless a legacy `metadata` dict supplied them.
SPEC_ONLY_FIELDS: tuple[str, ...] = (
    "recommended_seed",
    "adapter",
)

_TYPED_FIELDS: frozenset[str] = frozenset(METADATA_FIELDS + SPEC_ONLY_FIELDS)


@dataclasses.dataclass(frozen=True, init=False)
class TaskSpec:
  """A single benchmark evaluation task.

  Optional typed fields are `None` when the task does not set them; `None`
  fields are absent from `metadata`.

  Attributes:
    name: Human-readable name of the task.
    description: Behavior or failure mode description.
    split: Data split (e.g., 'iteration', 'evaluation', 'all').
    suite: Name of the benchmark suite this task belongs to (e.g.,
      'visual_symmetry').
    seed_instruction: Domain seed given to DISCO (and baselines) for the task.
    checker_rules: Rules the hypothesis checker enforces.
    procedural_autorater: Name of the procedural autorater scoring responses.
    autorater_params: Keyword parameters for the procedural autorater.
    null_proportion: The task's reference rate: the null that discovered
      hypotheses stating no rate of their own are tested against, and the rate
      baseline claims are phrased around (e.g. the population baseline of a T2I
      bias task).
    target_model: Default target model.
    modality: Generation modality (e.g., 'text_to_image').
    modalities: Input/output modalities (e.g., ['image', 'text']).
    recommended_target_models: Target models the task is designed for.
    aliases: Extra lookup names for the task.
    recommended_seed: Seed recommended for DISCO runs when `seed_instruction` is
      not set (the T2I tasks). Never part of `metadata`, so baselines that read
      `metadata['seed_instruction']` keep getting no seed.
    adapter: Name of the baseline task adapter.
    extra: Remaining legacy metadata keys without a typed field.
    metadata_keys: Keys of the legacy `metadata` view, in order.
  """

  name: str
  description: str
  split: str = "all"
  suite: str = ""
  seed_instruction: str | None = None
  checker_rules: list[str] | None = None
  procedural_autorater: str | None = None
  autorater_params: dict[str, Any] | None = None
  null_proportion: float | None = None
  target_model: str | None = None
  modality: str | None = None
  modalities: list[str] | None = None
  recommended_target_models: list[str] | None = None
  aliases: list[str] | None = None
  recommended_seed: str | None = None
  adapter: str | None = None
  extra: dict[str, Any] = dataclasses.field(default_factory=dict)
  metadata_keys: tuple[str, ...] = dataclasses.field(
      default=(), compare=False, repr=False
  )

  def __init__(
      self,
      name: str,
      description: str,
      split: str = "all",
      suite: str = "",
      metadata: Mapping[str, Any] | None = None,
      *,
      seed_instruction: str | None = None,
      checker_rules: list[str] | None = None,
      procedural_autorater: str | None = None,
      autorater_params: dict[str, Any] | None = None,
      null_proportion: float | None = None,
      target_model: str | None = None,
      modality: str | None = None,
      modalities: list[str] | None = None,
      recommended_target_models: list[str] | None = None,
      aliases: list[str] | None = None,
      recommended_seed: str | None = None,
      adapter: str | None = None,
      extra: Mapping[str, Any] | None = None,
      metadata_keys: Sequence[str] | None = None,
  ):
    """Builds a spec from typed fields, or from a legacy `metadata` dict.

    Args:
      name: See class docstring.
      description: See class docstring.
      split: See class docstring.
      suite: See class docstring.
      metadata: Legacy metadata dict. When given, it alone defines the
        `metadata` view: it overrides every field in `METADATA_FIELDS`, `extra`
        and `metadata_keys` (so `dataclasses.replace(task, metadata=...)`
        works), and overrides a `SPEC_ONLY_FIELDS` field only if it has that
        key.
      seed_instruction: See class docstring.
      checker_rules: See class docstring.
      procedural_autorater: See class docstring.
      autorater_params: See class docstring.
      null_proportion: See class docstring.
      target_model: See class docstring.
      modality: See class docstring.
      modalities: See class docstring.
      recommended_target_models: See class docstring.
      aliases: See class docstring.
      recommended_seed: See class docstring.
      adapter: See class docstring.
      extra: See class docstring. Must not use a typed field's name.
      metadata_keys: See class docstring. Defaults to the set typed
        `METADATA_FIELDS`, then the `extra` keys.

    Raises:
      ValueError: If `extra` repeats a typed field, or `metadata_keys` does
        not match the set fields.
    """
    typed: dict[str, Any] = {
        "seed_instruction": seed_instruction,
        "checker_rules": checker_rules,
        "procedural_autorater": procedural_autorater,
        "autorater_params": autorater_params,
        "null_proportion": null_proportion,
        "target_model": target_model,
        "modality": modality,
        "modalities": modalities,
        "recommended_target_models": recommended_target_models,
        "aliases": aliases,
        "recommended_seed": recommended_seed,
        "adapter": adapter,
    }
    if metadata is not None:
      for field_name in METADATA_FIELDS:
        typed[field_name] = metadata.get(field_name)
      for field_name in SPEC_ONLY_FIELDS:
        if field_name in metadata:
          typed[field_name] = metadata[field_name]
      extra = {k: v for k, v in metadata.items() if k not in _TYPED_FIELDS}
      metadata_keys = tuple(metadata)
    extra = dict(extra or {})
    clashes = sorted(_TYPED_FIELDS.intersection(extra))
    if clashes:
      raise ValueError(f"{name!r}: extra repeats typed fields {clashes}")
    if metadata_keys is None:
      metadata_keys = [
          f for f in METADATA_FIELDS if typed[f] is not None
      ] + list(extra)
    metadata_keys = tuple(metadata_keys)
    unknown = [
        k for k in metadata_keys if k not in _TYPED_FIELDS and k not in extra
    ]
    hidden = [
        k
        for k in [*METADATA_FIELDS, *extra]
        if k not in metadata_keys and (k in extra or typed[k] is not None)
    ]
    if unknown or hidden or len(set(metadata_keys)) != len(metadata_keys):
      raise ValueError(
          f"{name!r}: metadata_keys {list(metadata_keys)} do not match the set"
          f" fields (unknown: {unknown}, missing: {hidden})"
      )
    object.__setattr__(self, "name", name)
    object.__setattr__(self, "description", description)
    object.__setattr__(self, "split", split)
    object.__setattr__(self, "suite", suite)
    for field_name, value in typed.items():
      object.__setattr__(self, field_name, value)
    object.__setattr__(self, "extra", extra)
    object.__setattr__(self, "metadata_keys", metadata_keys)

  def __replace__(
      self, *, metadata: Mapping[str, Any] | None = None, **changes: Any
  ) -> Self:
    """Returns a copy with `changes`; `metadata=` is accepted as in `__init__`.

    Spelled out so that type checkers accept `dataclasses.replace(task,
    metadata=...)`.

    Args:
      metadata: Legacy metadata dict (see `__init__`).
      **changes: Field values to change.
    """
    kwargs: dict[str, Any] = {
        f.name: getattr(self, f.name) for f in dataclasses.fields(self)
    }
    kwargs.update(changes)
    if metadata is not None:
      kwargs["metadata"] = metadata
    return type(self)(**kwargs)

  @property
  def key(self) -> str:
    """Returns a normalized key suitable for indexing and lookup."""
    return normalize_key(self.name)

  @property
  def alias_keys(self) -> tuple[str, ...]:
    """Returns the normalized aliases."""
    return tuple(normalize_key(a) for a in self.aliases or ())

  @property
  def metadata(self) -> dict[str, Any]:
    """Returns the legacy metadata dict, rebuilt from the typed fields.

    A new dict on every call, so mutating it does not change the spec.
    """
    return {
        k: self.extra[k] if k in self.extra else getattr(self, k)
        for k in self.metadata_keys
    }

  def to_dict(self) -> dict[str, Any]:
    """Returns a JSON-serializable dictionary representation of the task."""
    result: dict[str, Any] = {
        "name": self.name,
        "description": self.description,
        "split": self.split,
    }
    if self.suite:
      result["suite"] = self.suite
    metadata = self.metadata
    if metadata:
      result["metadata"] = metadata
    return result


class _HasKey(Protocol):

  @property
  def key(self) -> str:
    ...


_K = TypeVar("_K", bound=_HasKey)


def require(value: _T | None, what: str) -> _T:
  """Returns `value`, raising if the task data left it unset.

  Args:
    value: A typed TaskSpec field.
    what: Description of the field for the error message.

  Raises:
    ValueError: If `value` is None.
  """
  if value is None:
    raise ValueError(f"Task data does not set {what}")
  return value


def index_by_key(
    tasks: Sequence[TaskSpec],
    skip_aliases: frozenset[str] = frozenset(),
) -> dict[str, TaskSpec]:
  """Returns the legacy per-module `TASKS_BY_KEY` index.

  Every task key, in task order, followed by every alias, in task order.

  Args:
    tasks: Tasks to index.
    skip_aliases: Aliases left out of the index.
  """
  index = {t.key: t for t in tasks}
  for task in tasks:
    for alias in task.alias_keys:
      if alias not in skip_aliases:
        index[alias] = task
  return index


def legacy_match(
    query: str,
    index: Mapping[str, _K],
    items: Sequence[_K],
    what: str,
) -> list[_K]:
  """Exact `index` hit, else the legacy substring-of-key fallback.

  Shared by the per-module accessors (`get_hedging_tasks`, ...). Logs a
  warning when only the substring fallback matched.

  Args:
    query: Normalized query.
    index: Exact lookup table (keys and, for some modules, aliases).
    items: Items searched by substring of their key.
    what: Accessor name, for the warning.

  Returns:
    `[index[query]]`, else the items whose key contains `query` (possibly
    empty).
  """
  if query in index:
    return [index[query]]
  matched = [item for item in items if query in item.key]
  if matched:
    logging.warning(
        "%s(%r) matched %s only by substring; use an exact key or alias.",
        what,
        query,
        [item.key for item in matched],
    )
  return matched
