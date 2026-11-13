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

"""Golden cases pinning the discovery-baseline task adapters.

Pins, for every adapter class and a representative set of tasks and seeds:

  * the attributes the adapter derives from its task (null rate, labels, ...),
  * every prompt it builds (direct interrogation, red team, looped
    investigator) and its fixed probe configs,
  * `format_empirical_predictions` on fixed rates, and
  * for a few probes and `k_samples` settings, every call `execute_probe`
    makes to the target model (model, contents, config) and to the T2I judge,
    plus the observation it returns.

Target and judge responses come from deterministic fakes keyed on the request
contents, so the golden does not depend on call scheduling.
"""

import asyncio
from collections.abc import Callable
import hashlib
import io
import types as pytypes
from typing import Any
from unittest import mock

import numpy as np
from PIL import Image

from disco import backbone_client as backbone_lib
from disco import standalone_tools
from disco.baselines.adapters import base as adapters_base
from disco.baselines.adapters import registry as adapters_registry
from disco.tasks import data_classes
from disco.tasks import hedging_tasks
from disco.tasks import recipe_tasks
from disco.tasks import t2i_cat_bias_tasks
from disco.tasks import t2i_handedness_bias_tasks
from disco.tasks.visual_symmetry import hypotheses as visual_symmetry_hypotheses

_SEED = "Probe the target model and report where it behaves unusually."

# `k_samples` settings each probe is executed with.
_PROBE_SETTINGS: tuple[int, ...] = (1, 3, 2)

# How many fixed probes per adapter go through execute_probe, plus one config
# with no recognised keys (exercises the defaults).
_NUM_FIXED_PROBES = 2

_RATES: dict[str, float] = {
    "square": 0.95,
    "english_letter": 0.5,
    "soups": 0.05,
    "nutrition_sweeteners": 0.33,
    "writing a letter": 0.2,
}

# Text the fake target returns; picked by a hash of the request.
_TEXT_RESPONSES: tuple[str, ...] = (
    "It is definitely true. Everyone agrees on this point.",
    "It may possibly be the case, although some evidence suggests otherwise.",
    "Disclaimer: I am not a doctor. Perhaps consult a professional.",
    "one two three four five six seven eight nine ten eleven twelve",
    "",
)

# Judge answers; picked by a hash of the generation prompt. With this order the
# first handedness and cat probes match the bias label and the second do not.
_JUDGE_LABELS: tuple[str, ...] = ("Left", "Right", "Black", "Other", "Orange")


def _digest(data: bytes | str) -> int:
  if isinstance(data, str):
    data = data.encode("utf-8")
  return int.from_bytes(hashlib.sha256(data).digest()[:4], "big")


def _sha(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()[:16]


def _pixels(png: bytes) -> bytes:
  """Size and RGB pixels of an encoded image.

  Images are hashed by their pixels: the PNG bytes the adapters encode differ
  across machines (zlib builds), while the pixels do not.

  Args:
    png: Encoded image.

  Returns:
    Dimensions header and raw RGB pixel bytes.
  """
  image = Image.open(io.BytesIO(png)).convert("RGB")
  return f"{image.width}x{image.height}:".encode() + image.tobytes()


def _task_cases() -> dict[str, data_classes.Task]:
  return {
      "visual_symmetry": visual_symmetry_hypotheses.VISUAL_SYMMETRY_TASK,
      "t2i_handedness": next(
          iter(t2i_handedness_bias_tasks.TASKS_BY_NAME.values())
      ),
      "t2i_cat": next(iter(t2i_cat_bias_tasks.TASKS_BY_NAME.values())),
      "recipe_french_long": recipe_tasks.RECIPE_TRANSLATE_FRENCH_LONG_TASK,
      "recipe_english_long": recipe_tasks.RECIPE_TRANSLATE_ENGLISH_LONG_TASK,
      **{f"hedging_{task.key}": task for task in hedging_tasks.HEDGING_TASKS},
  }


def _adapter_variants() -> dict[str, adapters_base.TaskProbeAdapter]:
  """Adapters for every task case and seed setting."""
  variants = {}
  for name, task in _task_cases().items():
    for seed_name, seed in (("taskseed", ""), ("customseed", _SEED)):
      variants[f"{name}/{seed_name}"] = adapters_registry.get_task_adapter(
          task, seed_instruction=seed
      )
  variants["t2i_handedness/taskseed/judge"] = (
      adapters_registry.get_task_adapter(
          _task_cases()["t2i_handedness"], judge_model="gemini-judge-x"
      )
  )
  return variants


_PRIVATE_ATTRS: tuple[str, ...] = (
    "null_prop",
    "is_french_longer",
    "autorater_name",
    "autorater_params",
    "metric",
    "image_save_dir",
    "_bias_category",
    "_modality",
    "_judge_model",
    "_is_handedness",
    "_is_cat_color",
    "_classification_question",
    "_bias_label",
    "_answer_options",
)


def _config(config: Any) -> Any:
  if config is None:
    return None
  return config.model_dump(mode="json", exclude_none=True)


def _contents(contents: Any) -> Any:
  """Summarizes request contents; images become a hash of their pixels."""
  if isinstance(contents, str):
    return contents
  out = []
  for content in contents:
    parts = []
    for part in content.parts:
      if part.text is not None:
        parts.append({"text": part.text})
      elif part.inline_data is not None:
        parts.append({
            "mime_type": part.inline_data.mime_type,
            "sha": _sha(_pixels(part.inline_data.data)),
        })
      else:
        parts.append({"other": repr(part)})
    out.append({"role": content.role, "parts": parts})
  return out


def _mirror_completion(png: bytes) -> Image.Image:
  """Mirrors the left half of a probe image onto its right half."""
  arr = np.array(Image.open(io.BytesIO(png)).convert("RGB"))
  width = arr.shape[1]
  half = width // 2
  arr[:, width - half :] = arr[:, :half][:, ::-1]
  return Image.fromarray(arr)


class FakeModels:
  """Stands in for `client.aio.models`; records every target call."""

  def __init__(self, calls: list[dict[str, Any]]):
    self._calls = calls

  async def generate_content(
      self, model: str, contents: Any, config: Any = None
  ) -> Any:
    """Records a target model call and returns a deterministic fake response."""
    self._calls.append({
        "target_model": model,
        "contents": _contents(contents),
        "config": _config(config),
    })
    if isinstance(contents, str):
      text = _TEXT_RESPONSES[_digest(contents) % len(_TEXT_RESPONSES)]
      return pytypes.SimpleNamespace(text=text)
    images: list[bytes] = []
    for content in contents:
      for part in content.parts:
        if part.inline_data is not None:
          images.append(part.inline_data.data)
    if images:
      # Visual symmetry: a mirror completion for half the probes, the
      # unmodified (half-erased) probe for the rest.
      png = images[0]
      if _digest(_pixels(png)) % 2:
        image = _mirror_completion(png)
      else:
        image = Image.open(io.BytesIO(png)).convert("RGB")
      return pytypes.SimpleNamespace(image=image, text="completed")
    # Text-to-image: an image whose colour depends on the prompt.
    text_parts: list[str] = []
    for content in contents:
      for part in content.parts:
        text_parts.append(part.text or "")
    text = "".join(text_parts)
    shade = _digest(text) % 256
    return pytypes.SimpleNamespace(
        image=Image.new("RGB", (8, 8), (shade, 64, 128)), text=""
    )


class _FakeJudge:
  """Stands in for `standalone_tools.AskQuestionJudge`; records calls."""

  def __init__(self, calls: list[dict[str, Any]], **kwargs: Any):
    self._calls = calls
    self._kwargs = kwargs

  async def judge_one(
      self, idx: int, probe_content: Any, response_content: Any, *, log_fn=None
  ) -> dict[str, Any]:
    """Records a judge call and returns a deterministic label."""
    del log_fn  # Unused.
    probe_text = probe_content.as_text()
    image = b"".join(
        chunk.data
        for chunk in response_content
        if isinstance(chunk.data, bytes)
    )
    self._calls.append({
        "judge": {
            "backbone": self._kwargs["backbone"],
            "question_text": self._kwargs["question_text"],
            "answer_options": self._kwargs["answer_options"],
            "model": self._kwargs["model"],
            "jitter": self._kwargs["jitter"],
        },
        "idx": idx,
        "probe_text": probe_text,
        "response_sha": _sha(_pixels(image)),
    })
    label = _JUDGE_LABELS[_digest(probe_text) % len(_JUDGE_LABELS)]
    return {"judgment": f" {label} "}


async def _execute(
    adapter: adapters_base.TaskProbeAdapter,
    probe: dict[str, Any],
    probe_idx: int,
    k_samples: int,
) -> dict[str, Any]:
  """Executes a single probe on `adapter` with fake target and judge models."""
  calls: list[dict[str, Any]] = []
  client = pytypes.SimpleNamespace(
      aio=pytypes.SimpleNamespace(models=FakeModels(calls))
  )

  def judge_factory(**kwargs: Any) -> _FakeJudge:
    return _FakeJudge(calls, **kwargs)

  with (
      mock.patch.object(standalone_tools, "AskQuestionJudge", judge_factory),
      mock.patch.object(
          backbone_lib, "get_backbone_client", lambda m: f"backbone:{m}"
      ),
  ):
    obs = await adapter.execute_probe(
        client,
        "fake-target-model",
        dict(probe),
        round_idx=2,
        probe_idx=probe_idx,
        k_samples=k_samples,
        log=lambda *args, **kwargs: None,
    )
  return {"calls": calls, "observation": obs.to_dict()}


def _execute_probe_cases(
    adapter: adapters_base.TaskProbeAdapter,
) -> dict[str, Any]:
  """Runs `execute_probe` across representative probes and `k_samples`."""
  probes = adapter.get_fixed_probe_categories()[:_NUM_FIXED_PROBES] + [
      {"unrecognised_key": "x"}
  ]
  result = {}
  for idx, probe in enumerate(probes, 1):
    for k_samples in _PROBE_SETTINGS:
      result[f"probe{idx}/k{k_samples}"] = asyncio.run(
          _execute(adapter, probe, idx, k_samples)
      )
  return result


def _adapter_snapshot(
    adapter: adapters_base.TaskProbeAdapter,
) -> dict[str, Any]:
  fixed = adapter.get_fixed_probe_categories()
  return {
      "class": type(adapter).__name__,
      "suite": adapter.suite,
      "name": adapter.name,
      "seed_instruction": adapter.seed_instruction,
      "is_multimodal": adapter.is_multimodal,
      "task_null_proportion": adapter.task_null_proportion,
      "task_null_pct": adapter.task_null_pct,
      "attrs": {
          attr: getattr(adapter, attr)
          for attr in _PRIVATE_ATTRS
          if hasattr(adapter, attr)
      },
      "direct_interrogation_prompts": (
          adapter.get_direct_interrogation_prompts()
      ),
      "red_team_system_prompt": adapter.get_red_team_system_prompt(),
      "looped_investigator_system_prompt": (
          adapter.get_looped_investigator_system_prompt()
      ),
      "target_generate_config": _config(adapter.target_generate_config(0.0)),
      # The recipe and hedging pools are pinned in full by category_pools.
      "num_fixed_probes": len(fixed),
      "fixed_probes": fixed if len(fixed) <= 300 else fixed[:3],
      "empirical_predictions": adapter.format_empirical_predictions(_RATES),
  }


def adapters_snapshot() -> dict[str, Any]:
  result = {}
  for name, adapter in _adapter_variants().items():
    snapshot = _adapter_snapshot(adapter)
    # Probe execution does not depend on the seed, so run it once per task.
    if name.endswith("/taskseed") or name.endswith("/judge"):
      snapshot["execute_probe"] = _execute_probe_cases(adapter)
    result[name] = snapshot
  return result


# Golden name -> case function, in the same shape as
# stats_golden_cases.ALL_CASES.
ALL_CASES: dict[str, Callable[[], Any]] = {
    "adapters": adapters_snapshot,
}
