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

"""Base class for the discovery baselines' task adapters.

An adapter supplies everything task-specific a discovery baseline needs: the
direct-interrogation questions, the fixed probe taxonomy, the investigator
prompts, how to run and score one probe on the target model, and how to turn
empirical pass rates into claims.
"""

import abc
import asyncio
from collections.abc import Sequence
import dataclasses
from typing import Any

from google.genai import types

from disco.baselines import common
from disco.models import genai_client
from disco.tasks import data_classes


def majority_vote(outcomes: Sequence[bool], k_samples: int) -> bool:
  """Whether a probe passes, given the pass/fail outcome of each sample.

  With more than one sample, a strict majority of the `k_samples` must pass;
  with one, the probe passes iff that sample did.

  Args:
    outcomes: Per-sample outcomes.
    k_samples: Number of samples requested for the probe.

  Returns:
    True if the probe passes, False otherwise.
  """
  return (
      (sum(outcomes) > k_samples / 2)
      if k_samples > 1
      else (outcomes[0] if outcomes else False)
  )


def _short_error(e: BaseException) -> str:
  """`e` on one line, at most ~200 characters, for a failure reason."""
  text = f"{type(e).__name__}: {e}".replace("\n", " ")
  return text if len(text) <= 200 else text[:200] + "..."


def failure_reason_from_error(e: BaseException, prefix: str = "") -> str:
  """Failure reason of a sample whose call raised `e`.

  Args:
    e: The exception.
    prefix: Names the failed step, e.g. "judge error". Defaults to "target call
      failed".

  Returns:
    Formatted failure reason string.
  """
  return f"{prefix or 'target call failed'}: {_short_error(e)}"


EMPTY_RESPONSE_REASON = "empty response"


def first_failure_reason(reasons: Sequence[str]) -> str:
  """The first non-empty reason in `reasons`, or ""."""
  return next((r for r in reasons if r), "")


async def query_text_once(
    client: Any,
    target_model: str,
    prompt_text: str,
    gen_config: types.GenerateContentConfig,
    semaphore: asyncio.Semaphore | None,
    log: common.LogFn,
) -> tuple[str, str]:
  """Sends a text prompt to the target once.

  Args:
    client: google.genai client.
    target_model: Target model name.
    prompt_text: Prompt to send.
    gen_config: Config of the call.
    semaphore: Bounds concurrent target calls, if set.
    log: Log function.

  Returns:
    (text, failure_reason). The sample is valid iff `failure_reason` is "":
    a call that raised after retries, or returned no or only blank text (empty
    or blocked), is not. `text` is "" for an invalid sample.
  """
  try:
    resp = await common.generate_content_with_retry(
        client=client,
        model=target_model,
        contents=prompt_text,
        config=gen_config,
        semaphore=semaphore,
        log_fn=log,
    )
  except common.LLM_CALL_ERRORS as e:
    return "", failure_reason_from_error(e)
  text = resp.text if (resp and hasattr(resp, "text") and resp.text) else ""
  if not text.strip():
    return "", EMPTY_RESPONSE_REASON
  return text, ""


@dataclasses.dataclass(frozen=True)
class TextSamples:
  """The target's answers to `k` samples of one text prompt.

  Attributes:
    texts: Each sample's text, in order; "" for an invalid sample.
    failure_reasons: Why each sample is invalid, aligned with `texts`; "" for a
      valid sample.
  """

  texts: list[str]
  failure_reasons: list[str]

  @property
  def n_samples(self) -> int:
    """Total number of samples requested."""
    return len(self.texts)

  @property
  def valid_texts(self) -> list[str]:
    """Texts of the valid samples, in order."""
    return [t for t, r in zip(self.texts, self.failure_reasons) if not r]

  @property
  def n_valid(self) -> int:
    """Number of valid samples."""
    return len(self.valid_texts)

  @property
  def first_failure_reason(self) -> str:
    """First non-empty failure reason across samples, or empty string."""
    return first_failure_reason(self.failure_reasons)


async def query_text_samples(
    client: Any,
    target_model: str,
    prompt_text: str,
    gen_config: types.GenerateContentConfig,
    k_samples: int,
    semaphore: asyncio.Semaphore | None,
    log: common.LogFn,
) -> TextSamples:
  """Sends a text prompt to the target `k_samples` times, concurrently.

  Args:
    client: google.genai client.
    target_model: Target model name.
    prompt_text: Prompt to send.
    gen_config: Config of each call.
    k_samples: Number of samples.
    semaphore: Bounds concurrent target calls, if set.
    log: Log function.

  Returns:
    Each sample's text and, for an invalid one, why (see `query_text_once`).
  """
  results = await asyncio.gather(*[
      query_text_once(
          client, target_model, prompt_text, gen_config, semaphore, log
      )
      for _ in range(k_samples)
  ])
  return TextSamples(
      texts=[text for text, _ in results],
      failure_reasons=[reason for _, reason in results],
  )


class TaskProbeAdapter(abc.ABC):
  """Abstract adapter defining probe generation, querying, and scoring for a task."""

  def __init__(
      self,
      task: data_classes.Task,
      seed_instruction: str = "",
  ):
    self.task = task
    self.seed_instruction: str = seed_instruction or str(
        task.metadata.get("seed_instruction", "")
    )

  @property
  def suite(self) -> str:
    return self.task.suite or "generic"

  @property
  def name(self) -> str:
    return self.task.name

  @property
  def is_multimodal(self) -> bool:
    return "image" in self.task.metadata.get("modalities", []) or (
        self.suite == "visual_symmetry"
    )

  @property
  def task_null_proportion(self) -> float:
    """Reference rate this task is defined against.

    Read from task metadata so the baseline and DISCO arms are tested against
    the same null proportion. Falls back to 0.75 if `null_proportion` is not
    set in task metadata.
    """
    return float(self.task.metadata.get("null_proportion", 0.75) or 0.75)

  @property
  def task_null_pct(self) -> int:
    """`task_null_proportion` as a whole percentage, for claim text."""
    return int(round(self.task_null_proportion * 100))

  @abc.abstractmethod
  def get_direct_interrogation_prompts(self) -> list[str]:
    """Returns direct textual questions for the Direct Asking baseline."""
    raise NotImplementedError

  @abc.abstractmethod
  def get_fixed_probe_categories(self) -> list[dict[str, Any]]:
    """Returns a list of category configs for the Simple Probing baseline."""
    raise NotImplementedError

  @abc.abstractmethod
  def _build_red_team_system_prompt(self) -> str:
    """Returns the task-specific body of the Red Team investigator prompt.

    Subclasses implement this. Callers must use the concrete
    `get_red_team_system_prompt` below, which appends the shared brief.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def _build_looped_investigator_system_prompt(self) -> str:
    """Returns the task-specific body of the multi-round investigator prompt.

    Subclasses implement this. Callers must use the concrete
    `get_looped_investigator_system_prompt` below.
    """
    raise NotImplementedError

  def _investigation_brief(self) -> str:
    """Formats `seed_instruction` as an investigation brief for the LLM investigator.

    Includes a note clarifying that the baseline investigator proposes probe
    configurations in JSON rather than calling DISCO's data-collection tools
    directly.

    Returns:
      Formatted investigation brief string, or the empty string when
      `seed_instruction` is empty.
    """
    seed = (self.seed_instruction or "").strip()
    if not seed:
      return ""
    return (
        "\n\n--- INVESTIGATION BRIEF ---\n"
        "This brief is given verbatim to every method under comparison,"
        " including DISCO. It defines what is being looked for and the"
        " reference accuracy rate that hypotheses are tested against.\n\n"
        f"{seed}\n\n"
        "Note: you do not call the data-collection tools the brief names."
        " Propose probe configurations in the format described above instead;"
        " the harness renders the probes, queries the target model and scores"
        " the completions for you.\n"
        "--- END BRIEF ---"
    )

  def get_red_team_system_prompt(self) -> str:
    """Red Team investigator prompt, with the shared investigation brief."""
    return self._build_red_team_system_prompt() + self._investigation_brief()

  def get_looped_investigator_system_prompt(self) -> str:
    """Multi-round investigator prompt, with the shared investigation brief."""
    return (
        self._build_looped_investigator_system_prompt()
        + self._investigation_brief()
    )

  def target_generate_config(
      self, temperature: float, system_instruction: str | None = None
  ) -> types.GenerateContentConfig:
    """Returns the config of a probe call to the target model.

    Args:
      temperature: Sampling temperature.
      system_instruction: Optional system instruction for this call (e.g. the
        symmetry canvas color).
    """
    kwargs: dict[str, Any] = {}
    if system_instruction is not None:
      kwargs["system_instruction"] = system_instruction
    return types.GenerateContentConfig(
        safety_settings=genai_client.DEFAULT_SAFETY_SETTINGS,
        temperature=temperature,
        **kwargs,
    )

  @abc.abstractmethod
  async def execute_probe(
      self,
      client: Any,
      target_model: str,
      probe_config: dict[str, Any],
      round_idx: int,
      probe_idx: int,
      k_samples: int = 1,
      semaphore: asyncio.Semaphore | None = None,
      log: common.LogFn = common.default_log,
  ) -> common.GenericProbeObservation:
    """Executes a probe on the target model and scores the observation asynchronously.

    The observation's `n_samples`, `n_valid_samples` and `failure_reason` say
    how many samples came back usable. It is scored and summarized over the
    valid samples only; with none, score is 0, passed is False and the summary
    is `common.no_usable_response_summary`. Callers decide whether to resample
    (`methods.base.run_probe_with_resample`).
    """
    raise NotImplementedError

  @abc.abstractmethod
  def format_empirical_predictions(
      self, empirical_rates: dict[str, float]
  ) -> list[str]:
    """Translates empirical pass rates into natural language predictions."""
    raise NotImplementedError
