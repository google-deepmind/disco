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

"""Helpers shared by the discovery baselines and their task adapters.

Target-model calls with retry (Gemini through google.genai), the
probe observation record, image helpers, the baseline result, checkpointing,
parsing of investigator output, and investigator calls.
"""

import asyncio
import dataclasses
import io
import json
import os
import random
import re
from typing import Any, Callable

from google.genai import types
from PIL import Image

from disco.core import checkpoint as checkpoint_lib
from disco.models import genai_client
from disco.models import registry as model_registry
from disco.tasks import data_classes
from disco.tasks.visual_symmetry import image_gen as visual_symmetry_image_gen

LogFn = Callable[..., None]
write_json_atomic = checkpoint_lib.write_json_atomic
LLM_CALL_ERRORS = genai_client.LLM_CALL_ERRORS


def default_log(*args: Any, **kwargs: Any) -> None:
  del kwargs
  print(*args)


def save_png(image: Any, path: str, log: LogFn = default_log) -> bool:
  """Writes a PIL image or HxWx3 array to `path`.

  Encodes in memory and writes the bytes to disk. Failures are logged, not
  raised: a missing debug image must not abort a run.

  Args:
    image: PIL Image or numpy array to write.
    path: Destination file path.
    log: Logging callback for warning messages.

  Returns:
    True on success, False if writing failed.
  """
  try:
    pil = image if isinstance(image, Image.Image) else Image.fromarray(image)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    with open(path, "wb") as f:
      f.write(buf.getvalue())
    return True
  except (OSError, ValueError, TypeError) as e:
    log(f"  [Warning] Could not save image {path}: {e}")
    return False


# Attempts `generate_content_with_retry` makes when its caller does not say.
# Set once per run from `--max_retries` (see `set_target_max_retries`).
_DEFAULT_MAX_RETRIES = 10
_target_max_retries = _DEFAULT_MAX_RETRIES


def set_target_max_retries(max_retries: int) -> None:
  """Sets the attempts `generate_content_with_retry` makes by default.

  Args:
    max_retries: Attempts per call, at least 1.
  """
  global _target_max_retries
  if max_retries < 1:
    raise ValueError(f"max_retries must be at least 1, got {max_retries}.")
  _target_max_retries = max_retries


def target_max_retries() -> int:
  """The attempts `generate_content_with_retry` makes by default."""
  return _target_max_retries


# Delay before the first retry when the caller does not say. Set once per run
# from `--base_retry_delay` (see `set_target_base_retry_delay`).
_DEFAULT_BASE_RETRY_DELAY = 3.0
_target_base_retry_delay = _DEFAULT_BASE_RETRY_DELAY


def set_target_base_retry_delay(base_delay: float) -> None:
  """Sets the first-retry delay `generate_content_with_retry` uses by default.

  Args:
    base_delay: Seconds before the first retry; later retries double it (with
      jitter, capped by `max_delay`). At least 0.
  """
  global _target_base_retry_delay
  if base_delay < 0:
    raise ValueError(f"base_delay must be at least 0, got {base_delay}.")
  _target_base_retry_delay = base_delay


def target_base_retry_delay() -> float:
  """The first-retry delay `generate_content_with_retry` uses by default."""
  return _target_base_retry_delay


_NON_RETRYABLE_GENERATION_ERRORS = (
    "Generated Chunk is neither text nor image content",
)


async def generate_content_with_retry(
    client: Any,
    model: str,
    contents: Any,
    config: types.GenerateContentConfig | None = None,
    semaphore: asyncio.Semaphore | None = None,
    log_fn: LogFn = default_log,
    max_retries: int | None = None,
    base_delay: float | None = None,
    max_delay: float = 60.0,
) -> Any:
  """Calls Gemini model with bounded semaphore & exponential backoff.

  Only exceptions are retried here. A call that returns an empty or blocked
  response is the caller's to judge (see `adapters.base.query_text_once`).

  Args:
    client: google.genai client.
    model: Model name.
    contents: Request contents.
    config: Request config.
    semaphore: Bounds concurrent calls, if set.
    log_fn: Log function.
    max_retries: Attempts before the last exception is raised; None means
      `target_max_retries()`.
    base_delay: Delay before the first retry, in seconds; None means
      `target_base_retry_delay()`.
    max_delay: Ceiling on the delay between attempts, in seconds.

  Returns:
    The model response.
  """
  if max_retries is None:
    max_retries = _target_max_retries
  if base_delay is None:
    base_delay = _target_base_retry_delay
  if not model or not model.strip():
    raise ValueError("Model name must not be empty.")

  clean_model = model_registry.normalize_model_address(model)

  for attempt in range(max_retries):
    try:
      if semaphore is not None:
        async with semaphore:
          return await client.aio.models.generate_content(
              model=clean_model,
              contents=contents,
              config=config,
          )
      else:
        return await client.aio.models.generate_content(
            model=clean_model,
            contents=contents,
            config=config,
        )
    except LLM_CALL_ERRORS as e:
      if any(marker in str(e) for marker in _NON_RETRYABLE_GENERATION_ERRORS):
        log_fn(
            "  [GenAI Error] Non-retryable generation failure on attempt"
            f" {attempt + 1}/{max_retries}: {e}"
        )
        raise
      if attempt < max_retries - 1:
        exp_delay = base_delay * (2**attempt)
        jitter = random.uniform(0.8, 1.2)
        delay = min(exp_delay * jitter, max_delay)
        log_fn(
            f"  [GenAI Retry] Attempt {attempt + 1}/{max_retries} failed ({e})."
            f" Retrying in {delay:.1f}s..."
        )
        await asyncio.sleep(delay)
      else:
        log_fn(f"  [GenAI Error] All {max_retries} attempts failed: {e}")
        raise


def no_usable_response_summary(reason: str) -> str:
  """Returns the `response_summary` of an observation with no valid sample.

  Distinct from any valid measurement so neither an investigator reading the
  observation nor a person reading the log mistakes a failed call for a
  result.

  Args:
    reason: Why the samples failed.
  """
  return f"[no usable target response: {reason or 'unknown'}]"


@dataclasses.dataclass
class GenericProbeObservation:
  """Observation from probing a target model on a specific task.

  Attributes:
    round_idx: Round the probe ran in.
    probe_idx: Index of the probe.
    category: Category of the probe.
    prompt_summary: What was sent, as shown to the investigator.
    response_summary: What came back, as shown to the investigator.
    score: Score over the valid samples; 0 when there are none.
    passed: Outcome over the valid samples; False when there are none.
    details: Adapter-specific details.
    n_samples: Samples requested from the target.
    n_valid_samples: Samples that came back usable (no exception, non-blank
      text, an image, a judgment). `score` and `passed` are over these only.
    failure_reason: The first reason a sample failed, or "".
    usable: Whether the observation counts: set by
      `methods.base.run_probe_with_resample`, False when it stayed short of
      usable samples. Methods charge budget for, rate, and show the investigator
      usable observations only.
  """

  round_idx: int
  probe_idx: int
  category: str
  prompt_summary: str
  response_summary: str
  score: float
  passed: bool
  details: dict[str, Any] = dataclasses.field(default_factory=dict)
  n_samples: int = 1
  n_valid_samples: int = 1
  failure_reason: str = ""
  usable: bool = True

  def to_dict(self) -> dict[str, Any]:
    """Serializes the probe observation to a dictionary."""
    return {
        "round_idx": self.round_idx,
        "probe_idx": self.probe_idx,
        "category": self.category,
        "prompt_summary": self.prompt_summary,
        "response_summary": self.response_summary,
        "score": round(self.score, 4),
        "passed": self.passed,
        "details": self.details,
        "n_samples": self.n_samples,
        "n_valid_samples": self.n_valid_samples,
        "failure_reason": self.failure_reason,
        "usable": self.usable,
    }


def extract_image_from_response(response: Any) -> Image.Image | None:
  """Extracts PIL Image from Gemini response candidate inline parts or wrapper."""
  if not response:
    return None
  if hasattr(response, "image") and response.image is not None:
    return response.image
  if hasattr(response, "image_bytes") and response.image_bytes is not None:
    try:
      return Image.open(io.BytesIO(response.image_bytes)).convert("RGB")
    except (OSError, ValueError, TypeError):
      pass
  if not hasattr(response, "candidates") or not response.candidates:
    return None
  for part in response.candidates[0].content.parts:
    if hasattr(part, "inline_data") and part.inline_data:
      return Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
  return None


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class GenericBaselineResult:
  """Container for generic baseline evaluation results."""

  method_name: str
  origin: str
  predictions: list[str]
  structured_hypotheses: list[data_classes.Hypothesis] = dataclasses.field(
      default_factory=list
  )
  reasoning: str = ""
  observations: list[GenericProbeObservation] = dataclasses.field(
      default_factory=list
  )
  total_target_queries: int = 0
  empirical_rates: dict[str, float] = dataclasses.field(default_factory=dict)
  metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

  def to_dict(self) -> dict[str, Any]:
    """Serializes the baseline evaluation result to a dictionary."""
    return {
        "method_name": self.method_name,
        "origin": self.origin,
        "predictions": self.predictions,
        "structured_hypotheses": [
            h.to_dict() for h in self.structured_hypotheses
        ],
        "reasoning": self.reasoning,
        "total_target_queries": self.total_target_queries,
        "empirical_rates": {
            k: round(v, 4) for k, v in self.empirical_rates.items()
        },
        "num_observations": len(self.observations),
        "observations": [o.to_dict() for o in self.observations],
        "metadata": self.metadata,
    }


# ---------------------------------------------------------------------------
# Checkpoint Serialization Helpers
# ---------------------------------------------------------------------------


def obs_from_dict(d: dict[str, Any]) -> GenericProbeObservation:
  """Reconstructs a GenericProbeObservation from a dictionary.

  Checkpoints written before the validity fields existed have none of them;
  their observations read as one valid sample out of one, usable.

  Args:
    d: A dictionary from `GenericProbeObservation.to_dict`.

  Returns:
    Reconstructed `GenericProbeObservation`.
  """
  return GenericProbeObservation(
      round_idx=d.get("round_idx", 1),
      probe_idx=d.get("probe_idx", 0),
      category=d.get("category", ""),
      prompt_summary=d.get("prompt_summary", ""),
      response_summary=d.get("response_summary", ""),
      score=float(d.get("score", 0.0)),
      passed=bool(d.get("passed", False)),
      details=d.get("details", {}),
      n_samples=int(d.get("n_samples", 1)),
      n_valid_samples=int(d.get("n_valid_samples", 1)),
      failure_reason=str(d.get("failure_reason", "")),
      usable=bool(d.get("usable", True)),
  )


def save_intermediate_checkpoint(
    checkpoint_dir: str,
    state: dict[str, Any],
    log: LogFn = default_log,
) -> None:
  """Atomically saves intermediate progress to checkpoint.json.

  A failure is logged, not raised: a lost checkpoint costs a restart its
  progress, but must not abort the run that is making it.

  Args:
    checkpoint_dir: Directory of the checkpoint; "" disables saving.
    state: What to save.
    log: Log function.
  """
  if not checkpoint_dir:
    return
  try:
    # Replaces the previous checkpoint only once the new one is complete.
    write_json_atomic(
        os.path.join(checkpoint_dir, "checkpoint.json"), state, indent=2
    )
  except (OSError, TypeError, ValueError) as e:
    log(
        f"  [Warning] Could not save checkpoint to {checkpoint_dir}:"
        f" {type(e).__name__}: {e}"
    )


def load_intermediate_checkpoint(
    checkpoint_dir: str,
) -> dict[str, Any] | None:
  """Loads intermediate progress from checkpoint.json if present."""
  if not checkpoint_dir:
    return None
  ckpt_path = os.path.join(checkpoint_dir, "checkpoint.json")
  if not os.path.exists(ckpt_path):
    return None
  try:
    with open(ckpt_path, "r", encoding="utf-8") as f:
      return json.load(f)
  except (OSError, json.JSONDecodeError, ValueError, TypeError):
    return None


# ---------------------------------------------------------------------------
# Parsing Helpers
# ---------------------------------------------------------------------------


def parse_predictions(text: str, max_predictions: int = 100) -> list[str]:
  """Parses predictions from <predictions> tags, numbered lists, or bullet points.

  Accumulates multi-line items (e.g., bold headings followed by description
  paragraphs) into a single coherent hypothesis.

  Args:
    text: Raw LLM output containing predictions.
    max_predictions: Maximum number of predictions to return.

  Returns:
    List of parsed prediction strings (up to `max_predictions`).
  """
  match = re.search(r"<predictions>(.*?)</predictions>", text, re.DOTALL)
  content = match.group(1) if match else text

  predictions: list[str] = []
  current_claim_lines: list[str] = []

  for raw_line in content.strip().split("\n"):
    line = raw_line.strip()
    if not line or line.startswith("<") or line.startswith("#"):
      continue

    # Match numbered or bulleted list item start
    # (e.g., '1.', '1)', '1 -', '[1]', '- ', '* ')
    m = re.search(r"^(\d+[.)\s\-]+|\[\d+\]\s*|[-*•]\s+)(.+)", line)
    if m:
      if current_claim_lines:
        full_claim = " ".join(current_claim_lines).strip()
        if full_claim and len(full_claim) > 5 and full_claim not in predictions:
          predictions.append(full_claim)
        current_claim_lines = []
      current_claim_lines.append(m.group(2).strip())
    else:
      # Continuation of current item or unnumbered block
      if current_claim_lines:
        current_claim_lines.append(line)
      elif len(line) > 15 and line not in predictions:
        predictions.append(line)

  if current_claim_lines:
    full_claim = " ".join(current_claim_lines).strip()
    if full_claim and len(full_claim) > 5 and full_claim not in predictions:
      predictions.append(full_claim)

  return predictions[:max_predictions]


def parse_probe_json_list(text: str) -> list[dict[str, Any]]:
  """Resiliently parses probe configurations from LLM text (JSON codeblock, array, or regex objects)."""
  if not text or not text.strip():
    return []

  # 1. Try markdown json code block
  m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
  if m:
    try:
      parsed = json.loads(m.group(1))
      if isinstance(parsed, list):
        return [p for p in parsed if isinstance(p, dict)]
    except (json.JSONDecodeError, ValueError, TypeError):
      pass

  # 2. Try direct json.loads
  try:
    parsed = json.loads(text.strip())
    if isinstance(parsed, list):
      return [p for p in parsed if isinstance(p, dict)]
    elif isinstance(parsed, dict):
      return [parsed]
  except (json.JSONDecodeError, ValueError, TypeError):
    pass

  # 3. Fallback: extract individual JSON objects using regex
  probes: list[dict[str, Any]] = []
  for obj_match in re.finditer(
      r"\{[^{}]*(?:dish_en|dish_fr|shape_type|prompt|category|activity|scene)[^{}]*\}",
      text,
      re.DOTALL,
  ):
    try:
      obj = json.loads(obj_match.group(0))
      if isinstance(obj, dict) and obj not in probes:
        probes.append(obj)
    except (json.JSONDecodeError, ValueError, TypeError):
      continue

  return probes


def build_structured_hypotheses_from_preds(
    predictions: list[str],
    suite_name: str,
    null_proportion: float = 0.75,
) -> list[data_classes.Hypothesis]:
  """Converts string prediction list into structured Hypothesis objects.

  Args:
    predictions: Natural-language claims produced by a baseline.
    suite_name: Task suite these hypotheses belong to.
    null_proportion: Reference rate the claims are tested against. Callers pass
      `adapter.task_null_proportion`; the 0.75 default survives only for suites
      whose task metadata sets no `null_proportion`.

  Returns:
    One structured Hypothesis per prediction.
  """
  hyps: list[data_classes.Hypothesis] = []
  for idx, p in enumerate(predictions):
    p_lower = p.lower()
    cond: dict[str, Any] = {"raw_claim": p}
    # Heuristics for visual symmetry shapes. Every family the probe generator
    # supports must appear here: a hypothesis naming a family that is missing
    # from this list gets no `shape_type` in its condition, so the verifier
    # cannot tell which probes to draw and the hypothesis is untestable.
    # Derived from the generator rather than restated, so the two cannot drift.
    #
    # Longest name first: a prefix match on a shorter family would otherwise
    # shadow a longer one (e.g. `chinese_char` swallowing `complex_chinese`).
    for s in sorted(
        (
            st
            for st in visual_symmetry_image_gen.list_shape_types()
            if st != "dynamic"
        ),
        key=len,
        reverse=True,
    ):
      if s in p_lower or s.replace("_", " ") in p_lower:
        cond["shape_type"] = s
        break

    hyps.append(
        data_classes.Hypothesis(
            name=f"Hypothesis_{idx + 1}",
            claim=p,
            task_suite=suite_name,
            condition=cond,
            null_proportion=null_proportion,
        )
    )
  return hyps


class Checkpointer:
  """Saves and restores one method's intermediate progress.

  Wraps `save_intermediate_checkpoint` and `load_intermediate_checkpoint`,
  tagging each checkpoint with the method that wrote it, so that a method never
  resumes from another method's checkpoint.
  """

  def __init__(
      self, checkpoint_dir: str, method: str, log: LogFn = default_log
  ):
    """Initializes the checkpointer.

    Args:
      checkpoint_dir: Directory of the checkpoint; "" disables checkpointing.
      method: Tag of the method, stored under "method" in the checkpoint.
      log: Where a failed save is reported.
    """
    self._checkpoint_dir = checkpoint_dir
    self._method = method
    self._log = log

  def load(self) -> dict[str, Any] | None:
    """Returns the checkpoint, or None if there is none from this method."""
    if not self._checkpoint_dir:
      return None
    ckpt = load_intermediate_checkpoint(self._checkpoint_dir)
    if ckpt and ckpt.get("method") == self._method:
      return ckpt
    return None

  def save(self, **state: Any) -> None:
    """Saves `state`, after the method tag, if checkpointing is enabled."""
    if self._checkpoint_dir:
      save_intermediate_checkpoint(
          self._checkpoint_dir, {"method": self._method, **state}, self._log
      )


async def ask_investigator(
    client: Any,
    investigator_model: str,
    prompt: str,
    temperature: float,
    semaphore: asyncio.Semaphore | None,
    log: LogFn,
    error_message: str,
) -> str:
  """Sends `prompt` to the investigator.

  Args:
    client: google.genai client.
    investigator_model: Investigator model.
    prompt: The prompt.
    temperature: Sampling temperature.
    semaphore: Bounds concurrent model calls, if set.
    log: Log function.
    error_message: Logged, followed by the error, if the call fails.

  Returns:
    The investigator's answer, or "" if it gave none or the call failed.
  """
  text = ""
  try:
    resp = await generate_content_with_retry(
        client=client,
        model=investigator_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            safety_settings=genai_client.DEFAULT_SAFETY_SETTINGS,
            temperature=temperature,
        ),
        semaphore=semaphore,
        log_fn=log,
    )
    text = resp.text if (resp and hasattr(resp, "text") and resp.text) else ""
  except LLM_CALL_ERRORS as e:
    log(f"{error_message}: {e}")
  return text


async def synthesize(
    client: Any,
    investigator_model: str,
    synthesis_prompt: str,
    max_predictions: int,
    semaphore: asyncio.Semaphore | None,
    log: LogFn,
) -> tuple[str, list[str]]:
  """Has the investigator answer `synthesis_prompt` with predictions.

  Args:
    client: google.genai client.
    investigator_model: Investigator model.
    synthesis_prompt: Prompt asking for predictions.
    max_predictions: Most predictions to keep.
    semaphore: Bounds concurrent model calls, if set.
    log: Log function.

  Returns:
    The investigator's answer ("" if the call failed) and the predictions
    parsed from it.
  """
  pred_text = await ask_investigator(
      client=client,
      investigator_model=investigator_model,
      prompt=synthesis_prompt,
      temperature=0.7,
      semaphore=semaphore,
      log=log,
      error_message="Investigator synthesis error",
  )
  return pred_text, parse_predictions(
      pred_text, max_predictions=max_predictions
  )
