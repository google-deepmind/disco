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

"""Standalone agent that drives the Gemini API directly with function calling.

Manages conversation history, tool declarations, and tool dispatch.
"""

import asyncio
import datetime
import functools
import io
import json
import os
import random
import threading
from typing import Any, Callable

from google.genai import types
import numpy as np
import PIL.Image

from disco import backbone_client as backbone_lib
from disco import standalone_tools as tools
from disco.agent import scrape as agent_scrape
from disco.models import content as content_api
from disco.models import genai_client
from disco.models import registry as model_registry
from disco.models import target_candidate
from disco.prompting import round_prompts
from disco.tasks.visual_symmetry import eval as visual_symmetry_eval
from disco.tasks.visual_symmetry import hypotheses as visual_symmetry_hypotheses
from disco.tasks.visual_symmetry import image_gen as visual_symmetry_image_gen

LogFn = Callable[..., None]

# Reference accuracy the visual symmetry task is defined against, read from the
# task registry rather than restated. The scoring tool quotes this number back
# to the agent as the null to test against, so it must be the same value the
# verifier uses or the two compute p-values against different references.
_SYMMETRY_NULL_PROPORTION: float = float(
    visual_symmetry_hypotheses.VISUAL_SYMMETRY_TASK.metadata["null_proportion"]
)


def _default_log(*args: Any, **kwargs: Any) -> None:
  del kwargs
  print(*args)


_CANDIDATE_CACHE: dict[str, Any] = {}


def _make_candidate(model_address: str) -> Any:
  """Creates a target model candidate instance."""
  return target_candidate.make_candidate(model_address)


def _get_or_create_candidate(target_model: str) -> Any:
  """Retrieves or instantiates a cached candidate instance."""
  norm_addr = model_registry.normalize_model_address(target_model)
  if norm_addr not in _CANDIDATE_CACHE:
    sep = "&" if "?" in norm_addr else "?"
    if "num_inner_retries" not in norm_addr:
      full_addr_with_options = f"{norm_addr}{sep}num_inner_retries=3"
    else:
      full_addr_with_options = norm_addr
    candidate = _make_candidate(full_addr_with_options)
    _CANDIDATE_CACHE[norm_addr] = candidate
  return _CANDIDATE_CACHE[norm_addr]


def get_or_create_candidate(target_model: str) -> Any:
  """Retrieves or instantiates a cached candidate instance."""
  return _get_or_create_candidate(target_model)


def prewarm_candidate(target_model: str) -> None:
  """Builds and caches the candidate instance for `target_model` ahead of time.

  Args:
    target_model: Target model address, e.g. `gemini-3.1-flash-image`.
  """
  _get_or_create_candidate(target_model)


class StandaloneAgent:
  """Agent that uses Gemini API directly with function calling.

  Manages conversation history, sends prompts, processes tool calls,
  and returns final text responses. Logs all intermediate steps via
  an injectable log function.
  """

  def __init__(
      self,
      agent_model: str,
      target_model: str,
      log_fn: LogFn = _default_log,
      output_dir: str = "",
      initial_scrape_round: int = 0,
      procedural_autorater: str = "",
      allow_continuous_means: bool = True,
      exclude_analyze: bool | None = None,
      rating_model: str = "",
      summarizer_model: str = "",
      scrape_bandwidth: int | None = None,
      agent_thinking_level: str | None = None,
      agent_thinking_budget: int | None = None,
      min_response_fraction: float = 0.0,
  ):
    self._agent_model = agent_model
    self._agent_thinking_level = agent_thinking_level
    self._agent_thinking_budget = agent_thinking_budget
    # `--min_response_fraction`: a text scrape still short of this fraction
    # after the regular backfill waves keeps topping up (see
    # `agent_scrape.scrape_with_backfill`). 0 disables top-up.
    self._min_response_fraction = min_response_fraction
    self._target_model = target_model
    if scrape_bandwidth is not None:
      self._scrape_bandwidth = scrape_bandwidth
    else:
      self._scrape_bandwidth = 20
    self._rating_model = (
        rating_model or f"models/{model_registry.DEFAULT_GEMINI_MODEL}"
    )
    self._summarizer_model = summarizer_model or self._rating_model
    self._log = log_fn
    self._output_dir = output_dir
    self._scrape_round = initial_scrape_round
    self._procedural_autorater = procedural_autorater
    self._allow_continuous_means = allow_continuous_means
    should_exclude_analyze = (
        exclude_analyze
        if exclude_analyze is not None
        else bool(procedural_autorater)
    )
    self._exclude_analyze = should_exclude_analyze
    self._tool_declarations = tools.get_tool_declarations(
        exclude_analyze=should_exclude_analyze,
        allow_continuous_means=allow_continuous_means,
    )
    self._client = genai_client.get_genai_client()
    self._history: list[types.Content] = []
    # Mirror every turn to a JSONL file. Appended, never truncated, so resuming
    # from a checkpoint extends the record rather than erasing it.
    self._transcript_path = (
        os.path.join(output_dir, "agent_transcript.jsonl") if output_dir else ""
    )
    self._transcript_lock = threading.Lock()
    self._last_responses: list[content_api.Response] = []
    # Index into the most recent prompt list for each entry of
    # `_last_responses`. Lets consumers recover pair membership even when some
    # prompts failed to produce a usable response.
    self._last_response_indices: list[int] = []
    self._last_prompt_count: int = 0
    self.all_responses: list[content_api.Response] = []
    self.last_judgments: list[dict[str, Any]] = []
    self.all_judgments: list[dict[str, Any]] = []
    self.last_scalar_values: list[float] = []
    self.last_scrape_status: str = "OK"
    self.last_valid_response_count: int = 0
    self.last_block_reason: str = ""
    # Request/receipt counts of the most recent scrape, and of every scrape
    # this agent has run.
    self.last_scrape_outcome: agent_scrape.ScrapeOutcome | None = None
    self.scrape_history: list[agent_scrape.ScrapeOutcome] = []
    self._stat_test_completed_this_step: bool = False
    self._candidate_cls = _get_or_create_candidate(self._target_model)

    # Visual symmetry state: stores generated test data for scoring.
    self._symmetry_gt_cells: list[np.ndarray] = []
    self._symmetry_prompt_images: list[np.ndarray] = []
    self._symmetry_output_images: list[np.ndarray] = []
    self._symmetry_gt_cells_by_probe: list[np.ndarray] = []
    self._symmetry_output_images_by_probe: list[list[np.ndarray]] = []
    self._symmetry_samples_per_probe: int = 1
    self._symmetry_rows: int = 10
    self._symmetry_cols: int = 10
    # Provenance for the last symmetry round. `_save_responses` compacts out
    # unanswered requests, so row N of `round_summary.json` is not probe N.
    # These two together recover the mapping.
    self._symmetry_probe_index_by_request: list[int] = []
    self._symmetry_answered_request_indices: list[int] = []
    self._symmetry_last_shape_type: str = ""
    # The background the most recent probe batch was rendered on. The scorer
    # extracts ink from the probe and the ground truth relative to this colour,
    # so scoring a navy-background batch as if it were white classifies the
    # whole canvas as ink and silently destroys every criterion.
    self._symmetry_last_bg_color: str = "white"

  @property
  def last_responses(self) -> list[content_api.Response]:
    return self._last_responses

  @property
  def last_response_indices(self) -> list[int]:
    """Prompt-list index each entry of `last_responses` was generated from."""
    return self._last_response_indices

  @property
  def last_prompt_count(self) -> int:
    """Number of prompts in the most recent scrape, including failed slots."""
    return self._last_prompt_count

  @property
  def symmetry_output_images(self) -> list[np.ndarray]:
    return self._symmetry_output_images

  @property
  def last_symmetry_shape_type(self) -> str:
    """Shape family of the most recent symmetry probe batch ("" if none)."""
    return self._symmetry_last_shape_type

  @property
  def scrape_round(self) -> int:
    """Number of scrape folders written so far, including the initial offset."""
    return self._scrape_round

  def reset_symmetry_state(self, *, per_probe: bool = False) -> None:
    """Clears the outputs of earlier symmetry probes before a new batch.

    Args:
      per_probe: Also clear the per-probe images and the scalar scores, as done
        before each arm of a contrastive round.
    """
    self._symmetry_output_images = []
    self.last_judgments = []
    if per_probe:
      self._symmetry_output_images_by_probe = []
      self.last_scalar_values = []

  def set_combined_responses(
      self, responses: list[content_api.Response]
  ) -> None:
    """Replaces the last responses with `responses` from several batches.

    The scrape status becomes "OK" if there is at least one response and
    "FAILED" otherwise. `last_response_indices` is left as is.

    Args:
      responses: Responses collected across the batches of one round.
    """
    self._last_responses = responses
    self.last_scrape_status = "OK" if responses else "FAILED"

  async def run_symmetry_probes(self, args: dict[str, Any]) -> str:
    """Runs the `generate_and_send_symmetry_probes` tool with `args`."""
    return await self._tool_generate_and_send_symmetry_probes(args)

  async def score_symmetry(self, args: dict[str, Any]) -> str:
    """Runs the `score_symmetry_completions` tool with `args`."""
    return await self._tool_score_symmetry_completions(args)

  # Reasoning longer than this threshold (in chars) will be summarized via
  # an LLM call before being stored in judgment records. Shorter reasoning
  # passes through unchanged.
  REASONING_SUMMARY_THRESHOLD: int = 300
  _REASONING_SUMMARY_THRESHOLD: int = REASONING_SUMMARY_THRESHOLD
  _JUDGE_JITTER: float = 1.0

  async def summarize_reasoning(self, reasoning: str) -> str:
    """Summarizes verbose judge reasoning into a concise 1-2 sentence summary.

    Uses a fast Gemini Flash call to compress long chain-of-thought reasoning
    (common in image generation judging) into a compact summary that preserves
    the specific failure pattern observed. Falls back to hard truncation if the
    LLM call fails.

    Args:
      reasoning: The full judge reasoning string.

    Returns:
      A summarized reasoning string (~150 chars), or a truncated fallback.
    """
    if len(reasoning) <= self.REASONING_SUMMARY_THRESHOLD:
      return reasoning

    try:
      backbone = backbone_lib.get_backbone_client(self._summarizer_model)
      summary = await backbone.generate(
          model=self._summarizer_model,
          prompt=round_prompts.reasoning_summary_prompt(reasoning),
          temperature=0.0,
      )
      if summary:
        # Ensure the summary is actually shorter than the original.
        if len(summary) < len(reasoning):
          return summary
    except genai_client.LLM_CALL_ERRORS as e:
      self._log(f"  [Warning] Reasoning summarization failed: {e}")

    # Fallback: hard truncation.
    return reasoning[: self.REASONING_SUMMARY_THRESHOLD]

  _summarize_reasoning = summarize_reasoning

  def _record_transcript(self, kind: str, **fields: Any) -> None:
    """Appends one record to the agent transcript.

    Best-effort and never raises: losing an audit record must not abort a run
    that is otherwise healthy. Failures are surfaced through `_log` rather than
    swallowed silently, so a systematically broken path
    is visible in `disco.log` instead of just producing an empty file.

    Args:
      kind: Record type, e.g. "prompt", "model_turn", "tool_call".
      **fields: Additional JSON-serialisable fields for this record.
    """
    if not self._transcript_path:
      return
    record: dict[str, Any] = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "kind": kind,
        # The scrape round is the join key between this transcript and the
        # `responses_round_NNN/` directories holding the probe images.
        "scrape_round": self._scrape_round,
    }
    record.update(fields)
    try:
      line = json.dumps(record, ensure_ascii=False, default=str)
    except (TypeError, ValueError, OverflowError) as e:
      self._log(f"  [Warning] Could not serialize transcript record: {e}")
      return
    with self._transcript_lock:
      try:
        parent = os.path.dirname(self._transcript_path)
        if parent:
          os.makedirs(parent, exist_ok=True)
        with open(self._transcript_path, "a", encoding="utf-8") as f:
          f.write(line + "\n")
      except OSError as e:
        self._log(f"  [Warning] Transcript write failed: {e}")

  def _save_responses(
      self,
      responses: list[content_api.Response],
      prompts_list: list[str],
  ) -> str:
    """Saves responses to disk and generates a JSON summary."""
    self._scrape_round += 1
    save_dir = os.path.join(
        self._output_dir,
        f"responses_round_{self._scrape_round:03d}",
    )
    os.makedirs(save_dir, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []

    for i, resp in enumerate(responses):
      prompt_text = prompts_list[i] if i < len(prompts_list) else ""
      text = resp.content.as_text(ignore_unsupported_types=True)

      # Save text response.
      txt_path = os.path.join(save_dir, f"response_{i:03d}.txt")
      with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

      # Save image chunks as PNGs.
      img_paths: list[str] = []
      image_chunks = resp.content.as_images(ignore_unsupported_types=True)
      for j, chunk in enumerate(image_chunks):
        if hasattr(chunk, "data") and chunk.data:
          img_name = f"response_{i:03d}_img_{j}.png"
          img_path = os.path.join(save_dir, img_name)
          with open(img_path, "wb") as f:
            f.write(chunk.data)
          img_paths.append(img_name)

      # Always add to the JSON summary.
      row_data: dict[str, Any] = {
          "response_idx": i,
          "round": self._scrape_round,
          "prompt": prompt_text,
          "response_text": text,
          "images": img_paths,
      }
      if self._symmetry_last_shape_type:
        row_data["shape_type"] = self._symmetry_last_shape_type
      summary_rows.append(row_data)

    # Always write structured JSON summary.
    summary_path = os.path.join(save_dir, "round_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
      f.write(json.dumps(summary_rows, indent=2, ensure_ascii=False))
    self._log(f"  [Saved] Responses: {save_dir}")
    self._log(f"  [Summary] {summary_path}")

    return save_dir

  # ── Tool implementations ─────────────────────────────────────────────

  def _record_scrape_outcome(self, outcome: agent_scrape.ScrapeOutcome) -> None:
    self.last_scrape_outcome = outcome
    self.scrape_history.append(outcome)

  async def scrape_responses(
      self,
      prompts_list: list[str],
      candidate_addr: str | None = None,
      chunk_size: int = 20,
      max_retries: int = 4,
      base_delay: float = 2.0,
      max_delay: float = 20.0,
      jitter: float = 1.0,
      bandwidth: int | None = None,
  ) -> list[content_api.Response]:
    """Scrapes responses from a target model in chunks with retries and exponential backoff."""
    if not prompts_list:
      return []

    effective_bandwidth = (
        bandwidth if bandwidth is not None else self._scrape_bandwidth
    )

    target_candidate_addr = candidate_addr or self._target_model
    if model_registry.normalize_model_address(
        target_candidate_addr
    ) != model_registry.normalize_model_address(self._target_model):
      self._log(
          "  [Tool Error] scrape_responses: candidate mismatch. "
          f"Expected '{self._target_model}', got '{target_candidate_addr}'."
      )
      raise ValueError(
          f"You must only test the target model '{self._target_model}'. "
          f"Do not attempt to test other models like '{target_candidate_addr}'."
      )

    candidate_cls = self._candidate_cls

    def _make_task(i: int) -> Callable[[], Any]:
      return functools.partial(
          candidate_cls.respond,
          content_api.Content(prompts_list[i]),
          jitter=jitter,
      )

    def _is_usable(resp: content_api.Response) -> bool:
      text = resp.content.as_text(ignore_unsupported_types=True)
      has_images = bool(resp.content.as_images(ignore_unsupported_types=True))
      return bool(text.strip() or has_images)

    def _mark_failed() -> None:
      self.last_scrape_status = "FAILED"
      self.last_valid_response_count = 0
      self._last_responses = []
      self._last_response_indices = []
      self._last_prompt_count = len(prompts_list)

    # Slots stay positionally aligned with `prompts_list`. Alignment is
    # load-bearing: paired experiments emit queries in strict A/B alternation,
    # so a response that is dropped or appended out of order silently
    # reassigns every later response to the wrong pair and the wrong arm.
    outcome = await agent_scrape.scrape_with_backfill(
        _make_task,
        len(prompts_list),
        is_usable=_is_usable,
        bandwidth=effective_bandwidth,
        labels=agent_scrape.TEXT_LABELS,
        log=self._log,
        chunk_size=chunk_size,
        chunk_retries=max_retries,
        chunk_base_delay=base_delay,
        chunk_max_delay=max_delay,
        on_chunk_failure=_mark_failed,
        min_fraction=self._min_response_fraction,
    )
    self._record_scrape_outcome(outcome)
    slots = outcome.slots
    missing_indices = outcome.missing_indices

    # Compact for the return value, but remember which prompt each surviving
    # response came from so downstream consumers can reconstruct pairs.
    valid_indices = [i for i, resp in enumerate(slots) if resp is not None]
    valid_responses: list[content_api.Response] = [
        resp for resp in slots if resp is not None
    ]
    valid_prompts: list[str] = [prompts_list[i] for i in valid_indices]

    if not valid_responses:
      self._log("  [Scrape Error] All scraped responses were empty or failed.")
      _mark_failed()
      raise RuntimeError(
          "Failed to obtain valid non-empty responses from target model."
      )

    if missing_indices:
      self._log(
          f"  [Scrape Warning] {len(missing_indices)} slot(s) still unfilled"
          f" after {outcome.n_backfill_attempts} backfill attempt(s). Prompt"
          f" indices: {missing_indices[:20]}"
      )

    self.last_scrape_status = "OK"
    self.last_valid_response_count = len(valid_responses)
    self._last_responses = valid_responses
    self._last_response_indices = valid_indices
    self._last_prompt_count = len(prompts_list)
    self.all_responses.extend(valid_responses)

    # Save valid responses to disk.
    self._save_responses(valid_responses, valid_prompts)
    return valid_responses

  async def _tool_generate_responses(self, args: dict[str, Any]) -> str:
    """Scrapes responses from a target model."""
    prompts_list = args["prompts"]
    candidate_addr = args.get("candidate", self._target_model)

    self._log(
        "\n  [Tool] generate_single_turn_responses:"
        f" {len(prompts_list)} prompts,"
        f" candidate='{candidate_addr}'"
    )

    try:
      valid_responses = await self.scrape_responses(
          prompts_list, candidate_addr=candidate_addr
      )
    except Exception as e:  # pylint: disable=broad-exception-caught  # Last-resort tool boundary: serialize scrape failures back to the agent turn.
      self._log(f"  [Scrape Error] Target model scraping failed: {e}")
      return f"Error: Target model scraping failed with exception: {e}"

    dropped = len(prompts_list) - len(valid_responses)
    dropped_note = (
        f" ({dropped} empty response(s) dropped)" if dropped > 0 else ""
    )
    save_dir = os.path.join(
        self._output_dir,
        f"responses_round_{self._scrape_round:03d}",
    )
    parts = [
        f"Generated {len(valid_responses)} valid response(s){dropped_note} from"
        f" '{candidate_addr}' for {len(prompts_list)} prompt(s).\n"
        f"Responses saved to: {save_dir}\n"
    ]
    for i, resp in enumerate(valid_responses):
      text = resp.content.as_text(ignore_unsupported_types=True)
      has_images = bool(resp.content.as_images(ignore_unsupported_types=True))
      display_text = (
          text
          if text.strip()
          else ("[Image generated]" if has_images else "[Empty]")
      )
      preview = (
          display_text[:200] + "..."
          if len(display_text) > 200
          else display_text
      )
      self._log(f"    Response {i+1}: {preview}")
      if len(display_text) > 500:
        display_text = display_text[:500] + "... [truncated]"
      parts.append(f"\n--- Response {i+1} ---\n{display_text}\n")

    result = "\n".join(parts)
    self._log(f"  [Tool Result] {len(valid_responses)} valid responses scraped")
    return result

  async def _tool_analyze_responses(self, args: dict[str, Any]) -> str:
    """Asks a question about each stored response via the backbone."""
    if self._exclude_analyze:
      self._log(
          "  [Tool Blocked] analyze_responses invoked but exclude_analyze=True."
      )
      return (
          "Error: analyze_responses is disabled for this task. Do not use an"
          " LLM judge. Use score_symmetry_completions to score completions via"
          " continuous mask IoU (or run_procedural_autorater for procedural"
          " tasks)."
      )

    question_text = args["question"]
    answer_options = args["answer_options"]

    self._log(
        "\n  [Tool] analyze_responses:"
        f" question='{question_text}',"
        f" options={answer_options}"
    )

    if not self._last_responses:
      return (
          "Error: No responses to analyze."
          " Call generate_single_turn_responses first."
      )

    judge = tools.AskQuestionJudge(
        backbone=backbone_lib.get_backbone_client(self._rating_model),
        question_text=question_text,
        answer_options=answer_options,
        model=self._rating_model,
        jitter=self._JUDGE_JITTER,
    )
    judgments_list = await judge.judge_all(
        self._last_responses,
        log_fn=self._log,
    )

    # Keep each judgment tied to the prompt slot it came from so paired
    # analyses can recover pair membership (see stats_utils.reduce_pairs).
    judgment_indices = self._last_response_indices or list(
        range(len(judgments_list))
    )
    for rec, prompt_index in zip(judgments_list, judgment_indices):
      rec["prompt_index"] = prompt_index

    counts: dict[str, int] = {k: 0 for k in answer_options}
    for rec in judgments_list:
      counts[rec["judgment"]] = counts.get(rec["judgment"], 0) + 1

    # Load existing round_summary.json to retroactively update it with ratings
    # & reasoning.
    save_dir = os.path.join(
        self._output_dir,
        f"responses_round_{self._scrape_round:03d}",
    )
    summary_path = os.path.join(save_dir, "round_summary.json")
    existing_summary: list[dict[str, Any]] = []

    if os.path.exists(summary_path):
      try:
        with open(summary_path, "r", encoding="utf-8") as f:
          existing_summary = json.loads(f.read())
      except (OSError, json.JSONDecodeError, ValueError, TypeError) as e:
        self._log(f"  [Warning] Could not load round summary: {e}")

    for i, rec in enumerate(judgments_list):
      # Update in-place in the round_summary.json — always keep FULL reasoning
      # on disk for post-hoc analysis.
      if i < len(existing_summary):
        existing_summary[i]["judgment"] = rec["judgment"]
        existing_summary[i]["judgment_reasoning"] = rec["reasoning"]

    # Batch-summarize verbose reasoning before storing in judgment records.
    # The summarized versions are used in downstream prompts to avoid token
    # explosion; full reasoning is preserved in round_summary.json on disk.
    summarize_tasks = [
        self.summarize_reasoning(rec["reasoning"]) for rec in judgments_list
    ]
    summarized = await asyncio.gather(*summarize_tasks)
    for rec, summary in zip(judgments_list, summarized):
      rec["reasoning"] = summary

    # Write updated round_summary.json back to disk.
    if existing_summary:
      try:
        with open(summary_path, "w", encoding="utf-8") as f:
          f.write(json.dumps(existing_summary, indent=2, ensure_ascii=False))
        self._log(
            f"  [Updated Summary] {summary_path} with ratings & reasoning"
        )
      except (OSError, TypeError, ValueError) as e:
        self._log(f"  [Error] Failed to save updated summary: {e}")

    total = len(judgments_list)
    parts = [f"Analyzed {total} response(s).\nQuestion: '{question_text}'\n"]
    for ans, count in counts.items():
      pct = count / total * 100 if total > 0 else 0
      parts.append(f"'{ans}': {count} ({pct:.1f}%)")

    self._log(f"  [Tool Result] Judgement counts: {counts}")

    self.last_judgments = judgments_list
    self.all_judgments.extend(judgments_list)

    # Wrap both counts and detailed judgments list so caller can parse them.
    dispatch_result = {
        "summary": "\n".join(parts),
        "counts": counts,
        "judgments": judgments_list,
    }
    return json.dumps(dispatch_result)

  async def _tool_run_procedural_autorater(self, args: dict[str, Any]) -> str:
    """Runs a registered procedural autorater on all stored responses."""
    autorater_name = args.get("autorater_name", "")
    params = args.get("params", {})

    if autorater_name not in tools.PROCEDURAL_AUTORATERS:
      available = list(tools.PROCEDURAL_AUTORATERS.keys())
      return json.dumps({
          "error": (
              f"Unknown autorater '{autorater_name}'. Available: {available}"
          )
      })

    if not self._last_responses:
      return json.dumps({
          "error": (
              "No responses to analyze. Call generate_single_turn_responses"
              " first."
          )
      })

    autorater_fn = tools.PROCEDURAL_AUTORATERS[autorater_name]
    values: list[float] = []

    for resp in self._last_responses:
      text = resp.content.as_text(ignore_unsupported_types=True)
      value = autorater_fn(text, params)
      values.append(value)

    n = len(values)
    mean = sum(values) / n if n > 0 else 0.0
    std = (sum((v - mean) ** 2 for v in values) / n) ** 0.5 if n > 0 else 0.0

    output = {
        "autorater": autorater_name,
        "total_responses": n,
        "values": values,
        "mean": mean,
        "std": std,
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }

    judgments_list: list[dict[str, Any]] = []
    # `_last_response_indices` maps each response back to its slot in the
    # prompt list. Carrying it into the judgment records is what lets paired
    # analyses recover which pair (and which arm) a judgment belongs to.
    indices = self._last_response_indices or list(
        range(len(self._last_responses))
    )
    for idx, (resp, value) in zip(indices, zip(self._last_responses, values)):
      prompt_text = (
          resp.probe.content.as_text(ignore_unsupported_types=True)
          if resp and resp.probe
          else ""
      )
      resp_text = (
          resp.content.as_text(ignore_unsupported_types=True) if resp else ""
      )
      judgments_list.append({
          "prompt": prompt_text,
          "response": resp_text,
          "value": value,
          "judgment": f"Score: {value:.1f}",
          "reasoning": f"{autorater_name} = {value:.1f}",
          "prompt_index": idx,
      })
    self.last_judgments = judgments_list
    self.all_judgments.extend(judgments_list)
    self.last_scalar_values = values

    self._log(
        f"  [Procedural Autorater] {autorater_name}: "
        f"n={n}, mean={mean:.1f}, std={std:.1f}"
    )
    return json.dumps(output)

  # ── Visual symmetry tools ──────────────────────────────────────────

  # Shared canonical prompt so discovery and verification send the exact same
  # text with every probe image.
  SYMMETRY_PROMPT: str = visual_symmetry_hypotheses.SYMMETRY_PROMPT
  _SYMMETRY_PROMPT: str = SYMMETRY_PROMPT

  async def _tool_generate_symmetry_test_image(
      self, args: dict[str, Any]
  ) -> str:
    """Generates a single symmetry test image and stores it."""
    shape_type = args.get("shape_type", "english_letter")
    rows = 10
    cols = 10
    seed = int(args.get("seed", 0))
    bg_color = args.get("bg_color", "white")
    fg_color = args.get("fg_color", "blue")
    letter = args.get("letter", None)

    self._log(
        "\n  [Tool] generate_symmetry_test_image: "
        f"shape={shape_type}, seed={seed}"
    )

    kwargs: dict[str, Any] = {}
    if fg_color:
      kwargs["fg_color"] = fg_color
    if letter:
      kwargs["letter"] = letter

    try:
      gt_cells, prompt_cells, _ = (
          visual_symmetry_image_gen.generate_symmetry_image(
              shape_type=shape_type,
              rows=rows,
              cols=cols,
              bg_color=bg_color,
              seed=seed,
              show_midline=args.get("show_midline", True),
              **kwargs,
          )
      )
    except ValueError as e:
      return json.dumps({"error": str(e)})

    # Store for later use.
    self._symmetry_gt_cells = [gt_cells]
    self._symmetry_prompt_images = [prompt_cells]
    self._symmetry_rows = rows
    self._symmetry_cols = cols

    return json.dumps({
        "shape_type": shape_type,
        "seed": seed,
        "bg_color": bg_color,
        "message": (
            f"Generated 1 symmetry test image ({shape_type}, seed={seed})."
            " Use generate_and_send_symmetry_probes for batch testing, or send"
            " this image to the model manually."
        ),
    })

  async def _tool_generate_and_send_symmetry_probes(
      self, args: dict[str, Any]
  ) -> str:
    """Generates multiple symmetry test images and sends them to the target."""
    shape_type = args.get("shape_type", "english_letter")
    self._symmetry_last_shape_type = shape_type
    rows = 10
    cols = 10
    n_probes = int(args.get("n_probes", 25))
    samples_per_probe = max(1, int(args.get("samples_per_probe", 1)))
    seed_start = int(args.get("seed_start", 0))
    bg_color = args.get("bg_color", "white")
    self._symmetry_last_bg_color = bg_color
    fg_color = args.get("fg_color", None)
    letter = args.get("letter", None)
    size = args.get("size", None)
    radius = args.get("radius", None)
    show_midline = args.get("show_midline", True)
    # Enforce standard neutral reflection prompt without disclosing pattern
    # identity.
    text_prompt = self._SYMMETRY_PROMPT

    self._log(
        "\n  [Tool] generate_and_send_symmetry_probes: "
        f"shape={shape_type}, n={n_probes}, "
        f"samples_per_probe={samples_per_probe}"
    )

    # Generate all test images.
    gt_cells_list: list[np.ndarray] = []
    prompt_images: list[np.ndarray] = []
    kwargs: dict[str, Any] = {"show_midline": show_midline}
    if fg_color:
      kwargs["fg_color"] = fg_color
    if letter:
      kwargs["letter"] = letter
    if size is not None:
      kwargs["size"] = int(size)
    if radius is not None:
      kwargs["radius"] = float(radius)

    for i in range(n_probes):
      seed = seed_start + i
      try:
        gt_cells, prompt_cells, _ = (
            visual_symmetry_image_gen.generate_symmetry_image(
                shape_type=shape_type,
                rows=rows,
                cols=cols,
                bg_color=bg_color,
                seed=seed,
                **kwargs,
            )
        )
      except ValueError as e:
        return json.dumps({"error": str(e), "probes_generated": i})

      gt_cells_list.append(gt_cells)
      prompt_images.append(prompt_cells)

    self._log(
        f"  Generated {n_probes} input probe images. Sending "
        f"{n_probes * samples_per_probe} total requests to target model..."
    )

    # Convert numpy images to PIL for content_api.
    pil_images: list[PIL.Image.Image] = []
    for img_arr in prompt_images:
      pil_img = PIL.Image.fromarray(img_arr)
      pil_images.append(pil_img)

    # Send each image + text prompt to the target model (samples_per_probe
    # times per image).
    candidate_cls = self._candidate_cls
    probes = []
    probe_to_input_idx: list[int] = []
    probe_to_sample_idx: list[int] = []
    for i, pil_img in enumerate(pil_images):
      for s in range(samples_per_probe):
        probe_content = content_api.Content([pil_img, text_prompt])
        probes.append(
            content_api.Probe(
                content=probe_content,
                timestamp=datetime.datetime.now(),
                id=f"{i}_sample_{s}",
            )
        )
        probe_to_input_idx.append(i)
        probe_to_sample_idx.append(s)

    def _extract_output_image(resp: Any) -> np.ndarray | None:
      """Returns the first decodable RGB image in a response, else None.

      This is the single definition of "usable" for a symmetry response. The
      backfill loop and the scoring loop below both go through it, so the set
      of slots we retry can never drift from the set of slots we score.

      Args:
        resp: Response object returned by the target model.

      Returns:
        Decoded RGB image array, or None if no valid image is present.
      """
      if resp is None or not resp.content:
        return None
      image_chunks = resp.content.as_images(ignore_unsupported_types=True)
      if not image_chunks:
        return None
      chunk = image_chunks[0]
      try:
        if hasattr(chunk, "data") and chunk.data:
          pil_out = PIL.Image.open(io.BytesIO(chunk.data)).convert("RGB")
          return np.array(pil_out)
        if hasattr(chunk, "pil_image"):
          return np.array(chunk.pil_image.convert("RGB"))
      except (OSError, ValueError, TypeError, AttributeError):
        return None
      return None

    max_bandwidth = 10
    # Tolerant gather plus in-place backfill: a probe that exhausts
    # `GeminiTargetCandidate.respond`'s own retries leaves an empty slot instead
    # of discarding its siblings, and is re-requested up to MAX_BACKFILL times.
    # Slots stay aligned with `probes` because `probe_to_input_idx` maps a
    # position back to the probe image it answers. A probe the model never
    # answered is missing data, not a failed completion.
    outcome = await agent_scrape.scrape_with_backfill(
        lambda i: functools.partial(candidate_cls.respond, probes[i].content),
        len(probes),
        is_usable=lambda resp: _extract_output_image(resp) is not None,
        bandwidth=max_bandwidth,
        labels=agent_scrape.SYMMETRY_LABELS,
        log=self._log,
    )
    self._record_scrape_outcome(outcome)
    slots = outcome.slots
    missing_indices = outcome.missing_indices
    n_missing_initially = outcome.n_missing_initially
    backfill_attempts = outcome.n_backfill_attempts
    max_backfill = agent_scrape.MAX_BACKFILL
    n_backfilled = outcome.n_backfilled
    if backfill_attempts:
      self._log(
          f"  [Scrape Replenish] Recovered {n_backfilled}/"
          f"{n_missing_initially} response(s) over {backfill_attempts}"
          " attempt(s)."
      )

    if missing_indices:
      self._log(
          f"  [Scrape Warning] {len(missing_indices)} symmetry probe slot(s)"
          f" still unfilled after {max_backfill} backfill attempt(s). They are"
          " recorded as unanswered, NOT as failed completions. Probe indices:"
          f" {missing_indices[:20]}"
      )

    # Extract output images from responses grouped by probe index.
    output_images_by_probe: list[list[np.ndarray]] = [
        [] for _ in range(n_probes)
    ]
    output_images_flat: list[np.ndarray] = []
    valid_gt_flat: list[np.ndarray] = []
    n_empty = 0
    for idx, resp in enumerate(slots):
      i = probe_to_input_idx[idx]
      arr = _extract_output_image(resp)
      if arr is None:
        n_empty += 1
        continue
      output_images_by_probe[i].append(arr)
      output_images_flat.append(arr)
      valid_gt_flat.append(gt_cells_list[i])

    # Store for scoring.
    self._symmetry_gt_cells_by_probe = gt_cells_list
    self._symmetry_output_images_by_probe = output_images_by_probe
    self._symmetry_gt_cells = valid_gt_flat
    self._symmetry_output_images = output_images_flat
    # The probe images are needed for the left-half fidelity term: a completion
    # only counts if the model preserved the half it was handed.
    self._symmetry_prompt_images = list(prompt_images)
    self._symmetry_samples_per_probe = samples_per_probe
    self._symmetry_rows = rows
    self._symmetry_cols = cols

    # Also save responses for standard DISCO agent flow. Compact the slots the
    # same way `scrape_responses` does, and record which probe each
    # survivor came from so downstream consumers are not left reading stale
    # indices from an earlier text-path scrape.
    valid_indices = [i for i, resp in enumerate(slots) if resp is not None]
    valid_responses: list[content_api.Response] = [
        resp for resp in slots if resp is not None
    ]
    self.last_scrape_status = "OK" if valid_responses else "FAILED"
    self.last_valid_response_count = len(valid_responses)
    self._last_responses = valid_responses
    self._last_response_indices = valid_indices
    self._last_prompt_count = len(probes)
    self.all_responses.extend(valid_responses)
    # Needed by `_tool_score_symmetry_completions` to annotate the correct rows
    # of `round_summary.json`: `valid_responses` has been compacted, so row N
    # corresponds to request `valid_indices[N]`, which answers probe
    # `probe_to_input_idx[valid_indices[N]]`.
    self._symmetry_probe_index_by_request = probe_to_input_idx
    self._symmetry_answered_request_indices = valid_indices
    prompts_list = [text_prompt] * len(valid_responses)
    save_dir = self._save_responses(valid_responses, prompts_list)

    # Save input images, ground truth, and text prompt for reproducibility.
    inputs_dir = os.path.join(save_dir, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)

    # Save text prompt.
    with open(
        os.path.join(inputs_dir, "text_prompt.txt"), "w", encoding="utf-8"
    ) as f:
      f.write(text_prompt)

    # Record probe provenance. `_save_responses` numbers files by position in
    # the list it is handed, so compacting out unanswered slots would otherwise
    # make `response_NNN.png` untraceable back to the probe image it answers.
    # `answered_request_indices[k]` is the flat request index that produced
    # `response_{k:03d}`, and `probe_image_index_by_request` maps that on to the
    # probe image. This also preserves which requests went unanswered.
    provenance = {
        "n_probes": n_probes,
        "samples_per_probe": samples_per_probe,
        "total_requests": len(probes),
        "probe_image_index_by_request": probe_to_input_idx,
        "sample_index_by_request": probe_to_sample_idx,
        "answered_request_indices": valid_indices,
        "unanswered_request_indices": missing_indices,
        "n_backfill_attempts": backfill_attempts,
        "n_backfilled": n_backfilled,
    }
    with open(
        os.path.join(inputs_dir, "probe_provenance.json"),
        "w",
        encoding="utf-8",
    ) as f:
      json.dump(provenance, f, indent=2)

    # Save input prompt images (half-erased grids sent to the model).
    for i, img_arr in enumerate(prompt_images):
      pil_img = PIL.Image.fromarray(img_arr)
      img_path = os.path.join(inputs_dir, f"prompt_{i:03d}.png")
      buf = io.BytesIO()
      pil_img.save(buf, format="PNG")
      with open(img_path, "wb") as f:
        f.write(buf.getvalue())

    # Save ground truth cell arrays (full symmetric pattern).
    for i, gt in enumerate(gt_cells_list):
      pil_gt = PIL.Image.fromarray(gt)
      gt_path = os.path.join(inputs_dir, f"ground_truth_{i:03d}.png")
      buf = io.BytesIO()
      pil_gt.save(buf, format="PNG")
      with open(gt_path, "wb") as f:
        f.write(buf.getvalue())

    self._log(
        f"  Saved {len(prompt_images)} input images and "
        f"{len(gt_cells_list)} ground truth images to {inputs_dir}"
    )

    unanswered_note = ""
    if n_empty:
      unanswered_note = (
          f" {n_empty} request(s) returned no image even after"
          f" {max_backfill} retries; those are UNANSWERED (missing data), not"
          " failed completions, and are excluded from scoring rather than"
          " counted against the model."
      )
    result = {
        "n_probes_sent": n_probes,
        "samples_per_probe": samples_per_probe,
        "total_requests": len(probes),
        "n_valid_outputs": len(output_images_flat),
        "n_empty": n_empty,
        "n_backfill_attempts": backfill_attempts,
        "n_backfilled": n_backfilled,
        "shape_type": shape_type,
        "save_dir": save_dir,
        "message": (
            f"Sent {n_probes} symmetry probes with {samples_per_probe} samples"
            f" each for {shape_type} ({len(probes)} requests total). Got"
            f" {len(output_images_flat)} valid image outputs."
            + unanswered_note
            + " Call score_symmetry_completions to evaluate (supports pass@k)."
        ),
    }
    self._log(
        f"  [Tool Result] {len(output_images_flat)}/{len(probes)} valid image"
        " outputs"
    )
    return json.dumps(result)

  async def _tool_score_symmetry_completions(self, args: dict[str, Any]) -> str:
    """Scores stored symmetry completion outputs against ground truth with pass@k support."""
    delta_e_threshold = float(args.get("delta_e_threshold", 10.0))
    max_incorrect = int(args.get("max_incorrect_for_success", 0))
    iou_threshold = float(
        args.get("iou_threshold", visual_symmetry_eval.DEFAULT_IOU_THRESHOLD)
    )
    left_fidelity_threshold = float(
        args.get(
            "left_fidelity_threshold",
            visual_symmetry_eval.DEFAULT_LEFT_FIDELITY_THRESHOLD,
        )
    )
    # `success_criterion` is the agent-facing name declared in the tool schema;
    # `success_mode` is the internal spelling, kept for direct callers.
    # Invalid names are returned as a tool error rather than raised so the
    # agent can correct the criterion name on the next call.
    try:
      success_mode = visual_symmetry_eval.resolve_success_mode(
          args.get("success_criterion") or args.get("success_mode")
      )
    except ValueError as e:
      return json.dumps({"error": str(e)})
    match_tolerance_px = float(
        args.get(
            "match_tolerance_px",
            visual_symmetry_eval.DEFAULT_MATCH_TOLERANCE_PX,
        )
    )
    # Not agent-settable: it must be the colour the probes were actually drawn
    # with, which is recorded when they are generated.
    bg_color = self._symmetry_last_bg_color
    pass_at_k_arg = args.get("pass_at_k", None)

    if (
        not self._symmetry_output_images_by_probe
        or not self._symmetry_gt_cells_by_probe
    ):
      if not self._symmetry_output_images or not self._symmetry_gt_cells:
        return json.dumps({
            "error": (
                "No symmetry outputs to score. "
                "Call generate_and_send_symmetry_probes first."
            )
        })
      output_by_probe = [[img] for img in self._symmetry_output_images]
      gt_by_probe = self._symmetry_gt_cells
    else:
      output_by_probe = self._symmetry_output_images_by_probe
      gt_by_probe = self._symmetry_gt_cells_by_probe

    if pass_at_k_arg is not None and int(pass_at_k_arg) > 0:
      k_eval = int(pass_at_k_arg)
      output_by_probe = [samples[:k_eval] for samples in output_by_probe]
    else:
      k_eval = self._symmetry_samples_per_probe

    rows = self._symmetry_rows
    cols = self._symmetry_cols

    self._log(
        "\n  [Tool] score_symmetry_completions: "
        f"scoring {len(output_by_probe)} input probes (pass@{k_eval}), "
        f"threshold={iou_threshold}, tolerance={match_tolerance_px}px, "
        f"bg={bg_color}"
    )

    # `preservation` raises inside the scorer when no probe images were stored,
    # because measuring "did it keep the left half" against nothing would pass
    # every probe. Surface that as a tool error for the same reason as above:
    # the agent can re-probe or pick another criterion, whereas an uncaught
    # exception here aborts the whole discovery run.
    try:
      eval_result = visual_symmetry_eval.score_symmetry_completions_pass_at_k(
          output_images_by_probe=output_by_probe,
          ground_truth_cells_by_probe=gt_by_probe,
          rows=rows,
          cols=cols,
          delta_e_threshold=delta_e_threshold,
          max_incorrect=max_incorrect,
          iou_threshold=iou_threshold,
          probe_images_by_probe=self._symmetry_prompt_images or None,
          left_fidelity_threshold=left_fidelity_threshold,
          success_mode=success_mode,
          match_tolerance_px=match_tolerance_px,
          bg_color=bg_color,
      )
    except ValueError as e:
      return json.dumps({"error": str(e)})

    # Log every criterion, not just the selected one, so the run log itself
    # records whether the headline number depends on the choice of criterion.
    def _fmt(v: float | None) -> str:
      return "n/a" if v is None else f"{v:.2%}"

    self._log(
        f"  [Criteria] mode={success_mode} "
        f"self_mirror={_fmt(eval_result.get('rate_self_mirror'))} "
        f"left_fidelity={_fmt(eval_result.get('rate_left_fidelity'))} "
        f"ground_truth={_fmt(eval_result.get('rate_ground_truth'))} "
        f"agreement={_fmt(eval_result.get('criterion_agreement'))}"
    )

    # Store judgments per probe for downstream DISCO hypothesis statistical
    # tests.
    # Probes with no returned completion are missing data, not failures, so they
    # must not enter the Bernoulli sequence the statistical test consumes.
    judgments_list: list[dict[str, Any]] = []
    for probe in eval_result["per_probe_results"]:
      if not probe.get("valid", True):
        continue
      is_pass = probe["passed"]
      mean_iou = sum(s.get("iou", 0.0) for s in probe["sample_scores"]) / max(
          1, len(probe["sample_scores"])
      )
      judgments_list.append({
          "prompt": f"symmetry_probe_{probe['probe_idx']}",
          "response": (
              f"pass@{eval_result['k']}={is_pass}"
              f" ({probe['n_samples_passed']}/{probe['n_samples']} samples"
              f" passed, mean_iou={mean_iou:.3f})"
          ),
          "value": 1.0 if is_pass else 0.0,
          "judgment": "Success" if is_pass else "Failure",
          "reasoning": (
              f"pass@{eval_result['k']}={'Success' if is_pass else 'Failure'}:"
              f" {probe['n_samples_passed']}/{probe['n_samples']} completions"
              f" succeeded (IoU >= {iou_threshold})."
          ),
      })
    n_invalid = eval_result.get("n_probes_invalid", 0)
    if n_invalid:
      self._log(
          f"  [Warning] {n_invalid} probe(s) returned no completion and were"
          " excluded from scoring as missing data (not counted as failures)."
      )
    self.last_judgments = judgments_list
    self.all_judgments.extend(judgments_list)
    self.last_scalar_values = [
        1.0 if p.get("passed", False) else 0.0
        for p in eval_result.get("per_probe_results", [])
        if p.get("valid", True)
    ]

    # Retroactively annotate round_summary.json with the algorithmic scores.
    #
    # Row N of the summary is NOT probe N. `_save_responses` is handed the
    # compacted `valid_responses` list, so row N corresponds to request
    # `answered[N]`, which answers probe `probe_by_request[answered[N]]`. With
    # pass@k there are also several requests per probe. Indexing the summary by
    # probe would therefore attribute the wrong pass/IoU to every row after the
    # first unanswered request -- silently, since the numbers still look
    # plausible. The statistics are unaffected (they come from
    # `per_probe_results` directly), but the saved artifact is what anyone
    # debugging a run reads, so it has to agree.
    summary_path = os.path.join(
        self._output_dir,
        f"responses_round_{self._scrape_round:03d}",
        "round_summary.json",
    )
    if os.path.exists(summary_path):
      try:
        with open(summary_path, "r", encoding="utf-8") as f:
          summary_data = json.load(f)

        per_probe = eval_result.get("per_probe_results", [])
        probe_by_request = self._symmetry_probe_index_by_request
        answered = self._symmetry_answered_request_indices

        # `sample_scores` holds only the samples that came back, in request
        # order, so a row's rank among its probe's answered requests is its
        # index into that list.
        samples_seen: dict[int, int] = {}
        for row_idx, req_idx in enumerate(answered):
          if row_idx >= len(summary_data):
            break
          if req_idx >= len(probe_by_request):
            continue
          p_idx = probe_by_request[req_idx]
          sample_rank = samples_seen.get(p_idx, 0)
          samples_seen[p_idx] = sample_rank + 1
          if p_idx >= len(per_probe):
            continue
          probe = per_probe[p_idx]
          row = summary_data[row_idx]
          row["probe_idx"] = p_idx
          row["sample_idx"] = sample_rank
          row["passed"] = probe.get("passed")
          sample_scores = probe.get("sample_scores", [])
          if sample_rank < len(sample_scores):
            sample = sample_scores[sample_rank]
            row["iou"] = sample.get("iou", 0.0)
            # Persist every criterion, not just the selected one.
            for key in (
                "iou_self_mirror",
                "left_fidelity",
                "iou_vs_ground_truth",
                "success_self_mirror",
                "success_left_fidelity",
                "success_ground_truth",
            ):
              if sample.get(key) is not None:
                row[key] = sample[key]
          if self._symmetry_last_shape_type:
            row["shape_type"] = self._symmetry_last_shape_type

        with open(summary_path, "w", encoding="utf-8") as f:
          json.dump(summary_data, f, indent=2)
      except (
          OSError,
          json.JSONDecodeError,
          ValueError,
          TypeError,
          KeyError,
      ) as e:
        self._log(f"  [Warning] Could not annotate {summary_path}: {e}")

    # The row annotation above is lossy by construction: unanswered probes have
    # no row to attach to, and it keeps one sample per row. Write the scorer's
    # own output verbatim alongside it so the full per-probe, per-sample record
    # -- thresholds, validity flags and all three criteria -- survives.
    scoring_path = os.path.join(
        self._output_dir,
        f"responses_round_{self._scrape_round:03d}",
        "scoring_results.json",
    )
    try:
      with open(scoring_path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=2, default=str)
    except (OSError, TypeError, ValueError) as e:
      self._log(f"  [Warning] Could not write {scoring_path}: {e}")

    n_valid = eval_result.get("n_probes_valid", eval_result["n_probes"])
    if eval_result.get("n_probes_invalid", 0):
      invalid_note = (
          f" {eval_result['n_probes_invalid']} probe(s) returned no"
          " completion and are excluded as missing data; do NOT count"
          " them as failures."
      )
    else:
      invalid_note = ""
    output = {
        "n_probes": eval_result["n_probes"],
        "n_probes_valid": n_valid,
        "n_probes_invalid": eval_result.get("n_probes_invalid", 0),
        "k": eval_result["k"],
        "n_probes_passed": eval_result["n_probes_passed"],
        "n_probes_failed": n_valid - eval_result["n_probes_passed"],
        "pass_at_k_rate": round(eval_result["pass_at_k_rate"], 4),
        "total_samples": eval_result["total_samples"],
        "total_samples_passed": eval_result["total_samples_passed"],
        "pass_at_1_rate": round(eval_result["pass_at_1_rate"], 4),
        "delta_e_threshold": delta_e_threshold,
        "max_incorrect_for_success": max_incorrect,
        "per_probe": eval_result["per_probe_results"],
        "message": (
            f"Scored {n_valid} visual symmetry completions"
            " algorithmically against ground truth. Results:"
            f" {eval_result['n_probes_passed']}/{n_valid} passed"
            f" (rate={eval_result['pass_at_k_rate']:.2%})."
            + invalid_note
            + " In Step 5, report this using"
            " `run_proportion_z_test(n_successes="
            f"{eval_result['n_probes_passed']}, n_obs={n_valid}, "
            # Read from the task definition so the agent and the verifier
            # test against the same reference null proportion.
            f"null_proportion={_SYMMETRY_NULL_PROPORTION:.2f},"
            " alternative='larger'|'smaller')`."
        ),
    }

    self._log(
        f"  [Tool Result] pass@{eval_result['k']}:"
        f" {eval_result['n_probes_passed']}/{n_valid} probes"
        f" passed (rate={eval_result['pass_at_k_rate']:.2%}),"
        f" pass@1={eval_result['pass_at_1_rate']:.2%}"
    )
    return json.dumps(output)

  # ── Dispatch ─────────────────────────────────────────────────────────

  async def _dispatch_tool(self, name: str, args: dict[str, Any]) -> str:
    """Routes a function call to the right implementation."""
    # Guard against duplicate statistical tests in the same step.
    if name in tools.STAT_TOOLS and self._stat_test_completed_this_step:
      self._log(
          f"  [Tool Blocked] Duplicate statistical test '{name}' blocked."
      )
      return (
          "Error: A statistical test has already been successfully computed for"
          " this step. You are NOT allowed to run multiple statistical tests or"
          " recalculate with different parameters in the same round. Please"
          " provide your final response text now reporting the p-value and"
          " result from your completed test."
      )

    # Guard against late scraping / measurement after statistical testing.
    if self._stat_test_completed_this_step and name in {
        "generate_single_turn_responses",
        "analyze_responses",
        "run_procedural_autorater",
        "generate_symmetry_test_image",
        "generate_and_send_symmetry_probes",
        "score_symmetry_completions",
    }:
      self._log(
          f"  [Tool Blocked] Tool '{name}' blocked after statistical test"
          " completion."
      )
      return (
          "Error: Statistical evaluation has already completed for this step."
          " Do NOT execute additional scraping or measurement tools. Provide"
          " your final response text now."
      )

    async_handlers: dict[str, Any] = {
        "generate_single_turn_responses": self._tool_generate_responses,
        "analyze_responses": self._tool_analyze_responses,
        "run_procedural_autorater": self._tool_run_procedural_autorater,
        "generate_symmetry_test_image": self._tool_generate_symmetry_test_image,
        "generate_and_send_symmetry_probes": (
            self._tool_generate_and_send_symmetry_probes
        ),
        "score_symmetry_completions": self._tool_score_symmetry_completions,
    }
    try:
      if name in async_handlers:
        return str(await async_handlers[name](args))
      elif name in tools.SYNC_TOOL_DISPATCH:
        args_str = json.dumps(args, default=str)
        self._log(f"\n  [Tool] {name}: {args_str}")
        result = str(tools.SYNC_TOOL_DISPATCH[name](args))
        self._log(f"  [Tool Result] {result}")
        if name in tools.STAT_TOOLS:
          # Lock only if the calculation was valid (not a syntax/parameter
          # error)
          if "Error:" not in result:
            self._stat_test_completed_this_step = True
        return result
      else:
        return f"Error: Unknown tool '{name}'."
    except Exception as e:  # pylint: disable=broad-exception-caught  # Last-resort tool boundary: return tool execution errors to the agent instead of crashing the session.
      self._log(f"  [Tool Error] {name}: {e}")
      return f"Error executing tool '{name}': {e}"

  def clear_history(self) -> None:
    """Clears conversation history to prevent quadratic token accumulation."""
    # Recorded because a clear is not a no-op for interpretation: every turn
    # after it was produced without sight of the turns before it.
    self._record_transcript("clear_history", turns_dropped=len(self._history))
    self._history = []

  # ── Main send loop ──────────────────────────────────────────────────

  async def send(self, prompt_text: str, enable_tools: bool = True) -> str:
    """Sends a prompt and handles multi-turn tool calling.

    Args:
      prompt_text: The user's prompt text.
      enable_tools: Whether tool declarations should be provided for this turn.

    Returns:
      The model's final text response.
    """
    self.last_block_reason = ""
    self._stat_test_completed_this_step = False
    self._history.append(
        types.Content(role="user", parts=[types.Part(text=prompt_text)])
    )
    self._record_transcript(
        "prompt",
        text=prompt_text,
        enable_tools=enable_tools,
        agent_model=self._agent_model,
        # A cleared history means the agent is answering this prompt without
        # the earlier turns, which changes how its answer should be read.
        history_turns_before=len(self._history) - 1,
    )

    max_tool_rounds = 10 if enable_tools else 1
    for round_idx in range(max_tool_rounds):

      @content_api.retry(num_retries=5)
      async def _generate_with_retry():
        if enable_tools:
          tools_config = [
              types.Tool(function_declarations=self._tool_declarations)
          ]
        else:
          tools_config = None
        thinking_config = None
        if self._agent_thinking_level:
          if self._agent_thinking_level.upper() == "OFF":
            thinking_config = types.ThinkingConfig(thinking_budget=0)
          else:
            thinking_config = types.ThinkingConfig(
                thinking_level=self._agent_thinking_level,
            )
        elif self._agent_thinking_budget is not None:
          thinking_config = types.ThinkingConfig(
              thinking_budget=self._agent_thinking_budget,
          )
        return await self._client.aio.models.generate_content(
            model=self._agent_model,
            contents=self._history,
            config=types.GenerateContentConfig(
                tools=tools_config,
                safety_settings=list(genai_client.DEFAULT_SAFETY_SETTINGS),
                temperature=1.0,
                thinking_config=thinking_config,
            ),
        )

      response = await _generate_with_retry()

      if not response.candidates:
        block_reason = ""
        block_message = ""
        if hasattr(response, "prompt_feedback") and response.prompt_feedback:
          pf = response.prompt_feedback
          if hasattr(pf, "block_reason") and pf.block_reason:
            block_reason = getattr(
                pf.block_reason, "value", str(pf.block_reason)
            )
          if hasattr(pf, "block_reason_message") and pf.block_reason_message:
            block_message = str(pf.block_reason_message)

        block_info = block_reason or "unknown"
        if block_message:
          block_info += f": {block_message}"

        self._log(
            f"  [Agent] No candidates in response (block_reason={block_info})"
        )
        self.last_block_reason = block_info
        self._record_transcript(
            "blocked",
            tool_round=round_idx + 1,
            block_reason=block_info,
        )
        return "[No response from model]"

      candidate = response.candidates[0]
      parts = (
          list(candidate.content.parts)
          if candidate.content and candidate.content.parts
          else []
      )
      self._history.append(types.Content(role="model", parts=parts))
      # Separate the thinking parts from the answer: the reasoning is the part
      # worth reviewing when a hypothesis looks wrong, and it is discarded
      # everywhere else in the pipeline.
      thought_text = "\n".join(
          p.text for p in parts if p.text and getattr(p, "thought", False)
      )
      answer_text = "\n".join(
          p.text for p in parts if p.text and not getattr(p, "thought", False)
      )
      self._record_transcript(
          "model_turn",
          tool_round=round_idx + 1,
          text=answer_text,
          thought_text=thought_text,
          function_calls=[
              {
                  "name": p.function_call.name,
                  "args": p.function_call.args,
              }
              for p in parts
              if p.function_call
          ],
          finish_reason=str(getattr(candidate, "finish_reason", "")),
      )

      fn_calls = [p for p in parts if p.function_call]
      if not fn_calls:
        text_parts = [p.text for p in parts if p.text]
        final_text = "\n".join(text_parts) if text_parts else ""
        return final_text

      # Log and execute tool calls.
      self._log(
          f"  [Agent] Tool round {round_idx + 1}: {len(fn_calls)} call(s)"
      )
      fn_responses = []
      has_error = False
      for part in fn_calls:
        fc = part.function_call
        if fc is None:
          continue
        fc_args: dict[str, Any] = fc.args if isinstance(fc.args, dict) else {}
        fc_name: str = fc.name or ""
        result = await self._dispatch_tool(fc_name, fc_args)
        call_failed = isinstance(result, str) and (
            result.startswith("Error:")
            or result.startswith("Error executing tool")
        )
        # `has_error` is loop-wide and drives the backoff below; the transcript
        # needs to attribute the failure to the individual call.
        has_error = has_error or call_failed
        self._record_transcript(
            "tool_call",
            tool_round=round_idx + 1,
            name=fc_name,
            args=fc_args,
            # Untruncated on purpose. The tool result is the evidence the agent
            # conditioned its next move on; a 1000-character prefix of it is
            # not enough to reconstruct why it did what it did.
            result=result,
            is_error=call_failed,
        )
        fn_responses.append(
            types.Part(
                function_response=types.FunctionResponse(
                    name=fc_name,
                    response={"result": result},
                )
            )
        )

      self._history.append(types.Content(role="user", parts=fn_responses))

      if has_error and round_idx < max_tool_rounds - 1:
        backoff_delay = min(4.0 * (1.5**round_idx), 30.0) + random.uniform(
            0.5, 2.5
        )
        self._log(
            f"  [Agent] Tool error in round {round_idx + 1}. Backing off"
            f" {backoff_delay:.1f}s before next tool round..."
        )
        await asyncio.sleep(backoff_delay)

    self._log("  [Agent] Max tool rounds exceeded")
    return "[Max tool rounds exceeded]"
