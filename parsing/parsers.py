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

"""Parsers for Disco interaction cycles."""

import asyncio
import dataclasses
import json
import logging
import random
import re
from typing import Any

from disco import backbone_client as backbone_lib
from disco.parsing import schemas
from disco.prompting import parser as parser_prompts
from disco.stats import plan


def _clean_json(resp: str) -> str:
  """Extracts the first valid JSON object from a string.

  Uses json.JSONDecoder.raw_decode to find the first parseable JSON object,
  which is more robust than a greedy regex when the response contains
  multiple JSON objects or curly braces in natural language.

  Args:
    resp: The raw response string potentially containing a JSON object.

  Returns:
    The extracted JSON string, or the stripped input if no JSON is found.
  """
  decoder = json.JSONDecoder()
  # Find the first '{' and try to parse a complete JSON object from there.
  # If that fails, advance to the next '{' and retry.
  start = 0
  while True:
    idx = resp.find("{", start)
    if idx == -1:
      break
    try:
      obj, _ = decoder.raw_decode(resp, idx)
      if isinstance(obj, dict):
        return json.dumps(obj)
    except json.JSONDecodeError:
      pass
    start = idx + 1
  return resp.strip()


class BaseParser:
  """A base class for parsing LLM responses using a schema and validation."""

  schema_cls = None
  err_prefix = (
      "BaseParser failed to extract valid structured JSON from response"
  )

  def __init__(self, model: str):
    """Initializes the BaseParser.

    Args:
      model: The name of the model to use for distillation.
    """
    self._model = model

  async def _distill_with_llm(self, resp: str) -> str:
    """Distills the LLM response using another LLM call with a schema.

    This method sends the raw LLM response to a Gemini model, configured to
    return JSON according to `self.schema_cls`. It also includes system
    instructions and safety settings.

    Args:
      resp: The raw string response from an LLM.

    Returns:
      A string containing the JSON-formatted distillation from the LLM,
      or the original response if an error occurs.
    """
    backbone = backbone_lib.get_backbone_client(self._model)
    text = await backbone.generate(
        model=self._model,
        prompt=resp,
        response_schema=self.schema_cls,
        system_instruction=parser_prompts.get_parser_system_instruction(),
    )
    if text:
      return text.strip()
    raise ValueError("Received empty distillation response.")

  async def parse(self, resp: str) -> Any:
    """Parses the response text, retrying up to 5 times.

    This method sends the response to an LLM for distillation, cleans the
    resulting JSON, validates it against the schema, and performs
    subclass-specific validation.

    Args:
      resp: The raw string response from the LLM to be parsed.

    Returns:
      The parsed and validated result. The type depends on the subclass (e.g.,
      HypothesisResult, StatsResult, int, or str).

    Raises:
      ValueError: If `schema_cls` is not defined on the parser subclass.
      schemas.InsufficientInformationError: If the distillation agent
        reports insufficient information.
      schemas.ParserExhaustedError: If all attempts to distill and parse
        the response fail.
    """
    if self.schema_cls is None:
      raise ValueError("Parser schema_cls is not defined.")
    last_err: Exception = Exception("No parse attempts made.")
    max_parse_attempts = 5
    base_delay = 2.0
    max_delay = 60.0
    for attempt in range(max_parse_attempts):
      try:
        llm_resp = await self._distill_with_llm(resp)
        cleaned = _clean_json(llm_resp)
        data = json.loads(cleaned)
        parsed = self.schema_cls.model_validate(data)
        # the 3rd boolean argument here is a fallback
        if getattr(parsed, "insufficient_information", False):
          # Propagates directly (not in LLM_CALL_ERRORS) so the main agent
          # regenerates a response containing the missing information.
          raise schemas.InsufficientInformationError(
              "Distillation agent reported INSUFFICIENT_INFORMATION."
          )
        return self._validate(parsed, resp)
      except backbone_lib.LLM_CALL_ERRORS as e:
        # this error is from the parser, and it should retry parsing
        last_err = e
        if attempt < max_parse_attempts - 1:
          delay = min(
              base_delay * (2**attempt) * random.uniform(0.8, 1.2), max_delay
          )
          logging.warning(
              "%s (attempt %d/%d failed: %s). Retrying distillation in"
              " %.1fs...",
              self.err_prefix,
              attempt + 1,
              max_parse_attempts,
              e,
              delay,
          )
          await asyncio.sleep(delay)
        else:
          logging.error(
              "%s: All %d distillation attempts failed: %s",
              self.err_prefix,
              max_parse_attempts,
              e,
          )
    # All parse/distillation attempts failed due to JSON, validation, or
    # similar errors (InsufficientInformationError is re-raised immediately
    # above and never reaches here). Raise ParserExhaustedError so the
    # caller asks the main agent to regenerate its response.
    raise schemas.ParserExhaustedError(
        f"{self.err_prefix}. Details: {str(last_err)}. Response: {resp}"
    ) from last_err

  def _validate(self, parsed: Any, original_resp: str) -> Any:
    raise NotImplementedError("Subclasses must implement the _validate method.")


class HypothesisParser(BaseParser):
  """A parser for extracting Hypothesis ID and text."""

  schema_cls = schemas.HypothesisExtraction
  err_prefix = (
      "HypothesisParser failed to resolve required HYPOTHESIS_ID or HYPOTHESIS"
      " fields"
  )

  def _validate(
      self, parsed: schemas.HypothesisExtraction, original_resp: str
  ) -> schemas.HypothesisResult:
    """Validates and extracts the Hypothesis ID and text.

    Args:
      parsed: The raw pydantic model extracted from LLM distillation.
      original_resp: The original LLM response.

    Returns:
      A HypothesisResult instance.

    Raises:
      ValueError: If required hypothesis attributes are missing.
    """
    del original_resp
    if not parsed.hypothesis_id or not parsed.hypothesis_text:
      raise ValueError("Required hypothesis attributes were missing.")
    return schemas.HypothesisResult(
        h_id=parsed.hypothesis_id, h_text=parsed.hypothesis_text
    )


class StatsParser(BaseParser):
  """Parses Statistical test results."""

  schema_cls = schemas.StatsExtraction
  err_prefix = (
      "StatsParser failed to map and validate statistical tests against tool"
      " calls"
  )

  def __init__(
      self,
      model: str,
      validate_stats_test: bool = False,
  ):
    super().__init__(model)
    self._validate_stats_test = validate_stats_test

  def _validate(
      self, parsed: schemas.StatsExtraction, original_resp: str
  ) -> plan.StatsResult:
    """Validates and extracts statistical test results.

    Checks that the extracted test type matches a supported tool and that the
    tool was actually invoked in the original response.

    Args:
      parsed: The raw pydantic model extracted from LLM distillation.
      original_resp: The original LLM response to check for tool calls.

    Returns:
      A StatsResult instance.

    Raises:
      ValueError: If required stats attributes are missing, p-value is invalid,
        no tool was called, or the extracted test type doesn't match the called
        tool.
    """
    test_type = parsed.test_type.lower() if parsed.test_type else ""
    if parsed.p_value is None:
      raise ValueError("Required P-VALUE was missing or could not be parsed.")
    p_value = parsed.p_value
    if not (0 <= p_value <= 1):
      raise ValueError(f"P-value must be between 0 and 1, got {p_value}.")

    if self._validate_stats_test:
      called_any_tool = any(t.value in original_resp for t in plan.StatsTool)
      if not called_any_tool:
        raise ValueError(
            "You failed to invoke any valid tool call handlers. It is possible"
            " you called them in a previous turn, but please do so again if so."
        )

      expected_tool = plan.match_test_to_tool(test_type)

      if not expected_tool:
        raise ValueError(
            f"The stated test type '{test_type}' did not match any statistical"
            " tool keywords."
        )

      if expected_tool.value not in original_resp:
        raise ValueError(
            f"The extracted test type '{test_type}' suggests using the tool"
            f" '{expected_tool.value}', but no call to that tool was found in"
            " the original response. It's likely the main agent claimed to run"
            " this test but failed to invoke the tool."
        )

    null_prop = parsed.null_proportion
    if null_prop is None and original_resp:
      m = re.search(
          r"Null proportion:\s*([0-9.]+)", original_resp, re.IGNORECASE
      )
      if not m:
        m = re.search(
            r'["\']null_proportion["\']\s*:\s*([0-9.]+)',
            original_resp,
            re.IGNORECASE,
        )
      if m:
        try:
          null_prop = float(m.group(1))
        except ValueError:
          pass

    alternative = parsed.alternative
    if alternative is None and original_resp:
      m = re.search(
          r"Alternative:\s*([a-zA-Z_-]+)", original_resp, re.IGNORECASE
      )
      if not m:
        m = re.search(
            r'["\']alternative["\']\s*:\s*["\']([a-zA-Z_-]+)["\']',
            original_resp,
            re.IGNORECASE,
        )
      if m:
        alternative = m.group(1).lower()

    sample_std = parsed.sample_std
    if sample_std is None and original_resp:
      m = re.search(
          r"(?:sample\s+std|std\s+dev|std\s+deviation|std_diff|std):\s*([0-9.]+)",
          original_resp,
          re.IGNORECASE,
      )
      if m:
        try:
          sample_std = float(m.group(1))
        except ValueError:
          pass

    return plan.StatsResult(
        test_type=test_type,
        p_value=p_value,
        group_a_mean=parsed.group_a_mean,
        group_a_obs=parsed.group_a_obs,
        group_b_mean=parsed.group_b_mean,
        group_b_obs=parsed.group_b_obs,
        null_proportion=null_prop,
        alternative=alternative,
        reasoning=parsed.reasoning,
        sample_std=sample_std,
    )


class TestPreregistrationParser(BaseParser):
  """A parser for extracting pre-registered statistical test plan."""

  schema_cls = schemas.TestPreregistrationExtraction
  err_prefix = (
      "TestPreregistrationParser failed to resolve valid pre-registered test"
      " plan items"
  )

  def _validate(
      self,
      parsed: schemas.TestPreregistrationExtraction,
      original_resp: str,
  ) -> plan.PreRegisteredTestPlan:
    """Validates and extracts the pre-registered test plan.

    Args:
      parsed: The raw pydantic model extracted from LLM distillation.
      original_resp: The original LLM response.

    Returns:
      A PreRegisteredTestPlan instance.

    Raises:
      ValueError: If required attributes are missing or invalid.
    """
    test_type = parsed.test_type.strip() if parsed.test_type else ""
    if not test_type:
      raise ValueError("Required TEST_TYPE was missing.")

    target_metric = parsed.target_metric.strip() if parsed.target_metric else ""
    if not target_metric:
      raise ValueError("Required TARGET_METRIC was missing.")

    null_baseline = parsed.null_baseline
    if null_baseline is None and original_resp:
      # Continuous tests declare a null in the metric's own units, which may be
      # negative, so the sign is part of the number rather than a separator.
      m = re.search(
          r"Null (?:baseline|proportion|value):\s*(-?[0-9.]+)",
          original_resp,
          re.IGNORECASE,
      )
      if not m:
        m = re.search(
            r'["\']null_(?:baseline|proportion|value)["\']\s*:\s*(-?[0-9.]+)',
            original_resp,
            re.IGNORECASE,
        )
      if m:
        try:
          null_baseline = float(m.group(1))
        except ValueError:
          pass

    if null_baseline is None:
      # 0.5 is the chance rate for a binary test, but it is a meaningless
      # default for a mean or a mean difference, where "no effect" is 0.
      null_baseline = 0.0 if plan.is_continuous_test(test_type) else 0.5

    alternative = (
        parsed.alternative.strip().lower() if parsed.alternative else ""
    )
    if not alternative and original_resp:
      m = re.search(
          r"Alternative:\s*([a-zA-Z_-]+)", original_resp, re.IGNORECASE
      )
      if not m:
        m = re.search(
            r'["\']alternative["\']\s*:\s*["\']([a-zA-Z_-]+)["\']',
            original_resp,
            re.IGNORECASE,
        )
      if m:
        alternative = m.group(1).lower()

    if alternative not in (
        "larger",
        "smaller",
        "two-sided",
        "twosided",
        "two_sided",
    ):
      alternative = "larger"
    elif alternative in ("twosided", "two_sided"):
      alternative = "two-sided"

    is_paired = bool(parsed.is_paired)
    if not is_paired and original_resp:
      m = re.search(
          r"Is[ _-]?Paired:\s*(true|yes|1)", original_resp, re.IGNORECASE
      )
      if m:
        is_paired = True

    return plan.PreRegisteredTestPlan(
        test_type=test_type,
        target_metric=target_metric,
        null_baseline=float(null_baseline),
        alternative=alternative,
        reasoning=parsed.reasoning,
        is_paired=is_paired,
    )


@dataclasses.dataclass
class Parsers:
  """A bundle of parser instances, each owning its own model."""

  hypothesis: HypothesisParser
  stats: StatsParser
  test_preregistration: TestPreregistrationParser | None = None

  @classmethod
  def from_model(
      cls, model: str, validate_stats_test: bool = False
  ) -> "Parsers":
    """Constructs a `Parsers` bundle using `model` for all parsers."""
    return cls(
        hypothesis=HypothesisParser(model=model),
        stats=StatsParser(model=model, validate_stats_test=validate_stats_test),
        test_preregistration=TestPreregistrationParser(model=model),
    )


async def parse_h(resp: str, parsers: Parsers) -> schemas.HypothesisResult:
  """Parses a hypothesis ID and text from an agent response."""
  return await parsers.hypothesis.parse(resp)


async def parse_and_validate_stats(
    resp: str, parsers: Parsers
) -> plan.StatsResult:
  """Parses and validates statistical test results from an agent response."""
  return await parsers.stats.parse(resp)


async def parse_and_validate_preregistration(
    resp: str, parsers: Parsers
) -> plan.PreRegisteredTestPlan:
  """Parses and validates pre-registered statistical test plan from an agent response."""
  if parsers.test_preregistration is None:
    raise ValueError(
        "test_preregistration parser is not initialized in Parsers."
    )
  return await parsers.test_preregistration.parse(resp)
