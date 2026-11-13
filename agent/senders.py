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

"""Agent senders and session containers for DISCO."""

import abc
import asyncio
import dataclasses
import random
from typing import Any, Callable, Coroutine, Protocol

from disco import behavior_discovery_prompts
from disco.core import prompts_protocol
from disco.models import genai_client


class AgentSender(Protocol):
  """Protocol for sending prompts to an agent and extracting results."""

  async def send_step(
      self,
      prompt_text: str,
      max_retries: int = 15,
      wait_time: float = 5.0,
      overload_base_wait: float = 15.0,
      max_delay: float = 120.0,
      log_fn: Callable[..., None] = print,
      enable_tools: bool = True,
  ) -> str:
    """Sends a prompt to the agent and returns the response text."""
    ...

  async def send_and_extract(
      self,
      prompt_text: str,
      parse_fn: Callable[..., Coroutine[Any, Any, Any]],
      max_retries: int = 2,
      fallback_value: Any = None,
      log_fn: Callable[[str], None] = print,
      retry_prompt_fn: Callable[[str, str], str] | None = None,
      enable_tools: bool = True,
      **kwargs,
  ) -> Any:
    """Sends a prompt, parses the response, and retries on failure."""
    ...

  def clear_history(self) -> None:
    """Clears conversation history if supported by the agent."""
    ...


# Concrete agent type that DiscoRunner can drive.
DiscoAgentType = Any


@dataclasses.dataclass
class DiscoSession:
  """Container for a fully initialized DISCO agent session.

  Groups the agent, its communication sender, and the prompts module it
  uses — everything produced by the agent factory and consumed by DiscoRunner.
  """

  agent: DiscoAgentType
  sender: AgentSender
  prompts_mod: prompts_protocol.PromptsModule


class BaseSender(abc.ABC):
  """Base class for agent senders providing unified retry and parsing logic."""

  def __init__(
      self,
      agent: Any,
      prompts_mod: prompts_protocol.PromptsModule = behavior_discovery_prompts,
  ):
    self.agent = agent
    self.prompts_mod = prompts_mod

  @abc.abstractmethod
  async def send_step(
      self,
      prompt_text: str,
      max_retries: int = 15,
      wait_time: float = 5.0,
      overload_base_wait: float = 15.0,
      max_delay: float = 120.0,
      log_fn: Callable[..., None] = print,
      enable_tools: bool = True,
  ) -> str:
    """Sends a prompt to the agent and returns the full response text."""
    ...

  async def send_and_extract(
      self,
      prompt_text: str,
      parse_fn: Callable[..., Coroutine[Any, Any, Any]],
      max_retries: int = 2,
      fallback_value: Any = None,
      log_fn: Callable[[str], None] = print,
      retry_prompt_fn: Callable[[str, str], str] | None = None,
      enable_tools: bool = True,
      **kwargs,
  ) -> Any:
    """Sends a prompt, parses the response, and retries on failure with self-correction."""
    if max_retries < 1:
      raise ValueError(f"max_retries must be >= 1, got {max_retries}")
    resp = ""
    last_error = ""
    for attempt in range(max_retries):
      if attempt == 0:
        cur_prompt = prompt_text
      else:
        # Clean up error message to remove long response chunks from prompt.
        clean_error = last_error.split(". Response:")[0]
        if retry_prompt_fn is not None:
          cur_prompt = retry_prompt_fn(clean_error, prompt_text)
        else:
          cur_prompt = self.prompts_mod.get_default_retry_prompt(
              clean_error, prompt_text
          )

      resp = await self.send_step(
          cur_prompt, log_fn=log_fn, enable_tools=enable_tools
      )
      try:
        return await parse_fn(resp, **kwargs)
      except Exception as e:  # pylint: disable=broad-exception-caught  # Last-resort guard: caller-supplied parse_fn may raise arbitrary validation errors to trigger retry.
        last_error = str(e)

    log_fn(
        "\n[Parsing Error] Model output could not be parsed after all"
        f" retries. Last error was '{last_error}'. Full response was:\n{resp}\n"
    )
    if fallback_value is not None:
      return fallback_value
    raise ValueError("Failed to extract expected content after retries.")

  def clear_history(self) -> None:
    """Clears conversation history if supported by the underlying agent."""
    if hasattr(self.agent, "clear_history"):
      self.agent.clear_history()


class StandaloneSender(BaseSender):
  """AgentSender implementation for StandaloneAgent."""

  async def send_step(
      self,
      prompt_text: str,
      max_retries: int = 15,
      wait_time: float = 5.0,
      overload_base_wait: float = 15.0,
      max_delay: float = 120.0,
      log_fn: Callable[..., None] = print,
      enable_tools: bool = True,
  ) -> str:
    """Sends a prompt to the standalone agent and returns the full response text.

    Retries on transient API exceptions with exponential backoff for
    overload/quota.

    Args:
      prompt_text: The prompt string to send to the agent.
      max_retries: Maximum number of retry attempts on transient errors.
      wait_time: Base wait time in seconds between standard retries.
      overload_base_wait: Base wait time in seconds when the model is
        overloaded.
      max_delay: Maximum backoff delay in seconds.
      log_fn: Logging callback function.
      enable_tools: Whether tool declarations are enabled for this step.

    Returns:
      The agent's full text response.

    Raises:
      RuntimeError: If all retry attempts are exhausted without returning.
    """
    for attempt in range(max_retries):
      try:
        text = await self.agent.send(prompt_text, enable_tools=enable_tools)
        preview = text[:1000] if len(text) > 1000 else text
        log_fn(f"\n  [Agent Response]\n{preview}")
        if len(text) > 1000:
          log_fn(f"  ... ({len(text)} chars total)")
        await asyncio.sleep(random.random() + wait_time)
        return text
      except genai_client.LLM_CALL_ERRORS as e:
        if attempt == max_retries - 1:
          raise
        err_str = str(e)
        if any(
            k in err_str
            for k in (
                "Overloaded",
                "overloaded",
                "429",
                "503",
                "UNAVAILABLE",
                "RESOURCE_EXHAUSTED",
                "ResourceExhausted",
                "ServerError",
                "ClientError",
                "APIError",
            )
        ):
          backoff = min(
              overload_base_wait * (1.6**attempt) + random.uniform(0.5, 2.0),
              max_delay,
          )
          log_fn(
              f"\n[Model Overload] {e}. Waiting"
              f" {backoff:.1f}s before retry ({attempt + 1}/{max_retries})..."
          )
          await asyncio.sleep(backoff)
        else:
          backoff = min(
              wait_time * (1.6**attempt) + random.uniform(0.5, 2.0),
              max_delay,
          )
          log_fn(
              f"\n[Transient Error] {e}. Waiting {backoff:.1f}s before retry"
              f" ({attempt + 1}/{max_retries})..."
          )
          await asyncio.sleep(backoff)
    raise RuntimeError("send_step exhausted all retries without returning.")
