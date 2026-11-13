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

"""Unified backbone client abstraction for DISCO.

Provides a `BackboneClient` interface that routes LLM calls to Gemini
(via ``genai.Client``). This lets DISCO's parsers, checkers, autoraters,
and helpers call backbone models through a clean async abstraction.

The agent's function-calling loop (``StandaloneAgent.send``) continues
to use ``genai.Client`` directly.
"""

import abc
import dataclasses
from typing import Any, Union

from google import genai
from google.genai import types
import pydantic

from disco.models import genai_client
from disco.models import registry as model_registry

LLM_CALL_ERRORS = genai_client.LLM_CALL_ERRORS

# ── Multimodal prompt types ────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True, slots=True)
class ImagePart:
  """Provider-agnostic container for an image within a multimodal prompt."""

  data: bytes
  mimetype: str = "image/png"


PromptPart = Union[str, ImagePart]
Prompt = Union[str, list[PromptPart]]


class BackboneClient(abc.ABC):
  """Async interface for backbone LLM generation."""

  @abc.abstractmethod
  async def generate(
      self,
      model: str,
      prompt: Prompt,
      *,
      temperature: float | None = None,
      system_instruction: str | None = None,
      response_schema: type[pydantic.BaseModel] | None = None,
      max_output_tokens: int | None = None,
  ) -> str:
    """Generates a text response from the backbone model."""
    ...


class GeminiBackboneClient(BackboneClient):
  """Routes backbone calls through the Gemini API."""

  def __init__(self, client: genai.Client | None = None) -> None:
    self._client_instance = client

  @property
  def _client(self) -> genai.Client:
    if self._client_instance is None:
      self._client_instance = genai_client.get_genai_client()
    return self._client_instance

  async def generate(
      self,
      model: str,
      prompt: Prompt,
      *,
      temperature: float | None = None,
      system_instruction: str | None = None,
      response_schema: type[pydantic.BaseModel] | None = None,
      max_output_tokens: int | None = None,
  ) -> str:
    """Generates a text or structured JSON response via the Gemini API."""
    config_kwargs: dict[str, Any] = {
        "safety_settings": list(genai_client.DEFAULT_SAFETY_SETTINGS),
    }
    if temperature is not None:
      config_kwargs["temperature"] = temperature
    if system_instruction:
      config_kwargs["system_instruction"] = system_instruction
    if response_schema is not None:
      config_kwargs["response_mime_type"] = "application/json"
      config_kwargs["response_schema"] = response_schema
    if max_output_tokens is not None:
      config_kwargs["max_output_tokens"] = max_output_tokens

    norm_model = model_registry.normalize_model_address(model)
    if not norm_model.startswith("models/") and "/" not in norm_model:
      norm_model = f"models/{norm_model}"

    contents = _prompt_to_gemini_contents(prompt)

    response = await self._client.aio.models.generate_content(
        model=norm_model,
        contents=contents,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    if not response.text:
      raise ValueError("Backbone model returned an empty response.")
    return response.text.strip()


def _prompt_to_gemini_contents(
    prompt: Prompt,
) -> Any:
  """Converts a ``Prompt`` to Gemini-compatible ``contents``."""
  if isinstance(prompt, str):
    return prompt
  parts: list[types.Part] = []
  for part in prompt:
    if isinstance(part, str):
      parts.append(types.Part.from_text(text=part))
    elif isinstance(part, ImagePart):
      parts.append(
          types.Part.from_bytes(
              data=part.data,
              mime_type=part.mimetype,
          )
      )
  return parts


_cached_backbones: dict[str, BackboneClient] = {}


def get_backbone_client(backbone_model: str = "") -> BackboneClient:
  """Returns a cached ``BackboneClient`` for the given model address."""
  if backbone_model in _cached_backbones:
    return _cached_backbones[backbone_model]

  client = GeminiBackboneClient()
  _cached_backbones[backbone_model] = client
  return client


def reset_backbone_client() -> None:
  """Resets the cached backbone clients (for testing)."""
  _cached_backbones.clear()
