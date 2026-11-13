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

"""Target model candidate wrapper backed by the google-genai SDK."""

from __future__ import annotations

import asyncio
import random
from typing import Any
import urllib.parse

from google.genai import types

from disco.models import content as content_lib
from disco.models import genai_client
from disco.models import registry


def _parse_bool(val: str) -> bool:
  return val.strip().lower() in ("1", "true", "yes", "on")


class GeminiTargetCandidate:
  """Target model candidate backed by `google.genai.Client`."""

  def __init__(
      self,
      model_name: str,
      *,
      num_inner_retries: int = 3,
      temperature: float | None = None,
      max_output_tokens: int | None = None,
      image_out: bool | None = None,
  ):
    self.model_name = registry.normalize_model_address(model_name)
    self.num_inner_retries = num_inner_retries
    self.temperature = temperature
    self.max_output_tokens = max_output_tokens
    if image_out is None:
      self.image_out = "image" in self.model_name.lower()
    else:
      self.image_out = image_out

  async def respond(
      self,
      content: content_lib.Content | str,
      jitter: float = 0.0,
  ) -> content_lib.Response:
    """Sends a single-turn multimodal or text query to the target Gemini model."""
    if isinstance(content, str):
      content = content_lib.Content(content)
    if jitter > 0:
      await asyncio.sleep(random.uniform(0, jitter))

    parts: list[Any] = []
    for chunk in content:
      if chunk.mimetype.startswith("image/"):
        parts.append(
            types.Part.from_bytes(data=chunk.data, mime_type=chunk.mimetype)
        )
      elif chunk.text:
        parts.append(types.Part.from_text(text=chunk.text))

    config_kwargs: dict[str, Any] = {
        "safety_settings": list(genai_client.DEFAULT_SAFETY_SETTINGS),
    }
    if self.temperature is not None:
      config_kwargs["temperature"] = self.temperature
    if self.max_output_tokens is not None:
      config_kwargs["max_output_tokens"] = self.max_output_tokens
    if self.image_out:
      config_kwargs["response_modalities"] = ["TEXT", "IMAGE"]

    config = types.GenerateContentConfig(**config_kwargs)
    client = genai_client.get_genai_client()

    delay = 1.0
    last_exc: Exception | None = None
    resp = None
    attempts = max(1, self.num_inner_retries)
    for attempt in range(attempts):
      try:
        resp = await client.aio.models.generate_content(
            model=self.model_name,
            contents=parts,
            config=config,
        )
        break
      except genai_client.LLM_CALL_ERRORS as exc:
        last_exc = exc
        if attempt + 1 >= attempts:
          raise
        await asyncio.sleep(delay * random.uniform(0.8, 1.2))
        delay *= 3.0

    if resp is None and last_exc is not None:
      raise last_exc

    out_chunks: list[content_lib.ContentChunk] = []
    if resp is not None and resp.candidates:
      cand_content = resp.candidates[0].content
      if cand_content and cand_content.parts:
        for part in cand_content.parts:
          inline_data = getattr(part, "inline_data", None)
          if inline_data is not None and getattr(inline_data, "data", None):
            out_chunks.append(
                content_lib.ContentChunk(
                    inline_data.data,
                    mimetype=getattr(inline_data, "mime_type", None)
                    or "image/png",
                )
            )
          elif getattr(part, "text", None):
            out_chunks.append(
                content_lib.ContentChunk(part.text, mimetype="text/plain")
            )
    if not out_chunks and resp is not None and getattr(resp, "text", None):
      out_chunks.append(
          content_lib.ContentChunk(resp.text, mimetype="text/plain")
      )

    return content_lib.Response(
        content=content_lib.Content(out_chunks),
        probe=content_lib.Probe(content=content),
    )


def make_candidate(model_address: str) -> GeminiTargetCandidate:
  """Constructs a `GeminiTargetCandidate` from a model address string."""
  if not model_address or not model_address.strip():
    raise ValueError("Model address must be non-empty.")
  base_addr, _, query_str = model_address.partition("?")
  norm_name = registry.normalize_model_address(base_addr)
  params = urllib.parse.parse_qs(query_str)

  num_inner_retries = int(params.get("num_inner_retries", ["3"])[0])
  temperature = (
      float(params["temperature"][0]) if "temperature" in params else None
  )
  max_output_tokens = (
      int(params["max_output_tokens"][0])
      if "max_output_tokens" in params
      else None
  )
  image_out = (
      _parse_bool(params["image_out"][0]) if "image_out" in params else None
  )

  return GeminiTargetCandidate(
      norm_name,
      num_inner_retries=num_inner_retries,
      temperature=temperature,
      max_output_tokens=max_output_tokens,
      image_out=image_out,
  )
