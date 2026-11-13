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

"""Lightweight multimodal Content, Probe, Response, and async retry utilities."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator, Sequence
import contextlib
import dataclasses
import functools
import inspect
import io
import random
import time
from typing import Any

import PIL.Image


class ContentChunk:
  """A single text or image chunk in a multimodal Content container."""

  def __init__(
      self,
      data_or_item: str | bytes | bytearray | PIL.Image.Image,
      mimetype: str | None = None,
  ):
    if isinstance(data_or_item, str):
      self.mimetype: str = mimetype or "text/plain"
      self.text: str = data_or_item
      self.data: bytes = data_or_item.encode("utf-8")
      self._pil_image: PIL.Image.Image | None = None
    elif isinstance(data_or_item, PIL.Image.Image):
      buf = io.BytesIO()
      data_or_item.save(buf, format="PNG")
      self.mimetype = mimetype or "image/png"
      self.data = buf.getvalue()
      self.text = ""
      self._pil_image = data_or_item
    elif isinstance(data_or_item, (bytes, bytearray)):
      self.mimetype = mimetype or "application/octet-stream"
      self.data = bytes(data_or_item)
      self.text = (
          self.data.decode("utf-8", errors="replace")
          if self.mimetype.startswith("text/")
          else ""
      )
      self._pil_image = None
    else:
      raise TypeError(f"Unsupported ContentChunk input: {type(data_or_item)!r}")

  @property
  def pil_image(self) -> PIL.Image.Image:
    """Decodes and caches a PIL Image from image bytes."""
    if self._pil_image is None:
      self._pil_image = PIL.Image.open(io.BytesIO(self.data))
    return self._pil_image


def _coerce_chunk(
    item: str | bytes | bytearray | PIL.Image.Image | ContentChunk,
) -> ContentChunk:
  if isinstance(item, ContentChunk):
    return item
  return ContentChunk(item)


class Content:
  """Container of ContentChunks representing a prompt or model response."""

  def __init__(
      self,
      *items: Any,
  ):
    self.chunks: list[ContentChunk] = []
    for arg in items:
      if arg is None:
        continue
      if isinstance(
          arg, (str, bytes, bytearray, PIL.Image.Image, ContentChunk)
      ):
        self.chunks.append(_coerce_chunk(arg))
      else:
        for sub in arg:
          self.chunks.append(_coerce_chunk(sub))

  def __iter__(self) -> Iterator[ContentChunk]:
    return iter(self.chunks)

  def __len__(self) -> int:
    return len(self.chunks)

  def __bool__(self) -> bool:
    return bool(self.chunks)

  def as_text(self, ignore_unsupported_types: bool = True) -> str:
    """Concatenates all text chunks."""
    parts: list[str] = []
    for chunk in self.chunks:
      if chunk.mimetype.startswith("text/"):
        parts.append(chunk.text)
      elif not ignore_unsupported_types:
        raise ValueError(f"Non-text chunk in Content.as_text: {chunk.mimetype}")
    return "".join(parts)

  def as_images(
      self, ignore_unsupported_types: bool = True
  ) -> list[ContentChunk]:
    """Returns all image chunks."""
    images: list[ContentChunk] = []
    for chunk in self.chunks:
      if chunk.mimetype.startswith("image/"):
        images.append(chunk)
      elif not ignore_unsupported_types:
        raise ValueError(
            f"Non-image chunk in Content.as_images: {chunk.mimetype}"
        )
    return images


@dataclasses.dataclass
class Probe:
  """A single query probe sent to a target model."""

  content: Content
  timestamp: Any = None
  id: str = ""


@dataclasses.dataclass
class Response:
  """A single response returned by a target model."""

  content: Content
  probe: Probe | None = None


def is_coroutine(func: Any) -> bool:
  """Returns True if `func` (unwrapping partials) is a coroutine function."""
  while isinstance(func, functools.partial):
    func = func.func
  return inspect.iscoroutinefunction(func)


async def to_threads_with_throttle(
    funcs: Sequence[Callable[[], Any]],
    *,
    desc: str | None = None,
    bandwidth: int | None = None,
) -> list[Any]:
  """Runs `funcs` concurrently with optional semaphore throttling."""
  del desc  # Unused in open-source runner.
  if not funcs:
    return []
  sem = (
      asyncio.Semaphore(bandwidth)
      if bandwidth is not None and bandwidth > 0
      else contextlib.nullcontext()
  )

  async def _run_one(fn: Callable[[], Any]) -> Any:
    async with sem:
      if is_coroutine(fn):
        return await fn()
      res = await asyncio.to_thread(fn)
      if inspect.isawaitable(res):
        return await res
      return res

  return list(await asyncio.gather(*(_run_one(f) for f in funcs)))


def retry(
    num_retries: int = 5,
    initial_delay: float = 1.0,
    factor: float = 2.0,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
  """Retries an async or sync callable with exponential backoff."""

  def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    if inspect.iscoroutinefunction(func):

      @functools.wraps(func)
      async def _async_wrapper(*args: Any, **kwargs: Any) -> Any:
        delay = initial_delay
        for attempt in range(max(1, num_retries)):
          try:
            return await func(*args, **kwargs)
          except Exception:  # pylint: disable=broad-exception-caught  # Last-resort guard: generic retry decorator wraps arbitrary async callables.
            if attempt + 1 >= max(1, num_retries):
              raise
            await asyncio.sleep(delay * random.uniform(0.8, 1.2))
            delay *= factor

      return _async_wrapper

    @functools.wraps(func)
    def _sync_wrapper(*args: Any, **kwargs: Any) -> Any:
      delay = initial_delay
      for attempt in range(max(1, num_retries)):
        try:
          return func(*args, **kwargs)
        except Exception:  # pylint: disable=broad-exception-caught  # Last-resort guard: generic retry decorator wraps arbitrary sync callables.
          if attempt + 1 >= max(1, num_retries):
            raise
          time.sleep(delay * random.uniform(0.8, 1.2))
          delay *= factor

    return _sync_wrapper

  return _decorator
