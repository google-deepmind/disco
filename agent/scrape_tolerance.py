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

"""Per-request fault tolerance for target-model scraping.

`content.to_threads_with_throttle` awaits every task it launches, so a
single task that raises propagates out of the whole call. For a batch of `n`
probes that turns an independent per-probe failure probability `p` into a batch
failure probability `1 - (1 - p)^n`: at n=100 even p=3% loses the batch 95% of
the time, and the 97 responses that *did* come back are discarded along with it.

That is how a verification run can report `sample_size: 0` while the target
model was answering almost every request. The failure is especially damaging
during verification, which deliberately issues one large batch per hypothesis,
whereas discovery spreads the same budget over many small batches and mostly
survives.

This module supplies `gather_tolerant`, a drop-in replacement that isolates
each task: a task that raises yields `None` in its slot instead of destroying
its siblings. Callers already keep positional slots and re-request the empty
ones, so `None` is exactly the signal that machinery expects.

Retrying is deliberately NOT done per task here. `GeminiTargetCandidate.respond`
already retries each request with exponential backoff, so an exception reaching
us means sustained rate limiting or server overload (429 RESOURCE_EXHAUSTED /
503 UNAVAILABLE), and hammering the endpoint again inside the same burst makes
that worse. Recovery belongs between waves, which is what `backoff_delays` is
for.
"""

from __future__ import annotations

import asyncio
import collections
from collections.abc import Callable, Sequence
import functools
import inspect
import random
from typing import Any

from disco.models import content as content_api

# Substrings identifying an error caused by load shedding or rate limiting
# rather than by a bad request. Matched against a lowercased `str(exc)`.
_THROTTLING_MARKERS = (
    "throttl",
    "resource_exhausted",
    "rate limit",
    "429",
    "quota",
    "exhausted",
    "unavailable",
    "deadline_exceeded",
    "timed out",
    "timeout",
)


def is_throttling_error(exc: BaseException | str) -> bool:
  """Returns True if `exc` looks like transient load shedding."""
  text = (exc if isinstance(exc, str) else str(exc)).lower()
  return any(marker in text for marker in _THROTTLING_MARKERS)


def _is_coroutine_fn(func: Any) -> bool:
  """Mirrors `content_api.is_coroutine`, unwrapping nested partials."""
  while isinstance(func, functools.partial):
    func = func.func
  return inspect.iscoroutinefunction(func)


def summarize_errors(errors: Sequence[BaseException]) -> str:
  """Renders a compact digest of `errors` grouped by kind.

  A failed batch produces one identical exception per probe, so logging them
  individually buries the run log without adding information. Grouping keeps
  the count (which is the diagnostic) and one exemplar (which is the cause).

  Args:
    errors: The exceptions raised by individual tasks.

  Returns:
    A single-line summary, or the empty string if `errors` is empty.
  """
  if not errors:
    return ""
  buckets: dict[str, list[BaseException]] = collections.defaultdict(list)
  for exc in errors:
    kind = "throttled" if is_throttling_error(exc) else type(exc).__name__
    buckets[kind].append(exc)
  parts = []
  for kind, group in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
    exemplar = str(group[0]).replace("\n", " ")
    if len(exemplar) > 240:
      exemplar = exemplar[:240] + "..."
    parts.append(f"{len(group)}x {kind}: {exemplar}")
  return " | ".join(parts)


async def gather_tolerant(
    funcs: Sequence[Callable[[], Any]],
    *,
    desc: str | None = None,
    bandwidth: int | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[list[Any | None], list[BaseException]]:
  """Runs `funcs` concurrently, isolating per-task failures.

  Unlike `content_api.to_threads_with_throttle`, this never raises because one
  task raised. The returned list always has the same length and ordering as
  `funcs`, with `None` in the slot of every task that failed. Positional
  alignment is load-bearing for every caller here: slot `i` is the answer to
  probe `i`, so a dropped element would silently reassign every later response
  to the wrong probe.

  Args:
    funcs: Zero-argument callables, sync or async.
    desc: Progress description forwarded to the underlying throttler.
    bandwidth: Maximum number of concurrently in-flight calls.
    log: Optional sink for a one-line failure digest.

  Returns:
    A `(results, errors)` pair. `results` is positionally aligned with `funcs`;
    `errors` holds the exceptions that were swallowed, in completion order.
  """
  if not funcs:
    return [], []

  errors: list[BaseException] = []
  lock = asyncio.Lock()

  def _wrap(func: Callable[[], Any]) -> Callable[[], Any]:
    is_async = _is_coroutine_fn(func)

    async def _run() -> Any:
      try:
        if is_async:
          return await func()
        # Sync callables must not run on the event loop: they would serialize
        # the whole batch behind whichever one is blocking on the network.
        result = await asyncio.to_thread(func)
        if inspect.isawaitable(result):
          # A sync callable that *returns* a coroutine, e.g.
          # `lambda: candidate.respond(content)`. `iscoroutinefunction` is
          # False for it, so without this the caller would receive an
          # un-awaited coroutine object in place of a response -- a silent
          # corruption rather than an error.
          return await result
        return result
      except Exception as exc:  # pylint: disable=broad-exception-caught  # Last-resort per-task isolation guard: record arbitrary task failure without aborting sibling tasks.
        async with lock:
          errors.append(exc)
        return None

    return _run

  wrapped = [_wrap(f) for f in funcs]
  results: list[Any | None] = await content_api.to_threads_with_throttle(
      wrapped,
      desc=desc,
      bandwidth=bandwidth,
  )

  if errors and log is not None:
    log(
        f"  [Scrape Partial] {len(errors)}/{len(funcs)} request(s) failed and"
        f" were left as empty slots: {summarize_errors(errors)}"
    )
  return results, errors


def backoff_delays(
    attempt: int,
    *,
    base_delay: float = 15.0,
    factor: float = 3.0,
    max_delay: float = 300.0,
) -> float:
  """Returns the seconds to wait before backfill attempt `attempt` (1-based).

  Sized for recovery from sustained load shedding rather than for a transient
  blip: `GeminiTargetCandidate.respond` has already retried each request with
  backoff by the time a slot is still empty, so a sub-second pause here would
  just re-issue into the same rate limit.

  Args:
    attempt: 1-based index of the backfill attempt about to be made.
    base_delay: Delay before the first retry wave.
    factor: Multiplicative growth per wave.
    max_delay: Ceiling on the returned delay, before fuzzing.

  Returns:
    A fuzzed delay in seconds.
  """
  delay = min(base_delay * (factor ** max(0, attempt - 1)), max_delay)
  return delay * random.uniform(0.8, 1.2)


class TargetOutageError(RuntimeError):
  """Too many consecutive discovery rounds were lost to the target model.

  Raised by `RoundGuard`. The run's checkpoint is saved before the guard runs,
  so rerunning the same job resumes where the outage stopped it.
  """


class RoundGuard:
  """Stops a discovery loop that keeps losing rounds to the target model.

  A skipped round charges no budget, so a target that stays down (or keeps
  returning too few usable responses) would otherwise make the loop spin
  forever. After each round, call `after_round` with the round's skip kind
  (`scoring.SKIP_KINDS`, or None for a round that produced an e-value):

  * "target_side" and "short" skips count as consecutive target failures. Each
    one waits `backoff_delays` with a short schedule (5s, 10s, 20s, 40s, then
    60s) before the next round. The `max_consecutive_skips`-th in a row raises
    `TargetOutageError`.
  * A round that produced an e-value resets the count.
  * "auditor_side" skips (the auditor model failed, not the target) neither
    count nor reset.

  The count is not checkpointed: a resumed run starts from zero.
  """

  COUNTED_KINDS = frozenset({"target_side", "short"})

  def __init__(
      self,
      max_consecutive_skips: int,
      log: Callable[[str], None],
  ):
    """Initializes the guard.

    Args:
      max_consecutive_skips: Consecutive target-side or short skips that raise.
        0 disables both the cap and the backoff.
      log: Logger.
    """
    self._max = max_consecutive_skips
    self._log = log
    self.consecutive = 0

  async def after_round(self, skip_kind: str | None) -> None:
    """Records one round's outcome; backs off or raises after a target skip.

    Args:
      skip_kind: Why the round was skipped, or None if it was not.

    Raises:
      TargetOutageError: On the `max_consecutive_skips`-th consecutive
        target-side or short skip.
    """
    if skip_kind is None:
      self.consecutive = 0
      return
    if skip_kind not in self.COUNTED_KINDS or self._max <= 0:
      return
    self.consecutive += 1
    if self.consecutive >= self._max:
      self._log(
          f"\n[Target Outage] {self.consecutive} consecutive rounds were"
          " skipped for target-side failures or too few responses"
          f" (--max_consecutive_skips={self._max}). Stopping; the checkpoint"
          " is saved, so rerunning resumes from here."
      )
      raise TargetOutageError(
          f"{self.consecutive} consecutive discovery rounds skipped"
          f" ({skip_kind} last); the target model looks unavailable."
      )
    delay = backoff_delays(
        self.consecutive, base_delay=5.0, factor=2.0, max_delay=60.0
    )
    self._log(
        f"  [Round Guard] {self.consecutive}/{self._max} consecutive target"
        f" skips; waiting {delay:.0f}s before the next round."
    )
    await asyncio.sleep(delay)
