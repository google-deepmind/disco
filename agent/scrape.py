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

"""Scrapes N requests from a target model, refilling empty slots in place.

`StandaloneAgent.scrape_responses` (text) and
`StandaloneAgent._tool_generate_and_send_symmetry_probes` (visual symmetry) used
to carry their own copies of the same loop. Both now call
`scrape_with_backfill`, and their differences are explicit arguments:

| | text | symmetry |
| --- | --- | --- |
| chunking | `chunk_size=20`, each chunk retried on a raised gather | one batch,
no retry |
| bandwidth | `scrape_bandwidth` | 10 |
| jitter | set by the caller's `make_task` | none |
| usable slot | non-blank text or any image | decodes to an RGB image |
| log wording | `TEXT_LABELS` | `SYMMETRY_LABELS` |
| top-up past `MAX_BACKFILL` | while short of `--min_response_fraction` |
never |

What happens when every slot is still empty (raise vs. report FAILED) stays in
the callers.
"""

import asyncio
import dataclasses
import random
import time
from typing import Any, Callable

from disco.agent import scrape_tolerance
from disco.verification import scoring

MAX_BACKFILL = 3

# Top-up waves run after the `MAX_BACKFILL` regular waves while fewer than
# `min_fraction` of the slots are filled (`--min_response_fraction`). Each one
# waits `scrape_tolerance.backoff_delays` first (~15s, 45s, 135s, 300s, 300s),
# and none starts once `TOP_UP_DEADLINE_S` has passed since the scrape began.
# A scrape with no usable response at all is not topped up (see below).
MAX_TOP_UP = 5
TOP_UP_DEADLINE_S = 20 * 60.0


@dataclasses.dataclass(frozen=True)
class ScrapeLabels:
  """Log wording for one scrape path.

  Attributes:
    initial_desc: Progress description of the first pass. Chunked scrapes append
      " <chunk>/<num_chunks>".
    backfill_desc: Progress description of a backfill wave, followed by "
      <attempt>/<max_backfill>".
    deficit_noun: What is missing, e.g. "response(s)".
    request_noun: What is re-requested, e.g. "prompt(s)".
    extras_noun: What responses are counted against when there are too many.
    log_blank_slots: Whether to log each first-pass response that came back but
      was not usable.
  """

  initial_desc: str
  backfill_desc: str
  deficit_noun: str
  request_noun: str
  extras_noun: str
  log_blank_slots: bool = False


TEXT_LABELS = ScrapeLabels(
    initial_desc="Scraping chunk",
    backfill_desc="Scraping backfill",
    deficit_noun="response(s)",
    request_noun="prompt(s)",
    extras_noun="prompts",
    log_blank_slots=True,
)

SYMMETRY_LABELS = ScrapeLabels(
    initial_desc="Sending symmetry probes",
    backfill_desc="Symmetry backfill",
    deficit_noun="symmetry response(s)",
    request_noun="probe(s)",
    extras_noun="probes",
)


@dataclasses.dataclass
class ScrapeOutcome:
  """Result of one `scrape_with_backfill` call.

  Attributes:
    slots: One entry per request, positionally aligned with the requests. A slot
      holds a usable response or None.
    n_requested: Number of requests.
    n_missing_initially: Empty slots after the first pass.
    n_backfill_attempts: Backfill waves that were run, top-up waves included.
    errors: Every per-request exception swallowed along the way.
    n_top_up_attempts: How many of the backfill waves were top-up waves (run
      past `max_backfill` to reach `min_fraction`).
  """

  slots: list[Any | None]
  n_requested: int
  n_missing_initially: int
  n_backfill_attempts: int
  errors: list[BaseException] = dataclasses.field(default_factory=list)
  n_top_up_attempts: int = 0

  @property
  def valid_indices(self) -> list[int]:
    """Indices of slots that received a valid response."""
    return [i for i, resp in enumerate(self.slots) if resp is not None]

  @property
  def missing_indices(self) -> list[int]:
    """Indices of slots that remain empty after scraping and backfill."""
    return [i for i, resp in enumerate(self.slots) if resp is None]

  @property
  def n_received(self) -> int:
    """Total number of valid responses received."""
    return len(self.valid_indices)

  @property
  def n_backfilled(self) -> int:
    """Number of initially missing responses recovered during backfill."""
    return self.n_missing_initially - len(self.missing_indices)


async def _gather_chunked(
    make_task: Callable[[int], Callable[[], Any]],
    n: int,
    *,
    chunk_size: int,
    bandwidth: int,
    labels: ScrapeLabels,
    log: Callable[[str], None],
    chunk_retries: int,
    chunk_base_delay: float,
    chunk_max_delay: float,
    on_chunk_failure: Callable[[], None] | None,
) -> tuple[list[Any | None], list[BaseException]]:
  """First pass in chunks; a chunk is retried only if its gather raises."""
  starts = list(range(0, n, chunk_size))
  num_chunks = len(starts)
  raw: list[Any | None] = []
  errors: list[BaseException] = []
  for chunk_idx, start in enumerate(starts):
    tasks = [make_task(i) for i in range(start, min(start + chunk_size, n))]
    for attempt in range(chunk_retries):
      try:
        raw_chunk, chunk_errors = await scrape_tolerance.gather_tolerant(
            tasks,
            desc=f"{labels.initial_desc} {chunk_idx + 1}/{num_chunks}",
            bandwidth=min(bandwidth, len(tasks)),
            log=log,
        )
        raw.extend(raw_chunk)
        errors.extend(chunk_errors)
        break
      except Exception as e:  # pylint: disable=broad-exception-caught  # Last-resort batch guard: retry chunk on arbitrary caller task errors.
        if attempt < chunk_retries - 1:
          # Exponential backoff with random jitter [0.8, 1.2], capped.
          exp_delay = chunk_base_delay * (2**attempt)
          jitter_mult = random.uniform(0.8, 1.2)
          delay = min(exp_delay * jitter_mult, chunk_max_delay)
          log(
              f"  [Scrape Retry] Chunk {chunk_idx + 1}/{num_chunks} failed"
              f" (attempt {attempt + 1}/{chunk_retries}), retrying in"
              f" {delay:.1f}s: {e}"
          )
          await asyncio.sleep(delay)
        else:
          log(
              f"  [Scrape Error] Chunk {chunk_idx + 1}/{num_chunks} failed"
              f" after {chunk_retries} attempts: {e}"
          )
          if on_chunk_failure is not None:
            on_chunk_failure()
          raise
  return raw, errors


async def scrape_with_backfill(
    make_task: Callable[[int], Callable[[], Any]],
    n: int,
    *,
    is_usable: Callable[[Any], bool],
    bandwidth: int,
    labels: ScrapeLabels,
    log: Callable[[str], None],
    chunk_size: int | None = None,
    chunk_retries: int = 4,
    chunk_base_delay: float = 2.0,
    chunk_max_delay: float = 20.0,
    max_backfill: int = MAX_BACKFILL,
    on_chunk_failure: Callable[[], None] | None = None,
    min_fraction: float = 0.0,
    max_top_up: int = MAX_TOP_UP,
    top_up_deadline_s: float = TOP_UP_DEADLINE_S,
    clock: Callable[[], float] = time.monotonic,
) -> ScrapeOutcome:
  """Issues `n` requests and refills unusable slots in place.

  Slot `i` always answers request `i`. Alignment is load-bearing: paired text
  batches alternate A/B, and symmetry requests map back to probe images by
  position, so an empty slot is refilled where it is rather than compacted.

  Args:
    make_task: Returns a zero-argument callable that issues request `i`. It is
      called again for each backfill attempt of `i`.
    n: Number of requests.
    is_usable: Whether a non-None response fills its slot.
    bandwidth: Maximum concurrent requests (further capped by batch size).
    labels: Log wording.
    log: Logger.
    chunk_size: If set, the first pass runs in chunks of this size, and a chunk
      whose gather raises is retried up to `chunk_retries` times. If None, the
      first pass is one batch with no retry.
    chunk_retries: Attempts per chunk.
    chunk_base_delay: Base of the exponential chunk-retry backoff.
    chunk_max_delay: Cap on the chunk-retry backoff.
    max_backfill: Maximum regular backfill waves.
    on_chunk_failure: Called before re-raising when a chunk runs out of retries.
    min_fraction: `--min_response_fraction`. While fewer than this fraction of
      the slots are filled after the regular waves, up to `max_top_up` more
      waves run, each after a `scrape_tolerance.backoff_delays` wait. 0 (the
      default) disables top-up. Slots, not pairs, are counted, so a paired batch
      can still come out short in complete pairs. A batch with every slot still
      empty is treated as an outage and gets no top-up.
    max_top_up: Maximum top-up waves.
    top_up_deadline_s: No top-up wave starts (or waits) past this many seconds
      after the scrape began.
    clock: Seconds clock for the deadline (injectable for tests).

  Returns:
    The aligned slots and counts.
  """
  start_time = clock()
  if chunk_size is None:
    tasks = [make_task(i) for i in range(n)]
    raw, initial_errors = await scrape_tolerance.gather_tolerant(
        tasks,
        desc=labels.initial_desc,
        bandwidth=min(bandwidth, len(tasks)),
        log=log,
    )
  else:
    raw, initial_errors = await _gather_chunked(
        make_task,
        n,
        chunk_size=chunk_size,
        bandwidth=bandwidth,
        labels=labels,
        log=log,
        chunk_retries=chunk_retries,
        chunk_base_delay=chunk_base_delay,
        chunk_max_delay=chunk_max_delay,
        on_chunk_failure=on_chunk_failure,
    )
  errors = list(initial_errors)

  slots: list[Any | None] = [None] * n
  if len(raw) > n:
    log(
        f"  [Scrape Warning] Received {len(raw)} responses for"
        f" {n} {labels.extras_noun}. Ignoring the extras."
    )
  for i, resp in enumerate(raw[:n]):
    if resp is None:
      # The request failed outright and `gather_tolerant` already logged it.
      continue
    if is_usable(resp):
      slots[i] = resp
    elif labels.log_blank_slots:
      log(
          f"  [Scrape Warning] Response {i + 1} was empty/blank."
          " Will attempt to refill this slot."
      )

  missing_indices = [i for i, resp in enumerate(slots) if resp is None]
  n_missing_initially = len(missing_indices)

  async def _refill(desc: str) -> list[BaseException]:
    """Re-requests every empty slot once; returns the swallowed errors."""
    backfill_tasks = [make_task(i) for i in missing_indices]
    raw_backfill, backfill_errors = await scrape_tolerance.gather_tolerant(
        backfill_tasks,
        desc=desc,
        bandwidth=min(bandwidth, len(backfill_tasks)),
        log=log,
    )
    errors.extend(backfill_errors)
    for j, resp in enumerate(raw_backfill[: len(missing_indices)]):
      if resp is not None and is_usable(resp):
        slots[missing_indices[j]] = resp
    return backfill_errors

  backfill_attempts = 0
  # Only load shedding is worth waiting out before the next wave.
  throttled = any(scrape_tolerance.is_throttling_error(e) for e in errors)
  while missing_indices and backfill_attempts < max_backfill:
    backfill_attempts += 1
    if throttled:
      # `GeminiTargetCandidate.respond` has already retried each request with
      # backoff, so the rate limiting is sustained; pause before the next wave.
      delay = scrape_tolerance.backoff_delays(backfill_attempts)
      log(
          f"  [Scrape Replenish] Load shedding detected; waiting {delay:.0f}s"
          f" before backfill attempt {backfill_attempts}/{max_backfill}."
      )
      await asyncio.sleep(delay)
    log(
        f"  [Scrape Replenish] Deficit of {len(missing_indices)}"
        f" {labels.deficit_noun}. Attempt {backfill_attempts}/{max_backfill}"
        f" for {len(missing_indices)} {labels.request_noun}..."
    )
    backfill_errors = await _refill(
        f"{labels.backfill_desc} {backfill_attempts}/{max_backfill}"
    )
    throttled = any(
        scrape_tolerance.is_throttling_error(e) for e in backfill_errors
    )
    missing_indices = [i for i, resp in enumerate(slots) if resp is None]

  # Top-up: keep refilling past `max_backfill` while the batch is short of
  # `min_fraction`, always backing off first so a struggling target is not
  # hammered. It is for partial batches: a scrape that got nothing back from
  # the first pass and every regular wave is an outage, which the round-level
  # guard (`scrape_tolerance.RoundGuard`) handles, so it is not topped up.
  top_up_attempts = 0
  if missing_indices and len(missing_indices) == n and min_fraction > 0.0:
    log(
        f"  [Scrape Top-up] Skipped: 0/{n} {labels.deficit_noun} received"
        f" after {backfill_attempts} backfill attempt(s); treating the target"
        " as down rather than topping up."
    )
  while (
      missing_indices
      and len(missing_indices) < n
      and top_up_attempts < max_top_up
      and scoring.is_short(n - len(missing_indices), n, min_fraction)
  ):
    delay = scrape_tolerance.backoff_delays(top_up_attempts + 1)
    elapsed = clock() - start_time
    if elapsed + delay > top_up_deadline_s:
      log(
          f"  [Scrape Top-up] Stopping: {n - len(missing_indices)}/{n}"
          f" {labels.deficit_noun} received (< {min_fraction:g} required),"
          f" but waiting {delay:.0f}s more would pass the"
          f" {top_up_deadline_s:.0f}s deadline ({elapsed:.0f}s elapsed)."
      )
      break
    top_up_attempts += 1
    log(
        f"  [Scrape Top-up] {n - len(missing_indices)}/{n}"
        f" {labels.deficit_noun} received (< {min_fraction:g} required);"
        f" waiting {delay:.0f}s before top-up attempt"
        f" {top_up_attempts}/{max_top_up} for {len(missing_indices)}"
        f" {labels.request_noun}..."
    )
    await asyncio.sleep(delay)
    await _refill(
        f"{labels.backfill_desc} top-up {top_up_attempts}/{max_top_up}"
    )
    missing_indices = [i for i, resp in enumerate(slots) if resp is None]

  return ScrapeOutcome(
      slots=slots,
      n_requested=n,
      n_missing_initially=n_missing_initially,
      n_backfill_attempts=backfill_attempts + top_up_attempts,
      errors=errors,
      n_top_up_attempts=top_up_attempts,
  )
