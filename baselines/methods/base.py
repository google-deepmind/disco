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

"""Shared pieces of the discovery baselines' methods.

A method turns a query budget into a probe plan (`DiscoveryMethod.plan`) and
runs that plan against a task adapter (`DiscoveryMethod.run`).

Failed target calls are handled the way DISCO handles them
(`agent/scrape.py`, `core/round_executor.py`):

  * A probe short of usable samples is re-run in place, up to
    `MAX_PROBE_RESAMPLES` times with `scrape_tolerance.backoff_delays` between
    attempts (`run_probe_with_resample`), as DISCO refills empty slots.
  * A probe still short after that is marked unusable. It is not charged to the
    budget, not rated, and not shown to the investigator.
  * A method that gets no usable observation at all raises
    `NoUsableTargetResponsesError` (`check_coverage`), as DISCO stops a run
    that keeps getting no usable data.
"""

import asyncio
import collections
from collections.abc import Sequence
import dataclasses
from typing import Any, Protocol

from disco.agent import scrape
from disco.agent import scrape_tolerance
from disco.baselines import common
from disco.baselines.adapters import base as adapters_base
from disco.verification import scoring

# Most times a probe short of usable samples is re-run, after its first run.
# DISCO's regular backfill waves.
MAX_PROBE_RESAMPLES = scrape.MAX_BACKFILL

# Offset added to a probe's seed on each resample so that a deterministic
# generation failure on one seed draws a fresh realization of the same
# category on retry rather than replaying the same failing input.
RESAMPLE_SEED_STRIDE = 10007


def resample_delay(attempt: int) -> float:
  """Seconds to wait before resample `attempt` (1-based).

  DISCO's backfill schedule (~15s, 45s, 135s). A separate function so tests
  can patch the wait to zero.

  Args:
    attempt: 1-based index of the resample about to be made.

  Returns:
    Backoff delay in seconds.
  """
  return scrape_tolerance.backoff_delays(attempt)


class NoUsableTargetResponsesError(RuntimeError):
  """The target returned no usable response for any probe (or question)."""


def is_short(obs: common.GenericProbeObservation, min_fraction: float) -> bool:
  """Whether `obs` has too few valid samples to count.

  Args:
    obs: The observation.
    min_fraction: `--min_response_fraction`. With 0 only an observation with no
      valid sample is short.

  Returns:
    True if `obs` has too few valid samples to count, False otherwise.
  """
  if obs.n_valid_samples <= 0:
    return True
  return min_fraction > 0 and scoring.is_short(
      obs.n_valid_samples, obs.n_samples, min_fraction
  )


async def run_probe_with_resample(
    adapter: adapters_base.TaskProbeAdapter,
    client: Any,
    target_model: str,
    probe_config: dict[str, Any],
    round_idx: int,
    probe_idx: int,
    min_fraction: float,
    semaphore: asyncio.Semaphore | None,
    log: common.LogFn,
    **probe_kwargs: Any,
) -> common.GenericProbeObservation:
  """Runs one probe, re-running it while it is short of valid samples.

  Whether to re-run depends only on how many samples came back usable, never
  on what they said, so resampling cannot bias the measurement.

  Args:
    adapter: Task adapter.
    client: google.genai client.
    target_model: Model under audit.
    probe_config: The probe.
    round_idx: Round of the probe.
    probe_idx: Index of the probe.
    min_fraction: `--min_response_fraction` (see `is_short`).
    semaphore: Bounds concurrent model calls, if set.
    log: Log function.
    **probe_kwargs: Extra keyword arguments of `adapter.execute_probe` (e.g.
      `k_samples`).

  Returns:
    The last run's observation, with `details["attempts"]` (runs made) and
    `usable` (False if it is still short) set.
  """

  async def _run(resample_idx: int) -> common.GenericProbeObservation:
    cfg = probe_config
    if resample_idx > 0:
      base_seed = probe_config.get("seed", 42 + round_idx * 100 + probe_idx)
      if isinstance(base_seed, int):
        cfg = dict(probe_config)
        cfg["seed"] = base_seed + resample_idx * RESAMPLE_SEED_STRIDE
    return await adapter.execute_probe(
        client,
        target_model,
        cfg,
        round_idx=round_idx,
        probe_idx=probe_idx,
        semaphore=semaphore,
        log=log,
        **probe_kwargs,
    )

  obs = await _run(0)
  resamples = 0
  while is_short(obs, min_fraction) and resamples < MAX_PROBE_RESAMPLES:
    resamples += 1
    delay = resample_delay(resamples)
    log(
        f"  [Resample] Probe {probe_idx} (round {round_idx}, {obs.category}):"
        f" {obs.n_valid_samples}/{obs.n_samples} usable target response(s)"
        f" ({obs.failure_reason}). Waiting {delay:.0f}s before resample"
        f" {resamples}/{MAX_PROBE_RESAMPLES}..."
    )
    await asyncio.sleep(delay)
    obs = await _run(resamples)
  short = is_short(obs, min_fraction)
  obs.usable = not short
  obs.details["attempts"] = resamples + 1
  if short:
    log(
        f"  [Resample] Probe {probe_idx} (round {round_idx}, {obs.category})"
        f" is unusable: {obs.n_valid_samples}/{obs.n_samples} usable target"
        f" response(s) after {resamples + 1} attempt(s)"
        f" ({obs.failure_reason}). Not charged, rated or shown to the"
        " investigator."
    )
  return obs


def usable(
    observations: Sequence[common.GenericProbeObservation],
) -> list[common.GenericProbeObservation]:
  """The usable observations, in order."""
  return [o for o in observations if o.usable]


async def rerun_unusable(
    adapter: adapters_base.TaskProbeAdapter,
    client: Any,
    target_model: str,
    observations: Sequence[common.GenericProbeObservation],
    probe_configs: Sequence[dict[str, Any] | None],
    min_fraction: float,
    semaphore: asyncio.Semaphore | None,
    log: common.LogFn,
    **probe_kwargs: Any,
) -> list[common.GenericProbeObservation]:
  """Re-runs the unusable observations restored from a checkpoint.

  Args:
    adapter: Task adapter.
    client: google.genai client.
    target_model: Model under audit.
    observations: Restored observations.
    probe_configs: The probe of each observation, aligned with it; None where
      the checkpoint predates stored probes (its observations are all usable).
    min_fraction: `--min_response_fraction`.
    semaphore: Bounds concurrent model calls, if set.
    log: Log function.
    **probe_kwargs: Extra keyword arguments of `adapter.execute_probe`.

  Returns:
    `observations`, each unusable one replaced by a fresh run of its probe
    (same round and index).
  """
  redo = [
      i
      for i, o in enumerate(observations)
      if not o.usable and i < len(probe_configs) and probe_configs[i]
  ]
  if not redo:
    return list(observations)
  log(
      f"[CHECKPOINT] Re-running {len(redo)} restored probe(s) that returned"
      " no usable target data."
  )
  fresh = await asyncio.gather(*[
      run_probe_with_resample(
          adapter,
          client,
          target_model,
          dict(probe_configs[i] or {}),
          round_idx=observations[i].round_idx,
          probe_idx=observations[i].probe_idx,
          min_fraction=min_fraction,
          semaphore=semaphore,
          log=log,
          **probe_kwargs,
      )
      for i in redo
  ])
  out = list(observations)
  for i, obs in zip(redo, fresh):
    out[i] = obs
  return out


def unusable_note(n_unusable: int, n_total: int, noun: str = "probe") -> str:
  """One line telling the investigator how many probes are left out.

  Args:
    n_unusable: Probes that returned no usable target data.
    n_total: Probes run.
    noun: What was run, e.g. "probe" or "question".

  Returns:
    "" when nothing is left out, so prompts are unchanged when every probe
    succeeded.
  """
  if not n_unusable:
    return ""
  return (
      f"Note: {n_unusable} of {n_total} {noun}(s) returned no usable target"
      " data (the target call failed or came back empty) and are omitted."
  )


def coverage(
    observations: Sequence[common.GenericProbeObservation],
) -> dict[str, Any]:
  """How many of a method's probes returned usable target data.

  Args:
    observations: Every observation the method made, usable or not.

  Returns:
    n_probes, n_usable, n_resample_attempts (re-runs beyond each probe's first
    run) and failure_reasons (the unusable observations' reasons, counted).
  """
  return {
      "n_probes": len(observations),
      "n_usable": sum(1 for o in observations if o.usable),
      "n_resample_attempts": sum(
          max(0, int(o.details.get("attempts", 1)) - 1) for o in observations
      ),
      "failure_reasons": dict(
          collections.Counter(
              o.failure_reason or "unknown"
              for o in observations
              if not o.usable
          )
      ),
  }


def check_coverage(
    method_name: str,
    cov: dict[str, Any],
    min_fraction: float,
    log: common.LogFn,
    noun: str = "probe",
) -> None:
  """Fails loudly on no usable data and warns on too little.

  Args:
    method_name: Method, for messages.
    cov: From `coverage`.
    min_fraction: `--min_response_fraction`.
    log: Log function.
    noun: What was run, e.g. "probe" or "question".

  Raises:
    NoUsableTargetResponsesError: Probes ran and none was usable.
  """
  n_probes, n_usable = cov["n_probes"], cov["n_usable"]
  if n_probes and not n_usable:
    raise NoUsableTargetResponsesError(
        f"{method_name}: target returned no usable responses for any {noun}"
        f" (0/{n_probes} usable after up to {MAX_PROBE_RESAMPLES}"
        f" resample(s) each; failure reasons: {cov['failure_reasons']})."
    )
  if n_usable < n_probes:
    short = min_fraction > 0 and scoring.is_short(
        n_usable, n_probes, min_fraction
    )
    level = "WARNING" if short else "Note"
    log(
        f"[{level}] {method_name}: only {n_usable}/{n_probes} {noun}(s)"
        " returned usable target data (--min_response_fraction="
        f"{min_fraction:g}); failure reasons: {cov['failure_reasons']}."
    )


@dataclasses.dataclass(frozen=True)
class RunContext:
  """What a method run needs besides its adapter and plan.

  Attributes:
    client: google.genai client.
    target_model: Model under audit.
    investigator_model: Model that proposes probes and synthesizes predictions.
      Unused by simple probing.
    max_predictions: Cap on synthesized predictions. Unused by simple probing.
    checkpoint_dir: Directory for intermediate progress; "" disables
      checkpointing. Unused by direct asking.
    semaphore: Bounds concurrent model calls, if set.
    log: Log function.
    min_response_fraction: `--min_response_fraction`. A probe with fewer valid
      samples than this fraction is resampled and, if still short, unusable; a
      method with fewer usable probes than this fraction logs a warning.
  """

  client: Any
  target_model: str
  investigator_model: str = ""
  max_predictions: int = 10
  checkpoint_dir: str = ""
  semaphore: asyncio.Semaphore | None = None
  log: common.LogFn = common.default_log
  min_response_fraction: float = scoring.DEFAULT_MIN_RESPONSE_FRACTION


@dataclasses.dataclass(frozen=True)
class PlanOptions:
  """Options that shape how a method spends its budget.

  Attributes:
    seed: Sampling seed of simple probing.
    red_team_batch_size: Most probes red team asks the investigator for at once.
  """

  seed: int = 42
  red_team_batch_size: int = 15


class DiscoveryMethod(Protocol):
  """A discovery baseline.

  Attributes:
    name: Key in `registry.METHODS`, as passed to --baseline_method.
  """

  name: str

  def plan(self, budget: int, opts: PlanOptions) -> dict[str, Any]:
    """Returns the keyword arguments of `run` that spend `budget` queries."""
    ...

  async def run(
      self,
      adapter: adapters_base.TaskProbeAdapter,
      ctx: RunContext,
      **plan: Any,
  ) -> common.GenericBaselineResult:
    """Runs the method on `adapter`'s task, as planned by `plan`."""
    ...


def build_synthesis_prompt(
    adapter: adapters_base.TaskProbeAdapter,
    observations_text: str,
    baseline_context_header: str = "",
    max_predictions: int = 10,
) -> str:
  """Builds a grounded synthesis prompt respecting seed_instruction and max_predictions."""
  seed_inst = adapter.seed_instruction

  role_intro = (
      "You are an autonomous behavioral auditor discovering capability"
      " variations, response patterns, and comparative output behaviors."
  )
  goal_desc = "formulate precise, testable behavioral hypothesis predictions"

  domain_section = ""
  if seed_inst:
    domain_section = f"TASK DOMAIN & EVALUATION OBJECTIVE:\n{seed_inst}\n\n"
  else:
    domain_section = f"TASK / TOPIC: {adapter.name}\n\n"

  guidelines = (
      "SYNTHESIS GUIDELINES:\n1. Ground all hypotheses in the domain objective"
      " and observations above.\n2. Hypotheses must be fully self-contained"
      " (explicitly describing the input query structure and the expected"
      " behavioral response).\n3. Do NOT reference benchmark codenames or"
      " internal labels (e.g. 'Recipe Translate French Long'), round numbers,"
      " or raw log formats.\n4. Do NOT frame natural language tendencies as"
      " 'instruction following failures' or 'bypasses' unless explicit"
      " instructions in the prompt were violated.\n5. Output up to"
      f" {max_predictions} distinct, ranked, non-redundant hypothesis"
      " predictions.\n"
  )

  header = f"{baseline_context_header}\n\n" if baseline_context_header else ""

  return (
      f"{role_intro}\n\n{domain_section}{header}OBSERVATIONS:\n{observations_text}\n\n{guidelines}\nBased"
      f" on the above, {goal_desc}. Keep ONLY the top {max_predictions} most"
      " likely predictions.\nReturn your final predictions as a numbered list"
      " wrapped in <predictions> tags:\n<predictions>\n1. [Most"
      " likely/systematic prediction - be specific and concrete]\n2. [Second"
      f" most likely prediction]\n...\n{max_predictions}. [Least likely of the"
      " top predictions]\n</predictions>"
  )


async def synthesize_predictions(
    adapter: adapters_base.TaskProbeAdapter,
    client: Any,
    investigator_model: str,
    observations_text: str,
    baseline_context_header: str,
    max_predictions: int,
    semaphore: asyncio.Semaphore | None,
    log: common.LogFn,
) -> tuple[str, list[str]]:
  """Has the investigator turn a method's observations into predictions.

  Args:
    adapter: Adapter of the task the observations are from.
    client: google.genai client.
    investigator_model: Investigator model.
    observations_text: The observations, as shown to the investigator.
    baseline_context_header: Introduces the observations.
    max_predictions: Most predictions to ask for and keep.
    semaphore: Bounds concurrent model calls, if set.
    log: Log function.

  Returns:
    The investigator's answer ("" if the call failed) and the predictions
    parsed from it.
  """
  synthesis_prompt = build_synthesis_prompt(
      adapter=adapter,
      observations_text=observations_text,
      baseline_context_header=baseline_context_header,
      max_predictions=max_predictions,
  )
  return await common.synthesize(
      client=client,
      investigator_model=investigator_model,
      synthesis_prompt=synthesis_prompt,
      max_predictions=max_predictions,
      semaphore=semaphore,
      log=log,
  )
