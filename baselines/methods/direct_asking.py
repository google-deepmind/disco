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

"""Direct asking -> "Self-Report" (Sheshadri et al., 2026).

Asks the target, in natural language, to describe its own behavior. Issues no
task probes, so it measures the target's metacognition rather than its
empirical behavior.
"""

import asyncio
from typing import Any

from disco.baselines import common
from disco.baselines.adapters import base as adapters_base
from disco.baselines.methods import base
from disco.verification import scoring


async def run_direct_asking_baseline(
    adapter: adapters_base.TaskProbeAdapter,
    client: Any,
    target_model: str,
    investigator_model: str,
    max_predictions: int = 10,
    semaphore: asyncio.Semaphore | None = None,
    log: common.LogFn = common.default_log,
    min_response_fraction: float = scoring.DEFAULT_MIN_RESPONSE_FRACTION,
) -> common.GenericBaselineResult:
  """1. Direct Asking Baseline -> "Self-Report" (Sheshadri et al., 2026).

  A question whose call fails or comes back blank is asked again, up to
  `base.MAX_PROBE_RESAMPLES` times with `base.resample_delay` between
  attempts. One still unanswered is left out of the synthesis prompt and not
  charged to `total_target_queries`; if none is answered the method raises
  `base.NoUsableTargetResponsesError`.

  Args:
    adapter: Task adapter.
    client: google.genai client.
    target_model: Model under audit.
    investigator_model: Model that synthesizes predictions from self-reports.
    max_predictions: Maximum number of predictions to keep.
    semaphore: Bounds concurrent model calls, if set.
    log: Log function.
    min_response_fraction: Minimum fraction of usable questions before warning.

  Returns:
    The `GenericBaselineResult` for the direct asking run.
  """
  log(
      f"\n--- [Baseline 1/5: Direct Asking] ({adapter.name},"
      f" max_predictions={max_predictions}) ---"
  )
  questions = adapter.get_direct_interrogation_prompts()

  async def _ask_one(idx: int, q: str) -> common.GenericProbeObservation:
    config = adapter.target_generate_config(temperature=0.0)
    ans, reason = await adapters_base.query_text_once(
        client, target_model, q, config, semaphore, log
    )
    retries = 0
    while reason and retries < base.MAX_PROBE_RESAMPLES:
      retries += 1
      delay = base.resample_delay(retries)
      log(
          f"  [Resample] Direct question {idx}: no usable answer ({reason})."
          f" Waiting {delay:.0f}s before retry"
          f" {retries}/{base.MAX_PROBE_RESAMPLES}..."
      )
      await asyncio.sleep(delay)
      ans, reason = await adapters_base.query_text_once(
          client, target_model, q, config, semaphore, log
      )
    if reason:
      log(
          f"  [Resample] Direct question {idx} is unanswered after"
          f" {retries + 1} attempt(s) ({reason}); leaving it out."
      )
      return common.GenericProbeObservation(
          round_idx=1,
          probe_idx=idx,
          category="direct_question",
          prompt_summary=q,
          response_summary=common.no_usable_response_summary(reason),
          score=0.0,
          passed=False,
          details={"answer": "", "attempts": retries + 1},
          n_valid_samples=0,
          failure_reason=reason,
          usable=False,
      )
    return common.GenericProbeObservation(
        round_idx=1,
        probe_idx=idx,
        category="direct_question",
        prompt_summary=q,
        response_summary=ans[:150].replace("\n", " "),
        score=1.0,
        passed=True,
        details={"answer": ans, "attempts": retries + 1},
    )

  observations = list(
      await asyncio.gather(
          *[_ask_one(idx, q) for idx, q in enumerate(questions, 1)]
      )
  )
  cov = base.coverage(observations)
  base.check_coverage(
      "DirectAskingBaseline", cov, min_response_fraction, log, noun="question"
  )
  answered = base.usable(observations)
  observations_text = "\n\n".join(
      f"Q: {o.prompt_summary}\nA: {o.details['answer']}" for o in answered
  )
  note = base.unusable_note(
      len(observations) - len(answered), len(observations), noun="question"
  )
  if note:
    observations_text += f"\n\n{note}"

  pred_text, predictions = await base.synthesize_predictions(
      adapter=adapter,
      client=client,
      investigator_model=investigator_model,
      observations_text=observations_text,
      baseline_context_header="Here is a direct self-interrogation transcript:",
      max_predictions=max_predictions,
      semaphore=semaphore,
      log=log,
  )

  return common.GenericBaselineResult(
      method_name="DirectAskingBaseline",
      origin="Self-Report (Sheshadri et al., 2026)",
      predictions=predictions,
      structured_hypotheses=common.build_structured_hypotheses_from_preds(
          predictions, adapter.suite, adapter.task_null_proportion
      ),
      reasoning=pred_text,
      observations=observations,
      total_target_queries=len(answered),
      metadata={"coverage": cov},
  )


class DirectAsking:
  """`base.DiscoveryMethod` for `run_direct_asking_baseline`."""

  name = "direct_asking"

  def plan(self, budget: int, opts: base.PlanOptions) -> dict[str, Any]:
    """Direct asking asks a fixed set of questions, whatever the budget."""
    del budget, opts  # Unused.
    return {}

  async def run(
      self,
      adapter: adapters_base.TaskProbeAdapter,
      ctx: base.RunContext,
      **plan: Any,
  ) -> common.GenericBaselineResult:
    """Runs the direct asking (self-report) baseline."""
    return await run_direct_asking_baseline(
        adapter=adapter,
        client=ctx.client,
        target_model=ctx.target_model,
        investigator_model=ctx.investigator_model,
        max_predictions=ctx.max_predictions,
        semaphore=ctx.semaphore,
        log=ctx.log,
        min_response_fraction=ctx.min_response_fraction,
        **plan,
    )
