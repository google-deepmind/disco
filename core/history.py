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

"""History and context text shown to the DISCO discovery agent.

Builders for the cross-hypothesis failure summary (and its LLM compression),
the selector's hypothesis history, and the per-round context prompt given to
each fresh round agent. Which details appear depends on the mode: DISCO mode
shows only coarsened status, probe counts and transition scores; baseline
modes also show aggregated e-values and concrete examples.

The prompt text is pinned by the smoke-test transcripts; change it only
deliberately.
"""

from collections.abc import Callable

from disco import backbone_client as backbone_lib
from disco.core import checkpoint as checkpoint_lib
from disco.core import config as config_lib
from disco.core import records as records_lib
from disco.prompting import round_prompts


def build_failure_summary(
    hypothesis_groups: dict[str, records_lib.HypothesisRecord],
    max_examples: int = 6,
) -> str:
  """Builds a concise summary of concrete failure examples across all hypotheses.

  Collects failure records from all HypothesisRecords and formats them for
  injection into prompts.

  Args:
    hypothesis_groups: The current map of hypothesis ID -> HypothesisRecord.
    max_examples: Maximum number of failure examples to include (prevents prompt
      size explosion).

  Returns:
    A formatted string summarising past failures, or an empty string if none.
  """
  all_failures: list[tuple[str, dict[str, str]]] = []
  for h_id, h_data in hypothesis_groups.items():
    for fail in h_data.failures:
      all_failures.append((h_id, fail))

  if not all_failures:
    return ""

  # Most-recent failures first; cap at max_examples.
  examples = all_failures[-max_examples:]

  lines: list[str] = []
  lines.append(
      f"Concrete examples seen so far ({len(all_failures)} total,"
      f" showing {len(examples)}):"
  )
  for i, (h_id, fail) in enumerate(examples, 1):
    prompt_snip = fail.get("prompt", "")[:200]
    response_snip = fail.get("response", "")[:300]
    reasoning_snip = fail.get("reasoning", "")[:200]
    judgment = fail.get("judgment", "Example")
    lines.append(
        f"\n  Example {i} [{judgment}] (hypothesis: {h_id}):"
        f"\n    Prompt   : {prompt_snip!r}"
        f"\n    Response : {response_snip!r}"
        f"\n    Reasoning: {reasoning_snip!r}"
    )

  lines.append(
      "\nUse this information to decide where a fruitful direction to probe"
      " might be. Remember your goal is to find as many unique, verified"
      " behavior patterns as possible."
  )

  return "\n".join(lines)


async def summarize_failure_block(
    raw_summary: str,
    seed_instruction: str,
    model: str,
    log_fn: Callable[..., None],
) -> str:
  """Compresses a raw failure/example summary block into 2-3 sentences via LLM.

  Args:
    raw_summary: The raw output of build_failure_summary().
    seed_instruction: The seed task/instruction for context.
    model: Model name for the summarization LLM call.
    log_fn: Logger function.

  Returns:
    A compressed 2-3 sentence summary string. Falls back to truncation
    if the LLM call fails.
  """
  if not raw_summary or not raw_summary.strip():
    return ""

  prompt = round_prompts.failure_block_compression_prompt(
      raw_summary, seed_instruction
  )

  backbone = backbone_lib.get_backbone_client(model)
  try:
    text = await backbone.generate(
        model=model,
        prompt=prompt,
    )
    if text and text.strip():
      return text.strip()
  except backbone_lib.LLM_CALL_ERRORS as e:
    log_fn(
        f"  [Warning] Failure summary compression LLM call failed: {e}. Falling"
        " back to truncation."
    )

  return raw_summary[:500].strip()


def _lookup_e_value(
    h_id: str,
    running_e_values: dict[str, float] | None,
    final_e_values: dict[str, float] | None = None,
) -> float:
  """Returns `h_id`'s running or final e-value, defaulting to 1.0."""
  if running_e_values and h_id in running_e_values:
    return running_e_values[h_id]
  if final_e_values and h_id in final_e_values:
    return final_e_values[h_id]
  return 1.0


def _evidence_tag(
    n_g: int,
    status: str,
    curr_e: float,
    e_min: float,
    e_max: float,
) -> str:
  """Returns the qualitative evidence tag based on coarsened bin / thresholds."""
  if n_g == 0:
    return ""
  if status == "H" or curr_e >= e_max:
    return " [STRONG EVIDENCE]"
  if status == "L" or status == "B_0" or curr_e < e_min:
    return " [WEAK EVIDENCE]"
  return " [MODERATE EVIDENCE]"


def disco_history_summary(
    hypothesis_groups: dict[str, records_lib.HypothesisRecord],
    active_hypotheses: set[str],
    terminal_statuses: dict[str, str],
    probe_counts: dict[str, int],
    transition_scores: dict[str, list[float]],
    e_min: float,
    e_max: float,
    show_failures: bool = False,
    baseline_mode: bool = False,
    baseline_type1_mode: bool = False,
    running_e_values: dict[str, float] | None = None,
    final_e_values: dict[str, float] | None = None,
) -> str:
  """DISCO or baseline history summary.

  In DISCO mode: shows hypothesis text, coarsened status D_g, probe count n_g,
  and per-probe transition scores. Does NOT show running e-values, p-values,
  or model outputs.
  In baseline mode: shows hypothesis text, probe count, aggregated e-values,
  qualitative evidence tags, and failure examples if available.
  When show_failures is True, also includes concrete failure examples
  for each hypothesis.

  Args:
    hypothesis_groups: Tracked hypothesis records keyed by hypothesis ID.
    active_hypotheses: Set of currently active hypothesis IDs.
    terminal_statuses: Terminal or active bin status per hypothesis ID.
    probe_counts: Number of probes executed per hypothesis ID.
    transition_scores: Per-probe coarsened transition scores per hypothesis ID.
    e_min: Lower strip boundary e_min.
    e_max: Upper strip boundary e_max.
    show_failures: Whether to include concrete failure examples in the summary.
    baseline_mode: Whether unrestricted e-BH baseline mode is enabled.
    baseline_type1_mode: Whether Type-I baseline mode is enabled.
    running_e_values: Currently active running e-values per hypothesis ID.
    final_e_values: Finalized e-values for deactivated hypotheses.

  Returns:
    Formatted multi-line hypothesis history summary string.
  """
  is_baseline = baseline_mode or baseline_type1_mode
  if baseline_type1_mode:
    header = "Baseline Mode (Type-I Error)"
  elif baseline_mode:
    header = "Baseline Mode"
  else:
    header = "DISCO Mode"
  lines = [f"\n--- Hypothesis History ({header}) ---\n"]
  if not hypothesis_groups:
    lines.append("No hypotheses yet.\n")
  else:
    for h_id, h_data in hypothesis_groups.items():
      n_g = probe_counts.get(h_id, 0)
      status = terminal_statuses.get(h_id, "A")
      curr_e = _lookup_e_value(h_id, running_e_values, final_e_values)
      evidence_tag = _evidence_tag(n_g, status, curr_e, e_min, e_max)

      if baseline_type1_mode:
        if h_id in active_hypotheses:
          status_tag = (
              f" [ACTIVE, {n_g} probes, Aggregated"
              f" E={curr_e:.4f}]{evidence_tag}"
          )
        elif status == "H":
          status_tag = (
              f" [FROZEN (crossed 1/alpha), {n_g} probes, Aggregated"
              f" E={curr_e:.4f}]{evidence_tag}"
          )
        elif status == "L":
          status_tag = (
              f" [INACTIVE - EXITED-LOW, {n_g} probes, Aggregated"
              f" E={curr_e:.4f}]{evidence_tag}"
          )
        else:
          status_tag = (
              f" [INACTIVE ({status}), {n_g} probes, Aggregated"
              f" E={curr_e:.4f}]{evidence_tag}"
          )
      elif baseline_mode:
        if h_id in active_hypotheses:
          status_tag = (
              f" [ACTIVE, {n_g} probes, Aggregated"
              f" E={curr_e:.4f}]{evidence_tag}"
          )
        elif status == "H":
          status_tag = (
              f" [INACTIVE - EXITED-HIGH (crossed K_bar/alpha), {n_g} probes,"
              f" Aggregated E={curr_e:.4f}]{evidence_tag}"
          )
        elif status == "L":
          status_tag = (
              f" [INACTIVE - EXITED-LOW, {n_g} probes, Aggregated"
              f" E={curr_e:.4f}]{evidence_tag}"
          )
        else:
          status_tag = (
              f" [INACTIVE ({status}), {n_g} probes, Aggregated"
              f" E={curr_e:.4f}]{evidence_tag}"
          )
      else:
        if h_id in active_hypotheses:
          status_tag = f" [ACTIVE, {n_g} probes]{evidence_tag}"
        elif status == "H":
          status_tag = f" [INACTIVE - EXITED-HIGH, {n_g} probes]{evidence_tag}"
        elif status == "L":
          status_tag = f" [INACTIVE - EXITED-LOW, {n_g} probes]{evidence_tag}"
        else:
          status_tag = (
              f" [INACTIVE - TIMED-OUT ({status}), {n_g} probes]{evidence_tag}"
          )

      lines.append(f"ID: {h_id} ({h_data.text}){status_tag}\n")

      t_scores = transition_scores.get(h_id, [])
      if t_scores and not is_baseline:
        lines.append(
            "  Probe transition scores: "
            + ", ".join(f"{ts:+.3f}" for ts in t_scores)
            + "\n"
        )

      if show_failures and h_data.failures:
        lines.append(h_data.format_failures(indent="  "))

  if baseline_type1_mode:
    lines.append(
        "Hypotheses crossing 1/alpha are frozen upon crossing. Direct Type-I"
        " thresholding (no e-BH) at"
        " finalization.\n------------------------------------------\n\n"
    )
  elif baseline_mode:
    lines.append(
        "All hypotheses will be tested at the end via e-BH.\n"
        "------------------------------------------\n\n"
    )
  else:
    lines.append(
        f"Thresholds: e_min={e_min}, e_max={e_max}\n"
        "All hypotheses will be tested at the end via e-BH.\n"
        "------------------------------------------\n\n"
    )
  return "".join(lines)


def build_disco_round_context(
    config: config_lib.DiscoConfig,
    state: checkpoint_lib.RunState,
    h_text: str,
    h_record: records_lib.HypothesisRecord,
    h_id: str,
    cross_hyp_summary: str = "",
) -> str:
  """Builds a context prompt for the fresh round agent.

  In DISCO mode: shows hypothesis text, per-probe transition scores, and
  coarsened status.
  In baseline mode: shows hypothesis text, aggregated e-values, and all
  failure examples.

  Args:
    config: Run configuration.
    state: Current run state.
    h_text: Hypothesis text.
    h_record: The hypothesis's record.
    h_id: Hypothesis id.
    cross_hyp_summary: Precomputed cross-hypothesis summary; built from `state`
      when empty and the mode shows all failures.

  Returns:
    The context prompt.
  """
  n_g = state.probe_counts.get(h_id, 0)
  status = state.terminal_statuses.get(h_id, "A")
  t_scores = state.transition_scores.get(h_id, [])

  curr_e = _lookup_e_value(h_id, state.running_e_values, state.final_e_values)
  evidence_tag = _evidence_tag(n_g, status, curr_e, config.e_min, config.e_max)

  is_baseline = config.baseline_mode or config.baseline_type1_mode
  if (
      config.show_all_failures
      and config.num_active_bins > 0
      and not is_baseline
  ):
    raise ValueError(
        "show_all_failures=True is not permitted when num_active_bins > 0 in"
        " DISCO mode; set num_active_bins=0 or show_all_failures=False."
    )
  if is_baseline:
    prompt = (
        "You are a fresh agent helper for a DISCO probe round. You are"
        f" assisting with the hypothesis:\nHYPOTHESIS: {h_text}\n\n"
        f"Status: Active (probed {n_g} times, Aggregated E-value:"
        f" {curr_e:.4f}){evidence_tag}\n\n"
    )
  else:
    prompt = (
        "You are a fresh agent helper for a DISCO probe round with"
        " coarsened e-value feedback. You are assisting with the"
        f" hypothesis:\nHYPOTHESIS: {h_text}\n\n"
        f"Status: Active (probed {n_g} times, coarsened status:"
        f" {status}){evidence_tag}\n"
        f"Thresholds: e_min={config.e_min},"
        f" e_max={config.e_max}\n\n"
    )

  if t_scores and not is_baseline:
    prompt += "Prior probe transition scores for this hypothesis:\n"
    for i, ts in enumerate(t_scores):
      prompt += f"  - Probe {i+1}: transition score: {ts:+.3f}\n"
    prompt += "\n"

  show_intra = (
      config.show_intra_hypothesis_failures
      or config.show_all_failures
      or is_baseline
  )

  # Intra-hypothesis reflection summary (from this hypothesis's prior probes).
  if show_intra and h_record.failure_reflections:
    intra_header = (
        "--- Summary of Salient Examples from Prior Probes of THIS"
        " Hypothesis ---\n"
    )
    intra_footer = "--- End of Summary ---\n\n"
    prompt += (
        f"{intra_header}{h_record.failure_reflections[-1]}\n{intra_footer}"
    )

  # Previously tested queries context
  if h_record.executed_queries:
    prompt += (
        "Previously tested queries for this hypothesis"
        f" ({len(h_record.executed_queries)} total):\n"
    )
    for q in sorted(list(h_record.executed_queries))[:10]:
      prompt += f'  - "{q}"\n'
    if len(h_record.executed_queries) > 10:
      prompt += f"  ... and {len(h_record.executed_queries) - 10} more.\n"
    prompt += (
        "Ensure you generate novel, distinct queries that do NOT duplicate"
        " previous prompts.\n\n"
    )

  # Intra-hypothesis failures / salient examples (from this hypothesis's prior
  # probes).
  if show_intra and h_record.failures:
    prompt += "Concrete examples from prior probes of THIS hypothesis:\n"
    for i, fail in enumerate(h_record.failures):
      prompt += (
          f"  Example {i+1} ({fail.get('judgment', 'FAILURE')}):\n"
          f"    - Prompt: \"{fail.get('prompt', '')}\"\n"
          f"    - Model Response: \"{fail.get('response', '')}\"\n"
          f"    - Judge Reasoning: \"{fail.get('reasoning', '')}\"\n"
      )
    prompt += "\n"

  # Cross-hypothesis failures (from ALL hypotheses).
  if not cross_hyp_summary and (config.show_all_failures or is_baseline):
    cross_hyp_summary = build_failure_summary(state.hypothesis_groups)
  if cross_hyp_summary:
    cross_header = "--- Summary of Salient Examples Across ALL Hypotheses ---\n"
    cross_footer = "--- End of Cross-Hypothesis Summary ---\n\n"
    prompt += f"{cross_header}{cross_hyp_summary}\n{cross_footer}"

  if config.show_all_failures or is_baseline:
    prompt += (
        "You can see salient examples from this and other hypotheses - use"
        " them as extra information if necessary.\n\n"
    )
  elif show_intra:
    prompt += (
        "You can see salient examples from this hypothesis - use them as"
        " extra information if necessary.\n\n"
    )
  else:
    prompt += (
        "With coarsened e-value feedback, you do NOT see the results"
        " (model responses, judgments, failure examples) of previous"
        " rounds. You only see which queries were tried and the coarsened"
        " transition scores of prior probes.\n\n"
    )

  prompt += (
      "Please acknowledge that you have received this hypothesis and"
      " context, and wait for the next instruction. Do not start"
      " generating queries or running tools yet."
  )
  return prompt
