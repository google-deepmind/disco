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

"""LLM selection of the most informative examples from a probing round."""

from collections.abc import Callable
from typing import Any

from disco import backbone_client as backbone_lib
from disco.core import records as records_lib
from disco.parsing import extract
from disco.prompting import round_prompts


async def select_salient_examples(
    candidate_judgments: list[dict[str, Any]],
    seed_instruction: str,
    hypothesis_groups: dict[str, records_lib.HypothesisRecord],
    current_h_id: str,
    current_h_text: str,
    model: str,
    log_fn: Callable[..., None],
    terminal_statuses: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
  """Extracts the 3 most salient / informative examples from the round's judgments using an LLM.

  Args:
    candidate_judgments: The list of judgment dicts from agent.last_judgments.
    seed_instruction: The seed task/instruction.
    hypothesis_groups: Map of tracked hypotheses.
    current_h_id: The ID of the hypothesis being probed.
    current_h_text: The text of the hypothesis being probed.
    model: Model name for LLM selection call.
    log_fn: Logger function.
    terminal_statuses: Optional coarsened terminal statuses dict.

  Returns:
    List of up to 3 selected judgment dictionaries.
  """
  pool: list[dict[str, Any]] = []
  for j in candidate_judgments:
    prompt = j.get("prompt", "")
    response = j.get("response", "")
    if not prompt and not response:
      continue
    judg_val = j.get("judgment", "")
    reasoning = j.get("reasoning", "")
    if judg_val:
      display_judgment = judg_val
    elif "value" in j:
      display_judgment = f"Score: {j.get('value', 0):.1f}"
    else:
      display_judgment = ""
    pool.append({
        "prompt": prompt,
        "response": response,
        "judgment": display_judgment,
        "reasoning": reasoning,
    })

  if not pool:
    return []

  if len(pool) <= 3:
    return pool

  # 2. Build selection prompt
  history: list[round_prompts.SalientHistoryEntry] = []
  for h_id, h_data in hypothesis_groups.items():
    status_str = ""
    if terminal_statuses and h_id in terminal_statuses:
      status_str = f" [{terminal_statuses[h_id]}]"
    history.append(
        round_prompts.SalientHistoryEntry(
            h_id=h_id,
            text=h_data.text,
            status=status_str,
            sub_hypotheses=h_data.sub_hypotheses,
        )
    )

  selection_prompt = round_prompts.salient_selection_prompt(
      seed_instruction, current_h_id, current_h_text, history, pool
  )

  backbone = backbone_lib.get_backbone_client(model)
  selected_indices = [0, 1, 2]
  try:
    text = await backbone.generate(
        model=model,
        prompt=selection_prompt,
    )
    if text:
      parsed = extract.parse_selected_example_indices(
          text, len(pool), num_to_select=3
      )
      if parsed:
        selected_indices = parsed
  except backbone_lib.LLM_CALL_ERRORS as e:
    log_fn(
        f"  [Warning] Salient example selection LLM call failed: {e}. Falling"
        " back to default slicing."
    )

  selected_examples = [pool[i] for i in selected_indices if i < len(pool)]
  return selected_examples[:3]
