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

"""System-agnostic prompts for independent hypothesis precision verification.

Provides prompts that do not assume DISCO discovery context, suitable for
verifying hypotheses from any source (baselines, human-written, etc.).
Re-exported by `verification_prompts`, which satisfies the FDRPromptsModule
protocol.
"""

import textwrap
from typing import Any


def get_fresh_agent_intro_prompt(
    target_model: str,
    seed_instruction: str,
    threshold: float,
    chunk_size: int = 25,
    baseline_mode: bool = False,
) -> str:
  """Returns the intro prompt for a verification agent running Steps 3-5 only.

  Args:
    target_model: The name of the model being tested.
    seed_instruction: The task / domain being evaluated.
    threshold: The E-value threshold for verified hypotheses.
    chunk_size: Exact number of queries to allocate per round.
    baseline_mode: Whether running in baseline mode (offline e-BH evaluation).
  """
  del baseline_mode  # The verification intro is the same in every mode.
  steps_desc = (
      "3. Generate queries directly to test the hypothesis H using the"
      " specific criteria defined in H.\n"
      "    4. Execute them using your scraping and analysis tools.\n"
      "    5. Calculate and report the P-value using statistics tools."
  )
  guidance_query = (
      "- Focus on patterns a normal user would encounter in practice.\n"
      "    - Only do what is requested in each step. When asked to generate"
      " queries, produce them directly based on the hypothesis"
      " specifications — do not call tools or execute anything else."
  )
  start_note = (
      "Don't start yet — wait for the hypothesis and context to be provided."
  )

  return textwrap.dedent(f"""\
    You are an independent verification agent. You are testing whether a \
    claimed hypothesis about the target model '{target_model}' on the \
    topic / task: '{seed_instruction}' is reproducible and statistically valid.

    You have been assigned a specific Hypothesis (H) to investigate. \
    You will NOT propose hypotheses — that has already been done. \
    Your job is to run exactly one round for the hypothesis you receive:

    {steps_desc}

    E-values greater than {threshold:.4f} are considered verified evidence \
    for the hypothesis — this is for your information only.

    Important guidelines:
    - Think carefully about the null proportion. When testing if a \
    behavior or failure occurs at an elevated rate, calibrate against a \
    reasonable null baseline proportion (e.g. standard expectation or 50%).
    - Use 1-sided tests when you want to show a rate is above or \
    below a threshold. Use 2-sided tests to compare the model's \
    behavior across two different settings.
    {guidance_query}
    - Be strategic with your query budget. You will use exactly \
    {chunk_size} queries per round to ensure statistical \
    validity and consistent evidence collection.

    {start_note}""")


def get_precision_round_context_prompt(
    h_text: str,
    prior_e_values: list[Any],
    failures: list[dict[str, str]] | None = None,
    failure_reflections: list[str] | None = None,
) -> str:
  """Returns the context-injection prompt for precision verification round.

  Args:
    h_text: Hypothesis text.
    prior_e_values: `(label, e_value)` pairs of the prior rounds of this
      verification run (`records.EValueEntry` unpacks to these).
    failures: Concrete examples from prior rounds, if shown.
    failure_reflections: Reflection summaries from prior rounds, if shown.
  """
  prompt = (
      "You are an independent verification agent. "
      "Your goal is to test if the following hypothesis H is true "
      f"and reproducible:\nHYPOTHESIS: {h_text}\n\n"
  )

  if prior_e_values:
    prompt += "Prior rounds of this verification run:\n"
    for i, (_, e_val) in enumerate(prior_e_values):
      prompt += f"- Round {i + 1} | E-value: {e_val:.4f}\n"
    prompt += "\n"

  if failures:
    prompt += (
        "Concrete evidence examples from prior probes of THIS hypothesis:\n"
    )
    for i, fail in enumerate(failures):
      prompt += (
          f"  Example {i+1} ({fail.get('judgment', 'EVIDENCE')}):\n"
          f"    - Prompt: \"{fail.get('prompt', '')}\"\n"
          f"    - Model Response: \"{fail.get('response', '')}\"\n"
          f"    - Judge Reasoning: \"{fail.get('reasoning', '')}\"\n"
      )
    prompt += "\n"

  if failure_reflections:
    prompt += (
        "--- Summary of Salient Examples from Prior Probes of THIS"
        " Hypothesis ---\n"
    )
    prompt += f"{failure_reflections[-1]}\n"
    prompt += "--- End of Summary ---\n\n"

  prompt += (
      "CRITICAL: Do NOT assume the hypothesis is true. You must test it "
      "objectively. Test cases will attempt to disprove it as well as prove "
      "it. We want to see if we can replicate the claimed pattern.\n\n"
      "Please acknowledge that you have received this hypothesis and context,"
      " and wait for the next instruction. Do not start generating queries or"
      " running tools yet."
  )
  return prompt
