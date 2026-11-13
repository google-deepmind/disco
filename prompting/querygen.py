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

"""Prompt for the isolated (context-free) query generator."""

import random
import textwrap


def get_isolated_query_generation_prompt(
    seed_instruction: str,
    hypothesis: str,
    num_queries: int,
    excluded_queries: list[str] | set[str] | None = None,
    procedural_autorater: str = "",
    paired: bool = False,
) -> str:
  """Returns a self-contained query generation prompt for an isolated subagent.

  This prompt does NOT assume any conversational context. It provides the full
  specification (seed, hypothesis) in a single shot so a fresh, zero-context
  agent can generate representative queries without being biased by failure
  history or E-values.

  Args:
    seed_instruction: The task / domain being tested.
    hypothesis: The hypothesis being tested.
    num_queries: Exact number of queries to generate.
    excluded_queries: Optional list or set of previously tested queries to avoid
      duplicating.
    procedural_autorater: If non-empty, the name of the procedural autorater
      being used (influences what kind of queries to generate).
    paired: Whether to generate matched pairs (Group A vs Group B) formatted as
      a JSON array.
  """
  exclusion_block = ""
  if excluded_queries:
    ex_list = list(excluded_queries)
    sampled = random.sample(ex_list, min(200, len(ex_list)))
    exclusion_list = "\n".join(f'  - "{q}"' for q in sampled)
    overflow_indicator = "\n  ... and more." if len(ex_list) > 200 else ""
    exclusion_block = (
        "\n\nEXCLUSION LIST — Do NOT generate queries identical or"
        " near-identical to any of these previously tested queries:\n"
        f"{exclusion_list}{overflow_indicator}"
    )

  autorater_note = ""
  if procedural_autorater:
    autorater_note = (
        "\n\nNote: Responses will be scored by a procedural autorater "
        f"('{procedural_autorater}'), not an LLM judge. Generate queries "
        "that produce measurable, scoreable outputs."
    )

  if paired:
    num_pairs = max(1, num_queries // 2)
    format_directive = textwrap.dedent(
        f"""\
      MATCHED PAIRS REQUIREMENT:
      You MUST generate exactly {num_pairs} matched pairs ({num_pairs * 2} total queries) comparing Group A and Group B.
      Each pair MUST test the exact same underlying scenario, subject, task, or concept under the two comparative conditions defined in the hypothesis (Group A condition vs Group B condition).

      Output the pairs as a valid JSON array of objects with the following schema:
      ```json
      [
        {{
          "concept_or_scenario": "<description of the matched scenario/item>",
          "group_a": "<The complete, self-contained prompt for Group A>",
          "group_b": "<The complete, self-contained prompt for Group B>"
        }}
      ]
      ```

      CRITICAL RULES:
      - Do NOT include labels like '[A]', '[B]', '[Group A]', '[Group B]', or metadata tags in 'group_a' or 'group_b'. Both prompts must be natural, clean requests as if asked directly by a user.
      - Output ONLY the JSON array (inside a ```json codeblock or plain JSON). Do not explain your reasoning or call any tools."""
    )
  else:
    format_directive = textwrap.dedent("""\
      Output the queries as a numbered list:
      1. <query_text>
      2. <query_text>
      ...

      Generate ONLY the queries. Do not explain your reasoning or call
      any tools.""")

  return textwrap.dedent(f"""\
    You are a query generator for a statistical hypothesis testing system.
    Your ONLY job is to generate diverse, representative test queries.

    TASK DOMAIN: {seed_instruction}
    HYPOTHESIS (H): {hypothesis}

    Generate exactly {num_queries} queries that test H.

    Requirements:
    - TEST NEUTRALITY: The goal is to test whether the hypothesized behavior
      or quirk occurs naturally/spontaneously under realistic prompts in this
      domain, NOT to directly instruct or command the model to exhibit the
      behavior (unless the hypothesis specifically claims the behavior occurs
      in response to an explicit user request).
    - Each query must be a SINGLE-TURN, ZERO-CONTEXT prompt. The target
      model has no conversational memory and no access to external
      documents. Each query must be completely self-contained.
    - If the test requires the model to process an input artifact (text,
      code, data), you MUST embed the complete artifact in the query.
    - Maximize DIVERSITY: vary phrasing, subject matter, context, edge
      cases, and complexity across the {num_queries} queries. They should
      be representative samples from the space of all possible queries
      that test H, not clustered around a single template.
    - Do NOT include group labels, tags, or metadata in query text.
    {exclusion_block}{autorater_note}

    {format_directive}""")
