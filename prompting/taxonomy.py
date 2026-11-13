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

"""Prompts for taxonomy-driven query generation (arXiv:2603.29791)."""

import textwrap


def get_taxonomy_generation_prompt(
    hypothesis: str,
    seed_instruction: str,
    taxonomy_depth: int = 3,
    best_of_n: int = 3,
    paired: bool = False,
) -> str:
  """Returns a prompt asking the agent to generate a taxonomy for the hypothesis.

  Implements the taxonomy generation step from the taxonomy-driven query
  generation framework (arXiv:2603.29791): the agent
  decomposes the hypothesis space into hierarchical factors of variation,
  enabling structured coverage of the query space.

  Args:
    hypothesis: The current hypothesis being tested.
    seed_instruction: The seed instruction for the DISCO run.
    taxonomy_depth: Desired depth of the taxonomy tree.
    best_of_n: Number of independent taxonomy proposals to generate (Best-of-N).
    paired: Whether to enforce paired (Group A vs Group B) leaf nodes.
  """
  paired_guidance = ""
  if paired:
    paired_guidance = textwrap.dedent("""
    PAIRED COMPARISON MODE:
    This hypothesis compares Group A against Group B. Your taxonomy MUST \
    generate paired leaf nodes. Each L3 leaf node must be a matched pair \
    in the format:
        L3: [A] <Group A item/scenario> vs [B] <Group B item/scenario>

    For example:
        L3: [A] Beef Wellington vs [B] Grilled beef medallions
        L3: [A] Chicken tikka masala with naan vs [B] Butter chicken with rice

    Each pair should be a fair, matched comparison where the ONLY systematic \
    difference is the Group A vs Group B distinction defined in the hypothesis. \
    Do NOT create unpaired leaf nodes.
""")

  return textwrap.dedent(f"""\
    You are generating queries using a taxonomy-driven reasoning framework. \
    Instead of directly listing queries, you will first create a structured \
    taxonomy that maps the space of possible queries for this hypothesis.

    HYPOTHESIS: {hypothesis}
    SEED: {seed_instruction}

    Step 1: Identify the key FACTORS OF VARIATION that differentiate \
    the types of queries you could generate to test this hypothesis. \
    Think about what dimensions make one test query meaningfully different \
    from another (e.g., subject matter, phrasing style, edge case type, \
    context, etc.).

    Step 2: For each factor, generate a HIERARCHICAL TAXONOMY of depth \
    {taxonomy_depth}. Start with broad categories at the root and get \
    increasingly specific at each level. Generate at least {best_of_n} \
    candidate children per node, then self-critique to refine: add missing \
    categories, remove redundant ones, and merge overlapping concepts.

    Output the taxonomy in the following structured format:
    ```
    FACTOR: <factor_name>
      L1: <category_1>
        L2: <subcategory_1a>
          L3: <specific_instance_1a_i>
          L3: <specific_instance_1a_ii>
        L2: <subcategory_1b>
          L3: <specific_instance_1b_i>
      L1: <category_2>
        L2: <subcategory_2a>
          ...
    FACTOR: <factor_name_2>
      ...
    ```

    Guidelines:
    - Aim for 2-4 factors and completeness at each level.
    - Each leaf node should correspond to a concrete, testable query angle.
    - Prioritize coverage: ensure edge cases, uncommon scenarios, and \
    boundary conditions are represented, not just common cases.
    - Apply self-critique: after generating each level, check if important \
    categories are missing before moving to the next level.
{paired_guidance}
    Output only the taxonomy. Do not generate queries yet.""")


def get_taxonomy_meta_prompt(
    taxonomy_nodes: list[str],
    hypothesis: str,
    num_queries: int,
    seed_instruction: str = "",
    paired: bool = False,
) -> str:
  """Returns a prompt asking the agent to convert taxonomy node combinations into queries.

  Implements the meta-prompting step from taxonomy-driven query generation
  (arXiv:2603.29791): given sampled taxonomy nodes,
  the agent synthesizes them into concrete, diverse test queries.

  Args:
    taxonomy_nodes: List of taxonomy node-set descriptions (sampled
      combinations).
    hypothesis: The current hypothesis.
    num_queries: Exact number of queries to produce.
    seed_instruction: The seed instruction for the DISCO run.
    paired: Whether queries must be formatted in strict (A, B) alternating
      order.
  """
  nodes_formatted = "\n".join(
      f"  {i+1}. {node}" for i, node in enumerate(taxonomy_nodes)
  )
  seed_block = f"\n    SEED: {seed_instruction}" if seed_instruction else ""

  if paired:
    pairing_directive = textwrap.dedent(
        f"""\
    You MUST generate exactly {num_queries} queries total, organized as \
    {num_queries // 2} MATCHED PAIRS. The node combinations below are \
    already arranged in (Group A, Group B) alternating order. Generate \
    one query per node combination, preserving this exact order:

    - Odd-numbered queries (1, 3, 5, ...) → Group A queries
    - Even-numbered queries (2, 4, 6, ...) → Group B queries

    CRITICAL RULES:
    - Strictly preserve the alternating (A, B) structure: Query 1 is Group A, Query 2 is Group B, Query 3 is Group A, Query 4 is Group B, etc.
    - Do NOT include group labels, tags like [A] or [B], or any identifiers in the query text itself. Each query must be a clean, natural request as if a real user were asking it."""
    )
  else:
    pairing_directive = textwrap.dedent(f"""\
    You MUST generate exactly {num_queries} queries total.
    If there are fewer node combinations than queries needed, generate \
    multiple diverse interpretations per combination. If there are more, \
    select the most diverse and relevant subset.""")

  return textwrap.dedent(f"""\
    Now convert the following taxonomy node combinations into concrete \
    test queries. Each node combination specifies the REQUIREMENTS that \
    the generated query must satisfy.

    HYPOTHESIS: {hypothesis}{seed_block}

    Node combinations to convert into queries:
    {nodes_formatted}

    For each node combination, generate a concrete query that:
    1. Satisfies ALL the specified taxonomy node requirements.
    2. Is a natural, realistic query that a real user might make.
    3. Is distinct from all other queries (maximize local diversity).

    CRITICAL — Self-Containment Requirement:
    Every query will be sent to the target model in a SINGLE-TURN, ZERO-CONTEXT session. The model has no conversational memory and no access to external documents. Therefore:
    a) Each query MUST be completely self-contained and immediately answerable without follow-up.
    b) If the task requires the model to edit, proofread, debug, translate, summarize, or reason over an input artifact (text, code, data, scenario), you MUST explicitly construct and embed the complete artifact within the query itself. Do NOT write "please review this draft" without including the draft text.
    c) For comparison or opinion questions, frame queries to elicit substantive explanatory responses rather than closed yes/no confirmations. Prefer "Compare X and Y" over "Do you prefer X or Y?"
    d) Do NOT write prompts that assume multi-turn interaction or reference prior context.

    {pairing_directive}

    Output each query as a numbered list:
    1. <query_text>
    2. <query_text>
    ...

    Generate ONLY the queries. Do not call execution tools.""")


def get_taxonomy_critic_prompt(
    queries: list[str],
    taxonomy_nodes: list[str],
    hypothesis: str,
    seed_instruction: str = "",
) -> str:
  """Returns a prompt for critic-based filtering of generated queries.

  Implements the double-critic step from taxonomy-driven query generation
  (arXiv:2603.29791): independently assess whether
  each query correctly addresses its taxonomy requirements and is a valid
  test of the hypothesis. Queries failing the critic check are flagged
  for regeneration.

  Args:
    queries: List of candidate queries to evaluate.
    taxonomy_nodes: The taxonomy node-sets each query was generated from.
    hypothesis: The current hypothesis.
    seed_instruction: The seed instruction for the DISCO run.
  """
  pairs_formatted = "\n".join(
      f"  Query {i+1}: {q}\n  Requirements:"
      f" {taxonomy_nodes[i] if i < len(taxonomy_nodes) else 'N/A'}"
      for i, q in enumerate(queries)
  )
  seed_block = f"\n    SEED: {seed_instruction}" if seed_instruction else ""
  return textwrap.dedent(f"""\
    You are a CRITIC evaluating the quality of generated test queries. \
    For each query, independently assess three criteria:

    HYPOTHESIS: {hypothesis}{seed_block}

    Queries and their taxonomy requirements:
    {pairs_formatted}

    For EACH query, evaluate:
    1. REQUIREMENT ADHERENCE: Does the query satisfy ALL the taxonomy \
    node requirements it was generated from? (YES/NO with brief reason)
    2. HYPOTHESIS RELEVANCE & SCOPE FIDELITY: Is this query a valid, non-trivial \
    test whose subject matter strictly and authentically falls within the target \
    population, category, and conditions defined in the HYPOTHESIS? (YES/NO with brief reason)
    - REJECT if the query's subject matter, entity, or task falls outside the explicit boundaries, prerequisites, or target category defined in the HYPOTHESIS.
    - In paired comparison mode, REJECT if the pair compares entities from unrelated categories or introduces confounding variables beyond the distinction defined in the HYPOTHESIS.
    3. EXECUTION VIABILITY: Is this query self-contained and immediately \
    executable in a single-turn, zero-context session?
    - REJECT if the query asks the model to review, edit, or analyze content that is NOT provided within the query itself.
    - REJECT if the query would most naturally elicit a clarifying question (e.g., "What text would you like me to correct?").
    - REJECT if the query is a closed yes/no question that is likely to produce a single-word response.

    A query PASSES the critic if ALL THREE checks are YES.
    A query FAILS if any check is NO.

    Output your assessment for each query in this format:
    QUERY_1: PASS or FAIL
    QUERY_2: PASS or FAIL
    ...

    After the individual assessments, output a summary line:
    PASSING_INDICES: <comma-separated list of passing query numbers>""")
