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

"""Taxonomy-driven query generation utilities (arXiv:2603.29791).

Implements the taxonomy-driven framework for reasoning-driven, taxonomy-based
query generation as an alternative to direct flat query listing in DISCO.

The pipeline consists of five phases:
  1. Taxonomy Generation — LLM builds hierarchical factors of variation.
  2. Taxonomic Sampling — Leaf nodes are sampled and combined.
  3. Meta-Prompting — Node-sets are converted into concrete queries.
  4. Critic Filtering — Double-critic quality check.
  5. Backfill / Replenishment — Replenishes rejected queries to meet target
  count.
"""

import random
import re
from typing import Callable

from disco.agent import senders as agent_senders
from disco.core import config as config_lib
from disco.core import prompts_protocol


def parse_taxonomy_leaf_nodes(taxonomy_text: str) -> list[str]:
  """Parses leaf nodes from a hierarchical taxonomy response.

  Extracts the deepest-level nodes from each factor's taxonomy tree.
  These leaf nodes represent the most specific query angles.

  Args:
    taxonomy_text: The raw taxonomy text from the agent.

  Returns:
    A list of leaf node path strings (e.g., "factor > L1 > L2 > L3").
  """
  leaf_nodes: list[str] = []
  current_path: list[str] = []

  for line in taxonomy_text.split("\n"):
    stripped = line.strip()
    if not stripped:
      continue

    if stripped.startswith("FACTOR:"):
      current_factor = stripped[len("FACTOR:") :].strip()
      current_path = [current_factor]
      continue

    # Determine level from L<N>: prefix.
    level_match = re.search(r"^L(\d+):\s+(.+)$", stripped)
    if level_match:
      level = int(level_match.group(1))
      node_text = level_match.group(2).strip()

      # Trim path to parent level and append this node.
      current_path = current_path[:level] + [node_text]
      # Track this node; it will only be kept if no deeper child follows.
      continue

    # Handle indented lines without L<N> prefix (treat as leaf labels).
    if stripped.startswith("-") or stripped.startswith("*"):
      node_text = stripped.lstrip("-* ").strip()
      if node_text and current_path:
        leaf_nodes.append(" > ".join(current_path + [node_text]))

  # If no explicit leaf markers, collect the deepest nodes per factor
  # by re-parsing with depth tracking.
  if not leaf_nodes:
    # Fallback: extract all L-prefixed nodes.
    for line in taxonomy_text.split("\n"):
      stripped = line.strip()
      level_match = re.search(r"^L(\d+):\s+(.+)$", stripped)
      if level_match:
        leaf_nodes.append(level_match.group(2).strip())

  # Deduplicate while preserving order.
  seen: set[str] = set()
  unique_nodes: list[str] = []
  for node in leaf_nodes:
    if node not in seen:
      seen.add(node)
      unique_nodes.append(node)

  if unique_nodes:
    return unique_nodes
  return [
      line.strip()
      for line in taxonomy_text.split("\n")
      if line.strip() and not line.strip().startswith("FACTOR:")
  ][:20]


def classify_leaf_nodes_paired(
    leaf_nodes: list[str],
) -> list[tuple[str, str]]:
  """Splits leaf nodes into matched (Group A, Group B) pairs.

  Preserves the 1:1 correspondence between A and B items by returning them
  as atomic tuples.

  Args:
    leaf_nodes: List of raw leaf node strings parsed from the taxonomy.

  Returns:
    List of (group_a_node, group_b_node) tuples.
  """
  pairs: list[tuple[str, str]] = []
  unpaired_a: list[str] = []
  unpaired_b: list[str] = []

  for node in leaf_nodes:
    stripped = node.strip()
    if not stripped:
      continue

    # Strip any leading L\d+: prefix (e.g. "L3: ")
    stripped = re.sub(r"^L\d+:\s*", "", stripped)

    # Case 1: The leaf node is a comparative pair (e.g.
    # "[A] Dish A vs [B] Dish B")
    if re.search(r"\s+vs\.?\s+", stripped, re.IGNORECASE):
      parts = re.split(
          r"\s+vs\.?\s+", stripped, maxsplit=1, flags=re.IGNORECASE
      )
      part_a = parts[0].strip()
      part_b = parts[1].strip()
      part_a = re.sub(r"^\[(?:A|B)\]\s*|^\((?:A|B)\)\s*", "", part_a)
      part_b = re.sub(r"^\[(?:A|B)\]\s*|^\((?:A|B)\)\s*", "", part_b)
      pairs.append((f"[A] {part_a}", f"[B] {part_b}"))
      continue

    # Case 2: Explicit [A] or (A) tag
    if (
        stripped.startswith("[A]")
        or stripped.startswith("(A)")
        or "[Group A]" in stripped
    ):
      unpaired_a.append(stripped)
    # Case 3: Explicit [B] or (B) tag
    elif (
        stripped.startswith("[B]")
        or stripped.startswith("(B)")
        or "[Group B]" in stripped
    ):
      unpaired_b.append(stripped)

  # Pair up any remaining unpaired items by index
  for a, b in zip(unpaired_a, unpaired_b):
    pairs.append((a, b))

  return pairs


def sample_paired_nodes(
    paired_nodes: list[tuple[str, str]],
    q_count: int,
) -> list[str]:
  """Samples matched (A, B) pairs and interleaves them: [A_1, B_1, A_2, B_2, ...].

  Args:
    paired_nodes: List of (group_a_node, group_b_node) tuples.
    q_count: Total number of queries/nodes to generate.

  Returns:
    Interleaved list of node strings: [A_1, B_1, A_2, B_2, ...].
  """
  n_pairs = max(1, q_count // 2)

  if len(paired_nodes) >= n_pairs:
    sampled = random.sample(paired_nodes, n_pairs)
  else:
    sampled = list(paired_nodes)
    while len(sampled) < n_pairs:
      sampled.append(random.choice(paired_nodes))

  interleaved: list[str] = []
  for a, b in sampled:
    interleaved.append(a)
    interleaved.append(b)

  return interleaved[:q_count]


def parse_critic_passing_indices(
    critic_response: str, total_queries: int
) -> list[int]:
  """Parses the passing query indices from a critic response.

  Args:
    critic_response: The critic's response text.
    total_queries: Total number of queries that were evaluated.

  Returns:
    List of 0-indexed passing query indices.
  """
  # Look for PASSING_INDICES: line.
  for line in critic_response.split("\n"):
    line = line.strip()
    if line.upper().startswith("PASSING_INDICES:"):
      indices_str = line.split(":", 1)[1].strip()
      try:
        # Parse comma-separated 1-indexed numbers.
        indices = [
            int(x.strip()) - 1
            for x in indices_str.split(",")
            if x.strip().isdigit()
        ]
        return [i for i in indices if 0 <= i < total_queries]
      except ValueError:
        pass

  # Fallback: look for QUERY_N: PASS lines.
  passing: list[int] = []
  for line in critic_response.split("\n"):
    match = re.search(r"^QUERY_(\d+)\s*:\s*(PASS)", line.strip(), re.IGNORECASE)
    if match:
      idx = int(match.group(1)) - 1
      if 0 <= idx < total_queries:
        passing.append(idx)

  # If critic parsing fails entirely, pass all queries through.
  return passing if passing else list(range(total_queries))


def parse_numbered_queries(response: str) -> list[str]:
  """Parses numbered queries from an LLM response.

  Args:
    response: The raw response text.

  Returns:
    A list of query strings.
  """
  queries: list[str] = []
  for line in response.split("\n"):
    line = line.strip()
    match = re.search(r"^\d+[\.\\)]\s+(.+)$", line)
    if match:
      queries.append(match.group(1).strip())
  if not queries:
    queries = [l.strip() for l in response.split("\n") if l.strip()]
  return queries


async def run_taxonomy_query_generation(
    config: config_lib.DiscoConfig,
    sender: agent_senders.AgentSender,
    prompts_mod: prompts_protocol.PromptsModule,
    remaining_queries: int,
    h_text: str,
    log_fn: Callable[..., None],
    target_num_queries: int | None = None,
    paired: bool | None = None,
) -> tuple[int, list[str]]:
  """Reasoning-driven taxonomy-based query generation (arXiv:2603.29791).

  Implements the taxonomy-driven framework (arXiv:2603.29791) for structured
  query generation. Instead of asking the agent to directly list queries, this
  function orchestrates a multi-step pipeline:

  1. Taxonomy Generation: The agent builds a hierarchical taxonomy of
     factors of variation for the hypothesis space.
  2. Taxonomic Sampling: Leaf nodes are sampled and combined into
     node-sets representing query requirements.
  3. Meta-Prompting: Node-sets are converted into concrete queries.
  4. Critic Filtering: A double-critic assesses query quality.
  5. Backfill / Replenishment: Replenishes rejected queries to meet target
     count.

  Args:
    config: The DiscoConfig containing taxonomy-driven parameters.
    sender: The agent sender.
    prompts_mod: The prompts module.
    remaining_queries: Number of queries remaining in the budget.
    h_text: The current hypothesis text.
    log_fn: Logging function.
    target_num_queries: Optional explicit target number of queries.
    paired: Whether to generate A/B pairs. Normally resolved by the caller from
      the config and the hypothesis's pre-registered plan, so taxonomy mode
      follows the same per-hypothesis choice as every other generation mode.
      Defaults to the run-level setting so an omitted argument cannot silently
      turn a paired run into an unpaired one.

  Returns:
    Tuple of (number of queries to run, list of query strings).
  """
  if target_num_queries is not None and target_num_queries > 0:
    q_count = target_num_queries
  else:
    q_count = min(config.chunk_size, remaining_queries)
  is_paired = config.is_paired if paired is None else paired

  log_fn("\nStep 3 (Taxonomy): Taxonomy-based query generation")
  log_fn(
      f"  Target: {q_count} queries | Taxonomy depth:"
      f" {config.taxonomy_depth} | Best-of-N:"
      f" {config.taxonomy_best_of_n} | Paired: {is_paired}"
  )

  # ── Phase 1: Taxonomy Generation ──────────────────────────────────
  log_fn("  [Phase 1] Generating taxonomy...")
  taxonomy_prompt = prompts_mod.get_taxonomy_generation_prompt(
      hypothesis=h_text,
      seed_instruction=config.seed_instruction,
      taxonomy_depth=config.taxonomy_depth,
      best_of_n=config.taxonomy_best_of_n,
      paired=is_paired,
  )
  taxonomy_response = await sender.send_step(taxonomy_prompt)
  log_fn(f"  [Taxonomy] Response preview: {taxonomy_response[:500]}")

  # ── Phase 2: Parse and Sample Taxonomy Nodes ──────────────────────
  leaf_nodes = parse_taxonomy_leaf_nodes(taxonomy_response)
  log_fn(f"  [Phase 2] Parsed {len(leaf_nodes)} leaf nodes")

  if not leaf_nodes:
    log_fn(
        "  [Warning] No leaf nodes parsed. Returning 0 — caller should"
        " fall back to standard query generation."
    )
    return 0, []

  paired_nodes: list[tuple[str, str]] = []
  if is_paired:
    paired_nodes = classify_leaf_nodes_paired(leaf_nodes)
    log_fn(f"  [Phase 2] Classified {len(paired_nodes)} matched pairs")
    if paired_nodes:
      sampled_node_sets = sample_paired_nodes(paired_nodes, q_count)
    else:
      log_fn(
          "  [Warning] Could not extract any matched (A, B) pairs from"
          " taxonomy leaf nodes. Returning 0 queries — caller should"
          " fall back to standard query generation."
      )
      return 0, []
  else:
    # Unpaired mode: sample node combinations. If we have multiple factors,
    # cross-sample; otherwise just use individual nodes.
    if len(leaf_nodes) <= q_count:
      # Use all leaf nodes and add additional diversity.
      sampled_node_sets = list(leaf_nodes)
      # Pad with random combinations if needed.
      while len(sampled_node_sets) < q_count and len(leaf_nodes) >= 2:
        combo = random.sample(leaf_nodes, min(2, len(leaf_nodes)))
        combo_str = " + ".join(combo)
        if combo_str not in sampled_node_sets:
          sampled_node_sets.append(combo_str)
    else:
      # Subsample to ensure diversity.
      indices = random.sample(range(len(leaf_nodes)), q_count)
      sampled_node_sets = [leaf_nodes[i] for i in indices]

  log_fn(f"  [Phase 2] Sampled {len(sampled_node_sets)} node combinations")

  # ── Phase 3: Meta-Prompting (Node-Sets → Queries) ─────────────────
  log_fn("  [Phase 3] Converting taxonomy nodes to queries...")
  meta_prompt = prompts_mod.get_taxonomy_meta_prompt(
      taxonomy_nodes=sampled_node_sets,
      hypothesis=h_text,
      num_queries=q_count,
      seed_instruction=config.seed_instruction,
      paired=is_paired,
  )
  meta_response = await sender.send_step(meta_prompt)
  queries = parse_numbered_queries(meta_response)
  log_fn(f"  [Phase 3] Generated {len(queries)} raw queries")

  # ── Phase 4: Critic Filtering ─────────────────────────────────────
  if config.taxonomy_use_critic and queries:
    log_fn("  [Phase 4] Running critic evaluation...")
    critic_prompt = prompts_mod.get_taxonomy_critic_prompt(
        queries=queries,
        taxonomy_nodes=sampled_node_sets,
        hypothesis=h_text,
        seed_instruction=config.seed_instruction,
    )
    critic_response = await sender.send_step(critic_prompt)
    passing_indices = set(
        parse_critic_passing_indices(critic_response, len(queries))
    )
    if is_paired:
      # Pair-safe filtering: retain pair k only if both 2*k and 2*k+1 pass
      # critic
      n_pairs = len(queries) // 2
      surviving_queries: list[str] = []
      rejected_pairs = 0
      for k in range(n_pairs):
        if (2 * k in passing_indices) and (2 * k + 1 in passing_indices):
          surviving_queries.extend([queries[2 * k], queries[2 * k + 1]])
        else:
          rejected_pairs += 1
      if rejected_pairs > 0:
        log_fn(
            f"  [Phase 4] Critic rejected {rejected_pairs}/{n_pairs} pairs"
            f" ({len(queries) - len(surviving_queries)} queries)"
        )
        queries = surviving_queries
      else:
        log_fn("  [Phase 4] All pairs passed critic check")
    else:
      n_rejected = len(queries) - len(passing_indices)
      if n_rejected > 0:
        log_fn(
            f"  [Phase 4] Critic rejected {n_rejected}/{len(queries)} queries"
        )
        queries = [queries[i] for i in sorted(passing_indices)]
      else:
        log_fn("  [Phase 4] All queries passed critic check")
  else:
    log_fn("  [Phase 4] Skipping critic filtering")

  # ── Phase 5: Backfill / Replenishment Loop ────────────────────────
  accumulated_queries = list(queries)
  target_count = q_count
  max_retries = 5
  retry_count = 0
  while len(accumulated_queries) < target_count and retry_count < max_retries:
    retry_count += 1
    deficit = target_count - len(accumulated_queries)
    if is_paired and deficit % 2 != 0:
      deficit += 1

    log_fn(
        f"  [Taxonomy Replenish] Deficit of {deficit} queries"
        f" ({len(accumulated_queries)}/{target_count} gathered)."
        f" Running replenish round {retry_count}/{max_retries}..."
    )

    if is_paired:
      batch_node_sets = sample_paired_nodes(paired_nodes, deficit)
    else:
      if len(leaf_nodes) <= deficit:
        batch_node_sets = list(leaf_nodes)
        while len(batch_node_sets) < deficit and len(leaf_nodes) >= 2:
          combo = random.sample(leaf_nodes, min(2, len(leaf_nodes)))
          combo_str = " + ".join(combo)
          if combo_str not in batch_node_sets:
            batch_node_sets.append(combo_str)
        while len(batch_node_sets) < deficit:
          batch_node_sets.append(random.choice(leaf_nodes))
      else:
        indices = random.sample(range(len(leaf_nodes)), deficit)
        batch_node_sets = [leaf_nodes[i] for i in indices]

    batch_meta_prompt = prompts_mod.get_taxonomy_meta_prompt(
        taxonomy_nodes=batch_node_sets,
        hypothesis=h_text,
        num_queries=deficit,
        seed_instruction=config.seed_instruction,
        paired=is_paired,
    )
    batch_meta_response = await sender.send_step(batch_meta_prompt)
    new_batch_queries = parse_numbered_queries(batch_meta_response)
    if not new_batch_queries:
      log_fn(
          "  [Taxonomy Replenish] No queries generated in replenish round,"
          " retrying..."
      )
      continue
    # Critic filtering on new batch
    if config.taxonomy_use_critic and new_batch_queries:
      critic_prompt = prompts_mod.get_taxonomy_critic_prompt(
          queries=new_batch_queries,
          taxonomy_nodes=batch_node_sets,
          hypothesis=h_text,
          seed_instruction=config.seed_instruction,
      )
      critic_response = await sender.send_step(critic_prompt)
      passing_indices = set(
          parse_critic_passing_indices(critic_response, len(new_batch_queries))
      )
      if is_paired:
        n_pairs = len(new_batch_queries) // 2
        surviving_new: list[str] = []
        rejected_pairs = 0
        for k in range(n_pairs):
          if (2 * k in passing_indices) and (2 * k + 1 in passing_indices):
            surviving_new.extend(
                [new_batch_queries[2 * k], new_batch_queries[2 * k + 1]]
            )
          else:
            rejected_pairs += 1
        log_fn(
            "  [Taxonomy Replenish] Critic accepted"
            f" {len(surviving_new) // 2}/{n_pairs} pairs"
            f" ({rejected_pairs} pairs rejected)"
        )
        new_batch_queries = surviving_new
      else:
        n_rejected = len(new_batch_queries) - len(passing_indices)
        log_fn(
            "  [Taxonomy Replenish] Critic accepted"
            f" {len(passing_indices)}/{len(new_batch_queries)} queries"
            f" ({n_rejected} rejected)"
        )
        new_batch_queries = [
            new_batch_queries[i] for i in sorted(passing_indices)
        ]

    accumulated_queries.extend(new_batch_queries)
    log_fn(
        "  [Taxonomy Replenish] Progress:"
        f" {len(accumulated_queries)}/{target_count} queries"
    )

  # ── Finalize: Return queries to caller for Step 4 injection ───────
  final_queries = (
      accumulated_queries[:target_count] if accumulated_queries else []
  )

  # Safety-net: strip any residual [A]/[B] tags from query text and ensure
  # even length in paired mode.
  if is_paired and final_queries:
    final_queries = [
        re.sub(r"^\[(?:A|B)\]\s*", "", q, flags=re.IGNORECASE).strip()
        for q in final_queries
    ]
    if len(final_queries) % 2 != 0:
      final_queries = final_queries[:-1]
    q_count = len(final_queries)
  else:
    q_count = len(final_queries)

  log_fn(f"\n[Taxonomy Q Count] Final queries to run: {q_count}")
  return q_count, final_queries
