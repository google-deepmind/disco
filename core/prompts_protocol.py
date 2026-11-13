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

"""Protocol definitions for DISCO prompts modules."""

from typing import Any, Protocol, runtime_checkable

from disco.stats import plan as stats_plan


@runtime_checkable
class PromptsModule(Protocol):
  """Protocol capturing the interface of standard DISCO prompts."""

  def get_scrape_and_judge_execution_prompt(self, q_count: int) -> str:
    """Returns the prompt for scraping and judging target responses."""
    ...

  def get_procedural_autorater_addendum(
      self, autorater_name: str, allow_continuous_means: bool = True
  ) -> str:
    """Returns the prompt addendum describing the procedural autorater."""
    ...

  def get_procedural_scrape_and_measure_prompt(
      self, q_count: int, autorater_name: str
  ) -> str:
    """Returns the prompt for scraping and procedurally measuring responses."""
    ...

  def get_recent_failures_reflection_prompt(self) -> str:
    """Returns the prompt for reflecting on recent failure examples."""
    ...

  def get_test_preregistration_prompt(self, hypothesis_text: str) -> str:
    """Returns the prompt for pre-registering a statistical test plan."""
    ...

  def get_default_preregistration_retry_prompt(
      self, error_msg: str, original_prompt: str = ...
  ) -> str:
    """Returns the retry prompt when test plan pre-registration fails to parse."""
    ...

  def get_stats_significance_prompt(
      self, plan: stats_plan.PreRegisteredTestPlan | None = None
  ) -> str:
    """Returns the prompt for running the statistical significance test."""
    ...

  def get_default_retry_prompt(
      self, error_msg: str, original_prompt: str = ...
  ) -> str:
    """Returns the default retry prompt when response parsing fails."""
    ...

  # Taxonomy-driven query generation prompts

  def get_taxonomy_generation_prompt(
      self,
      hypothesis: str,
      seed_instruction: str,
      taxonomy_depth: int,
      best_of_n: int,
      paired: bool = ...,
  ) -> str:
    """Returns the prompt for generating a query taxonomy tree."""
    ...

  def get_taxonomy_meta_prompt(
      self,
      taxonomy_nodes: list[str],
      hypothesis: str,
      num_queries: int,
      seed_instruction: str = ...,
      paired: bool = ...,
  ) -> str:
    """Returns the meta-prompt for generating queries across taxonomy nodes."""
    ...

  def get_taxonomy_critic_prompt(
      self,
      queries: list[str],
      taxonomy_nodes: list[str],
      hypothesis: str,
      seed_instruction: str = ...,
  ) -> str:
    """Returns the critic prompt for filtering and refining taxonomy queries."""
    ...

  def get_isolated_query_generation_prompt(
      self,
      seed_instruction: str,
      hypothesis: str,
      num_queries: int,
      excluded_queries: list[str] | set[str] | None = None,
      procedural_autorater: str = "",
      paired: bool = False,
  ) -> str:
    """Returns the prompt for isolated query generation."""
    ...


@runtime_checkable
class FDRPromptsModule(PromptsModule, Protocol):
  """Extended protocol for FDR-controlled DISCO prompts."""

  def get_fresh_agent_intro_prompt(
      self,
      target_model: str,
      seed_instruction: str,
      threshold: float,
      chunk_size: int = 25,
      baseline_mode: bool = False,
  ) -> str:
    """Returns the introductory prompt for a fresh discovery agent."""
    ...

  def get_disco_fresh_agent_intro_prompt(
      self,
      target_model: str,
      seed_instruction: str,
      chunk_size: int = 25,
      e_min: float = 0.01,
      e_max: float = 100.0,
      baseline_mode: bool = False,
      show_all_failures: bool = False,
  ) -> str:
    """Returns the introductory prompt for a DISCO round agent."""
    ...

  def get_disco_select_or_propose_prompt(
      self,
      disco_history_summary: str,
      active_hypotheses: dict[str, str],
      target_model: str,
      seed_instruction: str,
      failure_summary: str = "",
      e_min: float = 0.01,
      e_max: float = 100.0,
      baseline_mode: bool = False,
  ) -> str:
    """Returns the Step 1 prompt for selecting or proposing a hypothesis."""
    ...

  def get_precision_round_context_prompt(
      self,
      h_text: str,
      prior_e_values: list[Any],
      failures: list[dict[str, str]] | None = None,
      failure_reflections: list[str] | None = None,
  ) -> str:
    """Returns the round context prompt for a selected hypothesis."""
    ...
