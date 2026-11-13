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

"""Configuration and protocol for hypothesis precision verification."""

import dataclasses
import typing
from typing import Literal

from disco import verification_prompts
from disco.core import prompts_protocol
from disco.models import registry as model_registry
from disco.parsing import parsers as parsers_lib
from disco.verification import scoring


@typing.runtime_checkable
class VerificationConfig(typing.Protocol):
  """Minimal interface required by precision_estimation for verification.

  Both DiscoConfig and VerifierConfig implement this protocol.
  """

  target_model: str
  agent_model: str
  agent_thinking_level: str | None = None
  agent_thinking_budget: int | None = None
  rating_model: str = ""
  summarizer_model: str = ""
  alpha: float
  kappa: float
  chunk_size: int
  output_dir: str
  experiment_name: str
  seed_instruction: str
  procedural_autorater: str
  task_suite: str = ""
  task_null_proportion: float = 0.0
  show_intra_hypothesis_failures: bool
  show_all_failures: bool
  allow_continuous_means_test: bool
  log_to_console: bool
  query_generation_mode: Literal["standard", "taxonomy"] = "standard"
  taxonomy_depth: int = 3
  taxonomy_use_critic: bool = True
  taxonomy_best_of_n: int = 3
  paired: bool = False
  p_value_mode: Literal["randomized", "discrete", "discrete_matched"] = (
      "randomized"
  )
  use_direction_guard: bool = True
  random_seed: int | None = None
  isolated_query_gen_temperature: float = 1.0
  single_batch_num_queries: int = 100
  single_batch_alpha: float = 0.05
  single_batch_null_prob: float | None = None
  max_concurrent_hypotheses: int = 1
  scrape_bandwidth: int = 4
  min_response_fraction: float = scoring.DEFAULT_MIN_RESPONSE_FRACTION

  @property
  def is_paired(self) -> bool:
    ...

  def get_prompts_module(self) -> prompts_protocol.FDRPromptsModule:
    ...

  def build_parsers(self) -> parsers_lib.Parsers:
    ...


@dataclasses.dataclass
class VerifierConfig:
  """Lightweight config for standalone hypothesis verification.

  Satisfies VerificationConfig without the complexity of DiscoConfig's
  many fields for sequential discovery, active selection, etc.
  """

  target_model: str
  # Auxiliary models. `backbone_model` fills in every role left at its default
  # (agent, parser and rating; summarizer falls back to rating); set a role
  # explicitly to override it (see `__post_init__`).
  agent_model: str = model_registry.DEFAULT_GEMINI_MODEL
  agent_thinking_level: str | None = None
  agent_thinking_budget: int | None = None
  backbone_model: str = ""
  rating_model: str = model_registry.DEFAULT_GEMINI_MODEL
  summarizer_model: str = ""
  alpha: float = 0.05
  kappa: float = 0.1741
  chunk_size: int = 25
  output_dir: str = "./outputs/precision_verifier/"
  experiment_name: str = "verification"
  seed_instruction: str = "Verify the provided hypotheses"
  procedural_autorater: str = ""
  # Benchmark suite this verification run belongs to (e.g. "visual_symmetry").
  # Lets the precision engine dispatch on modality explicitly instead of relying
  # solely on substring-matching `seed_instruction`.
  task_suite: str = ""
  # Reference / null success rate this task is defined against, sourced from the
  # registry task's `metadata["null_proportion"]`. 0.0 means unset, in which
  # case the precision engine falls back to its own per-modality default.
  # Overridden by an explicit `single_batch_null_prob`, or by a rate stated in
  # the hypothesis or seed instruction.
  task_null_proportion: float = 0.0
  exclude_analyze: bool = False
  show_intra_hypothesis_failures: bool = True
  show_all_failures: bool = False
  allow_continuous_means_test: bool = True
  # Query generation for the verification probes; the taxonomy settings are
  # read only when `query_generation_mode` is "taxonomy".
  query_generation_mode: Literal["standard", "taxonomy"] = "standard"
  taxonomy_depth: int = 3
  taxonomy_use_critic: bool = True
  taxonomy_best_of_n: int = 3
  # Run-level floor on paired (Group A, Group B) query generation.
  paired: bool = False
  p_value_mode: Literal["randomized", "discrete", "discrete_matched"] = (
      "randomized"
  )
  use_direction_guard: bool = True
  # Seed of the per-hypothesis p-value RNGs. None seeds them from OS entropy.
  random_seed: int | None = None
  isolated_query_gen_temperature: float = 1.0
  verification_mode: Literal["single_batch_p_value"] = "single_batch_p_value"
  # Base number of queries per hypothesis (doubled if paired). A positive
  # `total_verification_budget` is split across the hypotheses instead.
  single_batch_num_queries: int = 100
  total_verification_budget: int = 0
  # Significance level of the single-batch test; the standalone verifier sets
  # it together with `alpha` from `--alpha`.
  single_batch_alpha: float = 0.05
  # Null baseline of the single-batch test. None extracts it per hypothesis.
  single_batch_null_prob: float | None = None
  # Concurrency.
  max_concurrent_hypotheses: int = 1
  scrape_bandwidth: int = 4
  # `--min_response_fraction` (`verification/min_response_fraction_flag`).
  min_response_fraction: float = scoring.DEFAULT_MIN_RESPONSE_FRACTION
  parser_model: str = model_registry.DEFAULT_GEMINI_MODEL
  validate_stats_test: bool = False
  log_to_console: bool = True

  def __post_init__(self):
    default_model = model_registry.DEFAULT_GEMINI_MODEL
    if self.backbone_model:
      if not self.agent_model or self.agent_model == default_model:
        self.agent_model = self.backbone_model
      if not self.parser_model or self.parser_model == default_model:
        self.parser_model = self.backbone_model
      if not self.rating_model or self.rating_model == default_model:
        self.rating_model = self.backbone_model
      if (
          not self.summarizer_model
          or self.summarizer_model == default_model
          or self.summarizer_model == self.backbone_model
      ):
        self.summarizer_model = self.rating_model
    else:
      if not self.agent_model:
        self.agent_model = default_model
      if not self.parser_model:
        self.parser_model = default_model
      if not self.rating_model:
        self.rating_model = default_model
      if not self.summarizer_model:
        self.summarizer_model = self.rating_model
    # Normalize every model name (aliases, `models/` prefixes) so a value set
    # from a YAML config is treated the same as one resolved from a flag.
    for model_field in (
        "target_model",
        "backbone_model",
        "agent_model",
        "parser_model",
        "rating_model",
        "summarizer_model",
    ):
      setattr(
          self,
          model_field,
          model_registry.resolve_model_url(getattr(self, model_field)),
      )

  @property
  def is_paired(self) -> bool:
    """Whether this run forces paired A/B collection on every hypothesis.

    Note this is the *run-level* floor: with it false a hypothesis can still
    go paired on its own, via its pre-registered test plan.
    """
    return bool(self.paired)

  def get_prompts_module(self) -> prompts_protocol.FDRPromptsModule:
    """Returns the system-agnostic verification prompts module."""
    return verification_prompts

  def build_parsers(self) -> parsers_lib.Parsers:
    """Builds parsers bundle using the configured parser model."""
    return parsers_lib.Parsers.from_model(
        self.parser_model,
        validate_stats_test=self.validate_stats_test,
    )
