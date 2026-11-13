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

"""Builds the StandaloneAgent and AgentSender used by DISCO and the verifier."""

from typing import Any, Callable, Protocol

from disco import standalone_agent
from disco.agent import senders
from disco.core import prompts_protocol
from disco.core import task_detection
from disco.models import registry as model_registry


class AgentBuildConfig(Protocol):
  """The config fields every construction site reads directly."""

  target_model: str
  agent_model: str
  seed_instruction: str
  procedural_autorater: str
  allow_continuous_means_test: bool


def resolve_tool_model(cfg: AgentBuildConfig) -> str:
  """Returns the model the StandaloneAgent runs on."""
  return cfg.agent_model or model_registry.DEFAULT_GEMINI_MODEL


def build_agent(
    cfg: AgentBuildConfig,
    log_fn: Callable[..., None],
    *,
    output_dir: str,
    initial_scrape_round: int = 0,
) -> standalone_agent.StandaloneAgent:
  """Builds a StandaloneAgent from a DISCO or verifier config."""
  tool_agent_model = resolve_tool_model(cfg)
  return standalone_agent.StandaloneAgent(
      agent_model=tool_agent_model,
      target_model=cfg.target_model,
      log_fn=log_fn,
      output_dir=output_dir,
      initial_scrape_round=initial_scrape_round,
      procedural_autorater=cfg.procedural_autorater,
      allow_continuous_means=cfg.allow_continuous_means_test,
      # `exclude_analyze` is a VerifierConfig field only; DISCO runs exclude
      # the analyze tool exactly when the task is visual symmetry.
      exclude_analyze=(
          getattr(cfg, "exclude_analyze", False)
          or task_detection.is_symmetry_task(cfg)
      ),
      rating_model=getattr(cfg, "rating_model", ""),
      summarizer_model=getattr(cfg, "summarizer_model", ""),
      scrape_bandwidth=getattr(cfg, "scrape_bandwidth", None),
      agent_thinking_level=getattr(cfg, "agent_thinking_level", None),
      agent_thinking_budget=getattr(cfg, "agent_thinking_budget", None),
      min_response_fraction=getattr(cfg, "min_response_fraction", 0.0),
  )


def build_sender(
    agent: Any,
    prompts_mod: prompts_protocol.PromptsModule,
) -> senders.BaseSender:
  """Builds the sender that drives `agent`."""
  return senders.StandaloneSender(agent, prompts_mod)


def build_agent_and_sender(
    cfg: AgentBuildConfig,
    log_fn: Callable[..., None],
    prompts_mod: prompts_protocol.PromptsModule,
    *,
    output_dir: str,
    initial_scrape_round: int = 0,
) -> tuple[standalone_agent.StandaloneAgent, senders.BaseSender]:
  """Builds an agent and its sender; see `build_agent` and `build_sender`."""
  agent = build_agent(
      cfg,
      log_fn,
      output_dir=output_dir,
      initial_scrape_round=initial_scrape_round,
  )
  sender = build_sender(agent, prompts_mod)
  return agent, sender


def build_isolated_sender(
    cfg: AgentBuildConfig,
    log_fn: Callable[..., None],
    prompts_mod: prompts_protocol.PromptsModule,
    output_dir: str,
    initial_scrape_round: int,
) -> senders.BaseSender:
  """Builds a fresh sender for isolated query generation."""
  agent = build_agent(
      cfg,
      log_fn,
      output_dir=output_dir,
      initial_scrape_round=initial_scrape_round,
  )
  return senders.StandaloneSender(agent, prompts_mod)
