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

"""Snapshot cases for prompts built outside the prompt modules.

These prompts are assembled inside runner, agent, checker and judge code
rather than in `behavior_discovery_prompts` / `prompts` /
`verification_prompts`, so `prompt_snapshot_cases` does not cover them. Each
group renders to one golden mapping case name -> prompt text:

*   `runner_inline`: history and round-context builders (`core/history`), the
    failure-block compression and salient-example selection prompts, the
    injected-queries preamble of the text Step 4, and the agent's judge
    reasoning summary prompt.
*   `checker`: hypothesis compliance check and rejection prompts.
*   `judge`: the `AskQuestionJudge` prompt.

Prompts that are only sent to an LLM are captured with a fake backbone client
or a fake sender. All sample values are synthetic placeholders.
"""

import asyncio
from collections.abc import Callable
import types
from typing import Any, cast
from unittest import mock

from disco import backbone_client as backbone_lib
from disco import hypothesis_checker
from disco import standalone_agent
from disco import standalone_tools
from disco.core import checkpoint as checkpoint_lib
from disco.core import config as config_lib
from disco.core import history
from disco.core import records as records_lib
from disco.core import salient
from disco.core import task_executor

_LONG_TEXT = "<LONG_TEXT>" + " lorem ipsum" * 40


class _CapturingBackbone:
  """Backbone client that records prompts and returns a fixed reply."""

  def __init__(self, reply: str):
    self.reply = reply
    self.prompts: list[backbone_lib.Prompt] = []

  async def generate(
      self, *, model: str, prompt: backbone_lib.Prompt, **kwargs: Any
  ) -> str:
    del model, kwargs
    self.prompts.append(prompt)
    return self.reply


class _CapturingSender:
  """Agent sender that records the prompts of each step."""

  def __init__(self):
    self.prompts: list[str] = []

  async def send_step(self, prompt: str, **kwargs: Any) -> str:
    del kwargs
    self.prompts.append(prompt)
    return "<STEP_RESULT>"


def _capture_backbone_prompt(
    reply: str, fn: Callable[[], Any]
) -> backbone_lib.Prompt:
  """Runs the coroutine returned by `fn` and returns the one prompt it sent."""
  backbone = _CapturingBackbone(reply)
  with mock.patch.object(
      backbone_lib, "get_backbone_client", return_value=backbone
  ):
    asyncio.run(fn())
  if len(backbone.prompts) != 1:
    raise AssertionError(f"Expected one LLM call, got {len(backbone.prompts)}")
  return backbone.prompts[0]


def _failure(i: int, judgment: str = "EXHIBITS_BEHAVIOR") -> dict[str, str]:
  return {
      "judgment": judgment,
      "prompt": f"<PROBE_PROMPT_{i}>",
      "reasoning": f"<JUDGE_REASONING_{i}>",
      "response": f"<MODEL_RESPONSE_{i}>",
  }


def _hypothesis_groups() -> dict[str, records_lib.HypothesisRecord]:
  """Four hypotheses covering active, exited-high/low and timed-out records."""
  long_failure = {
      "judgment": "DOES_NOT_EXHIBIT",
      "prompt": _LONG_TEXT,
      "reasoning": _LONG_TEXT,
      "response": _LONG_TEXT,
  }
  return {
      "H1": records_lib.HypothesisRecord(
          text="<HYPOTHESIS_1>",
          e_values=[
              records_lib.EValueEntry("<SUB_HYPOTHESIS_1A>", 2.0),
              records_lib.EValueEntry("<SUB_HYPOTHESIS_1B>", 0.5),
          ],
          sub_hypotheses=["<SUB_HYPOTHESIS_1A>", "<SUB_HYPOTHESIS_1B>"],
          failures=[_failure(1), long_failure],
          failure_reflections=["<REFLECTION_1A>", "<REFLECTION_1B>"],
          executed_queries={"<QUERY_1B>", "<QUERY_1A>"},
      ),
      "H2": records_lib.HypothesisRecord(
          text="<HYPOTHESIS_2>",
          e_values=[records_lib.EValueEntry("<SUB_HYPOTHESIS_2A>", 60.0)],
          sub_hypotheses=["<SUB_HYPOTHESIS_2A>"],
          failures=[_failure(2), _failure(3, judgment="")],
          executed_queries={f"<QUERY_2_{i:02d}>" for i in range(12)},
      ),
      "H3": records_lib.HypothesisRecord(
          text="<HYPOTHESIS_3>",
          e_values=[records_lib.EValueEntry("<SUB_HYPOTHESIS_3A>", 0.3)],
      ),
      "H4": records_lib.HypothesisRecord(text="<HYPOTHESIS_4>"),
  }


def _run_state() -> checkpoint_lib.RunState:
  return checkpoint_lib.RunState(
      remaining_budget=40,
      hypothesis_groups=_hypothesis_groups(),
      running_e_values={"H1": 1.5, "H2": 60.0, "H3": 0.3},
      probe_counts={"H1": 3, "H2": 6, "H3": 2},
      terminal_statuses={"H1": "A", "H2": "H", "H3": "L", "H4": "B_2"},
      transition_scores={"H1": [0.25, -0.5], "H2": [1.0]},
      active_hypotheses={"H1"},
  )


def _config(**overrides: Any) -> config_lib.DiscoConfig:
  """Returns a config with every field the prompts read pinned."""
  config = config_lib.DiscoConfig(
      target_model="<TARGET_MODEL>",
      seed_instruction="<SEED_INSTRUCTION>",
      budget=40,
      alpha=0.1,
      kappa=0.1741,
      e_min=0.05,
      e_max=50.0,
      chunk_size=4,
      method="disco",
      show_all_failures=False,
      show_intra_hypothesis_failures=False,
      procedural_autorater="",
  )
  for name, value in overrides.items():
    if not hasattr(config, name):
      raise AttributeError(f"DiscoConfig has no field {name!r}")
    setattr(config, name, value)
  return config


def _failure_summary_cases() -> dict[str, str]:
  groups = _hypothesis_groups()
  return {
      "build_failure_summary": history.build_failure_summary(groups),
      "build_failure_summary[max_examples=2]": history.build_failure_summary(
          groups, max_examples=2
      ),
      "build_failure_summary[no_failures]": history.build_failure_summary(
          {"H4": groups["H4"]}
      ),
  }


def _disco_history_cases() -> dict[str, str]:
  """Renders the selector's history summary in every mode."""
  state = _run_state()

  def render(**kwargs: Any) -> str:
    return history.disco_history_summary(
        state.hypothesis_groups,
        state.active_hypotheses,
        state.terminal_statuses,
        state.probe_counts,
        state.transition_scores,
        e_min=0.05,
        e_max=50.0,
        running_e_values=state.running_e_values,
        **kwargs,
    )

  return {
      "disco_history_summary": render(),
      "disco_history_summary[show_failures=True]": render(show_failures=True),
      "disco_history_summary[baseline_mode=True]": render(baseline_mode=True),
      "disco_history_summary[baseline_type1_mode=True]": render(
          baseline_type1_mode=True
      ),
      "disco_history_summary[empty]": history.disco_history_summary(
          {}, set(), {}, {}, {}, e_min=0.05, e_max=50.0
      ),
  }


def _round_context_cases() -> dict[str, str]:
  """Renders the fresh round agent's context for each mode and hypothesis."""
  state = _run_state()
  groups = state.hypothesis_groups

  def render(h_id: str, cross_hyp_summary: str = "", **overrides: Any) -> str:
    return history.build_disco_round_context(
        _config(**overrides),
        state,
        groups[h_id].text,
        groups[h_id],
        h_id,
        cross_hyp_summary=cross_hyp_summary,
    )

  return {
      "build_disco_round_context": render("H1"),
      "build_disco_round_context[show_intra_hypothesis_failures=True]": render(
          "H1", show_intra_hypothesis_failures=True
      ),
      "build_disco_round_context[show_all_failures=True]": render(
          "H1", num_active_bins=0, show_all_failures=True
      ),
      "build_disco_round_context[baseline_mode=True]": render(
          "H1", baseline_mode=True
      ),
      "build_disco_round_context[baseline_type1_mode=True]": render(
          "H1", baseline_type1_mode=True
      ),
      "build_disco_round_context[cross_hyp_summary]": render(
          "H1", cross_hyp_summary="<CROSS_HYPOTHESIS_SUMMARY>"
      ),
      "build_disco_round_context[many_queries]": render(
          "H2", show_intra_hypothesis_failures=True
      ),
      "build_disco_round_context[unprobed]": render("H4"),
  }


def _compress_cases() -> dict[str, str]:
  raw = history.build_failure_summary(_hypothesis_groups())
  prompt = _capture_backbone_prompt(
      "<COMPRESSED_SUMMARY>",
      lambda: history.summarize_failure_block(
          raw, "<SEED_INSTRUCTION>", "<MODEL>", lambda *_: None
      ),
  )
  return {"summarize_failure_block": cast(str, prompt)}


def _salient_cases() -> dict[str, str]:
  """Renders the salient-example selection prompt with each status source."""
  judgments: list[dict[str, Any]] = [
      _failure(1),
      {"prompt": "<PROBE_PROMPT_2>", "response": "<MODEL_RESPONSE_2>"},
      {"prompt": "", "response": ""},
      {"prompt": _LONG_TEXT, "response": _LONG_TEXT, "value": 3.25},
      _failure(4, judgment="DOES_NOT_EXHIBIT"),
  ]
  state = _run_state()

  def render(**kwargs: Any) -> str:
    return cast(
        str,
        _capture_backbone_prompt(
            "[1, 3, 4]",
            lambda: salient.select_salient_examples(
                judgments,
                "<SEED_INSTRUCTION>",
                state.hypothesis_groups,
                "H1",
                "<HYPOTHESIS_1>",
                "<MODEL>",
                lambda *_: None,
                **kwargs,
            ),
        ),
    )

  return {
      "select_salient_examples": render(),
      "select_salient_examples[terminal_statuses]": render(
          terminal_statuses={"H1": "A", "H2": "H"}
      ),
  }


def _text_step4_cases() -> dict[str, str]:
  """Renders the text Step 4 prompt with injected queries."""
  prompts_mod = types.SimpleNamespace(
      get_procedural_scrape_and_measure_prompt=lambda q, a: (
          f"<PROCEDURAL_SCRAPE_PROMPT q_count={q} autorater={a}>"
      ),
      get_scrape_and_judge_execution_prompt=lambda q: (
          f"<SCRAPE_AND_JUDGE_PROMPT q_count={q}>"
      ),
  )

  def render(
      injected_queries: list[str] | None, procedural_autorater: str = ""
  ) -> str:
    sender = _CapturingSender()
    executor = task_executor.TextTaskExecutor(
        _config(procedural_autorater=procedural_autorater), lambda *_: None
    )
    asyncio.run(
        executor.execute_round(
            cast(Any, sender),
            cast(Any, prompts_mod),
            3,
            "<HYPOTHESIS>",
            _run_state(),
            injected_queries=injected_queries,
        )
    )
    if len(sender.prompts) != 1:
      raise AssertionError(
          f"Expected 1 step prompt, got {len(sender.prompts)}"
      )
    return sender.prompts[0]

  queries = ["<QUERY_1>", "<QUERY_2>", "<QUERY_3>"]
  return {
      # Symmetry rounds fall through into the text Step 4 with no queries.
      "text_step4": render(None),
      "text_step4[injected_queries]": render(queries),
      "text_step4[injected_queries,procedural]": render(
          queries, procedural_autorater="<AUTORATER>"
      ),
  }


def _reasoning_summary_cases() -> dict[str, str]:
  """Renders the StandaloneAgent reasoning-summary prompt."""
  agent = types.SimpleNamespace(
      _REASONING_SUMMARY_THRESHOLD=(
          standalone_agent.StandaloneAgent.REASONING_SUMMARY_THRESHOLD
      ),
      REASONING_SUMMARY_THRESHOLD=(
          standalone_agent.StandaloneAgent.REASONING_SUMMARY_THRESHOLD
      ),
      _summarizer_model="<MODEL>",
      _log=lambda *_: None,
  )
  summarize = standalone_agent.StandaloneAgent.summarize_reasoning
  prompt = _capture_backbone_prompt(
      "<SUMMARY>", lambda: summarize(cast(Any, agent), _LONG_TEXT)
  )
  return {"summarize_reasoning": cast(str, prompt)}


def render_runner_inline() -> dict[str, str]:
  """Returns case name -> prompt for prompts built in runner/agent code."""
  return (
      _failure_summary_cases()
      | _disco_history_cases()
      | _round_context_cases()
      | _compress_cases()
      | _salient_cases()
      | _text_step4_cases()
      | _reasoning_summary_cases()
  )


def render_checker() -> dict[str, str]:
  """Returns case name -> prompt for the hypothesis compliance checker."""
  check = hypothesis_checker.get_hypothesis_check_prompt
  base = {
      "seed_instruction": "<SEED_INSTRUCTION>",
      "hypothesis_id": "<HYPOTHESIS_ID>",
      "hypothesis_text": "<HYPOTHESIS>",
  }
  rules = ["<RULE_1>", "<RULE_2>"]
  cases = {
      "get_hypothesis_check_prompt": check(**base),
      "get_hypothesis_check_prompt[rules]": check(**base, rules=rules),
  }

  rule_evaluations = [
      hypothesis_checker.RuleEvaluation(
          rule_index=1, rule_text="<RULE_1>", passed=True
      ),
      hypothesis_checker.RuleEvaluation(
          rule_index=2,
          rule_text="<RULE_2>",
          passed=False,
          violation_detail="<VIOLATION_DETAIL>",
      ),
      hypothesis_checker.RuleEvaluation(
          rule_index=3, rule_text="<RULE_3>", passed=False
      ),
  ]
  results = {
      "violations": hypothesis_checker.HypothesisCheckResult(
          compliant=False,
          violations=["<VIOLATION_1>", "<VIOLATION_2>"],
          reasoning="<CHECKER_REASONING>",
      ),
      "rule_evaluations": hypothesis_checker.HypothesisCheckResult(
          compliant=False,
          reasoning="<CHECKER_REASONING>",
          rule_evaluations=rule_evaluations,
      ),
      "no_details": hypothesis_checker.HypothesisCheckResult(
          compliant=False, reasoning="<CHECKER_REASONING>"
      ),
  }
  for label, result in results.items():
    cases[f"format_rejection_prompt[{label}]"] = (
        hypothesis_checker.format_rejection_prompt(
            result, "<HYPOTHESIS_ID>", "<HYPOTHESIS>"
        )
    )
  return cases


def _render_parts(prompt: backbone_lib.Prompt) -> str:
  if isinstance(prompt, str):
    return prompt
  return "".join(
      part if isinstance(part, str) else f"<IMAGE {part.mimetype}>"
      for part in prompt
  )


def render_judge() -> dict[str, str]:
  """Returns case name -> prompt for the `AskQuestionJudge`."""
  build = standalone_tools.AskQuestionJudge.build_judge_prompt
  args = ("<USER_PROMPT>", "<MODEL_RESPONSE>", "<QUESTION>", ["YES", "NO"])
  image = backbone_lib.ImagePart(data=b"", mimetype="image/png")
  return {
      "build_judge_prompt": _render_parts(build(*args)),
      "build_judge_prompt[images]": _render_parts(
          build(*args, prompt_images=[image], response_images=[image, image])
      ),
  }


INLINE_GROUPS: dict[str, Callable[[], dict[str, str]]] = {
    "runner_inline": render_runner_inline,
    "checker": render_checker,
    "judge": render_judge,
}

# Golden name -> case function, in the same shape as
# prompt_snapshot_cases.ALL_CASES.
ALL_CASES: dict[str, Callable[[], Any]] = {
    f"prompts_{name}": fn for name, fn in INLINE_GROUPS.items()
}
