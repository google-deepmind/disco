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

"""Hermetic fakes for end-to-end tests of the DISCO runners.

The runners talk to three external systems:

*   The auditor agent: a real `standalone_agent.StandaloneAgent` driving the
    Gemini API through `genai_client.get_genai_client()`.
    `FakeGenaiClient` replaces the API and answers with a scripted agent.
*   Single-shot LLM calls through `backbone_client.get_backbone_client()`:
    parser distillation, the hypothesis checker, the isolated query generator
    and the failure-summary compressor. `FakeBackbone` answers these.
*   The target model, reached through
    `standalone_agent._get_or_create_candidate()`. `FakeCandidate` returns
    canned answers whose word count is set by a tag in the query, and answers
    visual-symmetry probe images with a synthetic completion.

`hermetic_disco_env()` installs all three together with a zero-delay
`asyncio.sleep` and a seeded default numpy RNG. Everything else (the runner,
the real StandaloneAgent tool loop, the parsers, the statistics, and for the
symmetry task the procedural probe-image generator and the algorithmic scorer)
runs for real. The two StandaloneAgent symmetry tools are wrapped in
pass-through spies so the scripted agent can read the scorer's verdict even
when the runner, not the agent, invoked the tool.

Every fake fails fast on a request it does not recognize by raising
`UnexpectedFakeCallError`, which derives from `BaseException` so that the
`except Exception` retry loops in the code under test cannot swallow it. Every
fake also appends to a shared call log, which tests snapshot to pin the order
of external calls.
"""

import asyncio
from collections.abc import Callable, Iterator, Mapping, Sequence
import contextlib
import dataclasses
import io
import json
import random
import re
import types as pytypes
from typing import Any
from unittest import mock

from google.genai import types
import numpy as np
from PIL import Image

from disco import backbone_client
from disco import standalone_agent
from disco.models import content as content_api

# Word counts of the target model's answers, by the tag in the query. The
# i-th query of a batch (1-based) gets the i-th count, cycling. Against a
# one-sided t-test with null mean 20 and n=3 these give p ~ 1e-4 (LONG),
# p ~ 0.04 (MEDIUM) and p ~ 0.9996 (SHORT).
TAG_WORD_COUNTS: Mapping[str, tuple[int, ...]] = {
    "LONG": (50, 51, 52),
    "MEDIUM": (21, 23, 25),
    "SHORT": (3, 4, 5),
}

# Text tag whose queries the target model answers with an empty string, so
# every slot stays unusable after backfill and the scrape FAILS.
EMPTY_TAG = "EMPTY"

# Text tag answered like MEDIUM, except that every third query of a batch gets
# an empty answer, so a 3-query round keeps 2 of 3 responses after backfill:
# short of --min_response_fraction=0.9.
PARTIAL_TAG = "PARTIAL"

# Visual symmetry tags: how the target model answers a symmetry probe image.
# MIRROR completes it by reflecting the left half onto the right half, COPY
# returns the half-erased probe unchanged, and NOIMAGE answers with text only
# (an unusable slot, so the symmetry scrape FAILS).
SYMMETRY_TAGS: tuple[str, ...] = ("MIRROR", "COPY", "NOIMAGE")

# Null success rate the scripted agent tests symmetry pass rates against.
SYMMETRY_NULL_PROPORTION = 0.9

# Null mean the scripted agent pre-registers and tests against.
NULL_MEAN = 20.0

_TAG_PATTERN = "|".join((*TAG_WORD_COUNTS, EMPTY_TAG, PARTIAL_TAG))

# Agent prompt kinds, matched in order against the latest user turn. More
# specific markers come first: e.g. the execution prompt also mentions Step 5.
_AGENT_PROMPT_MARKERS: tuple[tuple[str, str], ...] = (
    ("execute", "Use EXACTLY these queries:"),
    ("symmetry_execute", "Now use your symmetry tools to"),
    # Text scrape-and-judge prompt (no procedural autorater). The runner only
    # sends it in symmetry rounds, after the symmetry step: see
    # `FakeGenaiClient.generate_content`.
    (
        "execute_judge",
        "against the target model and collect analysis judgements",
    ),
    ("stats", "Step 5: Statistical Significance Testing"),
    # Step 5 without a pre-registered plan (symmetry rounds).
    ("stats", "to calculate the p-value for your"),
    ("reflection", "Step 4b: Salient Pattern"),
    ("preregistration", "Step 2b: Statistical Test Pre-Registration"),
    ("taxonomy_critic", "You are a CRITIC evaluating the quality"),
    ("taxonomy_meta", "Now convert the following taxonomy node combinations"),
    ("taxonomy", "You are generating queries using a taxonomy-driven"),
    ("select", "Choose ONE of the following actions:"),
    ("context", "Please acknowledge that you have received this hypothesis"),
    ("symmetry_intro", "visual symmetry capability discovery system"),
    ("intro", "Don't start yet"),
)

# Leaf nodes of the scripted agent's taxonomy. Exactly one chunk's worth, so
# taxonomy sampling uses them all and draws no random numbers.
_TAXONOMY_LEAVES: tuple[str, ...] = (
    "Weeknight dinner",
    "Holiday dinner",
    "Picnic dinner",
)

_BACKBONE_PROMPT_MARKERS: tuple[tuple[str, str], ...] = (
    ("query_generator", "You are a query generator"),
    ("compress_failures", "You are compressing a block"),
    ("select_salient", "You are assisting an automated exploration"),
)

_LABELED_LINE = re.compile(r"^([A-Z][A-Z_\-]*):\s*(.*?)\s*$", re.MULTILINE)


class UnexpectedFakeCallError(BaseException):
  """A fake received a request that its script does not cover.

  Derives from BaseException rather than Exception on purpose: the code under
  test wraps LLM calls in `except Exception` retry loops, which would otherwise
  swallow this and retry (or spin forever) instead of failing the test.
  """


@dataclasses.dataclass(frozen=True)
class RoundScript:
  """What the scripted auditor does in one probe round.

  Attributes:
    h_id: Hypothesis ID the selector returns. Returning the ID of an active
      hypothesis re-probes it; any other ID proposes a new hypothesis.
    h_text: Hypothesis text the selector returns.
    tag: Key of TAG_WORD_COUNTS, EMPTY_TAG, PARTIAL_TAG, or one of
      SYMMETRY_TAGS. Text queries carry the tag, so it decides the target
      model's answer lengths and hence the round's p-value; for symmetry rounds
      it decides how the target model completes the probe images.
    empty_taxonomy: If True, the agent's taxonomy has no leaf nodes, so taxonomy
      query generation returns 0 queries and the runner falls back.
    stats_failure: If True, the agent's Step 5 answers carry no statistics, so
      stats parsing fails and the runner gets the sentinel p-value.
    symmetry_probe_shape: Shape family the agent probes in a symmetry round. If
      empty, the agent calls no symmetry tools and the runner runs the probes
      itself.
  """

  h_id: str
  h_text: str
  tag: str
  empty_taxonomy: bool = False
  stats_failure: bool = False
  symmetry_probe_shape: str = ""


CallLog = list[tuple[str, str]]


def _prompt_to_text(prompt: Any) -> str:
  if isinstance(prompt, str):
    return prompt
  return "".join(p for p in prompt if isinstance(p, str))


def _classify(text: str, markers: Sequence[tuple[str, str]]) -> str | None:
  for kind, marker in markers:
    if marker in text:
      return kind
  return None


def _content_text(content: types.Content) -> str:
  return "\n".join(p.text for p in content.parts or [] if p.text)


def _text_response(text: str) -> types.GenerateContentResponse:
  return _parts_response([types.Part(text=text)])


def _call_response(
    name: str, args: dict[str, Any]
) -> types.GenerateContentResponse:
  return _parts_response(
      [types.Part(function_call=types.FunctionCall(name=name, args=args))]
  )


def _parts_response(
    parts: list[types.Part],
) -> types.GenerateContentResponse:
  return types.GenerateContentResponse(
      candidates=[
          types.Candidate(
              content=types.Content(role="model", parts=parts),
              finish_reason=types.FinishReason.STOP,
          )
      ]
  )


def _numbered(lines: Sequence[str]) -> str:
  return "\n".join(f"{i}. {line}" for i, line in enumerate(lines, start=1))


def _required_int(pattern: str, text: str) -> int:
  match = re.search(pattern, text)
  if not match:
    raise UnexpectedFakeCallError(f"No match for {pattern!r} in: {text[:500]}")
  return int(match.group(1))


class FakeGenaiClient:
  """Stands in for `genai.Client`, playing a scripted auditor agent.

  Only `client.aio.models.generate_content` is implemented. The reply depends
  on the latest turn of `contents`: a user prompt is classified by marker
  text, and a function response continues the tool loop for that tool.
  Each select-or-propose prompt starts the next scripted round.

  Every StandaloneAgent shares this client. Turns of the isolated taxonomy
  query generator (a fresh agent whose history starts with the taxonomy
  prompt) are logged as "isolated_agent" instead of "agent".
  """

  def __init__(self, script: Sequence[RoundScript], call_log: CallLog):
    self._script = tuple(script)
    self._call_log = call_log
    self._rounds_started = 0
    self._current: RoundScript | None = None
    self._last_autorater: dict[str, Any] | None = None
    self._symmetry_score: dict[str, Any] | None = None
    self._query_batches = 0
    self.aio = pytypes.SimpleNamespace(
        models=pytypes.SimpleNamespace(generate_content=self.generate_content)
    )

  @property
  def rounds_started(self) -> int:
    return self._rounds_started

  @property
  def current_tag(self) -> str:
    """The tag of the current round, or "" before the first round."""
    return self._current.tag if self._current else ""

  def observe_symmetry_score(self, result: str) -> None:
    """Records the result of a `score_symmetry_completions` tool call."""
    scored = json.loads(result)
    if "error" in scored:
      self._call_log.append(("symmetry_scorer", "error"))
      self._symmetry_score = None
      return
    self._call_log.append((
        "symmetry_scorer",
        f"{scored['n_probes_passed']}/{scored['n_probes_valid']} passed",
    ))
    self._symmetry_score = scored

  async def generate_content(
      self, *, model: str, contents: list[types.Content], config: Any
  ) -> types.GenerateContentResponse:
    """Returns the scripted agent's next turn."""
    del model, config  # Unused.
    first_kind = _classify(_content_text(contents[0]), _AGENT_PROMPT_MARKERS)
    source = "isolated_agent" if first_kind == "taxonomy" else "agent"
    parts = list(contents[-1].parts or [])
    fn_responses = [p.function_response for p in parts if p.function_response]
    if fn_responses:
      return self._continue_tool_loop(fn_responses[-1])
    text = "\n".join(p.text for p in parts if p.text)
    kind = _classify(text, _AGENT_PROMPT_MARKERS)
    if kind is None:
      raise UnexpectedFakeCallError(f"Unrecognized agent prompt: {text[:500]}")
    self._call_log.append((source, kind))
    if kind in ("intro", "context", "symmetry_intro"):
      return _text_response("Acknowledged. Waiting for the next instruction.")
    if kind == "select":
      return self._start_round()
    current = self._require_round(kind)
    if kind == "preregistration":
      return _text_response(
          "TEST_TYPE: Continuous means test\n"
          "TARGET_METRIC: Word count of each answer\n"
          f"NULL_BASELINE: {NULL_MEAN:g}\n"
          "ALTERNATIVE: larger\n"
          "IS_PAIRED: False\n"
          "REASONING: Answers longer than the null mean support the hypothesis."
      )
    if kind == "execute":
      return _call_response(
          "generate_single_turn_responses",
          {"prompts": _injected_queries(text)},
      )
    if kind == "taxonomy":
      return _text_response(self._taxonomy(current))
    if kind == "taxonomy_meta":
      n = _required_int(r"You MUST generate exactly (\d+) queries", text)
      return _text_response(
          _numbered(self._make_queries(current, n, "taxonomy"))
      )
    if kind == "taxonomy_critic":
      n = len(re.findall(r"^\s*Query \d+:", text, re.MULTILINE))
      if not n:
        raise UnexpectedFakeCallError(f"No queries to criticize: {text[:500]}")
      verdicts = "\n".join(f"QUERY_{i}: PASS" for i in range(1, n + 1))
      indices = ", ".join(str(i) for i in range(1, n + 1))
      return _text_response(f"{verdicts}\nPASSING_INDICES: {indices}")
    if kind == "symmetry_execute":
      if not current.symmetry_probe_shape:
        return _text_response("I will describe my plan without calling tools.")
      return _call_response(
          "generate_and_send_symmetry_probes",
          {
              "shape_type": current.symmetry_probe_shape,
              "n_probes": _required_int(r"with EXACTLY (\d+) visual", text),
          },
      )
    if kind == "execute_judge":
      # Text rounds in the smoke cases use the procedural autorater (measure,
      # not judge), and symmetry rounds run only their own Step 4, so no
      # round should get the judge prompt.
      raise UnexpectedFakeCallError(f"Unexpected judge prompt: {text[:500]}")
    if kind == "reflection":
      return _text_response("- Answer length follows the prompt tag.")
    # kind == "stats"
    if current.stats_failure:
      return _text_response("INSUFFICIENT_INFORMATION: true")
    if current.tag in SYMMETRY_TAGS:
      if self._symmetry_score is None:
        raise UnexpectedFakeCallError("Symmetry stats step before scoring.")
      return _call_response(
          "run_proportion_z_test",
          {
              "n_successes": self._symmetry_score["n_probes_passed"],
              "n_obs": self._symmetry_score["n_probes_valid"],
              "null_proportion": SYMMETRY_NULL_PROPORTION,
              "alternative": "smaller",
          },
      )
    if self._last_autorater is None:
      raise UnexpectedFakeCallError("Stats step before any measurement.")
    return _call_response(
        "run_continuous_means_test",
        {
            "sample_mean": self._last_autorater["mean"],
            "sample_std": self._last_autorater["std"],
            "n_obs": self._last_autorater["total_responses"],
            "null_mean": NULL_MEAN,
            "alternative": "larger",
        },
    )

  def _start_round(self) -> types.GenerateContentResponse:
    if self._rounds_started >= len(self._script):
      raise UnexpectedFakeCallError(
          f"Script exhausted: select prompt #{self._rounds_started + 1} but"
          f" only {len(self._script)} rounds are scripted. A round probably"
          " spent no budget (e.g. a failed scrape)."
      )
    self._current = self._script[self._rounds_started]
    self._rounds_started += 1
    self._last_autorater = None
    self._symmetry_score = None
    return _text_response(
        f"HYPOTHESIS_ID: {self._current.h_id}\n"
        f"HYPOTHESIS: {self._current.h_text}"
    )

  def _require_round(self, kind: str) -> RoundScript:
    if self._current is None:
      raise UnexpectedFakeCallError(f"'{kind}' prompt before any round.")
    return self._current

  def _make_queries(self, current: RoundScript, n: int, how: str) -> list[str]:
    """Returns `n` agent-written queries carrying the round's tag."""
    if current.tag in SYMMETRY_TAGS:
      raise UnexpectedFakeCallError(f"Text queries in symmetry round: {how}")
    self._query_batches += 1
    return [
        f"Tagged {current.tag} request {self._query_batches}.{i}: {how}"
        " dinner request."
        for i in range(1, n + 1)
    ]

  def _taxonomy(self, current: RoundScript) -> str:
    if current.empty_taxonomy:
      # No L<N> or bullet lines: the parser finds no leaf nodes at all.
      return "FACTOR: Dinner occasion"
    leaves = "\n".join(f"    - {leaf}" for leaf in _TAXONOMY_LEAVES)
    return (
        f"FACTOR: Dinner occasion\n  L1: {current.tag} tagged"
        f" requests\n{leaves}"
    )

  def _continue_tool_loop(
      self, fn_response: types.FunctionResponse
  ) -> types.GenerateContentResponse:
    """Returns the agent's next turn after the tool `fn_response` returned."""
    name = fn_response.name or ""
    result = str((fn_response.response or {}).get("result", ""))
    self._call_log.append(("agent", f"after:{name}"))
    if name == "generate_single_turn_responses":
      if result.startswith("Error"):
        return _text_response("The target model returned no usable responses.")
      return _call_response(
          "run_procedural_autorater",
          {"autorater_name": "count_words", "params": {}},
      )
    if name == "run_procedural_autorater":
      measured = json.loads(result)
      if "error" in measured:
        raise UnexpectedFakeCallError(f"Autorater failed: {result}")
      self._last_autorater = measured
      return _text_response(
          "Collected and measured all responses with count_words."
      )
    if name == "run_continuous_means_test":
      match = re.search(r"P-value:\s*([0-9.eE+-]+)", result)
      if not match or self._last_autorater is None:
        raise UnexpectedFakeCallError(f"Unexpected t-test result: {result}")
      return _text_response(
          "TEST_TYPE: Continuous means test\n"
          f"GROUP_A_MEAN: {self._last_autorater['mean']}\n"
          f"GROUP_A_OBS: {self._last_autorater['total_responses']}\n"
          f"P-VALUE: {match.group(1)}\n"
          "REASONING: One-sample t-test of word counts against the null mean."
      )
    if name == "generate_and_send_symmetry_probes":
      return _call_response("score_symmetry_completions", {})
    if name == "score_symmetry_completions":
      return _text_response("Scored the symmetry completions algorithmically.")
    if name == "run_proportion_z_test":
      match = re.search(r"P-value:\s*([0-9.eE+-]+)", result)
      if not match or self._symmetry_score is None:
        raise UnexpectedFakeCallError(f"Unexpected binomial result: {result}")
      n_obs = self._symmetry_score["n_probes_valid"]
      rate = self._symmetry_score["n_probes_passed"] / n_obs
      # ALTERNATIVE matters: without it the runner's direction guard assumes
      # "larger" and flips the p-value of a low pass rate.
      return _text_response(
          "TEST_TYPE: Binomial test\n"
          f"GROUP_A_MEAN: {rate}\n"
          f"GROUP_A_OBS: {n_obs}\n"
          f"NULL_PROPORTION: {SYMMETRY_NULL_PROPORTION}\n"
          "ALTERNATIVE: smaller\n"
          f"P-VALUE: {match.group(1)}\n"
          "REASONING: Exact binomial test of the symmetry pass rate against"
          " the reference accuracy."
      )
    raise UnexpectedFakeCallError(f"Unexpected tool response: {name}: {result}")


def _injected_queries(prompt: str) -> list[str]:
  """Returns the numbered queries injected into a Step 4 prompt."""
  block = prompt.split("Use EXACTLY these queries:", 1)[1].lstrip("\n")
  block = block.split("\n\n", 1)[0]
  queries = re.findall(r"^\d+\.\s+(.+)$", block, re.MULTILINE)
  if not queries:
    raise UnexpectedFakeCallError(f"No injected queries in: {prompt[:500]}")
  return queries


def _extract_labeled_fields(text: str, schema: Any) -> dict[str, str]:
  """Distills `LABEL: value` lines whose label is a field alias of `schema`."""
  aliases = {field.alias or name for name, field in schema.model_fields.items()}
  return {
      label: value
      for label, value in _LABELED_LINE.findall(text)
      if label in aliases
  }


class FakeBackbone:
  """Stands in for `backbone_client.BackboneClient`.

  Structured calls (with `response_schema`) are parser distillations: the
  fake copies every `LABEL: value` line of the input whose label is a field
  alias of the schema, which is what a faithful distiller would do with the
  scripted agent's output. The hypothesis checker always passes. Free-text
  calls are classified by marker text.
  """

  def __init__(self, call_log: CallLog):
    self._call_log = call_log
    self._query_batches = 0

  async def generate(
      self,
      model: str,
      prompt: Any,
      *,
      temperature: float | None = None,
      system_instruction: str | None = None,
      response_schema: Any = None,
      max_output_tokens: int | None = None,
  ) -> str:
    """Returns the fake completion for `prompt`."""
    del model, temperature, system_instruction, max_output_tokens  # Unused.
    text = _prompt_to_text(prompt)
    if response_schema is not None:
      name = response_schema.__name__
      self._call_log.append(("backbone", name))
      if name == "HypothesisCheckResult":
        return json.dumps({
            "compliant": True,
            "violations": [],
            "reasoning": "Synthetic checker: no constraints violated.",
        })
      fields = _extract_labeled_fields(text, response_schema)
      if not fields:
        raise UnexpectedFakeCallError(
            f"Nothing to distill into {name} from: {text[:500]}"
        )
      return json.dumps(fields)
    kind = _classify(text, _BACKBONE_PROMPT_MARKERS)
    if kind is None:
      raise UnexpectedFakeCallError(
          f"Unrecognized backbone prompt: {text[:500]}"
      )
    self._call_log.append(("backbone", kind))
    if kind == "query_generator":
      return self._generate_queries(text)
    if kind == "compress_failures":
      return "Answer length follows the prompt tag."
    # kind == "select_salient"
    return "[1, 2, 3]"

  def _generate_queries(self, prompt: str) -> str:
    count = re.search(r"Generate exactly (\d+) queries", prompt)
    tag = re.search(rf"HYPOTHESIS \(H\):.*?\b({_TAG_PATTERN})\b", prompt)
    if not count or not tag:
      raise UnexpectedFakeCallError(f"Bad query-generator prompt: {prompt}")
    self._query_batches += 1
    return "\n".join(
        f"{i}. Tagged {tag.group(1)} request {self._query_batches}.{i}:"
        " suggest a dinner recipe."
        for i in range(1, int(count.group(1)) + 1)
    )


def _png_content(pixels: np.ndarray) -> content_api.Content:
  buf = io.BytesIO()
  Image.fromarray(pixels).save(buf, format="PNG")
  return content_api.Content(
      content_api.ContentChunk(buf.getvalue(), mimetype="image/png")
  )


class FakeCandidate:
  """Stands in for the target model's `content_api.Candidate`.

  Answers `Tagged <TAG> request <batch>.<i>: ...` queries with an answer of
  `TAG_WORD_COUNTS[TAG][i - 1]` synthetic words, or an empty answer for
  EMPTY_TAG. Answers a visual-symmetry probe (any query with an image) as the
  current round's symmetry tag says; see SYMMETRY_TAGS.
  """

  def __init__(self, call_log: CallLog, symmetry_mode: Callable[[], str]):
    self._call_log = call_log
    self._symmetry_mode = symmetry_mode

  async def respond(
      self, content: content_api.Content, jitter: float = 0.0
  ) -> content_api.Response:
    """Returns the canned answer for the query in `content`."""
    del jitter  # Unused.
    images = content.as_images(ignore_unsupported_types=True)
    if images:
      return content_api.Response(
          content=self._complete_symmetry_probe(images[0]),
          probe=content_api.Probe(content=content),
      )
    query = content.as_text(ignore_unsupported_types=True)
    match = re.search(rf"Tagged ({_TAG_PATTERN}) request \d+\.(\d+)", query)
    if not match:
      raise UnexpectedFakeCallError(f"Unrecognized target query: {query}")
    self._call_log.append(("candidate", query))
    if match.group(1) == EMPTY_TAG:
      answer = ""
    elif match.group(1) == PARTIAL_TAG and int(match.group(2)) % 3 == 0:
      answer = ""
    else:
      counts = TAG_WORD_COUNTS[
          "MEDIUM" if match.group(1) == PARTIAL_TAG else match.group(1)
      ]
      n_words = counts[(int(match.group(2)) - 1) % len(counts)]
      answer = " ".join(f"word{j}" for j in range(1, n_words + 1))
    return content_api.Response(
        content=content_api.Content(answer),
        probe=content_api.Probe(content=content),
    )

  def _complete_symmetry_probe(
      self, image: content_api.ContentChunk
  ) -> content_api.Content:
    """Returns a fake symmetry completion according to the round's tag."""
    mode = self._symmetry_mode()
    if mode not in SYMMETRY_TAGS:
      raise UnexpectedFakeCallError(f"Symmetry probe in a '{mode}' round.")
    self._call_log.append(("candidate", f"symmetry:{mode}"))
    if mode == "NOIMAGE":
      return content_api.Content("I cannot draw images.")
    pixels = np.array(image.pil_image.convert("RGB"))
    if mode == "MIRROR":
      half = pixels.shape[1] // 2
      pixels = pixels.copy()
      pixels[:, pixels.shape[1] - half :] = pixels[:, :half][:, ::-1]
    return _png_content(pixels)


@dataclasses.dataclass
class FakeWorld:
  """The installed fakes and their shared call log."""

  call_log: CallLog
  genai_client: FakeGenaiClient
  backbone: FakeBackbone
  candidate: FakeCandidate


@contextlib.contextmanager
def hermetic_disco_env(script: Sequence[RoundScript]) -> Iterator[FakeWorld]:
  """Replaces every external dependency of the DISCO runners with fakes.

  Installs FakeGenaiClient, FakeBackbone and FakeCandidate, makes
  `asyncio.sleep` yield without waiting (the senders sleep 5+ seconds per
  step), seeds `random` and makes unseeded `np.random.default_rng()` calls
  use seed 0. Also wraps the StandaloneAgent symmetry probe and scoring tools
  in pass-through spies that log their calls and hand the scores to the
  scripted agent.

  Args:
    script: One entry per probe round, in order.

  Yields:
    The installed fakes.
  """
  call_log: CallLog = []
  genai_client = FakeGenaiClient(script, call_log)
  world = FakeWorld(
      call_log=call_log,
      genai_client=genai_client,
      backbone=FakeBackbone(call_log),
      candidate=FakeCandidate(
          call_log, symmetry_mode=lambda: genai_client.current_tag
      ),
  )
  orig_sleep = asyncio.sleep
  orig_default_rng = np.random.default_rng
  agent_cls = standalone_agent.StandaloneAgent
  orig_send_probes = (
      agent_cls._tool_generate_and_send_symmetry_probes  # pylint: disable=protected-access
  )
  orig_score = (
      agent_cls._tool_score_symmetry_completions  # pylint: disable=protected-access
  )

  async def fast_sleep(delay: float, result: Any = None) -> Any:
    del delay  # Unused.
    await orig_sleep(0)
    return result

  def seeded_default_rng(seed: Any = None) -> np.random.Generator:
    return orig_default_rng(0 if seed is None else seed)

  async def spy_send_probes(
      agent: standalone_agent.StandaloneAgent, args: dict[str, Any]
  ) -> str:
    call_log.append(
        ("symmetry_probes", f"{args.get('shape_type')} x{args.get('n_probes')}")
    )
    return await orig_send_probes(agent, args)

  async def spy_score(
      agent: standalone_agent.StandaloneAgent, args: dict[str, Any]
  ) -> str:
    result = await orig_score(agent, args)
    genai_client.observe_symmetry_score(result)
    return result

  with contextlib.ExitStack() as stack:
    stack.enter_context(
        mock.patch.object(
            backbone_client, "get_backbone_client", return_value=world.backbone
        )
    )
    stack.enter_context(
        mock.patch.object(
            standalone_agent.genai_client,
            "get_genai_client",
            return_value=world.genai_client,
        )
    )
    stack.enter_context(
        mock.patch.object(
            standalone_agent,
            "_get_or_create_candidate",
            return_value=world.candidate,
        )
    )
    stack.enter_context(
        mock.patch.object(
            agent_cls,
            "_tool_generate_and_send_symmetry_probes",
            spy_send_probes,
        )
    )
    stack.enter_context(
        mock.patch.object(
            agent_cls, "_tool_score_symmetry_completions", spy_score
        )
    )
    stack.enter_context(mock.patch.object(asyncio, "sleep", fast_sleep))
    stack.enter_context(
        mock.patch.object(np.random, "default_rng", seeded_default_rng)
    )
    random.seed(0)
    yield world
