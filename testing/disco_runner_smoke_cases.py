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

"""Hermetic end-to-end smoke cases for `runners.DiscoRunner`.

Each case builds a `DiscoRunner` the way `eval_disco` does
(`DiscoRunner(config, config.build_parsers())`), runs `run_loop()` against
the fakes in `fakes.py` with a short scripted run, and returns a normalized
snapshot of:

*   the `DiscoResult` returned by the runner,
*   every file written under the experiment directory (except `disco.log`,
    which is free-form and has timestamps), parsed where it is JSON; PNG
    images are reduced to their size and mode,
*   the log of external calls made to the fakes.

The `eval_disco` CLI wrapper post-processing (results summary,
`evaluation_results.json`) is not covered here.

Every case runs in one of three runner modes:

*   `disco_post_hoc`: DISCO FDR with post-hoc alpha calibration (the default).
*   `baseline`: unrestricted exploration plus offline e-BH with a fixed K_bar.
*   `baseline_type1`: freeze at E >= 1/alpha, no e-BH.

The original cases are named after their mode, use the isolated query
generator in standard mode on a text task, and cover the terminal statuses:

*   `disco_post_hoc`: LONG exits high (H), SHORT exits low (L) and MEDIUM stays
    active (A) until the budget runs out.
*   `baseline`: the script re-probes the LONG hypothesis in round 3.
*   `baseline_type1`: LONG freezes; MEDIUM is probed twice and stays below the
    threshold.

Hypotheses are operationalized directly (there is no sub-hypothesis step) and
Step 3 always runs on the isolated generator, so there are no cases for
in-context query generation or sub-hypotheses.

The other cases pin the remaining branches of `RoundExecutor.run` and the
`DiscoRunner.run_loop` skip paths:

*   `taxonomy_isolated`: taxonomy query generation through the isolated sender;
    in round 2 the taxonomy has no leaves, so the runner falls back to the
    isolated backbone query generator.
*   `scrape_failure`: round 2's target answers are all empty, so the round is
    skipped without charging budget and a fourth round spends it.
*   `stats_failure`: round 2's stats cannot be parsed, so the runner gets the
    sentinel p-value: the round's queries are charged but no e-value is
    recorded.
*   `symmetry`: the visual-symmetry path, with synthetic images. Round 1 is
    run by the agent's tools; round 2's target returns no images (a symmetry
    scrape failure, skipped); in round 3 the agent calls no tools and the
    runner runs the probes itself; in round 4 the agent probes the wrong
    family for a hard-family hypothesis and the runner re-probes with the
    right one. Only round 3's target copies instead of mirroring, so only
    that hypothesis is rejected.

Budget 9 with chunk size 3 gives three charged rounds per case.
"""

import asyncio
from collections.abc import Callable, Mapping, Sequence
import dataclasses
import json
import os
import tempfile
from typing import Any

from absl.testing import flagsaver
from PIL import Image

from disco import runners
from disco.core import config as config_lib
from disco.core import records as records_lib
from disco.testing import fakes
from disco.testing import golden_utils

_EXPERIMENT_NAME = "smoke"
_OUT_PLACEHOLDER = "<OUT>"
_TIMEOUT_SECONDS = 120.0

_LONG = fakes.RoundScript(
    h_id="long_tags",
    h_text="Requests tagged LONG get answers longer than 20 words.",
    tag="LONG",
)
_SHORT = fakes.RoundScript(
    h_id="short_tags",
    h_text="Requests tagged SHORT get answers longer than 20 words.",
    tag="SHORT",
)
_MEDIUM = fakes.RoundScript(
    h_id="medium_tags",
    h_text="Requests tagged MEDIUM get answers longer than 20 words.",
    tag="MEDIUM",
)
_EMPTY = fakes.RoundScript(
    h_id="empty_tags",
    h_text="Requests tagged EMPTY get answers longer than 20 words.",
    tag=fakes.EMPTY_TAG,
)
_PARTIAL = fakes.RoundScript(
    h_id="partial_tags",
    h_text="Requests tagged PARTIAL get answers longer than 20 words.",
    tag=fakes.PARTIAL_TAG,
)

_SYMMETRY_SEED = (
    "Find visual symmetry completion patterns where the target model fails."
)


def _symmetry_round(
    shape: str, tag: str, probe_shape: str | None = None
) -> fakes.RoundScript:
  """Returns a symmetry round about `shape`; the agent probes `probe_shape`."""
  return fakes.RoundScript(
      h_id=f"{shape}_fails",
      h_text=(
          f"The target model fails to complete {shape.replace('_', ' ')}"
          " probes symmetrically."
      ),
      tag=tag,
      symmetry_probe_shape=shape if probe_shape is None else probe_shape,
  )


@dataclasses.dataclass(frozen=True)
class SmokeCase:
  """A scripted run and the outcomes it must produce.

  Attributes:
    mode: Runner mode: "disco_post_hoc", "baseline" or "baseline_type1".
    script: One entry per probe round, in order.
    terminal_statuses: Terminal status the runner reports, by hypothesis.
    probe_counts: Probes (e-value updates) per hypothesis.
    rejected: Hypotheses rejected at finalization.
    config_overrides: DiscoConfig fields that differ from the shared config.
    queries: Budget charged per hypothesis. Unchecked if empty.
    total_queries: Total budget charged. Defaults to the configured budget.
    e_value_counts: E-values recorded per hypothesis. Unchecked if empty.
    call_counts: Exact number of times each of these call-log entries appears.
      They prove that the case took the branch it is about.
  """

  mode: str
  script: tuple[fakes.RoundScript, ...]
  terminal_statuses: Mapping[str, str]
  probe_counts: Mapping[str, int]
  rejected: frozenset[str]
  config_overrides: Mapping[str, Any] = dataclasses.field(default_factory=dict)
  queries: Mapping[str, int] = dataclasses.field(default_factory=dict)
  total_queries: int | None = None
  e_value_counts: Mapping[str, int] = dataclasses.field(default_factory=dict)
  call_counts: Mapping[tuple[str, str], int] = dataclasses.field(
      default_factory=dict
  )


_ISOLATED_GEN = ("backbone", "query_generator")

# Smoke-case mode -> `DiscoConfig.method`.
_METHODS: Mapping[str, config_lib.Method] = {
    "disco_post_hoc": "disco",
    "baseline": "ebh_baseline",
    "baseline_type1": "type1_baseline",
}

CASES: Mapping[str, SmokeCase] = {
    # Baseline mode has no exits (baseline_enforce_exit_low defaults to False
    # and there is no e_max freeze), so everything stays active (A). Type-I
    # mode freezes LONG at E >= 1/alpha, which it reports as H.
    "disco_post_hoc": SmokeCase(
        mode="disco_post_hoc",
        script=(_LONG, _SHORT, _MEDIUM),
        terminal_statuses={
            "long_tags": "H",
            "short_tags": "L",
            "medium_tags": "A",
        },
        probe_counts={"long_tags": 1, "short_tags": 1, "medium_tags": 1},
        rejected=frozenset({"long_tags"}),
        call_counts={
            # Hypotheses are operationalized directly, and queries always come
            # from the isolated generator.
            _ISOLATED_GEN: 3,
            ("agent", "execute"): 3,
        },
    ),
    "baseline": SmokeCase(
        mode="baseline",
        script=(_LONG, _SHORT, _LONG),
        terminal_statuses={"long_tags": "A", "short_tags": "A"},
        probe_counts={"long_tags": 2, "short_tags": 1},
        rejected=frozenset({"long_tags"}),
    ),
    "baseline_type1": SmokeCase(
        mode="baseline_type1",
        script=(_LONG, _MEDIUM, _MEDIUM),
        terminal_statuses={"long_tags": "H", "medium_tags": "A"},
        probe_counts={"long_tags": 1, "medium_tags": 2},
        rejected=frozenset({"long_tags"}),
        queries={"long_tags": 3, "medium_tags": 6},
        e_value_counts={"long_tags": 1, "medium_tags": 2},
        call_counts={
            _ISOLATED_GEN: 3,
        },
    ),
    "taxonomy_isolated": SmokeCase(
        mode="disco_post_hoc",
        script=(
            _LONG,
            dataclasses.replace(_SHORT, empty_taxonomy=True),
            _MEDIUM,
        ),
        config_overrides={"query_generation_mode": "taxonomy"},
        terminal_statuses={
            "long_tags": "H",
            "short_tags": "L",
            "medium_tags": "A",
        },
        probe_counts={"long_tags": 1, "short_tags": 1, "medium_tags": 1},
        rejected=frozenset({"long_tags"}),
        queries={"long_tags": 3, "short_tags": 3, "medium_tags": 3},
        e_value_counts={"long_tags": 1, "short_tags": 1, "medium_tags": 1},
        call_counts={
            ("isolated_agent", "taxonomy"): 3,
            ("isolated_agent", "taxonomy_meta"): 2,
            ("isolated_agent", "taxonomy_critic"): 2,
            ("agent", "taxonomy"): 0,
            # Round 2's fallback.
            _ISOLATED_GEN: 1,
            ("agent", "execute"): 3,
        },
    ),
    "scrape_failure": SmokeCase(
        mode="disco_post_hoc",
        script=(_LONG, _EMPTY, _SHORT, _MEDIUM),
        # The skipped hypothesis is never probed, so it stays active.
        terminal_statuses={
            "long_tags": "H",
            "empty_tags": "A",
            "short_tags": "L",
            "medium_tags": "A",
        },
        probe_counts={
            "long_tags": 1,
            "empty_tags": 0,
            "short_tags": 1,
            "medium_tags": 1,
        },
        rejected=frozenset({"long_tags"}),
        queries={
            "long_tags": 3,
            "empty_tags": 0,
            "short_tags": 3,
            "medium_tags": 3,
        },
        e_value_counts={
            "long_tags": 1,
            "empty_tags": 0,
            "short_tags": 1,
            "medium_tags": 1,
        },
        call_counts={
            ("agent", "after:generate_single_turn_responses"): 4,
            # The skipped round never reaches reflection or stats.
            ("agent", "reflection"): 3,
            ("agent", "stats"): 3,
        },
    ),
    "short_round": SmokeCase(
        mode="disco_post_hoc",
        script=(_LONG, _PARTIAL, _SHORT, _MEDIUM),
        # The short round (2 of 3 responses) is skipped like a failed scrape:
        # nothing is charged or recorded, and the hypothesis stays active.
        terminal_statuses={
            "long_tags": "H",
            "partial_tags": "A",
            "short_tags": "L",
            "medium_tags": "A",
        },
        probe_counts={
            "long_tags": 1,
            "partial_tags": 0,
            "short_tags": 1,
            "medium_tags": 1,
        },
        rejected=frozenset({"long_tags"}),
        queries={
            "long_tags": 3,
            "partial_tags": 0,
            "short_tags": 3,
            "medium_tags": 3,
        },
        e_value_counts={
            "long_tags": 1,
            "partial_tags": 0,
            "short_tags": 1,
            "medium_tags": 1,
        },
        call_counts={
            (
                "agent",
                "after:generate_single_turn_responses",
            ): 4,
            # The short round never reaches reflection or stats.
            ("agent", "reflection"): 3,
            ("agent", "stats"): 3,
        },
    ),
    "stats_failure": SmokeCase(
        mode="disco_post_hoc",
        script=(
            _LONG,
            dataclasses.replace(_MEDIUM, stats_failure=True),
            _SHORT,
        ),
        # The sentinel round charges budget but records no e-value, so MEDIUM
        # stays active with no probes.
        terminal_statuses={
            "long_tags": "H",
            "medium_tags": "A",
            "short_tags": "L",
        },
        probe_counts={"long_tags": 1, "medium_tags": 0, "short_tags": 1},
        rejected=frozenset({"long_tags"}),
        queries={"long_tags": 3, "medium_tags": 3, "short_tags": 3},
        e_value_counts={"long_tags": 1, "medium_tags": 0, "short_tags": 1},
        call_counts={
            # Round 2 asks twice (one retry), then falls back to the sentinel.
            ("agent", "stats"): 4,
            ("backbone", "StatsExtraction"): 4,
            # ... and never runs a statistical test.
            ("agent", "after:run_continuous_means_test"): 2,
        },
    ),
    "symmetry": SmokeCase(
        mode="disco_post_hoc",
        script=(
            _symmetry_round("circle", "MIRROR"),
            _symmetry_round("square", "NOIMAGE"),
            _symmetry_round("dots", "COPY", probe_shape=""),
            _symmetry_round("wave_lines", "MIRROR", probe_shape="circle"),
        ),
        config_overrides={
            "seed_instruction": _SYMMETRY_SEED,
            "procedural_autorater": "",
            # One charge per round: 3 probes x 3 charged rounds.
            "budget": 9,
        },
        # The model mirrors circles and wave lines (3/3 passed: not failures)
        # but copies dots (0/3 passed against a 0.9 reference: rejected).
        terminal_statuses={
            "circle_fails": "L",
            "square_fails": "A",
            "dots_fails": "H",
            "wave_lines_fails": "L",
        },
        probe_counts={
            "circle_fails": 1,
            "square_fails": 0,
            "dots_fails": 1,
            "wave_lines_fails": 1,
        },
        rejected=frozenset({"dots_fails"}),
        queries={
            "circle_fails": 3,
            "square_fails": 0,
            "dots_fails": 3,
            "wave_lines_fails": 3,
        },
        total_queries=9,
        e_value_counts={
            "circle_fails": 1,
            "square_fails": 0,
            "dots_fails": 1,
            "wave_lines_fails": 1,
        },
        call_counts={
            ("agent", "symmetry_intro"): 4,
            ("agent", "symmetry_execute"): 4,
            ("agent", "preregistration"): 0,
            # Round 1: the agent probes; round 2: the agent probes and, with
            # no images back, the runner re-probes (Fallback) and the round
            # is skipped; round 3: the agent calls no tools, so only the
            # runner probes (Fallback); round 4: the agent probes circles and
            # the runner re-probes the hard wave-lines family (Correction).
            ("symmetry_probes", "circle x3"): 2,
            ("symmetry_probes", "square x3"): 2,
            ("symmetry_probes", "dots x3"): 1,
            ("symmetry_probes", "wave_lines x3"): 1,
            ("agent", "after:generate_and_send_symmetry_probes"): 3,
            ("symmetry_scorer", "3/3 passed"): 3,
            ("symmetry_scorer", "0/3 passed"): 1,
            ("symmetry_scorer", "0/0 passed"): 2,
            # Each image fails 3 retries plus the first try: 2 x 3 x 4.
            ("candidate", "symmetry:NOIMAGE"): 24,
            ("agent", "execute_judge"): 0,
            ("agent", "stats"): 3,
            ("agent", "after:run_proportion_z_test"): 3,
        },
    ),
}

CASE_NAMES: tuple[str, ...] = tuple(CASES)

# Keys holding wall-clock values.
_DROPPED_KEYS = frozenset({"ts"})

# Files whose content is free-form or has timestamps. Only their presence is
# snapshotted.
_CONTENT_EXCLUDED_FILES = frozenset({"disco.log"})


def golden_name(case: str) -> str:
  return f"disco_runner_smoke/{case}"


def make_config(case: str, out_dir: str) -> config_lib.DiscoConfig:
  """Returns the DiscoConfig for `case`, writing under `out_dir`.

  Every field is set explicitly (checked by `disco_runner_smoke_test`), so
  changing a `DiscoConfig` default does not change the smoke runs.

  Args:
    case: Name of a case in `CASES`.
    out_dir: Output directory of the run.

  Raises:
    ValueError: If `case` is unknown.
  """
  if case not in CASES:
    raise ValueError(f"Unknown case: {case}")
  smoke_case = CASES[case]
  config = config_lib.DiscoConfig(
      # Target and task.
      target_model="fake-target-model",
      seed_instruction=(
          "Find prompt tags that make the target model give long answers."
      ),
      task_suite="",
      task_null_proportion=0.0,
      # Statistics.
      budget=9,
      alpha=0.05,
      kappa=0.1741,
      e_min=0.99,
      e_max=50.0,
      enforce_exit_low=True,
      use_boosting=True,
      chunk_size=3,
      alpha_sweep_min=0.01,
      alpha_sweep_max=0.5,
      alpha_sweep_steps=20,
      alpha_calibration_mode="post_hoc",
      n_max=10,
      num_active_bins=1,
      n_sim=2000,
      use_direction_guard=True,
      p_value_mode="randomized",
      random_seed=None,
      # Models and agent.
      agent_model="fake-agent-model",
      agent_thinking_level=None,
      agent_thinking_budget=None,
      backbone_model="",
      parser_model="fake-parser-model",
      rating_model="fake-rater-model",
      summarizer_model="fake-rater-model",
      scrape_bandwidth=1,
      min_response_fraction=0.9,
      max_consecutive_skips=20,
      max_concurrent_hypotheses=5,
      # Output.
      output_dir=out_dir,
      experiment_name=_EXPERIMENT_NAME,
      log_to_console=False,
      # What the agent sees.
      show_intra_hypothesis_failures=True,
      show_all_failures=False,
      # Measurement.
      procedural_autorater="count_words",
      allow_continuous_means_test=True,
      validate_stats_test=False,
      # Query generation.
      query_generation_mode="standard",
      taxonomy_depth=3,
      taxonomy_use_critic=True,
      taxonomy_best_of_n=3,
      paired=False,
      isolated_query_gen_temperature=1.0,
      # Method and baselines.
      method=_METHODS[smoke_case.mode],
      baseline_enforce_exit_low=False,
      baseline_exit_high=False,
      # Hypothesis checker.
      enable_hypothesis_checker=True,
      hypothesis_checker_max_retries=2,
      hypothesis_checker_rules=[],
      # Verification.
      run_precision_check=True,
      single_batch_num_queries=100,
      single_batch_alpha=0.05,
      single_batch_null_prob=None,
  )
  if smoke_case.config_overrides:
    config = dataclasses.replace(config, **smoke_case.config_overrides)
  return config


def _normalize(value: Any, out_dir: str) -> Any:
  """Removes run-specific parts of a JSON-like value."""
  if isinstance(value, str):
    return value.replace(out_dir, _OUT_PLACEHOLDER)
  if isinstance(value, Mapping):
    return {
        key: _normalize(item, out_dir)
        for key, item in value.items()
        if key not in _DROPPED_KEYS
    }
  if isinstance(value, list):
    return [_normalize(item, out_dir) for item in value]
  return value


def _read_output(path: str) -> Any:
  """Returns the parsed content of an output file."""
  if path.endswith(".png"):
    # Pixel values depend on the image codecs, so only the shape is pinned.
    with Image.open(path) as image:
      return {"png_size": list(image.size), "png_mode": image.mode}
  with open(path) as f:
    text = f.read()
  if path.endswith(".json"):
    return json.loads(text)
  if path.endswith(".jsonl"):
    return [json.loads(line) for line in text.splitlines() if line.strip()]
  return text


def _snapshot_outputs(experiment_dir: str) -> tuple[list[str], dict[str, Any]]:
  """Returns the relative paths and parsed contents of the run's files."""
  files = []
  outputs = {}
  for root, _, names in os.walk(experiment_dir):
    for name in names:
      path = os.path.join(root, name)
      rel = os.path.relpath(path, experiment_dir)
      files.append(rel)
      if name not in _CONTENT_EXCLUDED_FILES:
        outputs[rel] = _read_output(path)
  return sorted(files), outputs


async def _run_runner(
    config: config_lib.DiscoConfig,
) -> records_lib.DiscoResult:
  """Builds and runs the runner exactly as `eval_disco._run_disco_loop` does."""
  runner = runners.DiscoRunner(config, config.build_parsers())
  return await runner.run_loop()


@dataclasses.dataclass(frozen=True)
class SmokeRun:
  """The raw outcome of a smoke run and its normalized snapshot."""

  result: records_lib.DiscoResult
  rounds_started: int
  call_log: fakes.CallLog
  snapshot: dict[str, Any]


def run_script(
    config: config_lib.DiscoConfig, script: Sequence[fakes.RoundScript]
) -> tuple[records_lib.DiscoResult, fakes.FakeWorld]:
  """Runs the runner hermetically on `script`, resuming any checkpoint.

  Args:
    config: Run config. A checkpoint under its output directory is resumed.
    script: One `RoundScript` per round this call is expected to run.

  Returns:
    The runner's result and the fake world (call log, rounds started).
  """
  with (
      fakes.hermetic_disco_env(script) as world,
      flagsaver.flagsaver(),
  ):
    result = asyncio.run(
        asyncio.wait_for(_run_runner(config), timeout=_TIMEOUT_SECONDS)
    )
  return result, world


def run_case(case: str, out_dir: str) -> SmokeRun:
  """Runs `case` hermetically, writing under `out_dir` (which must be empty).

  Args:
    case: One of CASE_NAMES.
    out_dir: Fresh output directory. The runner resumes from any checkpoint it
      finds there, so it must not be reused.

  Returns:
    The runner's result and the normalized snapshot.
  """
  result, world = run_script(make_config(case, out_dir), CASES[case].script)
  files, outputs = _snapshot_outputs(os.path.join(out_dir, _EXPERIMENT_NAME))
  snapshot = _normalize(
      golden_utils.to_jsonable({
          "result": result,
          "files": files,
          "outputs": outputs,
          "call_log": world.call_log,
      }),
      out_dir,
  )
  # Round-trip so the snapshot compares equal to one loaded from disk.
  snapshot = json.loads(json.dumps(snapshot, allow_nan=False))
  return SmokeRun(
      result=result,
      rounds_started=world.genai_client.rounds_started,
      call_log=list(world.call_log),
      snapshot=snapshot,
  )


def snapshot_case(case: str) -> dict[str, Any]:
  """Runs `case` in a fresh temporary directory and returns its snapshot."""
  with tempfile.TemporaryDirectory() as out_dir:
    return run_case(case, out_dir).snapshot


def _case_fn(case: str) -> Callable[[], dict[str, Any]]:
  return lambda: snapshot_case(case)


ALL_CASES: Mapping[str, Callable[[], dict[str, Any]]] = {
    golden_name(case): _case_fn(case) for case in CASE_NAMES
}
