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

"""DISCO run state and its JSON checkpoint.

`RunState` holds everything the discovery loop mutates and persists between
rounds. `CheckpointStore` writes it to
`<output_dir>/<experiment>/checkpoint.json`
and restores it on restart.

The JSON schema is read by external tools (such as
`tasks/visual_symmetry/verify_visual_symmetry`) and pinned by the smoke goldens,
so key names and order must stay stable:

  current_round, remaining_budget, total_scrape_rounds,
  hypothesis_groups: {h_id: {text, e_values: [{sub_hypothesis, e_value,
    cumulative_queries}], sub_hypotheses, failures, failure_reflections,
    executed_queries, queries, test_plan (only once pre-registered)}},
  runner_title, budget,
  extra: {running_e_values, final_e_values, probe_counts, terminal_statuses,
    bin_trajectories, prev_bins, transition_scores, active_hypotheses, k_total,
    skip_counts},
  pval_rng_state (only in seeded runs)

Not persisted: the p-value RNG state of an unseeded run, and the null tables /
bin edges (recomputed from config with a fixed seed).
"""

from collections.abc import Callable
import dataclasses
import json
import os
from typing import Any

from absl import logging

from disco.core import records as records_lib
from disco.stats import plan as stats_plan
from disco.verification import scoring

CHECKPOINT_FILENAME = "checkpoint.json"


def write_json_atomic(path: str, obj: Any, **json_kw: Any) -> None:
  """Atomically writes an object to a JSON file via a temporary file."""
  dir_name = os.path.dirname(path)
  if dir_name:
    os.makedirs(dir_name, exist_ok=True)
  tmp_path = f"{path}.tmp.{os.getpid()}"
  with open(tmp_path, "w", encoding="utf-8") as f:
    json.dump(obj, f, **json_kw)
  os.replace(tmp_path, path)


def empty_skip_counts() -> dict[str, int]:
  """Returns zeroed skipped-round counters, one per `scoring.SKIP_KINDS`."""
  return dict.fromkeys(scoring.SKIP_KINDS, 0)


@dataclasses.dataclass
class RunState:
  """Mutable per-run state of the DISCO discovery loop.

  Attributes:
    remaining_budget: Queries left to spend.
    current_round: 1-based round counter.
    total_scrape_rounds: Cumulative scrape folders written across agents.
    hypothesis_groups: Per-hypothesis history keyed by hypothesis id.
    running_e_values: Current cumulative e-value per hypothesis (private).
    final_e_values: E-value at deactivation per hypothesis (private).
    probe_counts: Number of probes per hypothesis.
    terminal_statuses: Terminal status per deactivated hypothesis.
    bin_trajectories: Bin index after each probe per hypothesis.
    prev_bins: Last bin index per hypothesis.
    transition_scores: Null transition score after each probe per hypothesis.
    active_hypotheses: Ids still being probed.
    k_total: Number of hypotheses proposed so far.
    skip_counts: Rounds skipped without an e-value, by kind (see
      `scoring.SKIP_KINDS`).
    pval_rng_state: `bit_generator.state` of the p-value RNG, saved only when
      the run is seeded (`random_seed`); None otherwise.
  """

  remaining_budget: int
  current_round: int = 1
  total_scrape_rounds: int = 0
  hypothesis_groups: dict[str, records_lib.HypothesisRecord] = (
      dataclasses.field(default_factory=dict)
  )
  running_e_values: dict[str, float] = dataclasses.field(default_factory=dict)
  final_e_values: dict[str, float] = dataclasses.field(default_factory=dict)
  probe_counts: dict[str, int] = dataclasses.field(default_factory=dict)
  terminal_statuses: dict[str, str] = dataclasses.field(default_factory=dict)
  bin_trajectories: dict[str, list[int]] = dataclasses.field(
      default_factory=dict
  )
  prev_bins: dict[str, int] = dataclasses.field(default_factory=dict)
  transition_scores: dict[str, list[float]] = dataclasses.field(
      default_factory=dict
  )
  active_hypotheses: set[str] = dataclasses.field(default_factory=set)
  k_total: int = 0
  skip_counts: dict[str, int] = dataclasses.field(
      default_factory=empty_skip_counts
  )
  pval_rng_state: dict[str, Any] | None = None


def serialize_record(record: records_lib.HypothesisRecord) -> dict[str, Any]:
  """Returns the checkpoint JSON form of one hypothesis record."""
  data: dict[str, Any] = {
      "text": record.text,
      "e_values": [
          {
              "sub_hypothesis": ev.sub_hypothesis,
              "e_value": ev.e_value,
              "cumulative_queries": ev.cumulative_queries,
          }
          for ev in record.e_values
      ],
      "sub_hypotheses": list(record.sub_hypotheses),
      "failures": list(record.failures),
      "failure_reflections": list(record.failure_reflections),
      "executed_queries": sorted(record.executed_queries),
      "queries": record.queries,
  }
  if record.test_plan is not None:
    data["test_plan"] = dataclasses.asdict(record.test_plan)
  return data


def deserialize_record(data: dict[str, Any]) -> records_lib.HypothesisRecord:
  """Inverse of `serialize_record` (missing optional keys get defaults).

  A checkpoint written before the test plan was persisted has no `test_plan`
  key; the record then has no plan and the next round elicits one.

  Args:
    data: One entry of the checkpoint's `hypothesis_groups`.

  Returns:
    Deserialized `HypothesisRecord` instance.
  """
  test_plan = data.get("test_plan")
  return records_lib.HypothesisRecord(
      text=data["text"],
      e_values=[
          records_lib.EValueEntry(
              sub_hypothesis=ev["sub_hypothesis"],
              e_value=ev["e_value"],
              cumulative_queries=ev.get("cumulative_queries", 0),
          )
          for ev in data.get("e_values", [])
      ],
      sub_hypotheses=data.get("sub_hypotheses", []),
      failures=data.get("failures", []),
      failure_reflections=data.get("failure_reflections", []),
      executed_queries=set(data.get("executed_queries", [])),
      queries=data.get("queries", 0),
      test_plan=(
          stats_plan.PreRegisteredTestPlan(**test_plan)
          if test_plan is not None
          else None
      ),
  )


def serialize_extra(state: RunState) -> dict[str, Any]:
  """Returns the checkpoint `extra` block (DISCO bookkeeping)."""
  return {
      "running_e_values": dict(state.running_e_values),
      "final_e_values": dict(state.final_e_values),
      "probe_counts": dict(state.probe_counts),
      "terminal_statuses": dict(state.terminal_statuses),
      "bin_trajectories": {
          k: list(v) for k, v in state.bin_trajectories.items()
      },
      "prev_bins": dict(state.prev_bins),
      "transition_scores": {
          k: list(v) for k, v in state.transition_scores.items()
      },
      "active_hypotheses": sorted(state.active_hypotheses),
      "k_total": state.k_total,
      "skip_counts": dict(state.skip_counts),
  }


def restore_extra(state: RunState, extra: dict[str, Any]) -> None:
  """Restores the `extra` block into `state`; an empty block is a no-op.

  A checkpoint written before skipped rounds were counted restores zeros.

  Args:
    state: State to overwrite.
    extra: The checkpoint's `extra` block.
  """
  if not extra:
    return
  state.running_e_values = extra.get("running_e_values", {})
  state.final_e_values = extra.get("final_e_values", {})
  state.probe_counts = extra.get("probe_counts", {})
  state.terminal_statuses = extra.get("terminal_statuses", {})
  state.bin_trajectories = {
      k: list(v) for k, v in extra.get("bin_trajectories", {}).items()
  }
  state.prev_bins = extra.get("prev_bins", {})
  state.transition_scores = {
      k: list(v) for k, v in extra.get("transition_scores", {}).items()
  }
  state.active_hypotheses = set(extra.get("active_hypotheses", []))
  state.k_total = extra.get("k_total", 0)
  state.skip_counts = {
      **empty_skip_counts(),
      **extra.get("skip_counts", {}),
  }


def to_checkpoint(
    state: RunState, *, runner_title: str, budget: int
) -> dict[str, Any]:
  """Returns the full checkpoint payload in its on-disk key order."""
  checkpoint: dict[str, Any] = {
      "current_round": state.current_round,
      "remaining_budget": state.remaining_budget,
      "total_scrape_rounds": state.total_scrape_rounds,
      "hypothesis_groups": {
          h_id: serialize_record(rec)
          for h_id, rec in state.hypothesis_groups.items()
      },
      "runner_title": runner_title,
      "budget": budget,
  }
  extra = serialize_extra(state)
  if extra:
    checkpoint["extra"] = extra
  if state.pval_rng_state is not None:
    checkpoint["pval_rng_state"] = state.pval_rng_state
  return checkpoint


def restore_checkpoint(
    state: RunState, checkpoint: dict[str, Any], *, budget_delta: int
) -> None:
  """Restores a loaded checkpoint payload into `state`.

  Args:
    state: State to overwrite.
    checkpoint: Parsed checkpoint JSON.
    budget_delta: Budget increase since the checkpoint; added to the restored
      remaining budget.
  """
  state.current_round = checkpoint["current_round"]
  state.remaining_budget = checkpoint["remaining_budget"] + budget_delta
  state.total_scrape_rounds = checkpoint.get("total_scrape_rounds", 0)
  state.hypothesis_groups = {
      h_id: deserialize_record(data)
      for h_id, data in checkpoint.get("hypothesis_groups", {}).items()
  }
  restore_extra(state, checkpoint.get("extra", {}))
  state.pval_rng_state = checkpoint.get("pval_rng_state")


class CheckpointStore:
  """Reads and writes `checkpoint.json` for one experiment directory."""

  def __init__(self, path: str):
    self.path = path

  def save(self, state: RunState, *, runner_title: str, budget: int) -> None:
    """Writes the checkpoint via a temp file, then moves it into place.

    The previous checkpoint is only replaced once the new one is complete
    (`write_json_atomic`). A failed save is logged, not raised: it costs a
    restart its latest progress, but must not abort the run.

    Args:
      state: Run state to save.
      runner_title: Recorded in the checkpoint.
      budget: Recorded in the checkpoint.
    """
    checkpoint = to_checkpoint(state, runner_title=runner_title, budget=budget)
    try:
      write_json_atomic(self.path, checkpoint, indent=2)
    except (OSError, TypeError, ValueError) as e:
      logging.warning(
          "Could not save checkpoint %s: %s: %s", self.path, type(e).__name__, e
      )

  def load_into(
      self,
      state: RunState,
      *,
      budget: int,
      log_fn: Callable[..., None] | None = None,
  ) -> bool:
    """Restores `state` from disk if a compatible checkpoint exists.

    A checkpoint whose budget exceeds `budget` is ignored (fresh start); a
    smaller one is resumed with the difference added to the remaining budget.

    Args:
      state: State to overwrite on success.
      budget: The current run's total query budget.
      log_fn: Optional logging function.

    Returns:
      True if a checkpoint was restored.
    """
    if not os.path.exists(self.path):
      return False

    try:
      with open(self.path, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as e:
      if log_fn:
        log_fn(
            "[CHECKPOINT] Warning: failed to load checkpoint"
            f" ({e}), starting fresh."
        )
      return False

    ckpt_budget = checkpoint.get("budget", 0)
    if budget < ckpt_budget:
      if log_fn:
        log_fn(
            f"[CHECKPOINT] Config budget ({budget}) is smaller than"
            f" checkpoint budget ({ckpt_budget}), starting fresh."
        )
      return False

    restore_checkpoint(state, checkpoint, budget_delta=budget - ckpt_budget)
    return True
