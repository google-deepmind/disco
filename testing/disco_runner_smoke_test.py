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

r"""Hermetic end-to-end smoke test for `runners.DiscoRunner`.

Runs the eval_disco pipeline for each case in `disco_runner_smoke_cases` against
fake LLMs and a fake target model, checks the headline outcomes and that the
case took the branch it is about, and compares the full normalized output
(result, files written, call log) to a golden.

To regenerate the goldens after an intentional behavior change (list only the
cases whose behavior changed):

  python -m disco.testing.regenerate_goldens \
    --output_dir=testing/testdata \
    --only=disco_runner_smoke/disco_post_hoc,disco_runner_smoke/baseline,...
"""

import ast
import collections
import dataclasses
import inspect
import json
import os
import textwrap

from absl.testing import absltest
from absl.testing import parameterized

from disco import runners
from disco.agent import scrape_tolerance
from disco.core import checkpoint
from disco.core import config as config_lib
from disco.testing import disco_runner_smoke_cases as cases
from disco.testing import golden_utils

_RTOL = 1e-6
_MAX_REPORTED_MISMATCHES = 30


class DiscoRunnerSmokeTest(parameterized.TestCase):

  def test_make_config_pins_every_field(self):
    # Unpinned fields would follow DiscoConfig defaults, so a default change
    # would silently change the smoke runs.
    tree = ast.parse(textwrap.dedent(inspect.getsource(cases.make_config)))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "DiscoConfig"
    ]
    self.assertLen(calls, 1)
    pinned = {kw.arg for kw in calls[0].keywords}
    fields = {
        f.name for f in dataclasses.fields(config_lib.DiscoConfig) if f.init
    }
    self.assertEmpty(fields - pinned, "Pin these fields in make_config.")

  @parameterized.parameters(*cases.CASE_NAMES)
  def test_case(self, case: str):
    smoke_case = cases.CASES[case]
    run = cases.run_case(case, self.create_tempdir().full_path)
    result = run.result

    with self.subTest("outcomes"):
      self.assertLen(smoke_case.script, run.rounds_started)
      self.assertEqual(dict(result.probe_counts), dict(smoke_case.probe_counts))
      self.assertEqual(
          set(result.rejected_hypotheses), set(smoke_case.rejected)
      )
      self.assertEqual(
          dict(result.terminal_statuses), dict(smoke_case.terminal_statuses)
      )

    with self.subTest("budget"):
      budget = cases.make_config(case, "").budget
      self.assertEqual(
          sum(record.queries for record in result.all_records.values()),
          budget
          if smoke_case.total_queries is None
          else smoke_case.total_queries,
      )
      if smoke_case.queries:
        self.assertEqual(
            {h: r.queries for h, r in result.all_records.items()},
            dict(smoke_case.queries),
        )
      if smoke_case.e_value_counts:
        self.assertEqual(
            {h: len(r.e_values) for h, r in result.all_records.items()},
            dict(smoke_case.e_value_counts),
        )

    with self.subTest("branches"):
      counts = collections.Counter(run.call_log)
      self.assertEqual(
          {entry: counts[entry] for entry in smoke_case.call_counts},
          dict(smoke_case.call_counts),
      )

    with self.subTest("golden"):
      expected = golden_utils.load_golden(cases.golden_name(case))
      mismatches = golden_utils.find_mismatches(
          run.snapshot, expected, rtol=_RTOL
      )
      self.assertEmpty(
          mismatches,
          f"{len(mismatches)} mismatches against the golden, e.g.:\n"
          + "\n".join(mismatches[:_MAX_REPORTED_MISMATCHES]),
      )

  @parameterized.parameters(*cases.CASE_NAMES)
  def test_resume_restores_checkpointed_state(self, case: str):
    out_dir = self.create_tempdir().full_path
    cases.run_case(case, out_dir)
    config = cases.make_config(case, out_dir)
    with open(
        os.path.join(
            out_dir, config.experiment_name, checkpoint.CHECKPOINT_FILENAME
        )
    ) as f:
      on_disk = json.load(f)

    resumed = runners.DiscoRunner(config, config.build_parsers())
    logs = []
    self.assertTrue(resumed._load_checkpoint(log_fn=logs.append))
    # `to_checkpoint` writes set-valued fields sorted, so the round trip is
    # exact.
    self.assertEqual(
        checkpoint.to_checkpoint(
            resumed.state,
            runner_title=resumed._runner_title,
            budget=config.budget,
        ),
        on_disk,
    )
    self.assertIs(resumed.running_e_values, resumed.state.running_e_values)
    self.assertLen(logs, 1)
    self.assertStartsWith(logs[0], "[CHECKPOINT] Restored from round ")
    # A restored plan is reused by `RoundExecutor.preregister` instead of
    # being elicited again after data has been seen.
    self.assertEqual(
        {
            h_id
            for h_id, record in resumed.hypothesis_groups.items()
            if record.test_plan is not None
        },
        {
            h_id
            for h_id, data in on_disk["hypothesis_groups"].items()
            if "test_plan" in data
        },
    )

  def test_target_outage_raises_then_resumes(self):
    long_round, partial_round, short_round, medium_round = cases.CASES[
        "short_round"
    ].script
    empty_round = cases.CASES["scrape_failure"].script[1]
    out_dir = self.create_tempdir().full_path
    config = dataclasses.replace(
        cases.make_config("short_round", out_dir), max_consecutive_skips=2
    )

    # A target-side skip then a short skip: the second one in a row raises.
    with self.assertRaises(scrape_tolerance.TargetOutageError):
      cases.run_script(config, (long_round, empty_round, partial_round))

    stopped = runners.DiscoRunner(config, config.build_parsers())
    self.assertTrue(stopped._load_checkpoint(log_fn=lambda _: None))
    # Only round 1 was charged; the checkpoint is past both skipped rounds.
    self.assertEqual(stopped.remaining_budget, config.budget - 3)
    self.assertEqual(stopped.current_round, 4)
    self.assertEqual(
        stopped.state.skip_counts,
        {"target_side": 1, "short": 1, "auditor_side": 0},
    )

    # Once the target recovers, rerunning in the same directory resumes.
    result, world = cases.run_script(config, (short_round, medium_round))
    self.assertEqual(world.genai_client.rounds_started, 2)
    self.assertEqual(
        sum(record.queries for record in result.all_records.values()),
        config.budget,
    )
    self.assertEqual(
        {h: n for h, n in result.probe_counts.items() if n},
        {"long_tags": 1, "short_tags": 1, "medium_tags": 1},
    )


if __name__ == "__main__":
  absltest.main()
