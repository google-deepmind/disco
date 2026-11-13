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

"""Golden (characterization) tests for the DISCO bandit simulators.

These tests pin the behavior of ``bandit_disco`` and ``run_bandit_disco`` so
that refactors can be verified as behavior-preserving. If a test here fails
after an intentional behavior change, regenerate the goldens with
``python -m disco.bandit.generate_bandit_goldens --golden_dir=bandit/testdata``
and review the diff.

Also pins the JSON output schema (``params`` / ``summary`` / ``bounds_report``
/ ``alpha_sweep`` keys and file naming).
"""

import contextlib
import io
import json
import os
import pathlib
from typing import Any

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized

from disco.bandit import golden_cases
from disco.bandit import run_bandit_disco

FLAGS = flags.FLAGS


def _load_golden(filename: str) -> dict[str, Any]:
  path = pathlib.Path(__file__).parent / "testdata" / filename
  with open(path, "r") as f:
    return json.load(f)


class BanditResultGoldenTest(parameterized.TestCase):
  """Library-level: BanditResult for each (config, runner) must not drift."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.goldens = _load_golden(golden_cases.RESULT_GOLDENS_FILENAME)

  def test_golden_file_covers_all_cases(self):
    self.assertSameElements(
        golden_cases.RESULT_CASES.keys(),
        self.goldens.keys(),
        "RESULT_CASES and the golden file are out of sync; regenerate.",
    )

  @parameterized.named_parameters(
      (name, name) for name in sorted(golden_cases.RESULT_CASES)
  )
  def test_result_matches_golden(self, name: str):
    actual = golden_cases.run_result_case(golden_cases.RESULT_CASES[name])
    diffs = golden_cases.diff_structures(self.goldens[name], actual)
    self.assertEmpty(
        diffs, f"{name}: {len(diffs)} mismatch(es):\n" + "\n".join(diffs[:50])
    )


class RunBanditDiscoE2EGoldenTest(parameterized.TestCase):
  """Binary-level: the JSON files written by run_bandit_disco must not drift."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.goldens = _load_golden(golden_cases.E2E_GOLDENS_FILENAME)
    cls.actuals: dict[str, dict[str, Any]] = {}

  def _actual(self, name: str) -> dict[str, Any]:
    if name not in self.actuals:
      tmp = self.create_tempdir(name).full_path
      self.actuals[name] = golden_cases.run_e2e_case(
          golden_cases.E2E_CASES[name], tmp
      )
    return self.actuals[name]

  def test_golden_file_covers_all_cases(self):
    self.assertSameElements(
        golden_cases.E2E_CASES.keys(),
        self.goldens.keys(),
        "E2E_CASES and the golden file are out of sync; regenerate.",
    )

  @parameterized.named_parameters(
      (name, name) for name in sorted(golden_cases.E2E_CASES)
  )
  def test_result_json_matches_golden(self, name: str):
    actual = self._actual(name)
    diffs = golden_cases.diff_structures(
        self.goldens[name]["result"], actual["result"]
    )
    self.assertEmpty(
        diffs,
        f"{name} result.json: {len(diffs)} mismatch(es):\n"
        + "\n".join(diffs[:50]),
    )

  @parameterized.named_parameters(
      (name, name) for name in sorted(golden_cases.E2E_CASES)
  )
  def test_runs_json_matches_golden(self, name: str):
    actual = self._actual(name)
    diffs = golden_cases.diff_structures(
        self.goldens[name]["runs"], actual["runs"]
    )
    self.assertEmpty(
        diffs,
        f"{name} result_runs.json: {len(diffs)} mismatch(es):\n"
        + "\n".join(diffs[:50]),
    )

  @parameterized.named_parameters(
      (name, name) for name in sorted(golden_cases.E2E_CASES)
  )
  def test_stdout_content_matches_golden(self, name: str):
    """Stdout is compared after whitespace normalization (content only)."""
    actual = self._actual(name)
    self.assertEqual(self.goldens[name]["stdout"], actual["stdout"])


# =============================================================================
# Output contract (JSON schema and file naming)
# =============================================================================


def _extract_stat(metric_entry: Any) -> tuple[float | None, float]:
  if isinstance(metric_entry, dict):
    mean_val = metric_entry.get("mean")
    if mean_val is None:
      return None, 0.0
    ci95_val = metric_entry.get("ci95")
    if ci95_val is None or ci95_val == 0.0:
      sem = metric_entry.get("sem", 0.0)
      ci95_val = 1.96 * float(sem) if sem is not None else 0.0
    return float(mean_val), float(ci95_val)
  elif isinstance(metric_entry, (int, float)):
    return float(metric_entry), 0.0
  return None, 0.0


_REQUIRED_PARAMS_KEYS = (
    "runner_type",
    "alpha_calibration_mode",
    "alpha",
    "baseline_rate_logit",
    "budget",
    "chunk_size",
    "kappa",
    "e_min",
    "e_max",
    "n_max",
    "num_active_bins",
    "p_value_mode",
)
_HEADLINE_SUMMARY_METRICS = (
    "cluster_coverage",
    "penalized_cluster_coverage",
    "fdr",
    "recall",
    "macro_recall",
    "delta_post",
    "nominal_alpha",
)
_REQUIRED_BOUNDS_KEYS = ("prehoc_boosted", "prehoc_unboosted")
_STAT_SUBKEYS = ("mean", "std", "sem", "ci95", "ci_low", "ci_high")
# Full metric set written per runner.
_ALL_SUMMARY_METRICS = (
    "fdr",
    "fdr_exceedances",
    "fdr_exceed_rate",
    "recall",
    "macro_recall",
    "cluster_coverage",
    "penalized_cluster_coverage",
    "delta_post",
    "k",
    "tp",
    "fp",
    "probes_null",
    "probes_alt",
    "nominal_alpha",
    "exits_high",
    "exits_low",
    "exits_timeout",
    "exits_budget_expired",
)
_PER_RUN_KEYS = (
    "cluster_coverage",
    "penalized_cluster_coverage",
    "recall",
    "macro_recall",
    "empirical_fdr",
    "delta_post",
    "epsilon_post",
    "true_positives",
    "false_positives",
    "total_hypotheses_tested",
    "avg_probes_null",
    "avg_probes_alt",
    "nominal_alpha_used",
)


class OutputContractTest(absltest.TestCase):
  """Asserts the JSON output schema on live output."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    tmp = absltest.get_default_test_tmpdir()
    cls.all3 = golden_cases.run_e2e_case(
        golden_cases.E2E_CASES["e2e_all_bernoulli_j3"],
        os.path.join(tmp, "contract_all3"),
    )

  def test_top_level_keys(self):
    self.assertSameElements(
        ["params", "summary", "bounds_report", "alpha_sweep"],
        self.all3["result"].keys(),
    )
    self.assertSameElements(
        ["params", "num_mc_runs", "runs"], self.all3["runs"].keys()
    )

  def test_params_are_exactly_the_run_bandit_disco_flags(self):
    """Every key in params must correspond to a defined flag."""
    params = self.all3["result"]["params"]
    for key in params:
      self.assertIn(key, FLAGS, f"params key {key!r} is not a defined flag")
    for key in _REQUIRED_PARAMS_KEYS:
      self.assertIn(key, params)
    self.assertEqual(params["runner_type"], "all")

  def test_summary_keyed_by_runner_with_full_metric_set(self):
    summary = self.all3["result"]["summary"]
    self.assertSameElements(["offline_ebh", "type1", "disco"], summary.keys())
    for runner, sub in summary.items():
      for metric in _ALL_SUMMARY_METRICS:
        self.assertIn(metric, sub, f"{runner} missing {metric}")
        if metric == "fdr_exceedances":
          self.assertIsInstance(sub[metric], int)
          continue
        self.assertSameElements(
            _STAT_SUBKEYS, sub[metric].keys(), f"{runner}.{metric}"
        )
      # J=3 run: disco carries a bin_breakdown.
      if runner == "disco":
        self.assertIn("bin_breakdown", sub)

  def test_headline_metrics_resolve_on_every_runner(self):
    summary = self.all3["result"]["summary"]
    for runner, sub in summary.items():
      for metric in _HEADLINE_SUMMARY_METRICS:
        mean, _ = _extract_stat(sub.get(metric, {}))
        self.assertIsNotNone(mean, f"{runner}.{metric}")

  def test_bounds_report_populated_for_disco(self):
    bounds = self.all3["result"]["bounds_report"]
    for key in _REQUIRED_BOUNDS_KEYS:
      self.assertIn(key, bounds)
      self.assertIsInstance(bounds[key], float)

  def test_bounds_report_empty_without_disco(self):
    tmp = os.path.join(absltest.get_default_test_tmpdir(), "contract_type1")
    out = golden_cases.run_e2e_case(
        golden_cases.E2E_CASES["e2e_type1_bernoulli"], tmp
    )
    self.assertEqual(out["result"]["bounds_report"], {})
    self.assertEqual(out["result"]["alpha_sweep"], [])
    self.assertSameElements(["type1"], out["result"]["summary"].keys())

  def test_alpha_sweep_records(self):
    sweep = self.all3["result"]["alpha_sweep"]
    self.assertNotEmpty(sweep)
    for rec in sweep:
      self.assertSameElements(
          [
              "alpha_target",
              "nominal_alpha",
              "num_discoveries",
              "empirical_fdr",
              "recall",
              "epsilon_post",
              "marginal_posthoc",
              "delta_post",
          ],
          rec.keys(),
      )

  def test_per_run_file_schema(self):
    runs = self.all3["runs"]
    self.assertEqual(runs["num_mc_runs"], 2)
    self.assertSameElements(
        ["offline_ebh", "type1", "disco"], runs["runs"].keys()
    )
    for runner, arrays in runs["runs"].items():
      self.assertSameElements(_PER_RUN_KEYS, arrays.keys(), runner)
      for key, arr in arrays.items():
        self.assertLen(arr, 2, f"{runner}.{key}")

  def test_no_per_run_file_when_disabled(self):
    tmp = os.path.join(absltest.get_default_test_tmpdir(), "contract_noruns")
    out = golden_cases.run_e2e_case(
        golden_cases.E2E_CASES["e2e_disco_no_per_run"], tmp
    )
    self.assertIsNone(out["runs"])
    self.assertFalse(os.path.exists(os.path.join(tmp, "result_runs.json")))
    self.assertTrue(os.path.exists(os.path.join(tmp, "result.json")))

  def test_output_dir_filename_convention(self):
    """Output files in output_dir follow <runner>_seed<seed>_*.json."""
    tmp = os.path.join(absltest.get_default_test_tmpdir(), "contract_outdir")
    overrides = dict(golden_cases.E2E_CASES["e2e_type1_bernoulli"])
    overrides["output_path"] = ""
    overrides["output_dir"] = tmp
    with flagsaver.flagsaver(**overrides):
      with contextlib.redirect_stdout(io.StringIO()):
        run_bandit_disco.main(["run_bandit_disco"])
    files = sorted(os.listdir(tmp))
    self.assertLen(files, 2)
    main_file = [f for f in files if not f.endswith("_runs.json")][0]
    self.assertRegex(main_file, r"^type1_seed42_\d+\.json$")
    self.assertEqual(
        files[1 - files.index(main_file)], main_file[:-5] + "_runs.json"
    )


if __name__ == "__main__":
  absltest.main()
