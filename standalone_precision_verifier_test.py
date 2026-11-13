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

"""Unit tests for the standalone precision verifier and hypothesis verification."""

import asyncio
import dataclasses
import json
import os
from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from google.genai import types

from disco import precision_estimation
from disco import standalone_agent
from disco import standalone_precision_verifier
from disco.core import records as records_lib
from disco.models import registry as model_registry
from disco.stats import plan as stats_plan


async def _fake_isolated_queries(**kwargs) -> list[str]:
  """Stands in for the isolated query generator: no LLM call."""
  return [f"query_{i}" for i in range(kwargs["q_count"])]


class StandalonePrecisionVerifierTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = self.create_tempdir().full_path

  def _patch_isolated_queries(self) -> None:
    """Replaces the isolated query generator for the rest of the test."""
    self.enter_context(
        mock.patch.object(
            precision_estimation,
            "_generate_queries_isolated",
            new=_fake_isolated_queries,
        )
    )

  def test_load_hypotheses_dict_of_strings(self):
    path = os.path.join(self.test_dir, "hypotheses_dict.json")
    with open(path, "w") as f:
      json.dump(
          {
              "h_counting": "Model miscounts objects when > 5",
              "h_refusal": "Model refuses medication dosage questions",
          },
          f,
      )

    hyps = standalone_precision_verifier.load_hypotheses(path)
    self.assertEqual(
        hyps,
        [
            ("h_counting", "Model miscounts objects when > 5"),
            ("h_refusal", "Model refuses medication dosage questions"),
        ],
    )

  def test_load_hypotheses_dict_of_dicts(self):
    path = os.path.join(self.test_dir, "hypotheses_dict_nested.json")
    with open(path, "w") as f:
      json.dump(
          {
              "h_1": {"text": "Hypothesis 1 text", "category": "math"},
              "h_2": {"hypothesis": "Hypothesis 2 text", "category": "logic"},
          },
          f,
      )

    hyps = standalone_precision_verifier.load_hypotheses(path)
    self.assertEqual(
        hyps,
        [
            ("h_1", "Hypothesis 1 text"),
            ("h_2", "Hypothesis 2 text"),
        ],
    )

  def test_load_hypotheses_top_level_key(self):
    path = os.path.join(self.test_dir, "hypotheses_wrapper.json")
    with open(path, "w") as f:
      json.dump(
          {
              "hypotheses": {
                  "h_a": "Wrapped hypothesis A",
                  "h_b": "Wrapped hypothesis B",
              }
          },
          f,
      )

    hyps = standalone_precision_verifier.load_hypotheses(path)
    self.assertEqual(
        hyps,
        [
            ("h_a", "Wrapped hypothesis A"),
            ("h_b", "Wrapped hypothesis B"),
        ],
    )

  def test_load_hypotheses_rejected_hypotheses_key(self):
    path = os.path.join(self.test_dir, "rejected_wrapper.json")
    with open(path, "w") as f:
      json.dump(
          {
              "rejected_hypotheses": [
                  "Rejected hypothesis 1",
                  "Rejected hypothesis 2",
              ]
          },
          f,
      )

    hyps = standalone_precision_verifier.load_hypotheses(path)
    self.assertEqual(
        hyps,
        [
            ("h_0", "Rejected hypothesis 1"),
            ("h_1", "Rejected hypothesis 2"),
        ],
    )

  def test_load_hypotheses_verified_hypotheses_key(self):
    path = os.path.join(self.test_dir, "verified_wrapper.json")
    with open(path, "w") as f:
      json.dump(
          {
              "verified_hypotheses": [
                  "Verified hypothesis 1",
                  "Verified hypothesis 2",
              ]
          },
          f,
      )

    hyps = standalone_precision_verifier.load_hypotheses(path)
    self.assertEqual(
        hyps,
        [
            ("h_0", "Verified hypothesis 1"),
            ("h_1", "Verified hypothesis 2"),
        ],
    )

  def test_load_hypotheses_bundled_example(self):
    example_path = os.path.join(
        os.path.dirname(__file__), "example_hypotheses.json"
    )
    hyps = standalone_precision_verifier.load_hypotheses(example_path)
    self.assertNotEmpty(hyps)
    self.assertIn("h_recipe_french_longer", [h[0] for h in hyps])

  def test_load_hypotheses_list_of_strings(self):
    path = os.path.join(self.test_dir, "hypotheses_list.json")
    with open(path, "w") as f:
      json.dump(
          [
              "List hypothesis 0",
              "List hypothesis 1",
          ],
          f,
      )

    hyps = standalone_precision_verifier.load_hypotheses(path)
    self.assertEqual(
        hyps,
        [
            ("h_0", "List hypothesis 0"),
            ("h_1", "List hypothesis 1"),
        ],
    )

  def test_load_hypotheses_list_of_dicts(self):
    path = os.path.join(self.test_dir, "hypotheses_list_dicts.json")
    with open(path, "w") as f:
      json.dump(
          [
              {"label": "custom_1", "text": "Hypothesis 1"},
              {"id": "custom_2", "text": "Hypothesis 2"},
          ],
          f,
      )

    hyps = standalone_precision_verifier.load_hypotheses(path)
    self.assertEqual(
        hyps,
        [
            ("custom_1", "Hypothesis 1"),
            ("custom_2", "Hypothesis 2"),
        ],
    )

  @mock.patch.object(
      standalone_precision_verifier.precision_estimation,
      "run_precision_check",
  )
  def test_run_verifier_success(self, mock_run_prec_check):
    hyp_path = os.path.join(self.test_dir, "hypotheses.json")
    with open(hyp_path, "w") as f:
      json.dump(
          {
              "h_counting": "Model miscounts objects when > 5",
              "h_dates": "Model hallucinates paper publication dates",
          },
          f,
      )

    mock_run_prec_check.return_value = precision_estimation.PrecisionResult(
        records=[
            precision_estimation.HypothesisPrecisionRecord(
                hypothesis_id="h_counting",
                hypothesis_text="Model miscounts objects when > 5",
                verified=True,
                final_e_product=8.5,
                n_subhyps_used=2,
                e_values=[("Sub 1", 2.0), ("Sub 2", 4.25)],
            ),
            precision_estimation.HypothesisPrecisionRecord(
                hypothesis_id="h_dates",
                hypothesis_text="Model hallucinates paper publication dates",
                verified=False,
                final_e_product=0.5,
                n_subhyps_used=5,
                e_values=[("Sub 1", 0.5)],
            ),
        ],
        precision=0.5,
        n_verified=1,
        n_total=2,
    )

    out_dir = os.path.join(self.test_dir, "output")
    with flagsaver.flagsaver(
        hypotheses_path=hyp_path,
        target_model="gemini-3.8-flash",
        output_dir=out_dir,
        alpha=0.5,
    ):
      standalone_precision_verifier.asyncio.run(
          standalone_precision_verifier.run_verifier()
      )

    mock_run_prec_check.assert_called_once()
    passed_config = mock_run_prec_check.call_args.kwargs["config"]
    self.assertEqual(
        passed_config.rating_model,
        model_registry.resolve_model_url(model_registry.DEFAULT_GEMINI_MODEL),
    )
    self.assertEqual(
        passed_config.summarizer_model,
        model_registry.resolve_model_url(model_registry.DEFAULT_GEMINI_MODEL),
    )
    out_path = os.path.join(out_dir, "verification_results.json")
    self.assertTrue(os.path.exists(out_path))

    with open(out_path, "r") as f:
      res = json.load(f)

    self.assertEqual(res["precision"], 0.5)
    self.assertEqual(res["n_verified"], 1)
    self.assertEqual(res["n_total"], 2)
    self.assertEqual(res["empirical_fdr"], 0.5)
    self.assertLen(res["records"], 2)
    self.assertEqual(res["records"][0]["label"], "h_counting")
    self.assertTrue(res["records"][0]["verified"])
    self.assertEqual(res["records"][1]["label"], "h_dates")
    self.assertFalse(res["records"][1]["verified"])
    self.assertEqual(res["config"]["target_model"], "gemini-3.8-flash")

    log_path = os.path.join(out_dir, "verifier.log")
    self.assertTrue(os.path.exists(log_path))
    with open(log_path, "r") as f:
      log_content = f.read()
    if isinstance(log_content, bytes):
      log_content = log_content.decode("utf-8")
    self.assertIn("=== Standalone Precision Verifier Log Start", log_content)
    self.assertIn("h_counting", log_content)

    flags_path = os.path.join(out_dir, "flags.json")
    self.assertTrue(os.path.exists(flags_path))

  @mock.patch.object(
      standalone_precision_verifier.precision_estimation,
      "run_precision_check",
  )
  def test_run_verifier_writes_data_quality(self, mock_run_prec_check):
    hyp_path = os.path.join(self.test_dir, "hypotheses_dq.json")
    with open(hyp_path, "w") as f:
      json.dump({"h_ok": "Hypothesis ok", "h_fail": "Hypothesis fail"}, f)

    mock_run_prec_check.return_value = precision_estimation.PrecisionResult(
        records=[
            precision_estimation.HypothesisPrecisionRecord(
                hypothesis_id="h_ok",
                hypothesis_text="Hypothesis ok",
                verified=True,
                sample_size=9,
                n_requested=10,
                n_received=9,
            ),
            precision_estimation.HypothesisPrecisionRecord(
                hypothesis_id="h_fail",
                hypothesis_text="Hypothesis fail",
                verified=False,
                status="SCRAPE_FAILED",
                n_requested=10,
                n_received=0,
                failure_reason="SCRAPE_FAILED: No valid responses returned",
            ),
        ],
        precision=1.0,
        n_verified=1,
        n_total=2,
    )

    out_dir = os.path.join(self.test_dir, "output_dq")
    with flagsaver.flagsaver(
        hypotheses_path=hyp_path,
        target_model="gemini-3.8-flash",
        output_dir=out_dir,
    ):
      standalone_precision_verifier.asyncio.run(
          standalone_precision_verifier.run_verifier()
      )

    with open(os.path.join(out_dir, "verification_results.json"), "r") as f:
      res = json.load(f)
    self.assertEqual(
        res["data_quality"],
        {
            "n_records": 2,
            "n_failed": 1,
            "by_status": {"OK": 1, "SCRAPE_FAILED": 1},
            "n_requested_total": 20,
            "n_received_total": 9,
            "response_coverage": 0.45,
        },
    )
    self.assertEqual(
        res["records"][1]["failure_reason"],
        "SCRAPE_FAILED: No valid responses returned",
    )
    self.assertEqual(res["records"][0]["n_received"], 9)
    with open(os.path.join(out_dir, "verifier.log"), "r") as f:
      log_content = f.read()
    if isinstance(log_content, bytes):
      log_content = log_content.decode("utf-8")
    self.assertIn("[Data Quality] WARNING: 1/2", log_content)

  def _write_results(self, out_dir, records, **extra):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "verification_results.json")
    data = {
        "precision": 1.0,
        "records": records,
        "config": {"target_model": "gemini-3.8-flash", "alpha": 0.05},
        "custom_key": "kept",
        **extra,
    }
    with open(path, "w") as f:
      json.dump(data, f)
    return path, data

  @mock.patch.object(
      standalone_precision_verifier.precision_estimation,
      "run_precision_check",
  )
  def test_retry_failed_reruns_only_failed_records(self, mock_run_prec_check):
    out_dir = os.path.join(self.test_dir, "output_retry")
    path, original = self._write_results(
        out_dir,
        [
            {
                "label": "h_ok",
                "hypothesis_text": "ok",
                "verified": True,
                "status": "OK",
            },
            {
                "label": "h_sf",
                "hypothesis_text": "sf",
                "verified": False,
                "status": "SCRAPE_FAILED",
            },
            {
                "label": "h_plan",
                "hypothesis_text": "plan",
                "verified": False,
                "status": "PLAN_INCONSISTENT",
            },
            {
                "label": "h_short",
                "hypothesis_text": "short",
                "verified": False,
                "status": "INSUFFICIENT_RESPONSES",
                "n_attempts": 2,
            },
        ],
    )
    mock_run_prec_check.return_value = precision_estimation.PrecisionResult(
        records=[
            precision_estimation.HypothesisPrecisionRecord(
                hypothesis_id="h_sf",
                hypothesis_text="sf",
                verified=True,
                sample_size=10,
                n_requested=10,
                n_received=10,
            ),
            precision_estimation.HypothesisPrecisionRecord(
                hypothesis_id="h_short",
                hypothesis_text="short",
                verified=False,
                status="INSUFFICIENT_RESPONSES",
                n_requested=10,
                n_received=5,
            ),
        ],
        precision=1.0,
        n_verified=1,
        n_total=2,
    )

    with flagsaver.flagsaver(
        target_model="gemini-3.8-flash",
        output_dir=out_dir,
        alpha=0.1,
        retry_failed=True,
    ):
      standalone_precision_verifier.asyncio.run(
          standalone_precision_verifier.run_verifier()
      )

    mock_run_prec_check.assert_called_once()
    checked = mock_run_prec_check.call_args.kwargs["verified_hypotheses"]
    self.assertEqual(
        [(label, h.text) for label, h in checked],
        [("h_sf", "sf"), ("h_short", "short")],
    )
    with open(
        os.path.join(out_dir, "verification_results.pre_retry.json")
    ) as f:
      self.assertEqual(json.load(f), original)
    with open(path) as f:
      res = json.load(f)
    self.assertEqual(
        [
            (r["label"], r["status"], r.get("n_attempts"))
            for r in res["records"]
        ],
        [
            ("h_ok", "OK", None),
            ("h_sf", "OK", 2),
            ("h_plan", "PLAN_INCONSISTENT", None),
            ("h_short", "INSUFFICIENT_RESPONSES", 3),
        ],
    )
    # Two verified OK records; PLAN_INCONSISTENT and the still-short record
    # are dropped (the default --failed_records=drop).
    self.assertEqual(res["precision"], 1.0)
    self.assertEqual(res["n_scored"], 2)
    self.assertEqual(
        res["n_failed_by_status"],
        {"INSUFFICIENT_RESPONSES": 1, "PLAN_INCONSISTENT": 1},
    )
    self.assertEqual(res["n_total"], 4)
    self.assertEqual(res["custom_key"], "kept")
    self.assertEqual(res["config"], original["config"])
    [entry] = res["retry_history"]
    self.assertEqual(entry["n_retried"], 2)
    self.assertEqual(entry["n_recovered"], 1)
    self.assertEqual(entry["config_drift"], ["alpha"])
    self.assertEqual(res["data_quality"]["n_failed"], 2)

    # A second retry keeps the first backup and adds to the history.
    mock_run_prec_check.reset_mock()
    mock_run_prec_check.return_value = precision_estimation.PrecisionResult(
        records=[
            precision_estimation.HypothesisPrecisionRecord(
                hypothesis_id="h_short", hypothesis_text="short", verified=True
            ),
        ],
        precision=1.0,
        n_verified=1,
        n_total=1,
    )
    with flagsaver.flagsaver(
        target_model="gemini-3.8-flash",
        output_dir=out_dir,
        alpha=0.05,
        retry_failed=True,
    ):
      standalone_precision_verifier.asyncio.run(
          standalone_precision_verifier.run_verifier()
      )
    self.assertLen(
        mock_run_prec_check.call_args.kwargs["verified_hypotheses"], 1
    )
    self.assertTrue(
        os.path.exists(
            os.path.join(out_dir, "verification_results.pre_retry.2.json")
        )
    )
    with open(path) as f:
      res = json.load(f)
    self.assertLen(res["retry_history"], 2)
    self.assertNotIn("config_drift", res["retry_history"][1])
    self.assertEqual(res["records"][3]["n_attempts"], 4)
    self.assertEqual(res["n_scored"], 3)

  @mock.patch.object(
      standalone_precision_verifier.precision_estimation,
      "run_precision_check",
  )
  def test_retry_failed_reuses_stored_test_plans(self, mock_run_prec_check):
    out_dir = os.path.join(self.test_dir, "output_retry_plans")
    plan = stats_plan.PreRegisteredTestPlan(
        test_type="one_sample_proportion_z_test",
        target_metric="rate",
        null_baseline=0.2,
        alternative="larger",
        reasoning="r",
        is_paired=False,
    )
    path, _ = self._write_results(
        out_dir,
        [
            {
                "label": "h_planned",
                "hypothesis_text": "planned",
                "verified": False,
                "status": "SCRAPE_FAILED",
                "test_plan": dataclasses.asdict(plan),
                "allocated_queries": 80,
            },
            # Written before plans were saved: pre-registers again.
            {
                "label": "h_legacy",
                "hypothesis_text": "legacy",
                "verified": False,
                "status": "SCRAPE_FAILED",
            },
        ],
    )
    mock_run_prec_check.return_value = precision_estimation.PrecisionResult(
        records=[
            precision_estimation.HypothesisPrecisionRecord(
                hypothesis_id="h_planned",
                hypothesis_text="planned",
                verified=True,
                test_plan=plan,
                allocated_queries=80,
            ),
            precision_estimation.HypothesisPrecisionRecord(
                hypothesis_id="h_legacy",
                hypothesis_text="legacy",
                verified=False,
            ),
        ],
        precision=0.5,
        n_verified=1,
        n_total=2,
    )

    with flagsaver.flagsaver(
        target_model="gemini-3.8-flash",
        output_dir=out_dir,
        retry_failed=True,
    ):
      standalone_precision_verifier.asyncio.run(
          standalone_precision_verifier.run_verifier()
      )

    self.assertEqual(
        mock_run_prec_check.call_args.kwargs["stored_plans"],
        {"h_planned": plan},
    )
    self.assertEqual(
        mock_run_prec_check.call_args.kwargs["retry_budget"],
        precision_estimation.RetryBudget(
            allocated_queries={"h_planned": 80, "h_legacy": 0}, n_original=2
        ),
    )
    with open(path) as f:
      res = json.load(f)
    self.assertEqual(
        [r["test_plan"] for r in res["records"]],
        [dataclasses.asdict(plan), None],
    )
    self.assertEqual([r["allocated_queries"] for r in res["records"]], [80, 0])

  @mock.patch.object(
      standalone_precision_verifier.precision_estimation,
      "run_precision_check",
  )
  def test_retry_failed_with_nothing_failed_changes_nothing(
      self, mock_run_prec_check
  ):
    out_dir = os.path.join(self.test_dir, "output_retry_noop")
    path, original = self._write_results(
        out_dir,
        [{
            "label": "h",
            "hypothesis_text": "t",
            "verified": True,
            "status": "OK",
        }],
    )
    with flagsaver.flagsaver(output_dir=out_dir, retry_failed=True):
      standalone_precision_verifier.asyncio.run(
          standalone_precision_verifier.run_verifier()
      )
    mock_run_prec_check.assert_not_called()
    with open(path) as f:
      self.assertEqual(json.load(f), original)
    self.assertFalse(
        os.path.exists(
            os.path.join(out_dir, "verification_results.pre_retry.json")
        )
    )

  @mock.patch.object(
      standalone_precision_verifier.precision_estimation,
      "run_precision_check",
  )
  def test_retry_failed_without_existing_result_runs_everything(
      self, mock_run_prec_check
  ):
    hyp_path = os.path.join(self.test_dir, "hypotheses_retry.json")
    with open(hyp_path, "w") as f:
      json.dump({"h_a": "A", "h_b": "B"}, f)
    mock_run_prec_check.return_value = precision_estimation.PrecisionResult(
        records=[], precision=None, n_verified=0, n_total=0
    )
    out_dir = os.path.join(self.test_dir, "output_retry_fresh")
    with flagsaver.flagsaver(
        hypotheses_path=hyp_path,
        output_dir=out_dir,
        retry_failed=True,
    ):
      standalone_precision_verifier.asyncio.run(
          standalone_precision_verifier.run_verifier()
      )
    self.assertLen(
        mock_run_prec_check.call_args.kwargs["verified_hypotheses"], 2
    )

  def test_retry_failed_requires_hypotheses_without_existing_result(self):
    with flagsaver.flagsaver(
        output_dir=os.path.join(self.test_dir, "nothing_here"),
        retry_failed=True,
    ):
      with self.assertRaisesRegex(ValueError, "--hypotheses_path"):
        standalone_precision_verifier.asyncio.run(
            standalone_precision_verifier.run_verifier()
        )

  def test_annotate_coverage_sums_scrapes_and_sets_reason(self):
    def outcome(req: int, rcv: int) -> mock.Mock:
      return mock.Mock(n_requested=req, n_received=rcv)

    agents = [
        mock.Mock(scrape_history=[outcome(5, 5), outcome(5, 2)]),
        mock.Mock(scrape_history=[outcome(4, 0)]),
        object(),  # An agent without scrape history adds nothing.
    ]
    failed = precision_estimation._annotate_coverage(
        precision_estimation.HypothesisPrecisionRecord(
            hypothesis_text="h",
            verified=False,
            status="SCRAPE_FAILED",
            block_reasons=["blocked", "empty"],
        ),
        agents,
    )
    self.assertEqual((failed.n_requested, failed.n_received), (14, 7))
    self.assertEqual(failed.failure_reason, "blocked; empty")

    inconsistent = precision_estimation._annotate_coverage(
        precision_estimation.HypothesisPrecisionRecord(
            hypothesis_text="h",
            verified=False,
            status="PLAN_INCONSISTENT",
            failure_reason="test type mismatch",
        ),
        [],
    )
    self.assertEqual(inconsistent.failure_reason, "test type mismatch")

    ok = precision_estimation._annotate_coverage(
        precision_estimation.HypothesisPrecisionRecord(
            hypothesis_text="h", verified=True
        ),
        [],
    )
    self.assertEqual(ok.failure_reason, "")

  @mock.patch.object(
      standalone_precision_verifier.precision_estimation,
      "run_precision_check",
  )
  def test_run_verifier_appends_to_existing_log(self, mock_run_prec_check):
    """A restarted/resumed run must not truncate the previous attempt's log."""
    hyp_path = os.path.join(self.test_dir, "hypotheses.json")
    with open(hyp_path, "w") as f:
      json.dump({"h_counting": "Model miscounts objects when > 5"}, f)

    mock_run_prec_check.return_value = precision_estimation.PrecisionResult(
        records=[
            precision_estimation.HypothesisPrecisionRecord(
                hypothesis_id="h_counting",
                hypothesis_text="Model miscounts objects when > 5",
                verified=True,
                final_e_product=8.5,
                n_subhyps_used=2,
                e_values=[("Sub 1", 2.0), ("Sub 2", 4.25)],
            ),
        ],
        precision=1.0,
        n_verified=1,
        n_total=1,
    )

    out_dir = os.path.join(self.test_dir, "append_output")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "verifier.log")
    sentinel = "PRIOR ATTEMPT LOG LINE - MUST SURVIVE RESTART"
    with open(log_path, "w") as f:
      f.write(sentinel + "\n")

    with flagsaver.flagsaver(
        hypotheses_path=hyp_path,
        target_model="gemini-3.8-flash",
        output_dir=out_dir,
    ):
      standalone_precision_verifier.asyncio.run(
          standalone_precision_verifier.run_verifier()
      )

    with open(log_path, "r") as f:
      log_content = f.read()
    if isinstance(log_content, bytes):
      log_content = log_content.decode("utf-8")

    # The previous attempt's content survives, and the new attempt's banner is
    # appended after it.
    self.assertIn(sentinel, log_content)
    self.assertIn("=== Standalone Precision Verifier Log Start", log_content)
    self.assertLess(
        log_content.index(sentinel),
        log_content.index("=== Standalone Precision Verifier Log Start"),
    )

  def _run_verifier_for_config(
      self, mock_run_prec_check, name: str, yaml_text: str = "", **flag_values
  ):
    """Runs the verifier on one hypothesis and returns the config it got.

    Args:
      mock_run_prec_check: The patched `run_precision_check`.
      name: Unique name for this run's files under the test directory.
      yaml_text: Contents of a `--config` YAML file, if any.
      **flag_values: Flags to set for the run.

    Returns:
      The VerifierConfig passed to `run_precision_check`.
    """
    hyp_path = os.path.join(self.test_dir, f"hypotheses_{name}.json")
    with open(hyp_path, "w") as f:
      json.dump({"h_1": "Hypothesis 1"}, f)
    mock_run_prec_check.return_value = precision_estimation.PrecisionResult(
        records=[], precision=1.0, n_verified=0, n_total=0
    )
    if yaml_text:
      config_path = os.path.join(self.test_dir, f"{name}.yaml")
      with open(config_path, "w") as f:
        f.write(yaml_text)
      flag_values["config"] = config_path
    with flagsaver.flagsaver(
        hypotheses_path=hyp_path,
        output_dir=os.path.join(self.test_dir, f"output_{name}"),
        **flag_values,
    ):
      standalone_precision_verifier.asyncio.run(
          standalone_precision_verifier.run_verifier()
      )
    mock_run_prec_check.assert_called_once()
    return mock_run_prec_check.call_args.kwargs["config"]

  @mock.patch.object(
      standalone_precision_verifier.precision_estimation,
      "run_precision_check",
  )
  def test_backbone_model_flag_fans_out_to_every_role(
      self, mock_run_prec_check
  ):
    cfg = self._run_verifier_for_config(
        mock_run_prec_check,
        "backbone",
        target_model="gemini-3.8-flash",
        backbone_model="models/gemini-3.1-flash-image",
    )
    self.assertEqual(cfg.backbone_model, "gemini-3.1-flash-image")
    self.assertEqual(cfg.agent_model, "gemini-3.1-flash-image")
    self.assertEqual(cfg.parser_model, "gemini-3.1-flash-image")
    self.assertEqual(cfg.rating_model, "gemini-3.1-flash-image")
    self.assertEqual(cfg.summarizer_model, "gemini-3.1-flash-image")

  @mock.patch.object(
      standalone_precision_verifier.precision_estimation,
      "run_precision_check",
  )
  def test_yaml_role_model_beats_backbone_fan_out(self, mock_run_prec_check):
    cfg = self._run_verifier_for_config(
        mock_run_prec_check,
        "role_override",
        yaml_text=(
            "rating_model: GEMINI_FLASH_IMAGE\nsummarizer_model: custom\n"
        ),
        target_model="gemini-3.8-flash",
        backbone_model="gemini-3.8-flash",
    )
    # YAML names are normalized like flag values.
    self.assertEqual(cfg.rating_model, "gemini-3.1-flash-image")
    self.assertEqual(cfg.summarizer_model, "custom")
    self.assertEqual(cfg.agent_model, "gemini-3.8-flash")
    self.assertEqual(cfg.parser_model, "gemini-3.8-flash")

  @mock.patch.object(
      standalone_precision_verifier.precision_estimation,
      "run_precision_check",
  )
  def test_alpha_and_num_queries_flags(self, mock_run_prec_check):
    cfg = self._run_verifier_for_config(
        mock_run_prec_check, "alpha", alpha=0.2, num_queries=40
    )
    # `--alpha` is the single significance level.
    self.assertEqual(cfg.alpha, 0.2)
    self.assertEqual(cfg.single_batch_alpha, 0.2)
    self.assertEqual(cfg.single_batch_num_queries, 40)
    # Concurrency is not a flag; the dataclass defaults apply.
    self.assertEqual(cfg.max_concurrent_hypotheses, 1)
    self.assertEqual(cfg.scrape_bandwidth, 4)

  @mock.patch.object(
      standalone_precision_verifier.precision_estimation,
      "run_precision_check",
  )
  def test_yaml_only_fields(self, mock_run_prec_check):
    cfg = self._run_verifier_for_config(
        mock_run_prec_check,
        "yaml_only",
        yaml_text=(
            "paired: true\n"
            "total_verification_budget: 600\n"
            "scrape_bandwidth: 20\n"
            "random_seed: 7\n"
            "procedural_autorater: count_words\n"
        ),
    )
    self.assertTrue(cfg.paired)
    self.assertTrue(cfg.is_paired)
    self.assertEqual(cfg.total_verification_budget, 600)
    self.assertEqual(cfg.scrape_bandwidth, 20)
    self.assertEqual(cfg.random_seed, 7)
    self.assertEqual(cfg.procedural_autorater, "count_words")
    # A procedural autorater withholds the agent's LLM-judge analysis tool.
    self.assertTrue(cfg.exclude_analyze)

  @mock.patch.object(
      standalone_precision_verifier.precision_estimation,
      "run_precision_check",
  )
  def test_autorater_from_run_metadata_excludes_analyze(
      self, mock_run_prec_check
  ):
    hyp_path = os.path.join(self.test_dir, "hypotheses_metadata.json")
    with open(hyp_path, "w") as f:
      json.dump(
          {
              "config": {"procedural_autorater": "count_hedges"},
              "hypotheses": {"h_1": "Hypothesis 1"},
          },
          f,
      )
    mock_run_prec_check.return_value = precision_estimation.PrecisionResult(
        records=[], precision=1.0, n_verified=0, n_total=0
    )
    with flagsaver.flagsaver(
        hypotheses_path=hyp_path,
        output_dir=os.path.join(self.test_dir, "output_metadata"),
    ):
      standalone_precision_verifier.asyncio.run(
          standalone_precision_verifier.run_verifier()
      )
    cfg = mock_run_prec_check.call_args.kwargs["config"]
    self.assertEqual(cfg.procedural_autorater, "count_hedges")
    self.assertTrue(cfg.exclude_analyze)

  @mock.patch.object(
      standalone_precision_verifier.precision_estimation,
      "run_precision_check",
  )
  def test_unknown_yaml_key_raises(self, mock_run_prec_check):
    with self.assertRaisesRegex(ValueError, "scrape_bandwith"):
      self._run_verifier_for_config(
          mock_run_prec_check, "typo", yaml_text="scrape_bandwith: 20\n"
      )
    mock_run_prec_check.assert_not_called()

  @mock.patch.object(
      standalone_precision_verifier.precision_estimation,
      "run_precision_check",
  )
  def test_removed_flags_are_rejected_as_yaml_keys(self, mock_run_prec_check):
    # Removed flags that never were config fields are unknown keys now.
    for key in ("output_filename", "retry_statuses", "taxonomy_paired"):
      with self.subTest(key):
        with self.assertRaisesRegex(ValueError, key):
          self._run_verifier_for_config(
              mock_run_prec_check, f"removed_{key}", yaml_text=f"{key}: x\n"
          )
    mock_run_prec_check.assert_not_called()

  @mock.patch.object(
      standalone_precision_verifier.precision_estimation,
      "run_precision_check",
  )
  def test_run_verifier_with_verification_failures(self, mock_run_prec_check):
    hyp_path = os.path.join(self.test_dir, "hypotheses_vf.json")
    with open(hyp_path, "w") as f:
      json.dump(
          {
              "h_good": "Model generates correct answers",
              "h_bad": "Model hallucinates",
              "h_blocked": "Model blocked by safety filters",
          },
          f,
      )

    mock_run_prec_check.return_value = precision_estimation.PrecisionResult(
        records=[
            precision_estimation.HypothesisPrecisionRecord(
                hypothesis_id="h_good",
                hypothesis_text="Model generates correct answers",
                verified=True,
                final_e_product=10.0,
                n_subhyps_used=1,
                e_values=[("Sub 1", 10.0)],
                status="OK",
            ),
            precision_estimation.HypothesisPrecisionRecord(
                hypothesis_id="h_bad",
                hypothesis_text="Model hallucinates",
                verified=False,
                final_e_product=0.2,
                n_subhyps_used=5,
                e_values=[("Sub 1", 0.2)],
                status="OK",
            ),
            precision_estimation.HypothesisPrecisionRecord(
                hypothesis_id="h_blocked",
                hypothesis_text="Model blocked by safety filters",
                verified=False,
                final_e_product=1.0,
                n_subhyps_used=0,
                e_values=[],
                status="VERIFICATION_FAILURE",
                block_reasons=["PROHIBITED_CONTENT: weapons"],
            ),
        ],
        precision=0.5,  # 1 verified / 2 evaluated
        n_verified=1,
        n_total=3,
        n_evaluated=2,
        n_statistical_failures=1,
        n_execution_errors=1,
        n_verification_failures=1,
    )

    out_dir = os.path.join(self.test_dir, "output_vf")
    with flagsaver.flagsaver(
        hypotheses_path=hyp_path,
        target_model="gemini-3.8-flash",
        output_dir=out_dir,
        alpha=0.5,
    ):
      standalone_precision_verifier.asyncio.run(
          standalone_precision_verifier.run_verifier()
      )

    out_path = os.path.join(out_dir, "verification_results.json")
    self.assertTrue(os.path.exists(out_path))

    with open(out_path, "r") as f:
      res = json.load(f)

    self.assertEqual(res["precision"], 0.5)
    self.assertEqual(res["n_verified"], 1)
    self.assertEqual(res["n_total"], 3)
    self.assertEqual(res["n_evaluated"], 2)
    self.assertEqual(res["n_statistical_failures"], 1)
    self.assertEqual(res["n_execution_errors"], 1)
    self.assertEqual(res["n_verification_failures"], 1)
    self.assertEqual(res["empirical_fdr"], 0.5)
    self.assertLen(res["records"], 3)
    self.assertEqual(res["records"][2]["status"], "VERIFICATION_FAILURE")
    self.assertEqual(
        res["records"][2]["block_reasons"],
        ["PROHIBITED_CONTENT: weapons"],
    )

  @mock.patch.object(
      standalone_precision_verifier.precision_estimation,
      "run_precision_check",
  )
  def test_run_verifier_with_yaml_paired(self, mock_run_prec_check):
    hyp_path = os.path.join(self.test_dir, "hypotheses_paired.json")
    config_path = os.path.join(self.test_dir, "paired.yaml")
    with open(config_path, "w") as f:
      f.write("paired: true\n")
    with open(hyp_path, "w") as f:
      json.dump({"h_paired": "English recipe is shorter than French"}, f)

    mock_run_prec_check.return_value = precision_estimation.PrecisionResult(
        records=[
            precision_estimation.HypothesisPrecisionRecord(
                hypothesis_id="h_paired",
                hypothesis_text="English recipe is shorter than French",
                verified=True,
                final_e_product=25.0,
                n_subhyps_used=1,
                e_values=[("Sub 1", 25.0)],
                status="OK",
            ),
        ],
        precision=1.0,
        n_verified=1,
        n_total=1,
    )

    out_dir = os.path.join(self.test_dir, "output_paired")
    with flagsaver.flagsaver(
        hypotheses_path=hyp_path,
        target_model="gemini-3.8-flash",
        output_dir=out_dir,
        config=config_path,
    ):
      standalone_precision_verifier.asyncio.run(
          standalone_precision_verifier.run_verifier()
      )

    mock_run_prec_check.assert_called_once()
    passed_config = mock_run_prec_check.call_args.kwargs["config"]
    self.assertTrue(passed_config.paired)
    self.assertTrue(passed_config.is_paired)

  @mock.patch.object(
      standalone_precision_verifier.precision_estimation,
      "run_precision_check",
  )
  def test_run_verifier_single_batch_mode(self, mock_run_prec_check):
    hyp_path = os.path.join(self.test_dir, "hypotheses_single_batch.json")
    with open(hyp_path, "w") as f:
      json.dump(
          {"h_single": "Model refuses medication queries > 10% of the time"}, f
      )

    mock_run_prec_check.return_value = precision_estimation.PrecisionResult(
        records=[
            precision_estimation.HypothesisPrecisionRecord(
                hypothesis_id="h_single",
                hypothesis_text=(
                    "Model refuses medication queries > 10% of the time"
                ),
                verified=True,
                p_value=0.001,
                sample_size=100,
                observed_successes=25,
                observed_rate=0.25,
                null_rate=0.10,
                ci_lower=0.17,
                ci_upper=0.34,
                status="OK",
            ),
        ],
        precision=1.0,
        n_verified=1,
        n_total=1,
    )

    out_dir = os.path.join(self.test_dir, "output_single_batch")
    with flagsaver.flagsaver(
        hypotheses_path=hyp_path,
        target_model="gemini-3.8-flash",
        output_dir=out_dir,
        num_queries=100,
        alpha=0.05,
    ):
      standalone_precision_verifier.asyncio.run(
          standalone_precision_verifier.run_verifier()
      )

    mock_run_prec_check.assert_called_once()
    passed_config = mock_run_prec_check.call_args.kwargs["config"]
    self.assertEqual(passed_config.verification_mode, "single_batch_p_value")
    self.assertEqual(passed_config.single_batch_num_queries, 100)
    self.assertEqual(passed_config.single_batch_alpha, 0.05)
    self.assertEqual(passed_config.scrape_bandwidth, 4)

    out_path = os.path.join(out_dir, "verification_results.json")
    self.assertTrue(os.path.exists(out_path))
    with open(out_path, "r") as f:
      res = json.load(f)

    self.assertEqual(res["precision"], 1.0)
    self.assertEqual(res["empirical_fdr"], 0.0)
    self.assertEqual(res["records"][0]["p_value"], 0.001)
    self.assertEqual(res["records"][0]["sample_size"], 100)
    self.assertEqual(res["records"][0]["observed_rate"], 0.25)
    self.assertEqual(res["records"][0]["null_rate"], 0.10)

  def test_resolve_task_autorater(self):
    # 1. Explicit CLI flag has highest precedence
    self.assertEqual(
        standalone_precision_verifier.resolve_task_autorater(
            {"config": {"procedural_autorater": "count_hedges"}},
            cli_flag_val="count_words",
        ),
        "count_words",
    )

    # 2. JSON config block
    self.assertEqual(
        standalone_precision_verifier.resolve_task_autorater(
            {"config": {"procedural_autorater": "count_words"}},
            cli_flag_val="",
        ),
        "count_words",
    )

    # 3. Top-level procedural_autorater
    self.assertEqual(
        standalone_precision_verifier.resolve_task_autorater(
            {"procedural_autorater": "count_hedges"},
            cli_flag_val="",
        ),
        "count_hedges",
    )

    # 4. Auto-detect from task_name matching tasks/registry.py (top-level)
    self.assertEqual(
        standalone_precision_verifier.resolve_task_autorater(
            {"task_name": "recipe_translate_french_long"},
            cli_flag_val="",
        ),
        "count_words",
    )

    # 5. Auto-detect from nested config task_name / task_suite
    self.assertEqual(
        standalone_precision_verifier.resolve_task_autorater(
            {"config": {"task_name": "Recipe Translate French Long"}},
            cli_flag_val="",
        ),
        "count_words",
    )
    self.assertEqual(
        standalone_precision_verifier.resolve_task_autorater(
            {"config": {"task_suite": "recipes"}},
            cli_flag_val="",
        ),
        "count_words",
    )

  def test_resolve_seed_instruction(self):
    # 1. Explicit CLI flag has highest precedence
    self.assertEqual(
        standalone_precision_verifier.resolve_seed_instruction(
            {"config": {"seed_instruction": "From config"}},
            cli_flag_val="From CLI",
        ),
        "From CLI",
    )

    # 2. JSON config block
    self.assertEqual(
        standalone_precision_verifier.resolve_seed_instruction(
            {"config": {"seed_instruction": "From config"}},
            cli_flag_val="",
        ),
        "From config",
    )

    # 3. JSON disco_seed fallback
    self.assertEqual(
        standalone_precision_verifier.resolve_seed_instruction(
            {"config": {"disco_seed": "From disco_seed"}},
            cli_flag_val="",
        ),
        "From disco_seed",
    )

    # 4. Auto-detect from task_name in task registry (top-level & nested)
    self.assertIn(
        "recipe",
        standalone_precision_verifier.resolve_seed_instruction(
            {"task_name": "recipe_translate_french_long"},
            cli_flag_val="",
        ).lower(),
    )
    self.assertIn(
        "recipe",
        standalone_precision_verifier.resolve_seed_instruction(
            {"config": {"task_name": "recipe_translate_french_long"}},
            cli_flag_val="",
        ).lower(),
    )

    # 5. Default fallback
    self.assertEqual(
        standalone_precision_verifier.resolve_seed_instruction(
            {"task_name": "unregistered_task_foo"},
            cli_flag_val="",
        ),
        "Verify the provided hypotheses",
    )

  @mock.patch.object(
      standalone_agent,
      "_get_or_create_candidate",
      return_value=mock.MagicMock(),
  )
  @mock.patch.object(
      standalone_agent.genai_client,
      "get_genai_client",
      return_value=mock.MagicMock(),
  )
  def test_standalone_agent_tool_declarations_exclude_analyze(
      self, mock_client, mock_candidate
  ):
    del mock_client, mock_candidate
    # When exclude_analyze is False, both analyze_responses and
    # run_procedural_autorater are available
    agent_both = standalone_agent.StandaloneAgent(
        agent_model="dummy",
        target_model="dummy",
        procedural_autorater="count_words",
        exclude_analyze=False,
    )
    tool_names_both = [t.name for t in agent_both._tool_declarations]
    self.assertIn("analyze_responses", tool_names_both)
    self.assertIn("run_procedural_autorater", tool_names_both)

    # When exclude_analyze is True, analyze_responses is excluded
    agent_excluded = standalone_agent.StandaloneAgent(
        agent_model="dummy",
        target_model="dummy",
        procedural_autorater="count_words",
        exclude_analyze=True,
    )
    tool_names_excluded = [t.name for t in agent_excluded._tool_declarations]
    self.assertNotIn("analyze_responses", tool_names_excluded)

  @mock.patch.object(
      standalone_agent,
      "_get_or_create_candidate",
      return_value=mock.MagicMock(),
  )
  @mock.patch.object(standalone_agent.genai_client, "get_genai_client")
  def test_standalone_agent_send_enable_tools_flag(
      self, mock_get_client, mock_candidate
  ):
    del mock_candidate
    mock_client = mock.MagicMock()
    mock_get_client.return_value = mock_client
    mock_candidate_resp = mock.MagicMock()
    mock_candidate_resp.candidates = [
        mock.MagicMock(
            content=mock.MagicMock(parts=[types.Part(text="Hello text only")])
        )
    ]
    mock_client.aio.models.generate_content = mock.AsyncMock(
        return_value=mock_candidate_resp
    )

    agent = standalone_agent.StandaloneAgent(
        agent_model="dummy",
        target_model="dummy",
        procedural_autorater="count_words",
    )

    # 1. When enable_tools=False, tools passed to generate_content config is
    # None
    res = asyncio.run(agent.send("Intro prompt", enable_tools=False))
    self.assertEqual(res, "Hello text only")
    call_config = mock_client.aio.models.generate_content.call_args[1]["config"]
    self.assertIsNone(call_config.tools)

    # 2. When enable_tools=True, tools passed to generate_content config is
    # populated
    res2 = asyncio.run(agent.send("Scrape prompt", enable_tools=True))
    self.assertEqual(res2, "Hello text only")
    call_config2 = mock_client.aio.models.generate_content.call_args[1][
        "config"
    ]
    self.assertIsNotNone(call_config2.tools)
    self.assertNotEmpty(call_config2.tools)

  @mock.patch.object(
      precision_estimation,
      "_check_one_hypothesis_single_batch",
  )
  def test_single_batch_precision_calculation_excludes_errors(
      self, mock_check_single
  ):
    async def _mock_check(h_id, *args, **kwargs):
      del args, kwargs
      if h_id == "h_verified":
        return precision_estimation.HypothesisPrecisionRecord(
            hypothesis_id=h_id,
            hypothesis_text="Verified hypothesis",
            verified=True,
            status="OK",
        )
      elif h_id == "h_stat_fail":
        return precision_estimation.HypothesisPrecisionRecord(
            hypothesis_id=h_id,
            hypothesis_text="Statistical failure hypothesis",
            verified=False,
            status="OK",
        )
      else:
        return precision_estimation.HypothesisPrecisionRecord(
            hypothesis_id=h_id,
            hypothesis_text="Scrape failed hypothesis",
            verified=False,
            status="SCRAPE_FAILED",
            block_reasons=["SCRAPE_FAILED: No valid responses returned"],
        )

    mock_check_single.side_effect = _mock_check

    verified_hyps = [
        (
            "h_verified",
            records_lib.HypothesisRecord(text="Verified hypothesis"),
        ),
        (
            "h_stat_fail",
            records_lib.HypothesisRecord(text="Statistical failure hypothesis"),
        ),
        (
            "h_scrape_fail",
            records_lib.HypothesisRecord(text="Scrape failed hypothesis"),
        ),
    ]
    cfg = precision_estimation.verifier_config.VerifierConfig(
        target_model="gemini-3.8-flash"
    )
    result = asyncio.run(
        precision_estimation.run_single_batch_precision_check(
            verified_hypotheses=verified_hyps,
            config=cfg,
            log_fn=lambda *a, **k: None,
        )
    )

    self.assertEqual(result.n_total, 3)
    self.assertEqual(result.n_evaluated, 2)
    self.assertEqual(result.n_verified, 1)
    self.assertEqual(result.n_statistical_failures, 1)
    self.assertEqual(result.n_execution_errors, 1)
    self.assertEqual(result.n_verification_failures, 1)
    assert result.precision is not None
    self.assertAlmostEqual(result.precision, 0.5)

  @mock.patch.object(
      precision_estimation,
      "_generate_queries_isolated",
      return_value=["Query 1", "Query 2"],
  )
  @mock.patch.object(asyncio, "sleep", return_value=None)
  @mock.patch.object(standalone_agent, "StandaloneAgent")
  @mock.patch.object(precision_estimation.agent_senders, "StandaloneSender")
  def test_check_one_hypothesis_single_batch_reprompt_retry_success(
      self, mock_sender_cls, mock_agent_cls, mock_sleep, mock_gen_queries
  ):
    del mock_gen_queries
    mock_agent = mock.MagicMock()
    mock_agent_cls.return_value = mock_agent

    mock_sender = mock.MagicMock()
    mock_sender.send_and_extract = mock.AsyncMock()
    mock_sender.send_step = mock.AsyncMock()
    mock_sender_cls.return_value = mock_sender

    plan = stats_plan.PreRegisteredTestPlan(
        test_type="one_sample_proportion_z_test",
        target_metric="rate",
        null_baseline=0.10,
        alternative="larger",
    )
    mock_sender.send_and_extract.side_effect = [
        plan,  # pre-registration plan
        stats_plan.StatsResult(
            test_type="one_sample_proportion_z_test",
            p_value=0.001,
            group_a_obs=10,
            group_a_mean=0.4,
            null_proportion=0.1,
        ),  # stats
    ]

    mock_resp = mock.MagicMock()
    mock_agent.last_scrape_status = ""
    mock_agent.last_responses = []

    scrape_calls = 0

    async def mock_send_step(prompt, *args, **kwargs):
      del prompt, args, kwargs
      nonlocal scrape_calls
      scrape_calls += 1
      if scrape_calls == 1:
        # Intro prompt is first send_step call
        return "Intro acknowledged."
      elif scrape_calls == 2:
        # Context prompt is second send_step call
        return "Context acknowledged."
      elif scrape_calls == 3:
        # First scrape attempt: agent returns plain text without calling tool
        mock_agent.last_scrape_status = ""
        mock_agent.last_responses = []
        return "I will analyze these queries..."
      else:
        # Second scrape attempt: agent calls tool
        mock_agent.last_scrape_status = "OK"
        mock_agent.last_responses = [mock_resp]
        mock_agent.all_responses = [mock_resp]
        return "Tool executed successfully."

    mock_sender.send_step.side_effect = mock_send_step

    cfg = precision_estimation.verifier_config.VerifierConfig(
        target_model="gemini-3.8-flash",
        # One response stands in for the whole batch.
        min_response_fraction=0.0,
    )
    record = asyncio.run(
        precision_estimation._check_one_hypothesis_single_batch(
            h_id="h_test",
            h_text="Test hypothesis text",
            config=cfg,
            log_fn=lambda *a, **k: None,
        )
    )

    self.assertEqual(scrape_calls, 4)
    self.assertEqual(record.status, "OK")
    self.assertTrue(record.verified)
    self.assertEqual(record.test_plan, plan)
    mock_sleep.assert_called_once()
    retry_prompt_call = mock_sender.send_step.call_args_list[3][0][0]
    self.assertIn(
        "CRITICAL ERROR: You did not execute the"
        " 'generate_single_turn_responses' tool",
        retry_prompt_call,
    )

  @mock.patch.object(
      precision_estimation,
      "_generate_queries_isolated",
      return_value=["Query 1", "Query 2"],
  )
  @mock.patch.object(asyncio, "sleep", return_value=None)
  @mock.patch.object(standalone_agent, "StandaloneAgent")
  @mock.patch.object(precision_estimation.agent_senders, "StandaloneSender")
  def test_check_one_hypothesis_single_batch_reuses_stored_plan(
      self, mock_sender_cls, mock_agent_cls, mock_sleep, mock_gen_queries
  ):
    del mock_sleep, mock_gen_queries
    mock_agent = mock.MagicMock()
    mock_agent_cls.return_value = mock_agent
    mock_resp = mock.MagicMock()
    mock_agent.last_scrape_status = "OK"
    mock_agent.last_responses = [mock_resp]
    mock_agent.all_responses = [mock_resp]

    mock_sender = mock.MagicMock()
    mock_sender.send_step = mock.AsyncMock(return_value="OK")
    # Only the stats step extracts; there is no pre-registration turn.
    mock_sender.send_and_extract = mock.AsyncMock(
        side_effect=[
            stats_plan.StatsResult(
                test_type="one_sample_proportion_z_test",
                p_value=0.001,
                group_a_obs=10,
                group_a_mean=0.4,
                null_proportion=0.1,
            )
        ]
    )
    mock_sender_cls.return_value = mock_sender

    stored = stats_plan.PreRegisteredTestPlan(
        test_type="one_sample_proportion_z_test",
        target_metric="rate",
        null_baseline=0.10,
        alternative="larger",
    )
    logs = []
    record = asyncio.run(
        precision_estimation._check_one_hypothesis_single_batch(
            h_id="h_test",
            h_text="Test hypothesis text",
            config=precision_estimation.verifier_config.VerifierConfig(
                target_model="gemini-3.8-flash", min_response_fraction=0.0
            ),
            log_fn=lambda msg, *a, **k: logs.append(msg),
            stored_plan=stored,
        )
    )

    self.assertEqual(mock_sender.send_and_extract.await_count, 1)
    self.assertTrue(record.verified)
    self.assertIs(record.test_plan, stored)
    self.assertTrue(any("reusing the plan pre-registered" in m for m in logs))

  @mock.patch.object(
      precision_estimation,
      "_generate_queries_isolated",
      return_value=["Query 1", "Query 2"],
  )
  @mock.patch.object(asyncio, "sleep", return_value=None)
  @mock.patch.object(standalone_agent, "StandaloneAgent")
  @mock.patch.object(precision_estimation.agent_senders, "StandaloneSender")
  def test_check_one_hypothesis_single_batch_scrape_failed_after_retries(
      self, mock_sender_cls, mock_agent_cls, mock_sleep, mock_gen_queries
  ):
    del mock_gen_queries
    mock_agent = mock.MagicMock()
    mock_agent_cls.return_value = mock_agent

    mock_sender = mock.MagicMock()
    mock_sender.send_and_extract = mock.AsyncMock()
    mock_sender.send_step = mock.AsyncMock()
    mock_sender_cls.return_value = mock_sender

    plan = stats_plan.PreRegisteredTestPlan(
        test_type="one_sample_proportion_z_test",
        target_metric="rate",
        null_baseline=0.10,
        alternative="larger",
    )
    mock_sender.send_and_extract.side_effect = [
        plan,  # pre-registration plan
    ]

    mock_agent.last_scrape_status = ""
    mock_agent.last_responses = []
    mock_agent.last_block_reason = ""
    mock_sender.send_step.return_value = "I did not call any tool"

    cfg = precision_estimation.verifier_config.VerifierConfig(
        target_model="gemini-3.8-flash"
    )
    record = asyncio.run(
        precision_estimation._check_one_hypothesis_single_batch(
            h_id="h_test",
            h_text="Test hypothesis text",
            config=cfg,
            log_fn=lambda *a, **k: None,
        )
    )

    # 1 intro call + 1 context call + 3 scrape retry calls = 5 send_step calls
    self.assertEqual(mock_sender.send_step.call_count, 5)
    self.assertEqual(mock_sleep.call_count, 2)
    self.assertEqual(record.status, "SCRAPE_FAILED")
    self.assertFalse(record.verified)

  def test_run_single_batch_precision_check_concurrency_throttled(self):
    active_tasks = 0
    max_active_tasks = 0

    async def mock_check_one_sb(
        h_id, text, config, log_fn, stored_plan=None, q_count=None
    ):
      del text, config, log_fn, stored_plan, q_count
      nonlocal active_tasks, max_active_tasks
      active_tasks += 1
      if active_tasks > max_active_tasks:
        max_active_tasks = active_tasks
      await asyncio.sleep(0.01)
      active_tasks -= 1
      return precision_estimation.HypothesisPrecisionRecord(
          hypothesis_text="test",
          verified=True,
          hypothesis_id=h_id,
      )

    with mock.patch.object(
        precision_estimation,
        "_check_one_hypothesis_single_batch",
        side_effect=mock_check_one_sb,
    ):
      cfg = precision_estimation.verifier_config.VerifierConfig(
          target_model="gemini-3.8-flash",
          verification_mode="single_batch_p_value",
          max_concurrent_hypotheses=5,
      )
      hyps = [
          (f"h_{i}", records_lib.HypothesisRecord(text=f"text_{i}"))
          for i in range(12)
      ]
      res = asyncio.run(
          precision_estimation.run_single_batch_precision_check(
              verified_hypotheses=hyps,
              config=cfg,
              log_fn=lambda *a, **k: None,
          )
      )
      self.assertEqual(res.n_verified, 12)
      self.assertEqual(res.n_total, 12)
      self.assertLessEqual(max_active_tasks, 5)

  def test_allocate_verification_budget_weights_paired_double(self):
    # The user-facing case: two paired and two unpaired hypotheses share 600
    # queries. Paired ones cost two queries per observation, so weight 2 buys
    # them the same 200 observations the unpaired ones get for 100 queries.
    self.assertEqual(
        precision_estimation.allocate_verification_budget([2, 2, 1, 1], 600),
        [200, 200, 100, 100],
    )

  def test_allocate_verification_budget_spends_exactly_the_budget(self):
    # 100 does not divide evenly by a weight total of 3; largest-remainder
    # hands out the leftovers instead of letting them evaporate.
    allocation = precision_estimation.allocate_verification_budget([2, 1], 100)
    self.assertEqual(sum(allocation), 100)
    self.assertEqual(allocation, [67, 33])

    allocation = precision_estimation.allocate_verification_budget(
        [1, 1, 1], 100
    )
    self.assertEqual(sum(allocation), 100)
    self.assertCountEqual(allocation, [34, 33, 33])

  def test_allocate_verification_budget_floors_at_one_query(self):
    # A zero allocation would surface as an execution error rather than a
    # measurement, so every hypothesis keeps at least one query even when that
    # overruns a tiny budget.
    allocation = precision_estimation.allocate_verification_budget([1] * 5, 2)
    self.assertEqual(allocation, [1, 1, 1, 1, 1])

  def test_allocate_verification_budget_empty(self):
    self.assertEqual(
        precision_estimation.allocate_verification_budget([], 10), []
    )

  def _make_setup(self, h_id, is_paired, base_q_count=100):
    return precision_estimation._SingleBatchSetup(
        h_id=h_id,
        h_text=f"text_{h_id}",
        agent=None,
        sender=None,
        parsers=None,
        prompts_mod=None,
        output_dir="",
        log_fn=lambda *a, **k: None,
        test_plan=None,
        is_paired=is_paired,
        base_q_count=base_q_count,
        null_rate=0.1,
        alpha=0.05,
        block_reasons=[],
    )

  def _run_with_mocked_phases(
      self,
      cfg,
      n_hyps,
      paired_ids,
      stored_plans=None,
      observed_plans=None,
      retry_budget=None,
  ):
    """Runs a single-batch check, recording the queries each hypothesis got."""
    observed = {}
    if observed_plans is None:
      observed_plans = {}

    async def mock_setup(h_id, h_text, config, log_fn, stored_plan=None):
      del h_text, config, log_fn
      observed_plans[h_id] = stored_plan
      return self._make_setup(h_id, is_paired=h_id in paired_ids)

    async def mock_execute(setup, q_count, config):
      del config
      observed[setup.h_id] = q_count
      return precision_estimation.HypothesisPrecisionRecord(
          hypothesis_text="test",
          verified=True,
          hypothesis_id=setup.h_id,
      )

    hyps = [
        (
            f"h_{i}",
            records_lib.HypothesisRecord(text=f"text_{i}"),
        )
        for i in range(n_hyps)
    ]
    with (
        mock.patch.object(
            precision_estimation,
            "_setup_and_preregister_single_batch",
            side_effect=mock_setup,
        ),
        mock.patch.object(
            precision_estimation,
            "_execute_single_batch",
            side_effect=mock_execute,
        ),
    ):
      asyncio.run(
          precision_estimation.run_single_batch_precision_check(
              verified_hypotheses=hyps,
              config=cfg,
              log_fn=lambda *a, **k: None,
              stored_plans=stored_plans,
              retry_budget=retry_budget,
          )
      )
    return observed

  def test_run_single_batch_precision_check_budget_favors_paired(self):
    cfg = precision_estimation.verifier_config.VerifierConfig(
        target_model="gemini-3.8-flash",
        verification_mode="single_batch_p_value",
        single_batch_num_queries=100,
        total_verification_budget=600,
    )
    observed = self._run_with_mocked_phases(
        cfg, n_hyps=4, paired_ids={"h_0", "h_1"}
    )
    self.assertEqual(observed, {"h_0": 200, "h_1": 200, "h_2": 100, "h_3": 100})

  def test_run_single_batch_precision_check_budget_all_unpaired(self):
    cfg = precision_estimation.verifier_config.VerifierConfig(
        target_model="gemini-3.8-flash",
        verification_mode="single_batch_p_value",
        single_batch_num_queries=100,
        total_verification_budget=100,
    )
    observed = self._run_with_mocked_phases(cfg, n_hyps=4, paired_ids=set())
    self.assertEqual(observed, {"h_0": 25, "h_1": 25, "h_2": 25, "h_3": 25})

  def test_run_single_batch_precision_check_no_budget_doubles_paired(self):
    # With no total budget the flat behaviour stands: every hypothesis gets
    # single_batch_num_queries, doubled if it is paired.
    cfg = precision_estimation.verifier_config.VerifierConfig(
        target_model="gemini-3.8-flash",
        verification_mode="single_batch_p_value",
        single_batch_num_queries=100,
        total_verification_budget=0,
    )
    observed = self._run_with_mocked_phases(cfg, n_hyps=2, paired_ids={"h_0"})
    self.assertEqual(observed, {"h_0": 200, "h_1": 100})

  def test_run_single_batch_precision_check_passes_stored_plans(self):
    cfg = precision_estimation.verifier_config.VerifierConfig(
        target_model="gemini-3.8-flash",
        verification_mode="single_batch_p_value",
        single_batch_num_queries=100,
        total_verification_budget=200,
    )
    plan = stats_plan.PreRegisteredTestPlan(
        test_type="one_sample_proportion_z_test", target_metric="rate"
    )
    observed_plans = {}
    self._run_with_mocked_phases(
        cfg,
        n_hyps=2,
        paired_ids=set(),
        stored_plans={"h_1": plan},
        observed_plans=observed_plans,
    )
    self.assertEqual(observed_plans, {"h_0": None, "h_1": plan})

  def test_single_batch_retry_keeps_original_allocations(self):
    cfg = precision_estimation.verifier_config.VerifierConfig(
        target_model="gemini-3.8-flash",
        verification_mode="single_batch_p_value",
        single_batch_num_queries=100,
        total_verification_budget=600,
    )
    # Retrying 2 of 4 hypotheses: h_0 keeps its recorded 37 queries; h_1 has
    # no record, so it gets the original share (600 // 4) at weight 2.
    observed = self._run_with_mocked_phases(
        cfg,
        n_hyps=2,
        paired_ids={"h_1"},
        retry_budget=precision_estimation.RetryBudget(
            allocated_queries={"h_0": 37}, n_original=4
        ),
    )
    self.assertEqual(observed, {"h_0": 37, "h_1": 300})

  def test_flat_single_batch_retry_uses_recorded_query_count(self):
    observed = {}

    async def mock_check(
        h_id, text, config, log_fn, stored_plan=None, q_count=None
    ):
      del text, config, log_fn, stored_plan
      observed[h_id] = q_count
      return precision_estimation.HypothesisPrecisionRecord(
          hypothesis_text="t", verified=True, hypothesis_id=h_id
      )

    with mock.patch.object(
        precision_estimation,
        "_check_one_hypothesis_single_batch",
        side_effect=mock_check,
    ):
      asyncio.run(
          precision_estimation.run_single_batch_precision_check(
              verified_hypotheses=[
                  (f"h_{i}", records_lib.HypothesisRecord(text=f"t{i}"))
                  for i in range(2)
              ],
              config=precision_estimation.verifier_config.VerifierConfig(
                  target_model="gemini-3.8-flash",
                  verification_mode="single_batch_p_value",
              ),
              log_fn=lambda *a, **k: None,
              retry_budget=precision_estimation.RetryBudget(
                  allocated_queries={"h_0": 64}, n_original=5
              ),
          )
      )
    # None means the flat per-hypothesis size, as without a retry.
    self.assertEqual(observed, {"h_0": 64, "h_1": None})

  def _create_paired_execution_setup(self, test_plan, stats_result, n_pairs=40):
    mock_agent = mock.MagicMock()
    mock_agent.last_scrape_status = "OK"
    mock_resp = mock.MagicMock()
    mock_agent.last_responses = [mock_resp] * (2 * n_pairs)
    mock_agent.all_responses = [mock_resp] * (2 * n_pairs)
    mock_agent.last_response_indices = list(range(2 * n_pairs))
    mock_agent.last_block_reason = None

    mock_sender = mock.MagicMock()
    mock_sender.send_step = mock.AsyncMock(return_value="OK")
    mock_sender.send_and_extract = mock.AsyncMock(return_value=stats_result)

    mock_prompts_mod = mock.MagicMock()
    mock_prompts_mod.get_scrape_and_judge_execution_prompt.return_value = (
        "scrape"
    )
    mock_prompts_mod.get_stats_significance_prompt.return_value = "stats"

    self._patch_isolated_queries()
    cfg = precision_estimation.verifier_config.VerifierConfig(
        target_model="gemini-3.8-flash",
    )
    setup = precision_estimation._SingleBatchSetup(
        h_id="h_paired_test",
        h_text="Paired test hypothesis",
        agent=mock_agent,
        sender=mock_sender,
        parsers=None,
        prompts_mod=mock_prompts_mod,
        output_dir="",
        log_fn=lambda *a, **k: None,
        test_plan=test_plan,
        is_paired=True,
        base_q_count=2 * n_pairs,
        null_rate=test_plan.null_baseline if test_plan else 0.5,
        alpha=0.05,
        block_reasons=[],
    )
    return setup, cfg

  def test_single_batch_paired_drop_ties_valid(self):
    # 40 pairs collected, 22 ties dropped, agent reports 18 discordant pairs
    # (all successes).
    plan = stats_plan.PreRegisteredTestPlan(
        test_type="Binomial test",
        target_metric="rate",
        null_baseline=0.50,
        alternative="larger",
        is_paired=True,
    )
    stats_res = stats_plan.StatsResult(
        test_type="Binomial test",
        p_value=0.00001,
        group_a_obs=18,
        group_a_mean=1.0,
        null_proportion=0.50,
    )
    setup, cfg = self._create_paired_execution_setup(
        plan, stats_res, n_pairs=40
    )
    record = asyncio.run(
        precision_estimation._execute_single_batch(
            setup=setup, q_count=80, config=cfg
        )
    )
    self.assertEqual(record.status, "OK")
    self.assertTrue(record.verified)
    self.assertEqual(record.sample_size, 18)
    self.assertEqual(record.observed_successes, 18)
    self.assertIsNotNone(record.paired_diagnostics)
    self.assertEqual(record.paired_diagnostics.n_pairs, 40)
    self.assertEqual(record.paired_diagnostics.n_tied, 22)

  def _paired_plan_and_stats(self):
    plan = stats_plan.PreRegisteredTestPlan(
        test_type="Binomial test",
        target_metric="rate",
        null_baseline=0.50,
        alternative="larger",
        is_paired=True,
    )
    stats_res = stats_plan.StatsResult(
        test_type="Binomial test",
        p_value=0.00001,
        group_a_obs=18,
        group_a_mean=1.0,
        null_proportion=0.50,
    )
    return plan, stats_res

  def test_single_batch_paired_min_response_fraction(self):
    plan, stats_res = self._paired_plan_and_stats()
    for name, q_count, min_fraction, ok in (
        # 40 complete pairs of the 50 requested: 0.8 < 0.9.
        ("short", 100, 0.9, False),
        # 40 of 44: 0.91 >= 0.9.
        ("enough", 88, 0.9, True),
        ("disabled", 100, 0.0, True),
    ):
      with self.subTest(name):
        setup, cfg = self._create_paired_execution_setup(
            plan, stats_res, n_pairs=40
        )
        cfg.min_response_fraction = min_fraction
        record = asyncio.run(
            precision_estimation._execute_single_batch(
                setup=setup, q_count=q_count, config=cfg
            )
        )
        if ok:
          self.assertEqual(record.status, "OK")
          self.assertTrue(record.verified)
          continue
        self.assertEqual(record.status, "INSUFFICIENT_RESPONSES")
        self.assertFalse(record.verified)
        self.assertEqual(record.sample_size, 0)
        self.assertEqual(
            record.failure_reason,
            "INSUFFICIENT_RESPONSES: 40/50 complete pairs received (< 0.9"
            " required)",
        )
        # Short batches never reach the stats step.
        setup.sender.send_and_extract.assert_not_called()

  def test_single_batch_unpaired_min_response_fraction_counts_responses(self):
    plan, stats_res = self._paired_plan_and_stats()
    setup, cfg = self._create_paired_execution_setup(
        plan, stats_res, n_pairs=40
    )
    setup = dataclasses.replace(
        setup,
        is_paired=False,
        test_plan=dataclasses.replace(plan, is_paired=False),
    )
    # 80 responses of the 100 queries sent.
    record = asyncio.run(
        precision_estimation._execute_single_batch(
            setup=setup, q_count=100, config=cfg
        )
    )
    self.assertEqual(record.status, "INSUFFICIENT_RESPONSES")
    self.assertEqual(
        record.failure_reason,
        "INSUFFICIENT_RESPONSES: 80/100 responses received (< 0.9 required)",
    )
    self.assertEqual(record.allocated_queries, 100)

  def _symmetry_single_batch(
      self, h_text, n_last_responses, contrast_outcomes=None, q_count=100
  ):
    agent = mock.MagicMock()
    agent.last_scrape_status = "OK"
    agent.last_block_reason = None
    agent.last_responses = [mock.MagicMock()] * n_last_responses
    agent.all_responses = []
    sender = mock.MagicMock()
    sender.send_step = mock.AsyncMock(return_value="OK")
    sender.send_and_extract = mock.AsyncMock(
        return_value=stats_plan.StatsResult(
            test_type="Two-proportion z-test",
            p_value=0.5,
            group_a_obs=n_last_responses,
            group_a_mean=0.5,
        )
    )
    cfg = precision_estimation.verifier_config.VerifierConfig(
        target_model="gemini-3.8-flash", task_suite="visual_symmetry"
    )
    setup = precision_estimation._SingleBatchSetup(
        h_id="h_sym",
        h_text=h_text,
        agent=agent,
        sender=sender,
        parsers=None,
        prompts_mod=mock.MagicMock(),
        output_dir="",
        log_fn=lambda *a, **k: None,
        test_plan=None,
        is_paired=False,
        base_q_count=q_count,
        null_rate=0.9,
        alpha=0.05,
        block_reasons=[],
        contrast=precision_estimation.visual_symmetry_prompts.infer_contrast(
            h_text
        ),
    )
    with (
        mock.patch.object(
            precision_estimation,
            "_run_symmetry_execution",
            mock.AsyncMock(return_value=None),
        ),
        mock.patch.object(
            precision_estimation,
            "_run_symmetry_contrast_execution",
            mock.AsyncMock(return_value=contrast_outcomes),
        ),
    ):
      record = asyncio.run(
          precision_estimation._execute_single_batch(
              setup=setup, q_count=q_count, config=cfg
          )
      )
    return record, sender

  def test_single_batch_symmetry_counts_scored_completions(self):
    h_text = "complex_chinese succeeds below 90 percent"
    record, sender = self._symmetry_single_batch(h_text, 80)
    self.assertEqual(record.status, "INSUFFICIENT_RESPONSES")
    self.assertEqual(
        record.failure_reason,
        "INSUFFICIENT_RESPONSES: 80/100 responses received (< 0.9 required)",
    )
    sender.send_and_extract.assert_not_called()

    record, sender = self._symmetry_single_batch(h_text, 90)
    self.assertNotEqual(record.status, "INSUFFICIENT_RESPONSES")
    sender.send_and_extract.assert_awaited()

  def test_single_batch_symmetry_contrast_checks_each_arm(self):
    h_text = "complex_chinese fails more often than chinese_char"
    outcome = precision_estimation.ConditionOutcome
    # 180/200 pooled, but arm B returned 80 of its 100.
    record, sender = self._symmetry_single_batch(
        h_text,
        180,
        contrast_outcomes=[
            outcome("complex chinese", "complex_chinese", "white", 100, 40),
            outcome("chinese char", "chinese_char", "white", 80, 70),
        ],
        q_count=200,
    )
    self.assertEqual(record.status, "INSUFFICIENT_RESPONSES")
    self.assertEqual(
        record.failure_reason,
        "INSUFFICIENT_RESPONSES: arm 'chinese char' 80/100 responses received"
        " (< 0.9 required per arm)",
    )
    sender.send_and_extract.assert_not_called()

    record, sender = self._symmetry_single_batch(
        h_text,
        185,
        contrast_outcomes=[
            outcome("complex chinese", "complex_chinese", "white", 95, 40),
            outcome("chinese char", "chinese_char", "white", 90, 70),
        ],
        q_count=200,
    )
    self.assertNotEqual(record.status, "INSUFFICIENT_RESPONSES")
    sender.send_and_extract.assert_awaited()

  def test_single_batch_paired_response_count_fails_as_inconsistent(self):
    # Agent mistakenly reports response count (80) instead of pair count (40).
    plan = stats_plan.PreRegisteredTestPlan(
        test_type="Binomial test",
        target_metric="rate",
        null_baseline=0.50,
        alternative="larger",
        is_paired=True,
    )
    stats_res = stats_plan.StatsResult(
        test_type="Binomial test",
        p_value=0.00001,
        group_a_obs=80,
        group_a_mean=1.0,
        null_proportion=0.50,
    )
    setup, cfg = self._create_paired_execution_setup(
        plan, stats_res, n_pairs=40
    )
    record = asyncio.run(
        precision_estimation._execute_single_batch(
            setup=setup, q_count=80, config=cfg
        )
    )
    self.assertEqual(record.status, "PLAN_INCONSISTENT")
    self.assertFalse(record.verified)
    self.assertEqual(record.p_value, 1.0)

  def test_single_batch_paired_zero_discordant_pairs_fails(self):
    # Under TiePolicy.DROP, 0 discordant pairs cannot evaluate significance.
    plan = stats_plan.PreRegisteredTestPlan(
        test_type="Binomial test",
        target_metric="rate",
        null_baseline=0.50,
        alternative="larger",
        is_paired=True,
    )
    stats_res = stats_plan.StatsResult(
        test_type="Binomial test",
        p_value=1.0,
        group_a_obs=0,
        group_a_mean=0.0,
        null_proportion=0.50,
    )
    setup, cfg = self._create_paired_execution_setup(
        plan, stats_res, n_pairs=40
    )
    record = asyncio.run(
        precision_estimation._execute_single_batch(
            setup=setup, q_count=80, config=cfg
        )
    )
    self.assertEqual(record.status, "PLAN_INCONSISTENT")
    self.assertFalse(record.verified)

  def test_single_batch_paired_count_as_failure_requires_full_denominator(self):
    # Non-symmetry null (0.20) requires keeping tied pairs in the denominator.
    plan = stats_plan.PreRegisteredTestPlan(
        test_type="Binomial test",
        target_metric="rate",
        null_baseline=0.20,
        alternative="larger",
        is_paired=True,
    )
    # Agent attempts to drop ties and reports 18 instead of full 40 pairs.
    stats_res = stats_plan.StatsResult(
        test_type="Binomial test",
        p_value=0.00001,
        group_a_obs=18,
        group_a_mean=1.0,
        null_proportion=0.20,
    )
    setup, cfg = self._create_paired_execution_setup(
        plan, stats_res, n_pairs=40
    )
    record = asyncio.run(
        precision_estimation._execute_single_batch(
            setup=setup, q_count=80, config=cfg
        )
    )
    self.assertEqual(record.status, "PLAN_INCONSISTENT")
    self.assertFalse(record.verified)

  def test_single_batch_paired_continuous_requires_full_denominator(self):
    # Continuous paired test requires retaining zero differences in the mean.
    plan = stats_plan.PreRegisteredTestPlan(
        test_type="Paired continuous test",
        target_metric="difference",
        null_baseline=0.0,
        alternative="larger",
        is_paired=True,
    )
    # Agent drops ties and reports 18 instead of full 40 pairs.
    stats_res = stats_plan.StatsResult(
        test_type="Paired continuous test",
        p_value=0.00001,
        group_a_obs=18,
        group_a_mean=2.5,
        null_proportion=0.0,
    )
    setup, cfg = self._create_paired_execution_setup(
        plan, stats_res, n_pairs=40
    )
    record = asyncio.run(
        precision_estimation._execute_single_batch(
            setup=setup, q_count=80, config=cfg
        )
    )
    self.assertEqual(record.status, "PLAN_INCONSISTENT")
    self.assertFalse(record.verified)

  def test_single_batch_direction_guard_uses_plan_null_for_continuous(self):
    # A paired difference test's null is 0.0 on the metric's scale. The rate
    # null stays at the 0.5 default because setup ignores a 0.0 plan null, so
    # guarding against it would read a positive mean difference of 0.3 as
    # the wrong direction and flip p to 0.999.
    plan = stats_plan.PreRegisteredTestPlan(
        test_type="Paired continuous test",
        target_metric="difference",
        null_baseline=0.0,
        alternative="larger",
        is_paired=True,
    )
    stats_res = stats_plan.StatsResult(
        test_type="Paired continuous test",
        p_value=0.001,
        group_a_obs=40,
        group_a_mean=0.3,
        alternative="larger",
    )
    setup, cfg = self._create_paired_execution_setup(
        plan, stats_res, n_pairs=40
    )
    setup.null_rate = 0.5
    record = asyncio.run(
        precision_estimation._execute_single_batch(
            setup=setup, q_count=80, config=cfg
        )
    )
    self.assertEqual(record.status, "OK")
    self.assertAlmostEqual(record.p_value, 0.001)
    self.assertTrue(record.verified)


class TaskNullProportionTest(absltest.TestCase):
  """Covers sourcing the null/reference rate from task metadata."""

  def test_resolves_from_registry_for_visual_symmetry(self):
    self.assertAlmostEqual(
        standalone_precision_verifier.resolve_task_null_proportion(
            None, "visual_symmetry"
        ),
        0.90,
    )

  def test_json_overrides_registry(self):
    self.assertAlmostEqual(
        standalone_precision_verifier.resolve_task_null_proportion(
            {"null_proportion": 0.6}, "visual_symmetry"
        ),
        0.6,
    )

  def test_reads_from_config_block(self):
    self.assertAlmostEqual(
        standalone_precision_verifier.resolve_task_null_proportion(
            {"config": {"null_proportion": 0.8}}, ""
        ),
        0.8,
    )

  def test_returns_zero_for_unknown_suite(self):
    self.assertEqual(
        standalone_precision_verifier.resolve_task_null_proportion(
            None, "no_such_suite"
        ),
        0.0,
    )

  def test_ignores_out_of_range_and_unparseable_values(self):
    for bad in (
        {"null_proportion": 1.5},
        {"null_proportion": "abc"},
        {"null_proportion": 0.0},
    ):
      self.assertEqual(
          standalone_precision_verifier.resolve_task_null_proportion(bad, ""),
          0.0,
      )

  def test_default_null_rate_prefers_task_metadata(self):
    cfg = precision_estimation.verifier_config.VerifierConfig(
        target_model="test-model",
        task_suite="visual_symmetry",
        task_null_proportion=0.6,
    )
    self.assertAlmostEqual(precision_estimation._default_null_rate(cfg), 0.6)

  def test_default_null_rate_falls_back_to_symmetry_constant(self):
    # task_null_proportion unset (0.0) -> per-modality fallback still applies.
    cfg = precision_estimation.verifier_config.VerifierConfig(
        target_model="test-model",
        task_suite="visual_symmetry",
    )
    self.assertAlmostEqual(
        precision_estimation._default_null_rate(cfg),
        precision_estimation.SYMMETRY_NULL_RATE,
    )

  def test_default_null_rate_generic_task(self):
    cfg = precision_estimation.verifier_config.VerifierConfig(
        target_model="test-model",
        task_suite="recipes",
    )
    self.assertAlmostEqual(
        precision_estimation._default_null_rate(cfg),
        precision_estimation.DEFAULT_NULL_RATE,
    )


class ContrastiveSymmetryTest(absltest.TestCase):
  """Covers the two-condition verification path."""

  def _setup(self, h_text: str, base_q_count: int = 100):
    return precision_estimation._SingleBatchSetup(
        h_id="h1",
        h_text=h_text,
        agent=None,
        sender=None,
        parsers=None,
        prompts_mod=None,
        output_dir="",
        log_fn=lambda *a, **k: None,
        test_plan=None,
        is_paired=False,
        base_q_count=base_q_count,
        null_rate=0.9,
        alpha=0.1,
        block_reasons=[],
        contrast=precision_estimation.visual_symmetry_prompts.infer_contrast(
            h_text
        ),
    )

  def test_split_is_balanced(self):
    self.assertEqual(
        precision_estimation.split_contrast_budget(200), (100, 100)
    )

  def test_split_gives_the_odd_probe_to_arm_a(self):
    # Arm A is the condition the hypothesis is about, so it takes the remainder.
    self.assertEqual(precision_estimation.split_contrast_budget(101), (51, 50))

  def test_split_sums_to_the_budget(self):
    for total in range(1, 50):
      n_a, n_b = precision_estimation.split_contrast_budget(total)
      self.assertEqual(n_a + n_b, total)

  def test_contrastive_hypothesis_gets_double_budget(self):
    setup = self._setup("complex_chinese fails more often than chinese_char")
    self.assertTrue(setup.is_two_sample)
    # 100 per arm rather than 50 per arm: a two-proportion comparison needs
    # roughly twice the observations of a one-sample test for equal power.
    self.assertEqual(setup.standalone_q_count, 200)
    self.assertEqual(setup.budget_weight, 2)

  def test_absolute_hypothesis_budget_is_unchanged(self):
    setup = self._setup("complex_chinese succeeds below 90 percent")
    self.assertFalse(setup.is_two_sample)
    self.assertEqual(setup.standalone_q_count, 100)
    self.assertEqual(setup.budget_weight, 1)

  def test_two_arm_execution_is_balanced_and_scored_between_arms(self):
    calls = []

    class FakeAgent:
      """Mimics the probe/score pair, including the overwrite semantics."""

      def __init__(self):
        self.symmetry_output_images = []
        self.last_judgments = []
        self.last_scalar_values = []
        self.last_block_reason = ""
        # Arm A: 20/50 pass. Arm B: 45/50 pass.
        self._pending = [[1.0] * 20 + [0.0] * 30, [1.0] * 45 + [0.0] * 5]

      def reset_symmetry_state(self, *, per_probe=False):
        calls.append(("reset", per_probe))
        self.symmetry_output_images = []
        self.last_judgments = []
        if per_probe:
          self.last_scalar_values = []

      async def run_symmetry_probes(self, args):
        calls.append(("probe", args))
        # Generating a batch destroys the previous batch's stored images,
        # which is precisely why scoring must be interleaved.
        self.symmetry_output_images = ["img"]
        return "{}"

      async def score_symmetry(self, args):
        calls.append(("score", args))
        self.last_scalar_values = self._pending.pop(0)
        return "{}"

    agent = FakeAgent()
    plan = precision_estimation.visual_symmetry_prompts.infer_contrast(
        "complex_chinese fails more often than chinese_char"
    )
    outcomes = asyncio.run(
        precision_estimation._run_symmetry_contrast_execution(
            agent=agent,
            plan=plan,
            q_count=100,
            success_criterion="both",
            log_fn=lambda *a, **k: None,
        )
    )

    # Each arm starts from a full reset, and probe and score strictly
    # alternate; arm A is never left unscored while arm B is generated.
    self.assertEqual(
        [c[0] for c in calls],
        ["reset", "probe", "score", "reset", "probe", "score"],
    )
    self.assertTrue(calls[0][1])
    self.assertTrue(calls[3][1])
    self.assertEqual(calls[1][1]["n_probes"], 50)
    self.assertEqual(calls[4][1]["n_probes"], 50)
    self.assertEqual(calls[1][1]["shape_type"], "complex_chinese")
    self.assertEqual(calls[4][1]["shape_type"], "chinese_char")

    self.assertLen(outcomes, 2)
    self.assertEqual((outcomes[0].n_success, outcomes[0].n_scored), (20, 50))
    self.assertEqual((outcomes[1].n_success, outcomes[1].n_scored), (45, 50))
    self.assertAlmostEqual(outcomes[0].rate, 0.4)
    self.assertAlmostEqual(outcomes[1].rate, 0.9)

  def test_two_arm_execution_rejects_an_absolute_plan(self):
    plan = precision_estimation.visual_symmetry_prompts.infer_contrast(
        "maze probes fail"
    )
    with self.assertRaises(ValueError):
      asyncio.run(
          precision_estimation._run_symmetry_contrast_execution(
              agent=None,
              plan=plan,
              q_count=100,
              success_criterion="both",
              log_fn=lambda *a, **k: None,
          )
      )

  def test_empty_arm_is_reported_as_zero_scored_not_as_zero_successes(self):
    class SilentAgent:

      def __init__(self):
        self.last_judgments = []
        self.last_scalar_values = []

      def reset_symmetry_state(self, *, per_probe=False):
        self.last_judgments = []
        if per_probe:
          self.last_scalar_values = []

      async def run_symmetry_probes(self, args):
        del args
        return "{}"

      async def score_symmetry(self, args):
        del args
        return "{}"

    plan = precision_estimation.visual_symmetry_prompts.infer_contrast(
        "complex_chinese fails more often than chinese_char"
    )
    outcomes = asyncio.run(
        precision_estimation._run_symmetry_contrast_execution(
            agent=SilentAgent(),
            plan=plan,
            q_count=100,
            success_criterion="both",
            log_fn=lambda *a, **k: None,
        )
    )
    # n_scored == 0 is the signal the caller turns into SCRAPE_FAILED. Reading
    # it as "0 successes out of 50" would report a maximal effect size for a
    # throttled arm.
    for outcome in outcomes:
      self.assertEqual(outcome.n_scored, 0)
      self.assertEqual(outcome.n_success, 0)

  def test_single_arm_execution_falls_back_when_agent_skips_tools(self):
    calls = []

    class IdleAgent:
      """An agent whose tool-calling step never runs the symmetry tools."""

      def __init__(self):
        self.symmetry_output_images = ["stale"]
        self.last_judgments = ["stale"]
        self.last_symmetry_shape_type = ""

      def reset_symmetry_state(self, *, per_probe=False):
        calls.append(("reset", per_probe))
        self.symmetry_output_images = []
        self.last_judgments = []

      async def run_symmetry_probes(self, args):
        calls.append(("probe", args))
        return "{}"

      async def score_symmetry(self, args):
        calls.append(("score", args))
        return "{}"

    class Sender:

      async def send_step(self, prompt):
        calls.append(("send", bool(prompt)))
        return "done"

    asyncio.run(
        precision_estimation._run_symmetry_execution(
            sender=Sender(),
            agent=IdleAgent(),
            q_count=7,
            h_text="maze probes fail",
            log_fn=lambda *a, **k: None,
        )
    )
    expected_shape = precision_estimation._infer_symmetry_shape_type(
        "maze probes fail"
    )
    self.assertEqual(
        calls,
        [
            ("reset", False),
            ("send", True),
            ("probe", {"shape_type": expected_shape, "n_probes": 7}),
            ("score", {}),
        ],
    )


if __name__ == "__main__":
  absltest.main()
