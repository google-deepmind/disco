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

"""Unit tests for behavior discovery prompts, runners, and query generation."""

import asyncio
import random
from unittest import mock

from absl.testing import absltest

from disco import agent_factory
from disco import backbone_client
from disco import behavior_discovery_prompts
from disco import flags_lib
from disco import runners
from disco import standalone_agent
from disco import taxonomy_utils
from disco import verification_prompts
from disco import verifier_config
from disco.agent import senders as agent_senders
from disco.core import config as config_lib
from disco.core import history
from disco.core import records as records_lib
from disco.core import round_executor
from disco.core import task_executor
from disco.models import genai_client
from disco.models import registry as model_registry
from disco.parsing import extract
from disco.parsing import parsers as parsers_lib
from disco.parsing import schemas
from disco.stats import plan as stats_plan


class BehaviorDiscoveryPromptsTest(absltest.TestCase):

  def test_get_prompts_module_resolution(self):
    self.assertIs(config_lib.get_prompts_module(), behavior_discovery_prompts)

  def test_fresh_agent_intro_prompt(self):
    prompt = behavior_discovery_prompts.get_fresh_agent_intro_prompt(
        target_model="gemini-3.8-flash",
        seed_instruction="Explain technical concepts using analogies",
        threshold=20.0,
    )
    self.assertIn("behavior pattern discovery system", prompt)
    self.assertIn("behavioral patterns", prompt)

  def test_scrape_and_judge_execution_prompt_categories(self):
    prompt = behavior_discovery_prompts.get_scrape_and_judge_execution_prompt(
        q_count=25
    )
    self.assertIn("EXHIBITS_BEHAVIOR", prompt)
    self.assertIn("DOES_NOT_EXHIBIT", prompt)
    self.assertIn("REFUSAL", prompt)
    self.assertIn("AMBIGUOUS", prompt)

  def test_recent_reflection_prompt(self):
    prompt = behavior_discovery_prompts.get_recent_failures_reflection_prompt()
    self.assertIn("Salient Pattern & Boundary Extraction", prompt)
    self.assertIn("Key Discriminators", prompt)
    self.assertIn("Boundaries & Counterexamples", prompt)

  def test_disco_fresh_agent_intro_prompt(self):
    prompt = behavior_discovery_prompts.get_disco_fresh_agent_intro_prompt(
        target_model="gemini-3.8-flash",
        seed_instruction="Explain technical concepts using analogies",
    )
    self.assertIn("Discover behavior patterns", prompt)
    self.assertIn("coarsened e-value feedback", prompt)
    self.assertIn("evaluated together via e-BH", prompt)
    self.assertIn(
        "With coarsened e-value feedback, you will NOT see the results", prompt
    )

  def test_disco_fresh_agent_intro_prompt_show_all_failures(self):
    prompt = behavior_discovery_prompts.get_disco_fresh_agent_intro_prompt(
        target_model="gemini-3.8-flash",
        seed_instruction="Explain technical concepts using analogies",
        show_all_failures=True,
    )
    self.assertIn("Discover behavior patterns", prompt)
    self.assertIn("coarsened e-value feedback", prompt)
    self.assertIn("evaluated together via e-BH", prompt)
    self.assertNotIn(
        "With coarsened e-value feedback, you will NOT see the results", prompt
    )
    self.assertIn("You will see previously tested queries and evidence", prompt)

  def test_disco_select_or_propose_prompt(self):
    prompt = behavior_discovery_prompts.get_disco_select_or_propose_prompt(
        disco_history_summary="History summary",
        active_hypotheses={"hyp_1": "Uses metaphors"},
        target_model="gemini-3.8-flash",
        seed_instruction="Explain technical concepts using analogies",
        e_max=50.0,
    )
    self.assertIn("Discover behavior patterns", prompt)
    self.assertIn("HYPOTHESIS_ID:", prompt)
    self.assertIn("hyp_1", prompt)
    self.assertIn("log(E_max)", prompt)
    self.assertIn(
        "Hypotheses marked [INACTIVE - ...] have finished evaluation and CANNOT"
        " be probed again.",
        prompt,
    )
    self.assertNotIn("Evidence Summary From All Hypotheses", prompt)

  def test_disco_select_or_propose_prompt_empty_active(self):
    prompt = behavior_discovery_prompts.get_disco_select_or_propose_prompt(
        disco_history_summary="History summary",
        active_hypotheses={},
        target_model="gemini-3.8-flash",
        seed_instruction="Explain technical concepts using analogies",
        e_max=50.0,
    )
    self.assertIn(
        "Currently ACTIVE hypotheses: None (all previous hypotheses in history"
        " are INACTIVE",
        prompt,
    )
    self.assertIn("you MUST propose a NEW hypothesis", prompt)

  def test_disco_select_or_propose_prompt_with_failure_summary(self):
    prompt = behavior_discovery_prompts.get_disco_select_or_propose_prompt(
        disco_history_summary="History summary",
        active_hypotheses={"hyp_1": "Uses metaphors"},
        target_model="gemini-3.8-flash",
        seed_instruction="Explain technical concepts using analogies",
        failure_summary="Example 1: Model failed on metaphor probe",
        e_max=50.0,
    )
    self.assertIn("Discover behavior patterns", prompt)
    self.assertIn("Evidence Summary From All Hypotheses", prompt)
    self.assertIn("Example 1: Model failed on metaphor probe", prompt)
    self.assertIn("log(E_max)", prompt)

  def test_disco_fresh_agent_intro_prompt_baseline_mode(self):
    prompt = behavior_discovery_prompts.get_disco_fresh_agent_intro_prompt(
        target_model="gemini-3.8-flash",
        seed_instruction="Explain technical concepts using analogies",
        baseline_mode=True,
    )
    self.assertIn("Discover behavior patterns", prompt)
    self.assertIn("evaluated together via e-BH", prompt)
    self.assertNotIn("coarsened e-value feedback", prompt)

  def test_disco_select_or_propose_prompt_baseline_mode(self):
    prompt = behavior_discovery_prompts.get_disco_select_or_propose_prompt(
        disco_history_summary="History summary",
        active_hypotheses={"hyp_1": "Uses metaphors"},
        target_model="gemini-3.8-flash",
        seed_instruction="Explain technical concepts using analogies",
        baseline_mode=True,
    )
    self.assertIn("Discover behavior patterns", prompt)
    self.assertIn("Evidence compounds across probe rounds", prompt)
    self.assertIn("evaluated jointly with e-BH", prompt)
    self.assertNotIn("log(E_max)", prompt)

  def test_disco_config_uses_behavior_prompts(self):
    cfg = config_lib.DiscoConfig(
        target_model="gemini-3.8-flash",
        seed_instruction="testing",
        budget=100,
        alpha=0.05,
        kappa=0.5,
    )
    self.assertIs(cfg.get_prompts_module(), behavior_discovery_prompts)

  def test_procedural_autorater_addendum(self):
    addendum = behavior_discovery_prompts.get_procedural_autorater_addendum(
        "count_words", allow_continuous_means=True
    )
    self.assertIn("here are available evaluation tools for this task", addendum)
    self.assertIn(
        "In future steps, you will be asked to evaluate outputs of queries",
        addendum,
    )
    self.assertIn("During this future evaluation step", addendum)
    self.assertIn("count_words", addendum)
    self.assertIn("run_procedural_autorater", addendum)
    self.assertIn("run_continuous_means_test", addendum)
    self.assertIn("run_paired_means_test", addendum)
    self.assertIn("analyze_responses", addendum)

    addendum_no_cont = (
        behavior_discovery_prompts.get_procedural_autorater_addendum(
            "count_words", allow_continuous_means=False
        )
    )
    self.assertIn("Do NOT use continuous means tests", addendum_no_cont)
    self.assertNotIn(
        "run_continuous_means_test if your hypothesis", addendum_no_cont
    )

  def test_procedural_scrape_and_measure_prompt(self):
    prompt = (
        behavior_discovery_prompts.get_procedural_scrape_and_measure_prompt(
            q_count=25, autorater_name="count_words"
        )
    )
    self.assertIn("EXACTLY the 25 queries", prompt)
    self.assertIn("count_words", prompt)
    self.assertIn("run_procedural_autorater", prompt)
    self.assertIn("analyze_responses", prompt)

  def test_disco_config_procedural_autorater(self):
    cfg = config_lib.DiscoConfig(
        target_model="gemini-3.8-flash",
        seed_instruction="testing",
        budget=100,
        alpha=0.05,
        kappa=0.5,
        procedural_autorater="count_words",
    )
    self.assertEqual(cfg.procedural_autorater, "count_words")

  def test_precision_round_context_prompt_without_failures(self):
    prompt_behavior = (
        behavior_discovery_prompts.get_precision_round_context_prompt(
            h_text="Model uses metaphors",
            prior_e_values=[("Animal metaphors", 2.5)],
        )
    )
    self.assertIn("DISCO precision verification round", prompt_behavior)
    self.assertIn("Model uses metaphors", prompt_behavior)
    self.assertIn("Round 1 | E-value: 2.5000", prompt_behavior)
    self.assertNotIn("Concrete evidence examples", prompt_behavior)

  def test_precision_round_context_prompt_with_failures(self):

    evidence = [{
        "prompt": "Explain photosynthesis",
        "response": "Think of it like a solar panel",
        "judgment": "EXHIBITS_BEHAVIOR",
        "reasoning": "Uses solar panel metaphor",
    }]
    prompt_behavior = (
        behavior_discovery_prompts.get_precision_round_context_prompt(
            h_text="Model uses analogies",
            prior_e_values=[("Science analogies", 5.0)],
            failures=evidence,
        )
    )
    self.assertIn(
        "Concrete evidence examples from prior probes of THIS hypothesis:",
        prompt_behavior,
    )
    self.assertIn("Explain photosynthesis", prompt_behavior)
    self.assertIn("Think of it like a solar panel", prompt_behavior)
    self.assertIn("Uses solar panel metaphor", prompt_behavior)

  def test_precision_round_context_prompt_with_failure_reflections(self):
    prompt_behavior = (
        behavior_discovery_prompts.get_precision_round_context_prompt(
            h_text="Model uses metaphors",
            prior_e_values=[("Animal metaphors", 2.5)],
            failure_reflections=[
                "The model tends to use nature-based metaphors for science"
                " topics."
            ],
        )
    )
    self.assertIn(
        "--- Summary of Salient Examples from Prior Probes of THIS"
        " Hypothesis ---",
        prompt_behavior,
    )
    self.assertIn(
        "nature-based metaphors for science topics",
        prompt_behavior,
    )
    self.assertIn("--- End of Summary ---", prompt_behavior)

    prompt_verif = verification_prompts.get_precision_round_context_prompt(
        h_text="Model uses analogies",
        prior_e_values=[("Science analogies", 5.0)],
        failure_reflections=[
            "Analogies consistently appear in biology explanations."
        ],
    )
    self.assertIn(
        "--- Summary of Salient Examples from Prior Probes of THIS"
        " Hypothesis ---",
        prompt_verif,
    )
    self.assertIn(
        "biology explanations",
        prompt_verif,
    )
    self.assertIn("--- End of Summary ---", prompt_verif)

  def test_precision_round_context_prompt_no_reflections_when_none(self):
    prompt = behavior_discovery_prompts.get_precision_round_context_prompt(
        h_text="Model uses metaphors",
        prior_e_values=[("Animal metaphors", 2.5)],
        failure_reflections=None,
    )
    self.assertNotIn("Summary of Salient", prompt)

  def test_get_stats_significance_prompt_directional_instructions(self):
    prompt_behavior = behavior_discovery_prompts.get_stats_significance_prompt()
    self.assertIn("alternative", prompt_behavior)
    self.assertIn("'larger'", prompt_behavior)
    self.assertIn("'smaller'", prompt_behavior)
    self.assertIn("REASONING:", prompt_behavior)
    self.assertIn("statistical test arguments", prompt_behavior)

  def test_get_test_preregistration_prompt(self):
    prompt = behavior_discovery_prompts.get_test_preregistration_prompt(
        hypothesis_text="Model produces overly long recipe responses",
    )
    self.assertIn("Step 2b: Statistical Test Pre-Registration", prompt)
    self.assertIn("Model produces overly long recipe responses", prompt)
    self.assertIn("TEST_TYPE:", prompt)
    self.assertIn("NULL_BASELINE:", prompt)
    self.assertIn("ALTERNATIVE:", prompt)
    self.assertIn("REASONING:", prompt)
    self.assertIn("'larger'", prompt)

  def test_get_default_preregistration_retry_prompt(self):
    prompt = (
        behavior_discovery_prompts.get_default_preregistration_retry_prompt(
            error_msg="Required TEST_TYPE was missing."
        )
    )
    self.assertIn("Required TEST_TYPE was missing.", prompt)
    self.assertIn("TEST_TYPE:", prompt)
    self.assertIn("NULL_BASELINE:", prompt)
    self.assertIn("ALTERNATIVE:", prompt)

  def test_get_stats_significance_prompt_with_plan(self):
    plan = stats_plan.PreRegisteredTestPlan(
        test_type="Binomial test",
        target_metric="French recipe word count > 500",
        null_baseline=0.5,
        alternative="larger",
        reasoning="Testing if frequency is higher than 50% baseline",
    )
    prompt = behavior_discovery_prompts.get_stats_significance_prompt(plan=plan)
    self.assertIn("PRE-REGISTERED TEST PLAN:", prompt)
    self.assertIn("Test Type: Binomial test", prompt)
    self.assertIn("Target Metric: French recipe word count > 500", prompt)
    self.assertIn("Null Baseline: 0.5", prompt)
    self.assertIn("Alternative Direction: larger", prompt)
    self.assertIn("EXACTLY ONE", prompt)

  def test_test_preregistration_parser_validate(self):
    p = parsers_lib.TestPreregistrationParser(model="gemini-3.8-flash")
    raw_extraction = schemas.TestPreregistrationExtraction(
        test_type="Binomial test",
        target_metric="Failure rate",
        null_baseline=0.5,
        alternative="larger",
        reasoning="Testing elevated failure rate",
    )
    plan = p._validate(raw_extraction, original_resp="")
    self.assertEqual(plan.test_type, "Binomial test")
    self.assertEqual(plan.target_metric, "Failure rate")
    self.assertEqual(plan.null_baseline, 0.5)
    self.assertEqual(plan.alternative, "larger")

  def test_test_preregistration_parser_missing_test_type(self):
    p = parsers_lib.TestPreregistrationParser(model="gemini-3.8-flash")
    raw_extraction = schemas.TestPreregistrationExtraction(
        test_type="",
        target_metric="Failure rate",
    )
    with self.assertRaises(ValueError):
      p._validate(raw_extraction, original_resp="")

  def test_test_preregistration_parser_with_is_paired(self):
    p = parsers_lib.TestPreregistrationParser(model="gemini-3.8-flash")
    raw_extraction = schemas.TestPreregistrationExtraction(
        test_type="Binomial test",
        target_metric="Behavior occurrence",
        null_baseline=0.1,
        alternative="larger",
        is_paired=True,
        reasoning="Test if behavior rate exceeds 10%",
    )
    plan = p._validate(raw_extraction, original_resp="")
    self.assertEqual(plan.test_type, "Binomial test")
    self.assertEqual(plan.target_metric, "Behavior occurrence")
    self.assertEqual(plan.null_baseline, 0.1)
    self.assertEqual(plan.alternative, "larger")
    self.assertTrue(plan.is_paired)
    self.assertTrue(plan.implies_paired)
    # The plan is the only channel for the pairing decision: the parser must
    # not retain it, or a later hypothesis whose parse fails would inherit it.
    self.assertFalse(hasattr(p, "last_is_paired"))

  def test_preregistration_plan_implies_paired_from_test_type(self):
    """A paired-only test type implies pairing even if IS_PAIRED is False."""
    p = parsers_lib.TestPreregistrationParser(model="gemini-3.8-flash")
    for test_type in ("McNemar's Test", "Paired continuous test"):
      with self.subTest(test_type=test_type):
        plan = p._validate(
            schemas.TestPreregistrationExtraction(
                test_type=test_type,
                target_metric="Per-pair difference",
                null_baseline=0.5,
                alternative="larger",
                is_paired=False,
            ),
            original_resp="",
        )
        self.assertFalse(plan.is_paired)
        self.assertTrue(plan.implies_paired)
        self.assertEqual(plan.unit_of_analysis, "pair")

  def test_extract_null_proportion_from_seed(self):
    seed_75 = (
        "Investigate length. Use 75 percent as the null hypothesis percentage."
    )
    self.assertAlmostEqual(
        runners._extract_null_proportion_from_seed(seed_75), 0.75
    )

    seed_pct = "Investigate length. Use 75% as the null hypothesis percentage."
    self.assertAlmostEqual(
        runners._extract_null_proportion_from_seed(seed_pct), 0.75
    )

    seed_default = "Discover behavior without explicit null mention."
    self.assertAlmostEqual(
        runners._extract_null_proportion_from_seed(seed_default, default=0.5),
        0.5,
    )

  def test_guard_null_uses_plan_null_for_continuous_tests(self):
    plan = stats_plan.PreRegisteredTestPlan(
        test_type="Continuous means test",
        target_metric="word count",
        null_baseline=20.0,
        alternative="larger",
    )
    # Mean 15 words is below the 20-word null, so a "larger" p of 0.02 is the
    # wrong tail. Against the old 0.5 fallback it looked like the right tail.
    res = stats_plan.StatsResult(
        test_type="Continuous means test",
        p_value=0.02,
        group_a_mean=15.0,
        group_a_obs=25,
        alternative="larger",
    )
    null = runners._guard_null(res, plan, 0.5)
    self.assertEqual(null, 20.0)
    p_adj = runners._guard_p_value_direction(
        res, lambda *_: None, null_value=null, use_direction_guard=True
    )
    self.assertAlmostEqual(p_adj, 0.98)

  def test_guard_null_uses_task_null_for_proportion_tests(self):
    plan = stats_plan.PreRegisteredTestPlan(
        test_type="Binomial test", target_metric="m", null_baseline=0.3
    )
    res = stats_plan.StatsResult(
        test_type="Binomial test", p_value=0.02, group_a_mean=0.4
    )
    self.assertEqual(runners._guard_null(res, plan, 0.5), 0.5)
    self.assertEqual(runners._guard_null(res, plan, 0.9), 0.9)
    self.assertEqual(runners._guard_null(res, None, 0.5), 0.5)

  def test_symmetry_guard_tests_against_the_symmetry_null(self):
    # A symmetry "fails" hypothesis: 7/10 completions succeed against the
    # task's 0.9 reference, so the "smaller" tail is the right one. Against
    # the seed-derived 0.5 the guard used to fall back to, 0.7 looked like
    # the wrong tail and p was flipped to 0.98.
    config = config_lib.DiscoConfig(
        target_model="t",
        seed_instruction="Probe visual symmetry completion.",
        budget=9,
        alpha=0.1,
        kappa=0.5,
        task_suite="visual_symmetry",
    )
    null = task_executor.resolve_task_executor(
        config, lambda *_: None
    ).null_rate()
    self.assertAlmostEqual(null, 0.9)
    res = stats_plan.StatsResult(
        test_type="Proportion z-test",
        p_value=0.02,
        group_a_mean=0.7,
        group_a_obs=10,
        alternative="smaller",
    )
    p_adj = runners._guard_p_value_direction(
        res,
        lambda *_: None,
        null_value=runners._guard_null(res, None, null),
        use_direction_guard=True,
    )
    self.assertAlmostEqual(p_adj, 0.02)
    p_old = runners._guard_p_value_direction(
        res, lambda *_: None, null_value=0.5, use_direction_guard=True
    )
    self.assertAlmostEqual(p_old, 0.98)

  def test_text_task_null_rate(self):
    def text_null(seed: str) -> float:
      config = config_lib.DiscoConfig(
          target_model="t",
          seed_instruction=seed,
          budget=9,
          alpha=0.1,
          kappa=0.5,
      )
      return task_executor.resolve_task_executor(
          config, lambda *_: None
      ).null_rate()

    self.assertEqual(text_null("Find refusals."), 0.5)
    self.assertEqual(text_null("Use 90% as the null."), 0.9)

  def test_direction_guard_one_sample(self):
    logs = []
    log_fn = logs.append

    # Case 1: Mean in wrong direction (0.20 <= 0.75) and two-sided p-value
    # reported as 0.0001
    res_wrong_side = stats_plan.StatsResult(
        test_type="Binomial test",
        p_value=0.0001,
        group_a_mean=0.20,
        group_a_obs=10,
        reasoning="Testing if success rate > 0.75",
    )
    p_adj = runners._guard_p_value_direction(
        res_wrong_side, log_fn, null_value=0.75, use_direction_guard=True
    )
    self.assertAlmostEqual(p_adj, 0.9999, places=4)
    self.assertTrue(any("Direction Guard" in l for l in logs))

    # Case 1b: Direction guard disabled -> p-value left unchanged
    p_unchanged = runners._guard_p_value_direction(
        res_wrong_side, log_fn, null_value=0.75, use_direction_guard=False
    )
    self.assertAlmostEqual(p_unchanged, 0.0001)

    # Case 2: Mean in right direction (0.90 > 0.75) with p-value 0.03
    res_right_side = stats_plan.StatsResult(
        test_type="Binomial test",
        p_value=0.03,
        group_a_mean=0.90,
        group_a_obs=10,
    )
    p_good = runners._guard_p_value_direction(
        res_right_side, log_fn, null_value=0.75, use_direction_guard=True
    )
    self.assertAlmostEqual(p_good, 0.03)

    # Case 3: Custom null_proportion in res (0.20): mean 0.40 > 0.20 -> NOT
    # triggered even when fallback null_value is 0.50!
    res_custom_null = stats_plan.StatsResult(
        test_type="Binomial test",
        p_value=0.061051,
        group_a_mean=0.40,
        group_a_obs=15,
        null_proportion=0.20,
        alternative="larger",
    )
    p_custom = runners._guard_p_value_direction(
        res_custom_null, log_fn, null_value=0.50, use_direction_guard=True
    )
    self.assertAlmostEqual(p_custom, 0.061051, places=6)

    # Case 4: alternative="smaller": mean (0.40) > null (0.20) but small
    # p=0.01 -> triggered!
    res_smaller_wrong = stats_plan.StatsResult(
        test_type="Binomial test",
        p_value=0.01,
        group_a_mean=0.40,
        group_a_obs=15,
        null_proportion=0.20,
        alternative="smaller",
    )
    p_smaller_adj = runners._guard_p_value_direction(
        res_smaller_wrong, log_fn, null_value=0.50, use_direction_guard=True
    )
    self.assertAlmostEqual(p_smaller_adj, 0.99, places=4)

    # Case 5: alternative="two-sided": guard is skipped
    res_twosided = stats_plan.StatsResult(
        test_type="Binomial test",
        p_value=0.01,
        group_a_mean=0.10,
        group_a_obs=15,
        null_proportion=0.20,
        alternative="two-sided",
    )
    p_twosided = runners._guard_p_value_direction(
        res_twosided, log_fn, null_value=0.50, use_direction_guard=True
    )
    self.assertAlmostEqual(p_twosided, 0.01)

  def test_direction_guard_two_sample(self):
    logs = []
    log_fn = logs.append

    # Group A (0.2) < Group B (0.8), but reported p=0.01 (two-sided)
    res_2sample_wrong = stats_plan.StatsResult(
        test_type="Two-Sample Proportion Test",
        p_value=0.01,
        group_a_mean=0.2,
        group_a_obs=10,
        group_b_mean=0.8,
        group_b_obs=10,
    )
    p_adj = runners._guard_p_value_direction(
        res_2sample_wrong, log_fn, use_direction_guard=True
    )
    self.assertAlmostEqual(p_adj, 0.99, places=4)

    # When disabled:
    p_unchanged = runners._guard_p_value_direction(
        res_2sample_wrong, log_fn, use_direction_guard=False
    )
    self.assertAlmostEqual(p_unchanged, 0.01)

  def test_taxonomy_prompts_with_seed_and_self_containment(self):
    meta_prompt = behavior_discovery_prompts.get_taxonomy_meta_prompt(
        taxonomy_nodes=["Factor: Category -> Subcategory -> Node"],
        hypothesis="Model responses are short",
        num_queries=10,
        seed_instruction="Do NOT propose length constraints",
    )
    self.assertIn("SEED: Do NOT propose length constraints", meta_prompt)
    self.assertIn("Self-Containment Requirement", meta_prompt)
    self.assertIn("SINGLE-TURN, ZERO-CONTEXT", meta_prompt)

    critic_prompt = behavior_discovery_prompts.get_taxonomy_critic_prompt(
        queries=["1. Query text"],
        taxonomy_nodes=["Factor: Category -> Subcategory -> Node"],
        hypothesis="Model responses are short",
        seed_instruction="Do NOT propose length constraints",
    )
    self.assertIn("SEED: Do NOT propose length constraints", critic_prompt)
    self.assertIn("EXECUTION VIABILITY", critic_prompt)
    self.assertIn("ALL THREE checks", critic_prompt)

  def test_taxonomy_paired_prompts(self):
    gen_prompt = behavior_discovery_prompts.get_taxonomy_generation_prompt(
        hypothesis="Dishes in Group A are longer than Group B",
        seed_instruction="Compare recipe word counts",
        taxonomy_depth=3,
        best_of_n=3,
        paired=True,
    )
    self.assertIn("PAIRED COMPARISON MODE", gen_prompt)
    self.assertIn("L3: [A] <Group A item/scenario> vs [B]", gen_prompt)

    meta_prompt = behavior_discovery_prompts.get_taxonomy_meta_prompt(
        taxonomy_nodes=[
            "[A] Beef Wellington",
            "[B] Grilled steak",
            "[A] Lasagna",
            "[B] Caprese salad",
        ],
        hypothesis="Dishes in Group A are longer than Group B",
        num_queries=4,
        seed_instruction="Compare recipe word counts",
        paired=True,
    )
    self.assertIn("MATCHED PAIRS", meta_prompt)
    self.assertIn("Odd-numbered queries (1, 3, 5, ...) → Group A", meta_prompt)
    self.assertIn("Even-numbered queries (2, 4, 6, ...) → Group B", meta_prompt)
    self.assertIn(
        "Do NOT include group labels, tags like [A] or [B]", meta_prompt
    )

  def test_classify_leaf_nodes_paired(self):
    leaf_nodes = [
        "L3: [A] Beef Wellington vs [B] Grilled Steak",
        "L3: [A] Lasagna Bolognese vs [B] Caprese Salad",
        "Ramen Bowl vs Cucumber Salad",
        "[A] Dish Solo A",
        "[B] Dish Solo B",
    ]
    pairs = taxonomy_utils.classify_leaf_nodes_paired(leaf_nodes)
    self.assertLen(pairs, 4)
    self.assertEqual(pairs[0], ("[A] Beef Wellington", "[B] Grilled Steak"))
    self.assertEqual(pairs[1], ("[A] Lasagna Bolognese", "[B] Caprese Salad"))
    self.assertEqual(pairs[2], ("[A] Ramen Bowl", "[B] Cucumber Salad"))
    self.assertEqual(pairs[3], ("[A] Dish Solo A", "[B] Dish Solo B"))

  def test_sample_paired_nodes(self):
    paired_nodes = [
        ("[A] Dish 1", "[B] Dish A"),
        ("[A] Dish 2", "[B] Dish B"),
        ("[A] Dish 3", "[B] Dish C"),
    ]
    sampled = taxonomy_utils.sample_paired_nodes(paired_nodes, q_count=6)
    self.assertLen(sampled, 6)
    # Check that odd indices (0, 2, 4) are Group A and even indices (1, 3, 5)
    # are Group B
    for i, node in enumerate(sampled):
      if i % 2 == 0:
        self.assertTrue(node.startswith("[A]"))
      else:
        self.assertTrue(node.startswith("[B]"))

  def test_paired_sampling_preserves_correspondence(self):
    """Regression test: sampling must not scramble A/B pairs."""
    paired_nodes = [
        ("[A] English croissants", "[B] French croissants"),
        ("[A] English soufflé", "[B] French soufflé"),
        ("[A] English macarons", "[B] French macarons"),
        ("[A] English éclair", "[B] French éclair"),
    ]
    random.seed(42)
    sampled = taxonomy_utils.sample_paired_nodes(paired_nodes, q_count=6)
    # For each pair in the output, the dish name must match
    for i in range(0, len(sampled), 2):
      a_dish = sampled[i].split("English ")[1]
      b_dish = sampled[i + 1].split("French ")[1]
      self.assertEqual(
          a_dish,
          b_dish,
          f"Pair mismatch at index {i}: A={sampled[i]}, B={sampled[i+1]}",
      )

  def test_precision_round_context_prompt_prior_rounds(self):
    prompt = behavior_discovery_prompts.get_precision_round_context_prompt(
        h_text="Model uses metaphors",
        prior_e_values=[("Animal metaphors", 2.5), ("Plant metaphors", 0.4)],
    )
    self.assertIn(
        "Prior rounds of this precision verification run:",
        prompt,
    )
    self.assertIn("- Round 1 | E-value: 2.5000", prompt)
    self.assertIn("- Round 2 | E-value: 0.4000", prompt)
    # The round labels (sub-hypothesis texts of old records) are not shown.
    self.assertNotIn("Animal metaphors", prompt)
    self.assertNotIn("DIVERSITY & GENERALIZATION REQUIREMENT:", prompt)

  def test_baseline_fresh_agent_intro_prompt(self):
    prompt_behavior = behavior_discovery_prompts.get_fresh_agent_intro_prompt(
        target_model="gemini-3.8-flash",
        seed_instruction="Find length differences",
        threshold=5.0,
        baseline_mode=True,
    )
    self.assertIn(
        "controlled offline at system termination via e-BH", prompt_behavior
    )
    self.assertNotIn("considered verified behavioral patterns", prompt_behavior)

  def test_build_failure_summary_phrasing(self):
    groups = {
        "h1": records_lib.HypothesisRecord(
            text="Hypothesis 1",
            failures=[{
                "prompt": "prompt 1",
                "response": "resp 1",
                "judgment": "FAILURE",
                "reasoning": "reason 1",
            }],
        ),
    }
    summary_beh = runners._build_failure_summary(groups)
    self.assertIn(
        "Use this information to decide where a fruitful direction to probe"
        " might be.",
        summary_beh,
    )
    self.assertIn("behavior patterns", summary_beh)

  def test_parse_selected_example_indices(self):
    # JSON array
    indices = extract.parse_selected_example_indices(
        "[1, 4, 7]", total_candidates=10, num_to_select=3
    )
    self.assertEqual(indices, [0, 3, 6])

    # Text containing numbers
    indices = extract.parse_selected_example_indices(
        "Selected examples: 2, 5, 8", total_candidates=10, num_to_select=3
    )
    self.assertEqual(indices, [1, 4, 7])

    # Fewer available
    indices = extract.parse_selected_example_indices(
        "[1]", total_candidates=2, num_to_select=3
    )
    self.assertEqual(indices, [0, 1])

  def test_summarize_failure_block_empty(self):
    res = asyncio.run(
        runners._summarize_failure_block(
            raw_summary="",
            seed_instruction="Seed",
            model="gemini-3.8-flash",
            log_fn=lambda _: None,
        )
    )
    self.assertEqual(res, "")

  def test_summarize_failure_block_success(self):
    mock_backbone = mock.MagicMock()
    mock_backbone.generate = mock.AsyncMock(
        return_value="This is a 2-sentence summary of the failures."
    )
    with mock.patch.object(
        backbone_client,
        "get_backbone_client",
        return_value=mock_backbone,
    ):
      res = asyncio.run(
          runners._summarize_failure_block(
              raw_summary="Concrete failures seen so far...",
              seed_instruction="Seed",
              model="gemini-3.8-flash",
              log_fn=lambda _: None,
          )
      )
    self.assertEqual(res, "This is a 2-sentence summary of the failures.")

  def test_summarize_failure_block_fallback_on_error(self):
    mock_backbone = mock.MagicMock()
    mock_backbone.generate = mock.AsyncMock(
        side_effect=RuntimeError("API error")
    )
    long_raw = "A" * 1000
    with mock.patch.object(
        backbone_client,
        "get_backbone_client",
        return_value=mock_backbone,
    ):
      res = asyncio.run(
          runners._summarize_failure_block(
              raw_summary=long_raw,
              seed_instruction="Seed",
              model="gemini-3.8-flash",
              log_fn=lambda _: None,
          )
      )
    self.assertEqual(res, "A" * 500)

  def test_disco_select_or_propose_prompt_direct_hypotheses(self):
    prompt = behavior_discovery_prompts.get_disco_select_or_propose_prompt(
        disco_history_summary="History",
        active_hypotheses={"hyp_1": "Uses metaphors"},
        target_model="gemini-3.8-flash",
        seed_instruction="Explain technical concepts",
        e_max=50.0,
    )
    self.assertIn("The hypothesis MUST be detailed and operationalized", prompt)
    self.assertIn(
        "Do NOT state or assume a different number of queries in your"
        " hypothesis text.",
        prompt,
    )
    self.assertNotIn("sub-hypothesis", prompt)

  def test_fresh_agent_intro_prompt_direct_hypotheses(self):
    prompt = behavior_discovery_prompts.get_fresh_agent_intro_prompt(
        target_model="gemini-3.8-flash",
        seed_instruction="Explain technical concepts",
        threshold=20.0,
        chunk_size=10,
    )
    self.assertIn("Generate queries directly to test H.", prompt)
    self.assertNotIn("Propose a Sub-Hypothesis", prompt)
    self.assertIn("do not call tools or execute anything else", prompt)

  def test_disco_runner_baseline_mode_init(self):
    cfg = config_lib.DiscoConfig(
        target_model="gemini-3.8-flash",
        seed_instruction="Explain technical concepts",
        budget=100,
        alpha=0.05,
        kappa=0.5,
        method="ebh_baseline",
    )
    runner = runners.DiscoRunner(cfg, cfg.build_parsers())
    self.assertIsNone(runner.null_tables)
    self.assertEmpty(runner.bin_edges)
    self.assertEqual(runner.init_bin, 1)

  def test_disco_runner_baseline_intro_prompt(self):
    cfg = config_lib.DiscoConfig(
        target_model="gemini-3.8-flash",
        seed_instruction="Explain technical concepts",
        budget=100,
        alpha=0.05,
        kappa=0.5,
        method="ebh_baseline",
    )
    runner = runners.DiscoRunner(cfg, cfg.build_parsers())
    prompt = runner._get_intro_prompt(cfg.get_prompts_module())
    self.assertIn("evaluated together via e-BH", prompt)
    self.assertNotIn("coarsened e-value feedback", prompt)

  def test_disco_runner_baseline_context_prompt(self):
    cfg = config_lib.DiscoConfig(
        target_model="gemini-3.8-flash",
        seed_instruction="Explain technical concepts",
        budget=100,
        alpha=0.05,
        kappa=0.5,
        method="ebh_baseline",
    )
    runner = runners.DiscoRunner(cfg, cfg.build_parsers())
    rec = records_lib.HypothesisRecord(text="Uses metaphors")
    rec.sub_hypotheses.append("Sub 1")
    runner.running_e_values["h1"] = 12.5
    prompt = history.build_disco_round_context(
        runner.config, runner.state, "Uses metaphors", rec, "h1"
    )
    self.assertIn("Aggregated E-value: 12.5000", prompt)
    self.assertNotIn("coarsened status", prompt)

  def test_get_disco_history_summary_baseline_mode(self):
    groups = {
        "h1": records_lib.HypothesisRecord(text="Uses metaphors"),
        "h2": records_lib.HypothesisRecord(text="Uses bullet points"),
        "h3": records_lib.HypothesisRecord(text="Refuses queries"),
        "h4": records_lib.HypothesisRecord(text="Unprobed"),
    }
    summary = runners._get_disco_history_summary(
        hypothesis_groups=groups,
        active_hypotheses={"h1", "h2", "h3", "h4"},
        terminal_statuses={"h1": "A", "h2": "A", "h3": "A", "h4": "A"},
        probe_counts={"h1": 2, "h2": 1, "h3": 1, "h4": 0},
        transition_scores={"h1": [0.5, 0.2]},
        e_min=0.01,
        e_max=100.0,
        baseline_mode=True,
        running_e_values={"h1": 150.0, "h2": 5.0, "h3": 0.005, "h4": 1.0},
    )
    self.assertIn("Baseline Mode", summary)
    self.assertIn("Aggregated E=150.0000", summary)
    self.assertIn("[STRONG EVIDENCE]", summary)
    self.assertIn("[MODERATE EVIDENCE]", summary)
    self.assertIn("[WEAK EVIDENCE]", summary)
    self.assertIn("All hypotheses will be tested at the end via e-BH", summary)
    self.assertNotIn("transition score", summary)

  def test_disco_runner_baseline_type1_mode_init(self):
    cfg = config_lib.DiscoConfig(
        target_model="gemini-3.8-flash",
        seed_instruction="Explain technical concepts",
        budget=100,
        alpha=0.05,
        kappa=0.5,
        method="type1_baseline",
    )
    runner = runners.DiscoRunner(cfg, cfg.build_parsers())
    self.assertIsNone(runner.null_tables)
    self.assertEmpty(runner.bin_edges)
    self.assertEqual(runner.init_bin, 1)

  def test_disco_runner_baseline_type1_intro_prompt(self):
    cfg = config_lib.DiscoConfig(
        target_model="gemini-3.8-flash",
        seed_instruction="Explain technical concepts",
        budget=100,
        alpha=0.05,
        kappa=0.5,
        method="type1_baseline",
    )
    runner = runners.DiscoRunner(cfg, cfg.build_parsers())
    prompt = runner._get_intro_prompt(cfg.get_prompts_module())
    self.assertIn("evaluated together via e-BH", prompt)
    self.assertNotIn("coarsened e-value feedback", prompt)

  def test_disco_runner_baseline_type1_context_prompt(self):
    cfg = config_lib.DiscoConfig(
        target_model="gemini-3.8-flash",
        seed_instruction="Explain technical concepts",
        budget=100,
        alpha=0.05,
        kappa=0.5,
        method="type1_baseline",
    )
    runner = runners.DiscoRunner(cfg, cfg.build_parsers())
    rec = records_lib.HypothesisRecord(text="Uses metaphors")
    rec.sub_hypotheses.append("Sub 1")
    runner.running_e_values["h1"] = 25.0
    prompt = history.build_disco_round_context(
        runner.config, runner.state, "Uses metaphors", rec, "h1"
    )
    self.assertIn("Aggregated E-value: 25.0000", prompt)
    self.assertNotIn("coarsened status", prompt)

  def test_get_disco_history_summary_baseline_type1_mode(self):
    groups = {
        "h1": records_lib.HypothesisRecord(text="Uses metaphors"),
        "h2": records_lib.HypothesisRecord(text="Uses bullet points"),
        "h3": records_lib.HypothesisRecord(text="Refuses queries"),
        "h4": records_lib.HypothesisRecord(text="Unprobed"),
    }
    summary = runners._get_disco_history_summary(
        hypothesis_groups=groups,
        active_hypotheses={"h2", "h3", "h4"},
        terminal_statuses={"h1": "H", "h2": "A", "h3": "A", "h4": "A"},
        probe_counts={"h1": 2, "h2": 1, "h3": 1, "h4": 0},
        transition_scores={"h1": [0.5, 0.2]},
        e_min=0.01,
        e_max=100.0,
        baseline_type1_mode=True,
        running_e_values={"h1": 25.0, "h2": 5.0, "h3": 0.2, "h4": 1.0},
    )
    self.assertIn("Baseline Mode (Type-I Error)", summary)
    self.assertIn("FROZEN (crossed 1/alpha)", summary)
    self.assertIn("Direct Type-I thresholding (no e-BH)", summary)

  def test_get_disco_history_summary_baseline_mode_statuses(self):
    """Baseline (k_bar e-BH) history labels frozen and exited-low hypotheses."""
    groups = {
        "h1": records_lib.HypothesisRecord(text="Frozen at K_bar/alpha"),
        "h2": records_lib.HypothesisRecord(text="Still active"),
        "h3": records_lib.HypothesisRecord(text="Exited low"),
    }
    summary = runners._get_disco_history_summary(
        hypothesis_groups=groups,
        active_hypotheses={"h2"},
        terminal_statuses={"h1": "H", "h2": "A", "h3": "L"},
        probe_counts={"h1": 1, "h2": 2, "h3": 1},
        transition_scores={},
        e_min=0.99,
        e_max=50.0,
        baseline_mode=True,
        running_e_values={"h2": 3.0},
        final_e_values={"h1": 400.0, "h3": 0.2},
    )
    self.assertIn("Baseline Mode", summary)
    self.assertIn(
        "ID: h1 (Frozen at K_bar/alpha) [INACTIVE - EXITED-HIGH (crossed"
        " K_bar/alpha), 1 probes, Aggregated E=400.0000] [STRONG EVIDENCE]",
        summary,
    )
    self.assertIn(
        "ID: h2 (Still active) [ACTIVE, 2 probes, Aggregated E=3.0000]"
        " [MODERATE EVIDENCE]",
        summary,
    )
    self.assertIn(
        "ID: h3 (Exited low) [INACTIVE - EXITED-LOW, 1 probes, Aggregated"
        " E=0.2000] [WEAK EVIDENCE]",
        summary,
    )
    self.assertIn("All hypotheses will be tested at the end via e-BH.", summary)

  def test_get_disco_history_summary_disco_mode_evidence_tags(self):
    groups = {
        "h1": records_lib.HypothesisRecord(text="High e-value hypothesis"),
        "h2": records_lib.HypothesisRecord(text="Moderate e-value hypothesis"),
        "h3": records_lib.HypothesisRecord(text="Low e-value hypothesis"),
        "h4": records_lib.HypothesisRecord(text="Untested hypothesis"),
    }
    summary = runners._get_disco_history_summary(
        hypothesis_groups=groups,
        active_hypotheses={"h2", "h4"},
        terminal_statuses={"h1": "H", "h2": "A", "h3": "L", "h4": "A"},
        probe_counts={"h1": 3, "h2": 2, "h3": 2, "h4": 0},
        transition_scores={"h1": [0.5, 0.8], "h2": [0.2], "h3": [-0.5]},
        e_min=0.01,
        e_max=100.0,
        baseline_mode=False,
        running_e_values={"h1": 120.0, "h2": 4.5, "h3": 0.05, "h4": 1.0},
    )
    self.assertIn("DISCO Mode", summary)
    self.assertIn(
        "[INACTIVE - EXITED-HIGH, 3 probes] [STRONG EVIDENCE]", summary
    )
    self.assertIn("[ACTIVE, 2 probes] [MODERATE EVIDENCE]", summary)
    self.assertIn("[INACTIVE - EXITED-LOW, 2 probes] [WEAK EVIDENCE]", summary)
    self.assertIn("ID: h4 (Untested hypothesis) [ACTIVE, 0 probes]\n", summary)
    self.assertIn("  Probe transition scores: +0.500, +0.800\n", summary)
    self.assertIn("  Probe transition scores: -0.500\n", summary)
    self.assertIn("All hypotheses will be tested at the end via e-BH", summary)
    self.assertNotIn("Aggregated E=", summary)

  def test_disco_runner_disco_context_prompt_evidence_tag(self):
    cfg = config_lib.DiscoConfig(
        target_model="gemini-3.8-flash",
        seed_instruction="Explain technical concepts",
        budget=100,
        alpha=0.05,
        kappa=0.5,
        method="disco",
        e_min=0.01,
    )
    runner = runners.DiscoRunner(cfg, cfg.build_parsers())
    rec = records_lib.HypothesisRecord(text="Uses metaphors")
    rec.sub_hypotheses.append("Sub 1")
    runner.running_e_values["h1"] = 0.2
    runner.probe_counts["h1"] = 1
    prompt = history.build_disco_round_context(
        runner.config, runner.state, "Uses metaphors", rec, "h1"
    )
    self.assertIn("[MODERATE EVIDENCE]", prompt)
    self.assertNotIn("[WEAK EVIDENCE]", prompt)
    self.assertIn("coarsened status: A", prompt)

  def test_get_disco_history_summary_uses_final_e_values_on_deactivation(self):
    groups = {
        "h1": records_lib.HypothesisRecord(text="Exited high"),
        "h2": records_lib.HypothesisRecord(text="Exited low"),
    }
    summary = runners._get_disco_history_summary(
        hypothesis_groups=groups,
        active_hypotheses=set(),
        terminal_statuses={"h1": "H", "h2": "L"},
        probe_counts={"h1": 2, "h2": 1},
        transition_scores={"h1": [1.2, 1.5], "h2": [-1.1]},
        e_min=0.01,
        e_max=100.0,
        baseline_mode=False,
        running_e_values={},
        final_e_values={"h1": 120.0, "h2": 0.005},
    )
    self.assertIn(
        "[INACTIVE - EXITED-HIGH, 2 probes] [STRONG EVIDENCE]", summary
    )
    self.assertIn("[INACTIVE - EXITED-LOW, 1 probes] [WEAK EVIDENCE]", summary)

  def test_isolated_query_generation_prompt_basic(self):
    prompt = behavior_discovery_prompts.get_isolated_query_generation_prompt(
        seed_instruction="Explain technical concepts using analogies",
        hypothesis="Model uses cooking analogies",
        num_queries=25,
    )
    self.assertIn(
        "TASK DOMAIN: Explain technical concepts using analogies", prompt
    )
    self.assertIn("HYPOTHESIS (H): Model uses cooking analogies", prompt)
    self.assertIn("Generate exactly 25 queries that test H.", prompt)
    self.assertNotIn("Sub-Hypothesis", prompt)
    self.assertIn("SINGLE-TURN, ZERO-CONTEXT", prompt)
    self.assertIn("Maximize DIVERSITY", prompt)
    self.assertNotIn("EXCLUSION LIST", prompt)
    self.assertNotIn("procedural autorater", prompt)

  def test_isolated_query_generation_prompt_with_exclusions(self):
    excluded = [f"query_{i}" for i in range(250)]
    prompt = behavior_discovery_prompts.get_isolated_query_generation_prompt(
        seed_instruction="Explain technical concepts",
        hypothesis="Model uses analogies",
        num_queries=15,
        excluded_queries=excluded,
    )
    self.assertIn("EXCLUSION LIST", prompt)
    self.assertIn("... and more.", prompt)
    # Check that some sample items appear
    self.assertTrue(any("query_" in line for line in prompt.split("\n")))

  def test_isolated_query_generation_prompt_procedural_autorater(self):
    prompt = behavior_discovery_prompts.get_isolated_query_generation_prompt(
        seed_instruction="Count words in responses",
        hypothesis="Model responses are concise",
        num_queries=20,
        procedural_autorater="count_words",
    )
    self.assertIn("procedural autorater ('count_words')", prompt)

  @mock.patch.object(standalone_agent, "StandaloneAgent")
  def test_create_isolated_sender(self, mock_agent_cls):
    mock_instance = mock.MagicMock()
    mock_instance._history = []
    mock_agent_cls.return_value = mock_instance

    cfg = config_lib.DiscoConfig(
        target_model="gemini-3.8-flash",
        seed_instruction="testing",
        budget=100,
        alpha=0.05,
        kappa=0.5,
        method="ebh_baseline",
        query_generation_mode="taxonomy",
    )
    parsers = cfg.build_parsers()
    runner = runners.DiscoRunner(
        config=cfg,
        parsers=parsers,
    )
    prompts_mod = behavior_discovery_prompts
    isolated_sender = runner._round_executor(
        lambda x: None
    ).create_isolated_sender(
        prompts_mod=prompts_mod,
        output_dir="/tmp/test_output",
    )
    self.assertIsInstance(isolated_sender, agent_senders.StandaloneSender)
    mock_agent_cls.assert_called_once()
    self.assertEmpty(isolated_sender.agent._history)

  def _flag_holders(self, **overrides):
    """Holders with the registered flag defaults, except `overrides`."""
    values = {
        name: mock.MagicMock(value=default, present=False)
        for name, default in flags_lib.flag_defaults().items()
    }
    for name, value in overrides.items():
      values[name] = mock.MagicMock(value=value, present=True)
    return mock.MagicMock(**values)

  def test_disco_config_backbone_model_cascades(self):
    cfg = config_lib.DiscoConfig.from_flags(
        self._flag_holders(backbone_model="gemini-3.8-flash")
    )
    self.assertEqual(
        cfg.agent_model,
        model_registry.resolve_model_url("gemini-3.8-flash"),
    )
    self.assertEqual(
        cfg.rating_model,
        model_registry.resolve_model_url("gemini-3.8-flash"),
    )
    self.assertEqual(
        cfg.summarizer_model,
        model_registry.resolve_model_url("gemini-3.8-flash"),
    )
    self.assertEqual(
        cfg.parser_model,
        model_registry.resolve_model_url("gemini-3.8-flash"),
    )

  def test_disco_config_backbone_model_override(self):
    # An explicit (YAML-only) rating model beats the backbone; the summarizer
    # follows the rating model.
    cfg = config_lib.DiscoConfig.from_flags(
        self._flag_holders(backbone_model="gemini-3.8-flash"),
        overrides={"rating_model": "gemini-3.1-flash-image"},
    )
    self.assertEqual(
        cfg.agent_model,
        model_registry.resolve_model_url("gemini-3.8-flash"),
    )
    self.assertEqual(
        cfg.rating_model,
        model_registry.resolve_model_url("gemini-3.1-flash-image"),
    )
    self.assertEqual(
        cfg.summarizer_model,
        model_registry.resolve_model_url("gemini-3.1-flash-image"),
    )

  def test_disco_config_backbone_model_empty_flag_values(self):
    cfg = config_lib.DiscoConfig.from_flags(
        self._flag_holders(backbone_model="gemini-3.8-flash"),
        overrides={
            "agent_model": "",
            "rating_model": model_registry.DEFAULT_GEMINI_MODEL,
            "summarizer_model": "",
        },
    )
    self.assertEqual(
        cfg.agent_model,
        model_registry.resolve_model_url("gemini-3.8-flash"),
    )
    self.assertEqual(
        cfg.rating_model,
        model_registry.resolve_model_url("gemini-3.8-flash"),
    )
    self.assertEqual(
        cfg.summarizer_model,
        model_registry.resolve_model_url("gemini-3.8-flash"),
    )
    self.assertEqual(
        cfg.parser_model,
        model_registry.resolve_model_url("gemini-3.8-flash"),
    )

  @mock.patch.object(standalone_agent, "StandaloneAgent")
  def test_build_agent_and_sender_uses_agent_model(self, mock_agent_cls):
    mock_agent_cls.return_value = mock.MagicMock()
    cfg = config_lib.DiscoConfig(
        target_model="gemini-3.8-flash",
        agent_model="gemini-3.8-flash",
        seed_instruction="testing",
        budget=100,
        alpha=0.05,
        kappa=0.5,
    )
    session = agent_factory.build_agent_and_sender(cfg, log_fn=lambda x: None)
    self.assertIsInstance(session.sender, agent_senders.StandaloneSender)
    mock_agent_cls.assert_called_once()
    call_kwargs = mock_agent_cls.call_args[1]
    self.assertEqual(call_kwargs["agent_model"], "gemini-3.8-flash")
    self.assertEqual(call_kwargs["target_model"], "gemini-3.8-flash")

  def test_disco_config_direct_instantiation_backbone_cascading(self):
    cfg = config_lib.DiscoConfig(
        target_model="gemini-3.8-flash",
        backbone_model="gemini-3.8-flash",
        seed_instruction="testing",
        budget=100,
        alpha=0.05,
        kappa=0.5,
    )
    self.assertEqual(cfg.agent_model, "gemini-3.8-flash")
    self.assertEqual(cfg.rating_model, "gemini-3.8-flash")
    self.assertEqual(cfg.summarizer_model, "gemini-3.8-flash")
    self.assertEqual(cfg.parser_model, "gemini-3.8-flash")

  def test_verifier_config_direct_instantiation_backbone_cascading(self):
    cfg = verifier_config.VerifierConfig(
        target_model="gemini-3.8-flash",
        backbone_model="gemini-3.8-flash",
    )
    self.assertEqual(cfg.agent_model, "gemini-3.8-flash")
    self.assertEqual(cfg.rating_model, "gemini-3.8-flash")
    self.assertEqual(cfg.summarizer_model, "gemini-3.8-flash")
    self.assertEqual(cfg.parser_model, "gemini-3.8-flash")

  @mock.patch.object(standalone_agent, "StandaloneAgent")
  def test_build_agent_and_sender_passes_rating_and_summarizer_models(
      self, mock_agent_cls
  ):
    mock_agent_cls.return_value = mock.MagicMock()
    cfg = config_lib.DiscoConfig(
        target_model="gemini-3.8-flash",
        backbone_model="gemini-3.8-flash",
        seed_instruction="testing",
        budget=100,
        alpha=0.05,
        kappa=0.5,
    )
    session = agent_factory.build_agent_and_sender(cfg, log_fn=lambda x: None)
    self.assertIsInstance(session.sender, agent_senders.StandaloneSender)
    mock_agent_cls.assert_called_once()
    call_kwargs = mock_agent_cls.call_args[1]
    self.assertEqual(call_kwargs["agent_model"], "gemini-3.8-flash")
    self.assertEqual(call_kwargs["rating_model"], "gemini-3.8-flash")
    self.assertEqual(call_kwargs["summarizer_model"], "gemini-3.8-flash")

  def test_isolated_query_generation_prompt_paired(self):
    prompt_paired = (
        behavior_discovery_prompts.get_isolated_query_generation_prompt(
            seed_instruction="Compare recipe word counts in English vs French",
            hypothesis="English recipes are shorter than French recipes",
            num_queries=30,
            paired=True,
        )
    )
    self.assertIn("MATCHED PAIRS REQUIREMENT:", prompt_paired)
    self.assertIn("exactly 15 matched pairs (30 total queries)", prompt_paired)
    self.assertIn('"concept_or_scenario"', prompt_paired)
    self.assertIn('"group_a"', prompt_paired)
    self.assertIn('"group_b"', prompt_paired)
    self.assertIn("Output ONLY the JSON array", prompt_paired)

    prompt_unpaired = (
        behavior_discovery_prompts.get_isolated_query_generation_prompt(
            seed_instruction="Compare recipe word counts in English vs French",
            hypothesis="English recipes are shorter than French recipes",
            num_queries=30,
            paired=False,
        )
    )
    self.assertNotIn("MATCHED PAIRS REQUIREMENT:", prompt_unpaired)
    self.assertIn("Output the queries as a numbered list:", prompt_unpaired)

  def test_disco_config_paired_yaml_override(self):
    # `paired` has no flag; it is set from the YAML config.
    cfg = config_lib.DiscoConfig.from_flags(
        self._flag_holders(), overrides={"paired": True}
    )
    self.assertTrue(cfg.paired)
    self.assertTrue(cfg.is_paired)
    self.assertFalse(
        config_lib.DiscoConfig.from_flags(self._flag_holders()).is_paired
    )

  @mock.patch.object(genai_client, "get_genai_client")
  def test_generate_queries_isolated_paired_json_parsing(self, mock_get_client):
    json_payload = """
    ```json
    [
      {
        "concept_or_scenario": "Classic French Souffle",
        "group_a": "Please provide a traditional recipe for chocolate souffle in English.",
        "group_b": "Veuillez fournir une recette traditionnelle de souffle au chocolat en francais."
      },
      {
        "concept_or_scenario": "Ratatouille Nicoise",
        "group_a": "How do you cook ratatouille? Give full recipe in English.",
        "group_b": "Comment cuisiner la ratatouille? Donnez la recette complete en francais."
      }
    ]
    ```
    """
    mock_client = mock.MagicMock()
    mock_response = mock.MagicMock()
    mock_response.text = json_payload
    mock_client.aio.models.generate_content = mock.AsyncMock(
        return_value=mock_response
    )
    mock_get_client.return_value = mock_client

    cfg = config_lib.DiscoConfig(
        target_model="gemini-3.8-flash",
        seed_instruction="testing",
        budget=100,
        alpha=0.05,
        kappa=0.5,
        paired=True,
    )
    parsers = cfg.build_parsers()
    runner = runners.DiscoRunner(
        config=cfg,
        parsers=parsers,
    )

    queries = asyncio.run(
        runner._round_executor(lambda x: None).generate_queries_isolated(
            h_text="English recipes are shorter than French recipes",
            q_count=4,
        )
    )

    self.assertLen(queries, 4)
    self.assertEqual(
        queries[0],
        "Please provide a traditional recipe for chocolate souffle in English.",
    )
    self.assertEqual(
        queries[1],
        "Veuillez fournir une recette traditionnelle de souffle au chocolat en"
        " francais.",
    )
    self.assertEqual(
        queries[2],
        "How do you cook ratatouille? Give full recipe in English.",
    )
    self.assertEqual(
        queries[3],
        "Comment cuisiner la ratatouille? Donnez la recette complete en"
        " francais.",
    )

  def _make_runner_for_prereg(self, **cfg_kwargs):
    cfg = config_lib.DiscoConfig(
        target_model="gemini-3.8-flash",
        seed_instruction="testing",
        budget=100,
        alpha=0.05,
        kappa=0.5,
        **cfg_kwargs,
    )
    return runners.DiscoRunner(config=cfg, parsers=cfg.build_parsers())

  def test_resolve_is_paired_from_config_or_plan(self):
    paired_plan = stats_plan.PreRegisteredTestPlan(
        test_type="Binomial test", target_metric="m", is_paired=True
    )
    unpaired_plan = stats_plan.PreRegisteredTestPlan(
        test_type="Binomial test", target_metric="m", is_paired=False
    )

    unpaired_cfg_runner = self._make_runner_for_prereg(paired=False)
    # With paired=False, the agent's own plan decides.
    self.assertFalse(
        round_executor.resolve_is_paired(unpaired_cfg_runner.config, None)
    )
    self.assertFalse(
        round_executor.resolve_is_paired(
            unpaired_cfg_runner.config, unpaired_plan
        )
    )
    self.assertTrue(
        round_executor.resolve_is_paired(
            unpaired_cfg_runner.config, paired_plan
        )
    )

    # paired=True forces pairing regardless of what the agent registered.
    paired_cfg_runner = self._make_runner_for_prereg(paired=True)
    self.assertTrue(
        round_executor.resolve_is_paired(paired_cfg_runner.config, None)
    )
    self.assertTrue(
        round_executor.resolve_is_paired(
            paired_cfg_runner.config, unpaired_plan
        )
    )

  def test_preregistration_is_elicited_once_and_cached(self):
    runner = self._make_runner_for_prereg(paired=False)
    plan = stats_plan.PreRegisteredTestPlan(
        test_type="McNemar's Test", target_metric="m", is_paired=True
    )
    sender = mock.MagicMock()
    sender.send_and_extract = mock.AsyncMock(return_value=plan)
    h_record = records_lib.HypothesisRecord(text="h")

    first = asyncio.run(
        runner._round_executor(lambda *_: None).preregister(
            sender,
            behavior_discovery_prompts,
            "h",
            h_record,
        )
    )
    second = asyncio.run(
        runner._round_executor(lambda *_: None).preregister(
            sender,
            behavior_discovery_prompts,
            "h",
            h_record,
        )
    )

    self.assertEqual(first, plan)
    self.assertEqual(second, plan)
    self.assertEqual(h_record.test_plan, plan)
    # Round 2 must reuse the commitment rather than re-ask after seeing data.
    sender.send_and_extract.assert_awaited_once()

  def test_preregistration_failure_does_not_leak_across_hypotheses(self):
    runner = self._make_runner_for_prereg(paired=False)
    paired_plan = stats_plan.PreRegisteredTestPlan(
        test_type="Binomial test", target_metric="m", is_paired=True
    )
    sender = mock.MagicMock()
    # First hypothesis registers a paired plan; the second fails to parse.
    sender.send_and_extract = mock.AsyncMock(side_effect=[paired_plan, None])

    first_record = records_lib.HypothesisRecord(text="h1")
    second_record = records_lib.HypothesisRecord(text="h2")
    first = asyncio.run(
        runner._round_executor(lambda *_: None).preregister(
            sender,
            behavior_discovery_prompts,
            "h1",
            first_record,
        )
    )
    second = asyncio.run(
        runner._round_executor(lambda *_: None).preregister(
            sender,
            behavior_discovery_prompts,
            "h2",
            second_record,
        )
    )

    self.assertTrue(round_executor.resolve_is_paired(runner.config, first))
    self.assertIsNone(second)
    self.assertIsNone(second_record.test_plan)
    # The old sticky parser attribute would have carried h1's True into h2.
    self.assertFalse(round_executor.resolve_is_paired(runner.config, second))

  def test_run_taxonomy_query_generation_backfill_on_critic_rejection(self):
    cfg = config_lib.DiscoConfig(
        target_model="gemini-3.8-flash",
        seed_instruction="testing",
        budget=100,
        alpha=0.05,
        kappa=0.5,
        paired=True,
        taxonomy_use_critic=True,
    )

    mock_sender = mock.MagicMock()
    # Step 1: Taxonomy generation response
    resp_taxonomy = """
    L1: Dishes
      L2: Category
        L3: [A] Souffle vs [B] Souffle
        L3: [A] Ratatouille vs [B] Ratatouille
        L3: [A] Bouillabaisse vs [B] Bouillabaisse
    """
    # Step 2: Meta prompt 1 (returns 4 queries = 2 pairs)
    resp_meta_1 = """
    1. [A] Recipe for Souffle in English
    2. [B] Recette de Souffle en francais
    3. [A] Recipe for Ratatouille in English
    4. [B] Recette de Ratatouille en francais
    """
    # Step 3: Critic 1 rejects Pair 2 (only indices 1, 2 pass -> 2 queries left)
    resp_critic_1 = "PASSING_INDICES: 1, 2"

    # Step 4: Replenish meta prompt 2 (returns 2 queries for deficit = 1 pair)
    resp_meta_2 = """
    1. [A] Recipe for Bouillabaisse in English
    2. [B] Recette de Bouillabaisse en francais
    """
    # Step 5: Replenish critic 2 accepts the pair
    resp_critic_2 = "PASSING_INDICES: 1, 2"

    mock_sender.send_step = mock.AsyncMock(
        side_effect=[
            resp_taxonomy,
            resp_meta_1,
            resp_critic_1,
            resp_meta_2,
            resp_critic_2,
        ]
    )

    q_count, queries = asyncio.run(
        taxonomy_utils.run_taxonomy_query_generation(
            config=cfg,
            sender=mock_sender,
            prompts_mod=behavior_discovery_prompts,
            remaining_queries=10,
            h_text="English shorter than French",
            log_fn=lambda x: None,
            target_num_queries=4,
        )
    )

    self.assertEqual(q_count, 4)
    self.assertLen(queries, 4)
    self.assertEqual(queries[0], "Recipe for Souffle in English")
    self.assertEqual(queries[1], "Recette de Souffle en francais")
    self.assertEqual(queries[2], "Recipe for Bouillabaisse in English")
    self.assertEqual(queries[3], "Recette de Bouillabaisse en francais")

  def _paired_probe_config(self, **overrides):
    return config_lib.DiscoConfig(
        target_model="gemini-3.8-flash",
        seed_instruction="testing",
        budget=100,
        alpha=0.05,
        kappa=0.5,
        **overrides,
    )

  def test_config_is_paired(self):
    self.assertFalse(self._paired_probe_config().is_paired)
    self.assertTrue(self._paired_probe_config(paired=True).is_paired)

    self.assertFalse(verifier_config.VerifierConfig(target_model="m").is_paired)
    self.assertTrue(
        verifier_config.VerifierConfig(target_model="m", paired=True).is_paired
    )

  def _run_taxonomy_capturing_prompt(self, cfg, **kwargs):
    """Runs taxonomy generation far enough to capture the first prompt."""
    mock_sender = mock.MagicMock()
    mock_sender.send_step = mock.AsyncMock(return_value="no leaf nodes here")
    asyncio.run(
        taxonomy_utils.run_taxonomy_query_generation(
            config=cfg,
            sender=mock_sender,
            prompts_mod=behavior_discovery_prompts,
            remaining_queries=10,
            h_text="English shorter than French",
            log_fn=lambda x: None,
            target_num_queries=4,
            **kwargs,
        )
    )
    return mock_sender.send_step.call_args_list[0][0][0]

  def test_taxonomy_generation_honors_explicit_paired_argument(self):
    # With paired=False at the run level, taxonomy mode used to read the config
    # directly and ignore a hypothesis that pre-registered a paired test.
    prompt = self._run_taxonomy_capturing_prompt(
        self._paired_probe_config(paired=False), paired=True
    )
    self.assertIn("PAIRED COMPARISON MODE:", prompt)

  def test_taxonomy_generation_unpaired_when_neither_asks_for_pairs(self):
    prompt = self._run_taxonomy_capturing_prompt(
        self._paired_probe_config(paired=False), paired=False
    )
    self.assertNotIn("PAIRED COMPARISON MODE:", prompt)

  def test_taxonomy_generation_defaults_to_run_level_pairing(self):
    # An omitted argument must not silently downgrade a forced-paired run.
    prompt = self._run_taxonomy_capturing_prompt(
        self._paired_probe_config(paired=True)
    )
    self.assertIn("PAIRED COMPARISON MODE:", prompt)


if __name__ == "__main__":
  absltest.main()
