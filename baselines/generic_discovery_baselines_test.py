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

"""Tests for generic discovery baselines and task adapters."""

import asyncio
import dataclasses
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver

from disco.baselines import common
from disco.baselines import eval_discovery_baselines
from disco.baselines import recipe_categories_pool
from disco.baselines.adapters import hedging as adapters_hedging
from disco.baselines.adapters import recipe as adapters_recipe
from disco.baselines.adapters import registry as adapters_registry
from disco.baselines.adapters import visual_symmetry as adapters_visual_symmetry
from disco.baselines.methods import base as methods_base
from disco.baselines.methods import looped
from disco.baselines.methods import red_team
from disco.baselines.methods import simple_probing
from disco.models import genai_client
from disco.tasks import data_classes
from disco.tasks import hedging_tasks
from disco.tasks import recipe_tasks
from disco.tasks.visual_symmetry import hypotheses as visual_symmetry_hypotheses
from disco.tasks.visual_symmetry import image_gen as visual_symmetry_image_gen


class GenericDiscoveryBaselinesTest(absltest.TestCase):

  def test_get_task_adapter_selection(self):
    vis_task = visual_symmetry_hypotheses.VISUAL_SYMMETRY_TASK
    vis_adapter = adapters_registry.get_task_adapter(vis_task)
    self.assertIsInstance(
        vis_adapter, adapters_visual_symmetry.VisualSymmetryTaskAdapter
    )
    self.assertTrue(vis_adapter.is_multimodal)

    recipe_task = recipe_tasks.RECIPE_TRANSLATE_FRENCH_LONG_TASK
    recipe_adapter = adapters_registry.get_task_adapter(recipe_task)
    self.assertIsInstance(recipe_adapter, adapters_recipe.RecipeTaskAdapter)
    self.assertFalse(recipe_adapter.is_multimodal)

    hedging_task = hedging_tasks.HEDGING_UNHEDGED_CONTROVERSIAL_TASK
    hedging_adapter = adapters_registry.get_task_adapter(hedging_task)
    self.assertIsInstance(hedging_adapter, adapters_hedging.HedgingTaskAdapter)
    self.assertFalse(hedging_adapter.is_multimodal)

  def test_get_task_adapter_dispatches_on_adapter_field(self):
    task = data_classes.Task(
        name="Recipe Hedging Probe",
        description="d",
        suite="hedging",
        adapter="hedging",
    )
    self.assertIsInstance(
        adapters_registry.get_task_adapter(task),
        adapters_hedging.HedgingTaskAdapter,
    )
    with self.assertRaisesRegex(ValueError, "sets no adapter"):
      adapters_registry.get_task_adapter(
          dataclasses.replace(task, adapter=None)
      )
    with self.assertRaisesRegex(ValueError, "unknown adapter"):
      adapters_registry.get_task_adapter(
          dataclasses.replace(task, adapter="nope")
      )

  def test_adapter_probe_categories(self):
    vis_task = visual_symmetry_hypotheses.VISUAL_SYMMETRY_TASK
    vis_adapter = adapters_registry.get_task_adapter(vis_task)
    vis_cats = vis_adapter.get_fixed_probe_categories()
    # The baseline arm must be able to reach every probe the DISCO arm can.
    # If DISCO explores the harder families and the ten backgrounds and the
    # baselines cannot, the comparison measures the probe menu, not the method.
    expected_shapes = {
        s
        for s in visual_symmetry_image_gen.list_shape_types()
        if s != "dynamic"
    }
    expected_bgs = set(visual_symmetry_image_gen.BACKGROUND_PALETTE)
    self.assertEqual({c["shape_type"] for c in vis_cats}, expected_shapes)
    self.assertEqual({c["bg_color"] for c in vis_cats}, expected_bgs)
    self.assertLen(vis_cats, len(expected_shapes) * len(expected_bgs))
    # Uniform probing groups configs by shape_type, so the background variants
    # must cycle *within* a category rather than fragmenting the budget.
    self.assertLen(
        [c for c in vis_cats if c["shape_type"] == "english_letter"],
        len(expected_bgs),
    )

    recipe_task = recipe_tasks.RECIPE_TRANSLATE_FRENCH_LONG_TASK
    recipe_adapter = adapters_registry.get_task_adapter(recipe_task)
    recipe_cats = recipe_adapter.get_fixed_probe_categories()
    self.assertLen(recipe_cats, 400)
    self.assertIn("dish_en", recipe_cats[0])
    self.assertIn("dish_fr", recipe_cats[0])
    self.assertEqual(recipe_cats[0]["category"], "braised_meats")

  def test_symmetry_claims_state_the_task_null_in_both_directions(self):
    """Baseline claim text must quote the task's null proportion."""
    vis_task = visual_symmetry_hypotheses.VISUAL_SYMMETRY_TASK
    adapter = adapters_registry.get_task_adapter(vis_task)

    task_null = float(vis_task.metadata["null_proportion"])
    self.assertAlmostEqual(adapter.task_null_proportion, task_null)
    pct = f"{adapter.task_null_pct}%"

    # One rate either side of the null, so both branches are exercised.
    preds = adapter.format_empirical_predictions({
        "complex_chinese": task_null - 0.30,
        "square": min(task_null + 0.05, 1.0),
    })
    self.assertLen(preds, 2)
    for p in preds:
      self.assertIn(pct, p)
      self.assertNotIn("75%", p)
      self.assertNotIn("50%", p)

    # The direction must follow the null, not a separate hardcoded threshold.
    self.assertIn("struggles", preds[0])
    self.assertIn("can perform", preds[1])

    # The rate the baseline interrogates the model about is the same one.
    self.assertTrue(
        any(pct in q for q in adapter.get_direct_interrogation_prompts()),
        "direct interrogation must ask about the task's null rate",
    )

    # And the structured hypotheses handed to the verifier carry it too.
    hyps = common.build_structured_hypotheses_from_preds(
        preds, adapter.suite, adapter.task_null_proportion
    )
    for h in hyps:
      self.assertAlmostEqual(h.null_proportion, task_null)

  def test_symmetry_probe_menu_offered_to_llm_baselines_is_complete(self):
    """The red-team prompt must offer every family and background."""
    vis_task = visual_symmetry_hypotheses.VISUAL_SYMMETRY_TASK
    adapter = adapters_registry.get_task_adapter(vis_task)
    prompt = adapter.get_red_team_system_prompt()
    for shape in visual_symmetry_image_gen.list_shape_types():
      if shape == "dynamic":
        continue
      self.assertIn(shape, prompt, f"{shape} missing from red-team menu")
    for bg in visual_symmetry_image_gen.BACKGROUND_PALETTE:
      self.assertIn(bg, prompt, f"{bg} missing from red-team menu")

  def test_llm_baseline_prompts_carry_the_shared_investigation_brief(self):
    """Every LLM-driven baseline must include the shared task seed brief."""
    vis_task = visual_symmetry_hypotheses.VISUAL_SYMMETRY_TASK
    adapter = adapters_registry.get_task_adapter(vis_task)
    seed = visual_symmetry_hypotheses.DEFAULT_SEED_INSTRUCTION

    for label, prompt in (
        ("red_team", adapter.get_red_team_system_prompt()),
        ("looped", adapter.get_looped_investigator_system_prompt()),
    ):
      self.assertIn("INVESTIGATION BRIEF", prompt, f"{label}: no brief")
      self.assertIn(seed.strip(), prompt, f"{label}: seed not verbatim")
      # The rate is what the discovered hypotheses are tested against (spelled
      # "90 percent" in the seed brief).
      self.assertIn(
          "90 percent", prompt, f"{label}: null rate absent from brief"
      )
      # The seed tells the agent to "use" tools the baselines do not have.
      self.assertIn("you do not call", prompt, f"{label}: no tool disclaimer")

  def test_investigation_brief_is_absent_when_there_is_no_seed(self):
    """Anti-tautology guard for the test above.

    Without this, a brief that were somehow always present -- or an assertion
    matching on text the prompt already contained -- would pass silently. This
    pins that the brief is genuinely seed-driven and that suites with no seed
    are byte-for-byte unaffected.
    """
    seedless = dataclasses.replace(
        visual_symmetry_hypotheses.VISUAL_SYMMETRY_TASK,
        metadata={
            **visual_symmetry_hypotheses.VISUAL_SYMMETRY_TASK.metadata,
            "seed_instruction": "",
        },
    )
    adapter = adapters_registry.get_task_adapter(seedless, seed_instruction="")
    self.assertEqual(adapter.seed_instruction, "")
    for label, prompt in (
        ("red_team", adapter.get_red_team_system_prompt()),
        ("looped", adapter.get_looped_investigator_system_prompt()),
    ):
      self.assertNotIn(
          "INVESTIGATION BRIEF", prompt, f"{label}: spurious brief"
      )

  def test_symmetry_hard_shape_claims_resolve_to_a_shape_type(self):
    """A claim naming a hard family must carry that family in its condition.

    Without it the verifier has no `shape_type` to draw probes for, so the
    hypothesis is silently untestable rather than failing loudly.
    """
    hard = ["complex_chinese", "multi_glyph", "snowflake", "fine_noise"]
    preds = [
        "The target model struggles with vertical symmetry completion on"
        f" {s} shapes on 1:1 images, succeeding less than 90% of the time."
        for s in hard
    ]
    hyps = common.build_structured_hypotheses_from_preds(
        preds, "visual_symmetry", 0.90
    )
    self.assertEqual(
        [h.condition.get("shape_type") for h in hyps],
        hard,
    )

  def test_parse_predictions(self):
    text = (
        "<predictions>\n"
        "1. English recipes are shorter than French recipes for soups"
        " >= 10% of the time.\n"
        "2. English recipes are longer than French recipes for baked"
        " desserts >= 60% of the time.\n"
        "</predictions>"
    )
    preds = common.parse_predictions(text)
    self.assertLen(preds, 2)
    self.assertIn("soups", preds[0])
    self.assertIn("baked desserts", preds[1])

  def test_format_empirical_predictions(self):
    # Reverse task (French longer, null=10%)
    rev_task = recipe_tasks.RECIPE_TRANSLATE_FRENCH_LONG_TASK
    rev_adapter = adapters_registry.get_task_adapter(rev_task)
    rates = {"soups": 0.25, "baked_desserts": 0.05}
    preds = rev_adapter.format_empirical_predictions(rates)
    # Only soups (25% >= 10%) should produce a prediction
    self.assertLen(preds, 1)
    self.assertIn("French recipe responses are longer", preds[0])
    self.assertIn("10%", preds[0])
    self.assertIn("soups", preds[0])
    # Ensure no inverted >= 90% hypothesis is generated for baked_desserts
    for p in preds:
      self.assertNotIn("90%", p)
      self.assertNotIn("baked_desserts", p)

    # Test snake_case normalization when meeting threshold
    rates_with_qualifying_desserts = {"baked_desserts": 0.20}
    dessert_preds = rev_adapter.format_empirical_predictions(
        rates_with_qualifying_desserts
    )
    self.assertLen(dessert_preds, 1)
    self.assertIn("baked desserts", dessert_preds[0])
    self.assertNotIn("baked_desserts", dessert_preds[0])

    # Forward task (English longer, null=60%)
    fwd_task = recipe_tasks.RECIPE_TRANSLATE_ENGLISH_LONG_TASK
    fwd_adapter = adapters_registry.get_task_adapter(fwd_task)
    fwd_rates = {"soups": 0.75, "baked_desserts": 0.40}
    fwd_preds = fwd_adapter.format_empirical_predictions(fwd_rates)
    self.assertLen(fwd_preds, 2)
    self.assertIn("English recipe responses are longer", fwd_preds[0])
    self.assertIn("60%", fwd_preds[0])
    self.assertIn("French recipe responses are longer", fwd_preds[1])
    self.assertIn("40%", fwd_preds[1])
    self.assertIn("baked desserts", fwd_preds[1])
    self.assertNotIn("baked_desserts", fwd_preds[1])

  def test_generate_content_with_retry_success(self):
    async def _test():
      mock_client = mock.MagicMock()
      mock_response = mock.MagicMock()
      mock_response.text = "Success response"
      mock_client.aio.models.generate_content = mock.AsyncMock(
          return_value=mock_response
      )

      res = await common.generate_content_with_retry(
          client=mock_client,
          model="models/gemini-3.8-flash",
          contents="Hello",
          max_retries=3,
          base_delay=0.01,
      )
      self.assertEqual(res.text, "Success response")
      self.assertEqual(mock_client.aio.models.generate_content.call_count, 1)

    asyncio.run(_test())

  def test_generate_content_with_retry_recovers_after_failure(self):
    async def _test():
      mock_client = mock.MagicMock()
      mock_response = mock.MagicMock()
      mock_response.text = "Recovered response"

      calls = 0

      async def _fake_generate(*args, **kwargs):
        del args, kwargs
        nonlocal calls
        calls += 1
        if calls < 2:
          raise RuntimeError("Simulated transient 429 quota error")
        return mock_response

      mock_client.aio.models.generate_content = mock.AsyncMock(
          side_effect=_fake_generate
      )

      res = await common.generate_content_with_retry(
          client=mock_client,
          model="models/gemini-3.8-flash",
          contents="Hello",
          max_retries=3,
          base_delay=0.01,
      )
      self.assertEqual(res.text, "Recovered response")
      self.assertEqual(calls, 2)

    asyncio.run(_test())

  def test_async_recipe_adapter_execute_probe(self):
    async def _test():
      recipe_task = recipe_tasks.RECIPE_TRANSLATE_FRENCH_LONG_TASK
      adapter = adapters_registry.get_task_adapter(recipe_task)

      mock_client = mock.MagicMock()
      en_response = mock.MagicMock()
      en_response.text = "Short recipe step one step two."
      fr_response = mock.MagicMock()
      fr_response.text = (
          "Une tres longue recette francaise avec beaucoup d ingredients et d"
          " etapes detaillees pour ce plat delicieux."
      )

      async def _fake_generate(model, contents, config=None):
        del model, config
        if "Donnez-moi" in contents:
          return fr_response
        return en_response

      mock_client.aio.models.generate_content = mock.AsyncMock(
          side_effect=_fake_generate
      )

      obs = await adapter.execute_probe(
          client=mock_client,
          target_model="models/gemini-3.8-flash",
          probe_config={
              "dish_en": "French onion soup",
              "dish_fr": "soupe à l'oignon",
              "category": "soups",
          },
          round_idx=1,
          probe_idx=1,
      )
      self.assertTrue(obs.passed)
      self.assertEqual(obs.category, "soups")
      self.assertEqual(obs.score, 1.0)
      self.assertEqual(obs.details["dish_en"], "French onion soup")
      self.assertEqual(obs.details["dish_fr"], "soupe à l'oignon")

    asyncio.run(_test())

  def test_async_simple_probing_baseline(self):
    async def _test():
      recipe_task = recipe_tasks.RECIPE_TRANSLATE_FRENCH_LONG_TASK
      adapter = adapters_registry.get_task_adapter(recipe_task)

      mock_client = mock.MagicMock()
      mock_response = mock.MagicMock()
      mock_response.text = "Recipe ingredients instructions details."
      mock_client.aio.models.generate_content = mock.AsyncMock(
          return_value=mock_response
      )

      res = await simple_probing.run_simple_probing_baseline(
          adapter=adapter,
          client=mock_client,
          target_model="models/gemini-3.8-flash",
          budget=4,
      )
      self.assertEqual(res.method_name, "SimpleProbingBaseline")
      self.assertEqual(res.total_target_queries, 4)
      self.assertLen(res.observations, 4)

    asyncio.run(_test())

  def test_async_simple_probing_stratified_seed_sampling(self):
    async def _test():
      recipe_task = recipe_tasks.RECIPE_TRANSLATE_FRENCH_LONG_TASK
      adapter = adapters_registry.get_task_adapter(recipe_task)

      mock_client = mock.MagicMock()
      mock_response = mock.MagicMock()
      mock_response.text = "Recipe instructions."
      mock_client.aio.models.generate_content = mock.AsyncMock(
          return_value=mock_response
      )

      # Run with seed=42, budget=25 (25 categories -> 1 dish per category)
      res1 = await simple_probing.run_simple_probing_baseline(
          adapter=adapter,
          client=mock_client,
          target_model="models/gemini-3.8-flash",
          budget=25,
          seed=42,
      )
      # Run with seed=43, budget=25
      res2 = await simple_probing.run_simple_probing_baseline(
          adapter=adapter,
          client=mock_client,
          target_model="models/gemini-3.8-flash",
          budget=25,
          seed=43,
      )

      cats1 = {o.category for o in res1.observations}
      cats2 = {o.category for o in res2.observations}
      self.assertLen(cats1, 25)
      self.assertLen(cats2, 25)

      dishes1 = [o.details.get("dish_en") for o in res1.observations]
      dishes2 = [o.details.get("dish_en") for o in res2.observations]
      # The randomized selection should produce different dish orderings/samples
      self.assertNotEqual(dishes1, dishes2)

    asyncio.run(_test())

  def test_adapter_seed_instruction(self):
    recipe_task = recipe_tasks.RECIPE_TRANSLATE_FRENCH_LONG_TASK
    # Default seed_instruction from metadata
    adapter = adapters_registry.get_task_adapter(recipe_task)
    self.assertIn("French", adapter.seed_instruction)

    # Explicit override
    adapter_custom = adapters_registry.get_task_adapter(
        recipe_task,
        seed_instruction="Custom recipe audit seed",
    )
    self.assertEqual(
        adapter_custom.seed_instruction, "Custom recipe audit seed"
    )

  def test_build_baseline_synthesis_prompt(self):
    recipe_task = recipe_tasks.RECIPE_TRANSLATE_FRENCH_LONG_TASK
    adapter = adapters_registry.get_task_adapter(
        recipe_task,
        seed_instruction="Analyze French vs English recipe lengths",
    )
    prompt = methods_base.build_synthesis_prompt(
        adapter=adapter,
        observations_text="Observation 1\nObservation 2",
        baseline_context_header="Here is the research log:",
        max_predictions=8,
    )
    self.assertIn("autonomous behavioral auditor", prompt)
    self.assertIn("Analyze French vs English recipe lengths", prompt)
    self.assertIn("Do NOT reference benchmark codenames", prompt)
    self.assertIn(
        "Do NOT frame natural language tendencies as 'instruction following"
        " failures'",
        prompt,
    )
    self.assertIn(
        "formulate precise, testable behavioral hypothesis predictions", prompt
    )
    self.assertIn("Keep ONLY the top 8", prompt)
    self.assertIn("8. [Least likely of the top predictions]", prompt)
    self.assertIn("Observation 1", prompt)

  def test_adapter_system_prompts(self):
    recipe_task = recipe_tasks.RECIPE_TRANSLATE_FRENCH_LONG_TASK
    adapter = adapters_registry.get_task_adapter(recipe_task)

    self.assertIn(
        "autonomous behavioral auditor", adapter.get_red_team_system_prompt()
    )
    self.assertIn(
        "investigating recipe length",
        adapter.get_looped_investigator_system_prompt(),
    )

  def test_async_recipe_adapter_execute_probe_continuous(self):
    async def _test():
      recipe_task = recipe_tasks.RECIPE_TRANSLATE_FRENCH_LONG_TASK
      adapter = adapters_registry.get_task_adapter(recipe_task)

      mock_client = mock.MagicMock()
      en_response = mock.MagicMock()
      en_response.text = "Short recipe step one step two."
      fr_response = mock.MagicMock()
      fr_response.text = (
          "Une tres longue recette francaise avec beaucoup d ingredients et d"
          " etapes detaillees pour ce plat delicieux."
      )

      async def _fake_generate(model, contents, config=None):
        del model, config
        if "Donnez-moi" in contents:
          return fr_response
        return en_response

      mock_client.aio.models.generate_content = mock.AsyncMock(
          side_effect=_fake_generate
      )

      obs = await adapter.execute_probe(
          client=mock_client,
          target_model="models/gemini-3.8-flash",
          probe_config={
              "dish_en": "French onion soup",
              "dish_fr": "soupe à l'oignon",
              "category": "soups",
          },
          round_idx=1,
          probe_idx=1,
          k_samples=3,
      )
      self.assertTrue(obs.passed)
      self.assertEqual(obs.category, "soups")
      self.assertEqual(obs.score, 1.0)
      self.assertEqual(obs.details["pair_results"], [True, True, True])
      self.assertIn("English words=", obs.response_summary)
      self.assertIn("French words=", obs.response_summary)

    asyncio.run(_test())

  def test_recipe_categories_structure(self):
    cats = recipe_categories_pool.RECIPE_CATEGORIES
    self.assertLen(cats, 400)
    distinct_cats = set()
    cat_dish_counts = {}
    for entry in cats:
      self.assertIn("dish_en", entry)
      self.assertIn("dish_fr", entry)
      self.assertIn("category", entry)
      self.assertTrue(entry["dish_en"])
      self.assertTrue(entry["dish_fr"])
      self.assertTrue(entry["category"])
      c = entry["category"]
      distinct_cats.add(c)
      cat_dish_counts[c] = cat_dish_counts.get(c, 0) + 1

    self.assertLen(distinct_cats, 25)
    for c, count in cat_dish_counts.items():
      self.assertEqual(count, 16, f"Category {c} does not have 16 dishes")

  def test_parse_probe_json_list_resilience(self):
    # 1. Markdown codeblock
    md_text = (
        "Here are your probes:\n```json\n"
        '[{"dish_en": "beef stew", "dish_fr": "ragout", "category":'
        ' "braised_meats"}]\n```'
    )
    res1 = common.parse_probe_json_list(md_text)
    self.assertLen(res1, 1)
    self.assertEqual(res1[0]["dish_en"], "beef stew")

    # 2. Raw JSON array
    raw_json = '[{"dish_en": "bread", "dish_fr": "pain", "category": "breads"}]'
    res2 = common.parse_probe_json_list(raw_json)
    self.assertLen(res2, 1)
    self.assertEqual(res2[0]["dish_en"], "bread")

    # 3. Fallback regex extraction from messy text
    messy = (
        'I recommend {"dish_en": "soup", "dish_fr": "soupe", "category":'
        ' "soups"} and also {"dish_en": "tart", "dish_fr": "tarte",'
        ' "category": "desserts"}.'
    )
    res3 = common.parse_probe_json_list(messy)
    self.assertLen(res3, 2)
    self.assertEqual(res3[0]["dish_en"], "soup")
    self.assertEqual(res3[1]["dish_en"], "tart")

    # 4. Fallback regex extraction for T2I activity / scene keys
    messy_t2i = (
        'Try {"activity": "writing a letter"} and {"scene": "a cat on a sofa"}.'
    )
    res4 = common.parse_probe_json_list(messy_t2i)
    self.assertEqual(
        res4,
        [{"activity": "writing a letter"}, {"scene": "a cat on a sofa"}],
    )

  def test_async_red_team_probing_baseline_batched(self):
    async def _test():
      recipe_task = recipe_tasks.RECIPE_TRANSLATE_FRENCH_LONG_TASK
      adapter = adapters_registry.get_task_adapter(recipe_task)

      mock_client = mock.MagicMock()
      en_response = mock.MagicMock()
      en_response.text = "Short recipe."
      fr_response = mock.MagicMock()
      fr_response.text = "Tres longue recette detaillee francaise."
      batch1_response = mock.MagicMock()
      batch1_response.text = (
          '```json\n[{"dish_en": "Bouillabaisse", "dish_fr": "Bouillabaisse",'
          ' "category": "seafood"}]\n```'
      )
      batch2_response = mock.MagicMock()
      batch2_response.text = (
          '```json\n[{"dish_en": "Consomme", "dish_fr": "Consomme",'
          ' "category": "clear_broths"}]\n```'
      )

      synth_response = mock.MagicMock()
      synth_response.text = (
          "<predictions>\n1. French recipes are longer for seafood >= 10% of"
          " the time.\n</predictions>"
      )

      inv_calls = 0

      async def _fake_generate(model, contents, config=None):
        del model, config
        nonlocal inv_calls
        if isinstance(contents, str) and (
            "<predictions>" in contents or "behavioral auditor" in contents
        ):
          if (
              "adversarial red-team" in contents
              or "Previous probe observations" in contents
          ):
            inv_calls += 1
            return batch2_response if inv_calls > 1 else batch1_response
          return synth_response
        if isinstance(contents, str) and "Donnez-moi" in contents:
          return fr_response
        return en_response

      mock_client.aio.models.generate_content = mock.AsyncMock(
          side_effect=_fake_generate
      )

      res = await red_team.run_red_team_probing_baseline(
          adapter=adapter,
          client=mock_client,
          target_model="models/gemini-3.8-flash",
          investigator_model="models/gemini-3.8-flash",
          budget=2,
          batch_size=1,
          max_predictions=5,
      )
      self.assertEqual(res.method_name, "RedTeamProbingBaseline")
      self.assertEqual(res.total_target_queries, 2)
      self.assertLen(res.observations, 2)
      self.assertGreaterEqual(inv_calls, 2)

    asyncio.run(_test())

  def test_async_looped_investigator_baseline(self):
    async def _test():
      recipe_task = recipe_tasks.RECIPE_TRANSLATE_FRENCH_LONG_TASK
      adapter = adapters_registry.get_task_adapter(recipe_task)

      mock_client = mock.MagicMock()
      en_response = mock.MagicMock()
      en_response.text = "Short recipe."
      fr_response = mock.MagicMock()
      fr_response.text = "Tres longue recette detaillee francaise."
      inv_probe_response = mock.MagicMock()
      inv_probe_response.text = (
          '```json\n[{"dish_en": "Eclair", "dish_fr": "Eclair au chocolat",'
          ' "category": "pastries"}]\n```'
      )
      synth_response = mock.MagicMock()
      synth_response.text = (
          "<predictions>\n1. French recipes are longer for pastries >= 10% of"
          " the time.\n</predictions>"
      )

      async def _fake_generate(model, contents, config=None):
        del model, config
        if isinstance(contents, str) and (
            "<predictions>" in contents or "behavioral auditor" in contents
        ):
          if "Propose" in contents or "follow-up" in contents:
            return inv_probe_response
          return synth_response
        if isinstance(contents, str) and "Donnez-moi" in contents:
          return fr_response
        return en_response

      mock_client.aio.models.generate_content = mock.AsyncMock(
          side_effect=_fake_generate
      )

      res = await looped.run_looped_investigator_baseline(
          adapter=adapter,
          client=mock_client,
          target_model="models/gemini-3.8-flash",
          investigator_model="models/gemini-3.8-flash",
          num_rounds=2,
          probes_per_round=1,
          max_predictions=5,
      )
      self.assertEqual(res.method_name, "LoopedInvestigatorBaseline")
      self.assertEqual(res.total_target_queries, 2)
      self.assertLen(res.observations, 2)

    asyncio.run(_test())

  def test_hedging_adapter_probe_categories(self):
    hedging_task = hedging_tasks.HEDGING_UNHEDGED_CONTROVERSIAL_TASK
    adapter = adapters_registry.get_task_adapter(hedging_task)
    cats = adapter.get_fixed_probe_categories()
    self.assertLen(cats, 400)
    self.assertIn("prompt", cats[0])
    self.assertIn("category", cats[0])
    self.assertEqual(cats[0]["category"], "nutrition_sweeteners")

  def test_hedging_format_empirical_predictions(self):
    hedging_task = hedging_tasks.HEDGING_UNHEDGED_CONTROVERSIAL_TASK
    adapter = adapters_registry.get_task_adapter(hedging_task)
    rates = {"nutrition_sweeteners": 0.45, "history_bronze_age": 0.005}
    preds = adapter.format_empirical_predictions(rates)
    # Only nutrition_sweeteners (45% >= 33%) should produce a prediction
    self.assertLen(preds, 1)
    self.assertIn("nutrition sweeteners", preds[0])
    self.assertIn("33%", preds[0])
    self.assertIn("zero hedging markers", preds[0])

  def test_hedging_execute_probe_unhedged_vs_hedged(self):
    async def _test():
      hedging_task = hedging_tasks.HEDGING_UNHEDGED_CONTROVERSIAL_TASK
      adapter = adapters_registry.get_task_adapter(hedging_task)

      mock_client = mock.MagicMock()

      # Unhedged response (dogmatic statement with 0 hedges -> failure =
      # score 1.0)
      unhedged_resp = mock.MagicMock()
      unhedged_resp.text = (
          "Seed oils cause massive systemic inflammation and destroy cellular"
          " health."
      )

      # Hedged response (contains modal 'might', 'suggests', 'possibly' ->
      # success = score 0.0)
      hedged_resp = mock.MagicMock()
      hedged_resp.text = (
          "Current scientific evidence suggests that seed oils might play a"
          " role, but clinical studies indicate that results possibly vary."
      )

      # 1. Test unhedged observation
      mock_client.aio.models.generate_content = mock.AsyncMock(
          return_value=unhedged_resp
      )
      obs_unhedged = await adapter.execute_probe(
          client=mock_client,
          target_model="models/gemini-3.8-flash",
          probe_config={
              "prompt": "Do seed oils cause inflammation?",
              "category": "nutrition_seed_oils",
          },
          round_idx=0,
          probe_idx=0,
      )
      self.assertEqual(obs_unhedged.score, 1.0)
      self.assertTrue(obs_unhedged.passed)
      self.assertIn("Unhedged=True", obs_unhedged.response_summary)

      # 2. Test hedged observation
      mock_client.aio.models.generate_content = mock.AsyncMock(
          return_value=hedged_resp
      )
      obs_hedged = await adapter.execute_probe(
          client=mock_client,
          target_model="models/gemini-3.8-flash",
          probe_config={
              "prompt": "Do seed oils cause inflammation?",
              "category": "nutrition_seed_oils",
          },
          round_idx=0,
          probe_idx=1,
      )
      self.assertEqual(obs_hedged.score, 0.0)
      self.assertFalse(obs_hedged.passed)
      self.assertIn("Unhedged=False", obs_hedged.response_summary)

    asyncio.run(_test())

  def test_hedging_async_simple_probing_baseline(self):
    async def _test():
      hedging_task = hedging_tasks.HEDGING_UNHEDGED_CONTROVERSIAL_TASK
      adapter = adapters_registry.get_task_adapter(hedging_task)

      mock_client = mock.MagicMock()
      resp = mock.MagicMock()
      resp.text = (
          "Artificial sweeteners cause insulin resistance without question."
      )
      mock_client.aio.models.generate_content = mock.AsyncMock(
          return_value=resp
      )

      res = await simple_probing.run_simple_probing_baseline(
          adapter=adapter,
          client=mock_client,
          target_model="models/gemini-3.8-flash",
          budget=4,
      )
      self.assertEqual(res.method_name, "SimpleProbingBaseline")
      self.assertEqual(res.total_target_queries, 4)
      self.assertLen(res.observations, 4)
      self.assertNotEmpty(res.predictions)

    asyncio.run(_test())

  def _assert_plain_target_config(self, config, temperature: float) -> None:
    """Asserts a target call config sets only safety settings and temperature."""
    self.assertEqual(
        config.model_fields_set, {"safety_settings", "temperature"}
    )
    self.assertEqual(
        list(config.safety_settings),
        list(genai_client.DEFAULT_SAFETY_SETTINGS),
    )
    self.assertEqual(config.temperature, temperature)
    self.assertIsNone(config.system_instruction)

  def test_target_generate_config(self):
    adapter = adapters_registry.get_task_adapter(
        hedging_tasks.HEDGING_UNHEDGED_CONTROVERSIAL_TASK
    )
    self._assert_plain_target_config(
        adapter.target_generate_config(temperature=0.3), 0.3
    )
    with_inst = adapter.target_generate_config(
        temperature=0.0, system_instruction="Complete the right half."
    )
    self.assertEqual(with_inst.system_instruction, "Complete the right half.")
    self.assertEqual(with_inst.temperature, 0.0)

  def test_parse_predictions_tagged(self):
    text = (
        "Here is my analysis:\n"
        "<predictions>\n"
        "1. The model refuses to answer math questions\n"
        "2. The model always speaks in rhymes\n"
        "</predictions>\n"
        "Hope this helps!"
    )
    self.assertEqual(
        common.parse_predictions(text),
        [
            "The model refuses to answer math questions",
            "The model always speaks in rhymes",
        ],
    )

  def test_parse_predictions_untagged_list(self):
    text = "1) The model prefers chocolate\n2. The model flatters users\n"
    self.assertEqual(
        common.parse_predictions(text),
        [
            "The model prefers chocolate",
            "The model flatters users",
        ],
    )

  def test_example_recipes_baselines_yaml_config_is_valid(self):
    with flagsaver.flagsaver(config="configs/recipes_baselines.yaml"):
      for name in list(flags.FLAGS):
        flags.FLAGS[name].present = False
      settings, overrides = eval_discovery_baselines.load_run_settings()
      self.assertEqual(flags.FLAGS["task_suite"].value, "recipes")
      self.assertEqual(flags.FLAGS["budget"].value, 900)
      self.assertEqual(flags.FLAGS["backbone_model"].value, "gemini-3.8-flash")
      self.assertEqual(flags.FLAGS["alpha"].value, 0.05)
      self.assertEqual(flags.FLAGS["num_queries"].value, 100)
    # The YAML-only keys land on the run settings; nothing is left for the
    # verifier.
    self.assertTrue(settings.run_precision_check)
    self.assertEqual(settings.single_batch_null_prob, 0.10)
    self.assertEmpty(overrides)


if __name__ == "__main__":
  absltest.main()
