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

"""Tests for standalone_tools: statistical tests, procedural autoraters, and tool declarations."""

import asyncio
import inspect
import io
import json
import os
from typing import Any
from unittest import mock

from absl.testing import absltest
from google.genai import types
import numpy as np
import PIL.Image
from scipy import stats

from disco import runners
from disco import standalone_agent
from disco import standalone_tools as tools
from disco import stats_utils
from disco.models import content as content_api
from disco.models import genai_client
from disco.stats import plan as stats_plan
from disco.tasks.visual_symmetry import image_gen as visual_symmetry_image_gen


async def fake_to_threads_with_throttle(tasks, **unused_kwargs):
  """Sequential stand-in for `content_api.to_threads_with_throttle`.

  Awaits any awaitable a task returns, which the real helper also does. A
  sync-only fake happens to work when the tasks are sync mocks, but silently
  returns bare coroutine objects the moment a caller wraps its tasks in an
  async shim -- so the fake has to honour the same contract as the function it
  replaces.

  Args:
    tasks: Zero-argument callables, sync or async.
    **unused_kwargs: `desc` / `bandwidth`, ignored.

  Returns:
    The task results, positionally aligned with `tasks`.
  """
  results = []
  for task in tasks:
    result = task()
    if inspect.isawaitable(result):
      result = await result
    results.append(result)
  return results


class StandaloneToolsTest(absltest.TestCase):

  def test_count_words_autorater(self):
    self.assertIn("count_words", tools.PROCEDURAL_AUTORATERS)
    fn = tools.PROCEDURAL_AUTORATERS["count_words"]

    # Basic space-separated text
    self.assertEqual(fn("hello world this is a test", {}), 6.0)
    # Extra whitespace, newlines, tabs
    self.assertEqual(fn("  hello \n\n world \t test   ", {}), 3.0)
    # Empty string
    self.assertEqual(fn("", {}), 0.0)
    # Single word
    self.assertEqual(fn("word", {}), 1.0)

  def test_register_autorater(self):
    @tools.register_autorater("custom_test_rater")
    def custom_rater(text: str, params: dict[str, Any]) -> float:
      del params
      return float(len(text))

    self.assertIn("custom_test_rater", tools.PROCEDURAL_AUTORATERS)
    self.assertEqual(
        tools.PROCEDURAL_AUTORATERS["custom_test_rater"]("abc", {}), 3.0
    )

  def test_get_tool_declarations(self):
    all_tools = tools.get_tool_declarations(exclude_analyze=False)
    names = [t.name for t in all_tools]
    self.assertIn("analyze_responses", names)
    self.assertIn("run_procedural_autorater", names)
    self.assertIn("generate_single_turn_responses", names)
    self.assertIn("run_continuous_means_test", names)

    filtered_tools = tools.get_tool_declarations(exclude_analyze=True)
    filtered_names = [t.name for t in filtered_tools]
    self.assertNotIn("analyze_responses", filtered_names)
    self.assertIn("run_procedural_autorater", filtered_names)
    self.assertIn("generate_single_turn_responses", filtered_names)
    self.assertIn("run_continuous_means_test", filtered_names)

    no_means_tools = tools.get_tool_declarations(
        exclude_analyze=True, allow_continuous_means=False
    )
    no_means_names = [t.name for t in no_means_tools]
    self.assertNotIn("analyze_responses", no_means_names)
    self.assertNotIn("run_continuous_means_test", no_means_names)
    self.assertIn("run_proportion_z_test", no_means_names)

  def test_run_proportion_z_test_requires_alternative(self):
    res = tools.run_proportion_z_test({"n_successes": 5, "n_obs": 10})
    self.assertIn("Error: 'alternative' parameter is required", res)

    res_invalid = tools.run_proportion_z_test(
        {"n_successes": 5, "n_obs": 10, "alternative": "invalid"}
    )
    self.assertIn("Error: 'alternative' parameter is required", res_invalid)

  def test_run_proportion_z_test_directional(self):
    # 10/10 successes with null 0.75 and alternative='larger'
    res_larger = tools.run_proportion_z_test({
        "n_successes": 10,
        "n_obs": 10,
        "null_proportion": 0.75,
        "alternative": "larger",
    })
    self.assertIn("P-value: 0.056314", res_larger)
    self.assertIn("Statistic: N/A (exact)", res_larger)
    self.assertIn("Method: Binomial test (exact)", res_larger)

    # 2/10 successes with null 0.75 and alternative='larger' (below null)
    res_below = tools.run_proportion_z_test({
        "n_successes": 2,
        "n_obs": 10,
        "null_proportion": 0.75,
        "alternative": "larger",
    })
    self.assertIn("P-value: 0.999970", res_below)
    self.assertIn("Note: Sample proportion (0.2000) is below null", res_below)

  def test_run_proportion_z_test_large_n_always_exact(self):
    # N = 50 >= 30, should still use Binomial test (exact)
    res = tools.run_proportion_z_test({
        "n_successes": 35,
        "n_obs": 50,
        "null_proportion": 0.5,
        "alternative": "larger",
    })
    self.assertIn("Method: Binomial test (exact)", res)
    self.assertIn("Statistic: N/A (exact)", res)
    expected_p = stats.binomtest(35, 50, 0.5, alternative="greater").pvalue
    self.assertIn(f"P-value: {expected_p:.6f}", res)

  def test_run_proportion_z_test_small_p_value_scientific_notation(self):
    res = tools.run_proportion_z_test({
        "n_successes": 90,
        "n_obs": 100,
        "null_proportion": 0.1,
        "alternative": "larger",
    })
    self.assertIn("Method: Binomial test (exact)", res)
    self.assertIn("e-", res)
    self.assertNotIn("P-value: 0.000000", res)

  def test_run_proportion_z_test_2_sample(self):
    res_missing = tools.run_proportion_z_test_2_sample({
        "n_successes_a": 8,
        "n_obs_a": 10,
        "n_successes_b": 2,
        "n_obs_b": 10,
    })
    self.assertIn("Error: 'alternative' parameter is required", res_missing)

    res = tools.run_proportion_z_test_2_sample({
        "n_successes_a": 8,
        "n_obs_a": 10,
        "n_successes_b": 2,
        "n_obs_b": 10,
        "alternative": "larger",
    })
    self.assertIn("Two-Sample Proportion Test Results", res)
    self.assertIn("Method: Fisher's Exact Test", res)
    self.assertIn("Alternative: larger", res)

  def test_run_proportion_z_test_2_sample_large_n_always_exact(self):
    # Both N = 50 >= 30, should still use Fisher's Exact Test
    res = tools.run_proportion_z_test_2_sample({
        "n_successes_a": 35,
        "n_obs_a": 50,
        "n_successes_b": 15,
        "n_obs_b": 50,
        "alternative": "larger",
    })
    self.assertIn("Method: Fisher's Exact Test", res)
    self.assertIn("Statistic: N/A (exact)", res)
    table = [[35, 15], [15, 35]]
    _, expected_p = stats.fisher_exact(table, alternative="greater")
    self.assertIn(f"P-value: {stats_utils.format_p_value(expected_p)}", res)

  def test_run_mcnemar_test_directional(self):
    res_missing = tools.run_mcnemar_test({
        "passed_a_failed_b": 8,
        "failed_a_passed_b": 2,
    })
    self.assertIn("Error: 'alternative' parameter is required", res_missing)

    res_larger = tools.run_mcnemar_test({
        "passed_a_failed_b": 8,
        "failed_a_passed_b": 2,
        "alternative": "larger",
    })
    self.assertIn("McNemar's Test Results", res_larger)
    self.assertIn("P-value: 0.054688", res_larger)

  def test_run_continuous_means_test(self):
    res_missing = tools.run_continuous_means_test({
        "sample_mean": 150.0,
        "sample_std": 20.0,
        "n_obs": 10,
        "null_mean": 100.0,
    })
    self.assertIn("Error: 'alternative' parameter is required", res_missing)

    res = tools.run_continuous_means_test({
        "sample_mean": 150.0,
        "sample_std": 20.0,
        "n_obs": 10,
        "null_mean": 100.0,
        "alternative": "larger",
    })
    self.assertIn("One-Sample Means Test Results", res)
    self.assertIn("Method: T-test (Student's t)", res)
    self.assertIn("Alternative: larger", res)

  def test_run_continuous_means_test_large_n_always_t_test(self):
    # N = 50 >= 30, should still use Student's t-test
    res = tools.run_continuous_means_test({
        "sample_mean": 105.0,
        "sample_std": 10.0,
        "n_obs": 50,
        "null_mean": 100.0,
        "alternative": "larger",
    })
    self.assertIn("Method: T-test (Student's t)", res)
    z = (105.0 - 100.0) / (10.0 / np.sqrt(50))
    expected_p = stats.t.sf(z, df=49)
    self.assertIn(f"P-value: {expected_p:.6f}", res)

  def test_run_continuous_means_test_zero_variance(self):
    # Case 1: sample_mean > null_mean with sample_std = 0.0
    res_larger = tools.run_continuous_means_test({
        "sample_mean": 15.0,
        "sample_std": 0.0,
        "n_obs": 10,
        "null_mean": 10.0,
        "alternative": "larger",
    })
    self.assertIn("Deterministic (zero variance)", res_larger)
    self.assertIn("Statistic: inf", res_larger)
    self.assertIn("P-value: 0.000000", res_larger)

    res_smaller = tools.run_continuous_means_test({
        "sample_mean": 15.0,
        "sample_std": 0.0,
        "n_obs": 10,
        "null_mean": 10.0,
        "alternative": "smaller",
    })
    self.assertIn("Statistic: inf", res_smaller)
    self.assertIn("P-value: 1.000000", res_smaller)

    # Case 2: sample_mean < null_mean with sample_std = 0.0
    res_below = tools.run_continuous_means_test({
        "sample_mean": 5.0,
        "sample_std": 0.0,
        "n_obs": 10,
        "null_mean": 10.0,
        "alternative": "smaller",
    })
    self.assertIn("Statistic: -inf", res_below)
    self.assertIn("P-value: 0.000000", res_below)

    # Case 3: sample_mean == null_mean with sample_std = 0.0
    res_equal = tools.run_continuous_means_test({
        "sample_mean": 10.0,
        "sample_std": 0.0,
        "n_obs": 10,
        "null_mean": 10.0,
        "alternative": "larger",
    })
    self.assertIn("Statistic: 0.0000", res_equal)
    self.assertIn("P-value: 1.000000", res_equal)

  def test_run_paired_means_test(self):
    res_missing = tools.run_paired_means_test({
        "mean_diff": 15.0,
        "std_diff": 5.0,
        "n_pairs": 10,
        "null_mean": 0.0,
    })
    self.assertIn("Error: 'alternative' parameter is required", res_missing)

    res = tools.run_paired_means_test({
        "mean_diff": 15.0,
        "std_diff": 5.0,
        "n_pairs": 10,
        "null_mean": 0.0,
        "alternative": "larger",
    })
    self.assertIn("Paired Means Test Results", res)
    self.assertIn("Method: Paired T-test (Student's t)", res)
    self.assertIn("Mean difference (A - B): 15.0000", res)
    self.assertIn("Std dev of differences: 5.0000", res)
    self.assertIn("Number of pairs: 10", res)
    t_stat = 15.0 / (5.0 / np.sqrt(10))
    expected_p = stats.t.sf(t_stat, df=9)
    self.assertIn(f"P-value: {stats_utils.format_p_value(expected_p)}", res)

  def test_run_paired_means_test_zero_variance(self):
    res_larger = tools.run_paired_means_test({
        "mean_diff": 10.0,
        "std_diff": 0.0,
        "n_pairs": 10,
        "null_mean": 0.0,
        "alternative": "larger",
    })
    self.assertIn("Deterministic (zero variance)", res_larger)
    self.assertIn("Statistic (t): inf", res_larger)
    self.assertIn("P-value: 0.000000", res_larger)

  def test_maybe_randomize_p_value(self):
    rng = np.random.default_rng(42)

    # 1. 1-sample binomial test
    res_1sample = stats_plan.StatsResult(
        test_type="Binomial test (exact)",
        p_value=0.05,
        group_a_mean=0.7,
        group_a_obs=10,
        null_proportion=0.5,
    )
    p_rand_1 = runners._maybe_randomize_p_value(0.05, res_1sample, rng)
    self.assertNotEqual(p_rand_1, 0.05)
    self.assertGreaterEqual(p_rand_1, 0.0)
    self.assertLessEqual(p_rand_1, 1.0)

    # 2. 2-sample Fisher's exact test
    res_fisher = stats_plan.StatsResult(
        test_type="Fisher's Exact Test",
        p_value=0.03,
        group_a_mean=0.8,
        group_a_obs=10,
        group_b_mean=0.2,
        group_b_obs=10,
    )
    p_rand_2 = runners._maybe_randomize_p_value(0.03, res_fisher, rng)
    self.assertNotEqual(p_rand_2, 0.03)
    self.assertGreaterEqual(p_rand_2, 0.0)
    self.assertLessEqual(p_rand_2, 1.0)

    # 3. McNemar's test
    res_mcnemar = stats_plan.StatsResult(
        test_type="McNemar's test (exact)",
        p_value=0.04,
        group_a_mean=0.8,
        group_a_obs=10,
    )
    p_rand_3 = runners._maybe_randomize_p_value(0.04, res_mcnemar, rng)
    self.assertNotEqual(p_rand_3, 0.04)
    self.assertGreaterEqual(p_rand_3, 0.0)
    self.assertLessEqual(p_rand_3, 1.0)

    # 4. Continuous means t-test: MUST NOT be randomized
    res_continuous = stats_plan.StatsResult(
        test_type="T-test (Student's t)",
        p_value=0.042,
        group_a_mean=105.0,
        group_a_obs=15,
    )
    p_rand_4 = runners._maybe_randomize_p_value(0.042, res_continuous, rng)
    self.assertEqual(p_rand_4, 0.042)

  @mock.patch.object(standalone_agent.genai_client, "get_genai_client")
  @mock.patch.object(standalone_agent, "_get_or_create_candidate")
  def test_generate_responses_preserves_image_chunks_and_drops_empty(
      self, mock_get_candidate, mock_get_client
  ):
    mock_get_client.return_value = mock.MagicMock()
    mock_candidate = mock.AsyncMock()
    mock_get_candidate.return_value = mock_candidate

    logs = []
    agent = standalone_agent.StandaloneAgent(
        agent_model="models/gemini-3.8-flash",
        target_model="gemini-3.1-flash-image",
        output_dir=self.create_tempdir().full_path,
        log_fn=logs.append,
    )

    probe_mock = mock.MagicMock(spec=content_api.Probe)

    # 1. Text response
    resp_text = content_api.Response(
        content=content_api.Content(
            [content_api.ContentChunk("A lovely horse.")]
        ),
        probe=probe_mock,
    )
    # 2. Pure image response (empty text, non-empty media chunk)
    resp_image = content_api.Response(
        content=content_api.Content([
            content_api.ContentChunk(
                b"\x89PNG\r\n\x1a\nfake_image_bytes", mimetype="image/png"
            )
        ]),
        probe=probe_mock,
    )
    # 3. Empty/whitespace response
    resp_empty = content_api.Response(
        content=content_api.Content(
            [content_api.ContentChunk("   \n  ", mimetype="text/plain")]
        ),
        probe=probe_mock,
    )
    # 4. Multimodal response (text + image)
    resp_multimodal = content_api.Response(
        content=content_api.Content([
            content_api.ContentChunk("Here is your picture"),
            content_api.ContentChunk(
                b"\x89PNG\r\n\x1a\nfake_bytes_2", mimetype="image/png"
            ),
        ]),
        probe=probe_mock,
    )

    def mock_respond_text(content, *args, **kwargs):
      del args, kwargs
      txt = content.as_text(ignore_unsupported_types=True)
      if "horse 1" in txt:
        return resp_text
      elif "horse 2" in txt:
        return resp_image
      elif "horse 3" in txt:
        return resp_empty
      elif "horse 4" in txt:
        return resp_multimodal
      return resp_empty

    mock_candidate.respond.side_effect = mock_respond_text
    mock_get_candidate.return_value = mock_candidate
    agent._candidate_cls = mock_candidate

    result = asyncio.run(
        agent._tool_generate_responses({
            "prompts": [
                "draw horse 1",
                "draw horse 2",
                "draw horse 3",
                "draw horse 4",
            ],
            "candidate": "gemini-3.1-flash-image",
        })
    )

    # Responses 1, 2, and 4 should be retained; 3 should be dropped.
    self.assertLen(agent._last_responses, 3)
    self.assertLen(agent.all_responses, 3)
    self.assertEqual(agent.last_valid_response_count, 3)
    self.assertEqual(agent._last_responses[0], resp_text)
    self.assertEqual(agent._last_responses[1], resp_image)
    self.assertEqual(agent._last_responses[2], resp_multimodal)

    # Verify formatting in returned result string
    self.assertIn("--- Response 1 ---\nA lovely horse.", result)
    self.assertIn("--- Response 2 ---\n[Image generated]", result)
    self.assertIn("--- Response 3 ---\nHere is your picture", result)

    # Verify log contains warning for empty response
    self.assertTrue(
        any(
            "[Scrape Warning] Response 3 was empty/blank." in log
            for log in logs
        )
    )

    # The failed prompt leaves a gap in the recorded indices rather than
    # shifting response 4 into slot 3.
    self.assertEqual(agent.last_response_indices, [0, 1, 3])
    self.assertEqual(agent.last_prompt_count, 4)

  @mock.patch.object(standalone_agent.genai_client, "get_genai_client")
  @mock.patch.object(standalone_agent, "_get_or_create_candidate")
  def test_backfilled_response_returns_to_its_original_slot(
      self, mock_get_candidate, mock_get_client
  ):
    """A refilled response must not be appended after later responses.

    Paired batches alternate Group A / Group B, so appending a backfill to the
    end would leave every subsequent response attributed to the wrong pair and
    the wrong arm.
    """
    mock_get_client.return_value = mock.MagicMock()
    mock_candidate = mock.AsyncMock()
    mock_get_candidate.return_value = mock_candidate

    logs = []
    agent = standalone_agent.StandaloneAgent(
        agent_model="models/gemini-3.8-flash",
        target_model="gemini-3.8-flash",
        output_dir=self.create_tempdir().full_path,
        log_fn=logs.append,
    )

    probe_mock = mock.MagicMock(spec=content_api.Probe)

    def make_response(text):
      return content_api.Response(
          content=content_api.Content([content_api.ContentChunk(text)]),
          probe=probe_mock,
      )

    # Prompt "b0" fails on its first attempt and succeeds on the backfill.
    seen_b0 = {"count": 0}

    def mock_respond(content, *args, **kwargs):
      del args, kwargs
      txt = content.as_text(ignore_unsupported_types=True)
      if txt == "b0":
        seen_b0["count"] += 1
        if seen_b0["count"] == 1:
          return make_response("   ")
      return make_response(f"ok:{txt}")

    mock_candidate.respond.side_effect = mock_respond
    agent._candidate_cls = mock_candidate

    asyncio.run(
        agent._tool_generate_responses({
            "prompts": ["a0", "b0", "a1", "b1"],
            "candidate": "gemini-3.8-flash",
        })
    )

    texts = [
        r.content.as_text(ignore_unsupported_types=True)
        for r in agent.last_responses
    ]
    self.assertEqual(texts, ["ok:a0", "ok:b0", "ok:a1", "ok:b1"])
    self.assertEqual(agent.last_response_indices, [0, 1, 2, 3])

    # The reduction must therefore see two intact pairs.
    reduction = stats_utils.reduce_pairs(
        [1.0, 2.0, 5.0, 3.0], agent.last_response_indices
    )
    self.assertEqual(reduction.n_pairs, 2)
    self.assertEqual(reduction.n_orphans, 0)
    self.assertEqual(reduction.diffs, [-1.0, 2.0])

  @mock.patch.object(standalone_agent.genai_client, "get_genai_client")
  @mock.patch.object(standalone_agent, "_get_or_create_candidate")
  def test_agent_multiple_scrape_accumulates(
      self, mock_get_candidate, mock_get_client
  ):
    del mock_get_client
    mock_candidate = mock.AsyncMock()
    mock_get_candidate.return_value = mock_candidate

    logs = []
    agent = standalone_agent.StandaloneAgent(
        agent_model="models/gemini-3.8-flash",
        target_model="gemini-3.1-flash-image",
        output_dir=self.create_tempdir().full_path,
        log_fn=logs.append,
    )
    probe_mock = mock.MagicMock(spec=content_api.Probe)

    resp_1 = content_api.Response(
        content=content_api.Content(
            [content_api.ContentChunk("Batch 1 response")]
        ),
        probe=probe_mock,
    )
    resp_2 = content_api.Response(
        content=content_api.Content(
            [content_api.ContentChunk("Batch 2 response")]
        ),
        probe=probe_mock,
    )

    mock_candidate.respond.side_effect = [resp_1, resp_2]
    agent._candidate_cls = mock_candidate

    # First tool call: 1 prompt
    asyncio.run(
        agent._tool_generate_responses({
            "prompts": ["prompt 1"],
            "candidate": "gemini-3.1-flash-image",
        })
    )
    self.assertLen(agent._last_responses, 1)
    self.assertLen(agent.all_responses, 1)
    self.assertEqual(agent._last_responses[0], resp_1)
    self.assertEqual(agent.all_responses[0], resp_1)

    # Second tool call: 1 prompt
    asyncio.run(
        agent._tool_generate_responses({
            "prompts": ["prompt 2"],
            "candidate": "gemini-3.1-flash-image",
        })
    )
    # _last_responses should only contain the second batch
    self.assertLen(agent._last_responses, 1)
    self.assertEqual(agent._last_responses[0], resp_2)
    # all_responses should accumulate across both calls
    self.assertLen(agent.all_responses, 2)
    self.assertEqual(agent.all_responses[0], resp_1)
    self.assertEqual(agent.all_responses[1], resp_2)

  def test_build_judge_prompt(self):
    prompt = tools.AskQuestionJudge.build_judge_prompt(
        prompt_text="Generate a photo of a person writing.",
        response_text="Here is a photo.",
        question_text="What hand is the person using?",
        answer_options=["Left", "Right"],
    )
    # Text-only: should return a plain string.
    self.assertIsInstance(prompt, str)
    self.assertIn("<CONVERSATION_HISTORY>", prompt)
    self.assertIn(
        "<USER>\nGenerate a photo of a person writing.\n</USER>", prompt
    )
    self.assertIn("<MODEL>\nHere is a photo.\n</MODEL>", prompt)
    self.assertIn("</CONVERSATION_HISTORY>", prompt)
    self.assertIn(
        "<QUESTION>\nWhat hand is the person using?\n</QUESTION>", prompt
    )
    self.assertIn("<ANSWER_OPTIONS>\n[Left, Right]\n</ANSWER_OPTIONS>", prompt)
    self.assertIn('"chain_of_thought": <your reasoning>', prompt)
    self.assertIn('"answer": <one of the options>', prompt)

  def test_build_judge_prompt_with_images(self):
    fake_img = tools.backbone_lib.ImagePart(
        data=b"\x89PNG_fake",
        mimetype="image/png",
    )
    prompt = tools.AskQuestionJudge.build_judge_prompt(
        prompt_text="Describe this image.",
        response_text="A cat sitting.",
        question_text="Is the description accurate?",
        answer_options=["Yes", "No"],
        prompt_images=[fake_img],
        response_images=[fake_img],
    )
    # With images: should return a list (multimodal prompt).
    self.assertIsInstance(prompt, list)
    # First element should be a text part containing the header and USER
    # section.
    self.assertIsInstance(prompt[0], str)
    self.assertIn("<USER>\nDescribe this image.\n</USER>", prompt[0])
    # Should contain ImagePart objects.
    image_parts = [
        p for p in prompt if isinstance(p, tools.backbone_lib.ImagePart)
    ]
    self.assertLen(image_parts, 2)
    # Should contain a MODEL section.
    text_parts = [p for p in prompt if isinstance(p, str)]
    model_texts = [t for t in text_parts if "<MODEL>" in t]
    self.assertLen(model_texts, 1)
    self.assertIn("A cat sitting.", model_texts[0])

  def test_extract_images_from_content(self):
    """Tests the _extract_images_from_content helper."""
    fake_png = b"\x89PNG_fake_data"
    # Create a content object with text and an image.
    content = content_api.Content(
        "Some text",
        content_api.ContentChunk(fake_png, mimetype="image/png"),
    )
    images = tools.AskQuestionJudge.extract_images_from_content(content)
    self.assertLen(images, 1)
    self.assertIsInstance(images[0], tools.backbone_lib.ImagePart)
    self.assertEqual(images[0].data, fake_png)
    self.assertEqual(images[0].mimetype, "image/png")

  def test_extract_images_from_text_only_content(self):
    """Tests that text-only content returns empty image list."""
    content = content_api.Content("Just text, no images.")
    images = tools.AskQuestionJudge.extract_images_from_content(content)
    self.assertEmpty(images)

  @mock.patch.object(standalone_agent.genai_client, "get_genai_client")
  @mock.patch.object(standalone_agent, "_get_or_create_candidate")
  @mock.patch.object(standalone_agent.backbone_lib, "get_backbone_client")
  def test_tool_analyze_responses_with_backbone(
      self, mock_get_backbone, mock_get_candidate, mock_get_client
  ):
    mock_get_client.return_value = mock.MagicMock()
    mock_get_candidate.return_value = mock.MagicMock()
    mock_backbone = mock.AsyncMock()
    mock_get_backbone.return_value = mock_backbone

    # Return valid judgment for response 1, invalid option for response 2
    async def mock_generate(model, prompt, **kwargs):
      del model, kwargs
      # Handle both str and list[PromptPart] prompts robustly.
      prompt_str = (
          prompt
          if isinstance(prompt, str)
          else "".join(p for p in prompt if isinstance(p, str))
      )
      if "Response 1" in prompt_str:
        return json.dumps({
            "chain_of_thought": "The person holds pen in right hand.",
            "answer": "Right",
        })
      return json.dumps({
          "chain_of_thought": "Cannot tell.",
          "answer": "Ambiguous",
      })

    mock_backbone.generate.side_effect = mock_generate

    logs = []
    agent = standalone_agent.StandaloneAgent(
        agent_model="models/gemini-3.8-flash",
        target_model="gemini-3.8-flash",
        output_dir=self.create_tempdir().full_path,
        log_fn=logs.append,
    )
    agent._JUDGE_JITTER = 0.0

    probe_mock = mock.MagicMock(spec=content_api.Probe)
    probe_mock.content = content_api.Content("Draw a person writing.")

    agent._last_responses = [
        content_api.Response(
            content=content_api.Content("Response 1"), probe=probe_mock
        ),
        content_api.Response(
            content=content_api.Content("Response 2"), probe=probe_mock
        ),
    ]

    result_str = asyncio.run(
        agent._tool_analyze_responses({
            "question": "What hand is the person using?",
            "answer_options": ["Left", "Right"],
        })
    )
    result = json.loads(result_str)

    self.assertEqual(result["counts"]["Right"], 1)
    self.assertEqual(result["counts"]["Left"], 0)
    self.assertEqual(result["counts"]["Unknown"], 1)
    self.assertLen(result["judgments"], 2)
    self.assertEqual(result["judgments"][0]["judgment"], "Right")
    self.assertEqual(result["judgments"][1]["judgment"], "Unknown")

  @mock.patch.object(standalone_agent.genai_client, "get_genai_client")
  @mock.patch.object(standalone_agent, "_get_or_create_candidate")
  @mock.patch.object(standalone_agent.backbone_lib, "get_backbone_client")
  def test_tool_analyze_responses_with_images(
      self, mock_get_backbone, mock_get_candidate, mock_get_client
  ):
    """Verifies that images in probe/response content are passed to backbone."""
    mock_get_client.return_value = mock.MagicMock()
    mock_get_candidate.return_value = mock.MagicMock()
    mock_backbone = mock.AsyncMock()
    mock_get_backbone.return_value = mock_backbone

    # Track what prompts are sent to the backbone.
    captured_prompts = []

    async def mock_generate(model, prompt, **kwargs):
      del model, kwargs
      captured_prompts.append(prompt)
      return json.dumps({
          "chain_of_thought": "The image shows a person.",
          "answer": "Yes",
      })

    mock_backbone.generate.side_effect = mock_generate

    logs = []
    agent = standalone_agent.StandaloneAgent(
        agent_model="models/gemini-3.8-flash",
        target_model="gemini-3.8-flash",
        output_dir=self.create_tempdir().full_path,
        log_fn=logs.append,
    )
    agent._JUDGE_JITTER = 0.0

    # Build probe with text + image.
    fake_png = b"\x89PNG_test_image"
    probe_content = content_api.Content(
        "Draw a cat.",
        content_api.ContentChunk(fake_png, mimetype="image/png"),
    )
    probe_mock = mock.MagicMock(spec=content_api.Probe)
    probe_mock.content = probe_content

    # Build response with text + image.
    resp_content = content_api.Content(
        "Here is a cat image.",
        content_api.ContentChunk(fake_png, mimetype="image/jpeg"),
    )

    agent._last_responses = [
        content_api.Response(content=resp_content, probe=probe_mock),
    ]

    result_str = asyncio.run(
        agent._tool_analyze_responses({
            "question": "Is this a cat?",
            "answer_options": ["Yes", "No"],
        })
    )
    result = json.loads(result_str)

    # Verify judgment succeeded.
    self.assertEqual(result["counts"]["Yes"], 1)
    self.assertLen(result["judgments"], 1)
    self.assertEqual(result["judgments"][0]["judgment"], "Yes")

    # Verify the prompt sent to backbone was multimodal (a list, not str).
    self.assertLen(captured_prompts, 1)
    self.assertIsInstance(captured_prompts[0], list)

    # Should contain ImagePart objects from both probe and response.
    image_parts = [
        p
        for p in captured_prompts[0]
        if isinstance(p, tools.backbone_lib.ImagePart)
    ]
    self.assertLen(image_parts, 2)
    self.assertEqual(image_parts[0].mimetype, "image/png")
    self.assertEqual(image_parts[1].mimetype, "image/jpeg")

  @mock.patch.object(tools.backbone_lib, "get_backbone_client")
  def test_ask_question_judge_model_normalization(self, mock_get_backbone):
    mock_backbone = mock.AsyncMock()
    mock_get_backbone.return_value = mock_backbone

    judge1 = tools.AskQuestionJudge(
        backbone=mock_backbone,
        question_text="test?",
        answer_options=["A", "B"],
        model="models/gemini-3.8-flash",
    )
    self.assertEqual(judge1._model, "gemini-3.8-flash")

    judge2 = tools.AskQuestionJudge(
        backbone=mock_backbone,
        question_text="test?",
        answer_options=["A", "B"],
        model="gemini-3.8-flash",
    )
    self.assertEqual(judge2._model, "gemini-3.8-flash")

  def test_ask_question_judge_custom_model(self):
    mock_backbone = mock.AsyncMock()
    judge = tools.AskQuestionJudge(
        backbone=mock_backbone,
        question_text="Is the output formal?",
        answer_options=["Yes", "No"],
        model="models/gemini-3.8-flash",
    )
    self.assertEqual(judge._model, "gemini-3.8-flash")

  @mock.patch.object(standalone_agent.content_api, "to_threads_with_throttle")
  @mock.patch.object(standalone_agent.genai_client, "get_genai_client")
  @mock.patch.object(standalone_agent, "_get_or_create_candidate")
  def test_symmetry_probes_and_scoring_pass_at_k(
      self, mock_get_candidate, mock_get_client, mock_to_threads
  ):
    mock_get_client.return_value = mock.MagicMock()
    mock_to_threads.side_effect = fake_to_threads_with_throttle
    logs = []
    agent = standalone_agent.StandaloneAgent(
        agent_model="models/gemini-3.8-flash",
        target_model="gemini-3.1-flash-image",
        output_dir=self.create_tempdir().full_path,
        log_fn=logs.append,
    )

    probe_mock = mock.MagicMock(spec=content_api.Probe)

    # Ground truth image for probe 0 (pass)
    _, _, gt0_img = visual_symmetry_image_gen.generate_symmetry_image(
        shape_type="square", rows=10, cols=10, seed=0
    )
    pil_gt0 = PIL.Image.fromarray(gt0_img)
    buf0 = io.BytesIO()
    pil_gt0.save(buf0, format="PNG")
    resp_pass_gt0 = content_api.Response(
        content=content_api.Content(
            [content_api.ContentChunk(buf0.getvalue(), mimetype="image/png")]
        ),
        probe=probe_mock,
    )

    # Empty image for fail
    pil_blank = PIL.Image.fromarray(np.full((100, 100, 3), 255, dtype=np.uint8))
    buf_blank = io.BytesIO()
    pil_blank.save(buf_blank, format="PNG")
    resp_fail = content_api.Response(
        content=content_api.Content([
            content_api.ContentChunk(buf_blank.getvalue(), mimetype="image/png")
        ]),
        probe=probe_mock,
    )

    # Probe 0: [fail, pass], Probe 1: [fail, fail]
    call_idx = 0
    resp_list = [resp_fail, resp_pass_gt0, resp_fail, resp_fail]

    def mock_respond_symmetry(unused_content, *args, **kwargs):
      del unused_content, args, kwargs
      nonlocal call_idx
      res = resp_list[call_idx]
      call_idx += 1
      return res

    mock_candidate = mock.MagicMock()
    mock_candidate.respond.side_effect = mock_respond_symmetry
    mock_get_candidate.return_value = mock_candidate
    agent._candidate_cls = mock_candidate

    gen_res_str = asyncio.run(
        agent._tool_generate_and_send_symmetry_probes({
            "shape_type": "square",
            "n_probes": 2,
            "samples_per_probe": 2,
        })
    )
    gen_res = json.loads(gen_res_str)
    self.assertEqual(gen_res["n_probes_sent"], 2)
    self.assertEqual(gen_res["samples_per_probe"], 2)
    self.assertEqual(gen_res["total_requests"], 4)
    self.assertEqual(gen_res["n_valid_outputs"], 4)

    score_res_str = asyncio.run(agent._tool_score_symmetry_completions({}))
    score_res = json.loads(score_res_str)
    self.assertEqual(score_res["n_probes"], 2)
    self.assertEqual(score_res["k"], 2)
    self.assertEqual(score_res["n_probes_passed"], 1)
    self.assertAlmostEqual(score_res["pass_at_k_rate"], 0.5)
    self.assertEqual(score_res["total_samples"], 4)
    self.assertEqual(score_res["total_samples_passed"], 1)
    self.assertAlmostEqual(score_res["pass_at_1_rate"], 0.25)
    self.assertLen(agent.last_judgments, 2)
    self.assertEqual(agent.last_judgments[0]["value"], 1.0)
    self.assertEqual(agent.last_judgments[1]["value"], 0.0)

  @mock.patch.object(standalone_agent.content_api, "to_threads_with_throttle")
  @mock.patch.object(standalone_agent.genai_client, "get_genai_client")
  @mock.patch.object(standalone_agent, "_get_or_create_candidate")
  def test_symmetry_round_summary_rows_track_probes_when_some_unanswered(
      self, mock_get_candidate, mock_get_client, mock_to_threads
  ):
    """Scores in round_summary.json must follow the probe, not the row index.

    Unanswered requests are compacted out before saving, so with probe 0
    unanswered the only saved row belongs to probe 1. Annotating by probe index
    would stamp probe 0's verdict onto probe 1's response.
    """
    mock_get_client.return_value = mock.MagicMock()
    mock_to_threads.side_effect = fake_to_threads_with_throttle
    output_dir = self.create_tempdir().full_path
    agent = standalone_agent.StandaloneAgent(
        agent_model="models/gemini-3.8-flash",
        target_model="gemini-3.1-flash-image",
        output_dir=output_dir,
        log_fn=lambda *a, **k: None,
    )

    probe_mock = mock.MagicMock(spec=content_api.Probe)

    # Probe i is generated with seed=i, so probe 1's ground truth uses seed 1.
    _, _, gt1_img = visual_symmetry_image_gen.generate_symmetry_image(
        shape_type="square", rows=10, cols=10, seed=1
    )
    buf = io.BytesIO()
    PIL.Image.fromarray(gt1_img).save(buf, format="PNG")
    resp_pass = content_api.Response(
        content=content_api.Content(
            [content_api.ContentChunk(buf.getvalue(), mimetype="image/png")]
        ),
        probe=probe_mock,
    )
    # No image chunk at all: this request stays unanswered through backfill.
    resp_no_image = content_api.Response(
        content=content_api.Content("refused"), probe=probe_mock
    )

    # Initial pass: probe 0 -> no image, probe 1 -> correct completion. Every
    # later call is a backfill retry for probe 0, which keeps failing.
    responses = [resp_no_image, resp_pass]
    call_idx = 0

    def mock_respond(unused_content, *args, **kwargs):
      del unused_content, args, kwargs
      nonlocal call_idx
      res = responses[call_idx] if call_idx < len(responses) else resp_no_image
      call_idx += 1
      return res

    mock_candidate = mock.MagicMock()
    mock_candidate.respond.side_effect = mock_respond
    mock_get_candidate.return_value = mock_candidate
    agent._candidate_cls = mock_candidate

    gen_res = json.loads(
        asyncio.run(
            agent._tool_generate_and_send_symmetry_probes({
                "shape_type": "square",
                "n_probes": 2,
            })
        )
    )
    self.assertEqual(gen_res["n_probes_sent"], 2)
    self.assertEqual(gen_res["n_valid_outputs"], 1)

    asyncio.run(agent._tool_score_symmetry_completions({}))

    summary_path = os.path.join(
        output_dir, "responses_round_001", "round_summary.json"
    )
    with open(summary_path) as f:
      summary = json.load(f)

    # Only the answered request was saved, and it must be attributed to probe 1.
    self.assertLen(summary, 1)
    self.assertEqual(summary[0]["probe_idx"], 1)
    self.assertEqual(summary[0]["sample_idx"], 0)
    self.assertTrue(summary[0]["passed"])
    self.assertGreaterEqual(summary[0]["iou"], 0.9)

  @mock.patch.object(standalone_agent.genai_client, "get_genai_client")
  @mock.patch.object(standalone_agent, "_get_or_create_candidate")
  def test_stat_guard_single_test_and_duplicate_rejection(
      self, mock_get_candidate, mock_client
  ):
    mock_client.return_value = mock.MagicMock()
    mock_get_candidate.return_value = mock.MagicMock()
    logs = []
    agent = standalone_agent.StandaloneAgent(
        agent_model="models/gemini-3.8-flash",
        target_model="gemini-3.8-flash",
        output_dir=self.create_tempdir().full_path,
        log_fn=logs.append,
    )

    # 1. Invalid syntax / missing alternative allows retry without locking
    res_err = asyncio.run(
        agent._dispatch_tool(
            "run_proportion_z_test", {"n_successes": 10, "n_obs": 15}
        )
    )
    self.assertIn("Error: 'alternative' parameter is required", res_err)
    self.assertFalse(agent._stat_test_completed_this_step)

    # 2. Valid statistical test calculation succeeds and locks further stat
    # calls
    res_ok = asyncio.run(
        agent._dispatch_tool(
            "run_proportion_z_test",
            {
                "n_successes": 10,
                "n_obs": 15,
                "alternative": "larger",
                "null_proportion": 0.5,
            },
        )
    )
    self.assertIn("One-Sample Proportion Test Results", res_ok)
    self.assertTrue(agent._stat_test_completed_this_step)

    # 3. Duplicate statistical test call in the same step is blocked
    res_duplicate = asyncio.run(
        agent._dispatch_tool(
            "run_proportion_z_test",
            {
                "n_successes": 12,
                "n_obs": 15,
                "alternative": "larger",
                "null_proportion": 0.5,
            },
        )
    )
    self.assertIn(
        "Error: A statistical test has already been successfully computed for"
        " this step",
        res_duplicate,
    )

    # 4. Late scraping / measurement attempt after stat test is blocked
    res_late_scrape = asyncio.run(
        agent._dispatch_tool(
            "generate_single_turn_responses",
            {"prompts": ["test query 1", "test query 2"]},
        )
    )
    self.assertIn(
        "Error: Statistical evaluation has already completed for this step",
        res_late_scrape,
    )

  @mock.patch.object(standalone_agent, "_get_or_create_candidate")
  @mock.patch.object(genai_client, "get_genai_client")
  def test_exclude_analyze_blocks_tool(self, mock_client, mock_get_candidate):
    mock_client.return_value = mock.MagicMock()
    mock_get_candidate.return_value = mock.MagicMock()
    agent = standalone_agent.StandaloneAgent(
        agent_model="models/gemini-3.8-flash",
        target_model="gemini-3.8-flash",
        output_dir=self.create_tempdir().full_path,
        exclude_analyze=True,
    )
    res = asyncio.run(
        agent._dispatch_tool(
            "analyze_responses",
            {"question": "Is it symmetric?", "answer_options": ["Yes", "No"]},
        )
    )
    self.assertIn("Error: analyze_responses is disabled for this task", res)

  @mock.patch.object(standalone_agent, "_get_or_create_candidate")
  @mock.patch.object(genai_client, "get_genai_client")
  def test_standalone_agent_thinking_config(
      self, mock_client_factory, mock_get_candidate
  ):
    mock_get_candidate.return_value = mock.MagicMock()
    mock_client = mock.MagicMock()
    mock_client_factory.return_value = mock_client

    mock_part = types.Part(text="Analysis complete.")
    mock_candidate = types.Candidate(
        content=types.Content(role="model", parts=[mock_part])
    )
    mock_response = mock.MagicMock()
    mock_response.candidates = [mock_candidate]
    mock_response.usage_metadata = None
    mock_client.aio.models.generate_content = mock.AsyncMock(
        return_value=mock_response
    )

    # 1. Thinking level LOW
    agent_low = standalone_agent.StandaloneAgent(
        agent_model="models/gemini-3.8-flash",
        target_model="gemini-3.8-flash",
        output_dir=self.create_tempdir().full_path,
        agent_thinking_level="LOW",
    )
    asyncio.run(agent_low.send("Test instruction"))
    call_args = mock_client.aio.models.generate_content.call_args
    self.assertIsNotNone(call_args)
    config: Any = call_args.kwargs.get("config")
    self.assertIsNotNone(config)
    self.assertIsNotNone(config.thinking_config)
    self.assertEqual(config.thinking_config.thinking_level, "LOW")
    self.assertIsNone(config.thinking_config.thinking_budget)

    # 2. Thinking level OFF -> thinking_budget=0
    agent_off = standalone_agent.StandaloneAgent(
        agent_model="models/gemini-3.8-flash",
        target_model="gemini-3.8-flash",
        output_dir=self.create_tempdir().full_path,
        agent_thinking_level="OFF",
    )
    asyncio.run(agent_off.send("Test instruction"))
    call_args = mock_client.aio.models.generate_content.call_args
    config = call_args.kwargs.get("config")
    self.assertIsNotNone(config)
    self.assertIsNotNone(config.thinking_config)
    self.assertEqual(config.thinking_config.thinking_budget, 0)

    # 3. Thinking budget explicit integer
    agent_budget = standalone_agent.StandaloneAgent(
        agent_model="models/gemini-3.8-flash",
        target_model="gemini-3.8-flash",
        output_dir=self.create_tempdir().full_path,
        agent_thinking_budget=2048,
    )
    asyncio.run(agent_budget.send("Test instruction"))
    call_args = mock_client.aio.models.generate_content.call_args
    config = call_args.kwargs.get("config")
    self.assertIsNotNone(config)
    self.assertIsNotNone(config.thinking_config)
    self.assertEqual(config.thinking_config.thinking_budget, 2048)


if __name__ == "__main__":
  absltest.main()
