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

"""Unit tests for per-round prompt builders."""

from absl.testing import absltest

from disco.prompting import round_prompts


class RoundPromptsTest(absltest.TestCase):

  def test_injected_queries_preamble_is_empty_without_queries(self):
    self.assertEqual(round_prompts.injected_queries_preamble([]), "")

  def test_injected_queries_preamble_numbers_queries(self):
    preamble = round_prompts.injected_queries_preamble(["a", "b"])
    self.assertStartsWith(preamble, "The following 2 queries")
    self.assertIn("1. a\n2. b\n\n", preamble)

  def test_salient_selection_prompt_without_history(self):
    prompt = round_prompts.salient_selection_prompt(
        "seed", "H1", "text", [], [{"prompt": "p", "response": "r"}]
    )
    self.assertIn(
        "Past Hypotheses & Statuses:\nNo previous hypotheses.", prompt
    )
    self.assertIn("1. Prompt: 'p'\n   Response: 'r'\n\n", prompt)

  def test_salient_selection_prompt_marks_missing_sub_hypotheses(self):
    entry = round_prompts.SalientHistoryEntry(
        h_id="H1", text="text", status=" [A]", sub_hypotheses=[]
    )
    prompt = round_prompts.salient_selection_prompt(
        "seed", "H1", "text", [entry], []
    )
    self.assertIn("- ID 'H1' (text) [A] | Sub-hypotheses: None", prompt)


if __name__ == "__main__":
  absltest.main()
