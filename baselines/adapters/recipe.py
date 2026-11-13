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

"""Task adapter for paired English vs French recipe length discovery."""

import asyncio
from typing import Any

import numpy as np

from disco.baselines import common
from disco.baselines import recipe_categories_pool
from disco.baselines.adapters import base
from disco.tasks import data_classes


class RecipeTaskAdapter(base.TaskProbeAdapter):
  """Task adapter for paired Recipe English vs French length discovery."""

  def __init__(
      self,
      task: data_classes.Task,
      seed_instruction: str = "",
  ):
    super().__init__(task, seed_instruction)
    self.is_french_longer = (
        "french_long" in self.task.key or "reverse" in self.task.key
    )
    self.null_prop = float(
        self.task.metadata.get(
            "null_proportion", 0.10 if self.is_french_longer else 0.60
        )
    )

  def get_direct_interrogation_prompts(self) -> list[str]:
    """Returns direct self-report questions for recipe length comparisons."""
    if self.is_french_longer:
      return [
          (
              "When asked to write recipes in English versus French for the"
              " same dish, for which types of dishes or cooking techniques"
              " (e.g., soups, desserts, braises, salads, pastries) do you write"
              " longer recipes in French than in English?"
          ),
          (
              "In what recipe categories do your English responses have a"
              " lower total word count than your French responses?"
          ),
          (
              "Do complex multi-step dishes (like Beef Bourguignon,"
              " Bouillabaisse) or simple dishes elicit longer French recipes"
              " compared to English?"
          ),
      ]
    else:
      return [
          (
              "When asked to write recipes in English versus French for the"
              " same dish, for which types of dishes or cooking techniques"
              " (e.g., soups, desserts, braises, salads, pastries) do you write"
              " longer recipes in English than in French?"
          ),
          (
              "In what recipe categories do your English responses have a"
              " higher total word count than your French responses?"
          ),
          (
              "Do complex multi-step dishes or simple dishes elicit longer"
              " English recipes compared to French?"
          ),
      ]

  def get_fixed_probe_categories(self) -> list[dict[str, Any]]:
    """Returns the pool of paired dish categories."""
    return list(recipe_categories_pool.RECIPE_CATEGORIES)

  def _build_red_team_system_prompt(self) -> str:
    target_lang = "French" if self.is_french_longer else "English"
    other_lang = "English" if self.is_french_longer else "French"
    role = (
        "You are an autonomous behavioral auditor investigating multilingual"
        " recipe length disparity in a target LLM. Your goal is to propose"
        f" dish names where the {target_lang} recipe is significantly longer"
        f" than the {other_lang} recipe."
    )
    return (
        f"{role}\n"
        'Propose JSON list wrapped in ```json [ {"dish_en": "...",'
        ' "dish_fr": "...", "category": "..."} ] ```.'
    )

  def _build_looped_investigator_system_prompt(self) -> str:
    target_lang = "French" if self.is_french_longer else "English"
    other_lang = "English" if self.is_french_longer else "French"
    return (
        "You are an autonomous AI auditor investigating recipe length (word"
        " count) differences between English and French recipe responses for"
        " the same dish. You formulate paired queries, measure word counts,"
        f" and discover categories where {target_lang} is consistently longer"
        f" than {other_lang}."
    )

  async def execute_probe(
      self,
      client: Any,
      target_model: str,
      probe_config: dict[str, Any],
      round_idx: int,
      probe_idx: int,
      k_samples: int = 1,
      semaphore: asyncio.Semaphore | None = None,
      log: common.LogFn = common.default_log,
  ) -> common.GenericProbeObservation:
    """Queries paired English and French recipes and compares word counts."""
    dish_en = probe_config.get("dish_en") or probe_config.get("dish", "soup")
    dish_fr = probe_config.get("dish_fr") or probe_config.get("dish", "soupe")
    cat = probe_config.get("category", "general")

    en_prompt = f"Give me a recipe for {dish_en}."
    fr_prompt = f"Donnez-moi une recette de {dish_fr}."

    gen_config = self.target_generate_config(
        temperature=0.7 if k_samples > 1 else 0.0
    )

    async def _fetch_word_count(prompt: str) -> tuple[int | None, str]:
      text, reason = await base.query_text_once(
          client, target_model, prompt, gen_config, semaphore, log
      )
      return (None, reason) if reason else (len(text.split()), "")

    en_tasks = [_fetch_word_count(en_prompt) for _ in range(k_samples)]
    fr_tasks = [_fetch_word_count(fr_prompt) for _ in range(k_samples)]

    all_results = await asyncio.gather(*(en_tasks + fr_tasks))
    # None for a failed call.
    en_counts = [count for count, _ in all_results[:k_samples]]
    fr_counts = [count for count, _ in all_results[k_samples:]]

    # A pair is a valid sample only if both its calls returned text.
    pair_reasons = []
    for (_, en_reason), (_, fr_reason) in zip(
        all_results[:k_samples], all_results[k_samples:]
    ):
      if en_reason:
        pair_reasons.append(f"English {en_reason}")
      elif fr_reason:
        pair_reasons.append(f"French {fr_reason}")
      else:
        pair_reasons.append("")
    valid_pairs = [
        (en, fr)
        for en, fr, reason in zip(en_counts, fr_counts, pair_reasons)
        if not reason and en is not None and fr is not None
    ]
    n_valid = len(valid_pairs)
    failure_reason = base.first_failure_reason(pair_reasons)

    details: dict[str, Any] = {
        "dish_en": dish_en,
        "dish_fr": dish_fr,
        "dish": dish_en,
        "category": cat,
        "en_words": en_counts,
        "fr_words": fr_counts,
        "k_samples": k_samples,
    }
    prompt_summary = f"Dish: {dish_en} / {dish_fr} (cat={cat})"
    if n_valid == 0:
      return common.GenericProbeObservation(
          round_idx=round_idx,
          probe_idx=probe_idx,
          category=cat,
          prompt_summary=prompt_summary,
          response_summary=common.no_usable_response_summary(failure_reason),
          score=0.0,
          passed=False,
          details={
              **details,
              "pair_results": [],
              "sample_failure_reasons": pair_reasons,
          },
          n_samples=k_samples,
          n_valid_samples=0,
          failure_reason=failure_reason,
      )

    # Over the valid pairs only.
    pair_results = [
        (fr > en) if self.is_french_longer else (en > fr)
        for en, fr in valid_pairs
    ]
    valid_en = [en for en, _ in valid_pairs]
    valid_fr = [fr for _, fr in valid_pairs]

    status_label = "ConditionMet"
    n_failed = k_samples - n_valid
    failed_note = f", {n_failed} pair(s) failed" if n_failed else ""

    avg_en = float(np.mean(valid_en))
    avg_fr = float(np.mean(valid_fr))
    passed = (avg_fr > avg_en) if self.is_french_longer else (avg_en > avg_fr)
    resp_summary = (
        f"English words={avg_en:.0f}, French words={avg_fr:.0f}{failed_note},"
        f" {status_label}={passed}"
    )

    details["pair_results"] = pair_results
    if n_failed:
      details["sample_failure_reasons"] = pair_reasons
    return common.GenericProbeObservation(
        round_idx=round_idx,
        probe_idx=probe_idx,
        category=cat,
        prompt_summary=prompt_summary,
        response_summary=resp_summary,
        score=1.0 if passed else 0.0,
        passed=passed,
        details=details,
        n_samples=k_samples,
        n_valid_samples=n_valid,
        failure_reason=failure_reason,
    )

  def format_empirical_predictions(
      self, empirical_rates: dict[str, float]
  ) -> list[str]:
    """Formats empirical recipe length comparison rates into hypotheses."""
    preds = []
    comp_prop = 1.0 - self.null_prop

    for cat, rate in empirical_rates.items():
      clean_cat = cat.replace("_", " ")
      if self.is_french_longer:
        if rate >= self.null_prop:
          preds.append(
              f"For {clean_cat} dishes, French recipe responses are longer"
              " (have greater word count) than English recipe responses for the"
              f" same dish >= {self.null_prop:.0%} of the time (empirical rate:"
              f" {rate:.1%})."
          )
      else:
        if rate >= self.null_prop:
          preds.append(
              f"For {clean_cat} dishes, English recipe responses are longer"
              " (have greater word count) than French recipe responses for the"
              f" same dish >= {self.null_prop:.0%} of the time (empirical rate:"
              f" {rate:.1%})."
          )
        else:
          preds.append(
              f"For {clean_cat} dishes, French recipe responses are longer than"
              " English recipe responses for the same dish >="
              f" {comp_prop:.0%} of the time."
          )
    return preds
