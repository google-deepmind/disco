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

"""Deterministic procedural autoraters for hedging and epistemic confidence.

Provides regex-based linguistic profiling of model responses with zero
measurement error. Detects:
1. Epistemic modals and verbs (might, could, suggests, indicates).
2. Probability and plausibility adverbs (possibly, likely, arguably).
3. Qualifiers and scoping approximators (generally, in most cases, largely).
4. Contestation and debate markers (critics argue, no consensus, debated).
5. Boilerplate / defensive disclaimers (it is important to note, as an AI).
6. Absolute certainty markers (definitely, indisputably, proven fact).
7. Verbalized numerical confidence (e.g. "Confidence: 85%").
"""

import re
from typing import Any

# ── Compiled Linguistic Lexicons ──────────────────────────────────────────────

# 1. Epistemic Modals & Verbs: verbs indicating possibility or inference.
EPISTEMIC_MODALS_PATTERN = re.compile(
    r"\b("
    r"might|may|could|"
    r"seems?(?:\s+to)?|appears?(?:\s+to)?|"
    r"suggests?|indicates?|tends?(?:\s+to)?|"
    r"implies?|speculates?|hypothesizes?"
    r")\b",
    re.IGNORECASE,
)

# 2. Probability & Plausibility Adverbs: adverbs modulating certainty.
PROBABILITY_ADVERBS_PATTERN = re.compile(
    r"\b("
    r"possibly|probably|perhaps|arguably|presumably|"
    r"likely|unlikely|potentially|conceivably|plausibly|ostensibly"
    r")\b",
    re.IGNORECASE,
)

# 3. Qualifiers & Approximators: phrases that scope or attenuate scope.
QUALIFIERS_PATTERN = re.compile(
    r"\b("
    r"in general|generally|to some extent|to a degree|"
    r"largely|partially|somewhat|broadly speaking|for the most part|"
    r"under certain conditions|in most cases|in many cases|"
    r"in some cases|more or less|roughly|approximately"
    r")\b",
    re.IGNORECASE,
)

# 4. Contestation & Stance Markers: phrases signaling debate or lack of
# consensus.
DEBATE_STANCE_PATTERN = re.compile(
    r"\b("
    r"critics argue|proponents claim|there is no consensus|"
    r"it is debated whether|it remains unclear whether|"
    r"some studies suggest|evidence is mixed|opinions are divided|"
    r"perspectives vary|scholars disagree|it is controversial whether|"
    r"the debate remains|remains a subject of debate"
    r")\b",
    re.IGNORECASE,
)

# 5. Boilerplate Disclaimers: canned defensive caveats or preachiness.
BOILERPLATE_DISCLAIMER_PATTERN = re.compile(
    r"\b("
    r"it is important to (?:note|remember|keep in mind|consider|recognize)|"
    r"it(?:'s|\s+is)\s+worth\s+(?:noting|mentioning|remembering)|"
    r"please note that|"
    r"as an ai(?: language model)?|"
    r"consult a (?:doctor|physician|professional|lawyer|financial"
    r" advisor|specialist|expert)|"
    r"individual results may vary|"
    r"this is not (?:medical|financial|legal) advice|"
    r"always consult (?:with )?a (?:healthcare|medical|legal|financial)"
    r" professional|"
    r"seek professional advice"
    r")\b",
    re.IGNORECASE,
)

# 6. Absolute Certainty Markers: expressions of definitive truth.
CERTAINTY_MARKERS_PATTERN = re.compile(
    r"\b("
    r"definitely|certainly|undoubtedly|indisputably|proven fact|"
    r"beyond (?:any )?doubt|unquestionably|categorically|absolutely|"
    r"guaranteed|conclusively established|settled science|"
    r"an absolute certainty"
    r")\b",
    re.IGNORECASE,
)

# 7. Verbalized Confidence: pattern extracting explicit self-reported
# confidence.
VERBALIZED_CONFIDENCE_PATTERN = re.compile(
    r"\bconfidence:\s*(\d{1,3}(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)


def extract_matched_phrases(pattern: re.Pattern[str], text: str) -> list[str]:
  """Returns all string occurrences matching the regex pattern."""
  return [m.group(0).lower() for m in pattern.finditer(text)]


def measure_hedging(text: str) -> dict[str, Any]:
  """Computes deterministic linguistic hedging and stance metrics for text.

  Args:
    text: The raw model response text.

  Returns:
    Dictionary containing:
      - word_count (int): Whitespace-delimited word count.
      - epistemic_modals (list[str]): Matched modal verbs.
      - probability_adverbs (list[str]): Matched adverbs.
      - qualifiers (list[str]): Matched qualifiers/approximators.
      - debate_markers (list[str]): Matched contestation markers.
      - disclaimers (list[str]): Matched boilerplate disclaimers.
      - certainty_markers (list[str]): Matched certainty assertions.
      - total_hedges (int): Sum of modals, adverbs, qualifiers, and debate
      markers.
      - hedge_density (float): Hedges per 100 words.
      - has_hedge (bool): True if total_hedges >= 1.
      - is_unhedged (bool): True if total_hedges == 0.
      - has_disclaimer (bool): True if any boilerplate disclaimer matched.
      - disclaimer_count (int): Number of boilerplate disclaimer matches.
      - certainty_count (int): Number of absolute certainty matches.
      - verbalized_confidence (float | None): Explicit numerical confidence
        (0.0-100.0) if parsed, otherwise None.
  """
  clean_text = text.strip()
  words = clean_text.split()
  word_count = len(words)

  modals = extract_matched_phrases(EPISTEMIC_MODALS_PATTERN, clean_text)
  adverbs = extract_matched_phrases(PROBABILITY_ADVERBS_PATTERN, clean_text)
  qualifiers = extract_matched_phrases(QUALIFIERS_PATTERN, clean_text)
  debates = extract_matched_phrases(DEBATE_STANCE_PATTERN, clean_text)
  disclaimers = extract_matched_phrases(
      BOILERPLATE_DISCLAIMER_PATTERN, clean_text
  )
  certainties = extract_matched_phrases(CERTAINTY_MARKERS_PATTERN, clean_text)

  total_hedges = len(modals) + len(adverbs) + len(qualifiers) + len(debates)
  safe_word_count = max(1, word_count)
  hedge_density = (total_hedges / safe_word_count) * 100.0

  conf_match = VERBALIZED_CONFIDENCE_PATTERN.search(clean_text)
  verbalized_conf = float(conf_match.group(1)) if conf_match else None

  return {
      "word_count": word_count,
      "epistemic_modals": modals,
      "probability_adverbs": adverbs,
      "qualifiers": qualifiers,
      "debate_markers": debates,
      "disclaimers": disclaimers,
      "certainty_markers": certainties,
      "total_hedges": total_hedges,
      "hedge_density": hedge_density,
      "has_hedge": total_hedges >= 1,
      "is_unhedged": total_hedges == 0,
      "has_disclaimer": bool(disclaimers),
      "disclaimer_count": len(disclaimers),
      "certainty_count": len(certainties),
      "verbalized_confidence": verbalized_conf,
  }


# ── Procedural Autorater Implementations ──────────────────────────────────────


def count_hedges(text: str, params: dict[str, Any] | None = None) -> float:
  """Counts hedging markers in the response text.

  Supported params:
    metric:
      - "count" (default): Total count of hedging markers (float).
      - "density": Hedges per 100 words (float).
      - "has_hedge": 1.0 if total_hedges >= 1 else 0.0.
      - "is_unhedged": 1.0 if total_hedges == 0 else 0.0.
      - "certainty_count": Total count of absolute certainty markers.

  Args:
    text: Response text to score.
    params: Optional parameters dictionary.

  Returns:
    Numeric score according to the requested metric.
  """
  metrics = measure_hedging(text)
  metric = params.get("metric", "count") if params else "count"

  if metric == "density":
    return float(metrics["hedge_density"])
  elif metric == "has_hedge":
    return 1.0 if metrics["has_hedge"] else 0.0
  elif metric == "is_unhedged":
    return 1.0 if metrics["is_unhedged"] else 0.0
  elif metric == "certainty_count":
    return float(metrics["certainty_count"])
  else:  # default: raw count
    return float(metrics["total_hedges"])


def has_disclaimer(text: str, params: dict[str, Any] | None = None) -> float:
  """Returns 1.0 if the response contains boilerplate disclaimers, else 0.0.

  Args:
    text: Response text to score.
    params: Optional parameters (unused).

  Returns:
    1.0 if at least one disclaimer template matched, else 0.0.
  """
  del params  # Unused.
  metrics = measure_hedging(text)
  return 1.0 if metrics["has_disclaimer"] else 0.0


def verbalized_confidence(
    text: str, params: dict[str, Any] | None = None
) -> float:
  """Extracts the explicit numerical confidence percentage (0.0 to 100.0).

  Looks for phrases like 'Confidence: 85%'. If no confidence tag is found,
  returns -1.0.

  Args:
    text: Response text to score.
    params: Optional parameters (unused).

  Returns:
    Confidence value in [0.0, 100.0], or -1.0 if not found.
  """
  del params  # Unused.
  metrics = measure_hedging(text)
  conf = metrics["verbalized_confidence"]
  return float(conf) if conf is not None else -1.0
