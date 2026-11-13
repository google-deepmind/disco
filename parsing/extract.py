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

"""Extraction helper functions for parsing unstructured or semi-structured model text."""

import json
import re


def parse_selected_example_indices(
    response_text: str, total_candidates: int, num_to_select: int = 3
) -> list[int]:
  """Parses 1-based selected example indices from an LLM response.

  Args:
    response_text: The LLM response containing a JSON array of 1-based integers.
    total_candidates: Total number of candidate examples available.
    num_to_select: Number of indices desired (default 3).

  Returns:
    List of 0-based indices into the candidate pool.
  """
  indices: list[int] = []
  if not response_text:
    return list(range(min(total_candidates, num_to_select)))

  # Try direct JSON parse or regex search for JSON array
  try:
    data = json.loads(response_text)
    if isinstance(data, list):
      for item in data:
        if isinstance(item, int) and 1 <= item <= total_candidates:
          if (item - 1) not in indices:
            indices.append(item - 1)
  except (ValueError, TypeError):
    pass

  if len(indices) < num_to_select:
    # Try finding digits / array in text via regex
    matches = re.findall(r"\b\d+\b", response_text)
    for m in matches:
      val = int(m)
      if 1 <= val <= total_candidates and (val - 1) not in indices:
        indices.append(val - 1)
        if len(indices) == num_to_select:
          break

  # Fallback if still empty or fewer: pad with available 0-based indices
  if len(indices) < num_to_select:
    for i in range(total_candidates):
      if i not in indices:
        indices.append(i)
      if len(indices) == min(total_candidates, num_to_select):
        break

  return indices[:num_to_select]
