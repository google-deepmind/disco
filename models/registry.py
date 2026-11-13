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

"""Model registry and identifier resolution for DISCO Gemini models."""

# Default Gemini models used across DISCO.
DEFAULT_GEMINI_MODEL: str = "gemini-3.8-flash"
DEFAULT_GEMINI_IMAGE_MODEL: str = "gemini-3.1-flash-image"

KNOWN_PREFIXES = ("models/",)

_GEMINI_MODEL_ALIASES: dict[str, str] = {
    "GEMINI_FLASH_LATEST": "gemini-3.8-flash",
    "GEMINI_FLASH_LITE_LATEST": "gemini-3.8-flash",
    "GEMINI_FLASH_3_8": "gemini-3.8-flash",
    "GEMINI_FLASH_IMAGE": "gemini-3.1-flash-image",
    "GEMINI_FLASH_3_1_IMAGE": "gemini-3.1-flash-image",
}


def normalize_model_address(address: str) -> str:
  """Strips known prefixes from a model address to get the bare model ID."""
  result = address.strip()
  for prefix in KNOWN_PREFIXES:
    result = result.removeprefix(prefix)
  return result


def resolve_model_url(model_name_or_url: str) -> str:
  """Resolves a model short name or alias to a bare Gemini model ID."""
  stripped = normalize_model_address(model_name_or_url)
  normalized_input = stripped.upper().replace("-", "").replace("_", "")

  for key, model_id in _GEMINI_MODEL_ALIASES.items():
    if key.replace("-", "").replace("_", "") == normalized_input:
      return model_id

  return stripped
