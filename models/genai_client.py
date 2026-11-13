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

"""Shared genai.Client setup and standard HTTP/safety settings."""

import os

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

# Expected transient, network, response-parsing, or client-configuration errors
# raised during Gemini/LLM API calls. Catching this tuple retries or degrades
# gracefully on operational failures while allowing programming bugs
# (AttributeError, TypeError, KeyError, NameError, etc.) to surface immediately.
LLM_CALL_ERRORS = (genai_errors.APIError, OSError, ValueError, RuntimeError)

DEFAULT_SAFETY_SETTINGS = (
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
)


def get_api_key() -> str | None:
  """Returns the Gemini API key from environment variables."""
  return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def get_default_http_options() -> types.HttpOptions:
  """Returns the default HTTP options for the genai client."""
  return types.HttpOptions(
      timeout=600 * 1000,  # in milliseconds
      retry_options=types.HttpRetryOptions(
          attempts=20,
          initial_delay=5.0,
          exp_base=1.6,
          max_delay=120.0,
          jitter=1.0,
          http_status_codes=[408, 429, 500, 502, 503, 504],
      ),
  )


_cached_client: genai.Client | None = None


def get_genai_client() -> genai.Client:
  """Returns a shared genai.Client with standard DISCO configuration."""
  global _cached_client
  if _cached_client is None:
    api_key = get_api_key()
    if api_key is None:
      raise RuntimeError(
          "Missing API key: set the GEMINI_API_KEY or GOOGLE_API_KEY"
          " environment variable."
      )
    _cached_client = genai.Client(
        api_key=api_key,
        http_options=get_default_http_options(),
    )
  return _cached_client


def reset_cached_client() -> None:
  """Resets the cached client singleton (intended for unit tests)."""
  global _cached_client
  _cached_client = None
