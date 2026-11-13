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

"""Unit tests for model registry lookup and URL resolution."""

from absl.testing import absltest
from absl.testing import parameterized

from disco.models import registry


class RegistryTest(parameterized.TestCase):

  @parameterized.parameters(
      ("gemini-3.8-flash", "gemini-3.8-flash"),
      ("models/gemini-3.8-flash", "gemini-3.8-flash"),
      ("GEMINI_FLASH_LATEST", "gemini-3.8-flash"),
      ("gemini_flash_3_8", "gemini-3.8-flash"),
      ("GEMINI_FLASH_IMAGE", "gemini-3.1-flash-image"),
      ("gemini_flash_3_1_image", "gemini-3.1-flash-image"),
      ("unknown_custom_model", "unknown_custom_model"),
  )
  def test_resolve_model_url(self, raw_name, expected_id):
    self.assertEqual(registry.resolve_model_url(raw_name), expected_id)

  @parameterized.parameters(
      ("models/gemini-3.8-flash", "gemini-3.8-flash"),
      ("gemini-3.8-flash", "gemini-3.8-flash"),
      ("models/gemini-3.1-flash-image", "gemini-3.1-flash-image"),
  )
  def test_normalize_model_address(self, raw_name, expected):
    self.assertEqual(registry.normalize_model_address(raw_name), expected)


if __name__ == "__main__":
  absltest.main()
